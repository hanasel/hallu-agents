"""Cross-dataset disagreement pilot: pool-based querying, panels formed post-hoc.

Runs the frozen 3x3 core pool (agents.panels.pool() — 3 families x 3 nominal
capability tiers) against either SimpleQA Verified (open-ended only) or
TruthfulQA (open-ended or multiple-choice, via --dataset/--prompt-format) and
reports the same panel-level disagreement/correctness analysis over both.
One grader (evaluation.grade_correct) and one set of panel definitions serve
every combination, rather than a separate script per dataset that can
silently diverge from this one — see grade_correct's docstring for what
happened the first time this project had two independently-maintained
graders.

Design (applies to every --dataset/--prompt-format combination)
-----------------------------------------------------------------
POOL, NOT PANEL. The query loop takes a *pool* of models and asks each one
every question exactly once. Panel composition is a post-hoc analysis choice
over the cached texts — the cache is keyed on (model, prompt, temperature,
max_tokens), not on panel membership, so every panel below is free once the
pool has been queried. Adding a model costs one pass over the questions;
adding a *panel* costs nothing.

ABSTENTION IS A THIRD STATE. Detected explicitly — prose regex patterns in
open-ended mode, a missing extractable answer letter in MC mode — excluded
from clustering, graded as NOT_ATTEMPTED rather than incorrect, and reported
per-agent. Abstention rate may well be a better hallucination signal than
disagreement on some datasets — but that is only visible if it isn't buried
inside the entropy.

EMPTY IS A FAILURE, NOT A DATA POINT. A response that is empty with
finish_reason='length' is a truncation, and is handled exactly like an API
error: the question is skipped and counted, and the run aborts past a
threshold. A results file can no longer silently contain unusable rows.

MATCHED-N PANELS. Every within-family pair is compared against every
cross-family pair at matched N=2, and every tier panel holds family
composition and size constant at N=3, so family/tier effects are isolated
from panel size rather than confounded with it.

AUROC, NOT MEANS. In open-ended mode, Jaccard is continuous while normalised
entropy over k singleton clusters is exactly 1.0 — their means are not on a
comparable scale. (In MC mode the primary measure isn't entropy at all; see
"MC mode" below.) Rank correlation and AUROC against graded incorrectness are
reported instead of raw means.

Grader independence
--------------------
In open-ended mode, by default the SAME CrossEncoderNLI instance does the
clustering and the grading. Their errors are therefore correlated in exactly
the direction that inflates "disagreement predicts wrongness" — with k
agents, k distinct clusters nearly forces at most one agent to entail the
gold. Pass --judge-model to regrade with an independent LLM judge; the report
prints both gradings side by side and the AUROC computed under each. Treat
any AUROC that only survives under the NLI grader as an artifact. MC mode has
no such judge: letter comparison is deterministic, so there is nothing for an
independent grader to adjudicate, and --judge-model is ignored there (with a
one-line note) rather than erroring.

MC mode is a different measurement, not a flag
--------------------------------------------------
Every MC response is a single letter, so semantic entropy (NLI on "A" vs "B")
has no signal and the open-ended abstention regexes (prose patterns) never
fire. MC mode instead uses disagreement.MCExactMatch as the primary measure —
still stored under the row's "semantic_entropy" key so every report function
keeps working unmodified; check "measure" in each panel dict, or the section
headers, to see which one it actually is — grades via
grade_correct(..., prompt_format="mc") with the short-answer string-match
branch skipped entirely, and treats "no letter could be extracted" as
abstention instead of the regexes.

Requires: OPENROUTER_API_KEY, plus torch + sentence-transformers for the NLI
model (open-ended mode only — MC mode never triggers those weights to load).
Responses are cached (agents/) so re-runs are fast and deterministic.

Run from the project root:
    python scripts/disagreement_pilot.py --dataset simpleqa --n 200
    python scripts/disagreement_pilot.py --dataset truthfulqa --prompt-format mc --n 200
    python scripts/disagreement_pilot.py --dataset truthfulqa --n 200 --judge-model <model-id>
    python scripts/disagreement_pilot.py --n 50 --models a/b,c/d   # extend the pool
    python scripts/disagreement_pilot.py --n 50 --analyse-only     # re-report, no queries
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import (                                                     # noqa: E402
    load_simpleqa, SIMPLEQA_QUERY_CONFIG,
    load_truthfulqa, TRUTHFULQA_QUERY_CONFIG,
)
from agents import (                                                   # noqa: E402
    query_agents,
    assert_models_available,
    PermanentAgentError,
    PERMANENT_ERROR_PREFIX,
)
from agents.panels import (                                            # noqa: E402
    pool as panels_pool,
    family_of,
    tier_of,
    CORE_MODELS,
    LLAMA_SMALL,
    LLAMA_LARGE,
)
from agents.providers import make_provider_agent                      # noqa: E402
from disagreement import (                                             # noqa: E402
    JaccardDisagreement,
    MCExactMatch,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
)
from evaluation import grade_correct                                   # noqa: E402
from evaluation.grading import _extract_mc_letter                     # noqa: E402
from scripts.harvest import _git_sha                                   # noqa: E402


# ---------------------------------------------------------------------------
# Dataset table — everything that varies by --dataset in one place. Adding a
# third dataset means adding one entry here, not branching through the rest
# of the script.
# ---------------------------------------------------------------------------

DATASETS = {
    "simpleqa": {
        "loader": load_simpleqa,
        "config": SIMPLEQA_QUERY_CONFIG,
        "meta": ("topic", "answer_type", "multi_step", "requires_reasoning"),
        "formats": ("open",),
        "out": "outputs/simpleqa_results.jsonl",
    },
    "truthfulqa": {
        "loader": load_truthfulqa,
        "config": TRUTHFULQA_QUERY_CONFIG,
        "meta": ("category", "question_type"),
        "formats": ("open", "mc"),
        "out": "outputs/truthfulqa_results.jsonl",
    },
}


def default_out_path(dataset: str, prompt_format: str) -> str:
    """--out default: DATASETS[dataset]['out'], with the prompt format folded
    into the filename whenever it isn't 'open'. MC responses are letters and
    open responses are prose — two formats writing to the same file would
    silently mix rows that no report function here could compare, so this
    keeps them apart by construction rather than by caller discipline."""
    base = Path(DATASETS[dataset]["out"])
    if prompt_format == "open":
        return str(base)
    return str(base.with_name(f"{base.stem}_{prompt_format}{base.suffix}"))


def select_prompt(s, prompt_format: str) -> str:
    """The prompt actually sent to agents, chosen explicitly on
    `prompt_format` rather than read off `s.prompt`. `.prompt` happens to
    default to open_prompt() on both sample classes today, but a run that
    trusted that default would silently follow it into MC territory the day
    it stops being true — TruthfulQASample has both an open_prompt() and a
    mc_prompt(), SimpleQAVerifiedSample only open_prompt() (no MC format:
    SimpleQA has no distractors)."""
    if prompt_format == "open":
        p = s.open_prompt()
    elif prompt_format == "mc":
        p = s.mc_prompt()
    else:
        raise ValueError(f"prompt_format must be 'open' or 'mc'; got {prompt_format!r}")
    assert (p or "").strip(), f"{prompt_format} prompt is empty for {s.uid}"
    return p


def check_prompt_selection(samples, dataset_cfg: dict, prompt_format: str) -> None:
    """One-time sanity check, run before any query is sent: on a dataset that
    supports both formats, the chosen prompt must actually differ from the
    other one. Silently defaulting an 'open' run onto the MC prompt (or vice
    versa) would invalidate every cross-dataset comparison while looking
    completely normal in the output — this turns that into a loud assertion
    failure at the top of the run instead."""
    formats = dataset_cfg["formats"]
    if len(formats) < 2:
        return
    s = samples[0]
    chosen = select_prompt(s, prompt_format)
    other_format = next(f for f in formats if f != prompt_format)
    other = select_prompt(s, other_format)
    assert chosen != other, (
        f"{prompt_format} and {other_format} prompts are identical for "
        f"{s.uid} — prompt selection is not actually routing on "
        f"--prompt-format"
    )


# ---------------------------------------------------------------------------
# Configuration you will want to edit as the pool grows
# ---------------------------------------------------------------------------

# Models whose answer is preceded by chain-of-thought. These need a much
# larger token budget (the reasoning is billed and counted against
# max_tokens), and they are the ones to watch for abstention behaviour.
# Substring match against the model id, so "gpt-oss" catches every size.
REASONING_MODEL_MARKERS = ("gpt-oss", "-thinking", "deepseek-r1", "/o1", "/o3", "qwq")

# Conservative abstention patterns. Deliberately anchored near the start of
# the response so that "... though I don't have the exact date" inside a real
# answer is not misread as a refusal. Audit these with --show-abstentions
# before trusting the abstention rate: a false positive here silently removes
# a genuine dissent from the cluster count.
_ABSTENTION_PATTERNS = [
    r"^\s*(i|we)\s+(do not|don't|cannot|can't|couldn't|could not)\s+"
    r"(have|find|know|recall|access|verify|confirm|determine)",
    r"^\s*(i|we)\s+(am|are)\s+not\s+(sure|certain|aware)",
    r"^\s*(i|we)\s+(am|are)\s+unable\s+to",
    r"^\s*(i|we)\s+(am|are)\s+sorry",          # "I'm sorry, but I am unable to..."
    r"^\s*(i|we)\s+(do not|don't)\s+have\s+"
    r"(the\s+|any\s+|enough\s+|specific\s+|reliable\s+)*information",   # * not ?
    r"^\s*(sorry|unfortunately)[,\s]",
    r"^\s*(no|insufficient|unknown|not enough)\s+(information|data|record)",
    r"^\s*(unknown|n/?a|not available)\s*[.!]?\s*$",
]
_ABSTENTION_RE = [re.compile(p, re.IGNORECASE) for p in _ABSTENTION_PATTERNS]

# Minimum examples in EACH class before an AUROC is reported. Below this the
# estimate is dominated by two or three questions and will not survive a viva.
MIN_CLASS_FOR_AUROC = 10


def is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(marker in m for marker in REASONING_MODEL_MARKERS)


def agent_is_reasoning(a) -> bool:
    """Reasoning is a property of the resolved request params, not the model id —
    the same model appears twice in the pool, with reasoning on and off."""
    rp = getattr(a, "reasoning_params", {}) or {}
    eff = (rp.get("reasoning") or {}).get("effort", rp.get("reasoning_effort"))
    if eff is not None:
        return str(eff).lower() != "none"
    return is_reasoning_model(a.model)


def _abstention_text(t: str) -> str:
    """Normalise contractions before matching, so 'I'm not sure' and 'I am not
    sure' hit the same pattern. Cheaper and less error-prone than threading
    optional apostrophe forms through every alternation."""
    t = (t or "").strip().replace("\u2019", "'")
    t = re.sub(r"\bi'm\b", "I am", t, flags=re.I)
    t = re.sub(r"\bwe're\b", "we are", t, flags=re.I)
    return t

def is_abstention(text: str) -> bool:
    t = _abstention_text(text)
    if not t:
        return False          # empty is a truncation failure, not an abstention
    return any(rx.search(t) for rx in _ABSTENTION_RE)


# ---------------------------------------------------------------------------
# Small stats helpers (kept dependency-free so the analysis path runs anywhere)
# ---------------------------------------------------------------------------

def _ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """P(score of a positive > score of a negative); ties count 0.5."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auroc_guarded(scores, labels) -> Tuple[Optional[float], str]:
    """AUROC plus a reason string when it is suppressed for being underpowered."""
    npos, nneg = sum(labels), len(labels) - sum(labels)
    if min(npos, nneg) < MIN_CLASS_FOR_AUROC:
        return None, f"underpowered (pos={npos}, neg={nneg})"
    return auroc(scores, labels), ""

def common_rows(rows, keys):
    """Rows where EVERY listed panel is scorable, so panels are compared on
    the same questions. Without this, a panel containing an agent that
    abstains often is scored on a subset selected by that agent's confidence."""
    return [r for r in rows
            if all(r["panels"].get(k) and r["panels"][k]["semantic_entropy"] is not None
                   for k in keys)]

def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def short(name: str) -> str:
    return name.split("/")[-1]


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _agent_factory():
    """Locate the repo's single-model agent constructor.

    NOTE — the one place this script guesses at your API. It needs a callable
    that turns a model id plus query kwargs into an Agent. If none of the
    names below exist, export one from agents/panels.py, e.g.

        def make_agent(model: str, **query_kwargs) -> Agent: ...

    and the --models flag will work. Without it the pool falls back to
    agents.panels.pool().
    """
    import agents as agents_mod
    from agents import panels as panels_mod
    for mod in (panels_mod, agents_mod):
        for attr in ("make_agent", "build_agent", "agent_for", "agent_from_model"):
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
    return None


def build_pool(extra_models: List[str], panel_kwargs: dict) -> List:
    """The full agents.panels.pool(), plus any --models ids, deduped by model."""
    pool = list(panels_pool(**panel_kwargs))
    seen = {a.model for a in pool}

    if extra_models:
        factory = _agent_factory()
        if factory is None:
            raise RuntimeError(
                "--models needs a single-model agent constructor. Export one "
                "from agents/panels.py named make_agent(model, **query_kwargs) "
                "— see _agent_factory()'s docstring."
            )
        for model in extra_models:
            model = model.strip()
            if model and model not in seen:
                pool.append(factory(model, **panel_kwargs))
                seen.add(model)
    return pool


# ---------------------------------------------------------------------------
# Panel specs — formed offline over the pool, so they are free
# ---------------------------------------------------------------------------

class PanelSpec:
    """A named subset of the pool, identified by agent name."""

    def __init__(self, key: str, label: str, members: List[str], kind: str):
        self.key = key
        self.label = label
        self.members = members          # agent names
        self.kind = kind                # 'core' | 'full' | 'within' | 'cross' | 'toggle'
                                         # | 'ablation' | 'tier' | 'loo'

    def __repr__(self) -> str:
        return f"PanelSpec({self.key!r}, n={len(self.members)})"


def build_panel_specs(agents) -> List[PanelSpec]:
    """Core pool, full pool, every matched-N=2 pair, tier-matched panels, LOO
    panels, and the reasoning ablation.

    "core" is the frozen 3×3 pool (`agents.panels.CORE_MODELS`) and is the
    headline panel — its key is stable regardless of what else `--models`
    adds to the queried pool, so headline numbers stay reproducible. "full"
    is every agent actually queried (core plus any --models extras), kept as
    an exploratory extension reported separately when it differs from core.

    The N=2 pairs are the point: comparing a 2-model same-family panel against
    a bigger mixed panel confounds family with size, so within-family and
    cross-family are both enumerated at N=2 and aggregated.

    Tier panels hold family composition and panel size constant (one core
    model per family, N=3) while nominal capability tier varies, so
    small/large/strong are comparable panels rather than confounded by N.

    LOO ("leave-one-out") panels score the core pool with one core agent's
    own answer removed, so that agent's per-agent AUROC isn't partly scored
    against a panel that includes its own answer.
    """
    names = [a.name for a in agents]
    fam = {a.name: family_of(a.model) for a in agents}
    model_of = {a.name: a.model for a in agents}

    core = [a.name for a in agents if a.model in CORE_MODELS]
    specs: List[PanelSpec] = [
        PanelSpec("core", f"frozen core pool (N={len(core)})", core, "core"),
        PanelSpec("full", f"full pool (N={len(names)})", names, "full"),
    ]

    for x, y in combinations(names, 2):
        if model_of[x] == model_of[y]:
            kind, tag = "toggle", "same model, reasoning on/off"
        else:
            same = fam[x] == fam[y]
            kind = "within" if same else "cross"
            tag = fam[x] if same else f"{fam[x]}|{fam[y]}"
        specs.append(PanelSpec(f"{kind}:{short(x)}+{short(y)}",
                               f"{kind} ({tag})", [x, y], kind))

    # Reasoning ablation: does the cross-family signal survive without the
    # reasoning model? If the full-pool result collapses here, "cross-family"
    # was really "the one model that thinks before answering".
    no_reasoning = [a.name for a in agents if not agent_is_reasoning(a)]
    if 2 <= len(no_reasoning) < len(names):
        specs.append(PanelSpec("full_no_reasoning",
                               f"full pool minus reasoning models (N={len(no_reasoning)})",
                               no_reasoning, "ablation"))

    # Matched-N swap: same panel size, one non-reasoning model replaced by a
    # reasoning one. full vs full_no_reasoning differs in N as well as in
    # inference mode, so it cannot isolate either.
    reasoning = [a.name for a in agents if agent_is_reasoning(a)]
    if len(no_reasoning) >= 2 and reasoning:
        for rname in reasoning:
            # If this reasoning agent has a same-model twin in the base panel,
            # drop THAT twin — then the swap differs from full_no_reasoning in
            # exactly one bit: reasoning on vs off, same weights, everything
            # else held constant. Dropping an arbitrary agent instead (the old
            # no_reasoning[:-1]) would vary model identity and inference mode
            # together, which is the confound the twin exists to remove.
            twin = [n for n in no_reasoning if model_of[n] == model_of[rname]]
            base = ([n for n in no_reasoning if n != twin[0]] if twin
                    else no_reasoning[:-1])
            specs.append(PanelSpec(
                f"swap:{short(rname)}",
                f"non-reasoning base, one swapped for {short(rname)}"
                + (" (same-weights twin)" if twin else ""),
                base + [rname], "ablation"))

    # Tier-matched panels: one core model per family per tier, so family
    # composition and panel size (N=3) are held constant while nominal
    # capability tier varies. See MODEL_TIER's docstring — tiers are a design
    # scaffold, not a measured ranking.
    by_tier = defaultdict(list)
    for a in agents:
        if a.model in CORE_MODELS:
            by_tier[tier_of(a.model)].append(a.name)
    for t in ("small", "large", "strong"):
        if len(by_tier[t]) >= 2:
            specs.append(PanelSpec(f"tier:{t}", f"{t}-tier panel (N={len(by_tier[t])})",
                                   by_tier[t], "tier"))

    # Leave-one-out panels: core minus one agent, so that agent's per-agent
    # AUROC is scored against a panel that never includes its own answer.
    for n in core:
        specs.append(PanelSpec(f"loo:{short(n)}", f"core minus {short(n)}",
                               [m for m in core if m != n], "loo"))

    return specs


def score_panel(spec: PanelSpec, texts_by_name: Dict[str, str],
                excluded_by_name: Dict[str, bool],
                jaccard, semantic, mc_exact, question: str, is_mc: bool) -> dict:
    """Disagreement over one panel, with excluded responses removed first.

    `excluded_by_name` covers both abstentions (model behaviour: declining to
    answer is not a dissent) and truncations (measurement failure: an empty or
    cut-off response would form its own NLI cluster and inflate disagreement).
    Panels left with fewer than two usable answers get None — undefined, not
    zero. The row records the two causes separately; only the union matters
    here, since both mean "no comparable answer from this agent".

    In MC mode the primary measure is `disagreement.MCExactMatch`, not
    semantic entropy: NLI on "A" vs "B" has no signal, and Jaccard over
    one-token strings is degenerate. The score is still stored under the
    "semantic_entropy" key below so every report function downstream keeps
    working unmodified — check "measure" in the returned dict (or the report
    section headers) to see which one it actually is. n_clusters/
    cluster_sizes/cluster_of are still populated in MC mode too, grouped by
    MCExactMatch's own extracted letter rather than by NLI entailment, so
    downstream panel-level logic (dominant_wrong, minority-cluster
    membership) keeps working on real clusters rather than degenerating.
    """
    attempted = [n for n in spec.members if not excluded_by_name[n]]
    n_excluded = len(spec.members) - len(attempted)
    measure = "mc_exact_match" if is_mc else "semantic_entropy"
    if len(attempted) < 2:
        return {"jaccard": None, "semantic_entropy": None, "n_clusters": None,
                "cluster_sizes": None, "n_scored": len(attempted),
                "n_excluded": n_excluded, "cluster_of": {}, "measure": measure}

    texts = [texts_by_name[n] for n in attempted]
    jac = jaccard.score(texts).score

    if is_mc:
        mc = mc_exact.score(texts)
        score_val = mc.score
        # Group attempted agents by their extracted letter. An unparseable
        # answer (letter=None) is never merged with anything — including
        # another unparseable one, since two unreadable answers are not
        # known to agree on a letter.
        letters = mc.details["extracted_letters"]
        by_letter: Dict[str, List[int]] = {}
        clusters: List[List[int]] = []
        for idx, letter in enumerate(letters):
            if letter is None:
                clusters.append([idx])
            else:
                by_letter.setdefault(letter, []).append(idx)
        clusters.extend(by_letter.values())
        n_clusters = len(clusters)
        cluster_sizes = [len(c) for c in clusters]
    else:
        sem = semantic.score(texts, question=question)
        score_val = sem.score
        clusters = sem.details["clusters"]
        n_clusters = sem.details["n_clusters"]
        cluster_sizes = sem.details["cluster_sizes"]

    cluster_of = {}
    for cid, members in enumerate(clusters):
        for member_idx in members:
            cluster_of[attempted[member_idx]] = cid

    return {"jaccard": jac, "semantic_entropy": score_val,
            "n_clusters": n_clusters, "cluster_sizes": cluster_sizes,
            "n_scored": len(attempted), "n_excluded": n_excluded,
            "cluster_of": cluster_of, "measure": measure}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalise(s: str) -> str:
    return _PUNCT_RE.sub(" ", (s or "").lower()).split() and " ".join(
        _PUNCT_RE.sub(" ", (s or "").lower()).split()) or ""


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (s or "").lower()).split())


def _numbers(s: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", (s or "").replace(",", "")))


def _date_parts(s: str) -> set:
    """{('y','2016'), ('m',12), ('d','6')} — order- and format-agnostic, so
    '28 July 2021' and 'July 28, 2021' compare equal."""
    t = _norm(s)
    parts = set()
    years = set(re.findall(r"\b(1\d{3}|20\d{2})\b", t))
    parts |= {("y", y) for y in years}
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b|\b{name[:3]}\b", t):
            parts.add(("m", num))
    for n in re.findall(r"\b(\d{1,2})\b", t):
        if n not in years:
            parts.add(("d", str(int(n))))
    return parts


def grade_short_answer(text: str, gold: str, answer_type: str):
    """True / False for short-form golds; None to defer to NLI or the judge."""
    gold_main = re.split(r"\(acceptable range", gold, flags=re.I)[0].strip()
    g, t = _norm(gold_main), _norm(text)
    if not g or not t:
        return None
    if g in t:
        return True

    at = (answer_type or "").lower()
    if at == "number":
        gn, tn = _numbers(gold_main), _numbers(text)
        return gn.issubset(tn) if (gn and tn) else None     # was: not isdisjoint
    if at == "date":
        gp, tp = _date_parts(gold_main), _date_parts(text)
        return gp.issubset(tp) if (gp and tp) else None

    if len(g.split()) > 4:
        return None                      # long gold — entailment is the right tool

    # Defer to the judge only on a DISTINCTIVE partial match. Deferring on a
    # shared stopword or common forename sent "John Burritt" to the NLI against
    # gold "John Strahan French" and got CORRECT back, three times on
    # simpleqa-2666.
    STOP = {"the", "of", "a", "an", "de", "la", "el", "st", "saint",
            "john", "mary", "jean", "juan"}
    gt = {w for w in g.split() if w not in STOP and len(w) > 2}
    if gt and gt & set(t.split()):
        return None                      # partial name match — ambiguous, defer
    return False

def grade_one(nli, text, sample, abstained, use_exact, is_mc):
    if abstained or not (text or "").strip():
        return None
    if is_mc:
        # Letter comparison only — grade_short_answer is built for short
        # factual golds and would compare a letter against an answer string.
        return grade_correct(nli, text, sample, prompt_format="mc")
    if use_exact:
        v = grade_short_answer(text, sample.correct_answer,
                               getattr(sample, "answer_type", None))
        if v is not None:
            return v
    return grade_correct(nli, text, sample, prompt_format="open")


JUDGE_PROMPT = """You are grading a short factual answer against a gold answer.

Question: {question}
Gold answer: {gold}
Candidate answer: {answer}

Reply with exactly one word:
CORRECT      - the candidate contains the gold answer, allowing for paraphrase,
               different formatting, or extra detail that does not contradict it.
INCORRECT    - the candidate states something that conflicts with the gold answer.
NOT_ATTEMPTED - the candidate declines, hedges without committing, or does not
               state an answer at all.

One word only."""


def judge_grade(judge_agent, question: str, gold: str, answer: str,
                attempts: int = 3) -> Optional[bool]:
    """Independent LLM grading, for the grader-independence check.

    The default grader shares its NLI instance with the clustering measure, so
    their errors are correlated in the direction that inflates 'disagreement
    predicts wrongness'. This gives a second opinion from a model outside the
    pool.
    """
    if not (answer or "").strip():
        return None
    prompt = JUDGE_PROMPT.format(question=question, gold=gold,
                                 answer=answer.strip()[:2000])
    last = (None, None)
    for i in range(attempts):
        # Only the first attempt may read the cache: if a bad verdict was
        # somehow stored, retrying against it would return the same failure
        # forever. GroqAgent.query's cache-write guard should prevent that,
        # but the retry must not depend on that guard being airtight.
        r = judge_agent.query(prompt, use_cache=(i == 0))
        if not r.is_error and (r.text or "").strip():
            break
        last = (r.finish_reason, r.error)
        time.sleep(2 ** i)
    else:
        raise RuntimeError(
            f"Judge returned no usable verdict after {attempts} attempts "
            f"(last: finish_reason={last[0]}, error={last[1]}). A silent None "
            f"here would grade every response NOT_ATTEMPTED. Check the judge's "
            f"token budget and reasoning params.")
    verdict = (r.text or "").strip().upper()
    if "NOT_ATTEMPTED" in verdict:
        return None
    if verdict.startswith("CORRECT") or "CORRECT" in verdict.split()[:1]:
        return True
    if "INCORRECT" in verdict:
        return False
    if "CORRECT" in verdict:
        return True
    return None


# ---------------------------------------------------------------------------
# Provenance manifest — mirrors scripts/harvest.py's _write_manifest /
# _check_manifest_compatible, so --resume can refuse to silently mix rows
# produced under different settings into one file. _git_sha itself is
# imported from there rather than reimplemented (see the import at the top).
# ---------------------------------------------------------------------------

def write_manifest(path: Path, *, agents, args) -> None:
    a0 = agents[0]   # _assert_uniform_query_settings guarantees these match pool-wide
    manifest = {
        "dataset": args.dataset,
        "prompt_format": args.prompt_format,
        "n": args.n,
        "seed": args.seed,
        "agents": [
            {"name": a.name, "model": a.model, "family": family_of(a.model),
             "tier": tier_of(a.model), "reasoning_params": a.reasoning_params}
            for a in agents
        ],
        # The settings actually used, read off the built agents rather than
        # off DATASETS[...]["config"] — --no-concise means the pool was
        # built with the dataset config skipped entirely, and the manifest
        # must record what happened, not what would have happened by default.
        "max_tokens": a0.max_tokens,
        "temperature": a0.temperature,
        "system_prompt": a0.system_prompt,
        "nli_model": args.nli_model,
        "clustering": {"strict_entailment": args.strict,
                       "linkage": "single" if args.single_linkage else "complete"},
        "judge_model": args.judge_model or None,
        "judge_provider": (args.judge_provider if args.judge_model else None),
        "git_commit": _git_sha(),
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_manifest_compatible(manifest: dict, *, agents, args) -> None:
    """Refuse to --resume under different settings.

    The agent-name check further down in run_queries catches a changed pool,
    but not a changed max_tokens, prompt format, or NLI model — any of which
    would silently mix incomparable rows into one file. This is that check,
    against the manifest written alongside --out.
    """
    a0 = agents[0]
    checks = {
        "dataset": (manifest.get("dataset"), args.dataset),
        "prompt_format": (manifest.get("prompt_format"), args.prompt_format),
        "seed": (manifest.get("seed"), args.seed),
        "agent_names": (sorted(a["name"] for a in manifest.get("agents", [])),
                        sorted(a.name for a in agents)),
        "max_tokens": (manifest.get("max_tokens"), a0.max_tokens),
        "system_prompt": (manifest.get("system_prompt"), a0.system_prompt),
        "nli_model": (manifest.get("nli_model"), args.nli_model),
    }
    mismatches = [f"{k} ({old!r} != {new!r})" for k, (old, new) in checks.items() if old != new]
    if mismatches:
        raise RuntimeError(
            "--resume: existing manifest doesn't match this invocation: "
            + "; ".join(mismatches) + ". Resuming would silently mix "
            "incomparable rows into one file — use a different --out."
        )


# ---------------------------------------------------------------------------
# Query phase
# ---------------------------------------------------------------------------

def run_queries(samples, agents, jaccard, semantic, mc_exact, nli, panel_specs,
                out_path: Path, args, judge_agent, dataset_cfg: dict,
                is_mc: bool) -> List[dict]:
    section("Scoring")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    meta_fields = dataset_cfg["meta"]
    prompt_format = args.prompt_format

    names = [a.name for a in agents]
    core_names = next(s.members for s in panel_specs if s.key == "core")
    rows: List[dict] = []
    done_uids: set = set()

    if args.resume:
        print("  [!] --resume is only valid when the pool, question set and seed are "
              "unchanged from the run that produced the existing file — a resumed "
              "run that silently mixes rows from two different pools would leave "
              "`responses` dicts with inconsistent keys, and every panel comparison "
              "downstream would be wrong in a way that is hard to spot.")
        if out_path.exists():
            if manifest_path.exists():
                old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                check_manifest_compatible(old_manifest, agents=agents, args=args)
            else:
                print(f"  [!] {out_path} exists but {manifest_path.name} does not — "
                      f"proceeding, but provenance for its existing rows can't be "
                      f"verified.")
            n_bad_lines = 0
            existing_names: Optional[set] = None
            for line in out_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A partially-written final line from a killed process —
                    # skip it rather than letting it abort the resume.
                    n_bad_lines += 1
                    continue
                if existing_names is None:
                    existing_names = set(row["responses"])
                    if existing_names != set(names):
                        raise RuntimeError(
                            "--resume: existing results were produced by a "
                            f"different agent pool.\n  existing agents: "
                            f"{sorted(existing_names)}\n  current agents : "
                            f"{sorted(names)}\nRefusing to append — this would "
                            "leave `responses` dicts with inconsistent keys "
                            "across rows.")
                rows.append(row)
                done_uids.add(row["uid"])
            if n_bad_lines:
                print(f"  [resume] ignored {n_bad_lines} malformed line(s) in "
                      f"{out_path} (a partial write from an interrupted run)")
            print(f"  [resume] {len(done_uids)} question(s) already in {out_path}; "
                  f"skipping them.")
            samples = [s for s in samples if s.uid not in done_uids]
        else:
            print(f"  [resume] {out_path} does not exist yet; starting fresh.")
    else:
        out_path.write_text("", encoding="utf-8")   # truncate; never append to a stale run

    # Written (or, on a compatible --resume, re-written identically) before
    # the first query goes out, so even a hard-killed run leaves a manifest
    # a later --resume can check itself against.
    write_manifest(manifest_path, agents=agents, args=args)

    n_bad_questions = 0
    skipped: List[Tuple[str, str]] = []

    for idx, s in enumerate(samples, start=1):
        responses = query_agents(agents, select_prompt(s, prompt_format))

        # An API error aborts the question outright — unlike a truncation
        # (handled below), there is no partial response to fall back on.
        errored = [(a, r) for a, r in zip(agents, responses) if r.is_error]
        if errored:
            n_bad_questions += 1
            for a, r in errored:
                print(f"  [{idx:>3}/{len(samples)}] {s.uid}  [!] {a.name}: {r.error}")
                skipped.append((s.uid, f"{short(a.name)}: error"))
            permanent = any(r.error.startswith(PERMANENT_ERROR_PREFIX) for _, r in errored)
            if permanent or n_bad_questions > args.max_bad_questions:
                raise RuntimeError(
                    f"Aborting after {n_bad_questions} unusable question(s). "
                    f"{len(rows)} good row(s) already in {out_path}."
                )
            continue

        # A truncated response is a failed measurement for THAT AGENT, not for
        # the question. Excluded like an abstention, but counted separately:
        # an abstention is model behaviour, a truncation is our budget failing.
        unusable = {a.name: (r.finish_reason == "length" or not (r.text or "").strip())
                    for a, r in zip(agents, responses)}
        for n, bad in unusable.items():
            if bad:
                print(f"  [{idx:>3}/{len(samples)}] {s.uid}  [~] {short(n)}: "
                      f"truncated, excluded from this question")

        texts = {a.name: r.text for a, r in zip(agents, responses)}
        # In MC mode, "not attempted" means no letter could be extracted from
        # the response at all — the prose abstention regexes never fire on a
        # single letter, so they're not the signal here.
        if is_mc:
            abstained = {n: (not unusable[n]) and _extract_mc_letter(t) is None
                        for n, t in texts.items()}
        else:
            abstained = {n: (not unusable[n]) and is_abstention(t) for n, t in texts.items()}
        excluded = {n: unusable[n] or abstained[n] for n in texts}

        if sum(not v for v in excluded.values()) < 2:
            n_bad_questions += 1
            print(f"  [{idx:>3}/{len(samples)}] {s.uid}  [!] fewer than 2 usable answers")
            if n_bad_questions > args.max_bad_questions:
                raise RuntimeError(f"Aborting: {n_bad_questions} unscorable questions.")
            continue

        grades = {n: grade_one(nli, texts[n], s, excluded[n], not args.no_exact_match, is_mc)
                  for n in names}
        judged = None
        if judge_agent is not None:
            judged = {n: (None if excluded[n] else
                          judge_grade(judge_agent, s.question, s.correct_answer, texts[n]))
                      for n in names}

        panels = {spec.key: score_panel(spec, texts, excluded, jaccard, semantic,
                                        mc_exact, s.question, is_mc)
                  for spec in panel_specs}

        core_panel = panels["core"]
        graded_core = [grades[n] for n in core_names if grades.get(n) is not None]
        row = {
            "uid": s.uid,
            **{f: getattr(s, f, None) for f in meta_fields},
            # Reporting code groups by r["topic"]; TruthfulQA calls it
            # `category`. Alias rather than branching in every report function.
            "topic": getattr(s, "topic", None) or getattr(s, "category", None),
            "question": s.question,
            "correct_answer": s.correct_answer,
            "responses": texts,
            "finish_reasons": {a.name: r.finish_reason for a, r in zip(agents, responses)},
            "completion_tokens": {a.name: (r.usage or {}).get("completion_tokens")
                                  for a, r in zip(agents, responses)},
            "unusable": unusable,
            "excluded": excluded,
            "abstained": abstained,
            "n_abstained": sum(abstained.values()),
            "n_unusable": sum(unusable.values()),
            "grades": grades,
            "grades_judge": judged,
            "panels": panels,
            # Convenience mirrors of the CORE panel (the frozen 3×3 pool) —
            # the headline numbers, reproducible regardless of what else
            # --models adds to the queried pool. "full" (all queried agents)
            # stays available under panels["full"] and is reported separately
            # as an exploratory extension when it differs from core.
            "jaccard": core_panel["jaccard"],
            "semantic_entropy": core_panel["semantic_entropy"],
            "n_clusters": core_panel["n_clusters"],
            "cluster_sizes": core_panel["cluster_sizes"],
            "cluster_of": core_panel["cluster_of"],
            "panel_agrees": (core_panel["n_clusters"] == 1) if core_panel["n_clusters"] else None,
            "panel_majority_wrong": (bool(graded_core) and
                                     sum(g is False for g in graded_core) > len(graded_core) / 2),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        mtag = "mc" if is_mc else "sem"
        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} [{row['topic']}] ---")
            print(f"      Q: {s.question}")
            print(f"      gold: {s.correct_answer}")
            for n in names:
                gl = {True: "OK  ", False: "WRONG", None: "N/A "}[grades[n]]
                ab = " [abstained]" if abstained[n] else ""
                print(f"      [{gl}] {short(n)}{ab}: {texts[n].strip()[:200]}")
            print(f"      -> clusters={core_panel['cluster_sizes']} "
                  f"{mtag}={core_panel['semantic_entropy']}")

        sem_s = ("  n/a" if core_panel["semantic_entropy"] is None
                  else f"{core_panel['semantic_entropy']:.2f}")
        jac_s = "  n/a" if core_panel["jaccard"] is None else f"{core_panel['jaccard']:.2f}"
        ex_s = ""
        if any(excluded.values()):
            ex_s = f"  abst={sum(abstained.values())} trunc={sum(unusable.values())}"
        print(f"  [{idx:>3}/{len(samples)}] {s.uid}  jac={jac_s}  {mtag}={sem_s}  "
              f"clusters={core_panel['cluster_sizes']}{ex_s}")

    if skipped:
        print(f"\n  [!] {n_bad_questions} question(s) skipped as unusable:")
        for uid, why in skipped[:15]:
            print(f"        {uid}: {why}")
        if len(skipped) > 15:
            print(f"        ... and {len(skipped) - 15} more")
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_abstention(rows, names) -> None:
    section("0. Abstention and truncation — per-agent behaviour")
    print("  ABSTAIN is model behaviour: declining to answer is not a dissent, so")
    print("  it is excluded from clustering rather than scored as disagreement.")
    print("  TRUNC is our measurement failing: the answer hit the token ceiling.")
    print("  Both are excluded from panels; they are counted apart because only")
    print("  one of them is a property of the model.")
    print("  TOKENS is the median completion-token cost per answered question —")
    print("  the reasoning-on vs reasoning-off ratio is itself a result.\n")
    print(f"    {'agent':<34} {'abstain':>12} {'trunc':>12} {'tokens':>8}")
    for n in names:
        ab = sum(1 for r in rows if r.get("abstained", {}).get(n))
        tr = sum(1 for r in rows if r.get("unusable", {}).get(n))
        toks = [r["completion_tokens"][n] for r in rows
                if r.get("completion_tokens", {}).get(n)
                and not r.get("excluded", {}).get(n)]
        med = f"{statistics.median(toks):.0f}" if toks else "-"
        print(f"    {short(n):<34} {ab:>4}/{len(rows)} ({ab/max(len(rows),1):>3.0%}) "
              f"{tr:>4}/{len(rows)} ({tr/max(len(rows),1):>3.0%}) {med:>8}")

    dropped = [r for r in rows
               if not (r["panels"].get("core") and r["panels"]["core"]["semantic_entropy"] is not None)]
    print(f"\n    questions with <2 usable core answers (unscorable): {len(dropped)}")
    worst = max(((sum(1 for r in rows if r.get("unusable", {}).get(n)), n)
                 for n in names), default=(0, None))
    if worst[0] > 0.1 * len(rows):
        print(f"    [!] {short(worst[1])} truncates on {worst[0]/len(rows):.0%} of")
        print("        questions. These are not random — they cluster on questions the")
        print("        model cannot answer, so any panel containing it runs on an")
        print("        easier subset. Report its completion rate alongside its results.")
    print("    Audit the abstention patterns with --show-abstentions: a false")
    print("    positive there silently deletes a genuine dissent.")

def show_abstention_examples(rows, names, k: int = 12) -> None:
    section("Abstention examples (audit the regexes)")
    shown = 0
    for r in rows:
        for n in names:
            if r["abstained"].get(n) and shown < k:
                print(f"  {short(n):<28} {r['responses'][n].strip()[:110]!r}")
                shown += 1
    if not shown:
        print("  None detected.")


def report_measures(rows, names, core_names, panel_specs, args, is_mc) -> None:
    measure_label = "MC exact-match" if is_mc else "semantic entropy"
    mtag = "mc" if is_mc else "sem"
    section(f"A. Answer-level (Jaccard) vs {measure_label} — frozen core pool")

    # Tolerant to rows from an older results file that predates the core/tier/
    # loo panels (Edits 3-5): r["panels"].get("core") is None for those, so
    # they simply score 0 here rather than raising a KeyError.
    scored = [r for r in rows
              if r["panels"].get("core") and r["panels"]["core"]["semantic_entropy"] is not None]
    jac = [r["jaccard"] for r in scored]
    sem = [r["semantic_entropy"] for r in scored]
    print(f"  scorable questions: {len(scored)} / {len(rows)}")
    if not scored:
        print("  [!] No row has a scorable core panel — this results file predates")
        print("      the core/tier/LOO panels (Edits 3-5) and needs re-querying")
        print("      against the current 3x3 pool. Nothing else in this section can")
        print("      be computed; see panels['full'] in the raw rows for the old")
        print("      full-pool numbers instead.")
        return
    print(f"  mean Jaccard          : {statistics.mean(jac):.3f}")
    print(f"  mean {measure_label:<17}: {statistics.mean(sem):.3f}")
    if is_mc:
        print("  [note] These are not comparable either. Jaccard here is raw token")
        print("  overlap on a single-letter response — a weak, noisy secondary signal,")
        print("  not a comparison point for mc_exact_match. Use the AUROCs below.")
    else:
        print("  [note] These means are NOT comparable. Jaccard is continuous;")
        print("  normalised entropy over k singletons is exactly 1.0. Their")
        print("  difference is a scale artifact, not a result. Use the AUROCs below.")

    if len(names) > len(core_names):
        extra = [n for n in names if n not in core_names]
        print(f"\n  [exploratory] full pool has {len(extra)} agent(s) beyond the frozen "
              f"core: {', '.join(short(n) for n in extra)}")
        scored_full = [r for r in rows
                       if r["panels"].get("full") and r["panels"]["full"]["semantic_entropy"] is not None]
        if scored_full:
            jac_full = [r["panels"]["full"]["jaccard"] for r in scored_full]
            sem_full = [r["panels"]["full"]["semantic_entropy"] for r in scored_full]
            print(f"    scorable questions (full) : {len(scored_full)} / {len(rows)}")
            print(f"    mean Jaccard (full)       : {statistics.mean(jac_full):.3f}")
            print(f"    mean semantic entropy (full): {statistics.mean(sem_full):.3f}")
        print("    Exploratory only — this number moves whenever --models changes,")
        print("    unlike the core figures above, which are reproducible regardless.")

    cc = Counter(r["n_clusters"] for r in scored)
    print(f"\n  n_clusters spread     : " + ", ".join(f"{k}->{cc[k]}" for k in sorted(cc)))

    if is_mc:
        print("\n  Spearman(jac, mc_exact_match) skipped in MC mode: both measures are")
        print("  computed over single letters, so there is no lexical-vs-semantic")
        print("  distinction to make.")
    else:
        rho = spearman(jac, sem)
        distinct = len(set(sem))
        biggest_tie = max(Counter(sem).values()) if sem else 0
        print(f"  Spearman(jac, sem)    : {rho:+.3f}")
        print(f"  [note] sem takes only {distinct} distinct value(s) here, largest tie "
              f"block {biggest_tie}/{len(sem)}.")
        print("  Ties bound |rho| well below 1 regardless of agreement, so read the")
        print("  sign and rough magnitude only. Strongly negative per-agent AUROC for")
        print("  Jaccard would mean token overlap predicts WRONGNESS — plausible if")
        print("  agents hedge in a shared house style when ignorant.")

    grade_dist_label = "letter match" if is_mc else "NLI"
    auroc_label = "letter-match grader" if is_mc else "NLI grader"
    grade_ctr = Counter()
    for r in scored:
        for g in r["grades"].values():
            grade_ctr[{True: "correct", False: "incorrect", None: "not_attempted"}[g]] += 1
    tot = sum(grade_ctr.values()) or 1
    print(f"\n  grades ({grade_dist_label})          : " + ", ".join(
        f"{k}={grade_ctr[k]} ({grade_ctr[k]/tot:.0%})"
        for k in ("correct", "incorrect", "not_attempted")))

    if any(r.get("grades_judge") for r in scored):
        jg = Counter()
        agree = tot_cmp = 0
        for r in scored:
            for n in names:
                a, b = r["grades"][n], (r["grades_judge"] or {}).get(n)
                jg[{True: "correct", False: "incorrect", None: "not_attempted"}[b]] += 1
                tot_cmp += 1
                agree += (a == b)
        print(f"  grades (LLM judge)    : " + ", ".join(
            f"{k}={jg[k]} ({jg[k]/max(tot_cmp,1):.0%})"
            for k in ("correct", "incorrect", "not_attempted")))
        print(f"  grader agreement      : {agree}/{tot_cmp} ({agree/max(tot_cmp,1):.0%})")

        print("\n  Abstention regex vs judge (NOT_ATTEMPTED):")
        tp = fp = fn = tn = 0
        for r in scored:
            for n in names:
                if not r.get("grades_judge"):
                    continue
                regex_says = bool(r["abstained"].get(n))
                judge_says = r["grades_judge"].get(n) is None
                tp += regex_says and judge_says
                fp += regex_says and not judge_says
                fn += (not regex_says) and judge_says
                tn += (not regex_says) and (not judge_says)
        print(f"    both abstention: {tp}   regex only (FALSE POSITIVE): {fp}")
        print(f"    judge only (MISS): {fn}   both answered: {tn}")
        print("    A regex false positive deletes a real answer from every panel it")
        print("    appears in — worse than a miss. Inspect any nonzero fp by hand.")

    # -- the numbers that actually answer the RQ ---------------------------
    print("\n  AUROC — does disagreement predict a wrong answer?")
    for grade_key, label in (("grades", auroc_label), ("grades_judge", "LLM judge")):
        if grade_key == "grades_judge" and not any(r.get("grades_judge") for r in scored):
            continue
        print(f"\n    [{label}]")
        if grade_key == "grades" and not is_mc:
            print("    WARNING: this grader shares its NLI instance with the")
            print("    clustering measure. With k agents, k distinct clusters nearly")
            print("    forces at most one agent to entail the gold, so a high sem")
            print("    AUROC here is partly circular. Compare against the judge.")

        labels = []
        for r in scored:
            g = [v for v in (r[grade_key] or {}).values() if v is not None] \
                if r.get(grade_key) else []
            labels.append(bool(g) and sum(v is False for v in g) > len(g) / 2)
        for mname, scores in (("jaccard", jac), ("semantic_entropy", sem)):
            a, why = auroc_guarded(scores, labels)
            val = f"{a:.3f}" if a is not None else f"n/a — {why}"
            print(f"      {mname:<18} vs majority-wrong : {val}")

        print("      per-agent (label = that agent graded incorrect):")
        print(f"      {mtag}(self) scores against the CORE panel including that agent's")
        print(f"      own answer (mild self-inclusion). {mtag}(LOO) scores against the")
        print("      core panel with that agent's own answer removed — use LOO as")
        print("      the primary figure; self is shown alongside to document the gap.")
        for n in names:
            loo_key = f"loo:{short(n)}"
            has_loo = any(r["panels"].get(loo_key) for r in scored)

            pairs = [(r, (r[grade_key] or {}).get(n)) for r in scored if r.get(grade_key)]
            pairs = [(r, g) for r, g in pairs if g is not None]
            labs = [g is False for _, g in pairs]
            a_j, why_j = auroc_guarded([r["jaccard"] for r, _ in pairs], labs)
            a_s_self, _ = auroc_guarded([r["semantic_entropy"] for r, _ in pairs], labs)

            a_s_loo, n_loo, n_loo_dropped = None, 0, 0
            if has_loo:
                loo_pairs = [(r, g) for r, g in pairs
                             if r["panels"].get(loo_key)
                             and r["panels"][loo_key]["semantic_entropy"] is not None]
                n_loo = len(loo_pairs)
                n_loo_dropped = len(pairs) - n_loo
                loo_labs = [g is False for _, g in loo_pairs]
                loo_scores = [r["panels"][loo_key]["semantic_entropy"] for r, _ in loo_pairs]
                a_s_loo, _ = auroc_guarded(loo_scores, loo_labs)

            if a_j is None and a_s_self is None and a_s_loo is None:
                print(f"        {short(n):<32} suppressed — {why_j}")
                continue
            f = lambda v: f"{v:.3f}" if v is not None else " n/a "
            line = (f"        {short(n):<32} n={len(pairs):<4} "
                    f"err={sum(labs)/max(len(labs),1):.0%}  "
                    f"jac={f(a_j)}  {mtag}(self)={f(a_s_self)}")
            if has_loo:
                line += f"  {mtag}(LOO)={f(a_s_loo)} (n={n_loo}"
                if n_loo_dropped:
                    line += f", {n_loo_dropped} dropped <2 usable"
                line += ")"
            print(line)

    if is_mc:
        print("\n  Pairwise high-Jaccard-but-same-meaning diagnostic skipped in MC mode:")
        print("  both measures are over single letters, so there is no lexical-vs-")
        print("  semantic distinction to make.")
    else:
        pair_keys = [s.key for s in panel_specs if s.kind in ("within", "cross")]
        n_fp = n_tot = 0
        examples = []
        for r in scored:
            for k in pair_keys:
                p = r["panels"].get(k)
                if p and p["semantic_entropy"] is not None:
                    n_tot += 1
                    if p["jaccard"] >= 0.5 and p["semantic_entropy"] == 0.0:
                        n_fp += 1
                        if len(examples) < 5:
                            examples.append((r, k, p["jaccard"]))
        print(f"\n  Pairwise: high Jaccard (>=0.50) but the pair agrees in meaning: "
              f"{n_fp}/{n_tot} agent-pairs ({n_fp/max(n_tot,1):.1%})")
        print("  Measured pairwise, not on the full core panel: with 9 agents a core")
        print("  semantic entropy of exactly 0 needs all nine in one cluster, which")
        print("  essentially never happens, so that version of the diagnostic reads 0")
        print("  regardless of the underlying phenomenon. At N=2 a single cluster just")
        print("  means 'these two agree', which stays comparable as the pool grows and")
        print("  across datasets. Expect this to be far higher on TruthfulQA — the")
        print("  false-positive reduction motivating semantic entropy is largely a")
        print("  long-form phenomenon.")
        for r, k, j in examples:
            print(f"    - {r['uid']} [{r['topic']}]  {k}  jac={j:.2f}")
            print(f"        Q: {r['question'][:80]}")


def report_panels(rows, agents, panel_specs, args, is_mc) -> None:
    # Only annotated with the measure name in MC mode — the open-ended title
    # is unchanged so results files from before --dataset/--prompt-format
    # existed report identically (see disagreement_pilot's regression check
    # against the pre-refactor script).
    title = "B. Shared-bias test — matched-N panels (family isolated from size)"
    if is_mc:
        title = "B. Shared-bias test — matched-N panels (MC exact-match, family isolated from size)"
    section(title)

    def mean_of(spec_keys, field="semantic_entropy"):
        vals = [r["panels"][k][field] for r in rows for k in spec_keys
                if r["panels"].get(k) and r["panels"][k][field] is not None]
        return (statistics.mean(vals), len(vals)) if vals else (None, 0)

    within = [s.key for s in panel_specs if s.kind == "within"]
    cross = [s.key for s in panel_specs if s.kind == "cross"]
    if not within or not cross:
        print("  Need >=2 families with >=2 members each for a matched comparison.")
        print("  Add a second model from at least one non-Llama family to the pool.")
        return

    mw, nw = mean_of(within)
    mc, nc = mean_of(cross)
    print(f"  UNPAIRED (different question subsets — see paired figure below)")
    print(f"    within-family pairs (N=2): {mw:.3f}  [{len(within)} pairs, {nw} obs]")
    print(f"    cross-family  pairs (N=2): {mc:.3f}  [{len(cross)} pairs, {nc} obs]")

    def avail_mean(r, keys):
        """Mean over the panels that scored on THIS question, or None."""
        vals = [r["panels"][k]["semantic_entropy"] for k in keys
                if r["panels"].get(k)
                and r["panels"][k]["semantic_entropy"] is not None]
        return statistics.mean(vals) if vals else None

    paired = []
    for r in rows:
        w, c = avail_mean(r, within), avail_mean(r, cross)
        if w is not None and c is not None:
            paired.append((r, c - w))
    diffs = [d for _, d in paired]
    if diffs:
        pos = sum(d > 0 for d in diffs)
        neg = sum(d < 0 for d in diffs)
        print(f"\n  PAIRED within→cross difference: {statistics.mean(diffs):+.3f}  "
              f"(n={len(diffs)}/{len(rows)} questions with >=1 pair scorable in "
              f"each bucket)")
        print(f"    higher on cross: {pos}   higher on within: {neg}   "
              f"tied: {len(diffs) - pos - neg}")
        if pos + neg >= 6:
            res = wilcoxon(diffs)
            # Rank-biserial is the effect size that belongs with this test:
            # p says the sign is reliable, r says whether it matters. At
            # n=1000 a trivial difference will be significant.
            rbc = (pos - neg) / (pos + neg)
            print(f"    Wilcoxon signed-rank p = {res.pvalue:.2e}, "
                  f"rank-biserial r = {rbc:+.3f}")
        print("  Each question contributes one within and one cross score, so the")
        print("  question set no longer differs between the two panels.")
    else:
        print("\n  [!] No question is scorable in both buckets — nothing to pair.")

    

        
    print("\n  Per-pair breakdown:")
    for s in panel_specs:
        if s.kind not in ("within", "cross"):
            continue
        m, n = mean_of([s.key])
        if m is not None:
            print(f"    {s.kind:<7} {short(s.members[0]):<24} + "
                  f"{short(s.members[1]):<24} sem={m:.3f}  (n={n})")

    toggles = [s for s in panel_specs if s.kind == "toggle"]
    if toggles:
        print("\n  Reasoning toggle (same weights, same prompt — inference mode only):")
        for s in toggles:
            m, n = mean_of([s.key])
            if m is not None:
                print(f"    {short(s.members[0]):<24} + {short(s.members[1]):<24} "
                      f"sem={m:.3f}  (n={n})")
        print("  Pretraining held exactly constant here — this is how much")
        print("  disagreement inference mode alone produces.")

    swaps = [s for s in panel_specs if s.key.startswith("swap:")]
    if swaps and "full_no_reasoning" in {s.key for s in panel_specs}:
        print("\n  Reasoning ablation (matched N):")
        deltas = []
        for s in swaps:
            paired = common_rows(rows, ["full_no_reasoning", s.key])
            if not paired:
                continue
            d = [r["panels"][s.key]["semantic_entropy"]
                 - r["panels"]["full_no_reasoning"]["semantic_entropy"] for r in paired]
            md = statistics.mean(d)
            deltas.append(md)
            print(f"    swapping in {short(s.members[-1]):<24} {md:+.3f}  (n={len(paired)})")
            direction = ("raises" if md > 0 else "lowers" if md < 0 else "does not change")
            print(f"      → the reasoning model {direction} panel disagreement "
                  f"relative to the non-reasoning model it replaced.")
        if deltas and max(abs(d) for d in deltas) < 0.02:
            print("    Effects are near zero: inference mode is NOT what drives the")
            print("    cross-family signal — a useful negative result, not a failure.")
        elif deltas:
            print("    Effects are small but consistently signed; inference mode")
            print("    contributes a little to panel disagreement, far less than")
            print("    the capability gap between agents (see the per-pair table).")
        
    # Which agents dissent from a family that agrees among itself? Generalised
    # with 9 agents "exactly one agent outside
    # the pair's cluster" essentially never fires, so count minority-cluster membership
    print("\n  Minority-cluster membership on the core panel:")
    print("  How often each agent sits in a cluster smaller than the largest one —")
    print("  i.e. dissents from the panel's dominant reading. An agent that")
    print("  dominates this column IS the diversity signal, and the cross-family")
    print("  claim would then rest on that one model.")
    minority = Counter()
    seen_rows = Counter()
    for r in rows:
        p = r["panels"].get("core")
        if not p or not p.get("cluster_sizes"):
            continue
        sizes = p["cluster_sizes"]
        biggest = max(sizes)
        for n, cid in p["cluster_of"].items():
            seen_rows[n] += 1
            if sizes[cid] < biggest:
                minority[n] += 1
    for n, k in minority.most_common():
        print(f"    {short(n):<32} {k:>4}/{seen_rows[n]}  ({k/max(seen_rows[n],1):.0%})")

    def dominant_wrong(r, key="core", frac=0.5, require_homogeneous=True):
        """Shared bias on a majority basis: does one meaning-cluster hold more
        than `frac` of the scored agents, and is that cluster wrong?

        Strict unanimity (n_clusters == 1) gets monotonically harder to satisfy
        as the pool grows — 7 agents agreeing is far rarer than 4 — so the
        count shrinks with N even when the underlying failure is unchanged.
        This criterion is stable across pool sizes.
        """
        p = r["panels"].get(key)
        if not p or not p.get("cluster_sizes"):
            return None
        sizes = p["cluster_sizes"]
        n = sum(sizes)
        if max(sizes) <= frac * n:
            return None
        cid = sizes.index(max(sizes))
        members = [a for a, c in p["cluster_of"].items() if c == cid]
        graded = [r["grades"][a] for a in members if r["grades"].get(a) is not None]
        if not graded:
            return None
        # The dominant cluster must be grade-HOMOGENEOUS. A cluster holding
        # both "equilux" (correct) and "equinox" (wrong) is an NLI merge
        # failure, not a shared prior — counting it as a shared-bias false
        # negative attributes a clustering artifact to the models.
        if require_homogeneous and any(g is True for g in graded):
            return None
        if sum(g is False for g in graded) <= len(graded) / 2:
            return None
        return (cid, len(members), n)

    

    fn_strict = [r for r in rows if r["panel_agrees"] and r["panel_majority_wrong"]]
    fn_homog = [r for r in rows if dominant_wrong(r)]
    fn_any = [r for r in rows if dominant_wrong(r, require_homogeneous=False)]
    print(f"\n  Shared-bias FALSE NEGATIVES")
    print(f"    strict (whole pool in ONE cluster, and wrong)  : {len(fn_strict)} / {len(rows)}")
    print(f"    majority, grade-homogeneous cluster            : {len(fn_homog)} / {len(rows)}")
    print(f"    majority, cluster may be mixed                 : {len(fn_any)} / {len(rows)}")
    print(f"    The gap between the last two ({len(fn_any) - len(fn_homog)}) is the NLI")
    print("    clustering error rate on questions that would otherwise count as")
    print("    shared bias: a cluster holding both a correct and an incorrect")
    print("    answer is a merge failure, not a shared prior.")
    print("    Report the grade-homogeneous figure.")
    for r in fn_homog[:5]:
        print(f"    - {r['uid']} [{r['topic']}]  sem={r['semantic_entropy']:.2f}")
        print(f"        Q: {r['question'][:80]}")
        print(f"        gold: {r['correct_answer'][:80]}")
        for n, t in r["responses"].items():
            print(f"          {short(n):<26} {t.strip()[:70]}")

    # --- tier-matched ablation (small vs large vs strong, N=3 each) --------
    # Tolerant to an older results file: tier_specs is simply empty when the
    # rows/pool predate this panel, and the block below no-ops.
    tier_specs = {s.key.split(":", 1)[1]: s for s in panel_specs if s.kind == "tier"}
    if len(tier_specs) >= 2:
        print("\n  Tier-matched ablation (one core model per family per tier, N=3):")
        print("  Tier labels are a nominal design scaffold, not a measured capability")
        print("  ranking — cite each agent's measured error rate (Part A) rather than")
        print("  assuming 'strong' beats 'large' beats 'small'.")
        for t in ("small", "large", "strong"):
            s = tier_specs.get(t)
            if s is None:
                continue
            vals = [r["panels"][s.key]["semantic_entropy"] for r in rows
                    if r["panels"].get(s.key) and r["panels"][s.key]["semantic_entropy"] is not None]
            coverage = len(vals) / len(rows) if rows else 0.0
            mstr = f"{statistics.mean(vals):.3f}" if vals else "n/a"
            print(f"    {t:<8} sem={mstr:<8} coverage={coverage:>5.0%}  ({len(vals)}/{len(rows)})")

        tier_keys = [s.key for s in tier_specs.values()]
        tier_common = common_rows(rows, tier_keys)
        print(f"\n    Paired comparison, same questions across all {len(tier_keys)} tiers: "
              f"n={len(tier_common)}/{len(rows)}")
        if tier_common:
            for t in ("small", "large", "strong"):
                s = tier_specs.get(t)
                if s is None:
                    continue
                vals = [r["panels"][s.key]["semantic_entropy"] for r in tier_common]
                print(f"      {t:<8} sem={statistics.mean(vals):.3f}  (n={len(vals)})")
            print("    [!] llama-3.1-8b (small tier) is expected to abstain on roughly")
            print("    half of questions. The small tier's UNPAIRED coverage above is")
            print("    gated by its own confidence — the easy tail — so trust this")
            print("    paired figure, not the three independent means, when comparing")
            print("    tiers.")
        else:
            print("    No question is scorable across every tier — nothing to pair.")


def report_cross_tier(rows, agents, panel_specs, args, is_mc) -> None:
    """D. Cross-tier target/reference — can a tier panel detect another
    tier's errors?

    For each core agent as TARGET, AUROC of each tier panel (as REFERENCE)
    against that target's own graded correctness — restricted to rows where
    both are scorable and the target is not itself a member of the reference
    panel (skipped by construction whenever the target's own tier equals the
    reference tier, since a tier panel holds exactly one core model per
    family per tier).
    """
    # Same reasoning as report_panels: only annotated in MC mode, so the
    # open-ended title stays byte-identical to the pre-refactor script.
    title = "D. Cross-tier target/reference — capable model, weak references?"
    if is_mc:
        title += " (MC exact-match)"
    section(title)
    print("  Hypothesis: detection is weakest for a STRONG target checked by a")
    print("  SMALL-tier reference panel — a capable model checked by weak")
    print("  references is the realistic deployment case, and nothing else in")
    print("  this design measures it directly.\n")

    tier_specs = {s.key.split(":", 1)[1]: s for s in panel_specs if s.kind == "tier"}
    core_agents = [a for a in agents if a.model in CORE_MODELS]
    if not tier_specs or not core_agents:
        print("  No tier panels available (older results file, or pool has no core "
              "agents) — nothing to report.")
        return

    print(f"    {'target':<26} {'target tier':<12} {'ref tier':<9} {'n':>5} {'AUROC':>10}")
    for a in core_agents:
        target_tier = tier_of(a.model)
        for t, s in tier_specs.items():
            if t == target_tier:
                continue  # target is a member of its own tier panel by construction
            pairs = [(r, r["grades"].get(a.name)) for r in rows
                     if r["panels"].get(s.key) and r["panels"][s.key]["semantic_entropy"] is not None]
            pairs = [(r, g) for r, g in pairs if g is not None]
            scores = [r["panels"][s.key]["semantic_entropy"] for r, _ in pairs]
            labs = [g is False for _, g in pairs]
            auc, why = auroc_guarded(scores, labs)
            val = f"{auc:.3f}" if auc is not None else f"n/a ({why})"
            flag = "  <-- cell of interest" if target_tier == "strong" and t == "small" else ""
            print(f"    {short(a.name):<26} {target_tier:<12} {t:<9} {len(pairs):>5} {val:>10}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-dataset disagreement pilot (pool-based).")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="simpleqa",
                    help="which dataset to query/report on")
    ap.add_argument("--prompt-format", choices=["open", "mc"], default="open",
                    help="'open' (free text) or 'mc' (multiple choice, letter-only "
                         "responses). Must be one of the chosen dataset's supported "
                         "formats — SimpleQA is open-only (no distractors); "
                         "TruthfulQA supports both.")
    ap.add_argument("--n", type=int, default=200, help="number of questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="",
                    help="comma-separated extra model ids to add to the pool")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--judge-model", default="",
                    help="model id for independent regrading (grader-independence "
                         "check). Should NOT be a member of the pool.")
    ap.add_argument("--judge-provider", default="openrouter",
                    help="provider tag for --judge-model (see agents/providers.py). "
                         "The independence that matters is the judge's training "
                         "lineage, not the endpoint serving it, so this defaults to "
                         "openrouter rather than requiring a separate Gemini key.")
    ap.add_argument("--strict", action="store_true",
                    help="strict clustering (merge only on mutual entailment). "
                         "Worth trying on SimpleQA: relaxed mode merges distinct "
                         "years/names that NLI labels neutral rather than "
                         "contradictory.")
    ap.add_argument("--single-linkage", action="store_true")
    ap.add_argument("--no-concise", action="store_true",
                    help="do NOT force short answers (reproduces the saturated baseline)")
    ap.add_argument("--no-exact-match", action="store_true",
                    help="disable the short-answer string-match grading branch")
    ap.add_argument("--max-bad-questions", type=int, default=5,
                    help="abort after this many questions with an errored or "
                         "truncated response")
    ap.add_argument("--show-abstentions", action="store_true",
                    help="print detected abstentions so the regexes can be audited")
    ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-report from an existing --out file; no queries")
    ap.add_argument("--resume", action="store_true",
                    help="skip questions already present in --out and append "
                         "rather than truncate. Only valid when the pool, "
                         "question set and seed match the run that produced "
                         "the existing file.")
    ap.add_argument("--out", default=None,
                    help="output path; defaults from --dataset/--prompt-format "
                         "(see DATASETS at the top of this file)")
    args = ap.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    if args.prompt_format not in dataset_cfg["formats"]:
        ap.error(f"--dataset {args.dataset!r} has no {args.prompt_format!r} prompt "
                  f"format — available: {dataset_cfg['formats']}")
    is_mc = args.prompt_format == "mc"

    out_path = Path(args.out) if args.out else Path(
        default_out_path(args.dataset, args.prompt_format))

    dataset_label = {"simpleqa": "SimpleQA Verified", "truthfulqa": "TruthfulQA"}[args.dataset]
    section(f"Loading {args.n} {dataset_label} questions (seed={args.seed})"
            + ("" if not is_mc else f" [{args.prompt_format}]"))
    samples = dataset_cfg["loader"](n=args.n, seed=args.seed)
    print(f"  Loaded {len(samples)} questions.")
    check_prompt_selection(samples, dataset_cfg, args.prompt_format)

    section("Building agent pool")
    panel_kwargs = {} if args.no_concise else dict(dataset_cfg["config"])
    extra = [m for m in args.models.split(",") if m.strip()]
    try:
        agents = build_pool(extra, panel_kwargs)
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)

    print(f"  answer style: "
          f"{'verbose (baseline)' if args.no_concise else 'concise (forced short answers)'}")
    by_family = defaultdict(list)
    for a in agents:
        by_family[family_of(a.model)].append(a)
    for a in agents:
        tag = " [reasoning]" if agent_is_reasoning(a) else ""
        print(f"  - {a.name:<44} [{family_of(a.model)}] [{tier_of(a.model)}]{tag}")

    core_agents = [a for a in agents if a.model in CORE_MODELS]
    if not extra and len(core_agents) == len(agents) == len(CORE_MODELS):
        # Only meaningful when the pool is exactly the frozen core (no --models
        # extras) — the 3x3 grid this pool is designed to be.
        assert len({a.name for a in core_agents}) == 9, \
            f"core pool has duplicate agent names: {[a.name for a in core_agents]}"
        core_fam_ctr = Counter(family_of(a.model) for a in core_agents)
        core_tier_ctr = Counter(tier_of(a.model) for a in core_agents)
        assert len(core_fam_ctr) == 3 and all(v == 3 for v in core_fam_ctr.values()), \
            f"core pool is not 3 families x 3: {dict(core_fam_ctr)}"
        assert len(core_tier_ctr) == 3 and all(v == 3 for v in core_tier_ctr.values()), \
            f"core pool is not 3 tiers x 3: {dict(core_tier_ctr)}"
        print("  [OK] core pool verified: 9 distinct agents, 3 per family, 3 per tier.")

    thin = [f for f, members in by_family.items() if len(members) < 2]
    if thin:
        print(f"\n  [!] families with only one member: {', '.join(thin)}")
        print("      A within-family pair needs >=2 members, so these families")
        print("      cannot contribute to the matched-N comparison. Add a second")
        print("      size from each family you want to make a claim about.")
    n_reasoning = sum(agent_is_reasoning(a) for a in agents)
    if n_reasoning == 1:
        print("  [!] exactly ONE reasoning model in the pool. Its dissent cannot be")
        print("      attributed to family rather than inference mode — two axes")
        print("      vary at once. The ablation below quantifies the exposure.")

    panel_specs = build_panel_specs(agents)
    kind_counts = Counter(s.kind for s in panel_specs)
    print(f"\n  {len(panel_specs)} panels formed post-hoc over the cached pool "
          f"(free — the cache is keyed on model+prompt, not panel): " +
          ", ".join(f"{k}={kind_counts[k]}" for k in sorted(kind_counts)))

    if not args.analyse_only:
        section("Preflight — model IDs")
        try:
            assert_models_available(agents)
        except PermanentAgentError as exc:
            print(f"\n  [ABORT] {exc}\n")
            sys.exit(2)
        print("  All pool models are live.")

        # The canary is deliberately a question a reasoning model must think
        # about, not 'capital of France' — the old canary passed precisely
        # because it was too easy to trigger a long chain of thought, and 21
        # truncations then showed up on the real questions.
        section("Preflight — token budget under a reasoning load")
        canary = ("In what year did the Treaty of Westphalia conclude, and which "
                  "two settlements comprised it? Answer concisely.")
        ok = True
        for a, r in zip(agents, query_agents(agents, canary)):
            empty = not (r.text or "").strip()
            ctoks = (r.usage or {}).get("completion_tokens")
            note = ""
            if empty:
                ok = False
                note = (f"  <-- ERROR: {r.error}" if r.error else
                        f"  <-- EMPTY (finish_reason={r.finish_reason}); raise max_tokens")
            print(f"  {a.name:<44} [finish_reason={r.finish_reason}, "
                  f"completion_tokens={ctoks}] -> {(r.text or '').strip()[:40]!r}{note}")
        if not ok:
            cfg_name = f"{args.dataset.upper()}_QUERY_CONFIG"
            print("\n  [ABORT] An agent produced no usable text on the canary.")
            print("  - finish_reason='length' => the reasoning trace consumed the")
            print(f"    budget. Raise {cfg_name}['max_tokens'], or set a")
            print("    lower reasoning effort via extra_body.")
            print("  - If you change reasoning params: make_cache_key()'s `extra`")
            print("    currently carries only system_prompt, so extra_body is NOT")
            print("    part of the cache identity. Change it without fixing that")
            print("    and you will silently get the old truncated responses back.\n")
            sys.exit(2)
        print("  All agents answered under a reasoning load.")

    print(f"\n  Loading NLI model: {args.nli_model} (first run downloads weights)...")
    nli = CrossEncoderNLI(model_name=args.nli_model)
    jaccard = JaccardDisagreement()
    mc_exact = MCExactMatch()
    linkage = "single" if args.single_linkage else "complete"
    semantic = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                           linkage=linkage)
    print(f"  clustering: {'strict' if args.strict else 'relaxed'} + {linkage}-linkage")

    judge_agent = None
    if args.judge_model:
        if is_mc:
            print("  [!] --judge-model ignored in --prompt-format mc: letter comparison")
            print("      is deterministic, so there is nothing for an independent judge")
            print("      to adjudicate — continuing without it.")
        elif any(a.model == args.judge_model for a in agents):
            print("  [!] the judge is a pool member — that defeats the purpose.")
            sys.exit(1)
        else:
            # No **panel_kwargs here — that would give the judge the panel's
            # "Answer in one short sentence" system prompt, which is not its
            # task. max_tokens=512 is deliberately generous (not sized to the
            # one-word verdict): a reasoning judge that gets truncated to
            # empty text would otherwise grade every response NOT_ATTEMPTED,
            # silently.
            judge_agent = make_provider_agent(
                args.judge_model, provider=args.judge_provider,
                temperature=0.0, max_tokens=512, system_prompt=None)
            print(f"  independent judge: {args.judge_provider}/{args.judge_model}")

    if args.analyse_only:
        if not out_path.exists():
            print(f"\n  [ABORT] {out_path} does not exist.\n")
            sys.exit(1)
        rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"\n  Re-reporting {len(rows)} cached rows from {out_path}.")
    else:
        rows = run_queries(samples, agents, jaccard, semantic, mc_exact, nli, panel_specs,
                           out_path, args, judge_agent, dataset_cfg, is_mc)

    if not rows:
        print("\n  No usable rows. Nothing to report.\n")
        sys.exit(1)

    names = [a.name for a in agents]
    core_names = [a.name for a in agents if a.model in CORE_MODELS]
    report_abstention(rows, names)
    if args.show_abstentions:
        show_abstention_examples(rows, names)
    report_measures(rows, names, core_names, panel_specs, args, is_mc)
    report_panels(rows, agents, panel_specs, args, is_mc)
    report_cross_tier(rows, agents, panel_specs, args, is_mc)

    section("Done")
    print(f"  Per-question rows written to: {out_path}")
    # Flags only shown when they're not the defaults, so the default
    # simpleqa/open invocation's hint stays exactly what it always was.
    non_default_flags = ""
    if args.dataset != "simpleqa" or args.prompt_format != "open":
        non_default_flags = f"--dataset {args.dataset} --prompt-format {args.prompt_format} "
    print(f"  Re-report without querying:   python {Path(__file__).name} "
          f"{non_default_flags}--analyse-only --out {out_path}")
    if len(rows) < 150:
        print(f"\n  [!] n={len(rows)}. The subgroup counts in Part B (tie-broken,")
        print("      rescued, false negatives) need several hundred questions before")
        print("      their intervals are narrow enough to support a claim.\n")


if __name__ == "__main__":
    main()
