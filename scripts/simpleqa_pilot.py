"""SimpleQA Verified pilot: pool-based querying, panels formed post-hoc.

The SimpleQA Verified counterpart to scripts/pilot.py, restructured after the
first 50-question run showed three problems that the original design couldn't
separate:

  1. gpt-oss-20b (the only reasoning model in the panel) was truncated on
     21/50 questions. Empty text forms its own NLI cluster, so ~40% of the
     "disagreement" signal was measuring token budget, not knowledge.
  2. Same-family (N=2) was compared against the full panel (N=4), which
     confounds family diversity with panel size — more agents mechanically
     means more chances of a distinct cluster.
  3. Abstention ("I don't have reliable information about X") was scored
     identically to dissent. Those are opposite epistemic states.

What changed
------------
POOL, NOT PANEL. The query loop takes a *pool* of models and asks each one
every question exactly once. Panel composition then becomes a post-hoc
analysis choice over the cached texts — the cache is keyed on
(model, prompt, temperature, max_tokens), not on panel membership, so every
panel below is free once the pool has been queried. Adding a model costs one
pass over the questions; adding a *panel* costs nothing.

ABSTENTION IS A THIRD STATE. Detected explicitly, excluded from clustering,
graded as NOT_ATTEMPTED rather than incorrect, and reported per-agent. This
follows SimpleQA's own CORRECT / INCORRECT / NOT_ATTEMPTED protocol. On this
dataset abstention rate may well be a better hallucination signal than
disagreement — but that is only visible if it isn't buried inside the entropy.

EMPTY IS A FAILURE, NOT A DATA POINT. A response that is empty with
finish_reason='length' is a truncation, and is handled exactly like an API
error: the question is skipped and counted, and the run aborts past a
threshold. A results file can no longer silently contain unusable rows.

MATCHED-N PANELS. Part B compares every within-family pair against every
cross-family pair, both at N=2, so family diversity is isolated from panel
size. The full-panel view is still reported, clearly labelled as unmatched.

AUROC, NOT MEANS. Jaccard is continuous; normalised entropy over k singleton
clusters is exactly 1.0. Their means are not on a comparable scale and their
difference means nothing. Part A reports rank correlation and AUROC against
graded incorrectness instead.

Grader independence
-------------------
By default the SAME CrossEncoderNLI instance does the clustering and the
grading. Their errors are therefore correlated in exactly the direction that
inflates "disagreement predicts wrongness" — with k agents, k distinct
clusters nearly forces at most one agent to entail the gold. Pass
--judge-model to regrade with an independent LLM judge; the report prints
both gradings side by side and the AUROC computed under each. Treat any
AUROC that only survives under the NLI grader as an artifact.

Requires: OPENROUTER_API_KEY, plus torch + sentence-transformers for the NLI
model. Responses are cached (agents/) so re-runs are fast and deterministic.

Run from the project root:
    python scripts/simpleqa_pilot.py --n 200
    python scripts/simpleqa_pilot.py --n 200 --judge-model <model-id>
    python scripts/simpleqa_pilot.py --n 50 --models a/b,c/d   # extend the pool
    python scripts/simpleqa_pilot.py --n 50 --analyse-only     # re-report, no queries
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_simpleqa, SIMPLEQA_QUERY_CONFIG                 # noqa: E402
from agents import (                                                   # noqa: E402
    query_agents,
    assert_models_available,
    PermanentAgentError,
    PERMANENT_ERROR_PREFIX,
)
from agents.panels import (                                            # noqa: E402
    pool as panels_pool,
    family_of,
    LLAMA_SMALL,
    LLAMA_LARGE,
)
from agents.providers import make_provider_agent                      # noqa: E402
from disagreement import (                                             # noqa: E402
    JaccardDisagreement,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
)
from evaluation import grade_correct                                   # noqa: E402


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
        self.kind = kind                # 'full' | 'within' | 'cross' | 'toggle' | 'ablation'

    def __repr__(self) -> str:
        return f"PanelSpec({self.key!r}, n={len(self.members)})"


def build_panel_specs(agents) -> List[PanelSpec]:
    """Full panel, every matched-N=2 pair, and the reasoning ablation.

    The N=2 pairs are the point: comparing a 2-model same-family panel against
    a 4-model mixed panel confounds family with size, so within-family and
    cross-family are both enumerated at N=2 and aggregated.
    """
    names = [a.name for a in agents]
    fam = {a.name: family_of(a.model) for a in agents}
    model_of = {a.name: a.model for a in agents}
    specs: List[PanelSpec] = [PanelSpec("full", f"full pool (N={len(names)})", names, "full")]

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
    return specs


def score_panel(spec: PanelSpec, texts_by_name: Dict[str, str],
                excluded_by_name: Dict[str, bool],
                jaccard, semantic, question: str) -> dict:
    """Disagreement over one panel, with excluded responses removed first.

    `excluded_by_name` covers both abstentions (model behaviour: declining to
    answer is not a dissent) and truncations (measurement failure: an empty or
    cut-off response would form its own NLI cluster and inflate disagreement).
    Panels left with fewer than two usable answers get None — undefined, not
    zero. The row records the two causes separately; only the union matters
    here, since both mean "no comparable answer from this agent".
    """
    attempted = [n for n in spec.members if not excluded_by_name[n]]
    n_excluded = len(spec.members) - len(attempted)
    if len(attempted) < 2:
        return {"jaccard": None, "semantic_entropy": None, "n_clusters": None,
                "cluster_sizes": None, "n_scored": len(attempted),
                "n_excluded": n_excluded, "cluster_of": {}}

    texts = [texts_by_name[n] for n in attempted]
    jac = jaccard.score(texts).score
    sem = semantic.score(texts, question=question)
    cluster_of = {}
    for cid, members in enumerate(sem.details["clusters"]):
        for member_idx in members:
            cluster_of[attempted[member_idx]] = cid
    return {"jaccard": jac, "semantic_entropy": sem.score,
            "n_clusters": sem.details["n_clusters"],
            "cluster_sizes": sem.details["cluster_sizes"],
            "n_scored": len(attempted), "n_excluded": n_excluded,
            "cluster_of": cluster_of}


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

def grade_one(nli, text, sample, abstained, use_exact):
    if abstained or not (text or "").strip():
        return None
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
# Query phase
# ---------------------------------------------------------------------------

def run_queries(samples, agents, jaccard, semantic, nli, panel_specs,
                out_path: Path, args, judge_agent) -> List[dict]:
    section("Scoring")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = [a.name for a in agents]
    rows: List[dict] = []
    done_uids: set = set()

    if args.resume:
        print("  [!] --resume is only valid when the pool, question set and seed are "
              "unchanged from the run that produced the existing file — a resumed "
              "run that silently mixes rows from two different pools would leave "
              "`responses` dicts with inconsistent keys, and every panel comparison "
              "downstream would be wrong in a way that is hard to spot.")
        if out_path.exists():
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

    n_bad_questions = 0
    skipped: List[Tuple[str, str]] = []

    for idx, s in enumerate(samples, start=1):
        responses = query_agents(agents, s.prompt)

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
        abstained = {n: (not unusable[n]) and is_abstention(t) for n, t in texts.items()}
        excluded = {n: unusable[n] or abstained[n] for n in texts}

        if sum(not v for v in excluded.values()) < 2:
            n_bad_questions += 1
            print(f"  [{idx:>3}/{len(samples)}] {s.uid}  [!] fewer than 2 usable answers")
            if n_bad_questions > args.max_bad_questions:
                raise RuntimeError(f"Aborting: {n_bad_questions} unscorable questions.")
            continue

        grades = {n: grade_one(nli, texts[n], s, excluded[n], not args.no_exact_match)
                  for n in names}
        judged = None
        if judge_agent is not None:
            judged = {n: (None if excluded[n] else
                          judge_grade(judge_agent, s.question, s.correct_answer, texts[n]))
                      for n in names}

        panels = {spec.key: score_panel(spec, texts, excluded, jaccard, semantic,
                                        s.question)
                  for spec in panel_specs}

        full = panels["full"]
        graded = [g for g in grades.values() if g is not None]
        row = {
            "uid": s.uid,
            "topic": s.topic,
            "answer_type": s.answer_type,
            "multi_step": s.multi_step,
            "requires_reasoning": s.requires_reasoning,
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
            # Convenience mirrors of the full panel, so downstream scripts that
            # already read the old flat schema keep working.
            "jaccard": full["jaccard"],
            "semantic_entropy": full["semantic_entropy"],
            "n_clusters": full["n_clusters"],
            "cluster_sizes": full["cluster_sizes"],
            "cluster_of": full["cluster_of"],
            "panel_agrees": (full["n_clusters"] == 1) if full["n_clusters"] else None,
            "panel_majority_wrong": (bool(graded) and
                                     sum(g is False for g in graded) > len(graded) / 2),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} [{s.topic}] ---")
            print(f"      Q: {s.question}")
            print(f"      gold: {s.correct_answer}")
            for n in names:
                gl = {True: "OK  ", False: "WRONG", None: "N/A "}[grades[n]]
                ab = " [abstained]" if abstained[n] else ""
                print(f"      [{gl}] {short(n)}{ab}: {texts[n].strip()[:200]}")
            print(f"      -> clusters={full['cluster_sizes']} "
                  f"sem={full['semantic_entropy']}")

        sem_s = "  n/a" if full["semantic_entropy"] is None else f"{full['semantic_entropy']:.2f}"
        jac_s = "  n/a" if full["jaccard"] is None else f"{full['jaccard']:.2f}"
        ex_s = ""
        if any(excluded.values()):
            ex_s = f"  abst={sum(abstained.values())} trunc={sum(unusable.values())}"
        print(f"  [{idx:>3}/{len(samples)}] {s.uid}  jac={jac_s}  sem={sem_s}  "
              f"clusters={full['cluster_sizes']}{ex_s}")

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

    dropped = [r for r in rows if r["panels"]["full"]["semantic_entropy"] is None]
    print(f"\n    questions with <2 usable answers (unscorable): {len(dropped)}")
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


def report_measures(rows, names, args) -> None:
    section("A. Answer-level (Jaccard) vs semantic entropy")

    scored = [r for r in rows if r["panels"]["full"]["semantic_entropy"] is not None]
    jac = [r["jaccard"] for r in scored]
    sem = [r["semantic_entropy"] for r in scored]
    print(f"  scorable questions: {len(scored)} / {len(rows)}")
    print(f"  mean Jaccard          : {statistics.mean(jac):.3f}")
    print(f"  mean semantic entropy : {statistics.mean(sem):.3f}")
    print("  [note] These means are NOT comparable. Jaccard is continuous;")
    print("  normalised entropy over k singletons is exactly 1.0. Their")
    print("  difference is a scale artifact, not a result. Use the AUROCs below.")

    cc = Counter(r["n_clusters"] for r in scored)
    print(f"\n  n_clusters spread     : " + ", ".join(f"{k}->{cc[k]}" for k in sorted(cc)))

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

    grade_ctr = Counter()
    for r in scored:
        for g in r["grades"].values():
            grade_ctr[{True: "correct", False: "incorrect", None: "not_attempted"}[g]] += 1
    tot = sum(grade_ctr.values()) or 1
    print(f"\n  grades (NLI)          : " + ", ".join(
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
    for grade_key, label in (("grades", "NLI grader"), ("grades_judge", "LLM judge")):
        if grade_key == "grades_judge" and not any(r.get("grades_judge") for r in scored):
            continue
        print(f"\n    [{label}]")
        if grade_key == "grades":
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
        for n in names:
            pairs = [(r, (r[grade_key] or {}).get(n)) for r in scored if r.get(grade_key)]
            pairs = [(r, g) for r, g in pairs if g is not None]
            labs = [g is False for _, g in pairs]
            a_j, why_j = auroc_guarded([r["jaccard"] for r, _ in pairs], labs)
            a_s, _ = auroc_guarded([r["semantic_entropy"] for r, _ in pairs], labs)
            if a_j is None and a_s is None:
                print(f"        {short(n):<32} suppressed — {why_j}")
                continue
            f = lambda v: f"{v:.3f}" if v is not None else " n/a "
            print(f"        {short(n):<32} n={len(pairs):<4} "
                  f"err={sum(labs)/max(len(labs),1):.0%}  "
                  f"jac={f(a_j)}  sem={f(a_s)}")

    fp = [r for r in scored if r["jaccard"] is not None
          and r["jaccard"] >= 0.5 and r["semantic_entropy"] == 0.0]
    print(f"\n  Jaccard flags (>=0.50) but agents agree in meaning: {len(fp)} / {len(scored)}")
    print("  On short-form factuality this count is expected to be small — the")
    print("  false-positive reduction that motivates semantic entropy is largely")
    print("  a long-form phenomenon. TruthfulQA vs SimpleQA is the clean contrast.")
    for r in fp[:5]:
        print(f"    - {r['uid']} [{r['topic']}]  jac={r['jaccard']:.2f}")
        print(f"        Q: {r['question'][:80]}")


def report_panels(rows, agents, panel_specs, args) -> None:
    section("B. Shared-bias test — matched-N panels (family isolated from size)")

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
        for s in swaps:
            paired = common_rows(rows, ["full_no_reasoning", s.key])
            if not paired:
                continue
            d = [r["panels"][s.key]["semantic_entropy"]
                 - r["panels"]["full_no_reasoning"]["semantic_entropy"] for r in paired]
            md = statistics.mean(d)
            print(f"    swapping in {short(s.members[-1]):<24} {md:+.3f}  (n={len(paired)})")
            direction = ("raises" if md > 0 else "lowers" if md < 0 else "does not change")
            print(f"      → the reasoning model {direction} panel disagreement "
                  f"relative to the non-reasoning model it replaced.")
        print("    A near-zero effect means inference mode is NOT what drives the")
        print("    cross-family signal — a useful negative result, not a failure.")
        
    # Who actually breaks the ties?
    llama_names = [a.name for a in agents if a.model in (LLAMA_SMALL, LLAMA_LARGE)]
    llama_pair = [s for s in panel_specs
                  if s.kind == "within" and set(s.members) == set(llama_names)]
    if llama_pair and len(llama_names) == 2:
        key = llama_pair[0].key
        tie = [r for r in rows
               if r["panels"][key]["n_clusters"] == 1
               and r["panels"]["full"]["n_clusters"]
               and r["panels"]["full"]["n_clusters"] > 1]
        print(f"\n  Llamas agree but the full pool splits: {len(tie)} / {len(rows)}")

        dissenter = Counter()
        for r in tie:
            co = r["panels"]["full"]["cluster_of"]
            llama_cluster = {co[n] for n in llama_names if n in co}
            outside = [n for n, c in co.items()
                       if n not in llama_names and c not in llama_cluster]
            if len(outside) == 1:
                dissenter[outside[0]] += 1
        print("  Sole dissenter, by agent:")
        for n, k in dissenter.most_common():
            print(f"    {short(n):<32} {k:>3}")
        print("  If one agent dominates this column, 'cross-family' means that")
        print("  agent, and the claim needs a second model from its family.")

        rescued = [r for r in tie
                   if all(r["grades"].get(n) is False for n in llama_names)]
        print(f"\n  ...of which both Llamas graded WRONG (shared-bias surfaced): "
              f"{len(rescued)}")
        print(f"  [!] {len(rescued)}/{len(rows)} is a subgroup count. At n={len(rows)} its")
        print("  95% interval is wide; it also conjoins two judgements from the")
        print("  same NLI, so an NLI failure corrupts both at once. Report an")
        print("  interval, or run the judge, before drawing a conclusion.")
        for r in rescued[:5]:
            print(f"    - {r['uid']} [{r['topic']}]")
            print(f"        Q: {r['question'][:80]}")
            print(f"        gold: {r['correct_answer'][:80]}")

    def dominant_wrong(r, key="full", frac=0.5):
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
        return (cid, len(members), n) if sum(g is False for g in graded) > len(graded) / 2 else None

    fn_strict = [r for r in rows if r["panel_agrees"] and r["panel_majority_wrong"]]
    fn = [r for r in rows if dominant_wrong(r)]
    print(f"\n  Shared-bias FALSE NEGATIVES")
    print(f"    strict (whole pool in ONE cluster, and wrong) : {len(fn_strict)} / {len(rows)}")
    print(f"    majority (>50% of agents in one wrong cluster): {len(fn)} / {len(rows)}")
    print("    The strict count falls as the pool grows — unanimity across 7 agents")
    print("    is rarer than across 4 — so it measures panel size as much as shared")
    print("    bias. Report the majority figure; keep strict for comparability with")
    print("    the earlier 4-agent runs.")
    for r in fn[:5]:
        print(f"    - {r['uid']} [{r['topic']}]  sem={r['semantic_entropy']:.2f}")
        print(f"        Q: {r['question'][:80]}")
        print(f"        gold: {r['correct_answer'][:80]}")
        for n, t in r["responses"].items():
            print(f"          {short(n):<26} {t.strip()[:70]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="SimpleQA Verified disagreement pilot (pool-based).")
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
    ap.add_argument("--out", default="outputs/simpleqa_pilot_results.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)

    section(f"Loading {args.n} SimpleQA Verified questions (seed={args.seed})")
    samples = load_simpleqa(n=args.n, seed=args.seed)
    print(f"  Loaded {len(samples)} questions.")

    section("Building agent pool")
    panel_kwargs = {} if args.no_concise else dict(SIMPLEQA_QUERY_CONFIG)
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
        print(f"  - {a.name:<44} [{family_of(a.model)}]{tag}")

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
    print(f"\n  {len(panel_specs)} panels formed post-hoc over the cached pool "
          f"(free — the cache is keyed on model+prompt, not panel).")

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
            note = ""
            if empty:
                ok = False
                note = (f"  <-- ERROR: {r.error}" if r.error else
                        f"  <-- EMPTY (finish_reason={r.finish_reason}); raise max_tokens")
            print(f"  {a.name:<44} -> {(r.text or '').strip()[:40]!r}{note}")
        if not ok:
            print("\n  [ABORT] An agent produced no usable text on the canary.")
            print("  - finish_reason='length' => the reasoning trace consumed the")
            print("    budget. Raise SIMPLEQA_QUERY_CONFIG['max_tokens'], or set a")
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
    linkage = "single" if args.single_linkage else "complete"
    semantic = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                           linkage=linkage)
    print(f"  clustering: {'strict' if args.strict else 'relaxed'} + {linkage}-linkage")

    judge_agent = None
    if args.judge_model:
        if any(a.model == args.judge_model for a in agents):
            print("  [!] the judge is a pool member — that defeats the purpose.")
            sys.exit(1)
        # No **panel_kwargs here — that would give the judge the panel's
        # "Answer in one short sentence" system prompt, which is not its task.
        # max_tokens=512 is deliberately generous (not sized to the one-word
        # verdict): a reasoning judge that gets truncated to empty text would
        # otherwise grade every response NOT_ATTEMPTED, silently.
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
        rows = run_queries(samples, agents, jaccard, semantic, nli, panel_specs,
                           out_path, args, judge_agent)

    if not rows:
        print("\n  No usable rows. Nothing to report.\n")
        sys.exit(1)

    names = [a.name for a in agents]
    report_abstention(rows, names)
    if args.show_abstentions:
        show_abstention_examples(rows, names)
    report_measures(rows, names, args)
    report_panels(rows, agents, panel_specs, args)

    section("Done")
    print(f"  Per-question rows written to: {out_path}")
    print(f"  Re-report without querying:   python {Path(__file__).name} "
          f"--analyse-only --out {out_path}")
    if len(rows) < 150:
        print(f"\n  [!] n={len(rows)}. The subgroup counts in Part B (tie-broken,")
        print("      rescued, false negatives) need several hundred questions before")
        print("      their intervals are narrow enough to support a claim.\n")


if __name__ == "__main__":
    main()
