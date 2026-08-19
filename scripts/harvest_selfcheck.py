"""SelfCheckGPT-style within-model sampling harvest.

Queries a SINGLE model (on any OpenAI-compatible endpoint registered in
agents/providers.py — Groq by default, or e.g. --provider openrouter) k
times per TruthfulQA question, at temperature > 0, and scores the k draws
with the same disagreement-measure menu
scripts/harvest.py uses across DIFFERENT models. Where harvest.py's signal
is cross-model disagreement (same question, independent models, T=0),
this script's signal is within-model sampling variance (same question, same
model, independent draws) — the classic SelfCheckGPT setup (Manakul et al.
2023), run through this project's existing disagreement/grading machinery so
the numbers land in the same units and are directly comparable to the
cross-family panels.

Terminology note: TruthfulQA calls each item a "sample" (see
data.truthfulqa.TruthfulQASample) — but this script also draws k SAMPLES
(completions) per question. To keep those apart, everything below calls the
former a "question" and the latter a "draw".

Output (three files, derived from --out)
-----------------------------------------
  <out>                    one row per QUESTION: gold, k, disagreement scores
                            (Jaccard, MC exact-match, semantic entropy at
                            strict_entailment=True/False) pooled over the k
                            draws, and the majority-of-k hallucination label.
  <out stem>.responses.jsonl
                            one row per (question, draw): the full response
                            text, reasoning, finish_reason, token usage,
                            error/error_kind, that draw's own grade, and its
                            `sample_index` (0..k-1) — so a k=10 harvest can be
                            subset offline to k=5 or k=3 with no further API
                            calls.
  <out stem>.manifest.json
                            provenance: model, k, seeds, temperature,
                            prompt_format, query settings, resolved reasoning
                            params, NLI model, semantic-entropy settings,
                            dataset source/n/seed, git commit, UTC start/end.

Reproducible-but-distinct draws
--------------------------------
Draw i is issued by its own GroqAgent instance, built via
agents.providers.make_provider_agent(model, provider=..., extra_params=
{"seed": i}, ...) — the seed is merged into that request's extra_params,
which (a) is forwarded to the endpoint as a best-effort determinism hint and
(b) is baked into GroqAgent's cache key (agents/cache.py's `extra` dict). So
draw i is stable across re-runs (same cache key -> cache hit, no re-billing)
while draws i != j never collide on one cache entry — the mechanism
scripts/pilot_sampled.py already relies on for the same reason (there via
agents.panels.make_agent, Groq-only; here via make_provider_agent, so --model
can live on any registered provider).

Resumability
------------
Keyed on (uid, sample_index). Coarsely, a question already fully written to
--out is skipped outright on resume (mirrors scripts/harvest.py). Finely,
even a question NOT yet written to --out only re-issues draws whose
(model, prompt, temperature, max_tokens, seed) cache key isn't already on
disk — each draw's own GroqAgent cache is content-addressed on exactly that
tuple, which is equivalent to (uid, sample_index) since prompt<->uid and
seed<->sample_index are both bijective here. Net effect: a k=5 harvest
re-run from scratch makes zero additional API calls.

Run from the project root:
    python scripts/harvest_selfcheck.py --model openai/gpt-oss-20b \\
        --k 5 --temperature 0.7 --prompt-format open --n 50 \\
        --out outputs/harvest_selfcheck.csv

    # Same, via OpenRouter instead of Groq (needs OPENROUTER_API_KEY in .env;
    # --model must be an OpenRouter model id, e.g. 'meta-llama/llama-3.3-70b-instruct'):
    python scripts/harvest_selfcheck.py --model meta-llama/llama-3.3-70b-instruct \\
        --provider openrouter --k 5 --temperature 0.7 --prompt-format open \\
        --n 50 --out outputs/harvest_selfcheck_openrouter.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa, TRUTHFULQA_QUERY_CONFIG                # noqa: E402
from agents import (                                                     # noqa: E402
    query_agents,
    assert_models_available,
    PermanentAgentError,
)
from agents.panels import _assert_uniform_query_settings                 # noqa: E402
from agents.providers import make_provider_agent, PROVIDERS              # noqa: E402
from disagreement import (                                               # noqa: E402
    JaccardDisagreement,
    MCExactMatch,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
)
from evaluation import grade_correct                                     # noqa: E402


NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Same headroom rationale as scripts/harvest.py's HARVEST_MAX_TOKENS: a
# reasoning model's hidden trace can consume the whole completion budget
# before it gets to an answer, and --model here may well be one.
HARVEST_MAX_TOKENS = 300

CSV_FIELDS = [
    "uid", "category", "question_type", "question",
    "correct_answer", "correct_letter",
    "model", "k", "n_valid", "hallucinated",
    "jaccard",
    "mc_exact_match",
    "semantic_entropy_strict", "semantic_entropy_strict_nats", "n_clusters_strict",
    "semantic_entropy_relaxed", "semantic_entropy_relaxed_nats", "n_clusters_relaxed",
]

RESPONSE_FIELDS = [
    "uid", "model", "sample_index", "text", "reasoning", "finish_reason",
    "completion_tokens", "prompt_tokens", "total_tokens",
    "error", "error_kind", "grade", "latency_s", "timestamp",
]

# Consecutive-run-independent: total questions abandoned to a transient error
# in THIS invocation. Same circuit breaker as scripts/harvest.py.
MAX_TRANSIENT_SKIPS = 3


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _majority_hallucinated(grades: List[Optional[bool]]) -> Optional[bool]:
    """True iff the majority of DECIDED (non-None) grades in `grades` are
    False (wrong). False iff the majority are True (correct). None if there
    are no decided grades, or the vote is tied.

    Identical convention to scripts/harvest.py's `_majority_hallucinated`,
    applied to a question's k draws instead of a panel's per-model answers —
    same polarity (majority-wrong -> True), so labels are directly
    comparable between the two harvests.
    """
    decided = [g for g in grades if g is not None]
    if not decided:
        return None
    n_wrong = sum(1 for g in decided if g is False)
    n_right = len(decided) - n_wrong
    if n_wrong == n_right:
        return None
    return n_wrong > n_right


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def _load_completed_uids(out_path: Path) -> set:
    with out_path.open("r", newline="", encoding="utf-8") as fh:
        return {row["uid"] for row in csv.DictReader(fh)}


def _check_manifest_compatible(manifest: Dict[str, Any], *, model: str,
                                args: argparse.Namespace, query_config: dict) -> None:
    """Refuse to resume a harvest under different settings.

    A resumed run that silently used a different model, k, prompt format, or
    query config would mix incomparable rows into one file with no way to
    tell them apart later — exactly the kind of corruption the provenance
    manifest exists to prevent. Mirrors scripts/harvest.py's check, plus `k`
    (changing k mid-file would leave earlier rows scored over a different
    number of draws than later ones).
    """
    checks = {
        "model": (manifest.get("model"), model),
        "provider": (manifest.get("provider"), args.provider),
        "k": (manifest.get("k"), args.k),
        "prompt_format": (manifest.get("prompt_format"), args.prompt_format),
        "source": (manifest.get("source"), args.source),
        "max_tokens": (manifest.get("max_tokens"), query_config["max_tokens"]),
        "system_prompt": (manifest.get("system_prompt"), query_config.get("system_prompt")),
        "temperature": (manifest.get("temperature"), query_config.get("temperature")),
    }
    mismatches = [f"{k} ({old!r} != {new!r})" for k, (old, new) in checks.items() if old != new]
    if mismatches:
        raise SystemExit(
            "[ABORT] --out exists with a manifest whose config doesn't match this "
            "invocation: " + "; ".join(mismatches) + ". Resuming with different "
            "settings would silently mix incomparable rows in one file — use a "
            "different --out."
        )


def _write_manifest(path: Path, *, agents: List[Any], args: argparse.Namespace,
                     query_config: dict, start_utc: str, end_utc: str) -> None:
    manifest = {
        "model": args.model,
        "provider": args.provider,
        "k": args.k,
        "seeds": [a.extra_params.get("seed") for a in agents],
        "prompt_format": args.prompt_format,
        "temperature": query_config["temperature"],
        "max_tokens": query_config["max_tokens"],
        "system_prompt": query_config.get("system_prompt"),
        "reasoning_params": agents[0].reasoning_params if agents else {},
        "nli_model": NLI_MODEL_NAME,
        "semantic_entropy_settings": [
            {"strict_entailment": True, "linkage": "complete", "normalise": True},
            {"strict_entailment": False, "linkage": "complete", "normalise": True},
        ],
        "dataset": "truthfulqa",
        "source": args.source,
        "n": args.n,
        "seed": args.seed,
        "git_commit": _git_sha(),
        "start_utc": start_utc,
        "end_utc": end_utc,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Summary (re-derived from the on-disk files, so it's correct whether or not
# this invocation resumed a prior partial run)
# ---------------------------------------------------------------------------

def _mean_sd(values: List[float]) -> str:
    if not values:
        return "n=0"
    if len(values) < 2:
        return f"mean={values[0]:.3f}  sd=n/a  n=1"
    return f"mean={statistics.mean(values):.3f}  sd={statistics.stdev(values):.3f}  n={len(values)}"


def _print_breakdown(label: str, csv_rows: List[dict], resp_rows: List[dict]) -> None:
    uids = {r["uid"] for r in csv_rows}
    sub_resp = [r for r in resp_rows if r["uid"] in uids]

    n_scored = sum(1 for r in csv_rows if r["jaccard"] != "")
    n_unscored = len(csv_rows) - n_scored
    n_empty = sum(1 for r in sub_resp if not (r["text"] or "").strip())
    n_truncated = sum(1 for r in sub_resp if r["finish_reason"] == "length")
    grades = [r["grade"] for r in sub_resp]
    graded = [g for g in grades if g is not None]
    n_unclear = sum(1 for g in grades if g is None)
    # "Positive" = hallucinated (grade False), matching the disagreement
    # measures' convention (higher score = more disagreement = elevated
    # hallucination risk) — same polarity as scripts/harvest.py.
    pos_rate = (sum(1 for g in graded if g is False) / len(graded)) if graded else None

    print(f"\n  -- {label} ({len(csv_rows)} questions) --")
    print(f"  n_scored           : {n_scored}")
    print(f"  n_unscored (<2 valid draws): {n_unscored}")
    print(f"  n_empty draws      : {n_empty} / {len(sub_resp)}")
    print(f"  n_truncated        : {n_truncated} / {len(sub_resp)}")
    print(f"  unclear grades     : {n_unclear} / {len(grades)}")
    print(f"  positive base rate : {pos_rate:.3f}" if pos_rate is not None else
          "  positive base rate : n/a")
    for metric in ("jaccard", "mc_exact_match",
                   "semantic_entropy_strict", "semantic_entropy_relaxed"):
        vals = [float(r[metric]) for r in csv_rows if r[metric] != ""]
        print(f"    {metric:<26} {_mean_sd(vals)}")


def print_summary(out_path: Path, responses_path: Path, n_skipped_transient: int) -> None:
    section("Summary")
    with out_path.open("r", newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    resp_rows = []
    if responses_path.exists():
        with responses_path.open("r", encoding="utf-8") as fh:
            resp_rows = [json.loads(line) for line in fh if line.strip()]

    print(f"  total questions in {out_path.name}: {len(csv_rows)}")
    print(f"  n_skipped (transient errors, this run): {n_skipped_transient}")

    _print_breakdown("overall", csv_rows, resp_rows)
    for qt in sorted({r["question_type"] for r in csv_rows}):
        _print_breakdown(f"question_type={qt}",
                          [r for r in csv_rows if r["question_type"] == qt], resp_rows)


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="SelfCheckGPT-style within-model sampling harvest.")
    ap.add_argument("--model", required=True,
                     help="single model ID on --provider's endpoint")
    ap.add_argument("--provider", default="groq", choices=sorted(PROVIDERS),
                     help="which OpenAI-compatible endpoint to query --model on "
                          "(see agents/providers.py's registry); default 'groq'")
    ap.add_argument("--k", type=int, required=True, help="samples (draws) per question")
    ap.add_argument("--temperature", type=float, required=True,
                     help="sampling temperature for the k draws; must be > 0")
    ap.add_argument("--prompt-format", required=True, choices=["mc", "open"],
                     help="query with sample.mc_prompt() or sample.open_prompt() — "
                          "no default, must be explicit")
    ap.add_argument("--n", type=int, default=None, help="subset size (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="subset seed")
    ap.add_argument("--source", choices=["hf", "github"], default="github",
                     help="TruthfulQA source (see data.load_truthfulqa)")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()

    if args.k < 2:
        ap.error("--k must be >= 2 — every disagreement measure needs >=2 responses "
                  "to compare.")
    if args.temperature <= 0:
        ap.error(f"--temperature must be > 0 (got {args.temperature}) — k independent "
                  "draws from one model are identical at T=0, so there is nothing to "
                  "disagree about. Use scripts/harvest.py for deterministic cross-model "
                  "panels.")

    out_path = Path(args.out)
    responses_path = out_path.with_name(out_path.stem + ".responses.jsonl")
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Query settings: dataset defaults, max_tokens bumped for reasoning
    # models (see module docstring), temperature is the CLI's (not the
    # dataset config's 0.0 — this harvest's whole point is T>0).
    query_config = {
        "max_tokens": HARVEST_MAX_TOKENS,
        "system_prompt": TRUTHFULQA_QUERY_CONFIG["system_prompt"],
        "temperature": args.temperature,
    }

    section(f"Building {args.k} same-model draw agent(s) for {args.provider}/{args.model}")
    # One GroqAgent per draw, each with its own seed baked into extra_params
    # (-> the cache key, per make_agent's docstring in agents/panels.py) so
    # draw i is reproducible but never collides with draw j. make_provider_agent
    # resolves base_url/api_key_env from the --provider registry and (as of
    # the fix above) keeps Groq's REASONING_PARAMS table from leaking into a
    # non-Groq request for a colliding model id.
    agents = [
        make_provider_agent(
            args.model, provider=args.provider,
            extra_params={"seed": i},
            temperature=args.temperature,
            max_tokens=query_config["max_tokens"],
            system_prompt=query_config["system_prompt"],
        )
        for i in range(args.k)
    ]
    _assert_uniform_query_settings(agents)
    for a in agents:
        print(f"  - {a.name:<40} seed={a.extra_params.get('seed')} "
              f"reasoning_params={a.reasoning_params}")

    # ---- Preflight: the model must be live BEFORE spending a token. ----
    section("Preflight — model availability")
    try:
        assert_models_available(agents)
    except PermanentAgentError as exc:
        print(f"\n[ABORT] {exc}\n")
        sys.exit(1)
    print("  Model is live.")

    # ---- Resumability ----
    resuming = out_path.exists()
    completed_uids: set = set()
    start_utc = _utc_now()
    if resuming:
        completed_uids = _load_completed_uids(out_path)
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _check_manifest_compatible(old_manifest, model=args.model, args=args,
                                        query_config=query_config)
            start_utc = old_manifest.get("start_utc", start_utc)
        else:
            print(f"  [!] {out_path} exists but {manifest_path.name} does not — "
                  f"proceeding, but provenance for its existing rows can't be verified.")
        print(f"  Resuming: {len(completed_uids)} question(s) already complete, skipping.")

    section(f"Loading TruthfulQA (source={args.source}, n={args.n}, seed={args.seed})")
    samples = load_truthfulqa(n=args.n, seed=args.seed, source=args.source)
    samples = [s for s in samples if s.uid not in completed_uids]
    print(f"  {len(samples)} question(s) to process.")

    # One NLI backend for every score that needs one: loads the model once,
    # and memoises predictions across grading AND both semantic-entropy calls.
    nli = CrossEncoderNLI(model_name=NLI_MODEL_NAME)
    jaccard = JaccardDisagreement()
    mc_exact = MCExactMatch()
    sem_strict = SemanticEntropyDisagreement(nli=nli, strict_entailment=True)
    sem_relaxed = SemanticEntropyDisagreement(nli=nli, strict_entailment=False)

    csv_fh = out_path.open("a" if resuming else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    if not resuming:
        writer.writeheader()
        csv_fh.flush()

    responses_fh = responses_path.open("a", encoding="utf-8")

    def abort(message: str, code: int = 1) -> None:
        print(f"\n[ABORT] {message}\n")
        csv_fh.close()
        responses_fh.close()
        _write_manifest(manifest_path, agents=agents, args=args,
                         query_config=query_config, start_utc=start_utc, end_utc=_utc_now())
        sys.exit(code)

    section("Harvesting")
    n_transient_skips = 0
    for idx, s in enumerate(samples, start=1):
        prompt = s.mc_prompt() if args.prompt_format == "mc" else s.open_prompt()
        responses = query_agents(agents, prompt)

        # Permanent error on any draw: the config itself is broken (bad
        # model id, revoked key, decommissioned model) — no amount of
        # skipping fixes that, so stop the whole harvest here.
        permanent = [(a, r) for a, r in zip(agents, responses) if r.is_permanent]
        if permanent:
            a0, r0 = permanent[0]
            abort(f"permanent error on {s.uid} ({a0.name}): {r0.error}")

        # Transient error: skip this question (don't write a row, so a
        # resumed run retries it), count it, and give up after too many.
        transient = [(a, r) for a, r in zip(agents, responses) if r.is_transient]
        if transient:
            n_transient_skips += 1
            a0, r0 = transient[0]
            print(f"  [{idx:>4}/{len(samples)}] {s.uid}  SKIPPED "
                  f"(transient: {a0.name}: {r0.error})")
            if n_transient_skips >= MAX_TRANSIENT_SKIPS:
                abort(f"{n_transient_skips} questions hit transient errors — "
                      f"likely a systemic rate-limit/quota problem, not bad luck.")
            continue

        valid_texts: List[str] = []
        draw_grades: List[Optional[bool]] = []
        n_empty_here = 0
        for sample_index, (a, r) in enumerate(zip(agents, responses)):
            text = r.text or ""
            if not text.strip():
                n_empty_here += 1
            if r.finish_reason == "length":
                print(f"  [!] {a.name} draw {sample_index} truncated on {s.uid} "
                      f"(finish_reason='length')")

            grade = grade_correct(nli, text, s, prompt_format=args.prompt_format)
            draw_grades.append(grade)
            response_row = {
                "uid": s.uid,
                "model": a.model,
                "sample_index": sample_index,
                "text": text,
                "reasoning": r.reasoning,
                "finish_reason": r.finish_reason,
                "completion_tokens": r.usage.get("completion_tokens"),
                "prompt_tokens": r.usage.get("prompt_tokens"),
                "total_tokens": r.usage.get("total_tokens"),
                "error": r.error,
                "error_kind": r.error_kind,
                "grade": grade,
                "latency_s": r.latency_s,
                "timestamp": r.timestamp,
            }
            assert response_row.keys() == set(RESPONSE_FIELDS)
            responses_fh.write(json.dumps(
                {k: response_row[k] for k in RESPONSE_FIELDS}, ensure_ascii=False,
            ) + "\n")

            # Empty text scores as maximum disagreement under every measure —
            # a silent false positive, not a real response — so it's recorded
            # above but excluded from the question-level scoring inputs below.
            if text.strip():
                valid_texts.append(text)
        responses_fh.flush()

        hallucinated = _majority_hallucinated(draw_grades)
        # Self-check the label's polarity at the point of use — see
        # scripts/harvest.py's identical guard for why this matters.
        decided = [g for g in draw_grades if g is not None]
        if decided:
            n_wrong = sum(1 for g in decided if g is False)
            n_right = len(decided) - n_wrong
            assert (hallucinated is None) == (n_wrong == n_right), (
                f"{s.uid}: hallucinated={hallucinated} inconsistent with tie "
                f"state (n_wrong={n_wrong}, n_right={n_right})"
            )
            assert hallucinated is None or hallucinated == (n_wrong > n_right), (
                f"{s.uid}: hallucinated={hallucinated} does not match majority "
                f"of valid grades being False (n_wrong={n_wrong}, n_right={n_right})"
            )
        else:
            assert hallucinated is None

        row = {
            "uid": s.uid,
            "category": s.category,
            "question_type": s.question_type,
            "question": s.question,
            "correct_answer": s.correct_answer,
            "correct_letter": s.correct_letter,
            "model": args.model,
            "k": args.k,
            "n_valid": len(valid_texts),
            "hallucinated": hallucinated,
            "jaccard": None, "mc_exact_match": None,
            "semantic_entropy_strict": None, "semantic_entropy_strict_nats": None,
            "n_clusters_strict": None,
            "semantic_entropy_relaxed": None, "semantic_entropy_relaxed_nats": None,
            "n_clusters_relaxed": None,
        }

        if len(valid_texts) >= 2:
            jac = jaccard.score(valid_texts)
            mce = mc_exact.score(valid_texts)
            se_s = sem_strict.score(valid_texts, context=s.question)
            se_r = sem_relaxed.score(valid_texts, context=s.question)
            row.update({
                "jaccard": jac.score,
                "mc_exact_match": mce.score,
                "semantic_entropy_strict": se_s.details["semantic_entropy_normalised"],
                "semantic_entropy_strict_nats": se_s.details["semantic_entropy_nats"],
                "n_clusters_strict": se_s.details["n_clusters"],
                "semantic_entropy_relaxed": se_r.details["semantic_entropy_normalised"],
                "semantic_entropy_relaxed_nats": se_r.details["semantic_entropy_nats"],
                "n_clusters_relaxed": se_r.details["n_clusters"],
            })
            status = f"jac={jac.score:.2f} sem_strict={se_s.score:.2f}"
        else:
            status = "UNSCORED (<2 valid draws)"

        writer.writerow(row)
        csv_fh.flush()

        note = f"  [{n_empty_here} empty]" if n_empty_here else ""
        print(f"  [{idx:>4}/{len(samples)}] {s.uid}  {status}{note}")

    csv_fh.close()
    responses_fh.close()
    _write_manifest(manifest_path, agents=agents, args=args, query_config=query_config,
                     start_utc=start_utc, end_utc=_utc_now())

    print_summary(out_path, responses_path, n_transient_skips)
    section("Done")
    print(f"  Questions -> {out_path}")
    print(f"  Responses -> {responses_path}")
    print(f"  Manifest  -> {manifest_path}")


if __name__ == "__main__":
    main()
