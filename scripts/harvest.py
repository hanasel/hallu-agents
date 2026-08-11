"""Reusable TruthfulQA harvesting runner.

Queries an arbitrary set of Groq models on a TruthfulQA subset, grades each
response, and scores every sample with the full disagreement-measure menu.
Unlike scripts/pilot.py (fixed cross-family panel, open-ended only), this is
the general-purpose harvester: any model list, either prompt format, meant to
be re-run as the panel composition or format changes over the course of the
thesis experiments.

Output (three files, derived from --out)
-----------------------------------------
  <out>                    one row per SAMPLE: gold, panel-level disagreement
                            scores (Jaccard, MC exact-match, semantic entropy
                            at strict_entailment=True/False).
  <out stem>.responses.jsonl
                            one row per (sample, model): the full response
                            text, reasoning, finish_reason, token usage,
                            error/error_kind, and that response's own grade.
                            This is what makes offline re-scoring of ANY
                            subset of the harvested models possible without
                            another API call — the panel-level CSV only
                            scores the full --models list, but every
                            response's raw text is preserved here.
  <out stem>.manifest.json
                            provenance: models, prompt_format, query
                            settings, resolved reasoning params per model,
                            NLI model, semantic-entropy settings, dataset
                            source/n/seed, git commit, UTC start/end.

Resumability
------------
If --out already exists, its manifest is checked against this invocation's
config (mismatch aborts — silently mixing settings inside one harvest would
corrupt it) and its completed uids are skipped. Each GroqAgent also keeps its
own on-disk response cache, so a model that already answered a still-pending
sample before an interruption is replayed from cache, not re-billed.

Run from the project root:
    python scripts/harvest.py --models llama-3.3-70b-versatile openai/gpt-oss-20b \\
        qwen/qwen3.6-27b --prompt-format open --n 50 --out outputs/harvest.csv
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
    GroqAgent,
    query_agents,
    assert_models_available,
    PermanentAgentError,
)
from agents.panels import _assert_uniform_query_settings                 # noqa: E402
from disagreement import (                                               # noqa: E402
    JaccardDisagreement,
    MCExactMatch,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
)
from evaluation import grade_correct                                     # noqa: E402


NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# gpt-oss's reasoning tokens count against the completion budget, and there is
# no reasoning_effort="none" for that family (unlike qwen) to turn generation
# off outright — 300 gives real headroom over TRUTHFULQA_QUERY_CONFIG's 100.
HARVEST_MAX_TOKENS = 300

CSV_FIELDS = [
    "uid", "category", "question_type", "question",
    "correct_answer", "correct_letter",
    "n_agents", "n_valid",
    "jaccard",
    "mc_exact_match",
    "semantic_entropy_strict", "semantic_entropy_strict_nats", "n_clusters_strict",
    "semantic_entropy_relaxed", "semantic_entropy_relaxed_nats", "n_clusters_relaxed",
]

RESPONSE_FIELDS = [
    "uid", "model", "text", "reasoning", "finish_reason",
    "completion_tokens", "prompt_tokens", "total_tokens",
    "error", "error_kind", "grade", "latency_s", "timestamp",
]

# Consecutive-run-independent: total samples abandoned to a transient error in
# THIS invocation. Mirrors the n_question_errors > 3 circuit breaker already
# used by scripts/pilot.py and scripts/rgt_pilot.py.
MAX_TRANSIENT_SKIPS = 3


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _check_manifest_compatible(manifest: Dict[str, Any], *, models: List[str],
                                args: argparse.Namespace, query_config: dict) -> None:
    """Refuse to resume a harvest under different settings.

    A resumed run that silently used a different model list, prompt format,
    or query config would mix incomparable rows into one file with no way to
    tell them apart later — exactly the kind of corruption the provenance
    manifest exists to prevent.
    """
    checks = {
        "models": (sorted(manifest.get("models", [])), sorted(models)),
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


def _write_manifest(path: Path, *, agents: List[GroqAgent], args: argparse.Namespace,
                     query_config: dict, start_utc: str, end_utc: str) -> None:
    manifest = {
        "models": [a.model for a in agents],
        "prompt_format": args.prompt_format,
        "temperature": query_config["temperature"],
        "max_tokens": query_config["max_tokens"],
        "system_prompt": query_config.get("system_prompt"),
        "reasoning_params": {a.model: a.reasoning_params for a in agents},
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
    pos_rate = (sum(1 for g in graded if g is True) / len(graded)) if graded else None

    print(f"\n  -- {label} ({len(csv_rows)} samples) --")
    print(f"  n_scored           : {n_scored}")
    print(f"  n_unscored (<2 valid responses): {n_unscored}")
    print(f"  n_empty responses  : {n_empty} / {len(sub_resp)}")
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

    print(f"  total samples in {out_path.name}: {len(csv_rows)}")
    print(f"  n_skipped (transient errors, this run): {n_skipped_transient}")

    _print_breakdown("overall", csv_rows, resp_rows)
    for qt in sorted({r["question_type"] for r in csv_rows}):
        _print_breakdown(f"question_type={qt}",
                          [r for r in csv_rows if r["question_type"] == qt], resp_rows)


# ---------------------------------------------------------------------------
# Harvest
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Reusable TruthfulQA harvesting runner.")
    ap.add_argument("--models", nargs="+", required=True,
                     help="one or more Groq model IDs")
    ap.add_argument("--prompt-format", required=True, choices=["mc", "open"],
                     help="query with sample.mc_prompt() or sample.open_prompt() — "
                          "no default, must be explicit")
    ap.add_argument("--n", type=int, default=None, help="subset size (default: all)")
    ap.add_argument("--seed", type=int, default=42, help="subset seed")
    ap.add_argument("--source", choices=["hf", "github"], default="github",
                     help="TruthfulQA source (see data.load_truthfulqa)")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()

    if len(args.models) < 2:
        ap.error("--models needs at least 2 model IDs — every disagreement "
                  "measure needs >=2 responses to compare.")

    out_path = Path(args.out)
    responses_path = out_path.with_name(out_path.stem + ".responses.jsonl")
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Query settings: dataset defaults, max_tokens bumped for reasoning
    # models (see module docstring). Every agent gets the identical dict, so
    # _assert_uniform_query_settings below is a structural guarantee, not
    # just a check of something that happened to be true this run.
    query_config = dict(TRUTHFULQA_QUERY_CONFIG)
    query_config["max_tokens"] = HARVEST_MAX_TOKENS

    section(f"Building {len(args.models)} agent(s)")
    agents = [GroqAgent(model=m, **query_config) for m in args.models]
    _assert_uniform_query_settings(agents)
    for a in agents:
        print(f"  - {a.name:<40} reasoning_params={a.reasoning_params}")

    # ---- Preflight: every model must be live BEFORE spending a token. The
    # guard for a mid-run model disappearance (e.g. the Aug 16 Llama
    # deprecation) — must fail loudly here, not silently produce a short
    # panel 40 questions in.
    section("Preflight — model availability")
    try:
        assert_models_available(agents)
    except PermanentAgentError as exc:
        print(f"\n[ABORT] {exc}\n")
        sys.exit(1)
    print("  All requested models are live.")

    # ---- Resumability ----
    resuming = out_path.exists()
    completed_uids: set = set()
    start_utc = _utc_now()
    if resuming:
        completed_uids = _load_completed_uids(out_path)
        if manifest_path.exists():
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _check_manifest_compatible(old_manifest, models=[a.model for a in agents],
                                        args=args, query_config=query_config)
            start_utc = old_manifest.get("start_utc", start_utc)
        else:
            print(f"  [!] {out_path} exists but {manifest_path.name} does not — "
                  f"proceeding, but provenance for its existing rows can't be verified.")
        print(f"  Resuming: {len(completed_uids)} sample(s) already complete, skipping.")

    section(f"Loading TruthfulQA (source={args.source}, n={args.n}, seed={args.seed})")
    samples = load_truthfulqa(n=args.n, seed=args.seed, source=args.source)
    samples = [s for s in samples if s.uid not in completed_uids]
    print(f"  {len(samples)} sample(s) to process.")

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

        # Permanent error on any response: the config itself is broken (bad
        # model id, revoked key, decommissioned model) — no amount of
        # skipping fixes that, so stop the whole harvest here.
        permanent = [(a, r) for a, r in zip(agents, responses) if r.is_permanent]
        if permanent:
            a0, r0 = permanent[0]
            abort(f"permanent error on {s.uid} ({a0.name}): {r0.error}")

        # Transient error: skip this sample (don't write a row, so a resumed
        # run retries it), count it, and give up after too many in a row.
        transient = [(a, r) for a, r in zip(agents, responses) if r.is_transient]
        if transient:
            n_transient_skips += 1
            a0, r0 = transient[0]
            print(f"  [{idx:>4}/{len(samples)}] {s.uid}  SKIPPED "
                  f"(transient: {a0.name}: {r0.error})")
            if n_transient_skips >= MAX_TRANSIENT_SKIPS:
                abort(f"{n_transient_skips} samples hit transient errors — "
                      f"likely a systemic rate-limit/quota problem, not bad luck.")
            continue

        valid_texts: List[str] = []
        n_empty_here = 0
        for a, r in zip(agents, responses):
            text = r.text or ""
            if not text.strip():
                n_empty_here += 1
            if r.finish_reason == "length":
                print(f"  [!] {a.name} truncated on {s.uid} (finish_reason='length')")

            grade = grade_correct(nli, text, s, prompt_format=args.prompt_format)
            response_row = {
                "uid": s.uid,
                "model": a.model,
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
            # above but excluded from the panel-level scoring inputs below.
            if text.strip():
                valid_texts.append(text)
        responses_fh.flush()

        row = {
            "uid": s.uid,
            "category": s.category,
            "question_type": s.question_type,
            "question": s.question,
            "correct_answer": s.correct_answer,
            "correct_letter": s.correct_letter,
            "n_agents": len(agents),
            "n_valid": len(valid_texts),
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
            status = "UNSCORED (<2 valid responses)"

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
    print(f"  Samples  -> {out_path}")
    print(f"  Responses -> {responses_path}")
    print(f"  Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
