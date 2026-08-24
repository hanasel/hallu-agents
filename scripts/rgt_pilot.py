"""RAGTruth QA pilot: cross-family panel disagreement vs corpus hallucination.

The RAGTruth counterpart to scripts/pilot.py. Where that pilot regenerates
answers to TruthfulQA questions and grades them against gold, this one takes
RAGTruth QA sources, sends each source's *exact corpus prompt* (Mode B) to our
cross-family panel, and asks whether the panel's disagreement predicts whether
the source is one where hallucination occurs.

Protocol (framing "a" from the Stage-3 design note)
---------------------------------------------------
  - Unit of prediction: the source (a question + its retrieved passages).
  - Signal: panel disagreement (Jaccard + semantic entropy) over N fresh
    responses our panel generates for that source's prompt.
  - Label: the corpus hallucination outcome for that source, taken from the six
    RAGTruth reference responses — either `any_hallucinated` (>=1 of 6) or the
    count `n_hallucinated` (0-6).

Known caveat, stated plainly
----------------------------
Our panel (recent Llama-3.x / GPT-OSS / Gemini) is stronger than the 2023
corpus models (GPT-4-0613 ... Llama-2). So we are predicting whether *those*
models hallucinated from *our* panel's disagreement. The link is indirect: it
tests whether a source is intrinsically hallucination-prone (ambiguous or
under-specified context), not whether one specific response is unfaithful. The
response-conditioned verification framing (give each agent the candidate
response and ask "is this supported?") is the alternative — it matches
RAGTruth's per-span evaluation but is an ensemble judge, not inter-agent
disagreement. Pick deliberately; this script implements framing (a).

The near-free lexical floor for comparison is scripts/rgt_corpus_disagreement.py.

Requires: OPENROUTER_API_KEY (+ any panel provider keys), torch + sentence-transformers
for the NLI model. Responses are cached, so re-runs are cheap.

Run from the project root:
    python scripts/rgt_pilot.py --n 40
    python scripts/rgt_pilot.py --n 40 --no-cross      # same-family (2 Llama) panel
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
from scipy.stats import spearmanr                                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score     # noqa: E402

from data import load_ragtruth_by_source                              # noqa: E402
from agents import (                                                  # noqa: E402
    query_agents,
    assert_models_available,
    PermanentAgentError,
    PERMANENT_ERROR_PREFIX,
)
from agents.panels import (                                           # noqa: E402
    cross_family_panel,
    same_family_panel,
    family_of,
)
from disagreement import (                                            # noqa: E402
    JaccardDisagreement,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def pick_sources(grouped: dict, n: int, seed: int) -> list:
    """Deterministically pick n complete (6-response) sources."""
    complete = [(sid, rs) for sid, rs in grouped.items() if len(rs) == 6]
    rng = random.Random(seed)
    rng.shuffle(complete)
    return complete[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGTruth QA cross-family disagreement pilot.")
    ap.add_argument("--n", type=int, default=40, help="number of RAGTruth QA sources")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cross", action="store_true",
                    help="use the same-family (2 Llama) panel only")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--strict", action="store_true",
                    help="strict clustering (merge only on mutual entailment)")
    ap.add_argument("--single-linkage", action="store_true")
    ap.add_argument("--peek", type=int, default=0,
                    help="print responses for the first N sources")
    ap.add_argument("--out", default="outputs/rgt_pilot_results.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth QA [test], {args.n} sources (seed={args.seed})")
    grouped = load_ragtruth_by_source(split="test", task_types=["qa"])
    sources = pick_sources(grouped, args.n, args.seed)
    print(f"  Selected {len(sources)} sources "
          f"(of {sum(1 for rs in grouped.values() if len(rs) == 6)} complete).")

    section("Building agent panel")
    try:
        agents = same_family_panel() if args.no_cross else cross_family_panel()
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)
    for a in agents:
        print(f"  - {a.name:<40} [{family_of(a.model)}]")
    has_cross = not args.no_cross

    # Preflight — model IDs, same as scripts/pilot.py: catch a dead/typo'd
    # model in one API call instead of after MAX_RETRIES of backoff on source 1.
    section("Preflight — model IDs")
    try:
        assert_models_available(agents)
    except PermanentAgentError as exc:
        print(f"\n  [ABORT] {exc}\n")
        sys.exit(2)
    print("  All panel models are live.")

    # Preflight — same canary logic as scripts/pilot.py.
    section("Preflight — checking every agent actually answers")
    canary = "What is the capital of France? Answer in one word."
    ok = True
    for a, r in zip(agents, query_agents(agents, canary)):
        empty = not r.text.strip()
        note = ""
        if empty:
            ok = False
            note = (f"  <-- ERROR: {r.error}" if r.error
                    else f"  <-- EMPTY (finish_reason={r.finish_reason}); raise max_tokens")
        print(f"  {a.name:<40} -> {r.text.strip()[:40]!r}{note}")
    if not ok:
        print("\n  [ABORT] An agent returned no usable text.\n")
        sys.exit(2)
    print("  All agents responded.")

    print(f"\n  Loading NLI model: {args.nli_model} (first run downloads weights)...")
    nli = CrossEncoderNLI(model_name=args.nli_model)
    jaccard = JaccardDisagreement()
    linkage = "single" if args.single_linkage else "complete"
    semantic = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                           linkage=linkage)

    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    rows = []
    n_question_errors = 0
    for idx, (sid, responses) in enumerate(sources, start=1):
        ref = responses[0]                       # all six share prompt/question
        prompt = ref.prompt
        question = ref.question

        # Corpus labels for this source (the thing we're trying to predict).
        n_hallucinated = sum(1 for r in responses if r.is_hallucinated)
        any_hallucinated = n_hallucinated > 0

        panel = query_agents(agents, prompt)
        errored = [(a, r) for a, r in zip(agents, panel) if r.is_error]
        if errored:
            # Same reasoning as scripts/pilot.py: don't score a row built from
            # a failed call, and don't let a systematic failure run silently
            # to the end and produce a results file short of `n` rows.
            n_question_errors += 1
            for a, r in errored:
                print(f"  [{idx:>2}/{len(sources)}] {sid}  [!] {a.name}: {r.error}")
            permanent = any(r.error.startswith(PERMANENT_ERROR_PREFIX) for _, r in errored)
            if permanent or n_question_errors > 3:
                raise RuntimeError(
                    f"Aborting after {n_question_errors} source(s) with agent "
                    f"errors (latest: {errored[-1][0].name}: {errored[-1][1].error}). "
                    f"{len(rows)} good row(s) already written to {out_path} — "
                    f"cached answers won't be re-billed on re-run, so fix the "
                    f"underlying issue and re-run."
                )
            continue

        texts = [r.text for r in panel]
        empties = [not t.strip() for t in texts]

        jac = jaccard.score(panel).score
        sem = semantic.score(panel, question=question)

        row = {
            "source_id": sid,
            "question": question,
            "n_hallucinated_corpus": n_hallucinated,
            "any_hallucinated_corpus": any_hallucinated,
            "jaccard": jac,
            "semantic_entropy": sem.score,
            "n_clusters": sem.details["n_clusters"],
            "cluster_sizes": sem.details["cluster_sizes"],
            "responses": {a.name: t for a, t in zip(agents, texts)},
            "finish_reasons": {a.name: r.finish_reason for a, r in zip(agents, panel)},
            "empty_count": sum(empties),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {sid} ---")
            print(f"      Q: {question}")
            print(f"      corpus #hallucinated: {n_hallucinated}/6")
            for a, t in zip(agents, texts):
                print(f"      {a.name.split('/')[-1]}: {t.strip()[:160]}")

        print(f"  [{idx:>2}/{len(sources)}] {sid}  "
              f"jac={jac:.2f}  sem={sem.score:.2f}  "
              f"corpus_hallu={n_hallucinated}/6")

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    section("A. Panel disagreement vs corpus hallucination")
    total_empty = sum(r["empty_count"] for r in rows)
    if total_empty:
        print(f"  [!] {total_empty} EMPTY responses — an agent truncated; fix and re-run.")

    jac_arr = np.array([r["jaccard"] for r in rows])
    sem_arr = np.array([r["semantic_entropy"] for r in rows])
    nh = np.array([r["n_hallucinated_corpus"] for r in rows])
    anyh = np.array([r["any_hallucinated_corpus"] for r in rows], dtype=float)

    print(f"  mean Jaccard            : {jac_arr.mean():.3f}")
    print(f"  mean semantic entropy   : {sem_arr.mean():.3f}")
    print(f"  sources with any hallu  : {anyh.mean():.1%}")

    for name, x in (("jaccard", jac_arr), ("semantic_entropy", sem_arr)):
        rho, p = spearmanr(x, nh)
        print(f"\n  Spearman({name}, #hallucinated): rho={rho:+.3f}  p={p:.3g}")
        if anyh.min() != anyh.max():
            auc = roc_auc_score(anyh, x)
            apr = average_precision_score(anyh, x)
            print(f"  AUC-ROC({name} -> any-hallucinated): {auc:.3f}  "
                  f"AUC-PR={apr:.3f} (base {anyh.mean():.3f})")

    # Saturation guard, as in scripts/pilot.py.
    from collections import Counter
    cc = Counter(r["n_clusters"] for r in rows)
    print(f"\n  n_clusters spread: " + ", ".join(f"{k}->{cc[k]}" for k in sorted(cc)))
    if cc.get(len(agents), 0) > 0.7 * len(rows):
        print("  [!] semantic entropy looks SATURATED (most sources all-distinct) — "
              "expected on long RAG answers; this is why claim-level is the next measure.")

    section("Done")
    print(f"  Compare against the near-free lexical floor:")
    print(f"    python scripts/rgt_corpus_disagreement.py --task qa")
    print(f"  Rows written to: {out_path}\n")


if __name__ == "__main__":
    main()