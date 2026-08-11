"""Sampled pilot: k temperature samples per model, pooled semantic entropy.

Motivation
----------
With one answer per model (the plain pilot), semantic entropy over few
responses can only take a handful of discrete values. That coarse
quantisation flattens the rank statistic AUROC rewards, so the metric's real
class separation is hidden. This script draws k samples from each of 4 base
models (temperature>0, seeded for reproducibility) — LLAMA_SMALL + the
cross-family panel's 3 (LLAMA_LARGE, GPT-OSS, Qwen) — pools the 4·k
responses, and clusters the pool — giving semantic entropy a continuous
range (0 … ln(4k)). This is the standard Farquhar/Kuhn multi-sample
estimator, applied across an inter-agent panel. LLAMA_SMALL is included
alongside cross_family_panel's models (rather than sampling only those 3) so
the same-family (2-Llama) diversity contrast below has a genuine same-family
pair to compare against, identified by model id — not by list position.

Output rows are schema-compatible with scripts/evaluate.py and
scripts/audit_grader.py, so the same evaluation runs unchanged:

    python scripts/pilot_sampled.py --n 50 --k 5
    python scripts/evaluate.py --file outputs/pilot_sampled.jsonl

Labels
------
Disagreement is measured over the pooled samples. The hallucination LABEL is
kept at the base-model level for comparability with the plain pilot: each base
model is graded by MAJORITY VOTE over its k samples (more robust than a single
draw), then any_wrong / majority_wrong are computed over the 4 base grades.

Requires GROQ_API_KEY + torch + sentence-transformers. Samples are cached per
(model, prompt, temperature, seed), so re-runs are free and reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa, TRUTHFULQA_QUERY_CONFIG              # noqa: E402
from agents import query_agents                                        # noqa: E402
from agents.panels import (                                            # noqa: E402
    sampling_panel, SAMPLING_BASE_MODELS, LLAMA_SMALL, LLAMA_LARGE)
from disagreement import (                                             # noqa: E402
    JaccardDisagreement, SemanticEntropyDisagreement, CrossEncoderNLI)
from evaluation import grade_correct                                   # noqa: E402


def section(t: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def majority_grade(grades: List[Optional[bool]]) -> Optional[bool]:
    """Majority vote over a base model's k sample grades; None if all unclear/tied."""
    graded = [g for g in grades if g is not None]
    if not graded:
        return None
    n_wrong = sum(g is False for g in graded)
    n_right = sum(g is True for g in graded)
    if n_wrong == n_right:
        return None
    return False if n_wrong > n_right else True


def main() -> None:
    ap = argparse.ArgumentParser(description="Sampled (k-per-model) disagreement pilot.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=5, help="samples per base model")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--single-linkage", action="store_true")
    ap.add_argument("--out", default="outputs/pilot_sampled.jsonl")
    ap.add_argument("--peek", type=int, default=0)
    args = ap.parse_args()

    section(f"Loading {args.n} questions; k={args.k} samples/model @ T={args.temperature}")
    samples = load_truthfulqa(n=args.n, seed=args.seed)
    # TRUTHFULQA_QUERY_CONFIG minus 'temperature': this script's whole point is
    # k samples at args.temperature > 0 (sampling_panel's own parameter), which
    # would collide with (and be overridden by) the config's deterministic 0.0.
    tqa_kwargs = {k: v for k, v in TRUTHFULQA_QUERY_CONFIG.items() if k != "temperature"}
    # LLAMA_SMALL + cross_family_panel's 3 (union, not concatenation — Llama
    # large is only in the latter): gives the same-family contrast below a
    # genuine 2-Llama pair to compare against.
    sampled_base_models = [LLAMA_SMALL] + SAMPLING_BASE_MODELS
    groups, base_models = sampling_panel(args.k, temperature=args.temperature,
                                          base_models=sampled_base_models, **tqa_kwargs)
    flat_agents = [a for g in groups for a in g]
    print(f"  {len(samples)} questions x {len(flat_agents)} agents "
          f"({len(base_models)} models x {args.k} samples)")
    # Identity, not position: base_models[:2] no longer means "the Llama
    # pair" once cross_family_panel's models are folded in.
    llama_positions = [i for i, m in enumerate(base_models) if m in (LLAMA_SMALL, LLAMA_LARGE)]

    # Preflight on the first sample of each model.
    section("Preflight")
    firsts = [g[0] for g in groups]
    ok = True
    for a, r in zip(firsts, query_agents(firsts, "What is the capital of France? Answer in one word.")):
        empty = not r.text.strip()
        ok = ok and not empty
        print(f"  {a.name:<40} -> {r.text.strip()[:30]!r}"
              f"{'  <-- EMPTY: ' + str(r.error) if empty else ''}")
    if not ok:
        print("  [ABORT] an agent returned no text; check groq_agent.py / token budget.")
        sys.exit(2)

    nli = CrossEncoderNLI(model_name=args.nli_model)
    jaccard = JaccardDisagreement()
    linkage = "single" if args.single_linkage else "complete"
    semantic = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                           linkage=linkage)
    print(f"  clustering: {'strict' if args.strict else 'relaxed'} + {linkage}-linkage")

    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    rows = []
    for idx, s in enumerate(samples, start=1):
        responses = query_agents(flat_agents, s.prompt)
        texts = [r.text for r in responses]

        # Regroup flat responses back into per-base-model sample lists.
        per_model_texts: List[List[str]] = []
        pos = 0
        for g in groups:
            per_model_texts.append(texts[pos:pos + len(g)])
            pos += len(g)
        pooled = [t for grp in per_model_texts for t in grp]

        sem = semantic.score(pooled, question=s.question)
        jac = jaccard.score(pooled).score

        # Base-model grades by majority vote over that model's k samples.
        # This pilot only ever samples the open-ended prompt, so grading
        # routes through the NLI proxy, not letter extraction.
        base_grades = {
            m: majority_grade([grade_correct(nli, t, s, prompt_format="open") for t in grp])
            for m, grp in zip(base_models, per_model_texts)
        }
        grades_list = list(base_grades.values())
        graded = [g for g in grades_list if g is not None]
        any_wrong = any(g is False for g in grades_list)
        majority_wrong = bool(graded) and sum(g is False for g in graded) > len(graded) / 2

        # Same-family (2 Llamas) pooled samples for the diversity contrast,
        # found by model id (llama_positions) — not by list position.
        same_pool = [t for i in llama_positions for t in per_model_texts[i]]
        sem_same = semantic.score(same_pool, question=s.question)

        panel_agrees = sem.details["n_clusters"] == 1

        row = {
            "uid": s.uid, "category": s.category, "question": s.question,
            "correct_answer": s.correct_answer,
            "k": args.k, "n_samples": len(pooled),
            "responses": {m: grp[0] for m, grp in zip(base_models, per_model_texts)},
            "all_samples": {m: grp for m, grp in zip(base_models, per_model_texts)},
            "grades": base_grades,
            "jaccard": jac,
            "semantic_entropy": sem.score,
            "n_clusters": sem.details["n_clusters"],
            "cluster_sizes": sem.details["cluster_sizes"],
            "semantic_entropy_same_family": sem_same.score,
            "panel_agrees": panel_agrees,
            "panel_majority_wrong": majority_wrong,
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} ---  Q: {s.question[:70]}")
            for m, grp in zip(base_models, per_model_texts):
                print(f"    {m.split('/')[-1]} (grade={base_grades[m]}):")
                for t in grp:
                    print(f"        {t.strip()[:80]}")
            print(f"    -> pooled clusters={sem.details['cluster_sizes']} sem={sem.score:.3f}")
        print(f"  [{idx:>2}/{len(samples)}] {s.uid}  sem={sem.score:.3f} "
              f"clusters={sem.details['cluster_sizes']}")

    # -- summary --------------------------------------------------------
    section("Summary")
    mean_sem = statistics.mean(r["semantic_entropy"] for r in rows)
    distinct = sorted({round(r["semantic_entropy"], 3) for r in rows})
    print(f"  mean semantic entropy : {mean_sem:.3f}")
    print(f"  distinct sem values   : {len(distinct)} "
          f"(was {len(base_models)} with k=1) -> "
          f"{'CONTINUOUS' if len(distinct) > 5 else 'still coarse'}")
    grade_ctr = Counter()
    for r in rows:
        for g in r["grades"].values():
            grade_ctr[{True: "correct", False: "incorrect", None: "unclear"}[g]] += 1
    print(f"  base-model grades     : correct={grade_ctr['correct']} "
          f"incorrect={grade_ctr['incorrect']} unclear={grade_ctr['unclear']}")
    sbfn = sum(r["panel_agrees"] and r["panel_majority_wrong"] for r in rows)
    print(f"  shared-bias FNs       : {sbfn} / {len(rows)}")
    print(f"\n  Rows -> {out_path}")
    print(f"  Evaluate: python scripts/evaluate.py --file {out_path}")


if __name__ == "__main__":
    main()