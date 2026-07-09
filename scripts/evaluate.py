"""Evaluate whether disagreement predicts hallucination — from a pilot JSONL.

Reads outputs/pilot_results.jsonl (written by scripts/pilot.py) and asks the
core research question: does the semantic-entropy disagreement score separate
questions where the panel hallucinated from questions where it didn't? And
does it beat the lexical Jaccard baseline at that?

It reports, for each predictor (semantic entropy, Jaccard) and each label:
  - AUROC (rank-based Mann-Whitney; 0.5 = chance, 1.0 = perfect separation)
  - mean score on hallucinated vs clean questions (the separation)
and prints the shared-bias false negatives (panel agrees AND is wrong) as the
structural recall ceiling — cases no disagreement signal can catch.

Hallucination labels (both reported; they answer different questions):
  any_wrong      : at least one agent graded incorrect  — "did anyone slip?"
  majority_wrong : most graded answers incorrect        — "did the panel fail?"

CAVEATS, stated plainly because they bound the numbers:
  - The label comes from the NLI correctness *proxy* in pilot.py, not human
    judgement — it is noisy, especially on verbose/negated answers.
  - Questions where every answer graded 'unclear' are unlabeled and excluded.
  - AUROC on n=50 is a rough estimate; report a confidence interval before
    drawing strong conclusions. This is a pilot signal, not a final result.

    python scripts/evaluate.py
    python scripts/evaluate.py --file outputs/pilot_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def section(t: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Rank-based AUROC (equivalent to the Mann-Whitney U statistic).

    scores : predictor values; labels : 1 = positive (hallucination), 0 = clean.
    Handles ties via average ranks. Returns None if one class is empty.
    """
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    # Average ranks (1-based) with tie handling.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # average of ranks i+1..j+1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    sum_ranks_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def label_any_wrong(row) -> Optional[int]:
    grades = list(row["grades"].values())
    if any(g is False for g in grades):
        return 1
    if any(g is True for g in grades):
        return 0
    return None  # all unclear -> unlabeled


def label_majority_wrong(row) -> Optional[int]:
    graded = [g for g in row["grades"].values() if g is not None]
    if not graded:
        return None
    return 1 if sum(g is False for g in graded) > len(graded) / 2 else 0


def evaluate_predictor(rows, score_key: str, label_fn) -> dict:
    scores, labels = [], []
    for r in rows:
        lab = label_fn(r)
        if lab is None:
            continue
        scores.append(r[score_key])
        labels.append(lab)
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    return {
        "n": len(labels),
        "n_pos": sum(labels),
        "auroc": auroc(scores, labels),
        "mean_pos": statistics.mean(pos) if pos else None,
        "mean_neg": statistics.mean(neg) if neg else None,
    }


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/pilot_results.jsonl")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    section(f"Evaluating {len(rows)} questions from {path}")
    if len(rows) < 30:
        print("  [!] n < 30 — AUROC will be very noisy; treat as indicative only.")

    for label_name, label_fn in (("any_wrong", label_any_wrong),
                                 ("majority_wrong", label_majority_wrong)):
        section(f"Label: {label_name}")
        labeled = [r for r in rows if label_fn(r) is not None]
        n_pos = sum(label_fn(r) for r in labeled)
        print(f"  labeled questions: {len(labeled)} / {len(rows)}  "
              f"(positives={n_pos}, unlabeled/unclear={len(rows) - len(labeled)})")
        if not labeled or n_pos == 0 or n_pos == len(labeled):
            print("  Cannot compute AUROC (need both hallucinated and clean, labeled).")
            continue

        print(f"\n  {'predictor':<20}{'AUROC':>8}{'mean(halluc)':>14}{'mean(clean)':>13}")
        for key, nice in (("semantic_entropy", "semantic entropy"),
                          ("jaccard", "jaccard (baseline)")):
            r = evaluate_predictor(labeled, key, label_fn)
            print(f"  {nice:<20}{fmt(r['auroc']):>8}{fmt(r['mean_pos']):>14}"
                  f"{fmt(r['mean_neg']):>13}")
        print("\n  AUROC > 0.5 means the score is higher on hallucinated questions.")

    # -- shared-bias false negatives: the recall ceiling --------------------
    section("Shared-bias false negatives (recall ceiling)")
    sbfn = [r for r in rows if r.get("panel_agrees") and r.get("panel_majority_wrong")]
    total_wrong = [r for r in rows if label_majority_wrong(r) == 1]
    print(f"  Whole panel agrees AND is wrong: {len(sbfn)} / {len(rows)}")
    if total_wrong:
        print(f"  As a share of panel-failure questions: "
              f"{len(sbfn)}/{len(total_wrong)} = "
              f"{len(sbfn) / len(total_wrong):.0%}")
    print("  These score ~0 disagreement yet are wrong — no disagreement signal")
    print("  can catch them. They cap achievable recall and motivate a grounding")
    print("  signal as a complement.")
    for r in sbfn[:8]:
        print(f"    - {r['uid']} [{r['category']}]  sem={r['semantic_entropy']:.2f}")

    print()


if __name__ == "__main__":
    main()