"""Diagnose a pilot run from its results JSONL — no API/model needed.

Reads outputs/pilot_results.jsonl (as written by scripts/pilot.py) and reports
the things that reveal whether the disagreement metrics are actually
discriminating or have saturated:

  - response length distribution per agent (long answers => NLI out-of-dist)
  - histogram of semantic-entropy values and cluster counts
  - histogram of Jaccard values
  - correctness-grade distribution (correct / incorrect / unclear)
  - a few full example rows so you can eyeball the actual answers

Run from the project root:
    python scripts/inspect_pilot.py
    python scripts/inspect_pilot.py --file outputs/pilot_results.jsonl --examples 5
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def section(t: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def hist(values, edges):
    """Simple bucket counts for a list of floats given bucket right-edges."""
    counts = [0] * (len(edges) + 1)
    labels = []
    lo = 0.0
    for e in edges:
        labels.append(f"[{lo:.2f},{e:.2f})")
        lo = e
    labels.append(f"[{lo:.2f},1.00]")
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v < e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return list(zip(labels, counts))


def bar(n: int, total: int, width: int = 30) -> str:
    filled = int(round(width * n / total)) if total else 0
    return "#" * filled + "." * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/pilot_results.jsonl")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"No such file: {path}")

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(rows)
    section(f"Loaded {n} rows from {path}")

    # -- response lengths per agent -------------------------------------
    section("Response length per agent (words)")
    agent_names = list(rows[0]["responses"].keys())
    for name in agent_names:
        lengths = [len(r["responses"][name].split()) for r in rows]
        print(f"  {name:<40} "
              f"median={statistics.median(lengths):>4.0f}  "
              f"mean={statistics.mean(lengths):>5.1f}  "
              f"max={max(lengths):>4d}")
    print("\n  (Sentence-level NLI expects SHORT answers. Medians >> ~15-20 words"
          "\n   are a strong hint the clustering is out of distribution.)")

    # -- semantic entropy + cluster-count distribution ------------------
    section("Semantic entropy distribution")
    sem = [r["semantic_entropy"] for r in rows]
    print(f"  mean={statistics.mean(sem):.3f}  "
          f"min={min(sem):.3f}  max={max(sem):.3f}")
    for label, c in hist(sem, [0.001, 0.30, 0.60, 0.90]):
        print(f"    {label:<14} {bar(c, n)} {c}")
    cluster_counts = Counter(r["n_clusters"] for r in rows)
    print("\n  n_clusters per question:")
    for k in sorted(cluster_counts):
        print(f"    {k} cluster(s): {bar(cluster_counts[k], n)} {cluster_counts[k]}")
    if cluster_counts.get(len(agent_names), 0) > 0.7 * n:
        print("\n  >>> WARNING: most questions put EVERY agent in its own cluster."
              "\n      The metric has saturated — nothing is merging. See length"
              "\n      stats above; switch agents to concise answers and re-run.")

    # -- jaccard --------------------------------------------------------
    section("Jaccard distribution")
    jac = [r["jaccard"] for r in rows]
    print(f"  mean={statistics.mean(jac):.3f}  min={min(jac):.3f}  max={max(jac):.3f}")
    for label, c in hist(jac, [0.30, 0.60, 0.90]):
        print(f"    {label:<14} {bar(c, n)} {c}")

    # -- grade distribution ---------------------------------------------
    section("Correctness-grade distribution (NLI proxy)")
    grade_counter = Counter()
    for r in rows:
        for g in r["grades"].values():
            grade_counter[{True: "correct", False: "incorrect", None: "unclear"}[g]] += 1
    total_grades = sum(grade_counter.values())
    for k in ("correct", "incorrect", "unclear"):
        c = grade_counter.get(k, 0)
        print(f"    {k:<10} {bar(c, total_grades)} {c}")
    if grade_counter.get("unclear", 0) > 0.5 * total_grades:
        print("\n  >>> WARNING: most answers grade as 'unclear' — the NLI grader"
              "\n      can't match verbose answers to the short gold answer. This"
              "\n      masks shared-bias failures (both_llamas_wrong needs False,"
              "\n      not None). Concise answers fix this too.")

    # -- examples -------------------------------------------------------
    section(f"First {args.examples} questions (full answers)")
    for r in rows[: args.examples]:
        print(f"\n  {r['uid']} [{r['category']}]  "
              f"jac={r['jaccard']:.2f} sem={r['semantic_entropy']:.2f} "
              f"clusters={r['cluster_sizes']}")
        print(f"    Q: {r['question']}")
        print(f"    gold: {r['correct_answer']}")
        for name, text in r["responses"].items():
            g = r["grades"][name]
            gl = {True: "OK ", False: "WRONG", None: "?  "}[g]
            print(f"    [{gl}] {name}: {text.strip()[:160]}")

    print()


if __name__ == "__main__":
    main()
