"""Audit the NLI correctness grader from a pilot JSONL — for hand-checking.

The AUROC in evaluate.py is only as trustworthy as the 'hallucinated' label,
which comes from the NLI correctness proxy in pilot.py. This tool dumps the
graded rows in a readable form, defaulting to the ones most likely to be WRONG
(graded 'incorrect' or 'unclear'), so you can eyeball whether the label is
right before drawing conclusions.

If the pilot stored per-agent cluster membership ('cluster_of'), it also flags
an automatic inconsistency: two answers the clusterer put in the SAME meaning
cluster but the grader scored DIFFERENTLY. Same meaning => same correctness, so
one of those grades is almost certainly wrong — a cheap way to find grader
errors without reading everything.

    python scripts/audit_grader.py                       # incorrect + unclear
    python scripts/audit_grader.py --filter all
    python scripts/audit_grader.py --filter incorrect --limit 40
    python scripts/audit_grader.py --flag-only           # only inconsistencies
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

GRADE_STR = {True: "OK   ", False: "WRONG", None: "?    "}


def section(t: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def find_inconsistencies(row):
    """Pairs of agents in the same cluster but with different grades."""
    cluster_of = row.get("cluster_of")
    if not cluster_of:
        return []
    grades = row["grades"]
    names = list(cluster_of.keys())
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if cluster_of[a] == cluster_of[b]:
                ga, gb = grades.get(a), grades.get(b)
                # Different, and not merely one being 'unclear' vs a real grade.
                if ga is not None and gb is not None and ga != gb:
                    out.append((a, b, ga, gb))
    return out


def print_row(row) -> None:
    print(f"\n  {row['uid']} [{row.get('category','?')}]  "
          f"sem={row.get('semantic_entropy', float('nan')):.2f} "
          f"jac={row.get('jaccard', float('nan')):.2f}")
    print(f"    Q:    {row['question']}")
    print(f"    gold: {row['correct_answer']}")
    for name, text in row["responses"].items():
        g = row["grades"].get(name)
        cl = ""
        if row.get("cluster_of"):
            cl = f" (cluster {row['cluster_of'].get(name)})"
        print(f"    [{GRADE_STR[g]}]{cl} {name.split('/')[-1]}: {str(text).strip()[:150]}")
    for a, b, ga, gb in find_inconsistencies(row):
        print(f"    ** INCONSISTENT: {a.split('/')[-1]} ({GRADE_STR[ga].strip()}) and "
              f"{b.split('/')[-1]} ({GRADE_STR[gb].strip()}) are in the same cluster **")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/pilot_results.jsonl")
    ap.add_argument("--filter", choices=["all", "correct", "incorrect", "unclear"],
                    default=None,
                    help="which questions to show; default shows incorrect+unclear")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--flag-only", action="store_true",
                    help="show only questions with a same-cluster/different-grade flag")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    section(f"Grader audit — {len(rows)} questions from {path}")
    grade_ctr = Counter()
    for r in rows:
        for g in r["grades"].values():
            grade_ctr[{True: "correct", False: "incorrect", None: "unclear"}[g]] += 1
    tot = sum(grade_ctr.values())
    print(f"  grades: correct={grade_ctr['correct']}  incorrect={grade_ctr['incorrect']}  "
          f"unclear={grade_ctr['unclear']}  ({tot} answers)")
    if tot:
        print(f"  unclear rate: {grade_ctr['unclear']/tot:.0%}  "
              f"(high => grader can't judge; label noise)")

    has_clusters = any(r.get("cluster_of") for r in rows)
    inconsistent = [r for r in rows if find_inconsistencies(r)]
    if has_clusters:
        print(f"  same-cluster/different-grade inconsistencies: "
              f"{len(inconsistent)} / {len(rows)} questions "
              f"(each is a likely grader error)")
    else:
        print("  (no per-agent cluster membership in this file — re-run the pilot "
              "to enable the automatic inconsistency flag; hand-check below.)")

    # -- select rows to display -----------------------------------------
    def is_incorrect(r):
        return any(g is False for g in r["grades"].values())

    def is_unclear(r):
        return any(g is None for g in r["grades"].values())

    if args.flag_only:
        show = inconsistent
        section(f"Inconsistency-flagged questions ({len(show)})")
    elif args.filter == "all":
        show = rows
        section("All questions")
    elif args.filter == "correct":
        show = [r for r in rows if not is_incorrect(r) and not is_unclear(r)]
        section(f"Questions with all-correct grades ({len(show)})")
    elif args.filter == "incorrect":
        show = [r for r in rows if is_incorrect(r)]
        section(f"Questions with an 'incorrect' grade ({len(show)})")
    elif args.filter == "unclear":
        show = [r for r in rows if is_unclear(r)]
        section(f"Questions with an 'unclear' grade ({len(show)})")
    else:  # default: incorrect or unclear
        show = [r for r in rows if is_incorrect(r) or is_unclear(r)]
        section(f"Suspect questions — incorrect or unclear ({len(show)}). "
                f"Hand-check these first.")

    for r in show[: args.limit]:
        print_row(r)

    print("\n  Tip: compare each answer to the gold. If a WRONG matches the gold's "
          "meaning\n  (or a ? clearly agrees/disagrees), the grader mislabeled it — "
          "that noise\n  flows straight into the AUROC.\n")


if __name__ == "__main__":
    main()