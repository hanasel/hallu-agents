#!/usr/bin/env python3
"""Per-model within-model AUC for the RAGTruth corpus-response probe.

Replaces the "Within-model AUC (reliable models)" row of the appendix table,
which reports a range without recording which models it ranges over. Six
per-model numbers need no reliability filter: a model whose AUC is suppressed
is named, with the reason.

The script does not assume a schema. It prints the fields it finds, picks the
label and grouping columns, and computes an AUC for every plausible numeric
score column, so whichever field holds the dissent score, its answer is in the
output.

These files are one row per SOURCE with a per-model dict inside; the script
unpacks that automatically to one row per (source, model).

Run from the repository root:

    python rgt_within_model_auc.py --score dissent outputs/rgt_corpus_*.jsonl

Pass --score dissent explicitly. Without it the script scores every numeric
column, which includes n_hallucinated -- that is the label aggregated over the
source's six responses, so its "AUC" is leakage, not a result.

Then paste the LaTeX block it prints at the end into the appendix.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

MIN_CLASS = 10  # matches MIN_CLASS_FOR_AUROC in scripts/simpleqa_pilot.py

LABEL_FIELDS = ["is_hallucinated", "hallucinated", "label", "y", "is_hallu"]
GROUP_FIELDS = ["source_model", "model", "response_model", "agent"]
SKIP_SUBSTR = ("id", "index", "n_spans", "split", "seed")


def auc(scores: list[float], labels: list[int]) -> float | None:
    """Rank-based AUC with ties counted as half-wins. No dependencies."""
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return None
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


NESTED_FIELDS = ["per_model", "per_response", "models", "responses"]


def load(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def flatten(rows: list[dict]) -> tuple[list[dict], str | None]:
    """The corpus-probe files are one row per SOURCE, with a per-model dict
    inside. Flatten to one row per (source, model) so the rest of the script
    can treat every file the same way. Returns (rows, the field it unpacked)."""
    if not rows:
        return rows, None
    nested = next((f for f in NESTED_FIELDS
                   if isinstance(rows[0].get(f), dict)), None)
    if nested is None:
        return rows, None
    flat: list[dict] = []
    for r in rows:
        outer = {k: v for k, v in r.items() if k != nested}
        for model, inner in (r.get(nested) or {}).items():
            if not isinstance(inner, dict):
                continue
            rec = dict(outer)
            rec["model"] = model
            rec.update(inner)
            flat.append(rec)
    return flat, nested


def pick(rows: list[dict], candidates: list[str]) -> str | None:
    keys = set(rows[0])
    for c in candidates:
        if c in keys:
            return c
    return None


def numeric_fields(rows: list[dict], exclude: set[str]) -> list[str]:
    out = []
    for k, v in rows[0].items():
        if k in exclude or any(s in k.lower() for s in SKIP_SUBSTR):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            vals = {r.get(k) for r in rows if isinstance(r.get(k), (int, float))}
            if len(vals) > 2:          # a constant or binary column is not a score
                out.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="rgt_corpus_*.jsonl (globs are expanded)")
    ap.add_argument("--label", help="override the label field")
    ap.add_argument("--group", help="override the per-model grouping field")
    ap.add_argument("--score", help="compute only this score field")
    args = ap.parse_args()

    paths: list[str] = []
    for f in args.files:
        paths.extend(sorted(glob.glob(f)) or [f])

    for path in paths:
        rows = load([path])
        if not rows:
            print(f"\n{path}: empty or unreadable\n")
            continue

        print("=" * 74)
        print(f"  {os.path.basename(path)}   ({len(rows)} rows)")
        print("=" * 74)
        rows, nested = flatten(rows)
        if nested:
            print(f"  unpacked nested field {nested!r} -> {len(rows)} (source, model) rows")
        print("  fields:", ", ".join(sorted(rows[0])))

        label = args.label or pick(rows, LABEL_FIELDS)
        group = args.group or pick(rows, GROUP_FIELDS)
        if not label or not group:
            print(f"\n  Could not identify label ({label}) or group ({group}).")
            print("  Re-run with --label and --group naming the right fields.\n")
            continue

        scores = [args.score] if args.score else numeric_fields(rows, {label, group})
        if not scores:
            print("\n  No numeric score column found. Pass --score explicitly.\n")
            continue
        print(f"  label={label}  group={group}  scores={scores}\n")

        for score in scores:
            by_model: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for r in rows:
                s, y = r.get(score), r.get(label)
                if isinstance(s, (int, float)) and y is not None:
                    by_model[str(r.get(group))].append((float(s), int(bool(y))))

            print(f"  --- score: {score} ---")
            print(f"  {'model':44s} {'n':>5} {'pos':>5} {'neg':>5}  AUC")
            kept, dropped = [], []
            for model in sorted(by_model):
                pairs = by_model[model]
                ss = [p[0] for p in pairs]
                yy = [p[1] for p in pairs]
                pos, neg = sum(yy), len(yy) - sum(yy)
                a = auc(ss, yy)
                if a is None or min(pos, neg) < MIN_CLASS:
                    reason = "one class empty" if a is None else f"min class {min(pos, neg)} < {MIN_CLASS}"
                    dropped.append((model, len(yy), pos, neg, reason))
                    print(f"  {model:44s} {len(yy):5d} {pos:5d} {neg:5d}  -- ({reason})")
                else:
                    kept.append((model, a))
                    print(f"  {model:44s} {len(yy):5d} {pos:5d} {neg:5d}  {a:.3f}")
            if kept:
                vals = [a for _, a in kept]
                print(f"  {'range over reported models':44s} "
                      f"{'':17s}  {min(vals):.3f}-{max(vals):.3f}")
            print()

            if not args.score and score != scores[-1]:
                continue

        print("  LaTeX rows for the appendix (edit the score column you want):\n")
        for model in sorted(by_model):
            print(f"    \\texttt{{{model}}} & \\\\")
        print()
        if dropped:
            print("  Caption clause: AUC is not reported for "
                  + ", ".join(m for m, *_ in dropped)
                  + f" -- fewer than {MIN_CLASS} responses in one class.\n")


if __name__ == "__main__":
    main()