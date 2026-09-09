#!/usr/bin/env python3
"""Max/min AUROC per panel size from a panel_sweep9_*.csv, emitted as a
pgfplots coordinate block for Figure 4.3.

    python3 sweep_bars.py outputs/panel_sweep9_simpleqa.csv

Column names are auto-detected. If detection fails, pass them explicitly:

    python3 sweep_bars.py FILE --auroc auroc --size n
    python3 sweep_bars.py FILE --auroc auroc --members panel   # size = count of members
"""
import argparse, csv, re, sys
from collections import defaultdict

# Canonical judge-label means, Table 4.6. Offsets are computed against these so
# the figure cannot drift from the table. Override with --means for TruthfulQA:
#   open-ended  0.561 0.585 0.601 0.612 0.619 0.626 0.631 0.633
#   MC1         0.681 0.751 0.798 0.823 0.839 0.852 0.861 0.866
CANON = {2: 0.641, 3: 0.725, 4: 0.795, 5: 0.833,
         6: 0.856, 7: 0.871, 8: 0.881, 9: 0.889}


def pick(cols, *needles, avoid=()):
    for c in cols:
        low = c.lower()
        if any(n in low for n in needles) and not any(a in low for a in avoid):
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--auroc", help="AUROC column name")
    p.add_argument("--size", help="panel-size column name")
    p.add_argument("--members", help="panel-membership column; size = number of members")
    p.add_argument("--means", help="comma-separated means for N=2..9, overriding the canonical block")
    args = p.parse_args()

    with open(args.path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("empty file")
    cols = list(rows[0])
    print(f"columns: {cols}\nrows: {len(rows)}\n", file=sys.stderr)

    auroc_c = args.auroc or pick(cols, "auroc", "auc_roc", "roc_auc") or pick(cols, "auc", avoid=("pr", "prc"))
    size_c = args.size or pick(cols, "panel_size", "n_agents", "size") or ("n" if "n" in cols else None)
    memb_c = args.members or pick(cols, "panel", "members", "agents", "subset")
    if not auroc_c:
        sys.exit(f"could not find an AUROC column in {cols}; pass --auroc")

    def size_of(r):
        if size_c and str(r.get(size_c, "")).strip().isdigit():
            return int(r[size_c])
        if memb_c and r.get(memb_c):
            return len([t for t in re.split(r"[,;+|/\s]+", str(r[memb_c])) if t])
        return None

    by = defaultdict(list)
    skipped = 0
    for r in rows:
        n = size_of(r)
        try:
            v = float(r[auroc_c])
        except (TypeError, ValueError):
            skipped += 1
            continue
        if n is None:
            skipped += 1
            continue
        by[n].append(v)
    if not by:
        sys.exit("no rows parsed; check --auroc / --size / --members")
    print(f"using auroc={auroc_c!r} size={size_c!r} members={memb_c!r} "
          f"(skipped {skipped} rows)\n", file=sys.stderr)

    means = CANON.copy()
    if args.means:
        means = {n: float(v) for n, v in zip(range(2, 10), args.means.split(","))}

    print(f"{'N':>2} {'panels':>7} {'worst':>7} {'mean':>7} {'best':>7}   {'csv mean':>8}")
    for n in sorted(by):
        v = by[n]
        cm = sum(v) / len(v)
        m = means.get(n, cm)
        flag = "  <-- differs from canonical mean" if abs(cm - m) > 0.005 else ""
        print(f"{n:>2} {len(v):>7} {min(v):>7.3f} {m:>7.3f} {max(v):>7.3f}   {cm:>8.3f}{flag}")

    if 9 in by:
        print(f"\nN=9 in this file reads {max(by[9]):.4f}. Canonical is 0.8892 for "
              f"SimpleQA (0.633 open, 0.866 MC1). If it reads 0.892 this CSV was\n"
              f"written under the superseded judge label and the whiskers will be too.",
              file=sys.stderr)

    print("\n% paste into the \\addplot+ coordinates block")
    bad = False
    for n in sorted(by):
        m = means.get(n, sum(by[n]) / len(by[n]))
        up, lo = round(max(by[n]) - m, 3) + 0.0, round(m - min(by[n]), 3) + 0.0
        bad |= up < 0 or lo < 0
        print(f"  ({n},{m:.3f}) +- (0,{up:.3f}) -= (0,{lo:.3f})")
    if bad:
        print("\nNegative offset: the CSV and the canonical means disagree "
              "(wrong file, wrong label, or wrong --means).", file=sys.stderr)


if __name__ == "__main__":
    main()