"""Full panel-subset sweep, dose-response, partitions and held-out validation.

The nine-agent replacement for the analysis that produced Appendix B of the
report (Tables B.3-B.8) from the old five-agent harvest. That analysis was
never committed as code — outputs/panel_sweep*.csv exist with no producer —
so this script reconstructs it, generalised from 5 agents to N.

It reads a results file written by scripts/disagreement_pilot.py (which stores
every agent's raw response text, per-agent grades and per-agent abstention
flags on each row) and re-scores EVERY non-trivial subset of the pool offline.
No API calls: adding a panel is free once the pool has been queried, which is
the property scripts/disagreement_pilot.py's module docstring describes.

  9 agents -> 2^9 - 9 - 1 = 502 non-trivial panels.

What it produces
----------------
  A. Full sweep, one row per panel            -> <out>.csv   (report Table B.4)
  B. Mean lift by panel size                  -> report Sec 4.6 dose-response
  C. Mean lift by family count, within size   -> report Table B.5 / Sec 5.6.1
  D. Mean lift by weakest-agent presence      -> report Table B.6 / Sec 5.5
  E. Held-out validation, one row per agent   -> report Table B.3 / B.7, Table 8.1

The fixed label
---------------
The hallucination label is derived ONCE from the full pool — a question is
positive when a strict majority of the pool's graded responses are incorrect —
and held fixed across every subset. Only the disagreement score varies. This
is the discipline Section B.1 of the report describes: deriving the label
inside each subset would make the majority threshold depend on panel size and
render comparisons across sizes invalid. Because the label is constant, the
base rate is constant and AUC-PR is directly comparable down the whole sweep.

Disagreement, by format
-----------------------
  --format mc    disagreement.MCExactMatch over the subset's response texts.
                 Exactly the measurement the MC runs used; no NLI, no torch.
  --format open  normalised entropy of the FULL pool's meaning-clusters
                 RESTRICTED to the subset's members. Note this is not the same
                 as re-clustering each subset from scratch: complete-linkage
                 over 3 responses can merge pairs that stay apart among 9. It
                 is chosen deliberately — every panel is then scored under one
                 equivalence relation, so differences between panels are not
                 confounded with differences in how the NLI model happened to
                 cluster each subset. State this in the write-up; do not let a
                 reader assume subsets were re-clustered.

Run from the project root. On the free-text benchmarks --judge-only is required
to reproduce the report's figures; see the JUDGE_ONLY comment below. The
multiple-choice file has no judge grades at all (letter extraction is
deterministic), so it takes --grader nli and no --judge-only.
    python scripts/panel_sweep.py --file outputs/truthfulqa_results_mc.jsonl \
        --format mc --grader nli --out outputs/panel_sweep9_tqa_mc
    python scripts/panel_sweep.py --file outputs/truthfulqa_results.jsonl \
        --format open --grader judge --judge-only --out outputs/panel_sweep9_tqa_open
    python scripts/panel_sweep.py --file outputs/simpleqa_3x3_results_1000.jsonl \
        --format open --grader judge --judge-only --out outputs/panel_sweep9_sqa
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.panels import family_of, tier_of                          # noqa: E402
from disagreement.answer_level import MCExactMatch                    # noqa: E402

MIN_CLASS = 10          # matches the report's AUROC suppression rule (Sec 3.5.1)

# Set from --judge-only in main(). When False, grade_of() substitutes the NLI
# grade wherever the judge returned no verdict, which produces a HYBRID label
# rather than a judge label. On SimpleQA that fallback fires on 21 responses and
# flips one question, giving base rate 0.859 against the canonical 0.858 and
# AUROC 0.892 against the canonical 0.889 at N=9. The report's figures are the
# strict-judge ones, so pass --judge-only to reproduce them.
JUDGE_ONLY = False


# ---------------------------------------------------------------------------
# Metrics — kept local so this script has no sklearn dependency
# ---------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Rank-based AUROC with average ranks for ties (ties count as half-wins)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n, n_pos = len(pairs), sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_pr(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Average precision, with tied scores collapsed into one threshold.

    Ties matter here more than they usually do: on a small panel the
    disagreement score is supported only on values realisable by integer
    partitions of N, so a two-agent panel takes exactly two values and a
    threshold sweep that split ties would silently invent resolution the
    measure does not have.
    """
    n_pos = sum(labels)
    if n_pos == 0 or n_pos == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ap, tp, fp, prev_tp = 0.0, 0, 0, 0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if labels[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        ap += precision * (tp - prev_tp) / n_pos
        prev_tp = tp
        i = j + 1
    return ap


# ---------------------------------------------------------------------------
# Row access
# ---------------------------------------------------------------------------

def grade_of(row: dict, agent: str, grader: str) -> Optional[bool]:
    """True = correct, False = incorrect, None = not attempted / ungraded.

    With --judge-only, 'judge' means the judge and nothing else: a response the
    judge did not resolve is dropped from the label rather than back-filled from
    the NLI grader. See the JUDGE_ONLY comment above for why that matters.
    """
    if grader == "judge":
        g = (row.get("grades_judge") or {}).get(agent)
        if g is not None or JUDGE_ONLY:
            return g
    return (row.get("grades") or {}).get(agent)


def count_fallbacks(rows, agents: Sequence[str]) -> int:
    """Responses where a judge label is missing but an NLI grade would be used."""
    return sum(1 for r in rows for a in agents
               if (r.get("grades_judge") or {}).get(a) is None
               and (r.get("grades") or {}).get(a) is not None)


def usable(row: dict, agent: str) -> bool:
    """Agent contributed a comparable answer to this question.

    Abstentions are model behaviour and truncations are measurement failure,
    but both mean 'no answer to compare', so both are removed before scoring —
    the same union scripts/disagreement_pilot.py's score_panel applies.
    """
    if row.get("abstained", {}).get(agent):
        return False
    if row.get("unusable", {}).get(agent):
        return False
    if row.get("excluded", {}).get(agent):
        return False
    return bool((row.get("responses") or {}).get(agent))


def fixed_label(row: dict, agents: Sequence[str], grader: str) -> Optional[int]:
    """Majority-hallucinated over the FULL pool. None when undecidable."""
    graded = [grade_of(row, a, grader) for a in agents]
    graded = [g for g in graded if g is not None]
    if not graded:
        return None
    wrong = sum(1 for g in graded if g is False)
    return 1 if wrong * 2 > len(graded) else 0


# ---------------------------------------------------------------------------
# Disagreement over an arbitrary subset
# ---------------------------------------------------------------------------

_MC = MCExactMatch()


def subset_score(row: dict, panel: Sequence[str], fmt: str) -> Optional[float]:
    members = [a for a in panel if usable(row, a)]
    if len(members) < 2:
        return None                      # undefined, not zero
    if fmt == "mc":
        texts = [row["responses"][a] for a in members]
        return _MC.score(texts).score

    # open: normalised entropy of the full-pool clustering restricted to the
    # subset. See the module docstring for why this is not re-clustering.
    cluster_of = ((row.get("panels", {}).get("core") or {}).get("cluster_of")
                  or row.get("cluster_of") or {})
    ids = [cluster_of.get(a) for a in members]
    if any(i is None for i in ids):
        return None
    counts = defaultdict(int)
    for i in ids:
        counts[i] += 1
    n = len(ids)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(n) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def model_id(name: str) -> str:
    """Strip the provider prefix: 'openrouter/meta-llama/x' -> 'meta-llama/x'.

    agents.panels keys everything on the model id, not on the agent name that
    disagreement_pilot.py writes into the results file, so family_of/tier_of
    silently return 'unknown' if the provider segment is left on.
    """
    parts = name.split("/")
    return "/".join(parts[1:]) if len(parts) > 2 else parts[-1]


def short(name: str) -> str:
    return name.split("/")[-1]


def evaluate_panel(rows, panel, fmt, labels) -> Optional[dict]:
    scores, labs = [], []
    for row, lab in zip(rows, labels):
        if lab is None:
            continue
        s = subset_score(row, panel, fmt)
        if s is None:
            continue
        scores.append(s)
        labs.append(lab)
    if len(scores) < MIN_CLASS or min(sum(labs), len(labs) - sum(labs)) < MIN_CLASS:
        return None
    base = sum(labs) / len(labs)
    ap = auc_pr(scores, labs)
    return {
        "n_agents": len(panel),
        "n_families": len({family_of(model_id(a)) for a in panel}),
        "panel": "+".join(short(a) for a in panel),
        "n": len(scores),
        "base_rate": round(base, 3),
        "auc_pr": round(ap, 3) if ap is not None else None,
        "lift": round(ap / base, 2) if ap is not None and base else None,
        "auroc": round(auroc(scores, labs), 3),
        "mean_disagreement": round(statistics.mean(scores), 3),
    }


def held_out(rows, agents, fmt, grader) -> List[dict]:
    """Score each agent's own errors from the disagreement of the other N-1.

    Score and label then share no responses at all, which removes the coupling
    the report describes in Section B.3: on a fixed-option item answered by N
    agents, maximum disagreement mechanically entails a positive majority
    label, so any lift measured under the coupled label is inflated by
    construction.
    """
    out = []
    for target in agents:
        panel = [a for a in agents if a != target]
        scores, labs = [], []
        for row in rows:
            g = grade_of(row, target, grader)
            if g is None:
                continue
            s = subset_score(row, panel, fmt)
            if s is None:
                continue
            scores.append(s)
            labs.append(0 if g else 1)
        if not scores or min(sum(labs), len(labs) - sum(labs)) < MIN_CLASS:
            continue
        base = sum(labs) / len(labs)
        ap = auc_pr(scores, labs)
        out.append({
            "held_out_agent": short(target),
            "panel_size": len(panel),
            "n": len(scores),
            "base_rate": round(base, 3),
            "auc_pr": round(ap, 3),
            "lift": round(ap / base, 2),
            "auroc": round(auroc(scores, labs), 3),
        })
    return sorted(out, key=lambda r: -r["lift"])


def mean_lift(rows_) -> Optional[float]:
    v = [r["lift"] for r in rows_ if r.get("lift") is not None]
    return round(statistics.mean(v), 2) if v else None


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--file", required=True, help="disagreement_pilot results jsonl")
    ap_.add_argument("--format", choices=["mc", "open"], required=True)
    ap_.add_argument("--grader", choices=["nli", "judge"], default="judge",
                     help="which stored grading to use for labels; 'judge' falls "
                          "back to the NLI grades where no judge grade exists")
    ap_.add_argument("--judge-only", action="store_true",
                     help="with --grader judge, do NOT fall back to NLI grades "
                          "where the judge gave no verdict. This is the label "
                          "behind the report's figures (SimpleQA base 0.858, "
                          "N=9 AUROC 0.889); without it the label is a hybrid.")
    ap_.add_argument("--out", required=True, help="output path stem (no extension)")
    ap_.add_argument("--max-panels-printed", type=int, default=20)
    args = ap_.parse_args()

    global JUDGE_ONLY
    JUDGE_ONLY = args.judge_only

    rows = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip()]
    agents = list(rows[0]["responses"].keys())
    print(f"{len(rows)} questions, {len(agents)} agents, format={args.format}, "
          f"grader={args.grader}"
          f"{' (judge only, no NLI fallback)' if JUDGE_ONLY else ''}")

    # Multiple-choice files are graded deterministically by letter extraction,
    # so they carry no grades_judge at all. There --grader judge has always been
    # the NLI/deterministic path wearing a judge's name, via the fallback;
    # --judge-only would leave nothing to label with.
    n_judge = sum(1 for r in rows for a in agents
                  if (r.get("grades_judge") or {}).get(a) is not None)
    if args.grader == "judge" and n_judge == 0:
        sys.exit("\n  This file has no grades_judge entries at all, so there is "
                 "no judge label to\n  build. Multiple-choice responses are "
                 "graded by deterministic letter extraction\n  and live in "
                 "'grades'. Re-run with --grader nli (and without --judge-only);\n"
                 "  that is the same computation --grader judge was performing "
                 "here via the\n  NLI fallback, so the results are unchanged.")

    for a in agents:
        print(f"  - {short(a):<28} {family_of(model_id(a)):<8} {tier_of(model_id(a))}")

    if args.grader == "judge" and not JUDGE_ONLY:
        n_fb = count_fallbacks(rows, agents)
        if n_fb:
            print(f"\n  WARNING: the judge gave no verdict on {n_fb} responses; "
                  f"their NLI grades are being\n           substituted, so the "
                  f"label is a judge/NLI hybrid and will not match the report.\n"
                  f"           Re-run with --judge-only to reproduce the "
                  f"published figures.")

    labels = [fixed_label(r, agents, args.grader) for r in rows]
    decided = [l for l in labels if l is not None]
    if not decided:
        sys.exit(f"\n  No question could be labelled: every row's grades are "
                 f"missing or unresolved\n  under --grader {args.grader}"
                 f"{' --judge-only' if JUDGE_ONLY else ''}. Check which grading "
                 f"field this file\n  actually carries before re-running.")
    print(f"\nFixed majority-hallucinated label: {sum(decided)} positives of "
          f"{len(decided)} decidable questions (base rate "
          f"{sum(decided)/len(decided):.3f}); {len(labels)-len(decided)} undecidable")

    # ---- A. full sweep -----------------------------------------------------
    panels = [c for k in range(2, len(agents) + 1) for c in combinations(agents, k)]
    print(f"\nSweeping {len(panels)} non-trivial panels...")
    sweep = []
    for i, panel in enumerate(panels, 1):
        r = evaluate_panel(rows, list(panel), args.format, labels)
        if r:
            sweep.append(r)
        if i % 100 == 0:
            print(f"  {i}/{len(panels)}")
    sweep.sort(key=lambda r: (r["n_agents"], -(r["lift"] or 0)))

    if not sweep:
        sys.exit(f"\n  No panel met the reporting threshold of {MIN_CLASS} "
                 f"instances in each class.\n  Nothing to write.")

    out_csv = Path(f"{args.out}.csv")
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    print(f"\nWrote {len(sweep)} panels -> {out_csv}")

    # ---- B. dose-response --------------------------------------------------
    print("\n" + "=" * 72)
    print("  B. Mean lift by panel size  (report Sec 4.6)")
    print("=" * 72)
    by_n = defaultdict(list)
    for r in sweep:
        by_n[r["n_agents"]].append(r)
    print(f"  {'N':>3} {'panels':>7} {'mean lift':>10} {'best lift':>10} "
          f"{'best panel':<44} {'mean D':>7}")
    for n in sorted(by_n):
        grp = by_n[n]
        best = max(grp, key=lambda r: r["lift"] or 0)
        print(f"  {n:>3} {len(grp):>7} {mean_lift(grp):>10} {best['lift']:>10} "
              f"{best['panel'][:44]:<44} "
              f"{round(statistics.mean(r['mean_disagreement'] for r in grp), 3):>7}")

    # ---- C. family diversity ----------------------------------------------
    print("\n" + "=" * 72)
    print("  C. Mean lift by number of families, within panel size "
          "(report Table B.5 / Sec 5.6.1)")
    print("=" * 72)
    print(f"  {'N':>3} {'1 family':>16} {'2 families':>16} {'3 families':>16}")
    for n in sorted(by_n):
        cells = []
        for f in (1, 2, 3):
            grp = [r for r in by_n[n] if r["n_families"] == f]
            cells.append(f"{mean_lift(grp)} ({len(grp)})" if grp else "—")
        print(f"  {n:>3} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

    # ---- D. weakest-agent partition ---------------------------------------
    err = {}
    for a in agents:
        g = [grade_of(r, a, args.grader) for r in rows]
        g = [x for x in g if x is not None]
        err[a] = 1 - sum(g) / len(g) if g else None
    weakest = max(agents, key=lambda a: err[a] or 0)
    print("\n" + "=" * 72)
    print(f"  D. Mean lift by presence of the weakest agent "
          f"({short(weakest)}, error {err[weakest]:.3f})  (report Table B.6 / Sec 5.5)")
    print("=" * 72)
    print(f"  {'N':>3} {'excluding':>16} {'including':>16} {'delta':>8}")
    for n in sorted(by_n):
        exc = [r for r in by_n[n] if short(weakest) not in r["panel"].split("+")]
        inc = [r for r in by_n[n] if short(weakest) in r["panel"].split("+")]
        me, mi = mean_lift(exc), mean_lift(inc)
        d = round(me - mi, 2) if me is not None and mi is not None else None
        dtxt = "—" if d is None else (f"+{d}" if d > 0 else str(d))
        etxt = "—" if me is None else f"{me} ({len(exc)})"
        itxt = "—" if mi is None else f"{mi} ({len(inc)})"
        print(f"  {n:>3} {etxt:>16} {itxt:>16} {dtxt:>8}")

    # ---- E. held-out -------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  E. Held-out validation: N-1 panel predicts the excluded agent "
          f"(report Table B.3 / B.7, Table 8.1)")
    print("=" * 72)
    ho = held_out(rows, agents, args.format, args.grader)
    print(f"  {'held-out agent':<28} {'n':>5} {'base':>7} {'AUC-PR':>8} "
          f"{'lift':>6} {'AUROC':>7}")
    for r in ho:
        print(f"  {r['held_out_agent']:<28} {r['n']:>5} {r['base_rate']:>7} "
              f"{r['auc_pr']:>8} {r['lift']:>6} {r['auroc']:>7}")
    if ho:
        print(f"  {'MEAN':<28} {'':>5} {'':>7} {'':>8} "
              f"{round(statistics.mean(r['lift'] for r in ho), 2):>6} "
              f"{round(statistics.mean(r['auroc'] for r in ho), 3):>7}")
    Path(f"{args.out}.heldout.csv").write_text(
        "\n".join([",".join(ho[0].keys())] +
                  [",".join(str(v) for v in r.values()) for r in ho]) + "\n")
    print(f"\nWrote held-out table -> {args.out}.heldout.csv")


if __name__ == "__main__":
    main()