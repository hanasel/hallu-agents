"""Cross-task analysis of the RAGTruth verification runs (data2txt / summarization / qa).

Reads the three per-task result files produced by scripts/rgt_verify_pilot.py and
answers the chapter's two questions in one place:

  1. Does response-conditioned ensemble verification detect RAG hallucination
     consistently across tasks with very different base rates?
  2. RQ3 — does inter-judge DISAGREEMENT add anything over the ensemble verdict?

DEDUPLICATION (important)
-------------------------
The result files can contain duplicate rows: if two copies of the runner script
ran concurrently (e.g. a Ctrl-C killed `tail` but not the background job, and the
script was relaunched), each process loaded its own snapshot and appended, so the
same uid appears more than once. Naive reading double-counts those responses and
silently narrows every confidence interval.

This script deduplicates by uid and reports exactly what it dropped. Where
duplicates disagree (independent re-runs can differ slightly — decomposition is
not perfectly deterministic across runs), it keeps the LAST occurrence by default
and reports how many duplicate pairs actually disagreed on the signal, so you can
see whether the runs were reproducible.

STATISTICS
----------
Per-signal AUC-ROC with a percentile bootstrap 95% CI, plus AUC-PR always shown
next to its base rate (AUC-PR is base-rate dependent — a lower AUC-PR on a
low-prevalence task does NOT mean worse detection; compare the LIFT over base
rate, and use AUC-ROC for cross-task comparison).

The decisive RQ3 test is a PAIRED bootstrap of the AUC difference
(ensemble - disagreement) on the same resampled responses. An unpaired
comparison of two overlapping CIs is not a test; the paired delta is. If the
delta CI excludes 0, the ensemble is significantly better on that task.

It also fits a tiny 2-feature logistic model (ensemble + best disagreement
signal) and compares its cross-validated AUC against ensemble alone, which is
the direct answer to "does disagreement add incremental signal?".

Usage:
    python scripts/rgt_pool_analysis.py
    python scripts/rgt_pool_analysis.py --dir outputs --n-boot 5000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


TASKS = ["data2txt", "summarization", "qa"]

SIGNALS = [
    "ensemble_unsupported",
    "n_contested_claims",
    "mean_disagreement",
    "frac_high_disagreement",
    "max_disagreement",
]

DISAGREEMENT_SIGNALS = [s for s in SIGNALS if s != "ensemble_unsupported"]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Loading + deduplication
# ---------------------------------------------------------------------------

def load_task(path: Path, keep: str = "last") -> tuple[List[dict], dict]:
    """Load one task's JSONL, deduplicating by uid.

    Returns (rows, report). `report` records how many raw lines were read, how
    many duplicates were dropped, and how many duplicate pairs actually
    DISAGREED on ensemble_unsupported (a measure of run-to-run reproducibility).
    """
    raw: List[dict] = []
    bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1

    by_uid: "OrderedDict[str, dict]" = OrderedDict()
    seen_values: Dict[str, List[float]] = {}
    for r in raw:
        uid = r["uid"]
        seen_values.setdefault(uid, []).append(r.get("ensemble_unsupported", float("nan")))
        if uid in by_uid and keep == "first":
            continue
        by_uid[uid] = r

    n_dupe_uids = sum(1 for v in seen_values.values() if len(v) > 1)
    n_disagreeing = sum(
        1 for v in seen_values.values()
        if len(v) > 1 and not np.allclose(v, v[0], atol=1e-9, equal_nan=True)
    )

    report = {
        "raw_lines": len(raw),
        "unique": len(by_uid),
        "dropped": len(raw) - len(by_uid),
        "bad_lines": bad,
        "dupe_uids": n_dupe_uids,
        "dupe_uids_disagreeing": n_disagreeing,
    }
    return list(by_uid.values()), report


def scored_only(rows: List[dict]) -> List[dict]:
    """Responses that produced at least one claim (0-claim rows carry no signal)."""
    return [r for r in rows if r.get("n_claims", 0) > 0]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def auc_ci(y: np.ndarray, x: np.ndarray, n_boot: int, seed: int = 0):
    auc = roc_auc_score(y, x)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        boots.append(roc_auc_score(y[idx], x[idx]))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


def paired_auc_delta(y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray,
                     n_boot: int, seed: int = 0):
    """Bootstrap the AUC difference (a - b) on the SAME resampled responses.

    Pairing matters: two signals measured on the same responses are correlated,
    so comparing their independent CIs is not a test of whether they differ.
    Returns (delta, lo, hi, p_two_sided_approx).
    """
    delta = roc_auc_score(y, x_a) - roc_auc_score(y, x_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas.append(roc_auc_score(yb, x_a[idx]) - roc_auc_score(yb, x_b[idx]))
    if not deltas:
        return delta, float("nan"), float("nan"), float("nan")
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # Two-sided bootstrap p: fraction of resamples on the other side of 0, x2.
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return delta, lo, hi, min(p, 1.0)


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 5) -> float:
    """Cross-validated AUC of a logistic model on the given features.

    Used to ask whether ensemble+disagreement beats ensemble alone OUT OF SAMPLE.
    In-sample fits would always favour the larger model, so CV is required.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    n_splits = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=2000)
        model.fit(sc.transform(X[tr]), y[tr])
        preds[te] = model.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, preds)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def signal_table(y: np.ndarray, rows: List[dict], n_boot: int) -> None:
    print(f"\n  {'signal':<24}{'AUC-ROC':>9}{'95% CI':>18}{'AUC-PR':>9}"
          f"{'lift':>7}{'Spearman':>11}")
    base = y.mean()
    for name in SIGNALS:
        x = np.array([r[name] for r in rows], dtype=float)
        auc, lo, hi = auc_ci(y, x, n_boot)
        apr = average_precision_score(y, x)
        rho = spearmanr(x, y)[0]
        print(f"  {name:<24}{auc:>9.3f}   [{lo:.3f}, {hi:.3f}]{apr:>9.3f}"
              f"{apr/base:>6.1f}x{rho:>+11.3f}")
    print(f"\n  base rate (AUC-PR reference): {base:.3f}   "
          f"n={len(y)}, positives={int(y.sum())}")
    print("  'lift' = AUC-PR / base rate. Compare lift across tasks, not raw")
    print("  AUC-PR — AUC-PR falls with prevalence even when detection is equal.")


def rq3_block(y: np.ndarray, rows: List[dict], n_boot: int, seed: int = 0) -> None:
    """The decisive RQ3 comparison for one dataset (task or pooled)."""
    ens = np.array([r["ensemble_unsupported"] for r in rows], dtype=float)

    # Best disagreement signal by AUC, chosen on this data.
    best_name, best_auc = None, -1.0
    for name in DISAGREEMENT_SIGNALS:
        x = np.array([r[name] for r in rows], dtype=float)
        a = roc_auc_score(y, x)
        if a > best_auc:
            best_name, best_auc = name, a
    best = np.array([r[best_name] for r in rows], dtype=float)

    print(f"\n  RQ3 — ensemble vs best disagreement signal ({best_name}):")
    d, lo, hi, p = paired_auc_delta(y, ens, best, n_boot, seed)
    verdict = ("ensemble significantly better" if lo > 0 else
               "disagreement significantly better" if hi < 0 else
               "NOT significantly different")
    print(f"    paired ΔAUC (ensemble - {best_name}) = {d:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  p≈{p:.3f}")
    print(f"    -> {verdict}")

    rho, prho = spearmanr(ens, best)
    print(f"    Spearman(ensemble, {best_name}) = {rho:+.3f} (p={prho:.2g}) "
          f"— redundancy")

    # Does disagreement add anything OUT OF SAMPLE on top of the ensemble?
    auc_ens_cv = cv_auc(ens.reshape(-1, 1), y, seed)
    auc_both_cv = cv_auc(np.column_stack([ens, best]), y, seed)
    if not np.isnan(auc_both_cv):
        print(f"    5-fold CV AUC: ensemble alone = {auc_ens_cv:.3f}, "
              f"ensemble + {best_name} = {auc_both_cv:.3f} "
              f"({auc_both_cv - auc_ens_cv:+.3f})")
        print("    (a positive gain means disagreement carries INCREMENTAL signal)")


def confound_block(y: np.ndarray, rows: List[dict]) -> None:
    """Check the obvious confounds: does claim count leak the label or drive the signal?"""
    nc = np.array([r["n_claims"] for r in rows], dtype=float)
    ens = np.array([r["ensemble_unsupported"] for r in rows], dtype=float)
    ncont = np.array([r["n_contested_claims"] for r in rows], dtype=float)
    print("\n  Confound checks:")
    print(f"    Spearman(n_claims, label)                = {spearmanr(nc, y)[0]:+.3f}"
          "   (near 0 => claim count does not leak the label)")
    print(f"    Spearman(n_claims, ensemble_unsupported) = {spearmanr(nc, ens)[0]:+.3f}"
          "   (near 0 => signal is not a length artefact)")
    print(f"    Spearman(n_claims, n_contested_claims)   = {spearmanr(nc, ncont)[0]:+.3f}"
          "   (HIGH is expected — a raw count scales with claim count;")
    print("      this is why normalised frac_high_disagreement is the fairer")
    print("      disagreement signal to headline)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-task RAGTruth verification analysis.")
    ap.add_argument("--dir", default="outputs", help="directory holding the result files")
    ap.add_argument("--prefix", default="rgt_verify_together_",
                    help="filename prefix; files are <prefix><task>.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", choices=["first", "last"], default="last",
                    help="which copy of a duplicated uid to keep")
    ap.add_argument("--write-clean", action="store_true",
                    help="write deduplicated copies to <file>.dedup.jsonl")
    args = ap.parse_args()

    base_dir = Path(args.dir)

    # ------------------------------------------------------------------ #
    section("Loading + deduplication")
    per_task: Dict[str, List[dict]] = {}
    for task in TASKS:
        path = base_dir / f"{args.prefix}{task}.jsonl"
        if not path.exists():
            print(f"  [missing] {path} — skipping {task}")
            continue
        rows, rep = load_task(path, keep=args.keep)
        per_task[task] = rows
        flag = ""
        if rep["dropped"]:
            flag = ("  <-- DUPLICATES REMOVED (concurrent runner instances "
                    "appended to the same file)")
        print(f"  {task:<15} raw={rep['raw_lines']:>4}  unique={rep['unique']:>4}  "
              f"dropped={rep['dropped']:>4}{flag}")
        if rep["bad_lines"]:
            print(f"    [!] {rep['bad_lines']} unparseable line(s) skipped.")
        if rep["dupe_uids"]:
            print(f"    duplicated uids: {rep['dupe_uids']}, of which the two runs "
                  f"DISAGREED on ensemble_unsupported: {rep['dupe_uids_disagreeing']}")
            if rep["dupe_uids_disagreeing"]:
                frac = rep["dupe_uids_disagreeing"] / rep["dupe_uids"]
                print(f"    -> {frac:.0%} of repeated responses gave a different score on "
                      f"re-run.")
                print("       Decomposition/judging is not bit-identical across runs; "
                      "this is a")
                print("       reproducibility figure worth reporting, not a bug per se.")
        if args.write_clean:
            out = path.with_suffix(".dedup.jsonl")
            with out.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"    wrote {out}")

    if not per_task:
        print("\n  No result files found. Check --dir / --prefix.\n")
        return

    # ------------------------------------------------------------------ #
    section("Per-task results")
    summary_rows = []
    for task, rows in per_task.items():
        rows_s = scored_only(rows)
        excluded = len(rows) - len(rows_s)
        y = np.array([1.0 if r["is_hallucinated"] else 0.0 for r in rows_s])
        print(f"\n  ---- {task.upper()} ----")
        if excluded:
            print(f"  ({excluded} responses with 0 claims excluded)")
        if len(y) < 10 or y.min() == y.max():
            print("  too few / single-class; skipping.")
            continue
        signal_table(y, rows_s, args.n_boot)
        rq3_block(y, rows_s, args.n_boot, args.seed)
        confound_block(y, rows_s)

        ens = np.array([r["ensemble_unsupported"] for r in rows_s], dtype=float)
        auc, lo, hi = auc_ci(y, ens, args.n_boot)
        summary_rows.append((task, len(y), y.mean(), auc, lo, hi))

    # ------------------------------------------------------------------ #
    section("Headline table — ensemble verification across tasks")
    print(f"\n  {'task':<16}{'n':>5}{'base rate':>12}{'ensemble AUC':>15}{'95% CI':>20}")
    for task, n, base, auc, lo, hi in summary_rows:
        print(f"  {task:<16}{n:>5}{base:>11.1%}{auc:>15.3f}   [{lo:.3f}, {hi:.3f}]")
    if len(summary_rows) > 1:
        aucs = [r[3] for r in summary_rows]
        print(f"\n  spread across tasks: {min(aucs):.3f} – {max(aucs):.3f}")
        print("  Stable AUC across a wide base-rate range is the robustness claim:")
        print("  detection quality is not an artefact of how often hallucination occurs.")

    # ------------------------------------------------------------------ #
    section("Pooled across all tasks")
    pooled = []
    for task, rows in per_task.items():
        for r in scored_only(rows):
            rr = dict(r)
            rr["task"] = task
            pooled.append(rr)
    y = np.array([1.0 if r["is_hallucinated"] else 0.0 for r in pooled])
    print(f"  pooled responses: {len(y)}   hallucinated: {int(y.sum())} ({y.mean():.1%})")
    print("  Pooling mixes tasks with different base rates, so treat the pooled")
    print("  AUC as an aggregate rather than a clean estimate — its value is the")
    print("  larger positive count, which tightens the RQ3 comparison below.")
    if len(y) >= 10 and y.min() != y.max():
        signal_table(y, pooled, args.n_boot)
        rq3_block(y, pooled, args.n_boot, args.seed)
        confound_block(y, pooled)

    section("Done")
    print("  Reporting notes:")
    print("   - Quote AUC-ROC for cross-task comparison; pair every AUC-PR with its")
    print("     base rate (or quote the lift).")
    print("   - The paired ΔAUC, not the overlap of two CIs, is the RQ3 test.")
    print("   - Panel: 3 judges / 2 families (Llama-3.3-70B, GPT-OSS-20B, GPT-OSS-120B),")
    print("     Llama-3.3-70B decomposer, per-claim verification, cap 25 claims.\n")


if __name__ == "__main__":
    main()