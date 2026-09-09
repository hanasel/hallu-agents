"""Calibration, operating points, and cheap baselines.

Every detection number elsewhere in this project is a ranking measure — AUROC,
AUC-PR, lift. Those establish that a signal orders wrong answers above right
ones; none of them says where to put a threshold, what that threshold would
cost, or whether the detector beats the trivial alternative of flagging
everything. This script answers those questions from the three canonical
results files, without touching the pipeline (`disagreement/`, `agents/`,
`evaluation/`, the pilot) or the results files themselves.

Five things, per dataset (SimpleQA Verified, TruthfulQA open, TruthfulQA MC):

  1. Precision-recall curves for the primary signal vs. Jaccard, with the
     no-skill line at the base rate. AUC-PR and lift (AUC-PR / base rate).
  2. An operating point at fixed 80% recall (the rule, stated explicitly —
     "how much manual review does catching 80% of hallucinations cost?"),
     plus a second point at max F1.
  3. The trivial baseline: flag-everything precision (= prevalence) vs. the
     detector's precision at the 80%-recall point. On a saturated benchmark
     the detector may not beat it — that's a finding, not a bug.
  4. Threshold transferability: a threshold chosen on one dataset applied
     unchanged to the others.
  5. Cheap baselines — primary signal, Jaccard, mean response length,
     abstention count — AUROC/AUC-PR with bootstrap 95% CIs, so a 0.02
     difference is never reported as a finding when the intervals overlap.

Label provenance: the majority-wrong label is RECOMPUTED from `grades_judge`
(not read from the stored `panel_majority_wrong`, which is exact-match/NLI
grading — a different, noisier label source). On TruthfulQA MC there is no
judge (`grades_judge` is absent by construction — MC responses are graded by
letter match only), so the MC label falls back to `grades`. This choice
shifts the observed prevalence away from a same-file estimate computed off
the exact-match grades — see the script's own printed prevalence-provenance
note at startup, which exists precisely so that shift is never silent.

Run from the project root:
    python scripts/calibration.py
        # uses the three canonical files below; writes outputs/calibration/
    python scripts/calibration.py --simpleqa outputs/other.jsonl
        # override any input file (see --help for all three)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.disagreement_pilot import auroc_guarded, section  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = ["simpleqa", "truthfulqa_open", "truthfulqa_mc"]

DEFAULT_FILES = {
    "simpleqa": "outputs/simpleqa_3x3_results_1000_keys_v2.jsonl",
    "truthfulqa_open": "outputs/truthfulqa_results.jsonl",
    "truthfulqa_mc": "outputs/truthfulqa_results_mc.jsonl",
}

DATASET_TITLES = {
    "simpleqa": "SimpleQA Verified",
    "truthfulqa_open": "TruthfulQA (open)",
    "truthfulqa_mc": "TruthfulQA (MC)",
}

# The MC "core" panel's semantic_entropy field actually holds MC exact-match
# disagreement (see disagreement/answer_level.py:MCExactMatch) — same [0, 1]
# scale as normalised semantic entropy (both are mean pairwise disagreement
# under their respective equivalence rules), but a different measure. Labelled
# accordingly everywhere rather than called "semantic entropy" on this row.
PRIMARY_SIGNAL_LABEL = {
    "simpleqa": "semantic entropy",
    "truthfulqa_open": "semantic entropy",
    "truthfulqa_mc": "MC exact-match disagreement",
}

# grades_judge is the LLM-judge grade; on MC there is no judge (judge_model is
# null in that file's manifest — MC is graded by extracted-letter match only),
# so grades_judge is absent there by construction and we fall back to grades.
GRADE_KEY = {
    "simpleqa": "grades_judge",
    "truthfulqa_open": "grades_judge",
    "truthfulqa_mc": "grades",
}

RECALL_TARGET_DEFAULT = 0.80
N_BOOTSTRAP_DEFAULT = 2000
BOOTSTRAP_SEED_DEFAULT = 0
CI_LO, CI_HI = 2.5, 97.5


# ---------------------------------------------------------------------------
# IO + row extraction
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def queried_agents(row: dict) -> List[str]:
    """Every agent queried for this row — i.e. the core panel's full 9-name
    membership, abstained/truncated ones included.

    NOT `panels["core"]["cluster_of"].keys()`: score_panel() (disagreement_
    pilot.py) builds cluster_of only from `attempted` members, dropping any
    agent that was excluded (abstained OR truncated) before clustering even
    starts (see its docstring). Using cluster_of here would silently exclude
    every abstention from the abstention-count baseline — exactly the
    signal the spec's cheap-baselines section asks about — and make the
    mean-response-length baseline redundantly re-filter a set that already
    lacks the excluded agents. `responses`/`abstained`/`excluded` are keyed
    by the full queried set regardless of outcome, so read from there.
    """
    return list((row.get("responses") or {}).keys())


def majority_wrong(row: dict, grade_key: str) -> Optional[bool]:
    """Recomputed majority-wrong label — same rule as nli_sensitivity.py's
    `_majority_wrong`, applied here to `grade_key` (grades_judge, or grades
    as the MC fallback) rather than trusting the stored `panel_majority_wrong`
    field (which is always grades-based, regardless of grade_key)."""
    g = [v for v in (row.get(grade_key) or {}).values() if v is not None]
    if not g:
        return None
    return sum(v is False for v in g) > len(g) / 2


def mean_response_length(row: dict) -> Optional[float]:
    """Mean whitespace-token count of the core panel's responses, excluding
    agents marked `excluded` (truncated or abstained) — computed from the
    response TEXT, not the API's `completion_tokens` (which can include
    invisible reasoning tokens for some agents and isn't comparable across
    the panel)."""
    agents = queried_agents(row)
    excluded = row.get("excluded") or {}
    responses = row.get("responses") or {}
    lens = [len((responses.get(a) or "").split())
            for a in agents if not excluded.get(a, False)]
    return statistics.mean(lens) if lens else None


def abstention_count(row: dict) -> int:
    agents = queried_agents(row)
    abstained = row.get("abstained") or {}
    return sum(1 for a in agents if abstained.get(a, False))


class DatasetArrays:
    __slots__ = ("key", "signal", "jaccard", "resp_len", "abst_count", "label",
                 "n_total", "n_missing_score", "n_missing_label", "n_used")

    def __init__(self, key: str):
        self.key = key


def build_dataset(key: str, rows: List[dict]) -> DatasetArrays:
    grade_key = GRADE_KEY[key]
    sig, jac, rl, ac, lab = [], [], [], [], []
    n_missing_score = 0
    n_missing_label = 0
    for row in rows:
        core = (row.get("panels") or {}).get("core") or {}
        s, j = core.get("semantic_entropy"), core.get("jaccard")
        if s is None or j is None:
            n_missing_score += 1
            continue
        label = majority_wrong(row, grade_key)
        if label is None:
            n_missing_label += 1
            continue
        rlen = mean_response_length(row)
        sig.append(s)
        jac.append(j)
        rl.append(rlen if rlen is not None else 0.0)
        ac.append(abstention_count(row))
        lab.append(label)

    d = DatasetArrays(key)
    d.signal = np.asarray(sig, dtype=float)
    d.jaccard = np.asarray(jac, dtype=float)
    d.resp_len = np.asarray(rl, dtype=float)
    d.abst_count = np.asarray(ac, dtype=float)
    d.label = np.asarray(lab, dtype=bool)
    d.n_total = len(rows)
    d.n_missing_score = n_missing_score
    d.n_missing_label = n_missing_label
    d.n_used = len(sig)
    return d


# ---------------------------------------------------------------------------
# Curves, counts, operating points
# ---------------------------------------------------------------------------

def pr_curve(scores: np.ndarray, labels: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision, recall, thresholds


def aucpr_and_lift(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    aucpr = average_precision_score(labels, scores)
    prevalence = float(labels.mean())
    lift = aucpr / prevalence if prevalence > 0 else float("nan")
    return aucpr, lift


def counts_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred_pos = scores >= thr
    tp = int(np.sum(pred_pos & labels))
    fp = int(np.sum(pred_pos & ~labels))
    fn = int(np.sum(~pred_pos & labels))
    tn = int(np.sum(~pred_pos & ~labels))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not (np.isnan(precision) or np.isnan(recall)) else float("nan"))
    n = len(labels)
    return {
        "threshold": float(thr), "precision": precision, "recall": recall, "f1": f1,
        "frac_flagged": (tp + fp) / n, "fp_per_100": fp / n * 100.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def choose_operating_point(scores: np.ndarray, labels: np.ndarray,
                           target_recall: float) -> dict:
    """Highest-precision threshold that still achieves >= target_recall —
    i.e. the smallest overshoot above the recall target, by construction."""
    _, _, thresholds = pr_curve(scores, labels)
    candidates = [counts_at_threshold(scores, labels, t) for t in thresholds]
    qualifying = [c for c in candidates if c["recall"] >= target_recall]
    if not qualifying:
        return min(candidates, key=lambda c: c["threshold"])  # best achievable recall
    return max(qualifying, key=lambda c: c["threshold"])


def choose_max_f1_point(scores: np.ndarray, labels: np.ndarray) -> dict:
    _, _, thresholds = pr_curve(scores, labels)
    candidates = [counts_at_threshold(scores, labels, t) for t in thresholds]
    return max(candidates, key=lambda c: (c["f1"] if not np.isnan(c["f1"]) else -1.0))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_signal_metrics(signals: Dict[str, np.ndarray], labels: np.ndarray,
                             n_boot: int, seed: int) -> Dict[str, dict]:
    """Percentile-bootstrap AUROC/AUC-PR for every named signal, resampling
    ROW INDICES once per iteration and reusing them across all signals so the
    resulting intervals are directly comparable (a fair basis for the
    overlap check the spec asks for), not independently noisy per signal."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    names = list(signals.keys())
    boot_auroc = {name: [] for name in names}
    boot_aucpr = {name: [] for name in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        lab_b = labels[idx]
        if lab_b.sum() == 0 or lab_b.sum() == n:
            continue
        for name in names:
            s_b = signals[name][idx]
            boot_auroc[name].append(roc_auc_score(lab_b, s_b))
            boot_aucpr[name].append(average_precision_score(lab_b, s_b))

    out = {}
    for name in names:
        a = np.asarray(boot_auroc[name])
        p = np.asarray(boot_aucpr[name])
        out[name] = {
            "auroc": float(roc_auc_score(labels, signals[name])),
            "auroc_ci": (float(np.percentile(a, CI_LO)), float(np.percentile(a, CI_HI))),
            "aucpr": float(average_precision_score(labels, signals[name])),
            "aucpr_ci": (float(np.percentile(p, CI_LO)), float(np.percentile(p, CI_HI))),
            "n_boot_used": int(len(a)),
        }
    return out


def ci_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _f(v: Optional[float], fmt: str = ".3f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, fmt)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_prevalence(datasets: Dict[str, DatasetArrays]) -> str:
    lines = [f"{'dataset':<18} {'n_total':>8} {'n_used':>8} {'missing_score':>14} "
             f"{'missing_label':>14} {'prevalence':>11}"]
    for key in DATASETS:
        d = datasets[key]
        prevalence = float(d.label.mean())
        lines.append(f"{DATASET_TITLES[key]:<18} {d.n_total:>8} {d.n_used:>8} "
                     f"{d.n_missing_score:>14} {d.n_missing_label:>14} {prevalence:>11.3f}")
    return "\n".join(lines)


def make_pr_figure(key: str, d: DatasetArrays, out_path: Path) -> Tuple[float, float, float, float]:
    p_sig, r_sig, _ = pr_curve(d.signal, d.label)
    p_jac, r_jac, _ = pr_curve(d.jaccard, d.label)
    aucpr_sig, lift_sig = aucpr_and_lift(d.signal, d.label)
    aucpr_jac, lift_jac = aucpr_and_lift(d.jaccard, d.label)
    prevalence = float(d.label.mean())

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(r_sig, p_sig, label=f"{PRIMARY_SIGNAL_LABEL[key]} (AUC-PR={aucpr_sig:.2f})")
    ax.plot(r_jac, p_jac, label=f"Jaccard disagreement (AUC-PR={aucpr_jac:.2f})")
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1,
               label=f"no-skill (base rate={prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(DATASET_TITLES[key])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return aucpr_sig, lift_sig, aucpr_jac, lift_jac


def report_pr_aucpr(datasets: Dict[str, DatasetArrays], fig_dir: Path) -> Tuple[str, Dict[str, dict], List[Path]]:
    rows = []
    per_dataset = {}
    fig_paths = []
    for key in DATASETS:
        d = datasets[key]
        fig_path = fig_dir / f"pr_curve_{key}.pdf"
        aucpr_sig, lift_sig, aucpr_jac, lift_jac = make_pr_figure(key, d, fig_path)
        fig_paths.append(fig_path)
        per_dataset[key] = {"aucpr_signal": aucpr_sig, "lift_signal": lift_sig,
                            "aucpr_jaccard": aucpr_jac, "lift_jaccard": lift_jac,
                            "figure": str(fig_path)}
        rows.append(f"{DATASET_TITLES[key]:<18} "
                    f"{PRIMARY_SIGNAL_LABEL[key]:<28} AUC-PR={aucpr_sig:.3f} lift={lift_sig:.3f}   |   "
                    f"jaccard AUC-PR={aucpr_jac:.3f} lift={lift_jac:.3f}")
    return "\n".join(rows), per_dataset, fig_paths


def report_operating_points(datasets: Dict[str, DatasetArrays], target_recall: float) -> Tuple[str, Dict[str, dict]]:
    lines = [f"{'dataset':<18} {'rule':<12} {'thr':>7} {'precision':>10} {'recall':>7} "
             f"{'f1':>6} {'flagged%':>9} {'fp/100':>7}"]
    per_dataset = {}
    for key in DATASETS:
        d = datasets[key]
        rec_pt = choose_operating_point(d.signal, d.label, target_recall)
        f1_pt = choose_max_f1_point(d.signal, d.label)
        per_dataset[key] = {"recall_target": rec_pt, "max_f1": f1_pt}
        for rule, pt in (("recall>=80%", rec_pt), ("max F1", f1_pt)):
            lines.append(f"{DATASET_TITLES[key]:<18} {rule:<12} {pt['threshold']:>7.3f} "
                        f"{pt['precision']:>10.3f} {pt['recall']:>7.3f} {_f(pt['f1'],'.3f'):>6} "
                        f"{pt['frac_flagged']*100:>8.1f}% {pt['fp_per_100']:>7.2f}")
    return "\n".join(lines), per_dataset


def report_trivial_baseline(datasets: Dict[str, DatasetArrays],
                            operating: Dict[str, dict]) -> Tuple[str, Dict[str, dict]]:
    lines = [f"{'dataset':<18} {'prevalence':>11} {'detector_p@80%r':>16} {'diff':>8} {'beats_flag_everything':>22}"]
    per_dataset = {}
    for key in DATASETS:
        d = datasets[key]
        prevalence = float(d.label.mean())
        det_p = operating[key]["recall_target"]["precision"]
        diff = det_p - prevalence
        beats = diff > 0
        per_dataset[key] = {"prevalence": prevalence, "detector_precision": det_p,
                            "diff": diff, "beats_flag_everything": beats}
        lines.append(f"{DATASET_TITLES[key]:<18} {prevalence:>11.3f} {det_p:>16.3f} "
                    f"{diff:>+8.3f} {str(beats):>22}")
    return "\n".join(lines), per_dataset


def report_transferability(datasets: Dict[str, DatasetArrays],
                           operating: Dict[str, dict]) -> Tuple[str, List[dict]]:
    thr_sqa = operating["simpleqa"]["recall_target"]["threshold"]
    thr_tqa = operating["truthfulqa_open"]["recall_target"]["threshold"]
    combos = [
        ("simpleqa", thr_sqa, "truthfulqa_open"),
        ("truthfulqa_open", thr_tqa, "simpleqa"),
        ("simpleqa", thr_sqa, "truthfulqa_mc"),
        ("truthfulqa_open", thr_tqa, "truthfulqa_mc"),
    ]
    lines = [f"{'threshold from':<18} {'thr':>7} {'applied to':<18} {'precision':>10} {'recall':>7}"]
    results = []
    for src, thr, dst in combos:
        d = datasets[dst]
        pt = counts_at_threshold(d.signal, d.label, thr)
        results.append({"from": src, "threshold": thr, "to": dst,
                        "precision": pt["precision"], "recall": pt["recall"]})
        lines.append(f"{DATASET_TITLES[src]:<18} {thr:>7.3f} {DATASET_TITLES[dst]:<18} "
                    f"{pt['precision']:>10.3f} {pt['recall']:>7.3f}")
    return "\n".join(lines), results


def report_cheap_baselines(datasets: Dict[str, DatasetArrays], n_boot: int,
                           seed: int) -> Tuple[str, Dict[str, dict]]:
    signal_names = ["primary", "jaccard", "response_length", "abstention_count"]
    lines = []
    per_dataset = {}
    for key in DATASETS:
        d = datasets[key]
        signals = {
            "primary": d.signal,
            "jaccard": d.jaccard,
            "response_length": d.resp_len,
            "abstention_count": d.abst_count,
        }
        boot = bootstrap_signal_metrics(signals, d.label, n_boot, seed)
        per_dataset[key] = boot

        lines.append(f"\n  {DATASET_TITLES[key]} (n={d.n_used}, "
                     f"{n_boot}-resample bootstrap 95% CI):")
        lines.append(f"    {'signal':<20} {'AUROC':>7} {'95% CI':>17} "
                     f"{'AUC-PR':>7} {'95% CI':>17}")
        for name in signal_names:
            b = boot[name]
            disp = PRIMARY_SIGNAL_LABEL[key] if name == "primary" else {
                "jaccard": "jaccard disagreement",
                "response_length": "mean response length",
                "abstention_count": "abstention count",
            }[name]
            lines.append(f"    {disp:<20} {b['auroc']:>7.3f} "
                        f"[{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}]".ljust(46) +
                        f"{b['aucpr']:>7.3f} [{b['aucpr_ci'][0]:.3f}, {b['aucpr_ci'][1]:.3f}]")
        primary = boot["primary"]
        for name in ("jaccard", "response_length", "abstention_count"):
            b = boot[name]
            overlap_auroc = ci_overlap(primary["auroc_ci"], b["auroc_ci"])
            overlap_aucpr = ci_overlap(primary["aucpr_ci"], b["aucpr_ci"])
            verdict_auroc = "indistinguishable from primary" if overlap_auroc else "distinguishable from primary"
            verdict_aucpr = "indistinguishable from primary" if overlap_aucpr else "distinguishable from primary"
            lines.append(f"      -> {name}: AUROC {verdict_auroc}; AUC-PR {verdict_aucpr}")
    return "\n".join(lines), per_dataset


def report_verification(datasets: Dict[str, DatasetArrays], operating: Dict[str, dict],
                        trivial: Dict[str, dict], recall_target: float) -> str:
    lines = []
    ok_all = True
    for key in DATASETS:
        d = datasets[key]
        prevalence = float(d.label.mean())
        flag_everything_p = trivial[key]["prevalence"]
        v1 = abs(flag_everything_p - prevalence) < 1e-12
        rec_pt = operating[key]["recall_target"]
        v2 = rec_pt["recall"] >= recall_target - 1e-9
        auroc_sklearn = float(roc_auc_score(d.label, d.signal))
        auroc_custom, why = auroc_guarded(list(d.signal), list(d.label))
        v3 = auroc_custom is None or abs(auroc_sklearn - auroc_custom) < 1e-6
        ok_all = ok_all and v1 and v2 and v3
        lines.append(f"  {DATASET_TITLES[key]}:")
        lines.append(f"    [1] flag-everything precision == prevalence: "
                     f"{flag_everything_p:.6f} == {prevalence:.6f}  -> {'OK' if v1 else 'FAIL'}")
        lines.append(f"    [2] {recall_target:.0%}-recall operating point achieves recall="
                     f"{rec_pt['recall']:.3f} >= {recall_target:.2f} -> {'OK' if v2 else 'FAIL'}")
        lines.append(f"    [3] sklearn AUROC ({auroc_sklearn:.4f}) vs. project auroc_guarded "
                     f"({_f(auroc_custom, '.4f')}) -> {'OK' if v3 else 'FAIL'}"
                     + ("" if auroc_custom is not None else f"  [{why}]"))
    lines.append(f"\n  all checks: {'PASS' if ok_all else 'FAIL — see above'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# .tex fragments
# ---------------------------------------------------------------------------

def render_tex(pr: Dict[str, dict], operating: Dict[str, dict], trivial: Dict[str, dict],
              transfer: List[dict], cheap: Dict[str, dict]) -> str:
    parts = ["% Auto-generated by scripts/calibration.py — do not hand-edit.\n"]

    parts.append("% --- Table: PR / AUC-PR / lift ---")
    parts.append(r"\begin{tabular}{lrrrr}")
    parts.append(r"\toprule")
    parts.append(r"dataset & AUC-PR (signal) & lift (signal) & AUC-PR (Jaccard) & lift (Jaccard) \\")
    parts.append(r"\midrule")
    for key in DATASETS:
        p = pr[key]
        parts.append(f"{_tex_escape(DATASET_TITLES[key])} & {p['aucpr_signal']:.3f} & "
                    f"{p['lift_signal']:.3f} & {p['aucpr_jaccard']:.3f} & {p['lift_jaccard']:.3f} \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append("")

    parts.append("% --- Table: operating points ---")
    parts.append(r"\begin{tabular}{llrrrrrr}")
    parts.append(r"\toprule")
    parts.append(r"dataset & rule & threshold & precision & recall & F1 & flagged (\%) & FP/100 \\")
    parts.append(r"\midrule")
    for key in DATASETS:
        for rule, pt in (("recall $\\geq$ 80\\%", operating[key]["recall_target"]),
                         ("max F1", operating[key]["max_f1"])):
            parts.append(f"{_tex_escape(DATASET_TITLES[key])} & {rule} & {pt['threshold']:.3f} & "
                        f"{pt['precision']:.3f} & {pt['recall']:.3f} & {_f(pt['f1'])} & "
                        f"{pt['frac_flagged']*100:.1f} & {pt['fp_per_100']:.2f} \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append("")

    parts.append("% --- Table: trivial baseline (flag-everything) ---")
    parts.append(r"\begin{tabular}{lrrrl}")
    parts.append(r"\toprule")
    parts.append(r"dataset & prevalence & detector $p$@80\%r & diff & beats flag-everything \\")
    parts.append(r"\midrule")
    for key in DATASETS:
        t = trivial[key]
        parts.append(f"{_tex_escape(DATASET_TITLES[key])} & {t['prevalence']:.3f} & "
                    f"{t['detector_precision']:.3f} & {t['diff']:+.3f} & "
                    f"{'yes' if t['beats_flag_everything'] else 'no'} \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append("")

    parts.append("% --- Table: threshold transferability ---")
    parts.append(r"\begin{tabular}{lrlrr}")
    parts.append(r"\toprule")
    parts.append(r"threshold from & threshold & applied to & precision & recall \\")
    parts.append(r"\midrule")
    for r in transfer:
        parts.append(f"{_tex_escape(DATASET_TITLES[r['from']])} & {r['threshold']:.3f} & "
                    f"{_tex_escape(DATASET_TITLES[r['to']])} & {r['precision']:.3f} & {r['recall']:.3f} \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append("")

    parts.append("% --- Table: cheap baselines (bootstrap 95% CI) ---")
    parts.append(r"\begin{tabular}{llrrrr}")
    parts.append(r"\toprule")
    parts.append(r"dataset & signal & AUROC & AUROC 95\% CI & AUC-PR & AUC-PR 95\% CI \\")
    parts.append(r"\midrule")
    disp_names = {"primary": None, "jaccard": "jaccard disagreement",
                 "response_length": "mean response length", "abstention_count": "abstention count"}
    for key in DATASETS:
        boot = cheap[key]
        for name in ("primary", "jaccard", "response_length", "abstention_count"):
            disp = PRIMARY_SIGNAL_LABEL[key] if name == "primary" else disp_names[name]
            b = boot[name]
            parts.append(f"{_tex_escape(DATASET_TITLES[key])} & {_tex_escape(disp)} & "
                        f"{b['auroc']:.3f} & [{b['auroc_ci'][0]:.3f}, {b['auroc_ci'][1]:.3f}] & "
                        f"{b['aucpr']:.3f} & [{b['aucpr_ci'][0]:.3f}, {b['aucpr_ci'][1]:.3f}] \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibration, operating points, and cheap baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--simpleqa", default=DEFAULT_FILES["simpleqa"])
    ap.add_argument("--truthfulqa-open", default=DEFAULT_FILES["truthfulqa_open"])
    ap.add_argument("--truthfulqa-mc", default=DEFAULT_FILES["truthfulqa_mc"])
    ap.add_argument("--out-dir", default="outputs/calibration")
    ap.add_argument("--recall-target", type=float, default=RECALL_TARGET_DEFAULT)
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP_DEFAULT)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED_DEFAULT)
    args = ap.parse_args()

    paths = {
        "simpleqa": _resolve_path(args.simpleqa),
        "truthfulqa_open": _resolve_path(args.truthfulqa_open),
        "truthfulqa_mc": _resolve_path(args.truthfulqa_mc),
    }
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    section("Loading + label recomputation")
    datasets: Dict[str, DatasetArrays] = {}
    for key in DATASETS:
        path = paths[key]
        if not path.exists():
            print(f"  [!] {DATASET_TITLES[key]}: file not found: {path}")
            sys.exit(1)
        rows = load_rows(path)
        d = build_dataset(key, rows)
        datasets[key] = d
        print(f"  {DATASET_TITLES[key]:<18} {path.name}")
        print(f"    n_rows={d.n_total} n_used={d.n_used} "
              f"missing_score={d.n_missing_score} missing_label={d.n_missing_label} "
              f"(label from {GRADE_KEY[key]!r}) prevalence={float(d.label.mean()):.3f}")

    section("1. Precision-recall curves, AUC-PR, lift")
    pr_text, pr_data, fig_paths = report_pr_aucpr(datasets, out_dir)
    print(pr_text)
    for fp in fig_paths:
        print(f"  figure: {fp}")

    section(f"2. Operating points (fixed recall >= {args.recall_target:.0%}, and max F1)")
    op_text, op_data = report_operating_points(datasets, args.recall_target)
    print(op_text)

    section("3. Trivial baseline — flag everything")
    triv_text, triv_data = report_trivial_baseline(datasets, op_data)
    print(triv_text)

    section("4. Threshold transferability")
    transfer_text, transfer_data = report_transferability(datasets, op_data)
    print(transfer_text)

    section(f"5. Cheap baselines ({args.n_bootstrap}-resample bootstrap)")
    cheap_text, cheap_data = report_cheap_baselines(datasets, args.n_bootstrap, args.bootstrap_seed)
    print(cheap_text)

    section("Verification")
    verify_text = report_verification(datasets, op_data, triv_data, args.recall_target)
    print(verify_text)

    summary_txt_path = out_dir / "calibration_summary.txt"
    summary_tex_path = out_dir / "calibration_summary.tex"
    summary_json_path = out_dir / "calibration_summary.json"

    summary_txt_path.write_text(
        "\n\n".join([
            "== Loading ==",
            report_prevalence(datasets),
            "== 1. PR / AUC-PR / lift ==", pr_text,
            "== 2. Operating points ==", op_text,
            "== 3. Trivial baseline ==", triv_text,
            "== 4. Threshold transferability ==", transfer_text,
            "== 5. Cheap baselines ==", cheap_text,
            "== Verification ==", verify_text,
        ]) + "\n", encoding="utf-8")
    summary_tex_path.write_text(
        render_tex(pr_data, op_data, triv_data, transfer_data, cheap_data), encoding="utf-8")
    summary_json_path.write_text(json.dumps({
        "prevalence": {k: float(datasets[k].label.mean()) for k in DATASETS},
        "n_used": {k: datasets[k].n_used for k in DATASETS},
        "pr": pr_data, "operating_points": op_data, "trivial_baseline": triv_data,
        "transferability": transfer_data, "cheap_baselines": cheap_data,
    }, indent=2, ensure_ascii=False, default=lambda o: o if not isinstance(o, np.generic) else o.item()),
        encoding="utf-8")

    section("Done")
    print(f"  Summary table : {summary_txt_path}")
    print(f"  LaTeX fragment: {summary_tex_path}")
    print(f"  Structured    : {summary_json_path}")
    print(f"  Figures       : {out_dir}/pr_curve_{{dataset}}.pdf")


if __name__ == "__main__":
    main()
