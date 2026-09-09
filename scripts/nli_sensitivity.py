"""NLI backend and clustering-rule sensitivity sweep.

Every open-ended result in this project inherits the behaviour of one NLI
checkpoint (`cross-encoder/nli-deberta-v3-base`) under one clustering rule
(relaxed entailment, complete linkage). This script varies that one axis at a
time — five configurations per dataset (see RUN_SPECS) — and reports whether
four downstream conclusions (semantic entropy vs Jaccard, detectability vs
capability, family-diversity null result, tier-panel monotonicity) survive
the swap, or are artifacts of `canonical`.

Design
------
NOT a re-implementation of the pipeline. Each configuration is run by
shelling out to `scripts/disagreement_pilot.py` itself (see `run_one_config`)
with a distinct `--nli-model`/`--strict`/`--single-linkage` combination and
its own `--out` path — the flags already exist there, and re-deriving
build_pool/run_queries/score_panel here would be a second, driftable copy of
the same logic. Agent responses are cached (`agents/cache.py`, keyed on
model+prompt+temperature+max_tokens, independent of the NLI config) so this
costs NLI inference time, not API calls.

Judge grades are NOT re-queried per configuration. `grades_judge` is a
property of the (question, agent, response-text) triple, not of which NLI
model clustered it — since every configuration re-uses the same cached
responses, an existing judge-graded file can be backfilled onto every run's
`grades_judge` field for uids with byte-identical responses (see
`backfill_judge_grades`, the same pattern `eval_short_answer_branch.py` uses
and for the same reason: querying the judge nine agents x five configs times
over would spend real API calls to reproduce a verdict that already exists).

Metric extraction re-uses the pure, deterministic helpers `disagreement_pilot`
already defines (`auroc_guarded`, `spearman`, `common_rows`, `short`) rather
than re-deriving AUROC/rank-correlation math here — but does NOT re-derive
panel_specs. Panel keys ("within:...", "cross:...", "tier:small", "loo:...")
are a stable naming convention baked into every row's `panels` dict by
`build_panel_specs`/`score_panel`, so filtering `row["panels"]` by key prefix
recovers exactly the same panel groupings without reconstructing the agent
pool.

Run from the project root:
    python scripts/nli_sensitivity.py --dry-run
        # print the planned commands and run the label-order/availability
        # preflight only; nothing is executed or written.
    python scripts/nli_sensitivity.py --n 1000 \\
        --judge-source-simpleqa outputs/<settled-canonical-simpleqa>.jsonl \\
        --judge-source-truthfulqa outputs/<settled-canonical-truthfulqa>.jsonl \\
        --canonical-reference-simpleqa outputs/<settled-canonical-simpleqa>.jsonl \\
        --canonical-reference-truthfulqa outputs/<settled-canonical-truthfulqa>.jsonl
        # the full sweep. --judge-source-* backfills grades_judge (needed for
        # the judge-graded AUROC/Spearman figures); --canonical-reference-*
        # is compared byte-for-byte against this run's `canonical` cell
        # (verification step 1 — the sweep is not measuring what it claims
        # to if this does not match).
    python scripts/nli_sensitivity.py --n 300 --datasets simpleqa
        # the spec's fallback if the full sweep proves too slow: reduce N,
        # keep every configuration, rather than dropping configurations.

Do not run this until the short-answer-keys branch and whatever canonical
results file it lands on have settled — see this script's own docstring
note repeated in --help: the `canonical` cell must be the configuration
actually adopted, or the table compares against something that is no longer
the default.
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.disagreement_pilot import (                              # noqa: E402
    auroc_guarded, spearman, common_rows, short, section,
)
from disagreement import CrossEncoderNLI                              # noqa: E402


# ---------------------------------------------------------------------------
# Sweep definition — one axis at a time, per the spec's table.
# ---------------------------------------------------------------------------

# Execution order puts nli-large first (within each dataset): it is the
# configuration most likely to be impractically slow, and finding that out
# after the four cheap runs have already finished wastes the cheap runs'
# time budget on nothing. The FINAL TABLE still prints in TABLE_ORDER
# (canonical first) — RUN_ORDER only controls what runs first.
RUN_ORDER = ["nli-large", "canonical", "nli-alt", "strict", "single"]
TABLE_ORDER = ["canonical", "nli-large", "nli-alt", "strict", "single"]

RUN_SPECS = {
    "canonical": {"model_key": "base", "flags": []},
    "nli-large": {"model_key": "large", "flags": []},
    "nli-alt":   {"model_key": "alt", "flags": []},
    "strict":    {"model_key": "base", "flags": ["--strict"]},
    "single":    {"model_key": "base", "flags": ["--single-linkage"]},
}

DATASETS = ["simpleqa", "truthfulqa"]

# Baseline flags applied to EVERY configuration of a given dataset — not part
# of the axis being swept (RUN_SPECS above), but part of what "canonical" now
# means since the short-answer-keys branch was adopted. Without this, the
# `canonical` cell would run the pre-branch pipeline and fail verification
# step 1 (it would not reproduce the adopted canonical results file), and
# every other SimpleQA cell would be testing the robustness of a superseded
# configuration instead of the one actually in use.
#
# TruthfulQA gets nothing here: TruthfulQASample has no `answer_type` field
# (see DATASETS["truthfulqa"]["meta"] in disagreement_pilot.py), so
# make_short_answer_key_fn's date/number extraction never fires there — only
# the unconditional chemical-formula extractor could, and TruthfulQA answers
# are essentially never bare formulae. The flag is inert on this dataset, so
# it is left off rather than passed for no effect.
DEFAULT_DATASET_EXTRA_ARGS = {
    "simpleqa": "--short-answer-keys",
    "truthfulqa": "",
}

# Bucketing tolerances for the direction-holds check — see _bucket()'s
# docstring for why a raw sign comparison is the wrong tool near zero.
TOL_AUROC_GAP = 0.02
TOL_RANK_BISERIAL = 0.10
TOL_SPEARMAN = 0.05
TOL_TIER_MONOTONIC = 0.01


# ---------------------------------------------------------------------------
# Preflight: checkpoint availability + label-order sanity check
# ---------------------------------------------------------------------------

# Three pairs with an unambiguous gold label, used to verify a checkpoint's
# id2label resolves the way CrossEncoderNLI assumes BEFORE spending hours
# clustering with it. A silently mislabelled entailment/contradiction axis
# inverts every cluster and produces plausible-looking nonsense — see
# CrossEncoderNLI's own docstring on why label order is read from the
# checkpoint's config rather than hardcoded.
SANITY_PAIRS = [
    ("A man is playing a guitar on a busy street corner.",
     "A man is performing music in public.", "entailment"),
    ("The museum is closed on Mondays and reopens Tuesday morning.",
     "The museum is open every day of the week.", "contradiction"),
    ("The new restaurant on Main Street opened last month and has been busy "
     "every weekend.",
     "The restaurant serves Italian food.", "neutral"),
]


def sanity_check_checkpoint(model_name: str) -> Tuple[bool, str]:
    """Load `model_name` and verify it on SANITY_PAIRS.

    Returns (ok, message). ok=False covers BOTH failure modes the spec asks
    to guard against: the checkpoint could not be obtained at all (import/
    download/config error), or it loaded but got a known pair wrong (which
    would mean CrossEncoderNLI's id2label resolution failed for this
    checkpoint's config — see its docstring). Either way the checkpoint must
    not be used for the full sweep silently.
    """
    try:
        nli = CrossEncoderNLI(model_name=model_name)
        preds = [nli.predict(p, h) for p, h, _ in SANITY_PAIRS]
    except Exception as exc:                                    # noqa: BLE001
        return False, f"could not load/run {model_name!r}: {exc}"

    lines = []
    ok = True
    for (premise, hyp, expected), pred in zip(SANITY_PAIRS, preds):
        mark = "OK" if pred == expected else "MISMATCH"
        if pred != expected:
            ok = False
        lines.append(f"      [{mark}] expected={expected:<13} got={pred:<13} "
                     f"{premise[:44]!r} / {hyp[:44]!r}")
    msg = f"    {model_name}\n" + "\n".join(lines)
    return ok, msg


def run_preflight(model_names: Dict[str, str]) -> bool:
    """Sanity-check every distinct checkpoint the sweep will load.

    `model_names` maps model_key ("base"/"large"/"alt") -> checkpoint id.
    Returns True iff every checkpoint passed — the caller aborts the sweep
    otherwise rather than burning compute on a mislabelled model.
    """
    section("Preflight — checkpoint availability + label-order sanity check")
    all_ok = True
    for key, name in model_names.items():
        ok, msg = sanity_check_checkpoint(name)
        print(f"\n  [{key}] {'PASS' if ok else 'FAIL'}")
        print(msg)
        all_ok = all_ok and ok
    if not all_ok:
        print("\n  [ABORT] at least one checkpoint failed the sanity check — see above.")
        print("  Do not substitute silently: pass a working --alt-nli-model / "
              "--large-nli-model, or drop that configuration with --configs.")
    return all_ok


# ---------------------------------------------------------------------------
# Driver: one disagreement_pilot.py subprocess per configuration
# ---------------------------------------------------------------------------

def sens_out_path(out_dir: Path, dataset: str, run_id: str) -> Path:
    return out_dir / f"sens_{dataset}_{run_id}.jsonl"


def dataset_extra_flags(args: argparse.Namespace, dataset: str) -> List[str]:
    """Baseline flags for `dataset` (e.g. --short-answer-keys for SimpleQA,
    see DEFAULT_DATASET_EXTRA_ARGS) plus --pilot-args, applied on top of
    every configuration's own RUN_SPECS flags. Both are user-overridable
    (--simpleqa-extra-args / --truthfulqa-extra-args / --pilot-args) so a
    future settled-branch change doesn't require editing this file again.
    """
    per_dataset = {"simpleqa": args.simpleqa_extra_args,
                   "truthfulqa": args.truthfulqa_extra_args}[dataset]
    return shlex.split(per_dataset) + shlex.split(args.pilot_args)


def build_command(python_exe: str, dataset: str, n: int, seed: int,
                  nli_model: str, extra_flags: List[str], out_path: Path,
                  max_bad_questions: int) -> List[str]:
    return [
        python_exe, "scripts/disagreement_pilot.py",
        "--dataset", dataset, "--n", str(n), "--seed", str(seed),
        "--nli-model", nli_model, "--out", str(out_path),
        "--max-bad-questions", str(max_bad_questions),
    ] + extra_flags


def run_one_config(cmd: List[str], log_path: Path) -> Tuple[bool, float]:
    """Run one configuration as a subprocess; never raises.

    A literal subprocess (not an in-process call) so a hard crash or OOM in
    one checkpoint (deberta-v3-large is the likely candidate) cannot take
    down the whole sweep — the shell-loop-survives-individual-failures
    requirement from the spec's Compute section.
    """
    t0 = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   cwd=REPO_ROOT)
        ok = proc.returncode == 0
    except Exception as exc:                                    # noqa: BLE001
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[nli_sensitivity] subprocess raised: {exc}\n")
        ok = False
    return ok, time.time() - t0


# ---------------------------------------------------------------------------
# IO + judge-grade backfill
# ---------------------------------------------------------------------------

def load_rows_by_uid(path: Path) -> "Dict[str, dict]":
    rows: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[row["uid"]] = row
    return rows


def write_rows_jsonl(path: Path, rows_by_uid: "Dict[str, dict]") -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows_by_uid.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def backfill_judge_grades(rows_by_uid: "Dict[str, dict]",
                          judge_source: "Dict[str, dict]") -> int:
    """Copy `grades_judge` from `judge_source` onto `rows_by_uid` in place.

    Only when the response TEXT actually matches for that uid — a sweep run
    that somehow diverged (different pool, different truncation) must not
    silently inherit a verdict for a different answer. Same safeguard as
    eval_short_answer_branch.py's backfill_judge_grades, applied here across
    NLI configurations instead of across the short-answer-keys branch.
    """
    n = 0
    for uid, row in rows_by_uid.items():
        if row.get("grades_judge"):
            continue
        src = judge_source.get(uid)
        if not src or not src.get("grades_judge"):
            continue
        if row.get("responses") != src.get("responses"):
            continue
        row["grades_judge"] = src["grades_judge"]
        n += 1
    return n


def verify_canonical_reproduction(rows_by_uid: "Dict[str, dict]",
                                  reference_path: Path) -> None:
    """Verification step 1: `canonical` must reproduce the existing canonical
    results file exactly. Reports a diff rather than raising — a mismatch is
    informative (it means the sweep is not measuring what it claims to),
    not a bug in this script.
    """
    reference = load_rows_by_uid(reference_path)
    common = sorted(set(rows_by_uid) & set(reference))
    print(f"  {reference_path}: {len(common)} uid(s) in common "
          f"(this run has {len(rows_by_uid)}, reference has {len(reference)})")
    if not common:
        print("  [!] no uid in common — cannot verify.")
        return
    fields = ("semantic_entropy", "n_clusters", "jaccard", "cluster_sizes", "grades")
    mismatches = []
    for uid in common:
        a, b = rows_by_uid[uid], reference[uid]
        diffs = [f for f in fields if a.get(f) != b.get(f)]
        if diffs:
            mismatches.append((uid, diffs))
    if not mismatches:
        print(f"  [OK] all {len(common)} common row(s) match exactly on {fields}.")
    else:
        print(f"  [MISMATCH] {len(mismatches)}/{len(common)} row(s) differ:")
        for uid, diffs in mismatches[:10]:
            print(f"    {uid}: differs on {diffs}")
        if len(mismatches) > 10:
            print(f"    ... and {len(mismatches) - 10} more")
        print("  The sweep's canonical cell does not reproduce the reference file —")
        print("  investigate before trusting any other cell in this table.")


# ---------------------------------------------------------------------------
# Per-run metric extraction — mirrors the relevant disagreement_pilot.py
# report_* sections, but returns structured values instead of printing.
# ---------------------------------------------------------------------------

def _majority_wrong(row: dict, grade_key: str) -> Optional[bool]:
    """None (not False) when `row` has no grade under `grade_key` at all —
    e.g. a row grades_judge backfill never reached. Callers MUST drop None
    labels before computing AUROC: treating "no verdict" as "majority
    correct" silently dilutes the positive class with mislabelled rows (see
    conclusion1_gap and detection_metrics, which both do this filtering)."""
    g = [v for v in (row.get(grade_key) or {}).values() if v is not None]
    if not g:
        return None
    return sum(v is False for v in g) > len(g) / 2


def _scorable_core(rows: Sequence[dict]) -> List[dict]:
    return [r for r in rows
            if r.get("panels", {}).get("core")
            and r["panels"]["core"].get("semantic_entropy") is not None]


def _core_agent_names(rows: Sequence[dict]) -> List[str]:
    names: set = set()
    for r in rows:
        names.update((r.get("grades") or {}).keys())
    return sorted(names)


def _panel_keys_by_prefix(rows: Sequence[dict], prefix: str) -> List[str]:
    if not rows:
        return []
    return sorted(k for k in rows[0].get("panels", {}) if k.startswith(prefix))


def clustering_behaviour(rows: Sequence[dict]) -> dict:
    scored = _scorable_core(rows)
    sem = [r["semantic_entropy"] for r in scored]
    nclusters = [r["n_clusters"] for r in scored]
    nscored = [r["panels"]["core"]["n_scored"] for r in scored]
    saturated = sum(1 for nc, ns in zip(nclusters, nscored) if ns and nc == ns)
    return {
        "n": len(scored),
        "mean_entropy": statistics.mean(sem) if sem else None,
        "n_clusters_dist": dict(Counter(nclusters)),
        "saturated_frac": (saturated / len(scored)) if scored else None,
        "distinct_values": len(set(sem)),
        "largest_tie_block": max(Counter(sem).values()) if sem else 0,
    }


def detection_metrics(rows: Sequence[dict]) -> dict:
    scored = _scorable_core(rows)
    out: dict = {}

    for grade_key, label in (("grades", "nli"), ("grades_judge", "judge")):
        pairs = [(r, _majority_wrong(r, grade_key)) for r in scored]
        pairs = [(r, lab) for r, lab in pairs if lab is not None]
        if not pairs:
            out[f"panel_auroc_{label}"] = None
            continue
        labels = [lab for _, lab in pairs]
        sem_subset = [r["semantic_entropy"] for r, _ in pairs]
        auc, _why = auroc_guarded(sem_subset, labels)
        out[f"panel_auroc_{label}"] = auc

    loo_by_agent: dict = {}
    have_judge = any(r.get("grades_judge") for r in scored)
    if have_judge:
        for n in _core_agent_names(rows):
            key = f"loo:{short(n)}"
            pairs = [(r, (r.get("grades_judge") or {}).get(n)) for r in scored]
            pairs = [(r, g) for r, g in pairs if g is not None]
            loo_pairs = [(r, g) for r, g in pairs
                        if r["panels"].get(key)
                        and r["panels"][key]["semantic_entropy"] is not None]
            labs = [g is False for _, g in loo_pairs]
            scores = [r["panels"][key]["semantic_entropy"] for r, _ in loo_pairs]
            auc, _why = auroc_guarded(scores, labs)
            loo_by_agent[n] = {
                "auroc": auc, "n": len(loo_pairs),
                "err_rate": (sum(labs) / len(labs)) if labs else None,
            }
    out["loo_by_agent"] = loo_by_agent
    return out


def _dominant_wrong(row: dict, key: str = "core", frac: float = 0.5,
                    require_homogeneous: bool = True) -> Optional[tuple]:
    p = row.get("panels", {}).get(key)
    if not p or not p.get("cluster_sizes"):
        return None
    sizes = p["cluster_sizes"]
    n = sum(sizes)
    if not n or max(sizes) <= frac * n:
        return None
    cid = sizes.index(max(sizes))
    members = [a for a, c in p["cluster_of"].items() if c == cid]
    graded = [row["grades"][a] for a in members if row["grades"].get(a) is not None]
    if not graded:
        return None
    if require_homogeneous and any(g is True for g in graded):
        return None
    if sum(g is False for g in graded) <= len(graded) / 2:
        return None
    return (cid, len(members), n)


def shared_bias_false_negatives(rows: Sequence[dict]) -> dict:
    fn_strict = [r for r in rows if r.get("panel_agrees") and r.get("panel_majority_wrong")]
    fn_homog = [r for r in rows if _dominant_wrong(r)]
    fn_any = [r for r in rows if _dominant_wrong(r, require_homogeneous=False)]
    return {
        "strict": len(fn_strict),
        "majority_homogeneous": len(fn_homog),
        "majority_possibly_mixed": len(fn_any),
        "clustering_error_gap": len(fn_any) - len(fn_homog),
        "n_rows": len(rows),
    }


def conclusion1_gap(rows: Sequence[dict]) -> Optional[dict]:
    """Jaccard vs semantic-entropy AUROC, judge-graded majority-wrong.

    Only over rows with an actual judge verdict (_majority_wrong returns
    None for the rest — e.g. rows a grades_judge backfill never reached).
    Including ungraded rows as label=False would dilute the positive class
    with rows that were never actually adjudicated, understating both
    AUROCs and the gap between them.
    """
    scored = _scorable_core(rows)
    pairs = [(r, _majority_wrong(r, "grades_judge")) for r in scored]
    pairs = [(r, lab) for r, lab in pairs if lab is not None]
    if not pairs:
        return None
    labels = [lab for _, lab in pairs]
    jac = [r["jaccard"] for r, _ in pairs]
    sem = [r["semantic_entropy"] for r, _ in pairs]
    a_j, _ = auroc_guarded(jac, labels)
    a_s, _ = auroc_guarded(sem, labels)
    gap = (a_s - a_j) if (a_j is not None and a_s is not None) else None
    return {"jaccard_auroc": a_j, "semantic_auroc": a_s, "gap": gap, "n": len(pairs)}


def conclusion2_spearman(loo_by_agent: dict) -> Optional[float]:
    """Spearman(per-agent judge-graded error rate, LOO AUROC)."""
    xs, ys = [], []
    for d in loo_by_agent.values():
        if d["err_rate"] is None or d["auroc"] is None:
            continue
        xs.append(d["err_rate"])
        ys.append(d["auroc"])
    if len(xs) < 3:
        return None
    return spearman(xs, ys)


def conclusion3_family(rows: Sequence[dict]) -> Optional[dict]:
    """Paired within-family vs cross-family difference, Wilcoxon + rank-biserial."""
    within = _panel_keys_by_prefix(rows, "within:")
    cross = _panel_keys_by_prefix(rows, "cross:")
    if not within or not cross:
        return None

    def avail_mean(r, keys):
        vals = [r["panels"][k]["semantic_entropy"] for k in keys
                if r["panels"].get(k) and r["panels"][k]["semantic_entropy"] is not None]
        return statistics.mean(vals) if vals else None

    diffs = []
    for r in rows:
        w, c = avail_mean(r, within), avail_mean(r, cross)
        if w is not None and c is not None:
            diffs.append(c - w)
    if not diffs:
        return None
    pos = sum(d > 0 for d in diffs)
    neg = sum(d < 0 for d in diffs)
    result = {"mean_diff": statistics.mean(diffs), "n": len(diffs),
              "pos": pos, "neg": neg, "wilcoxon_p": None, "rank_biserial": None}
    if pos + neg >= 6:
        res = wilcoxon(diffs)
        result["wilcoxon_p"] = float(res.pvalue)
        result["rank_biserial"] = (pos - neg) / (pos + neg)
    return result


def conclusion4_tiers(rows: Sequence[dict]) -> Optional[dict]:
    """Paired tier-panel means, small/large/strong (same questions across tiers)."""
    present = {t: f"tier:{t}" for t in ("small", "large", "strong")
              if rows and f"tier:{t}" in rows[0].get("panels", {})}
    if len(present) < 2:
        return None
    paired = common_rows(rows, list(present.values()))
    if not paired:
        return {"means": {t: None for t in present}, "n_paired": 0}
    means = {t: statistics.mean(r["panels"][k]["semantic_entropy"] for r in paired)
            for t, k in present.items()}
    return {"means": means, "n_paired": len(paired)}


def compute_all_metrics(rows: Sequence[dict]) -> dict:
    det = detection_metrics(rows)
    return {
        "n_rows": len(rows),
        "clustering": clustering_behaviour(rows),
        "detection": det,
        "shared_bias_fn": shared_bias_false_negatives(rows),
        "c1": conclusion1_gap(rows),
        "c2": conclusion2_spearman(det["loo_by_agent"]),
        "c3": conclusion3_family(rows),
        "c4": conclusion4_tiers(rows),
    }


# ---------------------------------------------------------------------------
# Direction-holds — bucket each conclusion's figure so a near-zero canonical
# value ("the two converge", "no meaningful effect") isn't reported as
# flipping on noise alone. A raw sign comparison would call every sub-0.01
# wobble around zero a "flip"; bucketing by a tolerance band is the same
# judgement call disagreement_pilot.py itself makes with MIN_CLASS_FOR_AUROC
# and the rank-biserial-alongside-p convention.
# ---------------------------------------------------------------------------

def _bucket(value: Optional[float], tol: float) -> Optional[str]:
    if value is None:
        return None
    if abs(value) < tol:
        return "negligible"
    return "positive" if value > 0 else "negative"


def direction_holds(canonical: dict, run: dict) -> dict:
    out = {}

    c1c, c1r = canonical.get("c1"), run.get("c1")
    gc = c1c["gap"] if c1c else None
    gr = c1r["gap"] if c1r else None
    out["c1"] = (None if gc is None or gr is None
                else _bucket(gc, TOL_AUROC_GAP) == _bucket(gr, TOL_AUROC_GAP))

    sc, sr = canonical.get("c2"), run.get("c2")
    out["c2"] = (None if sc is None or sr is None
                else _bucket(sc, TOL_SPEARMAN) == _bucket(sr, TOL_SPEARMAN))

    c3c, c3r = canonical.get("c3"), run.get("c3")
    rc = c3c["rank_biserial"] if c3c else None
    rr = c3r["rank_biserial"] if c3r else None
    out["c3"] = (None if rc is None or rr is None
                else _bucket(rc, TOL_RANK_BISERIAL) == _bucket(rr, TOL_RANK_BISERIAL))

    c4c, c4r = canonical.get("c4"), run.get("c4")

    def _monotonic(means):
        if not means or any(means.get(t) is None for t in ("small", "large", "strong")):
            return None
        return (means["small"] >= means["large"] - TOL_TIER_MONOTONIC
                and means["large"] >= means["strong"] - TOL_TIER_MONOTONIC)

    mc = _monotonic(c4c["means"]) if c4c else None
    mr = _monotonic(c4r["means"]) if c4r else None
    out["c4"] = None if mc is None or mr is None else (mc == mr)
    return out


# ---------------------------------------------------------------------------
# Table rendering — text + LaTeX
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], fmt: str = "+.3f") -> str:
    return format(v, fmt) if v is not None else "n/a"


def _fmt_c1(c1: Optional[dict]) -> str:
    if not c1:
        return "n/a"
    return (f"{_fmt(c1['semantic_auroc'], '.3f')}/{_fmt(c1['jaccard_auroc'], '.3f')} "
           f"({_fmt(c1['gap'])}, n={c1.get('n', '?')})")


def _fmt_c2(c2: Optional[float]) -> str:
    return _fmt(c2)


def _fmt_c3(c3: Optional[dict]) -> str:
    if not c3:
        return "n/a"
    r = c3.get("rank_biserial")
    return f"{_fmt(c3['mean_diff'])} (r={_fmt(r) if r is not None else 'n/a'})"


def _fmt_c4(c4: Optional[dict]) -> str:
    if not c4:
        return "n/a"
    m = c4["means"]
    parts = [f"{t[0]}={_fmt(m.get(t), '.3f')}" for t in ("small", "large", "strong")]
    return "/".join(parts)


def render_text_table(results: "Dict[str, Dict[str, dict]]") -> str:
    header = (f"{'dataset':<11} {'config':<10} {'n':>5}  "
              f"{'C1 se/jac(gap)':<26} {'C2 rho':<8} {'C3 diff(r)':<20} "
              f"{'C4 s/l/st':<26} {'wall(s)':>9}")
    lines = [header, "-" * len(header)]
    for dataset in DATASETS:
        if dataset not in results:
            continue
        for run_id in TABLE_ORDER:
            r = results[dataset].get(run_id)
            if r is None:
                lines.append(f"{dataset:<11} {run_id:<10}  -- not run --")
                continue
            m = r["metrics"]
            lines.append(
                f"{dataset:<11} {run_id:<10} {m['n_rows']:>5}  "
                f"{_fmt_c1(m['c1']):<26} {_fmt_c2(m['c2']):<8} "
                f"{_fmt_c3(m['c3']):<20} {_fmt_c4(m['c4']):<26} "
                f"{r['wall_clock_s']:>9.1f}")
    return "\n".join(lines)


def render_direction_summary(results: "Dict[str, Dict[str, dict]]") -> str:
    labels = {
        "c1": "C1 semantic entropy > Jaccard (judge AUROC)",
        "c2": "C2 detectability rises with per-agent error rate (Spearman)",
        "c3": "C3 family diversity: no meaningful effect at matched N",
        "c4": "C4 disagreement falls monotonically small->large->strong tier",
    }
    lines = []
    for dataset in DATASETS:
        if dataset not in results or "canonical" not in results[dataset]:
            continue
        canonical_metrics = results[dataset]["canonical"]["metrics"]
        lines.append(f"\n  {dataset}:")
        flips = []
        for key, label in labels.items():
            per_run = {}
            for run_id in TABLE_ORDER:
                r = results[dataset].get(run_id)
                if r is None:
                    continue
                holds = direction_holds(canonical_metrics, r["metrics"])[key]
                per_run[run_id] = holds
            n_hold = sum(1 for v in per_run.values() if v)
            n_total = sum(1 for v in per_run.values() if v is not None)
            lines.append(f"    {label}: holds in {n_hold}/{n_total} run(s) "
                        f"({', '.join(f'{k}={v}' for k, v in per_run.items())})")
            for run_id, v in per_run.items():
                if v is False:
                    flips.append((run_id, key))
        if flips:
            lines.append(f"    [FLIP] direction did not hold for: "
                        + ", ".join(f"{k} in {r}" for r, k in flips))
    return "\n".join(lines)


def render_tex(results: "Dict[str, Dict[str, dict]]") -> str:
    rows_tex = []
    for dataset in DATASETS:
        if dataset not in results:
            continue
        for run_id in TABLE_ORDER:
            r = results[dataset].get(run_id)
            if r is None:
                continue
            m = r["metrics"]
            rows_tex.append(
                f"{dataset} & {run_id} & {m['n_rows']} & "
                f"{_fmt_c1(m['c1'])} & {_fmt_c2(m['c2'])} & "
                f"{_fmt_c3(m['c3'])} & {_fmt_c4(m['c4'])} & "
                f"{r['wall_clock_s']:.1f} \\\\"
            )
    body = "\n".join(rows_tex)
    return (
        "% Auto-generated by scripts/nli_sensitivity.py — do not hand-edit.\n"
        "\\begin{tabular}{llrlllll}\n"
        "\\toprule\n"
        "dataset & config & $n$ & C1 (SE/Jac AUROC, gap) & C2 ($\\rho$) & "
        "C3 (diff, $r$) & C4 (small/large/strong) & wall-clock (s) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NLI backend / clustering-rule sensitivity sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--datasets", default=",".join(DATASETS),
                    help="comma-separated subset of " + ",".join(DATASETS))
    ap.add_argument("--configs", default=",".join(RUN_ORDER),
                    help="comma-separated subset of " + ",".join(RUN_ORDER))
    ap.add_argument("--n", type=int, default=1000, help="questions per run")
    ap.add_argument("--seed", type=int, default=0,
                    help="held fixed across every configuration so all five "
                         "runs see the same question set")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--base-nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--large-nli-model", default="cross-encoder/nli-deberta-v3-large")
    ap.add_argument("--alt-nli-model", default="roberta-large-mnli",
                    help="any non-DeBERTa MNLI cross-encoder; this default is "
                         "the conventional RoBERTa-large-MNLI choice per the spec")
    ap.add_argument("--simpleqa-extra-args", default=DEFAULT_DATASET_EXTRA_ARGS["simpleqa"],
                    help="extra disagreement_pilot.py flags applied to every "
                         "SimpleQA configuration (shlex-split), on top of each "
                         "config's own flags in RUN_SPECS — default bakes in "
                         "the adopted short-answer-keys branch so `canonical` "
                         "means what it currently means")
    ap.add_argument("--truthfulqa-extra-args", default=DEFAULT_DATASET_EXTRA_ARGS["truthfulqa"],
                    help="same as --simpleqa-extra-args, for TruthfulQA — empty "
                         "by default, since short-answer-keys is inert there "
                         "(no answer_type field to gate date/number extraction)")
    ap.add_argument("--pilot-args", default="",
                    help="extra disagreement_pilot.py flags (shlex-split) "
                         "applied to EVERY run, both datasets — for anything "
                         "not covered by the per-dataset flags above")
    ap.add_argument("--judge-source-simpleqa", default="",
                    help="existing judge-graded SimpleQA results file to "
                         "backfill grades_judge from (see module docstring — "
                         "no fresh judge API calls are made by this script)")
    ap.add_argument("--judge-source-truthfulqa", default="")
    ap.add_argument("--canonical-reference-simpleqa", default="",
                    help="existing canonical SimpleQA results file; the "
                         "canonical run is diffed against it (verification 1)")
    ap.add_argument("--canonical-reference-truthfulqa", default="")
    ap.add_argument("--max-bad-questions", type=int, default=5)
    ap.add_argument("--skip-sanity-check", action="store_true",
                    help="skip the label-order preflight (not recommended)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-run a (dataset, config) whose --out file "
                         "already exists and is non-empty — resume a sweep "
                         "interrupted partway through")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned commands and run the preflight only; "
                         "execute nothing, write nothing")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in RUN_SPECS:
            ap.error(f"unknown config {c!r} — choose from {sorted(RUN_SPECS)}")
    for d in datasets:
        if d not in DATASETS:
            ap.error(f"unknown dataset {d!r} — choose from {DATASETS}")

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = {
        "base": args.base_nli_model,
        "large": args.large_nli_model,
        "alt": args.alt_nli_model,
    }
    used_keys = {RUN_SPECS[c]["model_key"] for c in configs}
    model_names = {k: v for k, v in model_names.items() if k in used_keys}

    if not args.skip_sanity_check:
        if not run_preflight(model_names):
            sys.exit(1)
    else:
        print("  [!] --skip-sanity-check: label-order verification NOT run. "
              "A mislabelled checkpoint would invert every cluster silently.")

    ordered_configs = [c for c in RUN_ORDER if c in configs]

    plan = [(d, c) for d in datasets for c in ordered_configs]
    section(f"Sweep plan — {len(plan)} run(s)")
    for dataset, run_id in plan:
        spec = RUN_SPECS[run_id]
        flags = spec["flags"] + dataset_extra_flags(args, dataset)
        cmd = build_command(args.python, dataset, args.n, args.seed,
                            model_names[spec["model_key"]], flags,
                            sens_out_path(out_dir, dataset, run_id),
                            args.max_bad_questions)
        print("  " + " ".join(cmd))

    if args.dry_run:
        print("\n  [dry-run] nothing executed.")
        return

    judge_sources = {}
    for dataset, flag in (("simpleqa", args.judge_source_simpleqa),
                          ("truthfulqa", args.judge_source_truthfulqa)):
        if flag:
            path = _resolve_path(flag)
            if not path.exists():
                print(f"  [!] --judge-source for {dataset} does not exist: {path} — "
                      f"continuing without judge backfill for this dataset.")
            else:
                judge_sources[dataset] = load_rows_by_uid(path)
        if dataset not in judge_sources:
            print(f"  [!] no --judge-source-{dataset}: grades_judge will not be "
                  f"backfilled, so conclusion 1/2 and the judge-graded panel AUROC "
                  f"will read n/a for {dataset}.")

    canonical_refs = {
        "simpleqa": args.canonical_reference_simpleqa,
        "truthfulqa": args.canonical_reference_truthfulqa,
    }

    results: Dict[str, Dict[str, dict]] = {}
    wall_clock_log: List[Tuple[str, str, float, bool]] = []
    first_run_done = False
    remaining = len(plan)

    for dataset, run_id in plan:
        remaining -= 1
        spec = RUN_SPECS[run_id]
        out_path = sens_out_path(out_dir, dataset, run_id)
        log_path = out_path.with_suffix(".log")

        flags = spec["flags"] + dataset_extra_flags(args, dataset)

        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            print(f"\n  [skip-existing] {dataset}/{run_id}: {out_path} already present.")
        else:
            section(f"Running {dataset}/{run_id}  (nli-model={model_names[spec['model_key']]}"
                    f"{' ' + ' '.join(flags) if flags else ''})")
            cmd = build_command(args.python, dataset, args.n, args.seed,
                                model_names[spec["model_key"]], flags,
                                out_path, args.max_bad_questions)
            ok, elapsed = run_one_config(cmd, log_path)
            wall_clock_log.append((dataset, run_id, elapsed, ok))
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {dataset}/{run_id} in {elapsed:.1f}s "
                  f"(log: {log_path})")
            if not ok:
                print(f"  [!] {dataset}/{run_id} failed — continuing with the "
                      f"remaining configurations (see log for the error).")
                continue
            if not first_run_done:
                first_run_done = True
                if remaining:
                    projected = elapsed * remaining
                    print(f"\n  [i] first run took {elapsed:.1f}s; if every "
                          f"remaining run took as long the sweep would need "
                          f"~{projected/60:.0f} more minute(s). Ctrl-C and "
                          f"re-run with a smaller --n if that's impractical — "
                          f"reduce the question count, not the configuration "
                          f"list (see this script's docstring).")

        if not out_path.exists() or out_path.stat().st_size == 0:
            print(f"  [!] {out_path} missing or empty — skipping analysis for "
                  f"{dataset}/{run_id}.")
            continue

        rows_by_uid = load_rows_by_uid(out_path)
        n_backfilled = 0
        if dataset in judge_sources:
            n_backfilled = backfill_judge_grades(rows_by_uid, judge_sources[dataset])
            if n_backfilled:
                write_rows_jsonl(out_path, rows_by_uid)
            print(f"  backfilled grades_judge onto {n_backfilled}/{len(rows_by_uid)} "
                  f"row(s) from the judge source.")

        if run_id == "canonical" and canonical_refs.get(dataset):
            ref_path = _resolve_path(canonical_refs[dataset])
            section(f"Verification 1 — {dataset} canonical reproduces {ref_path}")
            if ref_path.exists():
                verify_canonical_reproduction(rows_by_uid, ref_path)
            else:
                print(f"  [!] canonical reference does not exist: {ref_path}")

        metrics = compute_all_metrics(list(rows_by_uid.values()))
        results.setdefault(dataset, {})[run_id] = {
            "metrics": metrics,
            "wall_clock_s": next((e for d, r, e, ok in wall_clock_log
                                  if d == dataset and r == run_id and ok), 0.0),
            "n_backfilled_judge": n_backfilled,
        }

    section("Summary table")
    table_text = render_text_table(results)
    print(table_text)
    print(render_direction_summary(results))

    summary_txt_path = out_dir / "sens_summary.txt"
    summary_tex_path = out_dir / "sens_summary.tex"
    summary_json_path = out_dir / "sens_summary.json"
    summary_txt_path.write_text(
        table_text + "\n" + render_direction_summary(results) + "\n", encoding="utf-8")
    summary_tex_path.write_text(render_tex(results), encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    section("Wall-clock per configuration")
    for dataset, run_id, elapsed, ok in wall_clock_log:
        print(f"  {dataset:<11} {run_id:<10} {elapsed:>9.1f}s  {'OK' if ok else 'FAILED'}")

    section("Done")
    print(f"  Per-run results   : {out_dir}/sens_{{dataset}}_{{run}}.jsonl")
    print(f"  Summary table      : {summary_txt_path}")
    print(f"  LaTeX fragment     : {summary_tex_path}")
    print(f"  Structured summary : {summary_json_path}")


if __name__ == "__main__":
    main()
