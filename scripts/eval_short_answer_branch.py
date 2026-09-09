"""Evaluate the --short-answer-keys clustering branch against a baseline run.

`scripts/disagreement_pilot.py --short-answer-keys` injects an extracted-key
short-circuit into `SemanticEntropyDisagreement` (see
`disagreement/semantic.py`'s `key_fn` and `disagreement_pilot.py`'s
`make_short_answer_key_fn`) so that short-form answers — dates, numbers,
chemical formulae, and (behind --short-entity-keys) multi-entity lists — are
compared by extracted key instead of NLI entailment. DeBERTa-v3 reliably
labels two *different* short-form values NEUTRAL rather than CONTRADICTION,
so relaxed clustering merges them (simpleqa-3278: nine distinct chemical
formulae, six merged into one cluster).

This script is the evaluation half of that branch, not the implementation:
given a BASELINE results file (NLI-only, `--short-answer-keys` off) and a
BRANCH results file (`--short-answer-keys` on) over the *same questions and
pool*, it reports whether the branch is a net improvement, and precisely
what it costs.

WHY GRADE EQUALITY IS NOT A PROXY FOR ANSWER EQUALITY. An earlier version of
this script flagged any same-grade pair the branch split as a "false split".
That conflates "both wrong" with "both wrong in the same way" — two agents
can be incorrect in genuinely different directions, and separating them is
the clustering working, not a defect (simpleqa-3278's nine distinct wrong
chemical formulae are exactly this). This version partitions every pair by
what an INDEPENDENT judge's two grades actually license:

  Bucket A  correct/correct     : same claim — splitting is unambiguously
                                   wrong. -> false-split rate.
  Bucket B  correct/incorrect   : different claims — merging is unambiguously
                                   wrong. -> false-merge rate (the pairwise
                                   form of the old mixed-cluster metric).
  Bucket C  incorrect/incorrect : could be the same wrong answer or two
                                   different ones; grades cannot tell you
                                   which. No automatic verdict — sampled for
                                   manual adjudication instead.

Grading uses `grades_judge` (an LLM judge outside the pool), not `grades`
(the NLI grader) — the NLI grader shares its model with the clustering this
script evaluates, so using it here would make the bucket classification
circular with the thing being measured (see disagreement_pilot.py's own
"Grader independence" note). Rows need `grades_judge` populated (--judge-model
on the run that produced them) for their pairs to be classifiable.

Because Bucket C is exactly where a grade-based metric goes blind, this
script adds a grade-FREE structural measure alongside it: the size of the
largest cluster whose members are ALL graded incorrect. simpleqa-3278 (six
different wrong formulae in one cluster, 6 -> 1 after the branch) shows up
here even though it never counted as "mixed" and so was invisible to any
correctness-mixing metric.

Run from the project root:
    python scripts/eval_short_answer_branch.py \\
        --baseline outputs/simpleqa_3x3_results_1000.jsonl \\
        --branch outputs/simpleqa_results_keys.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.disagreement_pilot import (                              # noqa: E402
    auroc_guarded, section, short, make_short_answer_key_fn,
)

# simpleqa-1315 deliberately excluded — see report_named_cases: at the current
# nine-agent core pool NLI already splits it correctly, so there is nothing
# for the branch to fix. The original spec's description of it as a merge
# failure was written from a four-agent run and does not hold here.
DEFAULT_NAMED_CASES = ["simpleqa-3278", "simpleqa-0716"]
PANEL_KEY = "core"   # the frozen 3x3 pool — the headline panel throughout
                      # disagreement_pilot.py, and where the named cases live.


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        rows[row["uid"]] = row
    return rows


def backfill_judge_grades(baseline: Dict[str, dict], branch: Dict[str, dict],
                          uids: List[str]) -> Tuple[Dict[str, dict], int]:
    """Judge grades are a property of the (question, agent, response text)
    triple, not of which file clustered it. Where the branch run has no
    grades_judge (e.g. --judge-model wasn't passed, as in our own --n 50
    verification — the responses are byte-identical to the baseline's, so
    re-querying the judge would spend API calls to reproduce a verdict that
    already exists), backfill it from the baseline for uids in common.

    Only borrows a grade when the response TEXT actually matches — a branch
    run over a different pool or dataset slice must not silently inherit the
    wrong verdict. Returns (new branch dict, count of rows backfilled); every
    downstream function can then just read row["grades_judge"] directly
    instead of re-implementing this fallback (an earlier version of this
    script had the fallback in classify_pairs but not in
    largest_uniformly_wrong_cluster, which silently zeroed out the branch's
    entire over-merge measurement — see the fix note in report_overmerge).
    """
    effective = dict(branch)
    n_backfilled = 0
    for u in uids:
        b, br = baseline[u], branch[u]
        if br.get("grades_judge") or not b.get("grades_judge"):
            continue
        if b.get("responses") != br.get("responses"):
            continue   # different responses — do not borrow a grade
        merged = dict(br)
        merged["grades_judge"] = b["grades_judge"]
        effective[u] = merged
        n_backfilled += 1
    return effective, n_backfilled


# ---------------------------------------------------------------------------
# Pair classification — the three buckets
# ---------------------------------------------------------------------------

def classify_pairs(baseline: Dict[str, dict], branch: Dict[str, dict],
                   uids: List[str], key: str = PANEL_KEY
                   ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Every core-panel pair, for every common question, classified by the
    two agents' JUDGE grades into bucket A (correct/correct), B
    (correct/incorrect), or C (incorrect/incorrect). Pairs where either
    grade is missing are skipped entirely — no bucket has a verdict without
    both grades.

    Each record carries `baseline_merged`/`branch_merged` (were the two
    agents in the same cluster under that file's own clustering) so the
    caller can compute per-file rates without a second pass.
    """
    bucket_a: List[dict] = []
    bucket_b: List[dict] = []
    bucket_c: List[dict] = []
    for u in uids:
        b, a = baseline[u], branch[u]
        bp, ap = b.get("panels", {}).get(key), a.get("panels", {}).get(key)
        if not bp or not ap or not bp.get("cluster_of") or not ap.get("cluster_of"):
            continue
        # `branch` has already been through backfill_judge_grades (see main),
        # so its grades_judge is authoritative here — no fallback needed.
        gj = a.get("grades_judge") or {}
        members = sorted(set(bp["cluster_of"]) & set(ap["cluster_of"]) & set(gj))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                m1, m2 = members[i], members[j]
                g1, g2 = gj.get(m1), gj.get(m2)
                if g1 is None or g2 is None:
                    continue
                rec = {
                    "uid": u, "agent_a": m1, "agent_b": m2, "g1": g1, "g2": g2,
                    "answer_type": b.get("answer_type"),
                    "text_a": b["responses"].get(m1, ""),
                    "text_b": b["responses"].get(m2, ""),
                    "baseline_merged": bp["cluster_of"][m1] == bp["cluster_of"][m2],
                    "branch_merged": ap["cluster_of"][m1] == ap["cluster_of"][m2],
                }
                if g1 and g2:
                    bucket_a.append(rec)
                elif g1 != g2:
                    bucket_b.append(rec)
                else:
                    bucket_c.append(rec)
    return bucket_a, bucket_b, bucket_c


def report_three_buckets(bucket_a: List[dict], bucket_b: List[dict],
                         bucket_c: List[dict], n_samples: int = 20) -> List[dict]:
    section("Three-bucket pairwise analysis (judge-graded)")
    print(f"  bucket sizes — A (correct/correct): {len(bucket_a)}   "
          f"B (correct/incorrect): {len(bucket_b)}   "
          f"C (incorrect/incorrect): {len(bucket_c)}")

    print("\n  Bucket A — correct/correct: same claim, so a split is unambiguously")
    print("  wrong. False-split rate (lower is better):")
    n_a = len(bucket_a)
    base_split = sum(1 for r in bucket_a if not r["baseline_merged"])
    branch_split = sum(1 for r in bucket_a if not r["branch_merged"])
    if n_a:
        print(f"    baseline : {base_split}/{n_a} split ({base_split/n_a:.1%})")
        print(f"    branch   : {branch_split}/{n_a} split ({branch_split/n_a:.1%})")
    else:
        print("    n/a — no correct/correct pairs with both judge grades.")

    print("\n  Bucket B — correct/incorrect: different claims, so a merge is")
    print("  unambiguously wrong. False-merge rate (lower is better) — the pairwise")
    print("  form of the old mixed-cluster metric:")
    n_b = len(bucket_b)
    base_merge = sum(1 for r in bucket_b if r["baseline_merged"])
    branch_merge = sum(1 for r in bucket_b if r["branch_merged"])
    if n_b:
        print(f"    baseline : {base_merge}/{n_b} merged ({base_merge/n_b:.1%})")
        print(f"    branch   : {branch_merge}/{n_b} merged ({branch_merge/n_b:.1%})")
    else:
        print("    n/a — no correct/incorrect pairs with both judge grades.")

    print("\n  Bucket C — incorrect/incorrect: could be the same wrong answer or two")
    print("  different ones; the grades cannot tell you which. No automatic verdict.")
    disagreements = [r for r in bucket_c if r["baseline_merged"] != r["branch_merged"]]
    print(f"    baseline/branch disagree on {len(disagreements)}/{len(bucket_c)} pair(s) "
          f"whether to merge.")
    if disagreements:
        key_fn = make_short_answer_key_fn(short_entity=True)
        print(f"\n  Sampling {min(n_samples, len(disagreements))} of {len(disagreements)} "
              f"for manual adjudication:")
        for r in disagreements[:n_samples]:
            ka = key_fn(r["text_a"], r["answer_type"])
            kb = key_fn(r["text_b"], r["answer_type"])
            direction = ("baseline merged, branch split" if r["baseline_merged"]
                        else "baseline split, branch merged")
            print(f"\n    {r['uid']}  [{r['answer_type']}]  {direction}")
            print(f"      {short(r['agent_a']):<26} {r['text_a'].strip()[:100]!r}")
            print(f"        key: {sorted(ka) if ka else ka}")
            print(f"      {short(r['agent_b']):<26} {r['text_b'].strip()[:100]!r}")
            print(f"        key: {sorted(kb) if kb else kb}")
    return disagreements


# ---------------------------------------------------------------------------
# Over-merge — grade-free, catches what Bucket C can't verdict
# ---------------------------------------------------------------------------

def largest_uniformly_wrong_cluster(row: dict, key: str = PANEL_KEY
                                    ) -> Tuple[int, List[str]]:
    """Size (and members) of the largest cluster all of whose members are
    judge-graded incorrect; (0, []) if there is no such cluster (including
    when a member's judge grade is missing — a cluster can't be confirmed
    uniformly wrong without a grade for every member).

    Reads `row["grades_judge"]` directly — the caller (main, via
    backfill_judge_grades) is responsible for ensuring it's populated. An
    earlier version of this function read the branch file's OWN
    grades_judge, which was None (no --judge-model on that run); every
    branch-side cluster was then silently skipped for lacking a grade, and
    the whole distribution read a spuriously perfect 0.00 instead of the
    real (nonzero) post-branch sizes. Caught by inspecting simpleqa-3278's
    branch clustering directly — its 9 singleton clusters (all grade=False)
    should score 1, not 0."""
    p = row.get("panels", {}).get(key)
    gj = row.get("grades_judge") or {}
    if not p or not p.get("cluster_of"):
        return 0, []
    by_cluster: Dict[int, List[str]] = defaultdict(list)
    for member, cid in p["cluster_of"].items():
        by_cluster[cid].append(member)
    best_size, best_members = 0, []
    for members in by_cluster.values():
        graded = [gj.get(m) for m in members]
        if any(g is None for g in graded) or not all(g is False for g in graded):
            continue
        if len(members) > best_size:
            best_size, best_members = len(members), members
    return best_size, best_members


def report_overmerge(baseline: Dict[str, dict], branch: Dict[str, dict],
                     uids: List[str], top_n: int = 10) -> None:
    section("Over-merge — largest uniformly-wrong cluster per question")
    print("  Grade-free: catches simpleqa-3278-shaped failures (many DIFFERENT wrong")
    print("  answers merged into one cluster) that Bucket C above cannot verdict,")
    print("  because that cluster was never mixed-correctness in the first place.")

    base_size: Dict[str, int] = {}
    branch_size: Dict[str, int] = {}
    reductions: List[Tuple[int, str, int, int]] = []
    for u in uids:
        # Judge grades needed on both sides to confirm uniform wrongness —
        # skip a question if neither file has them for it at all.
        if not (baseline[u].get("grades_judge") or branch[u].get("grades_judge")):
            continue
        bs, _ = largest_uniformly_wrong_cluster(baseline[u])
        asz, _ = largest_uniformly_wrong_cluster(branch[u])
        base_size[u], branch_size[u] = bs, asz
        reductions.append((bs - asz, u, bs, asz))

    scored_uids = list(base_size)
    if not scored_uids:
        print("\n  n/a — no common question has grades_judge populated.")
        return

    cb = Counter(base_size[u] for u in scored_uids)
    ca = Counter(branch_size[u] for u in scored_uids)
    print(f"\n  n={len(scored_uids)} question(s) with judge grades on both sides")
    print(f"  baseline : mean={statistics.mean(base_size[u] for u in scored_uids):.2f}  "
          f"distribution (size->count): " + ", ".join(f"{k}->{cb[k]}" for k in sorted(cb)))
    print(f"  branch   : mean={statistics.mean(branch_size[u] for u in scored_uids):.2f}  "
          f"distribution (size->count): " + ", ".join(f"{k}->{ca[k]}" for k in sorted(ca)))

    reductions.sort(key=lambda t: (-t[0], t[1]))
    positive = [r for r in reductions if r[0] > 0]
    print(f"\n  {len(positive)} question(s) with a reduction; "
          f"the {min(top_n, len(positive))} largest:")
    for delta, u, bs, asz in positive[:top_n]:
        row = branch[u]
        print(f"\n    {u}  {bs} -> {asz}  Q: {row.get('question', '')[:90]}")
        _, members = largest_uniformly_wrong_cluster(baseline[u])
        for m in members:
            print(f"      {short(m):<26} {baseline[u]['responses'].get(m, '').strip()[:90]!r}")


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def report_coverage(branch: Dict[str, dict], uids: List[str], key: str = PANEL_KEY) -> None:
    section("Coverage — pairwise comparisons decided by the key vs NLI")
    overall = [0, 0]   # [key_decided, total]
    by_type: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for u in uids:
        p = branch[u].get("panels", {}).get(key)
        if not p:
            continue
        d = p.get("key_decided_comparisons", 0)
        t = p.get("total_comparisons", 0)
        overall[0] += d
        overall[1] += t
        at = branch[u].get("answer_type") or "(none)"
        by_type[at][0] += d
        by_type[at][1] += t

    if not overall[1]:
        print("  No comparisons recorded — branch file predates the key_fn "
              "instrumentation (score_panel's key_decided_comparisons/"
              "total_comparisons fields) or --short-answer-keys was off.")
        return

    print(f"  overall : {overall[0]}/{overall[1]} comparisons "
          f"({overall[0]/overall[1]:.1%}) decided by the key")
    print(f"\n  by answer_type:")
    for at in sorted(by_type):
        d, t = by_type[at]
        pct = f"{d/t:.1%}" if t else "n/a"
        print(f"    {at:<10} {d:>5}/{t:<5} ({pct})")

    interp_bits = []
    for at in ("Date", "Number"):
        d, t = by_type.get(at, [0, 0])
        if t:
            interp_bits.append(f"{at} {d/t:.1%}")
    if interp_bits:
        print(f"\n  Stated plainly: at {', '.join(interp_bits)} coverage, those answer")
        print("  types are effectively decided by exact key match rather than NLI")
        print("  entailment — not merely assisted by it. Exact matching may well be the")
        print("  more correct equivalence relation for a date or a count than entailment")
        print("  is, but that is a design choice being made here, not a neutral default.")


# ---------------------------------------------------------------------------
# Headline movement
# ---------------------------------------------------------------------------

def _majority_wrong(row: dict, grade_key: str = "grades") -> bool:
    g = [v for v in (row.get(grade_key) or {}).values() if v is not None]
    return bool(g) and sum(v is False for v in g) > len(g) / 2


def _scorable(rows: Sequence[dict], key: str = PANEL_KEY) -> List[dict]:
    return [r for r in rows
            if r.get("panels", {}).get(key)
            and r["panels"][key].get("semantic_entropy") is not None]


def _headline_stats(rows: List[dict]) -> dict:
    sem = [r["panels"][PANEL_KEY]["semantic_entropy"] for r in rows]
    n_clusters = [r["panels"][PANEL_KEY]["n_clusters"] for r in rows]
    n_scored = [r["panels"][PANEL_KEY]["n_scored"] for r in rows]
    saturated = sum(1 for nc, ns in zip(n_clusters, n_scored) if ns and nc == ns)
    aurocs = {}
    for grade_key, glabel in (("grades", "NLI grader"), ("grades_judge", "LLM judge")):
        if grade_key == "grades_judge" and not any(r.get("grades_judge") for r in rows):
            aurocs[glabel] = None
            continue
        labels = [_majority_wrong(r, grade_key) for r in rows]
        aurocs[glabel] = auroc_guarded(sem, labels)
    return {
        "n": len(rows),
        "mean_entropy": statistics.mean(sem) if sem else None,
        "n_clusters_spread": Counter(n_clusters),
        "saturated": saturated,
        "distinct": len(set(sem)),
        "tie_block": max(Counter(sem).values()) if sem else 0,
        "aurocs": aurocs,
    }


def _fmt_delta(a: Optional[float], b: Optional[float], pct: bool = False) -> str:
    if a is None or b is None:
        return "n/a"
    d = b - a
    return f"{d:+.1%}" if pct else f"{d:+.3f}"


def report_headline(baseline: Dict[str, dict], branch: Dict[str, dict],
                    uids: List[str]) -> None:
    section("Headline movement — before (baseline) vs after (branch)")
    base_scored = _scorable([baseline[u] for u in uids])
    branch_scored = _scorable([branch[u] for u in uids])
    print(f"  scorable questions: baseline {len(base_scored)}  branch {len(branch_scored)}  "
          f"(of {len(uids)} common)")

    sb, sa = _headline_stats(base_scored), _headline_stats(branch_scored)

    print(f"\n  {'metric':<38} {'baseline':>14} {'branch':>14} {'delta':>10}")
    me_b = f"{sb['mean_entropy']:.3f}" if sb['mean_entropy'] is not None else "n/a"
    me_a = f"{sa['mean_entropy']:.3f}" if sa['mean_entropy'] is not None else "n/a"
    print(f"  {'mean semantic entropy':<38} {me_b:>14} {me_a:>14} "
          f"{_fmt_delta(sb['mean_entropy'], sa['mean_entropy']):>10}")

    sat_b_pct = sb["saturated"] / max(sb["n"], 1)
    sat_a_pct = sa["saturated"] / max(sa["n"], 1)
    sat_b_str = f"{sb['saturated']}/{sb['n']} ({sat_b_pct:.1%})"
    sat_a_str = f"{sa['saturated']}/{sa['n']} ({sat_a_pct:.1%})"
    print(f"  {'at max cluster count (all singletons)':<38} {sat_b_str:>14} {sat_a_str:>14} "
          f"{_fmt_delta(sat_b_pct, sat_a_pct, pct=True):>10}")

    print(f"  {'distinct semantic-entropy values':<38} {sb['distinct']:>14} {sa['distinct']:>14} "
          f"{sa['distinct'] - sb['distinct']:>+10}")
    tb_b = f"{sb['tie_block']}/{sb['n']}"
    tb_a = f"{sa['tie_block']}/{sa['n']}"
    print(f"  {'largest tie block':<38} {tb_b:>14} {tb_a:>14} "
          f"{sa['tie_block'] - sb['tie_block']:>+10}")

    for glabel in ("NLI grader", "LLM judge"):
        vb, va = sb["aurocs"].get(glabel), sa["aurocs"].get(glabel)
        if vb is None and va is None:
            continue
        ab, whyb = vb if vb else (None, "n/a")
        aa, whya = va if va else (None, "n/a")
        ab_str = f"{ab:.3f}" if ab is not None else f"n/a ({whyb})"
        aa_str = f"{aa:.3f}" if aa is not None else f"n/a ({whya})"
        print(f"  {'AUROC vs majority-wrong [' + glabel + ']':<38} {ab_str:>14} {aa_str:>14} "
              f"{_fmt_delta(ab, aa):>10}")

    print(f"\n  n_clusters spread — baseline: " +
          ", ".join(f"{k}->{sb['n_clusters_spread'][k]}" for k in sorted(sb['n_clusters_spread'])))
    print(f"  n_clusters spread — branch  : " +
          ", ".join(f"{k}->{sa['n_clusters_spread'][k]}" for k in sorted(sa['n_clusters_spread'])))


# ---------------------------------------------------------------------------
# Named cases
# ---------------------------------------------------------------------------

def _print_case(row: Optional[dict], label: str) -> None:
    if row is None:
        print(f"    [{label}] not present in this file")
        return
    p = row.get("panels", {}).get(PANEL_KEY)
    print(f"    [{label}] n_clusters={p.get('n_clusters') if p else 'n/a'}  "
          f"cluster_sizes={p.get('cluster_sizes') if p else 'n/a'}")
    gj = row.get("grades_judge") or {}
    for name, cid in sorted((p or {}).get("cluster_of", {}).items()):
        nli_grade = row["grades"].get(name)
        judge_grade = gj.get(name)
        text = row["responses"].get(name, "").strip()[:80]
        print(f"      cluster {cid}  nli={nli_grade!s:<5} judge={judge_grade!s:<5}  "
              f"{short(name):<26} {text!r}")


def report_named_cases(baseline: Dict[str, dict], branch: Dict[str, dict],
                       uids: List[str]) -> None:
    section("Named cases")
    print("  simpleqa-1315 is deliberately NOT in this list. At the current nine-agent")
    print("  core pool NLI already splits it correctly (verified: 5 clusters, unchanged")
    print("  before/after the branch) — there is nothing for the branch to fix. The")
    print("  earlier description of it as a merge failure was written from a")
    print("  four-agent run; it does not hold at nine agents. Not a branch defect.")
    for uid in uids:
        print(f"\n  {uid}  Q: {(baseline.get(uid) or branch.get(uid) or {}).get('question', '')[:100]}")
        _print_case(baseline.get(uid), "before")
        _print_case(branch.get(uid), "after ")
        if uid == "simpleqa-0716":
            print("    [note] correctly out of scope, by design: this is a single-entity")
            print("    comparison (\"equilux\" vs \"equinox\"), and the >=2-entity threshold")
            print("    that exists specifically to prevent the 'New Guinea' / 'Papua New")
            print("    Guinea' false split necessarily excludes single-entity answers too.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate --short-answer-keys against a baseline results file.")
    ap.add_argument("--baseline", required=True,
                    help="results JSONL from a run WITHOUT --short-answer-keys")
    ap.add_argument("--branch", required=True,
                    help="results JSONL from a run WITH --short-answer-keys")
    ap.add_argument("--n-samples", type=int, default=20,
                    help="Bucket C disagreement pairs to print for manual adjudication")
    ap.add_argument("--top-n-reductions", type=int, default=10,
                    help="largest over-merge reductions to print with responses")
    ap.add_argument("--named-cases", default=",".join(DEFAULT_NAMED_CASES),
                    help="comma-separated uids to print cluster assignments for")
    args = ap.parse_args()

    baseline = load_rows(Path(args.baseline))
    branch = load_rows(Path(args.branch))

    uids = sorted(set(baseline) & set(branch))
    if not uids:
        print(f"\n  [ABORT] no uid in common between {args.baseline} and {args.branch}.\n")
        sys.exit(1)
    print(f"  {len(uids)} question(s) in common "
          f"(baseline has {len(baseline)}, branch has {len(branch)})")
    if len(uids) < len(baseline) or len(uids) < len(branch):
        print(f"  [!] comparing over the INTERSECTION only — the two files don't cover "
              f"exactly the same questions.")
    if not any(branch[u].get("grades_judge") or baseline[u].get("grades_judge") for u in uids):
        print(f"  [!] neither file has grades_judge for these questions — re-run with "
              f"--judge-model or the three-bucket analysis and over-merge metric below "
              f"will be empty.")

    branch, n_backfilled = backfill_judge_grades(baseline, branch, uids)
    if n_backfilled:
        print(f"  [i] backfilled grades_judge from the baseline onto {n_backfilled}/"
              f"{len(uids)} branch row(s) with matching response text (the branch run "
              f"itself had no --judge-model) — every function below now reads "
              f"row['grades_judge'] directly.")

    bucket_a, bucket_b, bucket_c = classify_pairs(baseline, branch, uids)
    report_three_buckets(bucket_a, bucket_b, bucket_c, n_samples=args.n_samples)
    report_overmerge(baseline, branch, uids, top_n=args.top_n_reductions)
    report_coverage(branch, uids)
    report_headline(baseline, branch, uids)
    report_named_cases(baseline, branch,
                       [u.strip() for u in args.named_cases.split(",") if u.strip()])

    section("Done")


if __name__ == "__main__":
    main()
