"""Decomposition census — how many atomic claims does each response actually yield?

Every response in the last run reported exactly 12 claims, which is the
--max-claims cap, not a property of the data. That means verification only ever
saw the OPENING of each response and any hallucination later in the text was
invisible to the method — a likely cap on recall and therefore on every AUC.

This script runs decomposition ONLY (one call per response, no judges), with an
effectively unlimited cap, and reports the true claim-count distribution plus a
cost projection so you can pick --max-claims deliberately.

Cache note
----------
The decomposition prompt, system prompt and temperature here are imported from
rgt_verify_pilot, so the cache keys are IDENTICAL. Every decomposition this
census performs replays for free in the real run — and any already done by
previous runs replays for free here. (max_claims is applied when parsing the
reply, not in the prompt, so it does not affect the cache key.)

Run:
    python scripts/rgt_decompose_census.py --task data2txt --n 150
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # import sibling script

import numpy as np                                                     # noqa: E402

from data import load_ragtruth                                        # noqa: E402
from agents.panels import make_agent, GPT_OSS_LARGE                   # noqa: E402
# Import the exact prompt + decompose() the pilot uses, so cache keys match.
from rgt_verify_pilot import DECOMPOSER_SYSTEM, decompose, section    # noqa: E402

NO_CAP = 500          # effectively unlimited; reveals the true count


def main() -> None:
    ap = argparse.ArgumentParser(description="RAGTruth decomposition census.")
    ap.add_argument("--task", default="data2txt",
                    choices=["qa", "summarization", "data2txt"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decomposer", default=GPT_OSS_LARGE)
    ap.add_argument("--n-judges", type=int, default=5,
                    help="panel size, for the cost projection only")
    ap.add_argument("--out", default="outputs/rgt_claim_census.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth {args.task.upper()} [test] — {args.n} responses")
    samples = load_ragtruth(split="test", task_types=[args.task],
                            n=args.n, seed=args.seed, quality="good")
    print(f"  Loaded {len(samples)} responses.")

    decomposer = make_agent(args.decomposer, system_prompt=DECOMPOSER_SYSTEM)
    print(f"  decomposer: {decomposer.name}")
    print(f"  (1 call per response — ~{len(samples)} calls if uncached)")

    section("Decomposing")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    aborted = False
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, s in enumerate(samples, start=1):
            claims = decompose(decomposer, s.response, NO_CAP)
            if claims is None:
                print(f"  [{idx}/{len(samples)}] {s.uid}  DECOMPOSER ERROR "
                      f"(likely quota) — stopping; cache preserved.")
                aborted = True
                break
            rec = {
                "uid": s.uid,
                "source_model": s.source_model,
                "is_hallucinated": s.is_hallucinated,
                "n_claims": len(claims),
                "response_words": len(s.response.split()),
                "claims": claims,
            }
            records.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if idx % 25 == 0 or idx == len(samples):
                print(f"  [{idx}/{len(samples)}] ...")

    if not records:
        print("  Nothing decomposed.")
        return

    counts = np.array([r["n_claims"] for r in records])

    section("Claim-count distribution")
    if aborted:
        print(f"  [!] INCOMPLETE — {len(records)}/{len(samples)} decomposed.\n")
    print(f"  responses      : {len(counts)}")
    print(f"  mean / median  : {counts.mean():.1f} / {np.median(counts):.0f}")
    print(f"  min / max      : {counts.min()} / {counts.max()}")
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p:<13}: {np.percentile(counts, p):.0f}")

    print("\n  histogram:")
    edges = [0, 1, 5, 10, 12, 15, 20, 25, 30, 40, 60, 10**9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = int(((counts >= lo) & (counts < hi)).sum())
        if k:
            bar = "#" * min(50, int(50 * k / len(counts)))
            label = f"{lo}-{hi-1}" if hi < 10**9 else f"{lo}+"
            print(f"    {label:>8} | {bar} {k}")

    section("Truncation under the old cap")
    for cap in (12, 15, 20, 25, 30):
        hit = int((counts > cap).sum())
        lost = int(np.clip(counts - cap, 0, None).sum())
        total = int(counts.sum())
        print(f"  cap={cap:<3} responses truncated: {hit:>4} ({hit/len(counts):>5.1%})   "
              f"claims lost: {lost:>5} ({lost/total:>5.1%} of all claims)")
    print("\n  Claims lost are claims never verified — hallucinations there are")
    print("  invisible to the method, capping recall and depressing every AUC.")

    section("Cost projection (per-claim mode)")
    print(f"  panel size {args.n_judges}; calls = n x (1 + n_judges x min(claims, cap))")
    for cap in (12, 20, 25, 30):
        eff = np.minimum(counts, cap)
        per_run = len(counts) + args.n_judges * int(eff.sum())
        scaled = per_run / len(counts) * 150
        print(f"  cap={cap:<3} this set ({len(counts)}): {per_run:>7,} calls   "
              f"| projected n=150: {scaled:>8,.0f} calls")

    section("Diagnostics")
    zero = int((counts == 0).sum())
    if zero:
        print(f"  [!] {zero} responses produced ZERO claims — inspect these; they")
        print(f"      are excluded from scoring by the pilot.")
    ceiling = int((counts >= 40).sum())
    if ceiling:
        print(f"  [!] {ceiling} responses at 40+ claims. If counts pile up at one")
        print(f"      value, the DECOMPOSER's max_tokens is truncating, not the data.")

    words = np.array([r["response_words"] for r in records])
    if len(counts) > 2:
        from scipy.stats import spearmanr
        rho, p = spearmanr(counts, words)
        print(f"\n  Spearman(n_claims, response_words): rho={rho:+.3f} p={p:.3g}")

    hal = counts[[r["is_hallucinated"] for r in records]]
    cln = counts[[not r["is_hallucinated"] for r in records]]
    if len(hal) and len(cln):
        print(f"  mean claims | hallucinated : {hal.mean():.1f}")
        print(f"  mean claims | clean        : {cln.mean():.1f}")
        print("  (a large gap means claim COUNT alone leaks the label — worth")
        print("   reporting as a baseline so signals aren't just counting claims)")

    print(f"\n  Claims written to: {out_path}\n")


if __name__ == "__main__":
    main()