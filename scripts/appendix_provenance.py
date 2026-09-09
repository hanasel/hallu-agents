#!/usr/bin/env python3
"""Fill the five remaining cells of Appendix Table H.2 (run provenance).

Prints, for each reported run: the git commit from its manifest, and the range
of response timestamps from the response cache. Results files carry no
timestamp -- only cached AgentResponse records do -- so the served date has to
come from the cache.

Run from the repository root:

    python appendix_provenance.py
    python appendix_provenance.py --cache cache/agent_responses --outputs outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import os


# results file -> the manifest that describes it
RUNS = [
    ("Nine-agent panel, TruthfulQA MC1", "outputs/truthfulqa_results_mc.jsonl"),
    ("Nine-agent panel, TruthfulQA open", "outputs/truthfulqa_results.jsonl"),
    ("Nine-agent panel, SimpleQA", "outputs/simpleqa_3x3_results_1000.jsonl"),
    ("Five-agent pilot, TruthfulQA MC1", "outputs/tqa790_mc.csv"),
    ("Five-agent pilot, TruthfulQA open", "outputs/tqa790_open.csv"),
    ("Verification, RAGTruth (data2txt)", "outputs/rgt_verify_together_data2txt.jsonl"),
    ("Verification, RAGTruth (summarisation)", "outputs/rgt_verify_together_summarization.jsonl"),
    ("Verification, RAGTruth (QA)", "outputs/rgt_verify_together_qa.jsonl"),
]


def manifest_path(results_path: str) -> str:
    root, _ = os.path.splitext(results_path)
    return root + ".manifest.json"


def read_manifest(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def cache_timestamp_range(cache_dir: str) -> dict[str, tuple[str, str, int]]:
    """Per cache file: (earliest timestamp, latest timestamp, n responses)."""
    out: dict[str, tuple[str, str, int]] = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*.jsonl"))):
        lo = hi = None
        n = 0
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = (rec.get("response") or {}).get("timestamp") or rec.get("timestamp")
                if not ts:
                    continue
                n += 1
                if lo is None or ts < lo:
                    lo = ts
                if hi is None or ts > hi:
                    hi = ts
        if n:
            out[os.path.basename(path)] = (lo, hi, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache/agent_responses")
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()

    print("=" * 72)
    print("  COMMITS  (git_commit field of each run's manifest)")
    print("=" * 72)
    for label, results in RUNS:
        results = results.replace("outputs/", args.outputs.rstrip("/") + "/", 1)
        man = read_manifest(manifest_path(results))
        if man is None:
            print(f"  {label:42s}  no manifest at {manifest_path(results)}")
            continue
        commit = man.get("git_commit") or "(absent from manifest)"
        n_req = man.get("n")
        judge = man.get("judge_model") or "-"
        nli = man.get("nli_model") or "-"
        print(f"  {label:42s}  {commit[:7]}   n_requested={n_req}")
        print(f"  {'':42s}  judge={judge}  nli={nli}")

    print()
    print("=" * 72)
    print("  SERVED DATES  (timestamp range per cached agent)")
    print("=" * 72)
    ranges = cache_timestamp_range(args.cache)
    if not ranges:
        print(f"  No timestamped cache entries under {args.cache!r}.")
        print("  Point --cache at the directory holding the per-agent .jsonl caches.")
        return
    for name, (lo, hi, n) in ranges.items():
        print(f"  {name:52s} {lo[:10]} .. {hi[:10]}  ({n} responses)")

    allts = [v for r in ranges.values() for v in (r[0], r[1])]
    print()
    print(f"  Overall harvest window: {min(allts)[:10]} .. {max(allts)[:10]}")
    print()
    print("  Note: the cache is keyed by model+prompt+settings, not by run, so a")
    print("  single agent's cache file spans every run it took part in. For a")
    print("  per-run date, filter the cache on the prompts belonging to that run,")
    print("  or quote the overall window above and say so in the caption.")


if __name__ == "__main__":
    main()