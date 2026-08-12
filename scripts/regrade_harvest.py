"""Re-grade a scripts/harvest.py .responses.jsonl sidecar with the current
NLI-based grader — no API calls, no re-running the harvest.

Why this exists
----------------
evaluation.grade_correct's open-ended path used to check a response only
against TruthfulQASample.correct_answer (one canonical string). It now checks
the full TruthfulQASample.correct_answers list (median 3, max 14 accepted
phrasings) — recovering enumerative answers ("Turkey, Ireland, the UK and
China consume more tea") that entail a *different* valid subset than the
single gold string and so scored neutral against it. A harvest run before
that fix has stale, over-pessimistic `grade` values baked into its sidecar.
This script recomputes them in place (into a NEW file — the original is left
untouched) using whichever grader is on PATH now, without spending a single
API call: the response TEXT is already in the sidecar, so only NLI inference
(local model, not the harvest) is needed.

Not scripts/regrade.py
-----------------------
That script re-grades scripts/pilot.py-style output with an LLM judge (Gemini
/ GPT-OSS). This one re-grades scripts/harvest.py's .responses.jsonl sidecar
with the same NLI grader the harvest itself used — different input format,
different judge, different purpose. Kept as a separate script rather than
overloading regrade.py's CLI with a second, incompatible mode.

How prompt_format/source are found
------------------------------------
Per-response rows don't carry them — they're read from the harvest's own
<stem>.manifest.json sidecar, which --file's basename must resolve to (see
scripts/harvest.py: <out>.csv / <out-stem>.responses.jsonl /
<out-stem>.manifest.json all share one stem).

Run from the project root:
    python scripts/regrade_harvest.py --file outputs/harvest.responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa                                        # noqa: E402
from disagreement import CrossEncoderNLI                                 # noqa: E402
from evaluation import grade_correct                                     # noqa: E402


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
RESPONSES_SUFFIX = ".responses.jsonl"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _grade_label(g: Optional[bool]) -> str:
    return {True: "correct", False: "incorrect", None: "unclear"}[g]


def _manifest_path_for(responses_path: Path) -> Path:
    name = responses_path.name
    if not name.endswith(RESPONSES_SUFFIX):
        raise SystemExit(
            f"[ABORT] {responses_path} doesn't look like a harvest responses "
            f"sidecar (expected a name ending in '{RESPONSES_SUFFIX}')."
        )
    stem = name[: -len(RESPONSES_SUFFIX)]
    return responses_path.with_name(f"{stem}.manifest.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-grade a harvest .responses.jsonl sidecar with the current grader."
    )
    ap.add_argument("--file", required=True,
                     help="path to an existing <stem>.responses.jsonl from scripts/harvest.py")
    ap.add_argument("--out", default=None,
                     help="output path (default: <stem>.regraded.responses.jsonl, "
                          "next to --file)")
    ap.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    args = ap.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        raise SystemExit(f"[ABORT] No such file: {in_path}")
    manifest_path = _manifest_path_for(in_path)
    if not manifest_path.exists():
        raise SystemExit(
            f"[ABORT] {manifest_path} not found — prompt_format/source can't be "
            f"determined without the harvest's manifest sidecar."
        )
    out_path = (Path(args.out) if args.out
                else in_path.with_name(in_path.name.replace(
                    RESPONSES_SUFFIX, f".regraded{RESPONSES_SUFFIX}")))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompt_format = manifest["prompt_format"]
    source = manifest["source"]

    section("Loading")
    print(f"  sidecar   : {in_path}")
    print(f"  manifest  : {manifest_path}  (prompt_format={prompt_format!r}, source={source!r})")
    rows = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {len(rows)} response row(s)")

    print(f"  Reloading TruthfulQA (source={source}) for gold answers...")
    by_uid = {s.uid: s for s in load_truthfulqa(n=None, source=source)}

    print(f"  NLI model : {args.nli_model} (loads lazily on first use; local, no API calls)")
    nli = CrossEncoderNLI(model_name=args.nli_model)

    section("Re-grading")
    before: Dict[str, int] = {"correct": 0, "incorrect": 0, "unclear": 0}
    after: Dict[str, int] = {"correct": 0, "incorrect": 0, "unclear": 0}
    n_unmatched = 0
    transitions: Dict[str, int] = {}

    for i, row in enumerate(rows, 1):
        old_grade = row.get("grade")
        before[_grade_label(old_grade)] += 1

        sample = by_uid.get(row["uid"])
        if sample is None:
            # Same failure mode data/truthfulqa.py already tolerates for
            # CSV/mc_task.json drift: skip rather than crash, count it.
            n_unmatched += 1
            after[_grade_label(old_grade)] += 1
            row["grade"] = old_grade
            continue

        new_grade = grade_correct(nli, row.get("text") or "", sample, prompt_format=prompt_format)
        after[_grade_label(new_grade)] += 1
        row["grade"] = new_grade

        if new_grade != old_grade:
            key = f"{_grade_label(old_grade)} -> {_grade_label(new_grade)}"
            transitions[key] = transitions.get(key, 0) + 1

        if i % 500 == 0:
            print(f"  ... {i}/{len(rows)}")

    if n_unmatched == len(rows):
        raise SystemExit(
            f"[ABORT] 0/{len(rows)} rows matched a TruthfulQA uid under "
            f"source={source!r} — this looks like a config mismatch, not data "
            f"drift. Check --file / the manifest's 'source'."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    section("Grade distribution: before -> after")
    for k in ("correct", "incorrect", "unclear"):
        print(f"  {k:<10} {before[k]:>5}  ->  {after[k]:>5}")
    if n_unmatched:
        print(f"\n  [!] {n_unmatched} row(s) had no matching TruthfulQA uid "
              f"(source={source!r}) — left unchanged, not re-graded.")

    print(f"\n  Transitions (old -> new, count):")
    if transitions:
        for k, v in sorted(transitions.items(), key=lambda kv: -kv[1]):
            flag = "  <-- was already decided, now flipped" if "unclear ->" not in k else ""
            print(f"    {k:<28} {v:>5}{flag}")
    else:
        print("    (none — every grade stayed the same)")

    recovered = transitions.get("unclear -> correct", 0) + transitions.get("unclear -> incorrect", 0)
    print(f"\n  Recovered from unclear: {recovered} / {before['unclear']}")
    flipped_decided = sum(v for k, v in transitions.items()
                           if "unclear" not in k)
    if flipped_decided:
        print(f"  [!] {flipped_decided} response(s) that were ALREADY decided changed "
              f"grade (not just unclear->decided) — inspect before trusting downstream "
              f"numbers; this can happen if correct_answer isn't exactly one of "
              f"correct_answers' entries for some question.")

    section("Done")
    print(f"  Regraded sidecar -> {out_path}")
    print(f"  (original left untouched: {in_path})")


if __name__ == "__main__":
    main()
