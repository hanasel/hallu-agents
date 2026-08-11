#!/usr/bin/env python3

"""
quickstart.py — validate that dataset loading works end-to-end.

Run from the project root:
    python quickstart.py

This script:
  1. Loads a small subset of TruthfulQA and prints sample prompts
  2. Loads a small subset of RAGTruth and prints sample prompts
  3. Exports both subsets to JSONL for inspection
  4. Prints dataset statistics

It's intended as a smoke test before running the full pipeline.
No API keys required — this only touches the HuggingFace Hub.
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from data import (
    load_truthfulqa,
    load_ragtruth,
    load_ragtruth_by_source,
    export_to_jsonl,
)


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main():
    # ------------------------------------------------------------------ #
    # 1. TruthfulQA
    # ------------------------------------------------------------------ #
    section("TruthfulQA — loading 10 MC1 questions")
    tqa_samples = load_truthfulqa(n=10, seed=0)
    print(f"Loaded {len(tqa_samples)} samples\n")

    sample = tqa_samples[0]
    print(f"Category       : {sample.category}")
    print(f"Question ID    : {sample.question_id}")
    print(f"Correct answer : [{sample.correct_letter}] {sample.correct_answer}")
    print(f"\n--- Prompt ---")
    print(sample.to_prompt())

    # ------------------------------------------------------------------ #
    # 2. TruthfulQA by category
    # ------------------------------------------------------------------ #
    section("TruthfulQA — filtering to 'Misconceptions' category")
    misconceptions = load_truthfulqa(categories=["Misconceptions"])
    print(f"Found {len(misconceptions)} Misconceptions questions")
    if misconceptions:
        print(f"Example: {misconceptions[0].question}")

    # ------------------------------------------------------------------ #
    # 3. RAGTruth test split
    # ------------------------------------------------------------------ #
    section("RAGTruth — loading 10 QA samples from test split")
    rgt_samples = load_ragtruth(split="test", task_types=["qa"], n=10, seed=0)
    print(f"Loaded {len(rgt_samples)} samples\n")

    rgt_sample = rgt_samples[0]
    print(f"Sample ID      : {rgt_sample.sample_id}")
    print(f"Source model   : {rgt_sample.source_model}")
    print(f"Task type      : {rgt_sample.task_type}")
    print(f"Hallucinated   : {rgt_sample.is_hallucinated}")
    print(f"Spans          : {len(rgt_sample.hallucination_spans)}")
    if rgt_sample.hallucination_spans:
        sp = rgt_sample.hallucination_spans[0]
        print(f"First span     : [{sp.label_type}] '{sp.text[:80]}...'")
    print(f"\n--- Prompt (truncated) ---")
    print(rgt_sample.to_rag_prompt()[:400] + "...")

    # ------------------------------------------------------------------ #
    # 4. RAGTruth grouped by source (multi-agent structure)
    # ------------------------------------------------------------------ #
    section("RAGTruth — grouped by source_id (n=60)")
    grouped = load_ragtruth_by_source(split="test", n=60, seed=0)
    print(f"Unique source IDs : {len(grouped)}")
    example_sid = next(iter(grouped))
    group = grouped[example_sid]
    print(f"Example source    : {example_sid}")
    print(f"  Models in group : {[s.source_model for s in group]}")
    print(f"  Hallucinated    : {[s.is_hallucinated for s in group]}")

    # ------------------------------------------------------------------ #
    # 5. Export to JSONL
    # ------------------------------------------------------------------ #
    section("Exporting to JSONL")
    tqa_path = export_to_jsonl(tqa_samples, "outputs/tqa_sample.jsonl")
    rgt_path = export_to_jsonl(rgt_samples, "outputs/rgt_sample.jsonl")
    print(f"\nInspect with:")
    print(f"  cat {tqa_path} | python -m json.tool | head -60")
    print(f"  cat {rgt_path} | python -m json.tool | head -60")

    # ------------------------------------------------------------------ #
    # Done
    # ------------------------------------------------------------------ #
    section("All checks passed")
    print("Dataset loading layer is working correctly.")
    print("Next step: implement the agent querying layer (agents/).\n")


if __name__ == "__main__":
    main()
