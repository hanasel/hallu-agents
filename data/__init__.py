"""
Unified dataset interface for the hallucination detection project.

Quick start
-----------
    from data import load_truthfulqa, load_ragtruth

    # TruthfulQA — all 790 MC1 questions (Jan 2025 github source, default)
    tqa = load_truthfulqa()

    # TruthfulQA — 100-sample subset, specific categories
    tqa_sub = load_truthfulqa(n=100, categories=["Misconceptions", "Health"])

    # TruthfulQA — pre-2025 HuggingFace hub snapshot (817 questions), for
    # reproducing older runs
    tqa_legacy = load_truthfulqa(source="hf")

    # RAGTruth test split — QA task only
    rgt = load_ragtruth(split="test", task_types=["qa"])

    # RAGTruth grouped by source (mirrors multi-agent structure)
    grouped = load_ragtruth_by_source(split="test", n=200)

    # SimpleQA Verified — 1,000 short-form factuality questions
    sqa = load_simpleqa(n=100, seed=42)

    # Per-dataset agent-query settings (max_tokens/system_prompt/temperature)
    # — agents.GroqAgent itself stays dataset-neutral; pass these explicitly.
    from data import TRUTHFULQA_QUERY_CONFIG, RAGTRUTH_QUERY_CONFIG, SIMPLEQA_QUERY_CONFIG
    agent = GroqAgent(**TRUTHFULQA_QUERY_CONFIG)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .schema import Sample, DatasetName, TaskType
from .truthfulqa import TruthfulQALoader, TruthfulQASample
from .ragtruth import RAGTruthLoader, RAGTruthSample, HallucinationSpan
from .simpleqa import SimpleQAVerifiedLoader, SimpleQAVerifiedSample
from .query_config import TRUTHFULQA_QUERY_CONFIG, RAGTRUTH_QUERY_CONFIG, SIMPLEQA_QUERY_CONFIG


def load_truthfulqa(
    n: Optional[int] = None,
    seed: int = 42,
    categories: Optional[list[str]] = None,
    question_types: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
    source: str = "github",
) -> list[TruthfulQASample]:
    """Load TruthfulQA MC1. Optionally subset and/or filter by category.

    `source="github"` (default) uses the Jan 2025 sylinrl/TruthfulQA update
    (790 questions, 37 categories). `source="hf"` uses the frozen pre-2025
    HuggingFace hub snapshot (817 questions, 38 categories), for reproducing
    older runs.
    """
    return TruthfulQALoader(cache_dir=cache_dir, source=source).load(
        n=n, seed=seed, categories=categories, question_types=question_types
    )


def load_ragtruth(
    split: str = "test",
    n: Optional[int] = None,
    seed: int = 42,
    task_types: Optional[list[str]] = None,
    source_models: Optional[list[str]] = None,
    hallucinated_only: bool = False,
    clean_only: bool = False,
    quality: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> list[RAGTruthSample]:
    """Load RAGTruth. Defaults to the test split (2,700 responses)."""
    loader = RAGTruthLoader(data_dir=cache_dir) if cache_dir is not None else RAGTruthLoader()
    return loader.load(
        split=split, n=n, seed=seed,
        task_types=task_types, source_models=source_models,
        hallucinated_only=hallucinated_only, clean_only=clean_only,
        quality=quality,
    )


def load_ragtruth_by_source(
    split: str = "test",
    cache_dir: Optional[str] = None,
    **kwargs,
) -> dict[str, list[RAGTruthSample]]:
    """
    Load RAGTruth grouped by source_id.
    Each key maps to up to 6 model responses for the same input.
    """
    loader = RAGTruthLoader(data_dir=cache_dir) if cache_dir is not None else RAGTruthLoader()
    return loader.load_by_source(split=split, **kwargs)


def load_simpleqa(
    n: Optional[int] = None,
    seed: int = 42,
    topics: Optional[list[str]] = None,
    answer_types: Optional[list[str]] = None,
    multi_step: Optional[bool] = None,
    requires_reasoning: Optional[bool] = None,
    cache_dir: Optional[str] = None,
) -> list[SimpleQAVerifiedSample]:
    """Load SimpleQA Verified. Optionally subset and/or filter.

    1,000 short-form factuality questions (Google DeepMind's revised,
    de-duplicated, error-corrected version of OpenAI's SimpleQA). See
    `data.simpleqa` for details.
    """
    return SimpleQAVerifiedLoader(cache_dir=cache_dir).load(
        n=n, seed=seed, topics=topics, answer_types=answer_types,
        multi_step=multi_step, requires_reasoning=requires_reasoning,
    )


def print_dataset_stats() -> None:
    """Print summary statistics for both datasets (requires network)."""
    print("=" * 60)
    print("TruthfulQA")
    print("=" * 60)
    loader_tqa = TruthfulQALoader()
    samples_tqa = loader_tqa.load()
    by_cat = loader_tqa.load_by_category()
    print(f"  Total questions : {len(samples_tqa)}")
    print(f"  Categories      : {len(by_cat)}")
    print(f"\n  Top 10 categories by size:")
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"    {cat:<35} {len(items):>3} questions")

    print()
    print("=" * 60)
    print("RAGTruth (test split)")
    print("=" * 60)
    stats = RAGTruthLoader().stats(split="test")
    print(f"  Total responses     : {stats['total']}")
    print(f"  Hallucinated        : {stats['hallucinated']} ({stats['hallucination_rate']:.1%})")
    print(f"\n  By task type:")
    for task, count in sorted(stats["by_task"].items()):
        print(f"    {task:<20} {count:>5} responses")
    print(f"\n  By source model:")
    for model, count in sorted(stats["by_model"].items()):
        print(f"    {model:<30} {count:>5} responses")

    print()
    print("=" * 60)
    print("SimpleQA Verified")
    print("=" * 60)
    loader_sqa = SimpleQAVerifiedLoader()
    samples_sqa = loader_sqa.load()
    by_topic = loader_sqa.load_by_topic()
    print(f"  Total questions : {len(samples_sqa)}")
    print(f"  Topics          : {len(by_topic)}")
    print(f"\n  Topics by size:")
    for topic, items in sorted(by_topic.items(), key=lambda x: -len(x[1])):
        print(f"    {topic:<35} {len(items):>3} questions")


def export_to_jsonl(samples, path: str | Path) -> Path:
    """Serialise samples to JSONL for caching / inspection.

    Works for any object satisfying the `Sample` protocol (i.e. with a
    `to_dict()` method).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            if not hasattr(sample, "to_dict"):
                raise TypeError(
                    f"Cannot serialise {type(sample).__name__}: "
                    f"missing to_dict() method. Object must satisfy the "
                    f"data.schema.Sample protocol."
                )
            fh.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    print(f"Exported {len(samples)} samples to {path}")
    return path
