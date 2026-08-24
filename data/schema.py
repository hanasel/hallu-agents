"""Shared types and the common Sample interface for dataset loaders.

`TruthfulQASample`, `RAGTruthSample`, and `SimpleQAVerifiedSample` all satisfy
the `Sample` protocol, exposing a stable `uid`, a ready-to-send `prompt`, and
a `dataset` discriminator.

This keeps the rich, dataset-specific dataclasses (with their domain methods like
`is_hallucinated`, `hallucination_rate`, `correct_letter`) while letting the
agent-querying layer treat any sample uniformly.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Protocol, runtime_checkable

DatasetName = Literal["truthfulqa", "ragtruth", "simpleqa"]
TaskType = Literal["qa", "summarization", "data2txt"]


@runtime_checkable
class Sample(Protocol):
    """Minimum interface every dataset sample must satisfy.

    Implementations live in:
      - `data.truthfulqa.TruthfulQASample`
      - `data.ragtruth.RAGTruthSample`
      - `data.simpleqa.SimpleQAVerifiedSample`

    Use this as the type hint anywhere code should work across datasets
    (e.g. agent querying). Evaluation code that branches on dataset can
    check `sample.dataset` directly.
    """

    @property
    def uid(self) -> str:
        """Stable, globally unique identifier across datasets.

        Format: '<dataset>-<local_id>', e.g. 'truthfulqa-0042', 'ragtruth-abc123'.
        """
        ...

    @property
    def prompt(self) -> str:
        """Ready-to-send prompt string for agent querying."""
        ...

    @property
    def dataset(self) -> DatasetName:
        """Which dataset this sample originates from."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSONL-friendly dict for caching / inspection."""
        ...
