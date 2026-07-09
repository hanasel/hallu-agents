"""
TruthfulQA dataset loader.

Loads the MC1 split of TruthfulQA from the Hugging Face Hub and returns
clean, validated dataclasses ready for the agent querying pipeline.

MC1 format: one correct answer among several options (single-label).
MC2 is deliberately excluded — multiple correct answers inflate the
false-positive rate of any disagreement-based signal.

Prompt formats
--------------
Ground truth is always MC1 (one correct answer per question). Two prompt
formats are supported for querying agents:

  - mc_prompt()   — multiple-choice with A/B/C/D options, asks for a letter.
                    Enables clean exact-match evaluation for answer-level
                    disagreement, but produces single-letter responses
                    that are useless for semantic clustering or
                    claim-level decomposition.

  - open_prompt() — just the bare question. Produces free-text responses
                    suitable for all three disagreement measures
                    (answer, semantic, claim-level). Requires fuzzy
                    matching against the correct/incorrect answer lists
                    for answer-level evaluation.

The `.prompt` property returns `open_prompt()` by default, since open-ended
responses are needed for two of the three disagreement measures in RQ2.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TruthfulQASample:
    """A single TruthfulQA MC1 question.

    Implements the `data.schema.Sample` protocol via the `uid`, `prompt`,
    and `dataset` properties below, so this class can be used uniformly
    by dataset-agnostic code (e.g. the agent-querying layer).
    """

    question_id: int
    question: str
    category: str

    # MC1: one correct choice, several incorrect choices
    correct_answer: str
    incorrect_answers: list[str]

    # All choices shuffled together (so agent order is randomised)
    choices: list[str]
    correct_index: int          # index into `choices` of the correct answer

    def mc_prompt(self) -> str:
        """Multiple-choice format: question + A/B/C/D options, asks for a letter.

        Produces single-token responses ('A', 'B', ...) suitable for
        answer-level disagreement via exact letter match. Not useful for
        semantic or claim-level measures — nothing to cluster or decompose.
        """
        lines = [f"Question: {self.question}", "Options:"]
        for i, choice in enumerate(self.choices):
            lines.append(f"  {chr(65 + i)}. {choice}")
        lines.append("\nAnswer with the letter of the correct option only (e.g. 'A').")
        return "\n".join(lines)

    def open_prompt(self) -> str:
        """Open-ended format: just the bare question.

        Produces free-text responses suitable for all three disagreement
        measures. No framing is added — the chat model's own instruction
        template handles that. Downstream answer-level evaluation requires
        fuzzy matching against the correct/incorrect answer lists.
        """
        return self.question

    def to_prompt(self, include_choices: bool = True) -> str:
        """Backward-compatible prompt formatter.

        `include_choices=True` returns `mc_prompt()`; False returns
        `open_prompt()`. Kept so existing code (e.g. quickstart.py) that
        uses this method continues to work; prefer the explicit
        `mc_prompt()` / `open_prompt()` methods in new code.
        """
        return self.mc_prompt() if include_choices else self.open_prompt()

    @property
    def correct_letter(self) -> str:
        return chr(65 + self.correct_index)

    # ------------------------------------------------------------------
    # Sample protocol (see data.schema.Sample) — uniform interface
    # ------------------------------------------------------------------

    @property
    def uid(self) -> str:
        """Stable cross-dataset identifier, e.g. 'truthfulqa-0042'."""
        return f"truthfulqa-{self.question_id:04d}"

    @property
    def prompt(self) -> str:
        """Default prompt for agent querying — open-ended (bare question).

        Returns `open_prompt()` because two of the three disagreement
        measures (semantic, claim-level) require free-text responses.
        For explicit multiple-choice querying, call `mc_prompt()` directly.
        """
        return self.open_prompt()

    @property
    def dataset(self) -> str:
        return "truthfulqa"

    def to_dict(self) -> dict:
        """Serialise to a JSONL-friendly dict."""
        return {
            "type": self.dataset,
            "uid": self.uid,
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "correct_answer": self.correct_answer,
            "incorrect_answers": list(self.incorrect_answers),
            "choices": list(self.choices),
            "correct_index": self.correct_index,
            "correct_letter": self.correct_letter,
            "prompt": self.prompt,           # what agents will actually see (open-ended)
            "mc_prompt": self.mc_prompt(),   # both variants exposed for inspection
            "open_prompt": self.open_prompt(),
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TruthfulQALoader:
    """
    Loads TruthfulQA MC1 from the Hugging Face datasets library.

    Usage
    -----
    >>> loader = TruthfulQALoader()
    >>> samples = loader.load()                       # all 817 questions
    >>> subset = loader.load(n=100, seed=42)          # reproducible 100-sample subset
    >>> by_cat = loader.load_by_category()            # dict[category -> list[sample]]

    Dataset card: https://huggingface.co/datasets/truthful_qa
    """

    HF_DATASET_NAME = "truthful_qa"
    HF_CONFIG = "multiple_choice"          # contains both MC1 and MC2 splits

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir
        self._raw: Optional[object] = None   # holds the HF DatasetDict once loaded
        self._cat_by_question: Optional[dict[str, str]] = None  # question -> category

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        n: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = False,
        categories: Optional[list[str]] = None,
    ) -> list[TruthfulQASample]:
        """
        Return a list of TruthfulQASample objects.

        Parameters
        ----------
        n          : if set, return a random subset of this size
        seed       : random seed for subsetting / shuffling
        shuffle    : shuffle the full dataset before subsetting
        categories : if set, filter to these categories only
        """
        raw = self._get_raw()
        # TruthfulQA only has a validation split in HF
        split = raw["validation"]

        samples = [self._parse_row(i, row) for i, row in enumerate(split)]

        if categories:
            samples = [s for s in samples if s.category in categories]

        if shuffle or n is not None:
            rng = random.Random(seed)
            samples = samples.copy()
            rng.shuffle(samples)

        if n is not None:
            samples = samples[:n]

        return samples

    def load_by_category(self, **kwargs) -> dict[str, list[TruthfulQASample]]:
        """Return samples grouped by TruthfulQA category."""
        samples = self.load(**kwargs)
        out: dict[str, list[TruthfulQASample]] = {}
        for s in samples:
            out.setdefault(s.category, []).append(s)
        return out

    def available_categories(self) -> list[str]:
        """Return the 38 TruthfulQA categories, sorted."""
        return sorted({s.category for s in self.load()})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw(self):
        if self._raw is not None:
            return self._raw
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required. Install it with:\n"
                "  pip install datasets"
            ) from exc

        # The MC1/MC2 config (`multiple_choice`) does NOT include the
        # `category` field — categories are only present in the `generation`
        # config. Load both and build a question-text -> category lookup
        # so we can attach categories to MC1 rows.
        self._raw = load_dataset(
            self.HF_DATASET_NAME,
            self.HF_CONFIG,
            cache_dir=self._cache_dir,
        )
        gen = load_dataset(
            self.HF_DATASET_NAME,
            "generation",
            cache_dir=self._cache_dir,
        )
        self._cat_by_question = {
            row["question"]: row.get("category", "Unknown")
            for row in gen["validation"]
        }
        return self._raw

    def _parse_row(self, idx: int, row: dict) -> TruthfulQASample:
        """Convert one raw HF row into a TruthfulQASample (MC1 only)."""
        question = row["question"]
        # MC1 rows don't carry `category` directly — look it up via the map
        # built from the `generation` config in _get_raw().
        category = "Unknown"
        if self._cat_by_question is not None:
            category = self._cat_by_question.get(question, "Unknown")

        # MC1 structure: {"choices": [...], "labels": [0/1, ...]}
        # Exactly one label == 1 (correct), rest are 0.
        mc1 = row["mc1_targets"]
        choices_raw: list[str] = mc1["choices"]
        labels: list[int] = mc1["labels"]

        correct_idx_raw = labels.index(1)
        correct_answer = choices_raw[correct_idx_raw]
        incorrect_answers = [c for i, c in enumerate(choices_raw) if i != correct_idx_raw]

        # Shuffle choices so correct answer isn't always first
        rng = random.Random(idx)          # deterministic per question
        shuffled = choices_raw.copy()
        rng.shuffle(shuffled)
        correct_index = shuffled.index(correct_answer)

        return TruthfulQASample(
            question_id=idx,
            question=question,
            category=category,
            correct_answer=correct_answer,
            incorrect_answers=incorrect_answers,
            choices=shuffled,
            correct_index=correct_index,
        )