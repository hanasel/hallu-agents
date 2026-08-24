"""
SimpleQA Verified dataset loader.

Loads Google DeepMind's "SimpleQA Verified" benchmark — a de-duplicated,
error-corrected revision of OpenAI's SimpleQA (Wei et al. 2024), released
2025 (arXiv:2509.07968). 1,000 short-form factuality questions, each with a
single canonical gold answer, a topic, an answer type, and >=2 supporting
source URLs.

Source: https://huggingface.co/datasets/google/simpleqa-verified
(simpleqa_verified.csv, MIT licensed). Downloaded once into `cache_dir`
(default `data/simpleqa_data/`) and reused on subsequent runs — same
download-and-cache pattern as `data.truthfulqa`'s github source.

Unlike TruthfulQA, SimpleQA Verified is not adversarial and has no
multiple-choice format or enumerated distractor answers — it is plain
open-ended short-form QA, one gold answer per question. `correct_answers`
and `incorrect_answers` are still exposed (as `[correct_answer]` and `[]`
respectively) purely so `SimpleQAVerifiedSample` has the same shape
`evaluation.grade_correct`'s "open" grading path expects from a
TruthfulQA-style sample (question / correct_answer / correct_answers /
incorrect_answers) — that NLI-proxy grader works here unmodified.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SIMPLEQA_CSV_URL = (
    "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv"
)

DEFAULT_CACHE_DIR = "data/simpleqa_data"


def _download_if_missing(url: str, dest: Path) -> None:
    if dest.exists():
        return
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimpleQAVerifiedSample:
    """A single SimpleQA Verified question.

    Implements the `data.schema.Sample` protocol via `uid`, `prompt`, and
    `dataset`. Also exposes `question` / `correct_answer` / `correct_answers`
    / `incorrect_answers` — the shape `evaluation.grade_correct`'s "open"
    path expects (see module docstring) — so no dataset-specific grading
    code was needed to reuse it.
    """

    original_index: int
    question: str
    correct_answer: str
    topic: str
    answer_type: str            # "Number" | "Person" | "Date" | "Place" | "Other"
    multi_step: bool
    requires_reasoning: bool
    urls: list[str] = field(default_factory=list)

    # Always [correct_answer] / [] — SimpleQA Verified gives one canonical
    # gold string per question, not TruthfulQA's enumerated phrasing/
    # distractor lists. Populated at construction time (see
    # SimpleQAVerifiedLoader._parse_row) so grade_correct's "open" path
    # works unmodified — see module docstring.
    correct_answers: list[str] = field(default_factory=list)
    incorrect_answers: list[str] = field(default_factory=list)

    def open_prompt(self) -> str:
        """Open-ended format: just the bare question, no framing added."""
        return self.question

    # ------------------------------------------------------------------
    # Sample protocol (see data.schema.Sample) — uniform interface
    # ------------------------------------------------------------------

    @property
    def uid(self) -> str:
        """Stable cross-dataset identifier, e.g. 'simpleqa-0042'."""
        return f"simpleqa-{self.original_index:04d}"

    @property
    def prompt(self) -> str:
        return self.open_prompt()

    @property
    def dataset(self) -> str:
        return "simpleqa"

    def to_dict(self) -> dict:
        """Serialise to a JSONL-friendly dict."""
        return {
            "type": self.dataset,
            "uid": self.uid,
            "original_index": self.original_index,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "correct_answers": list(self.correct_answers),
            "incorrect_answers": list(self.incorrect_answers),
            "topic": self.topic,
            "answer_type": self.answer_type,
            "multi_step": self.multi_step,
            "requires_reasoning": self.requires_reasoning,
            "urls": list(self.urls),
            "prompt": self.prompt,
            "open_prompt": self.open_prompt(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimpleQAVerifiedSample":
        """Reconstruct from a to_dict()-shaped dict.

        Derived entries (uid, prompt, open_prompt, type) are recomputed from
        the stored fields, not read back — they aren't constructor args.
        """
        return cls(
            original_index=d["original_index"],
            question=d["question"],
            correct_answer=d["correct_answer"],
            topic=d["topic"],
            answer_type=d["answer_type"],
            multi_step=d["multi_step"],
            requires_reasoning=d["requires_reasoning"],
            urls=list(d.get("urls", [])),
            correct_answers=list(d.get("correct_answers", [])) or [d["correct_answer"]],
            incorrect_answers=list(d.get("incorrect_answers", [])),
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class SimpleQAVerifiedLoader:
    """
    Loads SimpleQA Verified questions.

    Usage
    -----
    >>> loader = SimpleQAVerifiedLoader()
    >>> samples = loader.load()                        # all 1,000 questions
    >>> subset = loader.load(n=100, seed=42)            # reproducible 100-sample subset
    >>> by_topic = loader.load_by_topic()               # dict[topic -> list[sample]]
    >>> reasoning_only = loader.load(requires_reasoning=True)

    Source: https://huggingface.co/datasets/google/simpleqa-verified — CSV
    downloaded once into `cache_dir` (default `data/simpleqa_data/`).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = cache_dir
        self._samples: Optional[list[SimpleQAVerifiedSample]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        n: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = False,
        topics: Optional[list[str]] = None,
        answer_types: Optional[list[str]] = None,
        multi_step: Optional[bool] = None,
        requires_reasoning: Optional[bool] = None,
    ) -> list[SimpleQAVerifiedSample]:
        """
        Return a list of SimpleQAVerifiedSample objects.

        Parameters
        ----------
        n                  : if set, return a random subset of this size
        seed               : random seed for subsetting / shuffling
        shuffle            : shuffle the full dataset before subsetting
        topics             : if set, filter to these topics only
        answer_types       : if set, filter to these answer types only
        multi_step         : if set, filter to samples with this multi_step value
        requires_reasoning : if set, filter to samples with this requires_reasoning value
        """
        samples = self._load_all()

        if topics:
            samples = [s for s in samples if s.topic in topics]
        if answer_types:
            samples = [s for s in samples if s.answer_type in answer_types]
        if multi_step is not None:
            samples = [s for s in samples if s.multi_step == multi_step]
        if requires_reasoning is not None:
            samples = [s for s in samples if s.requires_reasoning == requires_reasoning]

        if shuffle or n is not None:
            rng = random.Random(seed)
            samples = samples.copy()
            rng.shuffle(samples)

        if n is not None:
            samples = samples[:n]

        return samples

    def load_by_topic(self, **kwargs) -> dict[str, list[SimpleQAVerifiedSample]]:
        """Return samples grouped by SimpleQA Verified topic."""
        samples = self.load(**kwargs)
        out: dict[str, list[SimpleQAVerifiedSample]] = {}
        for s in samples:
            out.setdefault(s.topic, []).append(s)
        return out

    def available_topics(self) -> list[str]:
        """Return the SimpleQA Verified topics, sorted (10)."""
        return sorted({s.topic for s in self.load()})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> list[SimpleQAVerifiedSample]:
        if self._samples is not None:
            return self._samples

        cache_dir = Path(self._cache_dir) if self._cache_dir else Path(DEFAULT_CACHE_DIR)
        csv_path = cache_dir / "simpleqa_verified.csv"
        _download_if_missing(_SIMPLEQA_CSV_URL, csv_path)

        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

        self._samples = [self._parse_row(row) for row in rows]
        return self._samples

    @staticmethod
    def _parse_row(row: dict) -> SimpleQAVerifiedSample:
        answer = row["answer"].strip()
        urls = [u.strip() for u in row["urls"].split(",") if u.strip()]
        return SimpleQAVerifiedSample(
            original_index=int(row["original_index"]),
            question=row["problem"].strip(),
            correct_answer=answer,
            topic=row["topic"],
            answer_type=row["answer_type"],
            multi_step=row["multi_step"].strip().lower() == "true",
            requires_reasoning=row["requires_reasoning"].strip().lower() == "true",
            urls=urls,
            correct_answers=[answer],
            incorrect_answers=[],
        )
