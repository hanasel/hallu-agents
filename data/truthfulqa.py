"""
TruthfulQA dataset loader.

Loads TruthfulQA and returns clean, validated dataclasses ready for the
agent querying pipeline. Two sources are supported:

  - "github" (default) — the canonical, actively maintained files from the
    sylinrl/TruthfulQA GitHub repo, including the Jan 2025 update: refined
    question wording, the "Indexical Error: Time" category dropped (37
    categories, 790 questions, down from 38/817), and a new binary
    multiple-choice format (`Best Answer` vs `Best Incorrect Answer`) that
    the authors now recommend over the original MC1/MC2 setup. Files are
    downloaded once into `cache_dir` (default `data/truthfulqa_data/`) and
    reused on subsequent runs.
  - "hf" — the original HuggingFace `truthful_qa` hub dataset (MC1/MC2,
    817 questions, 38 categories). Frozen at the pre-2025 version; kept so
    prior runs stay reproducible.

MC1 format: one correct answer among several options (single-label).
MC2 is deliberately excluded — multiple correct answers inflate the
false-positive rate of any disagreement-based signal.

Prompt formats
--------------
Ground truth is MC1 by default (one correct answer per question, several
distractors). Three prompt formats are supported for querying agents:

  - mc_prompt(mc_format="mc1")     — multiple-choice with A/B/C/D... options
                                      (the original format), asks for a letter.
  - mc_prompt(mc_format="binary")  — new Jan-2025 format: exactly two options,
                                      "Best Answer" vs "Best Incorrect Answer"
                                      (github source only). Recommended by the
                                      TruthfulQA authors for MC grading, but
                                      not the default here — see mc_prompt().
  - open_prompt()                  — just the bare question. Produces free-text
                                      responses suitable for all three
                                      disagreement measures (answer, semantic,
                                      claim-level). Requires fuzzy matching
                                      against the correct/incorrect answer
                                      lists for answer-level evaluation.

The `.prompt` property returns `open_prompt()` by default, since open-ended
responses are needed for two of the three disagreement measures in RQ2.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# GitHub source (default) — Jan 2025 update
# ---------------------------------------------------------------------------

_GITHUB_RAW_BASE = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main"
_GITHUB_CSV_URL = f"{_GITHUB_RAW_BASE}/TruthfulQA.csv"
_GITHUB_MC_TASK_URL = f"{_GITHUB_RAW_BASE}/data/mc_task.json"

DEFAULT_CACHE_DIR = "data/truthfulqa_data"


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
class TruthfulQASample:
    """A single TruthfulQA question.

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

    # All MC1 choices shuffled together (so agent order is randomised)
    choices: list[str]
    correct_index: int          # index into `choices` of the correct answer

    # New in the Jan 2025 update (github source). Populated best-effort
    # (from correct_answer / incorrect_answers[0]) when source="hf".
    question_type: str = "Unknown"    # "Adversarial" | "Non-Adversarial"
    source_url: str = ""
    best_answer: str = ""
    best_incorrect_answer: str = ""
    binary_choices: list[str] = field(default_factory=list)   # [Best Answer, Best Incorrect Answer], shuffled
    binary_correct_index: int = -1

    def mc_prompt(self, mc_format: str = "mc1") -> str:
        """Multiple-choice format: question + lettered options, asks for a letter.

        `mc_format="mc1"` (default) uses the original single-true format with
        the full distractor set (4-5+ options). `mc_format="binary"` uses the
        Jan-2025 two-option format (Best Answer vs Best Incorrect Answer),
        which the TruthfulQA authors now recommend for MC grading — but MC1
        stays the default here since binary collapses inter-agent disagreement
        to ~2 discrete outcomes and raises chance agreement well above MC1's.

        Produces single-token responses ('A', 'B', ...) suitable for
        answer-level disagreement via exact letter match. Not useful for
        semantic or claim-level measures — nothing to cluster or decompose.
        """
        if mc_format == "mc1":
            choices = self.choices
        elif mc_format == "binary":
            choices = self.binary_choices
        else:
            raise ValueError(f"mc_format must be 'mc1' or 'binary'; got {mc_format!r}")

        lines = [f"Question: {self.question}", "Options:"]
        for i, choice in enumerate(choices):
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

    def to_prompt(self, include_choices: bool = True, mc_format: str = "mc1") -> str:
        """Backward-compatible prompt formatter.

        `include_choices=True` returns `mc_prompt(mc_format)`; False returns
        `open_prompt()`. Kept so existing code (e.g. quickstart.py) that
        uses this method continues to work; prefer the explicit
        `mc_prompt()` / `open_prompt()` methods in new code.
        """
        return self.mc_prompt(mc_format=mc_format) if include_choices else self.open_prompt()

    @property
    def correct_letter(self) -> str:
        return chr(65 + self.correct_index)

    @property
    def binary_correct_letter(self) -> str:
        return chr(65 + self.binary_correct_index)

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
            "question_type": self.question_type,
            "source_url": self.source_url,
            "correct_answer": self.correct_answer,
            "incorrect_answers": list(self.incorrect_answers),
            "choices": list(self.choices),
            "correct_index": self.correct_index,
            "correct_letter": self.correct_letter,
            "best_answer": self.best_answer,
            "best_incorrect_answer": self.best_incorrect_answer,
            "binary_choices": list(self.binary_choices),
            "binary_correct_index": self.binary_correct_index,
            "binary_correct_letter": self.binary_correct_letter,
            "prompt": self.prompt,                              # what agents will actually see (open-ended)
            "mc_prompt": self.mc_prompt(mc_format="mc1"),        # all variants exposed for inspection
            "binary_mc_prompt": self.mc_prompt(mc_format="binary"),
            "open_prompt": self.open_prompt(),
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TruthfulQALoader:
    """
    Loads TruthfulQA questions.

    Usage
    -----
    >>> loader = TruthfulQALoader()                   # github source (default)
    >>> samples = loader.load()                       # all 790 questions
    >>> subset = loader.load(n=100, seed=42)          # reproducible 100-sample subset
    >>> by_cat = loader.load_by_category()            # dict[category -> list[sample]]
    >>> adversarial = loader.load(question_types=["Adversarial"])

    >>> legacy = TruthfulQALoader(source="hf")         # pre-2025 HF hub dataset

    Sources
    -------
    github (default): https://github.com/sylinrl/TruthfulQA — CSV + mc_task.json,
        downloaded once into `cache_dir` (default `data/truthfulqa_data/`).
    hf: https://huggingface.co/datasets/truthful_qa — frozen pre-2025 snapshot,
        kept for reproducing older runs.
    """

    HF_DATASET_NAME = "truthful_qa"
    HF_CONFIG = "multiple_choice"          # contains both MC1 and MC2 splits

    def __init__(self, cache_dir: Optional[str] = None, source: str = "github"):
        if source not in ("github", "hf"):
            raise ValueError(f"source must be 'github' or 'hf'; got {source!r}")
        self.source = source
        self._cache_dir = cache_dir
        self._raw: Optional[object] = None   # holds the HF DatasetDict once loaded (hf source only)
        self._cat_by_question: Optional[dict[str, str]] = None  # question -> category (hf source only)
        self._samples: Optional[list[TruthfulQASample]] = None  # full parsed set, either source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        n: Optional[int] = None,
        seed: int = 42,
        shuffle: bool = False,
        categories: Optional[list[str]] = None,
        question_types: Optional[list[str]] = None,
    ) -> list[TruthfulQASample]:
        """
        Return a list of TruthfulQASample objects.

        Parameters
        ----------
        n              : if set, return a random subset of this size
        seed           : random seed for subsetting / shuffling
        shuffle        : shuffle the full dataset before subsetting
        categories     : if set, filter to these categories only
        question_types : if set, filter to these question types only
                         (github source: "Adversarial" | "Non-Adversarial")
        """
        samples = self._load_all()

        if categories:
            samples = [s for s in samples if s.category in categories]
        if question_types:
            samples = [s for s in samples if s.question_type in question_types]

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
        """Return the TruthfulQA categories, sorted (37 for github, 38 for hf)."""
        return sorted({s.category for s in self.load()})

    # ------------------------------------------------------------------
    # Internal helpers — dispatch
    # ------------------------------------------------------------------

    def _load_all(self) -> list[TruthfulQASample]:
        if self._samples is not None:
            return self._samples
        self._samples = (
            self._load_from_github() if self.source == "github" else self._load_from_hf()
        )
        return self._samples

    # ------------------------------------------------------------------
    # Internal helpers — github source (default)
    # ------------------------------------------------------------------

    def _load_from_github(self) -> list[TruthfulQASample]:
        cache_dir = Path(self._cache_dir) if self._cache_dir else Path(DEFAULT_CACHE_DIR)
        csv_path = cache_dir / "TruthfulQA.csv"
        mc_task_path = cache_dir / "mc_task.json"
        _download_if_missing(_GITHUB_CSV_URL, csv_path)
        _download_if_missing(_GITHUB_MC_TASK_URL, mc_task_path)

        with csv_path.open(encoding="utf-8-sig", newline="") as fh:
            csv_by_question = {row["Question"].strip(): row for row in csv.DictReader(fh)}

        with mc_task_path.open(encoding="utf-8") as fh:
            mc_entries = json.load(fh)

        samples = []
        for idx, entry in enumerate(mc_entries):
            question = entry["question"].strip()
            csv_row = csv_by_question.get(question)
            if csv_row is None:
                # CSV and mc_task.json cover the same question set; skip
                # defensively rather than crash if the upstream repo drifts.
                continue
            samples.append(self._parse_github_entry(idx, entry, csv_row))
        return samples

    def _parse_github_entry(self, idx: int, entry: dict, csv_row: dict) -> TruthfulQASample:
        question = entry["question"].strip()
        category = csv_row["Category"]
        question_type = csv_row["Type"]
        source_url = csv_row["Source"]

        mc1 = entry["mc1_targets"]
        choices_raw: list[str] = list(mc1.keys())
        labels: list[int] = list(mc1.values())
        correct_idx_raw = labels.index(1)
        correct_answer = choices_raw[correct_idx_raw]
        incorrect_answers = [c for i, c in enumerate(choices_raw) if i != correct_idx_raw]

        # Shuffle choices so correct answer isn't always first
        rng = random.Random(idx)          # deterministic per question
        shuffled = choices_raw.copy()
        rng.shuffle(shuffled)
        correct_index = shuffled.index(correct_answer)

        best_answer = csv_row["Best Answer"]
        best_incorrect_answer = csv_row["Best Incorrect Answer"]
        binary_choices = [best_answer, best_incorrect_answer]
        rng.shuffle(binary_choices)
        binary_correct_index = binary_choices.index(best_answer)

        return TruthfulQASample(
            question_id=idx,
            question=question,
            category=category,
            correct_answer=correct_answer,
            incorrect_answers=incorrect_answers,
            choices=shuffled,
            correct_index=correct_index,
            question_type=question_type,
            source_url=source_url,
            best_answer=best_answer,
            best_incorrect_answer=best_incorrect_answer,
            binary_choices=binary_choices,
            binary_correct_index=binary_correct_index,
        )

    # ------------------------------------------------------------------
    # Internal helpers — hf source (legacy, pre-2025 snapshot)
    # ------------------------------------------------------------------

    def _get_raw(self):
        if self._raw is not None:
            return self._raw
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' library is required for source='hf'. Install it with:\n"
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

    def _load_from_hf(self) -> list[TruthfulQASample]:
        raw = self._get_raw()
        # TruthfulQA only has a validation split in HF
        split = raw["validation"]
        return [self._parse_hf_row(i, row) for i, row in enumerate(split)]

    def _parse_hf_row(self, idx: int, row: dict) -> TruthfulQASample:
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

        # Best-effort binary fields (github-only concept) — computed from a
        # separate rng stream so the MC1 shuffle above is untouched, keeping
        # old hf-source runs bit-for-bit reproducible.
        best_answer = correct_answer
        best_incorrect_answer = incorrect_answers[0] if incorrect_answers else ""
        binary_choices = [best_answer, best_incorrect_answer]
        random.Random(idx + 1_000_003).shuffle(binary_choices)
        binary_correct_index = binary_choices.index(best_answer)

        return TruthfulQASample(
            question_id=idx,
            question=question,
            category=category,
            correct_answer=correct_answer,
            incorrect_answers=incorrect_answers,
            choices=shuffled,
            correct_index=correct_index,
            best_answer=best_answer,
            best_incorrect_answer=best_incorrect_answer,
            binary_choices=binary_choices,
            binary_correct_index=binary_correct_index,
        )
