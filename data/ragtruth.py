"""
RAGTruth dataset loader.

RAGTruth is a hallucination corpus for the RAG setting, comprising ~18k
responses from 6 LLMs across 3 task types, with word-level span annotations.

Source: https://github.com/ParticleMedia/RAGTruth (official, canonical)
Paper:  Niu et al., ACL 2024.

The dataset ships as two JSONL files in the GitHub repo (not on HuggingFace).
This loader reads them from a local directory — by default `./datasets/ragtruth/`
relative to the project root. Use `download_ragtruth_files()` to fetch them,
or place them there manually.

Label types
-----------
  Evident Baseless Info  — unsupported content stated confidently
  Evident Conflict       — content that directly contradicts the source
  Subtle Baseless Info   — unsupported content with apparent hedging
  Subtle Conflict        — minor contradiction with source

Key fields on each span
-----------------------
  text         : the hallucinated text span
  label_type   : one of the four types above
  due_to_null  : hallucination caused by missing context (hardest cases)
  implicit_true: flagged content is actually correct but ungrounded
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TYPES = frozenset({
    "Evident Baseless Info",
    "Evident Conflict",
    "Subtle Baseless Info",
    "Subtle Conflict",
})

# Normalised task types used throughout this project. Raw RAGTruth values
# ("QA", "Summary", "Data2txt") are mapped to these via _normalise_task_type().
TASK_TYPES = frozenset({"qa", "summarization", "data2txt"})

# Source model identifiers as they appear in response.jsonl (lowercased).
SOURCE_MODELS = frozenset({
    "gpt-4-0613",
    "gpt-3.5-turbo-0613",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
    "mistral-7b-instruct",
})

# Default location to look for the RAGTruth JSONL files.
DEFAULT_DATA_DIR = "data/ragtruth_data"

# Canonical raw-file URLs (used by download_ragtruth_files()).
_RAW_BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset"
_RAW_FILES = ("source_info.jsonl", "response.jsonl")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HallucinationSpan:
    """A single annotated hallucination span within a RAGTruth response."""

    text: str
    label_type: str                  # one of LABEL_TYPES
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    due_to_null: bool = False        # caused by missing context
    implicit_true: bool = False      # actually correct but ungrounded

    @property
    def is_evident(self) -> bool:
        return self.label_type.startswith("Evident")

    @property
    def is_conflict(self) -> bool:
        return "Conflict" in self.label_type

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label_type": self.label_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "due_to_null": self.due_to_null,
            "implicit_true": self.implicit_true,
        }


@dataclass(frozen=True)
class RAGTruthSample:
    """
    One response from the RAGTruth corpus.

    A single source instance (question + retrieved passages) maps to
    6 RAGTruthSamples — one per source model. This multi-model structure
    partially mirrors the multi-agent setup of this project.

    Implements the `data.schema.Sample` protocol via the `uid`, `prompt`,
    and `dataset` properties below, so this class can be used uniformly
    by dataset-agnostic code (e.g. the agent-querying layer).
    """

    sample_id: str
    source_id: str                   # groups the 6 model responses per source
    task_type: str                   # qa | summarization | data2txt (normalised)
    source_model: str                # which LLM generated this response
    split: str                       # train | test

    # Inputs
    source: str                      # "MARCO" | "CNN/DM" | "Recent News" | "Yelp"
    source_info_text: str            # source context as a single string (for display)
    question: Optional[str]          # extracted from source_info for QA tasks
    prompt_text: str                 # exact prompt sent to the corpus models (Mode B)

    # Output
    response: str
    quality: str = "good"            # "good" | "incorrect_refusal" | "truncated"

    # Annotations
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_hallucinated(self) -> bool:
        """True if the response contains at least one annotated span."""
        return len(self.hallucination_spans) > 0

    @property
    def hallucination_rate(self) -> float:
        """Fraction of response characters covered by hallucination spans."""
        if not self.hallucination_spans or not self.response:
            return 0.0
        covered = set()
        for span in self.hallucination_spans:
            if span.start_char is not None and span.end_char is not None:
                covered.update(range(span.start_char, span.end_char))
        return len(covered) / len(self.response)

    @property
    def n_evident_spans(self) -> int:
        return sum(1 for s in self.hallucination_spans if s.is_evident)

    @property
    def n_conflict_spans(self) -> int:
        return sum(1 for s in self.hallucination_spans if s.is_conflict)

    def to_rag_prompt(self) -> str:
        """
        Return the exact prompt the corpus models received.

        This is the prompt field from source_info.jsonl — using it directly
        means our agents see exactly what the corpus models saw, so any
        disagreement reflects model differences rather than prompt-format
        differences. This is the Mode B design discussed in the project plan.
        """
        return self.prompt_text

    # ------------------------------------------------------------------
    # Sample protocol (see data.schema.Sample) — uniform interface
    # ------------------------------------------------------------------

    @property
    def uid(self) -> str:
        """Stable cross-dataset identifier, e.g. 'ragtruth-1472'."""
        return f"ragtruth-{self.sample_id}"

    @property
    def prompt(self) -> str:
        """Ready-to-send prompt for agent querying (RAG-formatted)."""
        return self.to_rag_prompt()

    @property
    def dataset(self) -> str:
        return "ragtruth"

    def to_dict(self) -> dict:
        """Serialise to a JSONL-friendly dict."""
        return {
            "type": self.dataset,
            "uid": self.uid,
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "task_type": self.task_type,
            "source": self.source,
            "source_model": self.source_model,
            "split": self.split,
            "question": self.question,
            "response": self.response,
            "quality": self.quality,
            "is_hallucinated": self.is_hallucinated,
            "n_spans": len(self.hallucination_spans),
            "hallucination_spans": [sp.to_dict() for sp in self.hallucination_spans],
            "prompt": self.prompt,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TASK_TYPE_MAP = {
    "qa": "qa",
    "question_answering": "qa",
    "summary": "summarization",
    "summarization": "summarization",
    "data2txt": "data2txt",
    "data-to-text": "data2txt",
}


def _normalise_task_type(raw: str) -> str:
    """Map raw RAGTruth task_type values to our normalised set."""
    key = (raw or "").strip().lower()
    return _TASK_TYPE_MAP.get(key, key)


def _extract_question(source_info, task_type: str) -> Optional[str]:
    """Pull the question out of source_info for QA tasks."""
    if task_type != "qa":
        return None
    if isinstance(source_info, dict):
        q = source_info.get("question")
        return str(q) if q else None
    return None


def _stringify_source_info(source_info) -> str:
    """Render source_info as a single string for display / inspection."""
    if isinstance(source_info, str):
        return source_info
    if isinstance(source_info, dict):
        # For QA: dict has "question" and "passages" -> show passages
        # For Data2txt: dict is Yelp business attributes -> JSON dump
        if "passages" in source_info:
            return str(source_info.get("passages", ""))
        return json.dumps(source_info, ensure_ascii=False, indent=2)
    return str(source_info)


def download_ragtruth_files(
    target_dir: str | Path = DEFAULT_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Download source_info.jsonl and response.jsonl from the official
    ParticleMedia GitHub repo into `target_dir`.

    Set force=True to overwrite existing files.
    """
    import urllib.request

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading RAGTruth files to {target.resolve()}")
    for fn in _RAW_FILES:
        dest = target / fn
        if dest.exists() and not force:
            print(f"  {fn} already present (skipping; pass force=True to overwrite)")
            continue
        url = f"{_RAW_BASE_URL}/{fn}"
        print(f"  fetching {url}")
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"    saved {dest.name} ({size_mb:.1f} MB)")
    return target


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class RAGTruthLoader:
    """
    Loads the RAGTruth corpus from local JSONL files.

    Usage
    -----
    >>> loader = RAGTruthLoader()                         # default ./datasets/ragtruth/
    >>> test_samples = loader.load(split="test")          # 2,700 responses
    >>> train_samples = loader.load(split="train")        # 15,090 responses
    >>> qa_only = loader.load(split="test", task_types=["qa"])
    >>> grouped = loader.load_by_source(split="test")     # dict[source_id -> list]

    If the JSONL files are not present, call `download_ragtruth_files()` first,
    or download them manually from:
      https://github.com/ParticleMedia/RAGTruth/tree/main/dataset

    Notes
    -----
    The standard evaluation protocol (Luna, LettuceDetect) uses the test split
    and reports example-level F1. The train split is provided for supervised
    baselines only — the proposed zero-resource method does not use it.
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self._samples: Optional[list[RAGTruthSample]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        split: Optional[str] = "test",
        n: Optional[int] = None,
        seed: int = 42,
        task_types: Optional[list[str]] = None,
        source_models: Optional[list[str]] = None,
        hallucinated_only: bool = False,
        clean_only: bool = False,
        quality: Optional[str] = None,
    ) -> list[RAGTruthSample]:
        """
        Return RAGTruth samples.

        Parameters
        ----------
        split            : "train", "test", or None for all
        n                : if set, return a random subset of this size
        seed             : random seed for subsetting
        task_types       : filter to any subset of {"qa", "summarization", "data2txt"}
        source_models    : filter to specific source model(s) (case-insensitive)
        hallucinated_only: keep only responses with ≥1 annotated span
        clean_only       : keep only responses with no annotated spans
        quality          : filter by quality flag (e.g. "good" to drop refusals/truncations)
        """
        if split is not None and split not in ("train", "test"):
            raise ValueError(f"split must be 'train', 'test', or None; got {split!r}")
        if hallucinated_only and clean_only:
            raise ValueError("hallucinated_only and clean_only are mutually exclusive")

        samples = list(self._load_all())

        # ---- Filtering ----
        if split is not None:
            samples = [s for s in samples if s.split == split]

        if task_types:
            wanted = {_normalise_task_type(t) for t in task_types}
            samples = [s for s in samples if s.task_type in wanted]

        if source_models:
            wanted_m = {m.lower() for m in source_models}
            samples = [s for s in samples if s.source_model.lower() in wanted_m]

        if hallucinated_only:
            samples = [s for s in samples if s.is_hallucinated]

        if clean_only:
            samples = [s for s in samples if not s.is_hallucinated]

        if quality:
            samples = [s for s in samples if s.quality == quality]

        # ---- Subsetting ----
        if n is not None:
            import random
            rng = random.Random(seed)
            samples = samples.copy()
            rng.shuffle(samples)
            samples = samples[:n]

        return samples

    def load_by_source(
        self, split: str = "test", **kwargs
    ) -> dict[str, list[RAGTruthSample]]:
        """
        Group samples by source_id so you can inspect all 6 model responses
        for the same input. This directly mirrors the multi-agent setup.
        """
        samples = self.load(split=split, **kwargs)
        out: dict[str, list[RAGTruthSample]] = {}
        for s in samples:
            out.setdefault(s.source_id, []).append(s)
        return out

    def stats(self, split: str = "test") -> dict:
        """Return summary statistics for the split."""
        samples = self.load(split=split)
        hallucinated = [s for s in samples if s.is_hallucinated]
        by_task: dict[str, int] = {}
        by_model: dict[str, int] = {}
        for s in samples:
            by_task[s.task_type] = by_task.get(s.task_type, 0) + 1
            by_model[s.source_model] = by_model.get(s.source_model, 0) + 1

        return {
            "split": split,
            "total": len(samples),
            "hallucinated": len(hallucinated),
            "hallucination_rate": len(hallucinated) / len(samples) if samples else 0.0,
            "by_task": by_task,
            "by_model": by_model,
        }

    # ------------------------------------------------------------------
    # Internal: file loading and joining
    # ------------------------------------------------------------------

    def _load_all(self) -> list[RAGTruthSample]:
        if self._samples is not None:
            return self._samples

        source_path = self.data_dir / "source_info.jsonl"
        response_path = self.data_dir / "response.jsonl"

        if not source_path.exists() or not response_path.exists():
            raise FileNotFoundError(
                f"RAGTruth JSONL files not found in {self.data_dir.resolve()}.\n"
                f"Expected:\n"
                f"  {source_path}\n"
                f"  {response_path}\n\n"
                f"To download them, run:\n"
                f"  from data.ragtruth import download_ragtruth_files\n"
                f"  download_ragtruth_files()\n\n"
                f"Or manually from:\n"
                f"  https://github.com/ParticleMedia/RAGTruth/tree/main/dataset"
            )

        # Index source_info.jsonl by source_id
        sources: dict[str, dict] = {}
        with source_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sources[str(row["source_id"])] = row

        # Stream responses, joining each with its source
        samples: list[RAGTruthSample] = []
        with response_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                resp = json.loads(line)
                src = sources.get(str(resp.get("source_id", "")))
                if src is None:
                    continue  # orphaned response — skip rather than crash
                samples.append(self._build_sample(resp, src))

        self._samples = samples
        return samples

    def _build_sample(self, resp: dict, src: dict) -> RAGTruthSample:
        """Construct a RAGTruthSample by joining a response with its source."""
        task_type = _normalise_task_type(src.get("task_type", ""))
        source_info_raw = src.get("source_info", "")

        # Parse hallucination spans
        spans: list[HallucinationSpan] = []
        for span_raw in resp.get("labels", []) or []:
            if isinstance(span_raw, str):
                try:
                    span_raw = json.loads(span_raw)
                except (json.JSONDecodeError, TypeError):
                    continue
            if not isinstance(span_raw, dict):
                continue
            spans.append(
                HallucinationSpan(
                    text=span_raw.get("text", ""),
                    label_type=span_raw.get("label_type", "Evident Baseless Info"),
                    start_char=span_raw.get("start"),
                    end_char=span_raw.get("end"),
                    due_to_null=bool(span_raw.get("due_to_null", False)),
                    implicit_true=bool(span_raw.get("implicit_true", False)),
                )
            )

        return RAGTruthSample(
            sample_id=str(resp.get("id", "")),
            source_id=str(resp.get("source_id", "")),
            task_type=task_type,
            source_model=str(resp.get("model", "unknown")).lower(),
            split=str(resp.get("split", "test")),
            source=str(src.get("source", "")),
            source_info_text=_stringify_source_info(source_info_raw),
            question=_extract_question(source_info_raw, task_type),
            prompt_text=str(src.get("prompt", "")),
            response=str(resp.get("response", "")),
            quality=str(resp.get("quality", "good")),
            hallucination_spans=spans,
        )
