"""Base types for disagreement measures.

Every disagreement measure — answer-level, semantic, claim-level — takes a
list of responses and returns a `DisagreementResult` in [0, 1].
Higher = more disagreement = elevated hallucination risk (§3.1.3).

Concrete measures live in sibling modules:
  - `answer_level.py`  → JaccardDisagreement, MCExactMatch
  - `semantic.py`      → (to come — Semantic Entropy style clustering)
  - `claim_level.py`   → (to come — FActScore-style atomic-claim NLI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, Union, runtime_checkable


ResponseLike = Union[str, Any]   # str or agents.base.AgentResponse


@dataclass(frozen=True)
class DisagreementResult:
    """Output of a single disagreement computation.

    Attributes
    ----------
    score        : float in [0, 1]. 0 = all responses agree, 1 = maximum
                   disagreement. The exact interpretation is measure-specific
                   but this range is guaranteed.
    measure      : identifier of the measure, e.g. 'jaccard', 'mc_exact_match'.
    n_responses  : how many responses were compared.
    details      : measure-specific extras (per-pair scores, clusters,
                   extracted answers, etc.) — kept for downstream analysis
                   without expanding the schema.
    """

    score: float
    measure: str
    n_responses: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "measure": self.measure,
            "n_responses": self.n_responses,
            "details": dict(self.details),
        }


@runtime_checkable
class DisagreementMeasure(Protocol):
    """The interface every disagreement measure satisfies.

    Implementations may accept either raw strings or AgentResponse objects
    in the `responses` list — a small helper (`_extract_text` in each
    concrete module) normalises this.
    """

    @property
    def name(self) -> str:
        """Short identifier for logging / result JSONL, e.g. 'jaccard'."""
        ...

    def score(self, responses: Sequence[ResponseLike]) -> DisagreementResult:
        """Compute a disagreement score for the given responses.

        Raises ValueError if fewer than 2 responses are provided.
        """
        ...
