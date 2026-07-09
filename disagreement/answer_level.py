"""Answer-level disagreement measures.

Two variants:

  JaccardDisagreement
      Token-set overlap for free-text responses.
      score = 1 - mean(pairwise Jaccard similarity).
      Matches the interim report's "token-level overlap" measure for
      open-ended answer-level disagreement (§3.1.3).

  MCExactMatch
      Exact letter match for multiple-choice responses.
      score = fraction of response pairs whose extracted letters differ.
      Use when agents were queried with `sample.mc_prompt()`.

Answer-level is the coarsest of the three disagreement measures. It ignores
semantic equivalence — two responses that mean the same thing but use
different words will register as disagreement. Semantic Entropy (Farquhar
et al. 2024) addresses this at the next level up, and will be implemented
in `semantic.py`.
"""

from __future__ import annotations

import re
import string
from typing import List, Optional, Sequence, Tuple

from .base import DisagreementResult, ResponseLike


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_text(r: ResponseLike) -> str:
    """Coerce a response (str or AgentResponse) into a text string.

    Defensive against None/missing text so an errored AgentResponse doesn't
    crash the pipeline — it just contributes an empty token set, which
    naturally scores as maximum disagreement against anything non-empty.
    """
    if isinstance(r, str):
        return r
    text = getattr(r, "text", None)
    return text if text is not None else ""


def _tokenise(text: str) -> set:
    """Lowercase, strip punctuation, whitespace-split, return a token set.

    Deliberately simple — Semantic Entropy and other higher-level measures
    handle semantic normalisation. At answer-level, we're checking whether
    two responses use similar wording, nothing more.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


def _pairs(n: int) -> List[Tuple[int, int]]:
    """All (i, j) index pairs with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ---------------------------------------------------------------------------
# JaccardDisagreement
# ---------------------------------------------------------------------------

class JaccardDisagreement:
    """1 - mean(pairwise Jaccard similarity) over token sets.

    Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Both-empty pairs are treated as agreeing (sim = 1.0).
    """

    @property
    def name(self) -> str:
        return "jaccard"

    def score(self, responses: Sequence[ResponseLike]) -> DisagreementResult:
        texts = [_extract_text(r) for r in responses]
        n = len(texts)
        if n < 2:
            raise ValueError(f"Need at least 2 responses to compute disagreement; got {n}")

        token_sets = [_tokenise(t) for t in texts]
        pairwise_disagreements: List[float] = []

        for i, j in _pairs(n):
            a, b = token_sets[i], token_sets[j]
            union = a | b
            if not union:
                similarity = 1.0    # both empty → treat as agreeing
            else:
                similarity = len(a & b) / len(union)
            pairwise_disagreements.append(1.0 - similarity)

        mean_disagreement = sum(pairwise_disagreements) / len(pairwise_disagreements)
        return DisagreementResult(
            score=mean_disagreement,
            measure=self.name,
            n_responses=n,
            details={
                "pairwise_disagreements": pairwise_disagreements,
                "token_set_sizes": [len(ts) for ts in token_sets],
            },
        )


# ---------------------------------------------------------------------------
# MCExactMatch
# ---------------------------------------------------------------------------

_MC_LETTER_RE = re.compile(r"\b([A-Z])\b")


def _extract_letter(text: str) -> Optional[str]:
    """Extract the first capital letter A-Z appearing as a whole word.

    - "A"                        -> "A"
    - "A."                       -> "A"
    - "The answer is C."         -> "C"     (I is not the first word-boundary letter here — 'The' starts lowercase 'he')
    - "IMO the answer is B."     -> "IMO" would match first, but we require
                                    a *single* letter surrounded by word boundaries,
                                    so we skip 'IMO' and match 'B'.
    - "yes, learning music..."   -> None    (no capital word-boundary letter)

    Returns None if no such letter is found, which will be treated as
    disagreement with everything by MCExactMatch.
    """
    for match in _MC_LETTER_RE.finditer(text.strip()):
        return match.group(1)
    return None


class MCExactMatch:
    """Answer-level disagreement for MC responses.

    Extracts one letter (A-Z) from each response and compares pairwise.
    Same letter -> 0.0 disagreement; different or unparseable -> 1.0.
    """

    @property
    def name(self) -> str:
        return "mc_exact_match"

    def score(self, responses: Sequence[ResponseLike]) -> DisagreementResult:
        texts = [_extract_text(r) for r in responses]
        n = len(texts)
        if n < 2:
            raise ValueError(f"Need at least 2 responses to compute disagreement; got {n}")

        letters: List[Optional[str]] = [_extract_letter(t) for t in texts]
        pairwise_disagreements: List[float] = []

        for i, j in _pairs(n):
            li, lj = letters[i], letters[j]
            if li is None or lj is None:
                pairwise_disagreements.append(1.0)   # unparseable → treated as disagreement
            elif li == lj:
                pairwise_disagreements.append(0.0)
            else:
                pairwise_disagreements.append(1.0)

        mean_disagreement = sum(pairwise_disagreements) / len(pairwise_disagreements)
        return DisagreementResult(
            score=mean_disagreement,
            measure=self.name,
            n_responses=n,
            details={
                "extracted_letters": letters,
                "pairwise_disagreements": pairwise_disagreements,
                "n_unparseable": sum(1 for l in letters if l is None),
            },
        )
