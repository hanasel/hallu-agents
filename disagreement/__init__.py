"""Disagreement measures over multi-agent responses.

Three levels, coarsest → finest:

  answer-level   JaccardDisagreement, MCExactMatch   (lexical / exact match)
  semantic-level SemanticEntropyDisagreement          (NLI meaning-clusters)
  claim-level    (future)                             (decompose & verify claims)

All measures share the `DisagreementMeasure` interface: `.name` and
`.score(responses) -> DisagreementResult`.
"""

from .base import (
    DisagreementResult,
    DisagreementMeasure,
    ResponseLike,
)
from .answer_level import JaccardDisagreement, MCExactMatch
from .semantic import (
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
    NLIBackend,
    NLILabel,
    semantically_equivalent,
)

__all__ = [
    "DisagreementResult",
    "DisagreementMeasure",
    "ResponseLike",
    "JaccardDisagreement",
    "MCExactMatch",
    "SemanticEntropyDisagreement",
    "CrossEncoderNLI",
    "NLIBackend",
    "NLILabel",
    "semantically_equivalent",
]