"""Semantic-level disagreement via NLI meaning-clustering (Semantic Entropy).

Why this exists — the answer-level false positive
-------------------------------------------------
`JaccardDisagreement` measures *lexical* overlap. Two responses that assert
the same fact in different words score as strong disagreement even though they
agree:

    "Neil Armstrong."
    "The first person to walk on the Moon was Neil Armstrong."
      → Jaccard disagreement ≈ 0.9  (almost no shared tokens)
      → but they mean the SAME thing → true disagreement should be ~0.

That inflates the false-positive rate of a disagreement-based hallucination
signal: questions where the agents actually agree get flagged. This module
removes that failure mode by grouping responses into *meaning-classes* using
bidirectional Natural Language Inference (NLI) entailment, then measuring the
entropy of the resulting cluster distribution (Kuhn et al. 2023,
"Semantic Uncertainty"; Farquhar et al. 2024, Nature).

Discrete semantic entropy (and why, not the likelihood-weighted version)
------------------------------------------------------------------------
Farquhar et al. weight each meaning-cluster by the summed sequence likelihood
of its members. That needs per-token log-probabilities, which are (a) not
exposed uniformly across providers and (b) not comparable across *different*
models — an 8B's logprobs and a 70B's logprobs live on different scales. In
this inter-agent setup each agent contributes exactly one response at
temperature 0, so we treat the N responses as an unweighted sample and take
the entropy over cluster proportions:

        H = - Σ_c  (n_c / N) · ln(n_c / N)

with n_c the size of cluster c. H = 0 when all responses share one meaning
(full agreement) and H = ln(N) when every response is its own meaning-class
(full disagreement). We report raw nats AND the normalised H / ln(N) ∈ [0, 1]
so `score` is directly comparable to `JaccardDisagreement`.

Resolution caveat
-----------------
With N responses the score can only take the values realisable by integer
partitions of N. For N = 3 agents that's just {0, 0.579, 1.0} (all-agree,
2-vs-1, all-differ). This is coarse *by design* at the inter-agent level — it
answers "how many distinct claims did the panel make?", not "how confident is
one model?". If finer resolution is needed later, sample each agent k times
and pool the N·k responses before clustering; the maths below is unchanged.

Clustering
----------
Greedy single-pass clustering with the first member as the cluster
representative (as in the reference implementations). Two responses join the
same cluster iff they are *semantically equivalent*:

  - strict_entailment=True  (default): a ⊨ b AND b ⊨ a   (both entail)
  - strict_entailment=False (relaxed): neither direction contradicts

For QA, NLI is run on question-conditioned statements — the bare answers
"1969" and "1970" are not contradictory in isolation, but
"When did X happen? 1969" vs "When did X happen? 1970" are. Pass the question
via `context=` (or `question=`) to enable this; it is prepended to both sides.

NLI backend
-----------
`SemanticEntropyDisagreement` depends on an injected `NLIBackend`, so the
clustering/entropy logic is unit-testable with a deterministic fake and never
requires `torch`. `CrossEncoderNLI` is the production backend, wrapping a
DeBERTa-v3 MNLI cross-encoder via `sentence-transformers`.
"""

from __future__ import annotations

import math
from typing import (
    Any, Callable, Dict, FrozenSet, List, Literal, Optional, Protocol,
    Sequence, Tuple, runtime_checkable,
)

from .base import DisagreementResult, ResponseLike


NLILabel = Literal["entailment", "neutral", "contradiction"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_text(r: ResponseLike) -> str:
    """Coerce a response (str or object with `.text`) into a clean string.

    Also strips reasoning-model chain-of-thought (`<think>...</think>`), which
    GPT-OSS / Qwen3 can leak into the answer body and which would otherwise
    dominate the NLI comparison. Belt-and-braces: agents should suppress
    reasoning at the API level (see `GroqAgent.REASONING_PARAMS` /
    `AgentResponse.reasoning`), but a provider change shouldn't silently
    corrupt the signal.
    """
    if isinstance(r, str):
        text = r
    else:
        text = getattr(r, "text", None) or ""
    return _strip_reasoning(text).strip()


def _strip_reasoning(text: str) -> str:
    """Remove a leading/inline `<think>...</think>` block if present."""
    lower = text.lower()
    start = lower.find("<think>")
    if start == -1:
        return text
    end = lower.find("</think>", start)
    if end == -1:
        # Unterminated think block — drop everything from <think> on.
        return text[:start]
    return (text[:start] + text[end + len("</think>"):])


# ---------------------------------------------------------------------------
# NLI backend protocol + production implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class NLIBackend(Protocol):
    """A directional NLI classifier: does `premise` entail `hypothesis`?

    Implementations return one of "entailment" / "neutral" / "contradiction".
    `predict_batch` exists so backends can vectorise; a trivial default is
    provided by looping over `predict` for simple/fake backends.
    """

    def predict(self, premise: str, hypothesis: str) -> NLILabel: ...

    def predict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLILabel]: ...


# DeBERTa-v3 MNLI cross-encoders expose logits in this class order.
# (cross-encoder/nli-deberta-v3-* → 0: contradiction, 1: entailment, 2: neutral)
_DEBERTA_V3_LABEL_ORDER: Tuple[NLILabel, NLILabel, NLILabel] = (
    "contradiction",
    "entailment",
    "neutral",
)


class CrossEncoderNLI:
    """Production NLI backend: a DeBERTa-v3 MNLI cross-encoder.

    Uses `sentence_transformers.CrossEncoder` (already in requirements.txt).
    The model is loaded lazily on first use so importing this module stays
    cheap and torch-free until you actually score something.

    Label order is read from the model config's `id2label` when available and
    falls back to the DeBERTa-v3 convention otherwise, so swapping in a
    different MNLI checkpoint won't silently mislabel entailment/contradiction.

    Predictions are memoised in-process — greedy clustering re-compares against
    cluster representatives, so the same (premise, hypothesis) recurs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self._model: Any = None
        self._label_order: Tuple[NLILabel, ...] = _DEBERTA_V3_LABEL_ORDER
        self._cache: Dict[Tuple[str, str], NLILabel] = {}

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "CrossEncoderNLI needs sentence-transformers (and torch). Install:\n"
                "  pip install sentence-transformers>=2.7.0 torch>=2.2.0\n"
                "Or inject your own NLIBackend into SemanticEntropyDisagreement."
            ) from exc

        self._model = CrossEncoder(self.model_name, device=self.device)

        # Prefer the checkpoint's own label mapping if it advertises one.
        cfg = getattr(getattr(self._model, "model", None), "config", None)
        id2label = getattr(cfg, "id2label", None)
        if id2label:
            order: List[NLILabel] = []
            for i in range(len(id2label)):
                raw = str(id2label[i]).lower()
                if "contradict" in raw:
                    order.append("contradiction")
                elif "entail" in raw:
                    order.append("entailment")
                else:
                    order.append("neutral")
            if len(order) == 3:
                self._label_order = tuple(order)

    def predict(self, premise: str, hypothesis: str) -> NLILabel:
        return self.predict_batch([(premise, hypothesis)])[0]

    def predict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLILabel]:
        self._ensure_model()

        results: List[Optional[NLILabel]] = [None] * len(pairs)
        todo: List[Tuple[str, str]] = []
        todo_idx: List[int] = []
        for i, pair in enumerate(pairs):
            cached = self._cache.get(pair)
            if cached is not None:
                results[i] = cached
            else:
                todo.append(pair)
                todo_idx.append(i)

        if todo:
            import numpy as np  # local import; numpy is a hard dep anyway

            logits = self._model.predict(
                list(todo), batch_size=self.batch_size, convert_to_numpy=True
            )
            logits = np.asarray(logits)
            if logits.ndim == 1:  # single pair edge case
                logits = logits.reshape(1, -1)
            arg = logits.argmax(axis=1)
            for j, pair in enumerate(todo):
                label = self._label_order[int(arg[j])]
                self._cache[pair] = label
                results[todo_idx[j]] = label

        return [r for r in results]  # type: ignore[return-value]


def semantically_equivalent(
    nli: NLIBackend,
    a: str,
    b: str,
    *,
    context: Optional[str] = None,
    strict_entailment: bool = True,
    context_template: str = "{context} {text}",
) -> bool:
    """Standalone meaning-equivalence check reusing an NLI backend.

    Useful outside clustering — e.g. grading an agent's answer against a gold
    answer with the SAME NLI model that drives the disagreement measure.
    """
    a, b = _strip_reasoning(a).strip(), _strip_reasoning(b).strip()
    if not a or not b:
        return False
    if context:
        a = context_template.format(context=context.strip(), text=a)
        b = context_template.format(context=context.strip(), text=b)
    fwd, bwd = nli.predict_batch([(a, b), (b, a)])
    if strict_entailment:
        return fwd == "entailment" and bwd == "entailment"
    return fwd != "contradiction" and bwd != "contradiction"


# ---------------------------------------------------------------------------
# SemanticEntropyDisagreement
# ---------------------------------------------------------------------------

class SemanticEntropyDisagreement:
    """Disagreement = normalised entropy of NLI meaning-clusters.

    Parameters
    ----------
    nli               : an NLIBackend. Defaults to `CrossEncoderNLI()`
                        (constructed lazily so no torch import at __init__).
    strict_entailment : True → cluster only on mutual entailment (default);
                        False → cluster unless a direction contradicts.
    normalise         : True → score = H / ln(N) ∈ [0, 1] (default, comparable
                        to Jaccard); False → score = raw entropy in nats.
    linkage           : 'complete' (default) → a response joins a cluster only
                        if compatible with every member; 'single' → only the
                        representative (Kuhn/Farquhar shortcut). Complete is
                        more robust to noisy NLI on heterogeneous answers.
    context_template  : how the question/context is prepended for NLI. Default
                        "{context} {text}". Only used when a context is given.
    key_fn            : optional `(text, answer_type, question) -> Optional[frozenset]`.
                        `question` lets the key extractor subtract atoms that
                        are given in the question rather than asserted by the
                        answer (e.g. a year restated from the question is not
                        part of the answer) — see `make_short_answer_key_fn`.
                        When both responses in a pair reduce to a non-empty
                        key, equivalence is decided by subset-or-equal
                        (`ka <= kb or kb <= ka`) and the NLI model is not
                        consulted for that pair — see `_equivalent`. `None`
                        (default) reproduces the NLI-only behaviour exactly.
    """

    def __init__(
        self,
        nli: Optional[NLIBackend] = None,
        *,
        strict_entailment: bool = True,
        normalise: bool = True,
        context_template: str = "{context} {text}",
        linkage: str = "complete",
        key_fn: Optional[
            Callable[[str, Optional[str], Optional[str]], Optional[FrozenSet]]
        ] = None,
    ):
        self._nli = nli
        self.strict_entailment = bool(strict_entailment)
        self.normalise = bool(normalise)
        self.context_template = context_template
        if linkage not in ("complete", "single"):
            raise ValueError("linkage must be 'complete' or 'single'")
        self.linkage = linkage
        self.key_fn = key_fn

    @property
    def name(self) -> str:
        return "semantic_entropy"

    @property
    def nli(self) -> NLIBackend:
        """Lazily construct the default cross-encoder backend if none injected."""
        if self._nli is None:
            self._nli = CrossEncoderNLI()
        return self._nli

    # -- NLI-conditioning ------------------------------------------------

    def _condition(self, text: str, context: Optional[str]) -> str:
        if not context:
            return text
        return self.context_template.format(context=context.strip(), text=text)

    def _equivalent(
        self, a: str, b: str, context: Optional[str],
        answer_type: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], int]:
        """Are answers `a` and `b` the same meaning-class?

        Empty responses (e.g. an errored agent) never merge — they form their
        own singleton, which correctly registers as extra disagreement.

        Returns `(is_equivalent, key_verdict, emptied_by_subtraction)`:
          - `key_verdict` is `"equality"` or `"subset"` when `key_fn` merged
            the pair, `"differ"` when it split it, or `None` when NLI decided
            (see `score`'s `details["key_decided_comparisons"]`, which counts
            every non-`None` verdict).
          - `emptied_by_subtraction` counts how many of a/b's keys (0, 1 or
            2) were non-empty before question-subtraction but emptied by it,
            and so fell through to NLI instead of being key-decided —
            diagnostic only (`details["key_emptied_by_subtraction"]`).
        """
        emptied = 0
        if self.key_fn is not None:
            ka, kb = self.key_fn(a, answer_type, context), self.key_fn(b, answer_type, context)
            if context:
                if ka is None and self.key_fn(a, answer_type, None) is not None:
                    emptied += 1
                if kb is None and self.key_fn(b, answer_type, None) is not None:
                    emptied += 1
            if ka is not None and kb is not None:
                # Subset, not equality: a terse answer whose key is contained
                # in a more detailed answer's key asserts the same thing plus
                # less context, and should not be split. Equality punished
                # verbosity asymmetry and drove the Bucket A regression in v1.
                if ka == kb:
                    return True, "equality", emptied
                if ka <= kb or kb <= ka:
                    return True, "subset", emptied
                return False, "differ", emptied
        return semantically_equivalent(
            self.nli,
            a,
            b,
            context=context,
            strict_entailment=self.strict_entailment,
            context_template=self.context_template,
        ), None, emptied

    def _cluster(
        self, texts: Sequence[str], context: Optional[str],
        answer_type: Optional[str] = None,
    ) -> Tuple[List[List[int]], int, int, int, int, int]:
        """Greedy clustering.

        linkage='complete' (default): a response joins a cluster only if it is
        semantically compatible with EVERY member — robust to noisy NLI, since
        it doesn't assume transitivity of entailment through a representative.
        linkage='single': compares only to the cluster's first member (the
        Kuhn/Farquhar shortcut). Faster, but a wrong label on the
        representative pair can wrongly merge or split a whole cluster.

        Returns `(clusters, key_decided_comparisons, total_comparisons,
        key_decided_by_equality, key_decided_by_subset,
        key_emptied_by_subtraction)`.
        """
        clusters: List[List[int]] = []
        key_decided = 0
        total = 0
        key_equality = 0
        key_subset = 0
        key_emptied = 0
        for i, _t in enumerate(texts):
            for cluster in clusters:
                members = cluster if self.linkage == "complete" else cluster[:1]
                all_equiv = True
                for j in members:
                    eq, verdict, emptied = self._equivalent(texts[j], texts[i], context, answer_type)
                    total += 1
                    key_emptied += emptied
                    if verdict is not None:
                        key_decided += 1
                        if verdict == "equality":
                            key_equality += 1
                        elif verdict == "subset":
                            key_subset += 1
                    if not eq:
                        all_equiv = False
                        break
                if all_equiv:
                    cluster.append(i)
                    break
            else:
                clusters.append([i])
        return clusters, key_decided, total, key_equality, key_subset, key_emptied

    # -- public API ------------------------------------------------------

    def score(
        self,
        responses: Sequence[ResponseLike],
        *,
        context: Optional[str] = None,
        question: Optional[str] = None,
        answer_type: Optional[str] = None,
    ) -> DisagreementResult:
        """Cluster `responses` by meaning and return their entropy.

        `context`/`question` (aliases) is the shared question, prepended to
        every response before NLI so answers are compared *in context*.
        `answer_type` is forwarded to `key_fn` (if set) — e.g. SimpleQA's
        "Number" / "Date" / "Other" — and is otherwise unused.
        """
        ctx = context if context is not None else question
        texts = [_extract_text(r) for r in responses]
        n = len(texts)
        if n < 2:
            raise ValueError(f"Need at least 2 responses to compute disagreement; got {n}")

        (clusters, key_decided, total_compared,
         key_equality, key_subset, key_emptied) = self._cluster(texts, ctx, answer_type)
        sizes = [len(c) for c in clusters]
        probs = [s / n for s in sizes]

        entropy_nats = -sum(p * math.log(p) for p in probs if p > 0.0)
        max_entropy = math.log(n)  # n >= 2 → > 0
        entropy_norm = entropy_nats / max_entropy if max_entropy > 0 else 0.0
        # Clamp tiny FP drift into [0, 1].
        entropy_norm = min(1.0, max(0.0, entropy_norm))

        score = entropy_norm if self.normalise else entropy_nats

        return DisagreementResult(
            score=score,
            measure=self.name,
            n_responses=n,
            details={
                "n_clusters": len(clusters),
                "clusters": clusters,                       # list of index lists
                "cluster_sizes": sizes,
                "semantic_entropy_nats": entropy_nats,
                "semantic_entropy_normalised": entropy_norm,
                "representatives": [texts[c[0]][:160] for c in clusters],
                "strict_entailment": self.strict_entailment,
                "linkage": self.linkage,
                "context_used": bool(ctx),
                "key_fn_used": self.key_fn is not None,
                "key_decided_comparisons": key_decided,
                "total_comparisons": total_compared,
                "key_decided_by_equality": key_equality,
                "key_decided_by_subset": key_subset,
                "key_emptied_by_subtraction": key_emptied,
            },
        )