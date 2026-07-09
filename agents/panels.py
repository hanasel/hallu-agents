"""Named agent panels for the disagreement experiments.

Centralises which models make up a "panel" so scripts don't each hard-code
model IDs — important right now because Groq's lineup is shifting (the Llama
3.x chat models were announced for deprecation on 2026-06-17 in favour of the
GPT-OSS and Qwen3 families).

Two panels matter for the shared-bias question (RQ3):

  same_family : two Meta Llama models of different sizes (8B, 70B). They share
                pretraining lineage, so a misconception baked into the family's
                data can fool BOTH — disagreement stays low even when both are
                wrong (the shared-bias failure mode).

  cross_family: adds a model from a DIFFERENT lineage (OpenAI GPT-OSS-20B) so
                the panel spans genuinely independent training data. If the
                shared-bias hypothesis holds, this agent should *break ties* on
                exactly the questions where the two Llamas agree-but-are-wrong.

GPT-OSS is a reasoning model; `reasoning_format="hidden"` keeps its
chain-of-thought out of the answer body (otherwise every disagreement measure
sees the reasoning trace, not the answer).
"""

from __future__ import annotations

from typing import List, Optional

from .groq_agent import GroqAgent


# --- model IDs (single source of truth; update here when Groq's lineup moves) --
LLAMA_SMALL = "llama-3.1-8b-instant"      # Meta, ~8B   (deprecation announced)
LLAMA_LARGE = "llama-3.3-70b-versatile"   # Meta, ~70B  (deprecation announced)
CROSS_FAMILY = "openai/gpt-oss-20b"       # OpenAI GPT-OSS, ~20B, different lineage

# Post-Llama fallback panel, once the Llama chat models are shut down.
GPT_OSS_SMALL = "openai/gpt-oss-20b"
GPT_OSS_LARGE = "openai/gpt-oss-120b"
QWEN = "qwen/qwen3.6-27b"                  # Alibaba (qwen3-32b was deprecated 2026)

# Reasoning models emit a chain-of-thought that consumes the completion budget
# before the answer. With effort='low' the trace is short (tens of tokens), so
# the 512 default is usually fine — but harder questions can reason longer, so
# give real headroom to avoid an occasional length-truncated (empty) answer.
# (NB: these params must reach Groq via extra_body, which GroqAgent handles;
# passing them as top-level create() kwargs raises TypeError -> empty stub.)
REASONING_MAX_TOKENS = 2048


def _reasoning_kwargs(model: str) -> dict:
    """Extra GroqAgent kwargs per model.

    Reasoning models need (a) a larger token budget so the answer isn't
    truncated to empty, and (b) reasoning suppressed/limited so the trace stays
    out of the answer body and stays comparable to the non-reasoning Llamas.
    The `reasoning_effort` vocabulary differs by family:
      - GPT-OSS: 'low' | 'medium' | 'high'   -> use 'low' for direct QA answers
      - Qwen3  : 'none' (disable) | 'default' -> use 'none' for a direct answer
    """
    if "gpt-oss" in model:
        return {
            "max_tokens": REASONING_MAX_TOKENS,
            "extra_params": {"reasoning_format": "hidden", "reasoning_effort": "low"},
        }
    if "qwen3" in model:
        return {
            "max_tokens": REASONING_MAX_TOKENS,
            "extra_params": {"reasoning_format": "hidden", "reasoning_effort": "none"},
        }
    return {}


def make_agent(model: str, *, seed: Optional[int] = None, **kwargs) -> GroqAgent:
    """GroqAgent for `model` with reasoning-format handling applied.

    If `seed` is given it is merged into the request body (via extra_params →
    extra_body) alongside any reasoning params. Groq honours `seed` on a
    best-effort basis for reproducibility; crucially it is part of GroqAgent's
    cache key, so k differently-seeded agents produce k distinct cached samples
    instead of colliding on one.
    """
    rk = _reasoning_kwargs(model)
    if seed is not None:
        extra = dict(rk.get("extra_params", {}))
        extra["seed"] = int(seed)
        rk = {**rk, "extra_params": extra}
    return GroqAgent(model=model, **{**rk, **kwargs})


def same_family_panel(**kwargs) -> List[GroqAgent]:
    """Two Meta Llama models (8B + 70B) — shared lineage."""
    return [make_agent(LLAMA_SMALL, **kwargs), make_agent(LLAMA_LARGE, **kwargs)]


def cross_family_panel(**kwargs) -> List[GroqAgent]:
    """Same-family pair + one cross-family (GPT-OSS) agent."""
    return same_family_panel(**kwargs) + [make_agent(CROSS_FAMILY, **kwargs)]


# Base models used for the sampling panel (same three as cross_family_panel).
SAMPLING_BASE_MODELS = [LLAMA_SMALL, LLAMA_LARGE, CROSS_FAMILY]


def sampling_panel(k: int, temperature: float = 0.7, **kwargs):
    """k temperature-sampled variants of each base model.

    Returns (groups, base_models) where groups[m] is the list of k seeded
    GroqAgents for base_models[m]. Pooling all k·len(base) responses per
    question and clustering them gives semantic entropy a continuous range
    (0 … ln(k·M)) instead of the coarse 3-value set you get with one sample
    per model — this is the Farquhar/Kuhn multi-sample setup.

    Requires temperature > 0 for the samples to actually differ.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 1 and temperature <= 0:
        raise ValueError("k>1 needs temperature>0, else all samples are identical")
    groups = [
        [make_agent(m, seed=s, temperature=temperature, **kwargs) for s in range(k)]
        for m in SAMPLING_BASE_MODELS
    ]
    return groups, list(SAMPLING_BASE_MODELS)