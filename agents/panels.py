"""Named agent panels for the disagreement experiments.

Centralises which models make up a "panel" so scripts don't each hard-code
model IDs — important right now because Groq's lineup is shifting (the Llama
3.x chat models were announced for deprecation on 2026-06-17 in favour of the
GPT-OSS and Qwen3 families).

Two panels matter for the shared-bias question (RQ3):

  same_family : two Meta Llama models of different sizes (8B, 70B). They share
                pretraining lineage, so a misconception baked into the family's
                data can fool BOTH — disagreement stays low even when both are
                wrong (the shared-bias failure mode). Kept as the size-ablation
                contrast to cross_family below.

  cross_family: one model per family — Meta (Llama-3.3-70B), OpenAI OSS
                (GPT-OSS-20B), Alibaba (Qwen3.6-27B) — so every member has a
                genuinely independent training lineage, unlike a same-family
                pair. This is the actual cross-family panel; it does NOT
                contain two Llamas (same_family_panel is the two-Llama
                composition, kept separately for the size-ablation contrast).

GPT-OSS and Qwen3 are reasoning models; GroqAgent keeps their
chain-of-thought out of `text` (via `groq_agent.REASONING_PARAMS`, resolved
automatically from the model id) — otherwise every disagreement measure sees
the reasoning trace, not the answer.
"""

from __future__ import annotations

from typing import List, Optional

from .groq_agent import GroqAgent


# --- model IDs (single source of truth; update here when Groq's lineup moves) --
LLAMA_SMALL = "llama-3.1-8b-instant"      # Meta, ~8B   (deprecation announced)
LLAMA_LARGE = "llama-3.3-70b-versatile"   # Meta, ~70B  (deprecation announced)
CROSS_FAMILY = "openai/gpt-oss-20b"       # OpenAI GPT-OSS, ~20B, different lineage
QWEN = "qwen/qwen3.6-27b"                 # Alibaba, ~27B, different lineage
                                           # (qwen3-32b was deprecated 2026; verified
                                           # working: finish_reason="stop", clean
                                           # output at reasoning_effort="none")

# Post-Llama fallback panel, once the Llama chat models are shut down.
GPT_OSS_SMALL = "openai/gpt-oss-20b"
GPT_OSS_LARGE = "openai/gpt-oss-120b"

# Family membership per model id, for panel-composition reporting (and for
# identifying "the Llama pair" by identity rather than by position in a
# panel — cross_family_panel contains one Llama, so "first two agents" is
# no longer a safe way to find the same-family subset).
MODEL_FAMILY: dict[str, str] = {
    LLAMA_SMALL: "Meta",
    LLAMA_LARGE: "Meta",
    CROSS_FAMILY: "OpenAI",
    GPT_OSS_LARGE: "OpenAI",
    QWEN: "Qwen",
}


def family_of(model: str) -> str:
    """Family name for `model` (falls back to the raw model id if unknown)."""
    return MODEL_FAMILY.get(model, model)


def make_agent(model: str, *, seed: Optional[int] = None, **kwargs) -> GroqAgent:
    """GroqAgent for `model`.

    Reasoning suppression (reasoning_format / reasoning_effort) is resolved
    automatically inside GroqAgent from `groq_agent.REASONING_PARAMS`, keyed
    on the model id — nothing to do here for that. A live probe on a
    TruthfulQA question confirmed REASONING_PARAMS alone is enough to keep
    completions well within a normal token budget and finish_reason="stop"
    (38 / 30 / 31 completion tokens for gpt-oss-20b / gpt-oss-120b /
    llama-3.1-8b-instant), so this no longer needs a bumped max_tokens for
    reasoning models — see `_assert_uniform_query_settings` below for why a
    per-model max_tokens would be a problem anyway.

    If `seed` is given it is merged into the request body (via extra_params →
    extra_body). Groq honours `seed` on a best-effort basis for
    reproducibility; crucially it is part of GroqAgent's cache key, so k
    differently-seeded agents produce k distinct cached samples instead of
    colliding on one.
    """
    rk = {"extra_params": {"seed": int(seed)}} if seed is not None else {}
    return GroqAgent(model=model, **{**rk, **kwargs})


def _assert_uniform_query_settings(agents: List[GroqAgent]) -> None:
    """All agents in a panel must share max_tokens, system_prompt, and temperature.

    Disagreement measures are sensitive to response length, so a panel is
    supposed to differ in *model* only — not in how generously or how
    concisely each model is asked. Reasoning params (resolved per-model from
    `groq_agent.REASONING_PARAMS`) are the one setting allowed to vary: they
    exist to *equalise* behaviour across models with incompatible reasoning
    mechanics (e.g. gpt-oss vs qwen3), not to differentiate it.
    """
    settings = {(a.max_tokens, a.system_prompt, a.temperature) for a in agents}
    assert len(settings) <= 1, (
        "Panel agents must share max_tokens/system_prompt/temperature — got "
        f"{ {a.name: (a.max_tokens, a.system_prompt, a.temperature) for a in agents} }"
    )


def same_family_panel(**kwargs) -> List[GroqAgent]:
    """Two Meta Llama models (8B + 70B) — shared lineage."""
    agents = [make_agent(LLAMA_SMALL, **kwargs), make_agent(LLAMA_LARGE, **kwargs)]
    _assert_uniform_query_settings(agents)
    return agents


def cross_family_panel(**kwargs) -> List[GroqAgent]:
    """One model per family (Meta, OpenAI OSS, Qwen) — genuinely independent
    training lineages. NOT a same-family pair plus one outsider; that
    composition (two Llamas) is `same_family_panel` instead."""
    agents = [
        make_agent(LLAMA_LARGE, **kwargs),
        make_agent(CROSS_FAMILY, **kwargs),
        make_agent(QWEN, **kwargs),
    ]
    _assert_uniform_query_settings(agents)
    return agents


# Base models used for the sampling panel (same three as cross_family_panel).
SAMPLING_BASE_MODELS = [LLAMA_LARGE, CROSS_FAMILY, QWEN]


def sampling_panel(k: int, temperature: float = 0.7, *,
                    base_models: Optional[List[str]] = None, **kwargs):
    """k temperature-sampled variants of each base model.

    `base_models` defaults to SAMPLING_BASE_MODELS (one model per family,
    same composition as cross_family_panel). Pass an explicit list to sample
    a different set — e.g. adding LLAMA_SMALL back in for a same-family
    contrast alongside the cross-family models.

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
    models = list(base_models) if base_models is not None else SAMPLING_BASE_MODELS
    groups = [
        [make_agent(m, seed=s, temperature=temperature, **kwargs) for s in range(k)]
        for m in models
    ]
    _assert_uniform_query_settings([a for g in groups for a in g])
    return groups, list(models)