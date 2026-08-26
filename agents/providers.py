"""Provider registry — run the same agent code against any OpenAI-compatible API.

`GroqAgent` is an OpenAI-compatible client with Groq defaults; swapping
`base_url` + key env points it at another provider with no other changes. This
module holds the endpoint table and a small factory.

Why this exists
---------------
1. Judge independence. The panel is Meta (Llama), OpenAI (GPT-OSS) and Qwen,
   so a judge drawn from any of those families is self-evaluation. What makes
   a judge independent is its training lineage, not which endpoint serves it
   — a Google (Gemini) model routed through OpenRouter is just as external to
   the panel as one reached through Google's own endpoint. `--judge-provider`
   on scripts/disagreement_pilot.py defaults to "openrouter" for exactly this
   reason; pass a different provider tag only if the judge model isn't
   available there.
2. The diversity experiment (RQ3) needs models from genuinely different
   developers, which means multi-provider regardless.

(Historically also motivated by free-tier request-quota headroom on Gemini's
own endpoint vs Groq's token-quota; that no longer applies on a paid
OpenRouter account, but the Gemini provider tag below still works if you ever
need Google's endpoint directly.)

Gemini exposes an OpenAI-compatible endpoint, so it needs no special client.
Set the relevant key in `.env`:  OPENROUTER_API_KEY=... / GROQ_API_KEY=... /
GEMINI_API_KEY=...

    from agents.providers import make_provider_agent
    judge = make_provider_agent("gemini-2.5-flash", temperature=0.0)
    judge = make_provider_agent("openai/gpt-oss-120b", provider="groq")
    judge = make_provider_agent("meta-llama/llama-3.3-70b-instruct", provider="openrouter")
"""

from __future__ import annotations

from typing import Optional

from .groq_agent import GroqAgent, GROQ_BASE_URL

# Provider tag -> (base_url, api-key env var)
PROVIDERS: dict[str, tuple[str, str]] = {
    "groq": (GROQ_BASE_URL, "GROQ_API_KEY"),
    # Google's OpenAI-compatibility layer for the Gemini API.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",
               "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    # Single OpenAI-compatible endpoint fronting many upstream providers, kept
    # as its own tag (not folded into any upstream's model-id namespace) since
    # OpenRouter's own model ids ('meta-llama/...', 'qwen/...', ...) are a
    # third naming scheme, distinct from both Groq's and each upstream's own.
    # `agents.panels` builds its panel agents on this provider by default —
    # Groq's free tier rate-limits and its Llama 3.x deprecation schedule made
    # it an unreliable panel backend; see agents/panels.py's module docstring.
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

# OpenRouter's reasoning-suppression request shape is a single nested
# `reasoning` object (`{"effort": ..., "exclude": ...}`), NOT Groq's flat
# `reasoning_effort`/`reasoning_format` keys — a different provider, a
# different API shape, so this is a separate table from
# `groq_agent.REASONING_PARAMS`, keyed by OpenRouter's own model ids (which
# mostly differ from Groq's, e.g. 'meta-llama/llama-3.3-70b-instruct' vs
# Groq's 'llama-3.3-70b-versatile' — except gpt-oss and this Qwen build,
# whose ids happen to be identical strings on both providers; see the
# `make_provider_agent` guard below for why that collision matters).
# `exclude: true` keeps the trace out of the response entirely (Groq's
# 'hidden' reasoning_format equivalent); `effort` controls how many tokens
# the model spends reasoning before answering — kept low/none, same
# rationale as groq_agent.REASONING_PARAMS: an unbounded hidden trace can
# eat a short max_tokens budget before any answer text is produced.
OPENROUTER_REASONING_PARAMS: dict[str, dict] = {
    "qwen/qwen3.6-27b": {"reasoning": {"effort": "none", "exclude": True}},
    "qwen/qwen3.6-35b-a3b": {"reasoning": {"effort": "none", "exclude": True}},
    "openai/gpt-oss-20b": {"reasoning": {"effort": "low", "exclude": True}},
    "openai/gpt-oss-120b": {"reasoning": {"effort": "low", "exclude": True}},
    "qwen/qwen3.6-plus": {"reasoning": {"effort": "none", "exclude": True}},
    "openai/gpt-5.6-sol": {"reasoning": {"effort": "none", "exclude": True}},
    "meta-llama/llama-4-maverick": {"reasoning": {"effort": "none", "exclude": True}},
}


def infer_provider(model: str) -> str:
    """Guess the provider from a model id.

    NOTE the trap: Groq namespaces its OpenAI open-weights models as
    'openai/gpt-oss-20b'. That leading 'openai/' is a Groq model id, NOT the
    OpenAI API — so anything that isn't clearly Gemini defaults to Groq here.
    `agents.panels` does NOT rely on this inference — it passes
    `provider="openrouter"` explicitly (see agents/panels.py) — so this
    fallback only matters for ad-hoc/CLI-supplied model ids elsewhere in the
    codebase (e.g. scripts/harvest_selfcheck.py's `--model` without
    `--provider`).
    """
    m = model.lower()
    if m.startswith("gemini") or m.startswith("models/gemini"):
        return "gemini"
    return "groq"


def make_provider_agent(
    model: str,
    provider: Optional[str] = None,
    *,
    apply_reasoning_defaults: bool = True,
    **kwargs,
) -> GroqAgent:
    """Build an agent for `model` on the right provider.

    `provider` is inferred from the model id when not given. Reasoning models
    (GPT-OSS / Qwen3) get their reasoning-suppression params resolved here,
    per provider, for the `apply_reasoning_defaults=True` case:
      - groq       : nothing to do here — GroqAgent's own
                     `groq_agent.REASONING_PARAMS`-by-model-id lookup applies
                     automatically when `reasoning_params` isn't given.
      - openrouter : resolved from `OPENROUTER_REASONING_PARAMS` above
                     (a different API shape — see that table's docstring).
      - anything else : no reasoning params (none defined for other
                     providers yet).
    `apply_reasoning_defaults=False` opts out of all of the above (e.g. an
    ablation run) by forcing an empty override — still overridable per-call
    via an explicit `reasoning_params=` kwarg, since that's merged in after.

    Table lookups are keyed by MODEL ID ONLY, not (provider, model) —
    GroqAgent.__init__ falls back to `groq_agent.REASONING_PARAMS` whenever
    `reasoning_params` isn't given, regardless of which provider is asking.
    Left unguarded, a model id that collides across providers (e.g.
    'openai/gpt-oss-20b' and 'qwen/qwen3.6-27b' both exist, under those exact
    strings, on Groq AND OpenRouter) would silently get one provider's
    reasoning params sent to the other's endpoint. Every non-groq branch
    below sets `reasoning_params` explicitly (falling back to `{}` if the
    model has no table entry) specifically to block that fallthrough.
    """
    prov = provider or infer_provider(model)
    if prov not in PROVIDERS:
        raise ValueError(f"Unknown provider {prov!r}. Known: {sorted(PROVIDERS)}")
    base_url, key_env = PROVIDERS[prov]

    extra: dict = {}
    if not apply_reasoning_defaults:
        extra = {"reasoning_params": {}}
    elif prov == "groq":
        pass  # GroqAgent's own REASONING_PARAMS-by-model-id lookup applies.
    elif prov == "openrouter":
        extra = {"reasoning_params": dict(OPENROUTER_REASONING_PARAMS.get(model, {}))}
    else:
        extra = {"reasoning_params": {}}

    return GroqAgent(
        model=model,
        base_url=base_url,
        provider=prov,
        api_key_env=key_env,
        **{**extra, **kwargs},
    )