"""Provider registry — run the same agent code against any OpenAI-compatible API.

`GroqAgent` is an OpenAI-compatible client with Groq defaults; swapping
`base_url` + key env points it at another provider with no other changes. This
module holds the endpoint table and a small factory.

Why this exists
---------------
1. Judge independence. The panel is Meta (Llama) + OpenAI (GPT-OSS), so every
   cheap Groq model is *in* the panel — grading with one is self-evaluation.
   A Google (Gemini) judge is outside the panel entirely.
2. Free-tier headroom. Groq's free tier caps tokens-per-day per model; Gemini's
   free tier is capped on requests-per-day instead, so a batched judge run fits.
3. The diversity experiment (RQ3) needs models from genuinely different
   developers, which means multi-provider regardless.

Gemini exposes an OpenAI-compatible endpoint, so it needs no special client.
Set the relevant key in `.env`:  GROQ_API_KEY=... / GEMINI_API_KEY=...

    from agents.providers import make_provider_agent
    judge = make_provider_agent("gemini-2.5-flash", temperature=0.0)
    judge = make_provider_agent("openai/gpt-oss-120b")   # -> groq
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
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def infer_provider(model: str) -> str:
    """Guess the provider from a model id.

    NOTE the trap: Groq namespaces its OpenAI open-weights models as
    'openai/gpt-oss-20b'. That leading 'openai/' is a Groq model id, NOT the
    OpenAI API — so anything that isn't clearly Gemini defaults to Groq, which
    is where this project's panel lives.
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

    `provider` is inferred from the model id when not given. Groq reasoning
    models (GPT-OSS / Qwen3) get their reasoning-suppression params applied
    automatically inside GroqAgent (from `groq_agent.REASONING_PARAMS`) —
    nothing to do here for the `apply_reasoning_defaults=True` case.
    `apply_reasoning_defaults=False` opts a Groq agent out of that (e.g. an
    ablation run) by forcing an empty override. Non-Groq providers never get
    reasoning params, since those params are Groq-specific.
    """
    prov = provider or infer_provider(model)
    if prov not in PROVIDERS:
        raise ValueError(f"Unknown provider {prov!r}. Known: {sorted(PROVIDERS)}")
    base_url, key_env = PROVIDERS[prov]

    extra: dict = {}
    if prov == "groq":
        if not apply_reasoning_defaults:
            extra = {"reasoning_params": {}}
        # else: leave reasoning_params unset, so GroqAgent's own
        # REASONING_PARAMS-by-model-id lookup applies, as documented above.
    else:
        # REASONING_PARAMS is keyed by MODEL ID ONLY, not (provider, model) —
        # GroqAgent.__init__ falls back to that table whenever reasoning_params
        # isn't given, regardless of which provider is asking. Left alone, a
        # non-Groq model id that happens to collide with a Groq table entry
        # (e.g. 'openai/gpt-oss-20b' exists on both Groq and OpenRouter) would
        # silently get Groq's reasoning_effort/reasoning_format sent to a
        # different endpoint. Force it empty here — still overridable via an
        # explicit `reasoning_params=` kwarg, since that's merged in after.
        extra = {"reasoning_params": {}}

    return GroqAgent(
        model=model,
        base_url=base_url,
        provider=prov,
        api_key_env=key_env,
        **{**extra, **kwargs},
    )