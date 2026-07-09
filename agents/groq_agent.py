"""GroqAgent — first concrete agent implementation.

Uses the OpenAI SDK against Groq's OpenAI-compatible endpoint. The exact
same code works for OpenAI, Together, HuggingFace Inference Providers,
or any other OpenAI-compatible provider by swapping `base_url` and API key.

Defaults chosen for this project:
  - Model: `llama-3.1-8b-instant` (Groq's fastest / cheapest)
  - Temperature: 0.0 — see rationale below.

Why temperature=0.0 (not 0.7 like SelfCheckGPT)
-----------------------------------------------
SelfCheckGPT samples the same model at HIGH temperature so that sampling
noise gives it a within-model consistency signal. This project's signal
comes from *model differences*, not sampling noise. Setting T=0.0 on each
agent means any disagreement observed across agents is attributable to
their differing knowledge, not to stochastic sampling — which is exactly
what we want for RQ1/RQ3.

Environment
-----------
Requires `GROQ_API_KEY` in the environment (or a `.env` file at the project
root; `python-dotenv` will pick it up).
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from .base import Agent, AgentResponse, _now_iso
from .cache import ResponseCache, make_cache_key


# Groq's OpenAI-compatible endpoint
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Sensible defaults for this project
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# Retry policy for transient failures
MAX_RETRIES = 4
BACKOFF_BASE_S = 1.0
BACKOFF_FACTOR = 2.0


class GroqAgent:
    """A single Groq-served open-weights model as an agent.

    Implements the `agents.base.Agent` protocol.
    """

    provider = "groq"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        cache_enabled: bool = True,
        cache_dir: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extra_params: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        model          : Groq model id, e.g. 'llama-3.1-8b-instant'.
        temperature    : 0.0 for deterministic (default; see module docstring).
        max_tokens     : Max response tokens.
        api_key        : Overrides GROQ_API_KEY from environment.
        cache          : Pre-built cache (advanced use). If None and
                         cache_enabled=True, a default cache is created.
        cache_enabled  : Set False to bypass caching entirely.
        cache_dir      : Directory for cache files.
        system_prompt  : Optional system message prepended to every call.
                         Baked into the cache key so different system prompts
                         don't collide.
        extra_params   : Optional dict of Groq-specific request-body params.
                         Sent via the OpenAI SDK's `extra_body` escape hatch —
                         NOT as top-level create() kwargs, which the SDK rejects
                         with TypeError. Use for provider-/model-specific
                         settings, e.g. reasoning models on Groq (GPT-OSS, Qwen3)
                         that must hide/limit chain-of-thought:
                             extra_params={"reasoning_format": "hidden",
                                           "reasoning_effort": "low"}
                         Without this, the reasoning trace leaks into the answer
                         body and corrupts every disagreement measure. Baked into
                         the cache key so runs with different params don't collide.
        """
        # Lazy import so the package can be imported without `openai` installed
        # (useful for anyone only using the data layer).
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GroqAgent. Install with:\n"
                "  pip install openai>=1.0"
            ) from exc

        # Load .env if present, so users don't have to set env vars manually.
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except ImportError:
            pass

        self._model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt
        self.extra_params = dict(extra_params) if extra_params else {}

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get one at https://console.groq.com "
                "and either set the env var or add it to a `.env` file at "
                "your project root:\n"
                "  GROQ_API_KEY=gsk_..."
            )

        self._client = OpenAI(base_url=GROQ_BASE_URL, api_key=resolved_key)

        if cache is not None:
            self._cache = cache
        elif cache_enabled:
            self._cache = ResponseCache(
                agent_name=self.name,
                cache_dir=cache_dir if cache_dir is not None else "cache/agent_responses",
            )
        else:
            self._cache = None

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"{self.provider}/{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def query(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **_: Any,
    ) -> AgentResponse:
        """Send `prompt` to the model and return the response.

        Cache is consulted first when enabled. On API failure, retries with
        exponential backoff; if all retries fail, returns an `AgentResponse`
        with `error` populated (never raises).
        """
        temp = self.temperature if temperature is None else float(temperature)
        maxt = self.max_tokens if max_tokens is None else int(max_tokens)

        # ---- Cache lookup ----
        # Everything that affects the response must be in the key. That
        # includes the system prompt and any extra_params (e.g. a reasoning
        # model's reasoning_format), so two agents that differ only in those
        # don't share cached responses.
        key_extra: dict = {}
        if self.system_prompt:
            key_extra["system_prompt"] = self.system_prompt
        if self.extra_params:
            key_extra["extra_params"] = self.extra_params
        cache_key = make_cache_key(
            model=self._model,
            prompt=prompt,
            temperature=temp,
            max_tokens=maxt,
            extra=key_extra or None,
        )
        if use_cache and self._cache is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        # ---- Build messages ----
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ---- API call with retries ----
        last_error: Optional[str] = None
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=maxt,
                    extra_body=self.extra_params or None,
                )
                latency = time.perf_counter() - t0
                choice = completion.choices[0]
                text = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)
                usage_obj = getattr(completion, "usage", None)
                usage = {}
                if usage_obj is not None:
                    usage = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                        "total_tokens": getattr(usage_obj, "total_tokens", 0),
                    }

                response = AgentResponse(
                    text=text,
                    model=self._model,
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=maxt,
                    latency_s=latency,
                    timestamp=_now_iso(),
                    finish_reason=finish_reason,
                    usage=usage,
                    error=None,
                    provider=self.provider,
                )
                if use_cache and self._cache is not None:
                    self._cache.put(cache_key, response)
                return response

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt)
                    time.sleep(delay)
                # else: fall through to final error response

        # ---- All retries exhausted ----
        return AgentResponse(
            text="",
            model=self._model,
            prompt=prompt,
            temperature=temp,
            max_tokens=maxt,
            latency_s=0.0,
            timestamp=_now_iso(),
            finish_reason=None,
            usage={},
            error=last_error or "unknown_error",
            provider=self.provider,
        )

    def cache_stats(self) -> Optional[dict]:
        """Cache statistics, or None if caching is disabled."""
        return self._cache.stats() if self._cache is not None else None


# Runtime protocol check (mostly for IDE / mypy peace of mind)
assert isinstance(GroqAgent.__mro__[0], type)