"""agents/openai_compatible.py

Generic agent for any OpenAI-compatible chat endpoint (Together, DeepInfra,
Cerebras, ...). Satisfies the same Agent protocol as GroqAgent and returns the
same AgentResponse, so the pilot and disagreement measures are unchanged.

Uses the OpenAI SDK pointed at a provider base_url rather than each provider's
bespoke client, so one class covers every host. Caching, retries and error
mapping mirror GroqAgent: query() never raises on an API error — it populates
AgentResponse.error so a batch run doesn't abort on one failure (the pilot's
error-propagation stop relies on this).
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from openai import OpenAI

from .base import AgentResponse, _now_iso
from .cache import ResponseCache, make_cache_key


PROVIDER_BASE_URLS = {
    "together": "https://api.together.xyz/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
    "cerebras": "https://api.cerebras.ai/v1",
}

PROVIDER_KEY_ENV = {
    "together": "TOGETHER_API_KEY",
    "deepinfra": "DEEPINFRA_API_TOKEN",
    "cerebras": "CEREBRAS_API_KEY",
}


class OpenAICompatibleAgent:
    """One judge on an OpenAI-compatible endpoint.

    name convention: 'provider/model', matching GroqAgent, so cache filenames
    and result JSONL keys stay consistent across providers.
    """

    def __init__(
        self,
        model: str,
        provider: str = "together",
        system_prompt: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        max_retries: int = 4,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        default_max_tokens: int = 512,
    ) -> None:
        self._model = model
        self._provider = provider
        self._system_prompt = system_prompt
        self._cache = cache if cache is not None else ResponseCache()
        self._max_retries = max_retries
        self._default_max_tokens = default_max_tokens

        base_url = base_url or PROVIDER_BASE_URLS.get(provider)
        key_env = api_key_env or PROVIDER_KEY_ENV.get(provider, "")
        api_key = os.environ.get(key_env)
        if not api_key:
            raise RuntimeError(
                f"{key_env} not set — needed for provider '{provider}'."
            )
        if not base_url:
            raise RuntimeError(f"No base_url known for provider '{provider}'.")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return f"{self._provider}/{self._model}"

    @property
    def model(self) -> str:
        return self._model

    def query(self, prompt: str, **kwargs: Any) -> AgentResponse:
        temperature = kwargs.get("temperature", 0.0)
        max_tokens = kwargs.get("max_tokens", self._default_max_tokens)

        # Cache key includes the system prompt so editing a verifier/decomposer
        # prompt correctly invalidates — same contract GroqAgent relies on.
        key = make_cache_key(
            model=self.name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=self._system_prompt or "",
        )
        cached = self._cache.get(key)
        if cached is not None:
            return AgentResponse.from_dict(cached)

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(self._max_retries):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                out = AgentResponse(
                    text=choice.message.content or "",
                    model=self._model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    latency_s=time.time() - t0,
                    timestamp=_now_iso(),
                    finish_reason=choice.finish_reason,
                    usage={
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    } if usage else {},
                    provider=self._provider,
                )
                # Only cache successful, non-empty results — mirrors the reason
                # your quota-interrupted runs could resume cleanly.
                if out.text.strip():
                    self._cache.set(key, out.to_dict())
                return out
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                # Exponential backoff on rate limits / transient errors.
                time.sleep(min(2 ** attempt, 20))

        return AgentResponse(
            text="",
            model=self._model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            latency_s=0.0,
            timestamp=_now_iso(),
            finish_reason=None,
            error=f"{type(last_err).__name__}: {last_err}",
            provider=self._provider,
        )