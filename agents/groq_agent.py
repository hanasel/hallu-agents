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

# Sensible, dataset-neutral defaults for this project. Dataset-specific
# tuning (e.g. TruthfulQA's short-answer budget) lives in the dataset layer
# — see data/query_config.py — and callers pass it in explicitly.
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_SYSTEM_PROMPT = None

# Retry policy for transient failures
MAX_RETRIES = 4
BACKOFF_BASE_S = 1.0
BACKOFF_FACTOR = 2.0

# HTTP status codes that mean "this will never succeed" — a bad model id, a
# revoked/invalid key, a decommissioned model. Retrying these just burns free-
# tier quota and stalls the run for MAX_RETRIES rounds of backoff per call.
PERMANENT_STATUS_CODES = frozenset({400, 401, 403, 404, 422})

# Error strings are prefixed with this so callers (harness scripts) can tell
# "give up, config is wrong" apart from "give up, retries exhausted" without
# the Agent protocol having to raise (see `Agent.query` in base.py — it must
# never raise, so batch runs don't abort on one bad question).
PERMANENT_ERROR_PREFIX = "PERMANENT"

# error_kind values on AgentResponse. `error` above stays a human-readable
# (PERMANENT-prefixed, where relevant) string; this is the field callers
# should branch on.
ERROR_KIND_PERMANENT = "permanent"
ERROR_KIND_TRANSIENT = "transient"

# Per-model Groq reasoning-parameter capability table. Groq's reasoning
# models each expose a different, incompatible subset of
# {reasoning_format, reasoning_effort}:
#   - qwen/qwen3.6-27b     : reasoning_format (hidden|raw|parsed) is supported.
#                            reasoning_format="hidden" alone drops the trace
#                            but still *generates* it internally before
#                            discarding it — without reasoning_effort="none"
#                            to actually turn reasoning off, a low max_tokens
#                            budget can be consumed entirely by the hidden
#                            trace, leaving zero tokens for the answer.
#   - openai/gpt-oss-20b,
#     openai/gpt-oss-120b  : reasoning_format is NOT supported — sending it
#                            errors. Reasoning comes back in a separate
#                            `reasoning` field on the assistant message
#                            regardless; reasoning_effort (low|medium|high)
#                            only controls how much of it there is.
# Non-reasoning models (llama-3.1-8b-instant, llama-3.3-70b-versatile,
# allam-2-7b, ...) are deliberately absent — sending either param to them may
# error, so they get nothing.
REASONING_PARAMS: dict[str, dict] = {
    "qwen/qwen3.6-27b": {"reasoning_effort": "none", "reasoning_format": "hidden"},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}


class PermanentAgentError(RuntimeError):
    """Raised at panel-construction time, not from query() — see
    `assert_models_available`. A config problem (wrong model id, dead model),
    not a transient API failure."""


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
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        extra_params: Optional[dict] = None,
        reasoning_params: Optional[dict] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        api_key_env: Optional[str] = None,
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
                         None (default) means no system message. Dataset-
                         specific prompts (e.g. TruthfulQA's short-answer
                         instruction) live in data/query_config.py — pass
                         them in explicitly, e.g.
                             GroqAgent(**TRUTHFULQA_QUERY_CONFIG)
                         Baked into the cache key so different system prompts
                         don't collide.
        extra_params   : Optional dict of arbitrary Groq-specific request-body
                         params (e.g. `seed`). Sent via the OpenAI SDK's
                         `extra_body` escape hatch — NOT as top-level create()
                         kwargs, which the SDK rejects with TypeError. Merged
                         with (and taking precedence over) the resolved
                         reasoning params below. Baked into the cache key so
                         runs with different params don't collide.
        reasoning_params : Override for this model's entry in REASONING_PARAMS
                         (module-level capability table). None (default) means
                         "use the table lookup for `model`" — most models
                         aren't in the table and get no reasoning params at
                         all. Pass an explicit dict (even {}) to replace the
                         table entry, e.g. to ablate reasoning_effort:
                             GroqAgent("openai/gpt-oss-120b",
                                       reasoning_params={"reasoning_effort": "high"})
                         Baked into the cache key so runs at different
                         reasoning settings never collide.
        base_url       : OpenAI-compatible endpoint. Defaults to Groq's. Set this
                         to point the same code at another provider (Gemini,
                         OpenAI, Together, ...) — see `agents/providers.py`.
        provider       : Short provider tag used in `.name` (and hence the cache
                         filename), e.g. 'groq', 'gemini'. Defaults to 'groq'.
        api_key_env    : Environment variable holding the key. Defaults to
                         'GROQ_API_KEY'.
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
        # None => look up this model's capability-table entry; an explicit
        # dict (including {}) replaces it outright — see the constructor
        # docstring for the ablation use case.
        self.reasoning_params: dict = (
            dict(REASONING_PARAMS.get(model, {}))
            if reasoning_params is None
            else dict(reasoning_params)
        )

        # Provider wiring. Defaults reproduce the original Groq-only behaviour
        # exactly; overriding these points the same client at any other
        # OpenAI-compatible endpoint.
        self.provider = provider or type(self).provider
        self._api_key_env = api_key_env or "GROQ_API_KEY"
        self._base_url = base_url or GROQ_BASE_URL

        resolved_key = api_key or os.environ.get(self._api_key_env)
        if not resolved_key:
            raise RuntimeError(
                f"{self._api_key_env} not set. Set the env var or add it to a "
                f"`.env` file at your project root:\n"
                f"  {self._api_key_env}=..."
            )

        self._client = OpenAI(base_url=self._base_url, api_key=resolved_key)

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

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> Any:
        """The underlying OpenAI-SDK client. Exposed for preflight checks
        (`assert_models_available`) that need to call `.models.list()`."""
        return self._client

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

        # ---- Resolve request-body extras ----
        # extra_params (arbitrary, e.g. seed) wins on key collisions with the
        # model's resolved reasoning params (reasoning_format/reasoning_effort).
        request_extra = {**self.reasoning_params, **self.extra_params}

        # ---- Cache lookup ----
        # Everything that affects the response must be in the key. That
        # includes the system prompt, extra_params, and the resolved
        # reasoning params, so two agents that differ only in those don't
        # share cached responses — a run at reasoning_effort='low' must never
        # collide with one at 'medium'.
        key_extra: dict = {}
        if self.system_prompt:
            key_extra["system_prompt"] = self.system_prompt
        if self.extra_params:
            key_extra["extra_params"] = self.extra_params
        if self.reasoning_params:
            key_extra["reasoning_params"] = self.reasoning_params
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
        # Imported lazily (not at module scope) so the package stays importable
        # without `openai` installed — see the module docstring.
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        last_error: Optional[str] = None
        last_error_kind: Optional[str] = None
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=maxt,
                    extra_body=request_extra or None,
                )
                latency = time.perf_counter() - t0
                choice = completion.choices[0]
                text = choice.message.content or ""
                # gpt-oss (and qwen in 'parsed' mode) return the
                # chain-of-thought in a separate `reasoning` field rather than
                # embedding it in `content` — capture it for inspection, but
                # it must never end up inside `text`.
                reasoning_text = getattr(choice.message, "reasoning", None) or None
                assert not reasoning_text or reasoning_text not in text, (
                    f"reasoning field leaked into response text for model {self._model!r}"
                )
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
                    reasoning=reasoning_text,
                    error_kind=None,
                )
                if use_cache and self._cache is not None:
                    self._cache.put(cache_key, response)
                return response

            except APIStatusError as exc:
                # Subsumes RateLimitError (429) and InternalServerError (5xx),
                # as well as the permanent 4xx errors below — all are
                # APIStatusError subclasses with a real .status_code.
                last_error = f"{exc.status_code} {type(exc).__name__}: {exc}"
                if exc.status_code in PERMANENT_STATUS_CODES:
                    # Config problem (bad model id, bad key, decommissioned
                    # model) — retrying cannot help. Stop hitting the API and
                    # go straight to the error response instead of burning
                    # MAX_RETRIES more calls (and the backoff sleep) on it.
                    last_error = f"{PERMANENT_ERROR_PREFIX} {last_error}"
                    last_error_kind = ERROR_KIND_PERMANENT
                    break
                last_error_kind = ERROR_KIND_TRANSIENT
                if attempt < MAX_RETRIES:
                    # Groq sends Retry-After on 429s; honour it instead of
                    # guessing with exponential backoff — on the free tier the
                    # server's number is usually both more accurate and
                    # shorter than ours.
                    retry_after = exc.response.headers.get("retry-after")
                    delay = (float(retry_after) if retry_after
                              else BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt))
                    time.sleep(delay)
                # else: fall through to final error response

            except (APIConnectionError, APITimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_error_kind = ERROR_KIND_TRANSIENT
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt)
                    time.sleep(delay)
                # else: fall through to final error response

            except Exception as exc:
                # Anything else (SDK bug, our own bug, non-API exception) —
                # unknown shape, so treat as transient rather than silently
                # mislabelling it permanent.
                last_error = f"{type(exc).__name__}: {exc}"
                last_error_kind = ERROR_KIND_TRANSIENT
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_BASE_S * (BACKOFF_FACTOR ** attempt)
                    time.sleep(delay)
                # else: fall through to final error response

        # ---- All retries exhausted (or a permanent error broke early) ----
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
            reasoning=None,
            error_kind=last_error_kind,
        )

    def cache_stats(self) -> Optional[dict]:
        """Cache statistics, or None if caching is disabled."""
        return self._cache.stats() if self._cache is not None else None


# Runtime protocol check (mostly for IDE / mypy peace of mind)
assert isinstance(GroqAgent.__mro__[0], type)


def assert_models_available(agents: "list[GroqAgent]") -> None:
    """Fail fast if any agent's model isn't currently served.

    Costs one `models.list()` call per distinct endpoint (grouped by
    `base_url`) — no tokens, no chat completion — run once before any
    question is sent. Turns a decommissioned or typo'd model id into an
    immediate, loud failure instead of what actually happens without this:
    every query for that model exhausts MAX_RETRIES of backoff (permanent
    errors now break early, but a caller that doesn't check `.error` per
    response still silently drops every one of them), so a partial-panel
    run keeps going and looks like real data until someone notices the
    numbers are off.

    Call this right after building a panel (`same_family_panel()` /
    `cross_family_panel()` / etc.), before the canary/preflight query.
    """
    by_endpoint: dict[str, list[GroqAgent]] = {}
    for a in agents:
        by_endpoint.setdefault(a.base_url, []).append(a)

    for base_url, group in by_endpoint.items():
        try:
            active = {m.id for m in group[0].client.models.list().data}
        except Exception as exc:
            raise PermanentAgentError(
                f"Could not list models at {base_url}: {type(exc).__name__}: {exc}"
            ) from exc
        wanted = {a.model for a in group}
        missing = wanted - active
        if missing:
            raise PermanentAgentError(
                f"Not available at {base_url}: {sorted(missing)}\n"
                f"Active models: {sorted(active)}"
            )