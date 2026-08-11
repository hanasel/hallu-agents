"""Tests for agents.groq_agent: reasoning-model support and error classification.

No network calls: `GroqAgent.__init__` never talks to the API (the OpenAI
client is lazy), so we build real agents and monkeypatch
`agent._client.chat.completions.create` directly. Retryable-error tests
construct real `openai.APIStatusError` instances (backed by a real
`httpx.Response`) so the `except APIStatusError` branch in
`GroqAgent.query` is exercised exactly as it runs in production.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from agents.cache import make_cache_key
from agents.groq_agent import MAX_RETRIES, REASONING_PARAMS, GroqAgent

API_KEY = "test-key"

REASONING_MODELS = sorted(REASONING_PARAMS)
NON_REASONING_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "allam-2-7b",
]


def build_agent(model: str, **kwargs) -> GroqAgent:
    kwargs.setdefault("cache_enabled", False)
    return GroqAgent(model=model, api_key=API_KEY, **kwargs)


def fake_completion(content: str = "Paris.", reasoning=None, finish_reason: str = "stop"):
    """A minimal stand-in for an SDK ChatCompletion, exposing only the
    attributes GroqAgent.query actually reads."""
    message = types.SimpleNamespace(content=content, reasoning=reasoning)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def api_status_error(status_code: int, retry_after: "str | None" = None) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        status_code=status_code,
        request=request,
        headers=headers,
        json={"error": {"message": "boom"}},
    )
    return openai.APIStatusError(f"{status_code} error", response=response, body=None)


# ---------------------------------------------------------------------------
# Per-model reasoning-parameter capability table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", REASONING_MODELS)
def test_reasoning_model_sends_exactly_its_table_entry(model):
    agent = build_agent(model)
    mock_create = MagicMock(return_value=fake_completion())
    agent._client.chat.completions.create = mock_create

    agent.query("What is the capital of France?")

    _, kwargs = mock_create.call_args
    assert kwargs["extra_body"] == REASONING_PARAMS[model]


@pytest.mark.parametrize("model", NON_REASONING_MODELS)
def test_non_reasoning_model_sends_no_reasoning_params(model):
    agent = build_agent(model)
    mock_create = MagicMock(return_value=fake_completion())
    agent._client.chat.completions.create = mock_create

    agent.query("What is the capital of France?")

    _, kwargs = mock_create.call_args
    assert kwargs["extra_body"] is None


def test_reasoning_params_overridable_for_ablation():
    # Ablating reasoning_effort on a gpt-oss model: the constructor arg
    # replaces (not merges with) the table entry.
    agent = build_agent("openai/gpt-oss-120b", reasoning_params={"reasoning_effort": "high"})
    mock_create = MagicMock(return_value=fake_completion())
    agent._client.chat.completions.create = mock_create

    agent.query("q")

    _, kwargs = mock_create.call_args
    assert kwargs["extra_body"] == {"reasoning_effort": "high"}


def test_reasoning_params_override_can_disable_table_entry():
    agent = build_agent("qwen/qwen3.6-27b", reasoning_params={})
    mock_create = MagicMock(return_value=fake_completion())
    agent._client.chat.completions.create = mock_create

    agent.query("q")

    _, kwargs = mock_create.call_args
    assert kwargs["extra_body"] is None


# ---------------------------------------------------------------------------
# `reasoning` field capture
# ---------------------------------------------------------------------------

def test_reasoning_field_captured_and_kept_out_of_text():
    agent = build_agent("openai/gpt-oss-120b")
    mock_create = MagicMock(return_value=fake_completion(
        content="Paris.",
        reasoning="Let me think step by step about French capitals...",
    ))
    agent._client.chat.completions.create = mock_create

    resp = agent.query("What is the capital of France?")

    assert resp.text == "Paris."
    assert resp.reasoning == "Let me think step by step about French capitals..."
    assert resp.reasoning not in resp.text


def test_absent_reasoning_field_leaves_reasoning_none():
    agent = build_agent("llama-3.1-8b-instant")
    mock_create = MagicMock(return_value=fake_completion(content="Paris.", reasoning=None))
    agent._client.chat.completions.create = mock_create

    resp = agent.query("q")

    assert resp.reasoning is None
    assert resp.text == "Paris."


# ---------------------------------------------------------------------------
# Cache-key isolation by reasoning params
# ---------------------------------------------------------------------------

def test_cache_key_differs_when_reasoning_params_differ():
    agent_low = build_agent("openai/gpt-oss-120b", reasoning_params={"reasoning_effort": "low"})
    agent_medium = build_agent("openai/gpt-oss-120b", reasoning_params={"reasoning_effort": "medium"})

    key_low = make_cache_key(
        model=agent_low.model, prompt="q", temperature=0.0, max_tokens=100,
        extra={"reasoning_params": agent_low.reasoning_params},
    )
    key_medium = make_cache_key(
        model=agent_medium.model, prompt="q", temperature=0.0, max_tokens=100,
        extra={"reasoning_params": agent_medium.reasoning_params},
    )

    assert key_low != key_medium


def test_query_does_not_reuse_cache_across_reasoning_params(tmp_path):
    cache_dir = str(tmp_path / "cache")

    agent_low = GroqAgent(
        model="openai/gpt-oss-120b", api_key=API_KEY,
        reasoning_params={"reasoning_effort": "low"},
        cache_enabled=True, cache_dir=cache_dir,
    )
    mock_create_low = MagicMock(return_value=fake_completion(content="Paris."))
    agent_low._client.chat.completions.create = mock_create_low
    agent_low.query("What is the capital of France?")
    assert mock_create_low.call_count == 1

    # Same model, same prompt, same cache directory (-> same cache file),
    # different reasoning_effort: must be a genuine miss, not served from
    # agent_low's cached entry.
    agent_medium = GroqAgent(
        model="openai/gpt-oss-120b", api_key=API_KEY,
        reasoning_params={"reasoning_effort": "medium"},
        cache_enabled=True, cache_dir=cache_dir,
    )
    mock_create_medium = MagicMock(return_value=fake_completion(content="Paris."))
    agent_medium._client.chat.completions.create = mock_create_medium
    agent_medium.query("What is the capital of France?")
    assert mock_create_medium.call_count == 1


# ---------------------------------------------------------------------------
# Structural error classification
# ---------------------------------------------------------------------------

def test_404_sets_error_kind_permanent():
    agent = build_agent("llama-3.1-8b-instant")
    mock_create = MagicMock(side_effect=api_status_error(404))
    agent._client.chat.completions.create = mock_create

    resp = agent.query("q")

    assert resp.is_error
    assert resp.error_kind == "permanent"
    assert resp.is_permanent
    assert not resp.is_transient
    assert resp.error.startswith("PERMANENT")
    # A permanent error must not burn retries.
    assert mock_create.call_count == 1


def test_429_exhausting_retries_sets_error_kind_transient(monkeypatch):
    monkeypatch.setattr("agents.groq_agent.time.sleep", lambda *_: None)
    agent = build_agent("llama-3.1-8b-instant")
    mock_create = MagicMock(side_effect=api_status_error(429, retry_after="0"))
    agent._client.chat.completions.create = mock_create

    resp = agent.query("q")

    assert resp.is_error
    assert resp.error_kind == "transient"
    assert resp.is_transient
    assert not resp.is_permanent
    assert mock_create.call_count == MAX_RETRIES + 1


def test_successful_response_has_no_error_kind():
    agent = build_agent("llama-3.1-8b-instant")
    mock_create = MagicMock(return_value=fake_completion())
    agent._client.chat.completions.create = mock_create

    resp = agent.query("q")

    assert not resp.is_error
    assert resp.error_kind is None
    assert not resp.is_permanent
    assert not resp.is_transient


def test_error_kind_round_trips_through_to_dict_and_from_dict():
    from agents.base import AgentResponse

    resp = AgentResponse(
        text="", model="m", prompt="p", temperature=0.0, max_tokens=10,
        latency_s=0.0, timestamp="2026-01-01T00:00:00+00:00",
        error="PERMANENT 404 NotFoundError: boom", error_kind="permanent",
    )
    restored = AgentResponse.from_dict(resp.to_dict())

    assert restored.error_kind == "permanent"
    assert restored.is_permanent
