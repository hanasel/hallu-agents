"""Agent protocol and response dataclass.

An `Agent` is anything that takes a prompt and returns an `AgentResponse`.
Concrete implementations (e.g. `GroqAgent`) satisfy this protocol; downstream
code (querying loops, disagreement measures) only depends on the protocol.

For the inter-agent disagreement setup, we deliberately query each agent
independently — no shared conversation, no cross-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentResponse:
    """A single response from an agent, with all metadata needed for analysis.

    Kept dataclass-serialisable (via `to_dict`) so responses can be cached
    to JSONL and inspected without instantiating any agent.
    """

    text: str                    # the response content
    model: str                   # exact model identifier used
    prompt: str                  # the prompt that produced this response
    temperature: float
    max_tokens: int
    latency_s: float             # wall-clock time for the API call
    timestamp: str               # ISO-8601 UTC when the response was received
    finish_reason: Optional[str] = None    # "stop", "length", "content_filter", ...
    usage: Dict[str, int] = field(default_factory=dict)   # prompt/completion/total tokens
    error: Optional[str] = None  # set only if the call ultimately failed after retries
    provider: str = ""           # "groq", "together", "openai", ...

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "prompt": self.prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "latency_s": self.latency_s,
            "timestamp": self.timestamp,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "error": self.error,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AgentResponse":
        return cls(
            text=d["text"],
            model=d["model"],
            prompt=d["prompt"],
            temperature=d["temperature"],
            max_tokens=d["max_tokens"],
            latency_s=d["latency_s"],
            timestamp=d["timestamp"],
            finish_reason=d.get("finish_reason"),
            usage=dict(d.get("usage", {})),
            error=d.get("error"),
            provider=d.get("provider", ""),
        )


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 (used by concrete agents)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@runtime_checkable
class Agent(Protocol):
    """The interface every concrete agent must satisfy.

    Implementations:
      - `agents.groq_agent.GroqAgent`
      - (later: TogetherAgent, HFAgent, OllamaAgent, ...)
    """

    @property
    def name(self) -> str:
        """Short identifier for logging / cache-file naming.

        Convention: 'provider/model', e.g. 'groq/llama-3.1-8b-instant'.
        """
        ...

    @property
    def model(self) -> str:
        """The exact model identifier used at the API level."""
        ...

    def query(self, prompt: str, **kwargs: Any) -> AgentResponse:
        """Send a prompt to the model and return an AgentResponse.

        Concrete implementations handle retries, caching, and error mapping.
        Should never raise on API errors — populate `AgentResponse.error`
        instead so batch runs don't abort on a single failure.
        """
        ...
