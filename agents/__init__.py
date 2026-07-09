"""Agent querying layer.

Public exports:
  - Agent           : the interface every agent satisfies
  - AgentResponse   : the response dataclass returned by every query
  - GroqAgent       : concrete agent for Groq-served open models
  - ResponseCache   : the JSONL cache used by agents
  - make_cache_key  : cache-key hashing helper (mostly internal)
  - query_agents    : fan out one prompt to multiple agents independently
"""

from .base import Agent, AgentResponse
from .cache import ResponseCache, make_cache_key, DEFAULT_CACHE_DIR
from .groq_agent import GroqAgent, GROQ_BASE_URL
from .multi import query_agents

__all__ = [
    "Agent",
    "AgentResponse",
    "GroqAgent",
    "ResponseCache",
    "make_cache_key",
    "DEFAULT_CACHE_DIR",
    "GROQ_BASE_URL",
    "query_agents",
]