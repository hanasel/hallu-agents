"""JSONL-backed response cache for agent queries.

Design
------
Every call to `agent.query(prompt)` is hashed into a stable cache key
covering everything that affects the response (model, prompt, temperature,
max_tokens). If the key exists in the cache, the stored `AgentResponse`
is returned immediately — no API call, no cost.

Storage layout
--------------
One JSONL file per agent, at `cache_dir/<safe_name>.jsonl`. Each line is:

    {"cache_key": "<sha256>", "response": {<AgentResponse.to_dict()>}}

The file is append-only during a run and re-read on startup, so the cache
survives across sessions and can be inspected/diffed in git.

Rationale
---------
- **Determinism.** Re-running the pipeline reproduces exact numbers.
- **Cost.** No wasted API calls on prompts we've already answered.
- **Debuggability.** Cache files are human-readable JSONL.
- **Simplicity.** Nothing to run — the cache lives with your code.

Non-goals
---------
Not thread-safe. Not designed for parallel writers (we're not there yet).
Not a general-purpose KV store — bespoke to `AgentResponse`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Optional

from .base import AgentResponse


DEFAULT_CACHE_DIR = "cache/agent_responses"


def make_cache_key(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    extra: Optional[Dict[str, object]] = None,
) -> str:
    """Deterministic hash covering everything that affects the response.

    `extra` is for anything additional (e.g. system prompt, seed) that we
    want to be part of the identity. Kept as an optional dict so callers
    don't have to think about it if unused.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": round(float(temperature), 6),
        "max_tokens": int(max_tokens),
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _safe_filename(name: str) -> str:
    """Turn 'groq/llama-3.1-8b-instant' into 'groq__llama-3.1-8b-instant'."""
    return re.sub(r"[^\w\-.]+", "__", name)


class ResponseCache:
    """Append-only JSONL cache keyed by (model, prompt, temperature, max_tokens).

    Usage
    -----
    >>> cache = ResponseCache(agent_name="groq/llama-3.1-8b-instant")
    >>> hit = cache.get(key)
    >>> if hit is None:
    ...     response = do_api_call(...)
    ...     cache.put(key, response)
    """

    def __init__(
        self,
        agent_name: str,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        self.agent_name = agent_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.cache_dir / f"{_safe_filename(agent_name)}.jsonl"

        # In-memory index for O(1) lookups
        self._index: Dict[str, AgentResponse] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self._index[row["cache_key"]] = AgentResponse.from_dict(row["response"])
                except (json.JSONDecodeError, KeyError):
                    # Skip corrupted lines silently — reduces friction if
                    # a run was killed mid-write. A real corruption will
                    # surface as a cache miss.
                    continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[AgentResponse]:
        """Return the cached response for `key`, or None on miss."""
        return self._index.get(key)

    def put(self, key: str, response: AgentResponse) -> None:
        """Store `response` under `key`. Appended to the JSONL immediately.

        Errored responses (`response.is_error`) are NOT cached — we want a
        rerun to actually retry them, not memorise the failure.
        """
        if response.is_error:
            return
        self._index[key] = response
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                {"cache_key": key, "response": response.to_dict()},
                ensure_ascii=False,
            ) + "\n")

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def stats(self) -> Dict[str, object]:
        """Summary statistics for logging/inspection."""
        return {
            "agent_name": self.agent_name,
            "cache_file": str(self.path),
            "n_entries": len(self._index),
            "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
