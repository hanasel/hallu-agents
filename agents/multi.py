"""Multi-agent querying utilities.

Currently sequential — parallelisation is trivial to add later when we're
running full-dataset experiments. Keeps the independence guarantee that
underpins the inter-agent disagreement approach (§2.4): each agent sees
only the prompt, with no information about other agents' responses.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from .base import Agent, AgentResponse


def query_agents(
    agents: Sequence[Agent],
    prompt: str,
    **kwargs: Any,
) -> List[AgentResponse]:
    """Query multiple agents on the same prompt independently.

    Returns responses in the same order as `agents`. Additional kwargs are
    forwarded to each agent's `query()` method (e.g. `temperature`,
    `max_tokens`, `use_cache`).

    No information is shared between agents. This is the parallel-independent
    querying protocol from §3.1.1 — critical for the disagreement signal to
    reflect genuine knowledge divergence rather than debate dynamics.
    """
    return [agent.query(prompt, **kwargs) for agent in agents]
