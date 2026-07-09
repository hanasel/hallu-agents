"""End-to-end smoke test for the agent-querying layer.

Loads 3 TruthfulQA samples, queries a Groq-served Llama-3.1-8B twice
(once with a cold cache, once with a warm one), and prints the results.
The second run should hit the cache and complete near-instantly.

Requires GROQ_API_KEY in the environment or in a .env file at the project root.

Run from the project root:
    python scripts/test_single_agent.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make imports work when run as `python scripts/test_single_agent.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa                         # noqa: E402
from agents import GroqAgent                             # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def summarise(agent: GroqAgent, samples, label: str) -> float:
    """Query `samples` sequentially and print a compact summary. Returns total wall time."""
    t0 = time.perf_counter()
    for i, sample in enumerate(samples, start=1):
        r = agent.query(sample.prompt)
        preview = r.text.strip().replace("\n", " ")[:180]
        status = "ERROR" if r.is_error else f"{r.latency_s:>5.2f}s"
        print(f"  [{i}] {status} → {preview!r}")
        if r.is_error:
            print(f"      error: {r.error}")
    elapsed = time.perf_counter() - t0
    print(f"  ({label} total: {elapsed:.2f}s)")
    return elapsed


def main() -> None:
    section("Loading 3 TruthfulQA samples")
    samples = load_truthfulqa(n=3, seed=0)
    for s in samples:
        print(f"  {s.uid}  [{s.category}]  {s.question[:70]}")

    section("Instantiating GroqAgent (Llama-3.1-8B-Instant, T=0.0)")
    try:
        agent = GroqAgent()          # defaults: llama-3.1-8b-instant, T=0.0
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)

    stats = agent.cache_stats()
    print(f"  agent name  : {agent.name}")
    print(f"  cache file  : {stats['cache_file']}")
    print(f"  cache size  : {stats['n_entries']} entries")

    section("First pass — cold cache (real API calls)")
    cold_t = summarise(agent, samples, label="cold")

    section("Second pass — warm cache (should hit)")
    warm_t = summarise(agent, samples, label="warm")

    section("Result")
    if warm_t < cold_t * 0.1 or warm_t < 0.05:
        print(f"  Cache working: warm run was {cold_t / max(warm_t, 1e-6):.0f}x faster.")
    else:
        print(f"  WARNING: warm run ({warm_t:.2f}s) not much faster than cold "
              f"({cold_t:.2f}s). Cache may not be working.")
    print(f"  Cache now holds {agent.cache_stats()['n_entries']} entries.")
    print()


if __name__ == "__main__":
    main()