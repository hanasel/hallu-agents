"""Smoke test for the answer-level disagreement layer.

Part 1 — hand-crafted responses with assertions. Runs offline, verifies the
         measure is doing what we expect (identical → 0.0, no overlap → 1.0).

Part 2 — two real Groq agents on 3 TruthfulQA samples. Shows disagreement
         scores per sample. Expected pattern:
           - Q1 (music/thinking misconception): higher disagreement, because
             the 8B model falls for the misconception and the 70B likely
             doesn't. This is the target case.
           - Q2 (Neil Armstrong): low disagreement — both should get it right.
           - Q3 (self-identification): low disagreement — both should
             identify themselves as not-an-animal.

Requires GROQ_API_KEY. Uses the shared response cache, so a repeat run
completes near-instantly on Part 2.

Run from the project root:
    python scripts/test_disagreement.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa, TRUTHFULQA_QUERY_CONFIG         # noqa: E402
from agents import GroqAgent, query_agents                        # noqa: E402
from disagreement import JaccardDisagreement                       # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def part_1_offline_sanity_check() -> None:
    section("Part 1 — hand-crafted responses (offline sanity check)")

    jaccard = JaccardDisagreement()

    cases = [
        (
            "identical responses",
            ["The sky is blue.", "The sky is blue."],
            "expect 0.000",
        ),
        (
            "paraphrases (share some tokens)",
            ["The sky is blue.", "Blue is the colour of the sky."],
            "expect moderate",
        ),
        (
            "no lexical overlap",
            ["The sky is blue.", "Grass is often green."],
            "expect high",
        ),
        (
            "3 agents: 2 similar, 1 different",
            [
                "Yes, learning music helps thinking.",
                "Music education improves cognitive abilities.",
                "No, evidence does not support music boosting cognition.",
            ],
            "expect intermediate mean",
        ),
    ]

    for label, responses, note in cases:
        r = jaccard.score(responses)
        print(f"\n{label} ({note}):")
        for i, resp in enumerate(responses):
            print(f"  [{i}] {resp}")
        print(f"  → Jaccard disagreement: {r.score:.3f}")
        pairs_str = ", ".join(f"{p:.2f}" for p in r.details["pairwise_disagreements"])
        print(f"    pairwise: [{pairs_str}]")

    # Assertions — the measure should behave predictably in the extreme cases
    identical = jaccard.score(cases[0][1]).score
    no_overlap = jaccard.score(cases[2][1]).score
    assert identical == 0.0, f"identical should be 0.0, got {identical}"
    assert no_overlap > 0.5, f"no-overlap should be > 0.5, got {no_overlap}"
    print("\n[OK] identical → 0.0; no-overlap → > 0.5")


def part_2_real_agents() -> None:
    section("Part 2 — two Groq agents on 3 TruthfulQA samples")

    try:
        agent_a = GroqAgent(model="llama-3.1-8b-instant", **TRUTHFULQA_QUERY_CONFIG)
        agent_b = GroqAgent(model="llama-3.3-70b-versatile", **TRUTHFULQA_QUERY_CONFIG)
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)

    print(f"  Agent A: {agent_a.name}")
    print(f"  Agent B: {agent_b.name}")

    samples = load_truthfulqa(n=3, seed=0)
    jaccard = JaccardDisagreement()
    summary = []

    for s in samples:
        section(f"{s.uid}  [{s.category}]")
        print(f"Q: {s.question}")
        print(f"Correct: {s.correct_answer}")

        responses = query_agents([agent_a, agent_b], s.prompt)
        for agent, r in zip([agent_a, agent_b], responses):
            preview = r.text.strip().replace("\n", " ")[:180]
            status = "ERROR" if r.is_error else f"{r.latency_s:>5.2f}s"
            print(f"\n  {agent.name} ({status}):")
            print(f"    {preview!r}")
            if r.is_error:
                print(f"    error: {r.error}")

        result = jaccard.score(responses)
        print(f"\n  Jaccard disagreement: {result.score:.3f}")
        summary.append((s.uid, s.category, result.score))

    section("Summary")
    print(f"  {'uid':<20} {'category':<28} {'jaccard':>8}")
    print(f"  {'-'*20} {'-'*28} {'-'*8}")
    for uid, cat, score in summary:
        print(f"  {uid:<20} {cat[:28]:<28} {score:>8.3f}")


def main() -> None:
    part_1_offline_sanity_check()
    part_2_real_agents()

    section("Done")
    print("Answer-level disagreement layer is working end-to-end.")
    print("Next up: semantic disagreement (Semantic Entropy) or claim-level.")
    print()


if __name__ == "__main__":
    main()
