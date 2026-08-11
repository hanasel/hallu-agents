"""Per-dataset agent-query settings.

`agents.GroqAgent` defaults are deliberately neutral (max_tokens=512, no
system prompt) — nothing dataset-specific lives in the agents layer. Callers
pass one of these configs in explicitly, e.g.:

    from data import TRUTHFULQA_QUERY_CONFIG
    from agents import GroqAgent

    agent = GroqAgent(**TRUTHFULQA_QUERY_CONFIG)
"""

from __future__ import annotations

# TruthfulQA reference answers are one sentence; a 512-token budget with no
# steering caused finish_reason='length' on 100% of a 30-sample pilot (the
# model rambles well past the point a grader can compare against the short
# gold answer).
TRUTHFULQA_QUERY_CONFIG = {
    "max_tokens": 100,
    "system_prompt": "Answer in one short sentence.",
    "temperature": 0.0,
}

# RAGTruth prompts carry their own task instructions (QA / summarization /
# data2txt) as part of the corpus prompt itself, and responses are graded at
# the span level, so no extra system prompt or short-answer budget is imposed.
RAGTRUTH_QUERY_CONFIG = {
    "max_tokens": 512,
    "system_prompt": None,
    "temperature": 0.0,
}
