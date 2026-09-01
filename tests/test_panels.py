"""Tests for agents/panels.py's Qwen metadata tables (generation, total
params, arch, the two within-family ladders) and the invariant that every
Qwen model has an OPENROUTER_REASONING_PARAMS entry.

A model missing from OPENROUTER_REASONING_PARAMS resolves to {} via
`.get(model, {})` in agents/providers.py's make_provider_agent — no error, no
warning, just an unsuppressed reasoning trace flowing into `text` and into
every disagreement measure. This file is what turns that into a loud failure
at test time instead of a silently-corrupted query-time run.
"""

from __future__ import annotations

from agents.panels import (
    MODEL_FAMILY,
    SIZE_LADDER_QWEN35,
    GEN_LADDER_QWEN_27B,
    QWEN_PANEL_NEW_MODELS,
    QWEN_PLUS,
    QWEN,
    QWEN_LARGE,
    generation_of,
    total_params_of,
    arch_of,
    family_of,
)
from agents.providers import OPENROUTER_REASONING_PARAMS


def _qwen_models():
    return [m for m, fam in MODEL_FAMILY.items() if fam == "Qwen"]


def test_every_qwen_model_has_a_reasoning_params_entry():
    missing = [m for m in _qwen_models() if m not in OPENROUTER_REASONING_PARAMS]
    assert not missing, (
        f"Qwen model(s) missing from OPENROUTER_REASONING_PARAMS: {missing} — "
        "a missing entry resolves to {} silently, letting the reasoning trace "
        "leak into every disagreement measure."
    )


def test_reasoning_params_entries_actually_suppress_reasoning():
    for m in _qwen_models():
        params = OPENROUTER_REASONING_PARAMS[m]
        assert params.get("reasoning", {}).get("exclude") is True, m


def test_existing_qwen36_reasoning_params_untouched():
    # Cache identity for the already-cached 1000-question SimpleQA run
    # depends on these three never changing (reasoning_params is part of
    # GroqAgent's cache key).
    assert OPENROUTER_REASONING_PARAMS["qwen/qwen3.6-27b"] == {
        "reasoning": {"effort": "none", "exclude": True}}
    assert OPENROUTER_REASONING_PARAMS["qwen/qwen3.6-35b-a3b"] == {
        "reasoning": {"effort": "none", "exclude": True}}
    assert OPENROUTER_REASONING_PARAMS["qwen/qwen3.6-plus"] == {
        "reasoning": {"effort": "none", "exclude": True}}


def test_size_ladder_members_all_have_disclosed_params():
    for m in SIZE_LADDER_QWEN35:
        assert total_params_of(m) is not None, m


def test_qwen36_plus_excluded_from_size_ladder_undisclosed_params():
    assert total_params_of(QWEN_PLUS) is None
    assert QWEN_PLUS not in SIZE_LADDER_QWEN35


def test_generation_ladder_spans_distinct_generations_at_matched_size():
    gens = [generation_of(m) for m in GEN_LADDER_QWEN_27B]
    assert all(g is not None for g in gens)
    assert len(set(gens)) == len(gens)
    for m in GEN_LADDER_QWEN_27B:
        params = total_params_of(m)
        assert params is not None and 25 <= params <= 32, (m, params)


def test_qwen_panel_new_models_matches_the_six_new_ids():
    assert len(QWEN_PANEL_NEW_MODELS) == 6
    assert len(set(QWEN_PANEL_NEW_MODELS)) == 6
    for m in QWEN_PANEL_NEW_MODELS:
        assert family_of(m) == "Qwen"
    # Not re-adding any of the three already-cached Qwen3.6 models.
    assert QWEN not in QWEN_PANEL_NEW_MODELS
    assert QWEN_LARGE not in QWEN_PANEL_NEW_MODELS
    assert QWEN_PLUS not in QWEN_PANEL_NEW_MODELS


def test_unknown_model_falls_back_gracefully():
    assert generation_of("someone/unknown-model") is None
    assert total_params_of("someone/unknown-model") is None
    assert arch_of("someone/unknown-model") is None
    assert family_of("someone/unknown-model") == "someone/unknown-model"


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    if failures:
        sys.exit(f"{failures} test(s) failed")
    print("\nAll tests passed.")
