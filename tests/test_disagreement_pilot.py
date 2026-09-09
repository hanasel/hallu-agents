"""Tests for scripts/disagreement_pilot.py: abstention detection and the
short-answer grading heuristics that gate the NLI/judge.

`scripts/` isn't a package (no __init__.py, and each script inserts the repo
root onto sys.path itself), so the module under test is loaded by file path
rather than a normal import.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "disagreement_pilot", REPO_ROOT / "scripts" / "disagreement_pilot.py")
pilot = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("disagreement_pilot", pilot)
_SPEC.loader.exec_module(pilot)

from agents.base import AgentResponse                                  # noqa: E402

is_abstention = pilot.is_abstention
grade_short_answer = pilot.grade_short_answer


ABSTENTION_TRUE = [
    "I'm not aware of any information about the author of *Batman R.I.P.*",
    "I'm not sure when Instagram introduced comment liking.",
    "I'm sorry, but I am unable to verify who built the general store",
    "I don't have the specific information about which soldier",
    "I couldn't find any information on the Arvida Theatre",
    "I cannot verify how many votes Alisher Usmanov received",
]

ABSTENTION_FALSE = [
    'There is no such neologism, as the term "equinox" has been used for centuries',
    'No soldier was conferred with the "Points of Light Honour"',
    "The premise is false; no such gathering occurred",
]


def test_abstention_contractions_and_qualifiers_detected():
    for text in ABSTENTION_TRUE:
        assert is_abstention(text), f"expected abstention: {text!r}"


def test_committed_factual_claims_not_flagged_as_abstention():
    for text in ABSTENTION_FALSE:
        assert not is_abstention(text), f"expected NOT abstention: {text!r}"


def test_grade_short_answer_numeric_requires_full_subset():
    assert grade_short_answer("44 tracks", "37", "number") is False
    assert grade_short_answer("appears in 10 sets", "10", "number") is True
    assert grade_short_answer("1 year and 5 days", "1 year, 161 days", "number") is False


def test_grade_short_answer_defers_only_on_distinctive_overlap():
    assert grade_short_answer("John Burritt", "John Strahan French", None) is False


# ---------------------------------------------------------------------------
# make_short_answer_key_fn: the extractor behind --short-answer-keys /
# --short-entity-keys. Built with short_entity=True so all four sub-
# extractors (chemical formula, date, number, entity) are exercised as one
# unit — the default CLI path (--short-answer-keys without --short-entity-
# keys) simply never reaches the entity branch, see make_short_answer_key_fn.
# ---------------------------------------------------------------------------

key_fn = pilot.make_short_answer_key_fn(short_entity=True)


def test_key_fn_separates_different_dates():
    a = key_fn("The U.S. Congress approved the Compact of Free Association "
              "in 1986.", "Date")
    b = key_fn("The U.S. Congress approved the Compact of Free Association "
              "in 1985.", "Date")
    assert a is not None and b is not None
    assert a != b


def test_key_fn_separates_different_numbers():
    a = key_fn("The album appears in 44 tracks.", "Number")
    b = key_fn("The album appears in 37 tracks.", "Number")
    assert a is not None and b is not None
    assert a != b


def test_key_fn_separates_different_chemical_formulae():
    a = key_fn("The chemical formula of atogepant is C29H31N5O3S.", "Other")
    b = key_fn("The chemical formula of atogepant is C₂₈H₃₁N₅O₄.", "Other")
    assert a is not None and b is not None
    assert a != b


def test_key_fn_chemical_formula_ignores_answer_type_gate():
    # Distinctive enough to fire on its own — the atogepant question's
    # answer_type is "Other", not "Number", so this must NOT be gated on it.
    assert key_fn("...is C29H31N5O3S.", "Other") is not None
    assert key_fn("...is C29H31N5O3S.", None) is not None


def test_key_fn_unicode_subscripts_match_ascii_digits():
    a = key_fn("C₂₈H₃₁N₅O₄", "Other")
    b = key_fn("C28H31N5O4", "Other")
    assert a is not None
    assert a == b


def test_key_fn_multi_entity_answers_are_order_invariant():
    a = key_fn("Papua New Guinea and Indonesia", "Place")
    b = key_fn("Indonesia and Papua New Guinea", "Place")
    assert a is not None and b is not None
    assert a == b


def test_key_fn_multi_entity_answers_separate_on_different_entity():
    # simpleqa-1315: substituting Australia for the correct Indonesia must
    # NOT compare equal just because both mention Papua New Guinea.
    a = key_fn("Anisotenes cacotechna is found in Australia and Papua New Guinea.", "Place")
    b = key_fn("Anisotenes cacotechna is found in Indonesia and Papua New Guinea.", "Place")
    assert a is not None and b is not None
    assert a != b


def test_key_fn_identical_dates_compare_equal():
    a = key_fn("...in 1986.", "Date")
    b = key_fn("...in 1986.", "Date")
    assert a is not None
    assert a == b


def test_key_fn_returns_none_on_prose_with_no_short_form_answer():
    # TruthfulQA-style: no date/number/formula, and too few (<2) proper-noun
    # phrases for the entity extractor to fire either — the branch must
    # no-op here regardless of --short-entity-keys.
    text = "It depends on who you ask and how you define happiness."
    assert key_fn(text, None) is None


def test_key_fn_single_entity_answers_defer_to_nli():
    # The exact false-split risk --short-entity-keys is gated against:
    # "New Guinea" vs "Papua New Guinea" must NOT get a key (>=2 entities
    # required), so a single-entity answer always falls through to NLI.
    assert key_fn("New Guinea", "Place") is None
    assert key_fn("Papua New Guinea", "Place") is None
    assert key_fn("Neil Armstrong.", "Person") is None


def test_key_fn_short_entity_extractor_is_opt_in():
    default_key_fn = pilot.make_short_answer_key_fn(short_entity=False)
    assert default_key_fn("Papua New Guinea and Indonesia", "Place") is None


# ---------------------------------------------------------------------------
# v2: question-anchoring (Edit 1) and subset matching (Edit 2). "Compare
# equal" below means the relation _equivalent now uses: ka <= kb or kb <= ka,
# not ka == kb — see disagreement/semantic.py.
# ---------------------------------------------------------------------------

ANDERSSEN_QUESTION = ("How many games did Adolf Anderssen lose in his 1864 "
                     "chess match against Berthold Suhle?")


def test_key_fn_question_subtraction_merges_terse_and_detailed_number():
    a = key_fn("...lost 1 game in his 1864 match", "Number", ANDERSSEN_QUESTION)
    b = key_fn("...had 1 loss in his 1864 match, with 4 wins and 3 draws",
              "Number", ANDERSSEN_QUESTION)
    assert a is not None and b is not None
    assert a <= b or b <= a


def test_key_fn_subset_merges_terse_and_detailed_answer_no_question_overlap():
    a = key_fn("About 300 people died.", "Number")
    b = key_fn("300 people died when the HMS Ontario sank in 1780.", "Number")
    assert a is not None and b is not None
    assert a <= b or b <= a


def test_key_fn_different_values_still_separate_after_subtraction():
    question = "In what year was the treaty signed?"
    a = key_fn("It was signed in 1986.", "Date", question)
    b = key_fn("It was signed in 1985.", "Date", question)
    assert a is not None and b is not None
    assert not (a <= b or b <= a)


def test_key_fn_emptied_by_question_subtraction_returns_none():
    # The only number in the response is restated verbatim from the
    # question, so subtraction empties the key entirely — must return None
    # (not an empty frozenset, which would compare equal to any other empty
    # key and silently merge unrelated pairs) so the pair falls through to
    # NLI instead.
    question = "How many people died when the HMS Ontario sank in 1780?"
    assert key_fn("It sank in 1780.", "Date", question) is None
    assert key_fn("1780 people were aboard.", "Number", question) is None


# ---------------------------------------------------------------------------
# judge_grade: retry on an unusable verdict, raise only after exhausting
# `attempts`. Covers the finish_reason='error'/error=None shape (HTTP 200,
# upstream failure passed through as a "success") that crashed the 200-
# question run at question 185 — GroqAgent.query's own error path always
# sets `error`, so `r.is_error` alone doesn't cover it; judge_grade must
# retry on unusable *text* too, not just on `r.is_error`.
# ---------------------------------------------------------------------------

def _judge_response(text="", finish_reason="stop", error=None):
    return AgentResponse(
        text=text, model="judge-model", prompt="p", temperature=0.0, max_tokens=64,
        latency_s=0.0, timestamp="2026-01-01T00:00:00+00:00",
        finish_reason=finish_reason, error=error,
        error_kind=("transient" if error else None),
    )


class _FlakyJudge:
    """Two unusable responses (finish_reason='error', error=None — the exact
    shape from the crash), then a real verdict on the third call."""

    def __init__(self, n_failures=2, verdict="CORRECT"):
        self.calls = 0
        self.n_failures = n_failures
        self.verdict = verdict

    def query(self, prompt, **kwargs):
        self.calls += 1
        if self.calls <= self.n_failures:
            return _judge_response(text="", finish_reason="error", error=None)
        return _judge_response(text=self.verdict)


class _AlwaysFailingJudge:
    def __init__(self):
        self.calls = 0

    def query(self, prompt, **kwargs):
        self.calls += 1
        return _judge_response(text="", finish_reason="error", error=None)


def test_judge_grade_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(pilot.time, "sleep", lambda *_: None)
    judge = _FlakyJudge(n_failures=2, verdict="CORRECT")

    result = pilot.judge_grade(judge, "Q?", "Paris", "some answer")

    assert result is True
    assert judge.calls == 3


def test_judge_grade_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(pilot.time, "sleep", lambda *_: None)
    judge = _AlwaysFailingJudge()

    with pytest.raises(RuntimeError):
        pilot.judge_grade(judge, "Q?", "Paris", "some answer", attempts=3)

    assert judge.calls == 3


# ---------------------------------------------------------------------------
# --resume: skip uids already in --out, tolerate a truncated final line,
# append rather than truncate, and abort on a pool mismatch.
# ---------------------------------------------------------------------------

@dataclass
class _FakeSample:
    uid: str
    prompt: str
    topic: str = "test"
    answer_type: str = "person"
    multi_step: bool = False
    requires_reasoning: bool = False
    question: str = "Who?"
    correct_answer: str = "Paris"

    def open_prompt(self) -> str:
        return self.prompt


class _FakeAgent:
    """Stands in for a GroqAgent: same `.name`/`.model`/`.query()` shape,
    no network, no cache. `reasoning_params` is set to `{}` (falsy either
    way), so `agent_is_reasoning` still falls back to the model-id heuristic
    via getattr's default — same behaviour as it being absent entirely.
    max_tokens/temperature/system_prompt are only here for
    write_manifest/check_manifest_compatible, exercised by the --resume
    tests below."""

    def __init__(self, name, model, answer_text):
        self.name = name
        self.model = model
        self.answer_text = answer_text
        self.calls = 0
        self.reasoning_params = {}
        self.max_tokens = 64
        self.temperature = 0.0
        self.system_prompt = None

    def query(self, prompt, **kwargs):
        self.calls += 1
        return AgentResponse(
            text=self.answer_text, model=self.model, prompt=prompt,
            temperature=0.0, max_tokens=64, latency_s=0.0,
            timestamp="2026-01-01T00:00:00+00:00", finish_reason="stop",
            usage={"completion_tokens": 3}, error=None,
        )


class _StubScore:
    def __init__(self, score):
        self.score = score


class _StubJaccard:
    def score(self, texts):
        return _StubScore(0.0)


class _StubSemantic:
    """Every panel 'agrees' (one cluster) — the disagreement value itself
    is irrelevant to the resume behaviour under test."""

    def score(self, texts, question=None, answer_type=None):
        n = len(texts)
        result = _StubScore(0.0)
        result.details = {"n_clusters": 1, "cluster_sizes": [n],
                          "clusters": [list(range(n))]}
        return result


def _make_agents():
    return [
        _FakeAgent("fake/agent-a", "fake-model-a", "The answer is Paris."),
        _FakeAgent("fake/agent-b", "fake-model-b", "The answer is Paris."),
    ]


def _resume_args():
    return types.SimpleNamespace(
        resume=True, max_bad_questions=5, no_exact_match=False, peek=0,
        # write_manifest / check_manifest_compatible (run unconditionally at
        # the top of run_queries, resume or not) read these off `args` too.
        dataset="simpleqa", prompt_format="open", n=10, seed=0,
        nli_model="test-nli-model", strict=False, single_linkage=False,
        judge_model="", judge_provider="openrouter",
    )


_DATASET_CFG = {"meta": ("topic", "answer_type", "multi_step", "requires_reasoning")}


def _write_resume_fixture(out_path, agent_names, n_done=10):
    lines = [
        json.dumps({"uid": f"q{i}", "responses": {n: f"resp-{i}" for n in agent_names}})
        for i in range(n_done)
    ]
    lines.append('{"uid": "q-broken", invalid json here')   # truncated mid-write
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resume_skips_done_uids_ignores_bad_line_and_appends(tmp_path):
    agents = _make_agents()
    out_path = tmp_path / "results.jsonl"
    _write_resume_fixture(out_path, [a.name for a in agents], n_done=10)

    samples = [_FakeSample(uid=f"q{i}", prompt=f"question {i}") for i in range(10)]
    samples.append(_FakeSample(uid="q10", prompt="question 10"))

    panel_specs = pilot.build_panel_specs(agents)
    rows = pilot.run_queries(samples, agents, _StubJaccard(), _StubSemantic(), None, None,
                             panel_specs, out_path, _resume_args(), None,
                             _DATASET_CFG, False)

    # 10 pre-existing rows carried over untouched, plus exactly 1 new row.
    assert len(rows) == 11
    assert {r["uid"] for r in rows} == {f"q{i}" for i in range(11)}
    for a in agents:
        assert a.calls == 1   # only q10 was actually queried; q0..q9 skipped

    # Appended, not truncated: original 10 rows + the malformed line survive,
    # with the new row after them.
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    assert json.loads(lines[-1])["uid"] == "q10"


def test_resume_aborts_on_agent_pool_mismatch(tmp_path):
    old_agents = _make_agents()
    out_path = tmp_path / "results.jsonl"
    _write_resume_fixture(out_path, [a.name for a in old_agents], n_done=10)

    new_agents = [
        _FakeAgent("fake/agent-c", "fake-model-c", "answer"),
        _FakeAgent("fake/agent-d", "fake-model-d", "answer"),
    ]
    panel_specs = pilot.build_panel_specs(new_agents)

    with pytest.raises(RuntimeError, match="different agent pool"):
        pilot.run_queries([], new_agents, _StubJaccard(), _StubSemantic(), None, None,
                          panel_specs, out_path, _resume_args(), None,
                          _DATASET_CFG, False)


# ---------------------------------------------------------------------------
# build_panel_specs: within-family pairs sub-classify into within:size /
# within:generation / within:other (agents/panels.py's generation_of /
# total_params_of / arch_of tables), without changing `kind` itself — every
# existing `s.kind == "within"` check downstream must keep working unmodified.
# ---------------------------------------------------------------------------

from agents.panels import (                                            # noqa: E402
    QWEN3_32B, QWEN35_9B, QWEN35_27B, QWEN as QWEN36_27B,
    LLAMA_SMALL, LLAMA_LARGE,
)


def _spec_for(specs, name_a, name_b):
    key = frozenset((name_a, name_b))
    for s in specs:
        if s.kind == "within" and frozenset(s.members) == key:
            return s
    raise AssertionError(f"no within-family spec for {name_a!r}/{name_b!r}")


def test_within_pair_classified_as_size_same_generation_different_capacity():
    agents = [
        _FakeAgent("fake/q35-9b", QWEN35_9B, "x"),
        _FakeAgent("fake/q35-27b", QWEN35_27B, "x"),
    ]
    specs = pilot.build_panel_specs(agents)
    spec = _spec_for(specs, "fake/q35-9b", "fake/q35-27b")
    assert spec.subkind == "size"
    assert spec.family == "Qwen"
    assert spec.kind == "within"   # unchanged — pooled within-vs-cross logic relies on this


def test_within_pair_classified_as_generation_matched_capacity_different_generation():
    agents = [
        _FakeAgent("fake/q3-32b", QWEN3_32B, "x"),
        _FakeAgent("fake/q36-27b", QWEN36_27B, "x"),
    ]
    specs = pilot.build_panel_specs(agents)
    spec = _spec_for(specs, "fake/q3-32b", "fake/q36-27b")
    assert spec.subkind == "generation"
    assert spec.kind == "within"


def test_within_pair_falls_back_to_other_when_tables_have_no_data():
    # Meta has no MODEL_GENERATION/MODEL_TOTAL_PARAMS_B entries at all —
    # must degrade to 'other', not raise.
    agents = [
        _FakeAgent("fake/llama-small", LLAMA_SMALL, "x"),
        _FakeAgent("fake/llama-large", LLAMA_LARGE, "x"),
    ]
    specs = pilot.build_panel_specs(agents)
    spec = _spec_for(specs, "fake/llama-small", "fake/llama-large")
    assert spec.subkind == "other"
    assert spec.kind == "within"


# ---------------------------------------------------------------------------
# Ladder reporting helpers (report_ladders / report_capability_regression):
# _slope_ci95 (OLS slope + 95% CI) and _ladder_agent_row (per-agent n/err/
# abstain/AUROC off the 'full' panel, since ladder members added via
# --models are never core members and so have no LOO panel of their own).
# ---------------------------------------------------------------------------

def test_slope_ci95_recovers_a_known_negative_slope():
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ys = [0.9 - 0.8 * x for x in xs]   # exact line, slope -0.8
    fit = pilot._slope_ci95(xs, ys)
    assert fit is not None
    slope, lo, hi, n = fit
    assert slope == pytest.approx(-0.8, abs=1e-6)
    assert lo <= slope <= hi
    assert n == 6


def test_slope_ci95_none_when_too_few_points_or_no_spread():
    assert pilot._slope_ci95([0.1, 0.2], [0.5, 0.6]) is None          # n < 3
    assert pilot._slope_ci95([0.3, 0.3, 0.3], [0.1, 0.5, 0.9]) is None  # no x-spread


def _row(uid, full_sem, grades, grades_judge=None, abstained=None):
    return {
        "uid": uid,
        "panels": {"full": {"semantic_entropy": full_sem}},
        "grades": grades,
        "grades_judge": grades_judge,
        "abstained": abstained or {},
    }


def test_ladder_agent_row_computes_err_rate_and_auroc_direction():
    # 'a' is wrong exactly when full-panel disagreement is high (sem=0.8),
    # so AUROC(scores vs a-is-wrong) should be a perfect 1.0. auroc_guarded
    # needs >= MIN_CLASS_FOR_AUROC (10) examples in EACH class, hence 12+12.
    rows = ([_row(f"wrong{i}", 0.8, {"a": False}) for i in range(12)]
            + [_row(f"right{i}", 0.2, {"a": True}) for i in range(12)])
    n, err, abst, auc, why = pilot._ladder_agent_row(rows, "a", "grades")
    assert n == 24
    assert err == pytest.approx(12 / 24)
    assert auc == pytest.approx(1.0)


def test_ladder_agent_row_none_grade_key_reports_no_graded_rows():
    rows = [_row("q1", 0.5, {"a": True}, grades_judge=None)]
    n, err, abst, auc, why = pilot._ladder_agent_row(rows, "a", "grades_judge")
    assert n == 0
    assert err is None
    assert auc is None
    assert why == "no graded rows"


if __name__ == "__main__":
    # Runnable without pytest too, for the fixture-free tests — see module
    # docstring. Tests that need pytest fixtures (monkeypatch, tmp_path) are
    # skipped here; run via `pytest` to exercise those.
    import inspect

    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if inspect.signature(fn).parameters:
                print(f"SKIP  {name} (needs pytest fixtures; run via `pytest`)")
                continue
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("\nAll tests passed.")
