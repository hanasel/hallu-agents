"""Offline sanity test for the semantic-entropy disagreement measure.

Runs with NO model download and NO API calls: it injects a deterministic
`FakeNLI` backend whose entailment relation is defined by hand-specified
meaning groups. This isolates and verifies the clustering + entropy maths.

The real backend (`CrossEncoderNLI`, DeBERTa-v3 MNLI) is exercised only in the
live pilot (`scripts/pilot.py`), which needs torch + a GROQ_API_KEY.

Run from the project root:
    python scripts/test_semantic.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disagreement import (                                    # noqa: E402
    SemanticEntropyDisagreement,
    JaccardDisagreement,
    NLILabel,
)


# ---------------------------------------------------------------------------
# Deterministic fake NLI backend
# ---------------------------------------------------------------------------

class FakeNLI:
    """NLI backend driven by explicit meaning groups.

    Each group is a list of substrings. A response text is assigned to the
    first group containing a substring that occurs in it. Two texts entail
    each other iff they land in the same group; otherwise they contradict.
    Context prefixes are ignored (the substrings are chosen to be unambiguous).
    """

    def __init__(self, groups: Sequence[Sequence[str]]):
        self.groups = [list(g) for g in groups]

    def _group_of(self, text: str) -> int:
        low = text.lower()
        for gi, subs in enumerate(self.groups):
            if any(s.lower() in low for s in subs):
                return gi
        return -1  # ungrouped → its own singleton (matches nothing)

    def predict(self, premise: str, hypothesis: str) -> NLILabel:
        gp, gh = self._group_of(premise), self._group_of(hypothesis)
        if gp != -1 and gp == gh:
            return "entailment"
        return "contradiction"

    def predict_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLILabel]:
        return [self.predict(p, h) for p, h in pairs]


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_agree_is_zero() -> None:
    # Lexically very different, semantically identical → the false positive
    # that Jaccard gets wrong and semantic entropy must get right.
    responses = [
        "Neil Armstrong.",
        "The first person to walk on the Moon was Neil Armstrong.",
        "Armstrong — he was the first human on the lunar surface.",
    ]
    nli = FakeNLI(groups=[["armstrong"]])
    se = SemanticEntropyDisagreement(nli=nli)
    r = se.score(responses, question="Who was the first person on the Moon?")

    jac = JaccardDisagreement().score(responses).score

    print(f"  semantic entropy : {r.score:.3f}  (n_clusters={r.details['n_clusters']})")
    print(f"  jaccard          : {jac:.3f}  <-- inflated false positive")
    assert r.details["n_clusters"] == 1, r.details
    assert approx(r.score, 0.0), r.score
    assert jac > 0.5, jac  # Jaccard wrongly reports strong disagreement
    print("  [OK] one meaning-class -> semantic entropy 0.0 while Jaccard is high")


def test_all_differ_is_one() -> None:
    responses = ["Paris.", "Berlin.", "Madrid."]
    nli = FakeNLI(groups=[["paris"], ["berlin"], ["madrid"]])
    se = SemanticEntropyDisagreement(nli=nli)
    r = se.score(responses, question="Capital?")
    print(f"  semantic entropy : {r.score:.3f}  (n_clusters={r.details['n_clusters']})")
    assert r.details["n_clusters"] == 3, r.details
    assert approx(r.score, 1.0), r.score
    print("  [OK] three meaning-classes -> normalised entropy 1.0")


def test_two_vs_one_split() -> None:
    # The music/thinking case from test_disagreement.py Part 1, case 4.
    responses = [
        "Yes, learning music helps thinking.",
        "Music education improves cognitive abilities.",     # agrees with #0
        "No, evidence does not support music boosting cognition.",  # dissents
    ]
    nli = FakeNLI(groups=[["helps thinking", "improves cognitive"], ["does not support"]])
    se = SemanticEntropyDisagreement(nli=nli)
    r = se.score(responses, question="Does learning music help thinking?")

    # sizes [2, 1] over N=3: H = -(2/3 ln 2/3 + 1/3 ln 1/3); H/ln3 = 0.5793...
    expected_nats = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
    expected_norm = expected_nats / math.log(3)
    print(f"  cluster sizes    : {r.details['cluster_sizes']}")
    print(f"  semantic entropy : {r.score:.4f}  (expected {expected_norm:.4f})")
    assert sorted(r.details["cluster_sizes"], reverse=True) == [2, 1], r.details
    assert approx(r.score, expected_norm), (r.score, expected_norm)
    print("  [OK] 2-vs-1 split -> 0.579, the correct intermediate value")


def test_raw_nats_mode() -> None:
    responses = ["Paris.", "Berlin.", "Madrid."]
    nli = FakeNLI(groups=[["paris"], ["berlin"], ["madrid"]])
    se = SemanticEntropyDisagreement(nli=nli, normalise=False)
    r = se.score(responses, question="Capital?")
    assert approx(r.score, math.log(3)), r.score
    print(f"  [OK] raw-nats mode -> ln(3) = {math.log(3):.4f}")


def test_empty_response_is_singleton() -> None:
    # An errored agent (empty text) must not merge into any cluster.
    responses = ["Armstrong.", "", "Neil Armstrong was first."]
    nli = FakeNLI(groups=[["armstrong"]])
    se = SemanticEntropyDisagreement(nli=nli)
    r = se.score(responses, question="Who?")
    # #0 and #2 merge (armstrong); "" is its own class -> 2 clusters, sizes [2,1]
    assert r.details["n_clusters"] == 2, r.details
    assert sorted(r.details["cluster_sizes"], reverse=True) == [2, 1], r.details
    print(f"  [OK] empty/errored response -> its own singleton cluster {r.details['cluster_sizes']}")


def test_relaxed_vs_strict() -> None:
    # Under relaxed clustering, 'neutral' pairs merge; under strict they don't.
    class NeutralBetween01(FakeNLI):
        def predict(self, premise: str, hypothesis: str) -> NLILabel:
            if "a-text" in premise and "b-text" in hypothesis:
                return "neutral"
            if "b-text" in premise and "a-text" in hypothesis:
                return "neutral"
            return super().predict(premise, hypothesis)

    responses = ["a-text here", "b-text here"]
    nli = NeutralBetween01(groups=[["a-text"], ["b-text"]])

    strict = SemanticEntropyDisagreement(nli=nli, strict_entailment=True)
    relaxed = SemanticEntropyDisagreement(nli=nli, strict_entailment=False)
    rs = strict.score(responses)
    rr = relaxed.score(responses)
    assert rs.details["n_clusters"] == 2, rs.details
    assert rr.details["n_clusters"] == 1, rr.details
    print(f"  [OK] strict keeps neutral pair apart (2 clusters); "
          f"relaxed merges it (1 cluster)")


def test_reasoning_strip() -> None:
    # A GPT-OSS/Qwen-style leaked chain-of-thought must be stripped before NLI.
    from disagreement.semantic import _extract_text
    raw = "<think>Let me reason step by step about the Moon landing...</think>\nNeil Armstrong."
    assert _extract_text(raw) == "Neil Armstrong.", repr(_extract_text(raw))
    print("  [OK] <think>...</think> stripped from response body")


def test_key_fn_overrides_nli_and_is_reported_in_details() -> None:
    # FakeNLI groups "1986" and "1985" together (both contain "19"), which is
    # exactly the failure mode the key_fn branch exists to short-circuit: a
    # date key_fn must split them even though the injected NLI would merge
    # them, and score() must report that the split was key- not NLI-decided.
    def date_key_fn(text, answer_type, question=None):
        digits = "".join(ch for ch in text if ch.isdigit())
        return frozenset([digits]) if digits else None

    nli = FakeNLI(groups=[["19"]])   # would merge "...1986" and "...1985"
    se = SemanticEntropyDisagreement(nli=nli, key_fn=date_key_fn)
    r = se.score(["It happened in 1986.", "It happened in 1985."],
                question="When?", answer_type="Date")

    assert r.details["n_clusters"] == 2, r.details
    assert r.details["key_fn_used"] is True
    assert r.details["key_decided_comparisons"] == 1
    print(f"  [OK] key_fn split a pair FakeNLI would have merged "
          f"(n_clusters={r.details['n_clusters']}, "
          f"key_decided={r.details['key_decided_comparisons']})")


def test_key_fn_none_falls_back_to_nli_unchanged() -> None:
    # key_fn returning None for both responses must fall straight through to
    # NLI, with nothing recorded as key-decided — the no-op path.
    def always_none(text, answer_type, question=None):
        return None

    nli = FakeNLI(groups=[["armstrong"]])
    se = SemanticEntropyDisagreement(nli=nli, key_fn=always_none)
    r = se.score(["Armstrong.", "Neil Armstrong."], question="Who?")

    assert r.details["n_clusters"] == 1, r.details
    assert r.details["key_fn_used"] is True
    assert r.details["key_decided_comparisons"] == 0
    print("  [OK] key_fn returning None everywhere defers entirely to NLI")


def test_key_fn_subset_merges_and_is_counted_separately_from_equality() -> None:
    # A terse key contained in a more detailed key must merge (Edit 2), and
    # the merge must be counted as "subset", not "equality" (Edit 3) — this
    # is what lets the eval attribute any improvement to the right edit.
    def number_key_fn(text, answer_type, question=None):
        digits = frozenset(ch for ch in text.split() if ch.isdigit())
        return digits or None

    nli = FakeNLI(groups=[])   # would never merge on its own
    se = SemanticEntropyDisagreement(nli=nli, key_fn=number_key_fn)
    r = se.score(["300", "300 1780"], question="How many?", answer_type="Number")

    assert r.details["n_clusters"] == 1, r.details
    assert r.details["key_decided_comparisons"] == 1
    assert r.details["key_decided_by_subset"] == 1
    assert r.details["key_decided_by_equality"] == 0
    print("  [OK] subset match merges a terse/detailed pair, counted as 'subset'")

    r2 = se.score(["300", "300"], question="How many?", answer_type="Number")
    assert r2.details["n_clusters"] == 1, r2.details
    assert r2.details["key_decided_by_equality"] == 1
    assert r2.details["key_decided_by_subset"] == 0
    print("  [OK] identical keys are counted as 'equality', not 'subset'")


def test_key_fn_emptied_by_subtraction_falls_through_to_nli() -> None:
    # A key_fn that empties under a question-conditioned key (Edit 1) must
    # fall through to NLI for that pair, and the emptying must be counted —
    # this is what lets the eval attribute any improvement to Edit 1 itself.
    def subtracting_key_fn(text, answer_type, question=None):
        nums = frozenset(ch for ch in text.split() if ch.isdigit())
        if question:
            q_nums = frozenset(ch for ch in question.split() if ch.isdigit())
            nums = nums - q_nums
        return nums or None

    nli = FakeNLI(groups=[["1864"]])   # NLI merges once the key is emptied
    se = SemanticEntropyDisagreement(nli=nli, key_fn=subtracting_key_fn)
    r = se.score(["1864", "1864"], question="It happened in 1864",
                answer_type="Number")

    assert r.details["n_clusters"] == 1, r.details
    assert r.details["key_decided_comparisons"] == 0
    assert r.details["key_emptied_by_subtraction"] == 2
    print("  [OK] both keys emptied by question-subtraction -> NLI decided, "
          f"key_emptied_by_subtraction={r.details['key_emptied_by_subtraction']}")


def main() -> None:
    section("Semantic entropy — offline clustering tests (FakeNLI)")
    for fn in (
        test_all_agree_is_zero,
        test_all_differ_is_one,
        test_two_vs_one_split,
        test_raw_nats_mode,
        test_empty_response_is_singleton,
        test_relaxed_vs_strict,
        test_reasoning_strip,
        test_key_fn_overrides_nli_and_is_reported_in_details,
        test_key_fn_none_falls_back_to_nli_unchanged,
        test_key_fn_subset_merges_and_is_counted_separately_from_equality,
        test_key_fn_emptied_by_subtraction_falls_through_to_nli,
    ):
        print(f"\n-- {fn.__name__}")
        fn()

    section("All semantic-entropy offline tests passed")
    print("Clustering + entropy verified. Live NLI runs in scripts/pilot.py.\n")


if __name__ == "__main__":
    main()
