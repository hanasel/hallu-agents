"""SUPERSEDED — see scripts/disagreement_pilot.py. Kept here for reference
only, moved out of scripts/ so it can't be imported by accident and diverge
from that script's grader the way it did before evaluation.grade_correct
unified the two (see that module's docstring for the history). This file's
correctness fixes (truncation gate, abstention as a third state, matched-N
panels, AUROC not means, grader independence, --resume, ...) were all ported
into disagreement_pilot.py, which now covers both TruthfulQA and SimpleQA
Verified, open-ended and multiple-choice, with one grader instead of two.

Original docstring follows.
---------------------------

First meaningful pilot: 50 TruthfulQA questions, cross-family panel.

Scales the 3-question smoke test (scripts/test_disagreement.py) up to a real
pilot and folds in the two new capabilities:

  1. Semantic-entropy disagreement (NLI meaning-clustering) alongside Jaccard,
     to show it removes the lexical false positives.
  2. Two cross-family agents (OpenAI GPT-OSS-20B, Qwen3.6-27B) added to the
     two Meta Llamas, to test whether genuine model diversity breaks the
     shared-bias failure.

In cross mode (default) the query panel is the union of same_family_panel
(2 Llamas) and cross_family_panel (1 Llama + GPT-OSS + Qwen) — 4 distinct
agents, not a concatenation, since both panels contain llama-3.3-70b-versatile.

What it reports
---------------
  A. Measure comparison  — mean Jaccard vs mean semantic entropy, and the
     false-positive reduction: questions where Jaccard flags disagreement but
     the agents actually agree in meaning (semantic entropy = 0).
  B. Shared-bias test     — questions where the two Llamas agree (same-family
     disagreement = 0) but a cross-family model dissents (full-panel
     disagreement > 0), and how often that dissent coincides with the Llamas
     being wrong (NLI-graded against the gold answer).

Correctness is an NLI proxy (same backend as the measure): an answer is
"correct" if it is mutually-entailing with the gold answer in the question's
context. This is a coarse grade, flagged as such — not a substitute for the
answer-level fuzzy matcher planned for full evaluation.

Requires: OPENROUTER_API_KEY, plus torch + sentence-transformers for the NLI model.
Responses are cached (agents/) so re-runs are fast and deterministic.

Run from the project root:
    python scripts/pilot.py --n 50
    python scripts/pilot.py --n 50 --no-cross        # same-family only
    python scripts/pilot.py --n 20 --nli-model cross-encoder/nli-deberta-v3-base
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa, TRUTHFULQA_QUERY_CONFIG              # noqa: E402
from agents import (                                                   # noqa: E402
    query_agents,
    assert_models_available,
    PermanentAgentError,
    PERMANENT_ERROR_PREFIX,
)
from agents.panels import (                                            # noqa: E402
    cross_family_panel,
    same_family_panel,
    family_of,
    LLAMA_SMALL,
    LLAMA_LARGE,
)
from disagreement import (                                             # noqa: E402
    JaccardDisagreement,
    SemanticEntropyDisagreement,
    CrossEncoderNLI,
    semantically_equivalent,
)
from evaluation import grade_correct                                   # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def _peek_nli(semantic, texts, question, names) -> None:
    """Print pairwise NLI labels + conditioned answers for one question."""
    nli = semantic.nli
    cond = [f"{question} {t}".strip() for t in texts]
    print("      pairwise NLI (premise -> hypothesis):")
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i == j:
                continue
            label = nli.predict(cond[i], cond[j])
            print(f"        {names[i].split('/')[-1]:>22} -> "
                  f"{names[j].split('/')[-1]:<22} {label}")


# ---------------------------------------------------------------------------
1# Pilot
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-family disagreement pilot.")
    ap.add_argument("--n", type=int, default=50, help="number of TruthfulQA questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cross", action="store_true",
                    help="use the same-family (2 Llama) panel only")
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--strict", action="store_true",
                    help="strict clustering (merge only on mutual entailment). "
                         "Default is relaxed (merge unless a pair contradicts), "
                         "which suits heterogeneous inter-agent answers.")
    ap.add_argument("--single-linkage", action="store_true",
                    help="use single-linkage clustering (compare only to a cluster "
                         "representative). Default is complete-linkage (compare to "
                         "every member), which is more robust to noisy NLI.")
    ap.add_argument("--no-concise", action="store_true",
                    help="do NOT force short answers (reproduces the saturated baseline)")
    ap.add_argument("--peek", type=int, default=0,
                    help="print answers + pairwise NLI labels for the first N questions")
    ap.add_argument("--out", default="outputs/pilot_results.jsonl")
    args = ap.parse_args()

    section(f"Loading {args.n} TruthfulQA MC1 questions (seed={args.seed})")
    samples = load_truthfulqa(n=args.n, seed=args.seed)
    print(f"  Loaded {len(samples)} questions.")

    section("Building agent panel")
    panel_kwargs = {} if args.no_concise else dict(TRUTHFULQA_QUERY_CONFIG)
    try:
        same_family_agents = same_family_panel(**panel_kwargs)
        if args.no_cross:
            agents = same_family_agents
        else:
            cross_agents = cross_family_panel(**panel_kwargs)
            # Union, not concatenation: cross_family_panel already contains
            # LLAMA_LARGE (one model per family), so re-adding it from
            # same_family_agents would query — and print — it twice.
            agents = same_family_agents + [a for a in cross_agents if a.model != LLAMA_LARGE]
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)
    print(f"  answer style: "
          f"{'verbose (baseline)' if args.no_concise else 'concise (forced short answers)'}")
    for a in agents:
        print(f"  - {a.name:<40} [{family_of(a.model)}]")
    has_cross = not args.no_cross
    # Identity, not position: cross mode's panel has one Llama mixed in with
    # two other-family models, so "first two agents" no longer means "the
    # Llama pair" — see the module docstring.
    llama_indices = [i for i, a in enumerate(agents) if a.model in (LLAMA_SMALL, LLAMA_LARGE)]

    # Preflight: confirm every panel model id is actually being served, before
    # spending a single token. A dead or typo'd model id (e.g. a model
    # decommissioned mid-run) fails here in ~1 API call instead of after
    # MAX_RETRIES rounds of backoff on question 1 of `args.n`.
    section("Preflight — model IDs")
    try:
        assert_models_available(agents)
    except PermanentAgentError as exc:
        print(f"\n  [ABORT] {exc}\n")
        sys.exit(2)
    print("  All panel models are live.")

    # Preflight: one canary question. Catch a mute/erroring agent here instead
    # of after 50 questions. An empty answer usually means either (a) the call
    # errored (bad param, auth, rate limit) — shown via r.error — or (b) a
    # reasoning model's answer was truncated by too small a token budget.
    section("Preflight — checking every agent actually answers")
    canary = "What is the capital of France? Answer in one word."
    ok = True
    for a, r in zip(agents, query_agents(agents, canary)):
        empty = not r.text.strip()
        note = ""
        if empty:
            ok = False
            if r.error:
                note = f"  <-- ERROR: {r.error}"
            else:
                note = (f"  <-- EMPTY, no error (finish_reason={r.finish_reason}); "
                        f"likely truncated — raise max_tokens")
        print(f"  {a.name:<40} -> {r.text.strip()[:40]!r}{note}")
    if not ok:
        print("\n  [ABORT] An agent returned no usable text (see the note above).")
        print("  - 'ERROR: TypeError ...unexpected keyword argument' => provider-")
        print("    specific params must go via extra_body, not top-level kwargs")
        print("    (agents/groq_agent.py's GroqAgent handles this already).")
        print("  - 'EMPTY, no error' with finish_reason='length' => raise")
        print("    TRUTHFULQA_QUERY_CONFIG['max_tokens'] in data/query_config.py")
        print("    (applies uniformly to every panel agent, by design — see")
        print("    agents/panels._assert_uniform_query_settings).\n")
        sys.exit(2)
    print("  All agents responded.")

    # One NLI backend shared by the measure AND the correctness grader, so
    # clustering and grading use identical semantics (and share the cache).
    print(f"\n  Loading NLI model: {args.nli_model} "
          f"(first run downloads weights)...")
    nli = CrossEncoderNLI(model_name=args.nli_model)

    jaccard = JaccardDisagreement()
    linkage = "single" if args.single_linkage else "complete"
    semantic = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                           linkage=linkage)
    print(f"  clustering mode: {'strict' if args.strict else 'relaxed'} + "
          f"{linkage}-linkage")

    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")   # truncate; avoid appending to a stale run

    rows = []
    n_question_errors = 0
    for idx, s in enumerate(samples, start=1):
        responses = query_agents(agents, s.prompt)
        errored = [(a, r) for a, r in zip(agents, responses) if r.is_error]
        if errored:
            # Don't silently score through a failed call — an agent error
            # means this row's disagreement numbers would be junk (comparing
            # real answers against ""), and a run that quietly drops rows
            # ends up with a results file whose row count doesn't match `n`.
            n_question_errors += 1
            for a, r in errored:
                print(f"  [{idx:>2}/{len(samples)}] {s.uid}  [!] {a.name}: {r.error}")
            permanent = any(r.error.startswith(PERMANENT_ERROR_PREFIX) for _, r in errored)
            if permanent or n_question_errors > 3:
                raise RuntimeError(
                    f"Aborting after {n_question_errors} question(s) with agent "
                    f"errors (latest: {errored[-1][0].name}: {errored[-1][1].error}). "
                    f"{len(rows)} good row(s) already written to {out_path} — "
                    f"cached answers won't be re-billed on re-run, so fix the "
                    f"underlying issue and re-run."
                )
            continue

        texts = [r.text for r in responses]
        empties = [not t.strip() for t in texts]

        jac = jaccard.score(responses).score
        sem = semantic.score(responses, question=s.question)

        # Same-family-only view (the two Llamas, found by model id via
        # llama_indices — see above) for the shared-bias comparison. Only
        # meaningful when cross-family agents were actually added.
        same_responses = [responses[i] for i in llama_indices]
        sem_same = semantic.score(same_responses, question=s.question) if has_cross else None

        # This pilot only ever queries the open-ended prompt (s.prompt), so
        # grading routes through the NLI proxy, not letter extraction.
        grades = [grade_correct(nli, t, s, prompt_format="open") for t in texts]
        llama_grades = [grades[i] for i in llama_indices]
        both_llamas_wrong = all(g is False for g in llama_grades)
        llamas_agree = sem_same is not None and sem_same.details["n_clusters"] == 1

        # Whole-panel agreement + correctness, for the shared-bias analysis.
        panel_agrees = sem.details["n_clusters"] == 1
        graded = [g for g in grades if g is not None]
        panel_majority_wrong = bool(graded) and sum(g is False for g in graded) > len(graded) / 2

        # Per-agent cluster id, for the grader audit's same-cluster/different-
        # grade inconsistency check (scripts/audit_grader.py).
        cluster_of = {}
        for cid, members in enumerate(sem.details["clusters"]):
            for member_idx in members:
                cluster_of[agents[member_idx].name] = cid

        row = {
            "uid": s.uid,
            "category": s.category,
            "question": s.question,
            "correct_answer": s.correct_answer,
            "responses": {a.name: t for a, t in zip(agents, texts)},
            "grades": {a.name: g for a, g in zip(agents, grades)},
            "jaccard": jac,
            "semantic_entropy": sem.score,
            "n_clusters": sem.details["n_clusters"],
            "cluster_sizes": sem.details["cluster_sizes"],
            "semantic_entropy_same_family": (sem_same.score if sem_same else None),
            "n_clusters_same_family": (sem_same.details["n_clusters"] if sem_same else None),
            "llamas_agree": llamas_agree,
            "both_llamas_wrong": both_llamas_wrong,
            "panel_agrees": panel_agrees,
            "panel_majority_wrong": panel_majority_wrong,
            "cluster_of": cluster_of,
            "finish_reasons": {a.name: r.finish_reason for a, r in zip(agents, responses)},
            "empty_count": sum(empties),
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} [{s.category}] ---")
            print(f"      Q: {s.question}")
            print(f"      gold: {s.correct_answer}")
            for a, t, g in zip(agents, texts, grades):
                gl = {True: "OK ", False: "WRONG", None: "?  "}[g]
                print(f"      [{gl}] {a.name.split('/')[-1]}: {t.strip()[:200]}")
            _peek_nli(semantic, texts, s.question, [a.name for a in agents])
            print(f"      -> clusters={sem.details['cluster_sizes']} sem={sem.score:.2f}")

        flag = ""
        if jac >= 0.5 and sem.score == 0.0:
            flag = "  <- Jaccard false positive removed"
        print(f"  [{idx:>2}/{len(samples)}] {s.uid}  "
              f"jac={jac:.2f}  sem={sem.score:.2f}  "
              f"clusters={sem.details['cluster_sizes']}{flag}")

    # ------------------------------------------------------------------ #
    # A. Measure comparison
    # ------------------------------------------------------------------ #
    section("A. Answer-level (Jaccard) vs semantic entropy")
    mean_jac = statistics.mean(r["jaccard"] for r in rows)
    mean_sem = statistics.mean(r["semantic_entropy"] for r in rows)
    print(f"  mean Jaccard disagreement          : {mean_jac:.3f}")
    print(f"  mean semantic-entropy disagreement : {mean_sem:.3f}")

    total_empty = sum(r["empty_count"] for r in rows)
    if total_empty:
        print(f"  [!] {total_empty} EMPTY responses slipped through — an agent was "
              "truncated. Results below are unreliable; fix token budget and re-run.")

    # Sanity: cluster-count spread. If almost every question is 'all distinct',
    # the metric has saturated (usually verbose answers vs sentence-level NLI).
    from collections import Counter
    cc = Counter(r["n_clusters"] for r in rows)
    print(f"  n_clusters spread                  : "
          + ", ".join(f"{k}->{cc[k]}" for k in sorted(cc)))
    if cc.get(len(agents), 0) > 0.7 * len(rows):
        print("  [!] metric appears SATURATED (most questions = all-distinct). "
              "Run scripts/inspect_pilot.py; check answers aren't verbose and "
              "that you're not forcing --strict.")

    # Grade distribution — shows whether the NLI proxy can actually grade
    # answers. High 'unclear' means shared-bias detection is being masked.
    grade_ctr = Counter()
    for r in rows:
        for g in r["grades"].values():
            grade_ctr[{True: "correct", False: "incorrect", None: "unclear"}[g]] += 1
    tot = sum(grade_ctr.values())
    print(f"  correctness grades                 : "
          + ", ".join(f"{k}={grade_ctr.get(k,0)}" for k in ("correct", "incorrect", "unclear"))
          + f"  ({tot} total)")
    if grade_ctr.get("unclear", 0) > 0.5 * tot:
        print("  [!] most answers grade 'unclear' — grader can't match verbose "
              "answers to short gold; Part B shared-bias counts are unreliable.")

    false_pos = [r for r in rows if r["jaccard"] >= 0.5 and r["semantic_entropy"] == 0.0]
    print(f"\n  Questions Jaccard flags (>=0.50) but agents AGREE in meaning "
          f"(semantic entropy = 0): {len(false_pos)} / {len(rows)}")
    print("  These are the false positives semantic entropy removes. Examples:")
    for r in false_pos[:5]:
        print(f"    - {r['uid']} [{r['category']}]  jac={r['jaccard']:.2f}")
        print(f"        Q: {r['question'][:80]}")

    # ------------------------------------------------------------------ #
    # B. Shared-bias / cross-family test
    # ------------------------------------------------------------------ #
    if has_cross:
        section("B. Shared-bias test — do the cross-family models break ties?")
        tie_broken = [r for r in rows
                      if r["n_clusters_same_family"] == 1 and r["n_clusters"] > 1]
        print(f"  Two Llamas agree but a cross-family model dissents: "
              f"{len(tie_broken)} / {len(rows)} questions.")

        # Of those, how many are cases where BOTH Llamas were actually wrong?
        # These are exactly the shared-bias failures the cross-family models
        # are meant to rescue: same-family disagreement said 'agree' (miss),
        # a cross-family model reintroduces the signal.
        rescued = [r for r in tie_broken if r["both_llamas_wrong"]]
        print(f"  ...of which both Llamas were graded WRONG (shared-bias "
              f"failures surfaced by a cross-family model): {len(rescued)}")
        for r in rescued[:5]:
            print(f"    - {r['uid']} [{r['category']}]")
            print(f"        Q: {r['question'][:80]}")
            print(f"        gold: {r['correct_answer'][:80]}")

        # Contrast: mean disagreement on the two panels.
        mean_sem_same = statistics.mean(
            r["semantic_entropy_same_family"] for r in rows
            if r["semantic_entropy_same_family"] is not None
        )
        print(f"\n  mean semantic entropy, same-family (2 Llamas)  : {mean_sem_same:.3f}")
        print(f"  mean semantic entropy, full panel ({len(agents)} models): {mean_sem:.3f}")
        print(f"  → diversity raises the panel's disagreement by "
              f"{mean_sem - mean_sem_same:+.3f} on average.")

        # The hard limit of any disagreement signal: when the WHOLE panel agrees
        # (1 cluster) but is wrong, no amount of agents helps — a shared prior
        # fooled everyone, cross-family included. These are the shared-bias
        # FALSE NEGATIVES the detector structurally cannot catch.
        shared_bias_fn = [r for r in rows
                          if r["panel_agrees"] and r["panel_majority_wrong"]]
        print(f"\n  Shared-bias FALSE NEGATIVES (whole panel agrees AND is wrong): "
              f"{len(shared_bias_fn)} / {len(rows)}")
        print("  Disagreement is blind to these — they motivate a grounding signal.")
        for r in shared_bias_fn[:5]:
            print(f"    - {r['uid']} [{r['category']}]  sem={r['semantic_entropy']:.2f}")
            print(f"        Q: {r['question'][:80]}")
            print(f"        gold: {r['correct_answer'][:80]}")

    section("Done")
    print(f"  Per-question rows written to: {out_path}")
    print(f"  Inspect with:  cat {out_path} | python -m json.tool | less\n")


if __name__ == "__main__":
    main()