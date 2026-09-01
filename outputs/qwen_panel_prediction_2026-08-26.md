# Pre-registered prediction: family effect on TruthfulQA vs SimpleQA

Recorded 2026-08-26, before any TruthfulQA run that includes the Qwen
size/generation ladder extension (`agents.panels.QWEN_PANEL_NEW_MODELS`,
run via `--qwen-ladders`). Written in advance so the TruthfulQA run is a
falsifiable check against a committed direction, not post-hoc pattern-matching
on whatever number comes out.

## Baseline, verified against the current cache (not the planning brief's figure)

The implementation brief this note accompanies (`qwen_panel_setup.md`) cited a
prior SimpleQA family effect of "+0.052, rank-biserial r=+0.090" over five
within-family pairs. That figure does not reproduce against this repo: no
occurrence of it exists anywhere in the codebase or git history, and the
brief's assumed starting point (a 7-agent pool, `scripts/simpleqa_pilot.py`,
`scripts/pilot.py`) does not match what's on `main` — see the implementation
notes for the full discrepancy list.

Re-running `scripts/disagreement_pilot.py --dataset simpleqa --analyse-only
--out outputs/simpleqa_3x3_results_1000.jsonl` against the actual current
9-agent 3x3 core pool (Meta/OpenAI/Qwen x small/large/strong; 1000 cached
SimpleQA Verified questions) gives, as of this run:

    PAIRED within→cross difference: +0.016  (n=1000/1000)
      Wilcoxon signed-rank p = 6.88e-02, rank-biserial r = -0.026

i.e. on the current pool the paired family effect is small, not clearly
signed, and not significant at p<0.05. This is the actual pre-extension
SimpleQA baseline this prediction is made relative to — not the brief's
unverified figure.

## The prediction

SimpleQA errors are long-tail recall failures — idiosyncratic, so models fail
differently and disagreement stays informative even within a family.
TruthfulQA errors are shared web misconceptions, which is precisely the
mechanism that should make same-lineage models fail *identically*.

**Prediction:** the family effect (paired within→cross difference in panel
disagreement, and its rank-biserial effect size) will be **larger on
TruthfulQA than the +0.016 / r=-0.026 baseline above**, once the Qwen
size/generation ladder extension is queried and included in the pool. The
`within:size` panels specifically (`agents.panels.SIZE_LADDER_QWEN35` —
identical corpus, different capacity) are predicted to show the largest
degradation in panel disagreement, since size varies with lineage and recipe
held exactly fixed.

Note the existing cached TruthfulQA run (`outputs/truthfulqa_results.jsonl`,
790 questions, current 9-agent core pool, no ladder extension) already shows a
much larger effect than SimpleQA even without the ladder extension: paired
within→cross difference +0.075, Wilcoxon p=1.03e-24, rank-biserial r=+0.399.
That is consistent with the direction of this prediction and is the number to
beat once the ladder models are added — the prediction is that adding
same-corpus, different-capacity Qwen pairs pushes the `within:size (Qwen)`
figure specifically below the pooled within-family average already observed
here, not just that TruthfulQA > SimpleQA in aggregate (which is already
true).

## What would falsify this

- `within:size (Qwen)` panel disagreement (SIZE_LADDER_QWEN35 pairs) on
  TruthfulQA comes out *higher* than `within:generation (Qwen)` or
  `within:other`, or comparable to the cross-family figure.
- The TruthfulQA paired within→cross effect size does not exceed the
  SimpleQA baseline recorded above once both are run on the same
  (ladder-extended) pool.
