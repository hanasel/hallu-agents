"""RAGTruth response-conditioned verification pilot (framing "c", claim-level).

[unchanged docstring — see prior version. Key idea: decompose each labelled
corpus response into atomic claims, have a diverse judge panel rule each claim
SUPPORTED/UNSUPPORTED/UNVERIFIABLE against the source, and compare ensemble
fact-checking against inter-judge disagreement as hallucination signals.]

This revision (v2):
  - 5-judge panel by default (finer disagreement resolution than 3).
  - max_disagreement and n_contested_claims promoted to reported signals;
    the n=20 pilot showed mean_disagreement washes out over ~12 claims while
    the MAX (any genuinely-contested claim) tracked the label as well as the
    ensemble. We report both and let the full run arbitrate.
  - bootstrap 95% CIs on every AUC, so results come with error bars — at
    n=20 the point estimates were not separable; this makes that explicit.
  - --diagnose-full dumps RAGTruth span annotations (label_type, implicit_true,
    due_to_null) for responses the ensemble calls fully-unsupported, to tell
    genuine catches from implicit_true / due_to_null false positives.

Run:
    python scripts/rgt_verify_pilot.py --task data2txt --n 200 --diagnose-full
    python scripts/rgt_verify_pilot.py --task summarization --n 150
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
from scipy.stats import spearmanr                                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score     # noqa: E402

from data import load_ragtruth                                        # noqa: E402
from agents import query_agents                                       # noqa: E402
from agents.panels import (                                           # noqa: E402
    make_agent,
    CROSS_FAMILY,
    GPT_OSS_LARGE,
    GPT_OSS_SMALL,
    LLAMA_SMALL,
    LLAMA_LARGE,
    QWEN,
)


# ---------------------------------------------------------------------------
# Prompts (unchanged)
# ---------------------------------------------------------------------------

DECOMPOSER_SYSTEM = (
    "You decompose a passage into atomic factual claims. An atomic claim is a "
    "single, self-contained, independently checkable statement. Split compound "
    "sentences. Resolve pronouns to their referents so each claim stands alone. "
    "Do NOT add any information not present in the text. Output one claim per "
    "line, plain text, with no numbering, bullets, or commentary."
)

VERIFIER_SYSTEM = (
    "You are a careful fact-checker. You are given SOURCE material and a single "
    "CLAIM. Decide whether the claim is fully supported by the source.\n"
    "Reply with exactly one word:\n"
    "  SUPPORTED    - the source clearly supports the claim.\n"
    "  UNSUPPORTED  - the source contradicts the claim, or the claim states "
    "information not present in the source.\n"
    "  UNVERIFIABLE - the source is insufficient to judge.\n"
    "Judge only against the source; ignore outside knowledge. Output the single "
    "word and nothing else."
)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# NEW: 5-judge cross-family verification panel
# ---------------------------------------------------------------------------

# Five judges spanning four model lineages (Meta / OpenAI-OSS / Alibaba), so
# disagreement reflects genuine cross-family divergence rather than sampling.
# All run at temperature 0 in the scoring loop. Falls back gracefully if a
# model id is unavailable — see build_judges().
VERIFY_PANEL_MODELS = [LLAMA_SMALL, LLAMA_LARGE, GPT_OSS_SMALL, GPT_OSS_LARGE, QWEN]


def build_judges(model_ids: List[str], system_prompt: str):
    """Instantiate judges, skipping any model that fails to construct."""
    judges = []
    for m in model_ids:
        try:
            judges.append(make_agent(m, system_prompt=system_prompt))
        except Exception as exc:                                       # noqa: BLE001
            print(f"  [skip] judge {m}: {exc}")
    if len(judges) < 3:
        raise RuntimeError(
            f"Only {len(judges)} judges available; need >=3. Check panel model ids."
        )
    return judges


# ---------------------------------------------------------------------------
# Parsing helpers (unchanged)
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def parse_claims(text: str, max_claims: int) -> List[str]:
    claims: List[str] = []
    for line in (text or "").splitlines():
        line = _BULLET_RE.sub("", line.strip())
        if not line or line.lower().startswith(("here are", "claims:", "atomic")):
            continue
        claims.append(line)
        if len(claims) >= max_claims:
            break
    return claims


def parse_verdict(text: str) -> str:
    t = (text or "").strip().upper()
    if "UNVERIFIABLE" in t:
        return "UNVERIFIABLE"
    if "UNSUPPORTED" in t or "NOT SUPPORTED" in t or "NOT FULLY" in t:
        return "UNSUPPORTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    return "UNVERIFIABLE"


def claim_disagreement(verdicts: List[str]) -> float:
    """1 - modal-verdict fraction. With 5 judges: 0, .2, .4, .6 resolution."""
    if not verdicts:
        return 0.0
    modal = Counter(verdicts).most_common(1)[0][1]
    return 1.0 - modal / len(verdicts)


# ---------------------------------------------------------------------------
# Pipeline steps (unchanged)
# ---------------------------------------------------------------------------

def decompose(decomposer, response_text: str, max_claims: int) -> List[str]:
    prompt = (
        "Decompose the following response into atomic factual claims.\n\n"
        f"<response>\n{response_text}\n</response>"
    )
    r = decomposer.query(prompt, temperature=0.0)
    if r.is_error:
        return []
    return parse_claims(r.text, max_claims)


def verify_claim(judges, source_text: str, claim: str) -> List[str]:
    prompt = f"SOURCE:\n{source_text}\n\nCLAIM: {claim}\n\nVerdict:"
    responses = query_agents(judges, prompt, temperature=0.0)
    return [parse_verdict(r.text) for r in responses]


# ---------------------------------------------------------------------------
# NEW: bootstrap CI for AUC
# ---------------------------------------------------------------------------

def auc_with_ci(y: np.ndarray, x: np.ndarray, n_boot: int = 2000,
                seed: int = 0) -> tuple:
    """Return (auc, lo, hi) with a percentile bootstrap 95% CI.

    Resamples (x, y) pairs with replacement. Skips resamples that end up
    single-class (AUC undefined). Reports the point estimate on the full data.
    """
    auc = roc_auc_score(y, x)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, xb = y[idx], x[idx]
        if yb.min() == yb.max():
            continue
        boots.append(roc_auc_score(yb, xb))
    if not boots:
        return auc, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auc, lo, hi


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RAGTruth response-conditioned verification pilot (v2).")
    ap.add_argument("--task", default="data2txt",
                    choices=["qa", "summarization", "data2txt"])
    ap.add_argument("--n", type=int, default=200, help="candidate responses to verify")  # CHANGED default
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decomposer", default=GPT_OSS_LARGE)
    ap.add_argument("--max-claims", type=int, default=12)
    ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--diagnose-full", action="store_true",                  # NEW
                    help="dump RAGTruth span annotations for responses the "
                         "ensemble calls FULLY unsupported (ens_unsup==1.0), "
                         "to separate genuine catches from implicit_true / "
                         "due_to_null false positives.")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for AUC CIs")  # NEW
    ap.add_argument("--out", default="outputs/rgt_verify_results.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth {args.task.upper()} [test] — {args.n} responses (seed={args.seed})")
    samples = load_ragtruth(split="test", task_types=[args.task],
                            n=args.n, seed=args.seed, quality="good")
    print(f"  Loaded {len(samples)} candidate responses.")
    base = statistics.mean(1.0 if s.is_hallucinated else 0.0 for s in samples)
    print(f"  hallucination base rate in this subset: {base:.1%}")

    section("Building agents")
    try:
        decomposer = make_agent(args.decomposer, system_prompt=DECOMPOSER_SYSTEM)
        judges = build_judges(VERIFY_PANEL_MODELS, VERIFIER_SYSTEM)          # NEW
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)
    print(f"  decomposer : {decomposer.name}")
    for j in judges:
        print(f"  judge      : {j.name}")
    print(f"  ({len(judges)} judges — disagreement resolution "
          f"{', '.join(f'{k/len(judges):.2f}' for k in range(len(judges)//2 + 1))})")

    # Preflight (unchanged logic)
    section("Preflight — decomposer + judges on a canary")
    canary_claims = decompose(decomposer, "Paris is the capital of France. It has 2 million residents.", 5)
    print(f"  decomposer produced {len(canary_claims)} claims: {canary_claims}")
    if not canary_claims:
        print("  [ABORT] decomposer returned no claims.")
        sys.exit(2)
    v = verify_claim(judges, "Paris is the capital of France.", canary_claims[0])
    print(f"  judge verdicts on claim 1: {v}")
    if all(x == "UNVERIFIABLE" for x in v):
        print("  [WARN] all judges UNVERIFIABLE — check verifier prompt / parsing.")

    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    rows = []
    full_unsup_diagnostics = []                                              # NEW
    for idx, s in enumerate(samples, start=1):
        claims = decompose(decomposer, s.response, args.max_claims)

        claim_rows = []
        for claim in claims:
            verdicts = verify_claim(judges, s.source_info_text, claim)
            majority = Counter(verdicts).most_common(1)[0][0]
            claim_rows.append({
                "claim": claim,
                "verdicts": {j.name: v for j, v in zip(judges, verdicts)},
                "majority": majority,
                "disagreement": claim_disagreement(verdicts),
            })

        n_claims = len(claim_rows)
        if n_claims == 0:
            ensemble_unsupported = mean_dis = max_dis = 0.0
            n_contested = 0
            frac_high = 0.0
        else:
            ensemble_unsupported = sum(
                1 for c in claim_rows if c["majority"] != "SUPPORTED"
            ) / n_claims
            dis = [c["disagreement"] for c in claim_rows]
            mean_dis = statistics.mean(dis)
            max_dis = max(dis)
            n_contested = sum(1 for d in dis if d > 0)                       # NEW
            frac_high = n_contested / n_claims

        row = {
            "uid": s.uid,
            "source_model": s.source_model,
            "is_hallucinated": s.is_hallucinated,
            "n_spans": len(s.hallucination_spans),
            "n_claims": n_claims,
            "ensemble_unsupported": ensemble_unsupported,
            "mean_disagreement": mean_dis,
            "max_disagreement": max_dis,                                     # promoted
            "n_contested_claims": n_contested,                              # NEW
            "frac_high_disagreement": frac_high,
            "claims": claim_rows,
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        # NEW: diagnose fully-unsupported responses against the gold spans.
        if args.diagnose_full and n_claims > 0 and ensemble_unsupported == 1.0:
            spans = [{
                "text": sp.text,
                "label_type": sp.label_type,
                "implicit_true": sp.implicit_true,
                "due_to_null": sp.due_to_null,
            } for sp in s.hallucination_spans]
            full_unsup_diagnostics.append({
                "uid": s.uid,
                "gold_hallucinated": s.is_hallucinated,
                "n_gold_spans": len(spans),
                "any_implicit_true": any(sp["implicit_true"] for sp in spans),
                "any_due_to_null": any(sp["due_to_null"] for sp in spans),
                "spans": spans,
            })

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} [{s.source_model}] "
                  f"gold_hallucinated={s.is_hallucinated} ---")
            for c in claim_rows:
                flag = "  <-- majority NOT supported" if c["majority"] != "SUPPORTED" else ""
                print(f"      [{c['majority']:<12} dis={c['disagreement']:.2f}] "
                      f"{c['claim'][:90]}{flag}")

        print(f"  [{idx:>3}/{len(samples)}] {s.uid}  claims={n_claims:>2}  "
              f"ens_unsup={ensemble_unsupported:.2f}  max_dis={max_dis:.2f}  "
              f"n_contested={n_contested:>2}  gold={'H' if s.is_hallucinated else '.'}")

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    section("Signals vs RAGTruth label (example-level, with bootstrap 95% CI)")
    y = np.array([1.0 if r["is_hallucinated"] else 0.0 for r in rows])
    print(f"  responses: {len(y)}   hallucinated: {int(y.sum())} ({y.mean():.1%})")
    empty = sum(1 for r in rows if r["n_claims"] == 0)
    if empty:
        print(f"  [!] {empty} responses yielded 0 claims (still scored; check decomposer).")

    if y.min() == y.max():
        print("  (degenerate: all responses same label; AUC undefined — widen --n / --seed)")
    else:
        print(f"\n  {'signal':<26}{'AUC-ROC':>9}{'95% CI':>18}{'AUC-PR':>9}{'Spearman':>11}")
        signal_names = (                                                     # CHANGED order/set
            "ensemble_unsupported",
            "max_disagreement",
            "n_contested_claims",
            "mean_disagreement",
            "frac_high_disagreement",
        )
        for name in signal_names:
            x = np.array([r[name] for r in rows], dtype=float)
            auc, lo, hi = auc_with_ci(y, x, n_boot=args.n_boot)
            apr = average_precision_score(y, x)
            rho, _ = spearmanr(x, y)
            print(f"  {name:<26}{auc:>9.3f}   [{lo:.3f}, {hi:.3f}]"
                  f"{apr:>9.3f}{rho:>+11.3f}")
        print(f"\n  base rate (AUC-PR reference): {y.mean():.3f}")
        print("  Read CIs first: if a disagreement signal's CI overlaps the")
        print("  ensemble's, they're statistically indistinguishable at this n.")

        # Redundancy: does disagreement add over the ensemble verdict?
        ens = np.array([r["ensemble_unsupported"] for r in rows])
        for name in ("max_disagreement", "mean_disagreement"):
            md = np.array([r[name] for r in rows])
            rho, p = spearmanr(ens, md)
            print(f"\n  Spearman(ensemble_unsupported, {name}): "
                  f"rho={rho:+.3f}  p={p:.3g}")
        print("  (low corr + competitive AUC => a distinct axis worth combining)")

    # NEW: fully-unsupported diagnostics
    if args.diagnose_full:
        section("Diagnostic — responses the ensemble called FULLY unsupported")
        if not full_unsup_diagnostics:
            print("  none.")
        else:
            n_fp = sum(1 for d in full_unsup_diagnostics if not d["gold_hallucinated"])
            n_it = sum(1 for d in full_unsup_diagnostics if d["any_implicit_true"])
            n_dn = sum(1 for d in full_unsup_diagnostics if d["any_due_to_null"])
            print(f"  {len(full_unsup_diagnostics)} responses at ens_unsup=1.00")
            print(f"    of which gold-CLEAN (false positives) : {n_fp}")
            print(f"    with any implicit_true span           : {n_it}")
            print(f"    with any due_to_null span             : {n_dn}")
            print("  (a gold-clean full-unsupported response is the method calling")
            print("   grounded content unfaithful — inspect its spans below)")
            for d in full_unsup_diagnostics[:8]:
                tag = "CLEAN(FP)" if not d["gold_hallucinated"] else "hallu"
                print(f"\n    {d['uid']} [{tag}] gold_spans={d['n_gold_spans']} "
                      f"implicit_true={d['any_implicit_true']} "
                      f"due_to_null={d['any_due_to_null']}")
                for sp in d["spans"][:3]:
                    print(f"      - [{sp['label_type']}] {sp['text'][:80]}")

    section("Done")
    print(f"  Rows written to: {out_path}\n")


if __name__ == "__main__":
    main()