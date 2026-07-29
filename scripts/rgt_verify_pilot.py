"""RAGTruth response-conditioned verification pilot (framing "c", claim-level).

Decompose each labelled corpus response into atomic claims, have a diverse
judge panel rule each claim SUPPORTED/UNSUPPORTED/UNVERIFIABLE against the
source, and compare ensemble fact-checking against inter-judge disagreement.

v3 changes
----------
  - ERROR PROPAGATION. A rate-limited judge used to return empty text, which
    parse_verdict() mapped to UNVERIFIABLE — a silent vote that inflated both
    ens_unsup and n_contested. Quota exhaustion therefore produced plausible
    numbers instead of stopping. Now any errored call aborts the run with the
    cache intact, so a partial run is honest and resumable.
  - BATCHED VERIFICATION (--batch-claims). One call per judge per RESPONSE
    (all claims in a numbered list) instead of one per judge per CLAIM:
    ~5 calls/response instead of ~60. Includes strict alignment validation and
    a per-claim fallback if a judge's reply can't be parsed.
  - --compare-batching runs both modes on the first K responses and reports
    verdict agreement, so batching is validated rather than assumed.
  - Responses that yield 0 claims are EXCLUDED from scoring rather than counted
    as confident negatives.

Run:
    # 1. validate batching is faithful (uses cached per-claim verdicts)
    python scripts/rgt_verify_pilot.py --task data2txt --compare-batching 10

    # 2. then scale
    python scripts/rgt_verify_pilot.py --task data2txt --n 200 \
        --batch-claims --max-claims 25 --diagnose-full
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
    make_agent, GPT_OSS_LARGE, GPT_OSS_SMALL,
    LLAMA_SMALL, LLAMA_LARGE, QWEN,
)


# ---------------------------------------------------------------------------
# Prompts
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

# NEW: batched variant. Deliberately mirrors the per-claim wording so the only
# difference is presentation, not the decision criterion.
BATCH_VERIFIER_SYSTEM = (
    "You are a careful fact-checker. You are given SOURCE material and a "
    "numbered list of CLAIMS. Judge each claim INDEPENDENTLY against the "
    "source; do not let one claim's verdict influence another.\n"
    "For each claim output exactly one line of the form:\n"
    "  <number>. SUPPORTED\n"
    "  <number>. UNSUPPORTED\n"
    "  <number>. UNVERIFIABLE\n"
    "SUPPORTED = the source clearly supports the claim. UNSUPPORTED = the "
    "source contradicts it, or it states information not present in the "
    "source. UNVERIFIABLE = the source is insufficient to judge.\n"
    "Judge only against the source; ignore outside knowledge. Output one line "
    "per claim, in order, for every claim, and nothing else."
)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# VERIFY_PANEL_MODELS = [LLAMA_SMALL, LLAMA_LARGE, GPT_OSS_SMALL, GPT_OSS_LARGE, QWEN]

from agents.openai_compatible import OpenAICompatibleAgent   # NEW

TOGETHER_PANEL = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3.5-9B",      
    "Qwen/Qwen3.7-Plus",
]

def build_judges(model_ids, system_prompt):
    judges = []
    for m in model_ids:
        try:
            judges.append(OpenAICompatibleAgent(
                model=m, provider="together", system_prompt=system_prompt))
        except Exception as exc:                                 # noqa: BLE001
            print(f"  [skip] judge {m}: {exc}")
    if len(judges) < 3:
        raise RuntimeError(f"Only {len(judges)} judges available; need >=3.")
    return judges


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")
_NUM_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.):\-]\s*(.+)$")          # NEW


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
    """Order matters: 'UNSUPPORTED' contains 'SUPPORTED' as a substring."""
    t = (text or "").strip().upper()
    if "UNVERIFIABLE" in t:
        return "UNVERIFIABLE"
    if "UNSUPPORTED" in t or "NOT SUPPORTED" in t or "NOT FULLY" in t:
        return "UNSUPPORTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    return "UNVERIFIABLE"


def parse_batch_verdicts(text: str, n_claims: int) -> Optional[List[str]]:   # NEW
    """Parse a judge's numbered reply into n_claims verdicts.

    Returns None if the reply cannot be ALIGNED to the claim list — i.e. any
    index is missing or out of range. Alignment failure must never be papered
    over: a silently shifted list would attach verdicts to the wrong claims,
    which is worse than an extra API call.
    """
    found: dict[int, str] = {}
    for line in (text or "").splitlines():
        m = _NUM_LINE_RE.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if 1 <= idx <= n_claims and idx not in found:
            found[idx] = parse_verdict(m.group(2))
    if len(found) != n_claims:
        return None
    return [found[i] for i in range(1, n_claims + 1)]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def decompose(decomposer, response_text: str, max_claims: int) -> Optional[List[str]]:
    prompt = ("Decompose the following response into atomic factual claims.\n\n"
              f"<response>\n{response_text}\n</response>")
    r = decomposer.query(prompt, temperature=0.0)
    if r.is_error:
        return None                      # CHANGED: None = call failed, [] = no claims
    return parse_claims(r.text, max_claims)


def verify_per_claim(judges, source_text: str, claims: List[str]) -> Optional[List[List[str]]]:
    """One call per (judge, claim). Returns [n_claims][n_judges] or None on error."""
    out = []
    for claim in claims:
        prompt = f"SOURCE:\n{source_text}\n\nCLAIM: {claim}\n\nVerdict:"
        responses = query_agents(judges, prompt, temperature=0.0)
        if any(r.is_error for r in responses):                          # CHANGED
            return None
        out.append([parse_verdict(r.text) for r in responses])
    return out


def verify_batched(judges, source_text: str, claims: List[str],          # NEW
                   stats: dict) -> Optional[List[List[str]]]:
    """One call per judge for ALL claims. Falls back to per-claim for a judge
    whose reply can't be aligned. Returns [n_claims][n_judges] or None on error."""
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    prompt = (f"SOURCE:\n{source_text}\n\nCLAIMS:\n{numbered}\n\n"
              f"Output exactly {len(claims)} lines, one verdict per claim.")

    per_judge: List[List[str]] = []
    for j in judges:
        r = j.query(prompt, temperature=0.0)
        if r.is_error:
            return None
        verdicts = parse_batch_verdicts(r.text, len(claims))
        if verdicts is None:
            # Alignment failed for this judge only — pay for per-claim calls
            # rather than guess. Counted so a high rate is visible.
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            verdicts = []
            for claim in claims:
                p = f"SOURCE:\n{source_text}\n\nCLAIM: {claim}\n\nVerdict:"
                rr = j.query(p, temperature=0.0)
                if rr.is_error:
                    return None
                verdicts.append(parse_verdict(rr.text))
        per_judge.append(verdicts)

    # transpose [n_judges][n_claims] -> [n_claims][n_judges]
    return [list(col) for col in zip(*per_judge)]


def claim_disagreement(verdicts: List[str]) -> float:
    if not verdicts:
        return 0.0
    modal = Counter(verdicts).most_common(1)[0][1]
    return 1.0 - modal / len(verdicts)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def auc_with_ci(y, x, n_boot=2000, seed=0):
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


def summarise(claim_rows):
    """Response-level signals from per-claim verdicts."""
    n = len(claim_rows)
    if n == 0:
        return dict(ensemble_unsupported=0.0, mean_disagreement=0.0,
                    max_disagreement=0.0, n_contested_claims=0,
                    frac_high_disagreement=0.0)
    dis = [c["disagreement"] for c in claim_rows]
    n_contested = sum(1 for d in dis if d > 0)
    return dict(
        ensemble_unsupported=sum(1 for c in claim_rows if c["majority"] != "SUPPORTED") / n,
        mean_disagreement=statistics.mean(dis),
        max_disagreement=max(dis),
        n_contested_claims=n_contested,
        frac_high_disagreement=n_contested / n,
    )


def build_claim_rows(claims, verdict_matrix, judges):
    rows = []
    for claim, verdicts in zip(claims, verdict_matrix):
        rows.append({
            "claim": claim,
            "verdicts": {j.name: v for j, v in zip(judges, verdicts)},
            "majority": Counter(verdicts).most_common(1)[0][0],
            "disagreement": claim_disagreement(verdicts),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RAGTruth verification pilot (v3).")
    ap.add_argument("--task", default="data2txt",
                    choices=["qa", "summarization", "data2txt"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decomposer", default=GPT_OSS_LARGE)
    ap.add_argument("--max-claims", type=int, default=12)
    ap.add_argument("--batch-claims", action="store_true",                   # NEW
                    help="one call per judge per response instead of per claim "
                         "(~12x fewer calls)")
    ap.add_argument("--compare-batching", type=int, default=0,               # NEW
                    help="run BOTH modes on the first K responses and report "
                         "verdict agreement, then exit")
    ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--diagnose-full", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="outputs/rgt_verify_results.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth {args.task.upper()} [test] — {args.n} responses (seed={args.seed})")
    samples = load_ragtruth(split="test", task_types=[args.task],
                            n=args.n, seed=args.seed, quality="good")
    print(f"  Loaded {len(samples)} candidate responses.")
    print(f"  hallucination base rate: "
          f"{statistics.mean(1.0 if s.is_hallucinated else 0.0 for s in samples):.1%}")

    section("Building agents")
    try:
        decomposer = make_agent(args.decomposer, system_prompt=DECOMPOSER_SYSTEM)
        sys_prompt = BATCH_VERIFIER_SYSTEM if args.batch_claims else VERIFIER_SYSTEM
        judges = build_judges(VERIFY_PANEL_MODELS, sys_prompt)
    except RuntimeError as exc:
        print(f"\n  {exc}\n")
        sys.exit(1)
    print(f"  decomposer : {decomposer.name}")
    for j in judges:
        print(f"  judge      : {j.name}")
    mode = "BATCHED" if args.batch_claims else "per-claim"
    per_resp = len(judges) + 1 if args.batch_claims else len(judges) * args.max_claims + 1
    print(f"  mode: {mode}  (~{per_resp} calls/response, "
          f"~{per_resp * len(samples):,} for this run if uncached)")

    # ------------------------------------------------------------------ #
    # NEW: batching validation
    # ------------------------------------------------------------------ #
    if args.compare_batching:
        section(f"Comparing batched vs per-claim verdicts on {args.compare_batching} responses")
        judges_pc = build_judges(VERIFY_PANEL_MODELS, VERIFIER_SYSTEM)
        judges_b = build_judges(VERIFY_PANEL_MODELS, BATCH_VERIFIER_SYSTEM)
        stats = {}
        agree = total = 0
        for s in samples[:args.compare_batching]:
            claims = decompose(decomposer, s.response, args.max_claims)
            if not claims:
                continue
            a = verify_per_claim(judges_pc, s.source_info_text, claims)
            b = verify_batched(judges_b, s.source_info_text, claims, stats)
            if a is None or b is None:
                print("  [stop] API error during comparison.")
                break
            for ra, rb in zip(a, b):
                for va, vb in zip(ra, rb):
                    total += 1
                    agree += (va == vb)
            sa, sb = summarise(build_claim_rows(claims, a, judges_pc)), \
                     summarise(build_claim_rows(claims, b, judges_b))
            print(f"  {s.uid}: ens_unsup {sa['ensemble_unsupported']:.2f} -> "
                  f"{sb['ensemble_unsupported']:.2f}   n_contested "
                  f"{sa['n_contested_claims']} -> {sb['n_contested_claims']}")
        if total:
            print(f"\n  verdict agreement: {agree}/{total} ({agree/total:.1%})")
            print(f"  alignment fallbacks: {stats.get('fallbacks', 0)}")
            print("  >=95% => batching is faithful; scale with --batch-claims.")
            print("  <90%  => batching changes the decision; keep per-claim.")
        return

    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    rows, diagnostics, stats = [], [], {}
    aborted = False
    for idx, s in enumerate(samples, start=1):
        claims = decompose(decomposer, s.response, args.max_claims)
        if claims is None:                                              # CHANGED
            print(f"  [{idx}/{len(samples)}] {s.uid}  DECOMPOSER ERROR "
                  f"(likely rate/quota limit) — stopping; cache preserved.")
            aborted = True
            break

        if claims:
            vm = (verify_batched(judges, s.source_info_text, claims, stats)
                  if args.batch_claims
                  else verify_per_claim(judges, s.source_info_text, claims))
            if vm is None:                                              # CHANGED
                print(f"  [{idx}/{len(samples)}] {s.uid}  JUDGE ERROR "
                      f"(likely rate/quota limit) — stopping; cache preserved.")
                aborted = True
                break
            claim_rows = build_claim_rows(claims, vm, judges)
        else:
            claim_rows = []

        sig = summarise(claim_rows)
        row = {"uid": s.uid, "source_model": s.source_model,
               "is_hallucinated": s.is_hallucinated,
               "n_spans": len(s.hallucination_spans),
               "n_claims": len(claim_rows), **sig, "claims": claim_rows}
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        if args.diagnose_full and claim_rows and sig["ensemble_unsupported"] == 1.0:
            spans = [{"text": sp.text, "label_type": sp.label_type,
                      "implicit_true": sp.implicit_true,
                      "due_to_null": sp.due_to_null}
                     for sp in s.hallucination_spans]
            diagnostics.append({"uid": s.uid, "gold_hallucinated": s.is_hallucinated,
                                "any_implicit_true": any(x["implicit_true"] for x in spans),
                                "any_due_to_null": any(x["due_to_null"] for x in spans),
                                "spans": spans})

        if idx <= args.peek:
            print(f"\n  --- peek {idx}: {s.uid} [{s.source_model}] "
                  f"gold_hallucinated={s.is_hallucinated} ---")
            for c in claim_rows:
                flag = "  <-- majority NOT supported" if c["majority"] != "SUPPORTED" else ""
                print(f"      [{c['majority']:<12} dis={c['disagreement']:.2f}] "
                      f"{c['claim'][:90]}{flag}")

        print(f"  [{idx:>3}/{len(samples)}] {s.uid}  claims={len(claim_rows):>2}  "
              f"ens_unsup={sig['ensemble_unsupported']:.2f}  "
              f"max_dis={sig['max_disagreement']:.2f}  "
              f"n_contested={sig['n_contested_claims']:>2}  "
              f"gold={'H' if s.is_hallucinated else '.'}")

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    section("Signals vs RAGTruth label (example-level, bootstrap 95% CI)")
    if aborted:
        print(f"  [!] RUN INCOMPLETE — {len(rows)}/{len(samples)} scored before an "
              f"API error. Re-run after the quota resets; cached work replays.")
    if args.batch_claims and stats.get("fallbacks"):
        print(f"  [!] {stats['fallbacks']} judge replies needed per-claim fallback "
              f"(batch parse failed).")

    scored = [r for r in rows if r["n_claims"] > 0]                     # CHANGED
    excluded = len(rows) - len(scored)
    if excluded:
        print(f"  [!] {excluded} responses yielded 0 claims — EXCLUDED from scoring.")
    y = np.array([1.0 if r["is_hallucinated"] else 0.0 for r in scored])
    print(f"  scored: {len(y)}   hallucinated: {int(y.sum())} "
          f"({y.mean():.1%} of scored)" if len(y) else "  nothing scored.")

    if len(y) >= 10 and y.min() != y.max():
        print(f"\n  {'signal':<26}{'AUC-ROC':>9}{'95% CI':>18}{'AUC-PR':>9}{'Spearman':>11}")
        for name in ("ensemble_unsupported", "max_disagreement",
                     "n_contested_claims", "mean_disagreement",
                     "frac_high_disagreement"):
            x = np.array([r[name] for r in scored], dtype=float)
            auc, lo, hi = auc_with_ci(y, x, n_boot=args.n_boot)
            print(f"  {name:<26}{auc:>9.3f}   [{lo:.3f}, {hi:.3f}]"
                  f"{average_precision_score(y, x):>9.3f}"
                  f"{spearmanr(x, y)[0]:>+11.3f}")
        print(f"\n  base rate (AUC-PR reference): {y.mean():.3f}")
        ens = np.array([r["ensemble_unsupported"] for r in scored])
        for name in ("max_disagreement", "mean_disagreement", "n_contested_claims"):
            md = np.array([r[name] for r in scored], dtype=float)
            rho, p = spearmanr(ens, md)
            print(f"  Spearman(ensemble_unsupported, {name}): rho={rho:+.3f} p={p:.3g}")
    else:
        print("  too few scored responses (or single-class) for AUCs.")

    if args.diagnose_full:
        section("Diagnostic — responses called FULLY unsupported")
        if not diagnostics:
            print("  none.")
        for d in diagnostics[:8]:
            tag = "CLEAN(FP)" if not d["gold_hallucinated"] else "hallu"
            print(f"    {d['uid']} [{tag}] implicit_true={d['any_implicit_true']} "
                  f"due_to_null={d['any_due_to_null']}")
            for sp in d["spans"][:3]:
                print(f"      - [{sp['label_type']}] {sp['text'][:80]}")

    section("Done")
    print(f"  Rows written to: {out_path}\n")


if __name__ == "__main__":
    main()