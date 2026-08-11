"""RAGTruth response-conditioned verification pilot (framing "c", claim-level).

Decompose each labelled corpus response into atomic claims, have a diverse
judge panel rule each claim SUPPORTED/UNSUPPORTED/UNVERIFIABLE against the
source, and compare ensemble fact-checking against inter-judge disagreement.

v4 changes
----------
  - PROVIDER: judges AND decomposer now run on Together (OpenAI-compatible),
    a single provider / rate-limit bucket. Groq dependency removed.
  - PANEL: four confirmed-serverless judges (Llama-3.3-70B-Turbo, GPT-OSS-20B,
    GPT-OSS-120B, Qwen3.5-9B) spanning three families. Qwen3.7-Plus (streaming-
    only) and DeepSeek (V4-Pro frontier / V3 dedicated-only) were rejected.
  - EMPTY = DROP-VOTE. A judge that returns empty text (a non-answer, distinct
    from an API error) no longer casts a silent UNVERIFIABLE vote — its vote is
    dropped and the claim is judged on the remaining judges. Disagreement
    resolution therefore varies with the number of live votes on a claim.
    Per-judge empty rates are reported so a systematically-mute judge is
    visible rather than silently shrinking the panel.
  - RESUME. The output file is no longer truncated each run (unless --fresh).
    Completed uids are skipped and new rows appended, so a run interrupted by a
    quota limit resumes across windows. A run_config guard refuses to mix
    incomparable rows (different panel / mode / cap) into one file.

Inherited from v3: error propagation (API error stops the run, cache intact),
--batch-claims (validated at ~88% agreement -> NOT faithful; keep per-claim),
--compare-batching, 0-claim responses excluded from scoring.

Run:
    # canary
    python scripts/rgt_verify_pilot.py --task data2txt --n 3 --peek 3 \
        --max-claims 25 --out outputs/rgt_verify_together_data2txt.jsonl

    # real run (re-run the SAME command each quota window; it resumes)
    python scripts/rgt_verify_pilot.py --task data2txt --n 100 \
        --max-claims 25 --diagnose-full \
        --out outputs/rgt_verify_together_data2txt.jsonl
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
from agents.openai_compatible import OpenAICompatibleAgent           # noqa: E402


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


TOGETHER_PANEL = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


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
_NUM_LINE_RE = re.compile(r"^\s*(\d{1,3})\s*[.):\-]\s*(.+)$")


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
    """Map a judge reply to a verdict.

    Returns 'EMPTY' for a blank (non-)response so it can be distinguished from a
    genuine UNVERIFIABLE and dropped from the vote upstream. Order matters:
    'UNSUPPORTED' contains 'SUPPORTED' as a substring.
    """
    t = (text or "").strip().upper()
    if not t:
        return "EMPTY"
    if "UNVERIFIABLE" in t:
        return "UNVERIFIABLE"
    if "UNSUPPORTED" in t or "NOT SUPPORTED" in t or "NOT FULLY" in t:
        return "UNSUPPORTED"
    if "SUPPORTED" in t:
        return "SUPPORTED"
    # A non-empty reply we couldn't classify is a real UNVERIFIABLE (the judge
    # said *something*), not a dropped non-answer.
    return "UNVERIFIABLE"


def parse_batch_verdicts(text: str, n_claims: int) -> Optional[List[str]]:
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


def _effective(verdicts: List[str]) -> List[str]:
    """Drop EMPTY non-answers — a judge that didn't respond doesn't vote."""
    return [v for v in verdicts if v != "EMPTY"]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def decompose(decomposer, response_text: str, max_claims: int) -> Optional[List[str]]:
    prompt = ("Decompose the following response into atomic factual claims.\n\n"
              f"<response>\n{response_text}\n</response>")
    r = decomposer.query(prompt, temperature=0.0)
    if r.is_error:
        return None                      # None = API error (retry); [] = no claims
    return parse_claims(r.text, max_claims)


def verify_per_claim(judges, source_text: str, claims: List[str]) -> Optional[List[List[str]]]:
    """One call per (judge, claim). Returns [n_claims][n_judges] verdict strings
    (which may include 'EMPTY'), or None if any call is an API ERROR (distinct
    from an empty reply — an API error means quota/rate limit and stops the run
    so it can resume, whereas an empty reply just drops that judge's vote)."""
    out = []
    for claim in claims:
        prompt = f"SOURCE:\n{source_text}\n\nCLAIM: {claim}\n\nVerdict:"
        responses = query_agents(judges, prompt, temperature=0.0)
        if any(r.is_error for r in responses):
            return None
        out.append([parse_verdict(r.text) for r in responses])
    return out


def verify_batched(judges, source_text: str, claims: List[str],
                   stats: dict) -> Optional[List[List[str]]]:
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
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            verdicts = []
            for claim in claims:
                p = f"SOURCE:\n{source_text}\n\nCLAIM: {claim}\n\nVerdict:"
                rr = j.query(p, temperature=0.0)
                if rr.is_error:
                    return None
                verdicts.append(parse_verdict(rr.text))
        per_judge.append(verdicts)

    return [list(col) for col in zip(*per_judge)]


def claim_disagreement(verdicts: List[str]) -> float:
    """1 - modal-verdict fraction over the LIVE votes passed in."""
    if not verdicts:
        return 0.0
    modal = Counter(verdicts).most_common(1)[0][1]
    return 1.0 - modal / len(verdicts)


def build_claim_rows(claims, verdict_matrix, judges):
    """Per-claim row. `verdicts` keeps the full per-judge map (incl. EMPTY) for
    transparency; majority and disagreement are computed over LIVE votes only."""
    rows = []
    for claim, verdicts in zip(claims, verdict_matrix):
        eff = _effective(verdicts)
        if eff:
            majority = Counter(eff).most_common(1)[0][0]
            disagreement = claim_disagreement(eff)
        else:
            majority = "EMPTY"           # every judge muted on this claim
            disagreement = 0.0
        rows.append({
            "claim": claim,
            "verdicts": {j.name: v for j, v in zip(judges, verdicts)},
            "n_votes": len(eff),
            "majority": majority,
            "disagreement": disagreement,
        })
    return rows


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
    """Response-level signals. Claims where EVERY judge was empty (majority ==
    'EMPTY') carry no information and are excluded from the response's score."""
    usable = [c for c in claim_rows if c["majority"] != "EMPTY"]
    n = len(usable)
    n_empty_claims = len(claim_rows) - n
    if n == 0:
        return dict(ensemble_unsupported=0.0, mean_disagreement=0.0,
                    max_disagreement=0.0, n_contested_claims=0,
                    frac_high_disagreement=0.0, n_empty_claims=n_empty_claims)
    dis = [c["disagreement"] for c in usable]
    n_contested = sum(1 for d in dis if d > 0)
    return dict(
        ensemble_unsupported=sum(1 for c in usable if c["majority"] != "SUPPORTED") / n,
        mean_disagreement=statistics.mean(dis),
        max_disagreement=max(dis),
        n_contested_claims=n_contested,
        frac_high_disagreement=n_contested / n,
        n_empty_claims=n_empty_claims,
    )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_completed(out_path: Path, config: dict):
    """Load prior rows so a quota-interrupted run resumes. Refuses to mix rows
    from an incomparable config (different panel / mode / cap) into one file."""
    if not out_path.exists():
        return [], set()
    rows, bad = [], 0
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"  [!] skipped {bad} unparseable line(s) (interrupted write).")
    if rows:
        stored = rows[0].get("run_config", {})
        mismatched = {k: (stored.get(k), v) for k, v in config.items()
                      if stored.get(k) != v}
        if mismatched:
            print("  [!] CONFIG MISMATCH vs existing file — results would be mixed:")
            for k, (old, new) in mismatched.items():
                print(f"        {k}: file={old!r}  now={new!r}")
            print("      Use --fresh, or point --out at a new file.")
            sys.exit(3)
    return rows, {r["uid"] for r in rows}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="RAGTruth verification pilot (v4).")
    ap.add_argument("--task", default="data2txt",
                    choices=["qa", "summarization", "data2txt"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decomposer", default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    help="Together model id used for claim decomposition")
    ap.add_argument("--max-claims", type=int, default=25)
    ap.add_argument("--batch-claims", action="store_true",
                    help="one call per judge per response (validated ~88%% "
                         "agreement — NOT faithful; kept for reference only)")
    ap.add_argument("--compare-batching", type=int, default=0)
    ap.add_argument("--peek", type=int, default=0)
    ap.add_argument("--diagnose-full", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fresh", action="store_true",
                    help="truncate the output file instead of resuming it")
    ap.add_argument("--out", default="outputs/rgt_verify_together.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth {args.task.upper()} [test] — {args.n} responses (seed={args.seed})")
    samples = load_ragtruth(split="test", task_types=[args.task],
                            n=args.n, seed=args.seed, quality="good")
    print(f"  Loaded {len(samples)} candidate responses.")
    print(f"  hallucination base rate: "
          f"{statistics.mean(1.0 if s.is_hallucinated else 0.0 for s in samples):.1%}")

    section("Building agents")
    try:
        decomposer = OpenAICompatibleAgent(
            model=args.decomposer, provider="together",
            system_prompt=DECOMPOSER_SYSTEM)
        sys_prompt = BATCH_VERIFIER_SYSTEM if args.batch_claims else VERIFIER_SYSTEM
        judges = build_judges(TOGETHER_PANEL, sys_prompt)
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
    # Batching validation
    # ------------------------------------------------------------------ #
    if args.compare_batching:
        section(f"Comparing batched vs per-claim verdicts on {args.compare_batching} responses")
        judges_pc = build_judges(TOGETHER_PANEL, VERIFIER_SYSTEM)
        judges_b = build_judges(TOGETHER_PANEL, BATCH_VERIFIER_SYSTEM)
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
            sa = summarise(build_claim_rows(claims, a, judges_pc))
            sb = summarise(build_claim_rows(claims, b, judges_b))
            print(f"  {s.uid}: ens_unsup {sa['ensemble_unsupported']:.2f} -> "
                  f"{sb['ensemble_unsupported']:.2f}   n_contested "
                  f"{sa['n_contested_claims']} -> {sb['n_contested_claims']}")
        if total:
            print(f"\n  verdict agreement: {agree}/{total} ({agree/total:.1%})")
            print(f"  alignment fallbacks: {stats.get('fallbacks', 0)}")
            print("  >=95% => batching is faithful; <90% => keep per-claim.")
        return

    # ------------------------------------------------------------------ #
    # Scoring (resumable)
    # ------------------------------------------------------------------ #
    section("Scoring")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = {
        "task": args.task,
        "seed": args.seed,
        "decomposer": args.decomposer,
        "max_claims": args.max_claims,
        "batch_claims": bool(args.batch_claims),
        "judges": [j.name for j in judges],
    }
    if args.fresh:
        out_path.write_text("", encoding="utf-8")
        rows, done = [], set()
    else:
        rows, done = load_completed(out_path, run_config)
        if done:
            print(f"  Resuming: {len(done)} already scored; "
                  f"{len(samples) - len(done)} remaining.")

    diagnostics, stats = [], {}
    aborted = False
    for idx, s in enumerate(samples, start=1):
        if s.uid in done:
            continue

        claims = decompose(decomposer, s.response, args.max_claims)
        if claims is None:
            print(f"  [{idx}/{len(samples)}] {s.uid}  DECOMPOSER ERROR "
                  f"(likely rate/quota limit) — stopping; cache preserved.")
            aborted = True
            break

        if claims:
            vm = (verify_batched(judges, s.source_info_text, claims, stats)
                  if args.batch_claims
                  else verify_per_claim(judges, s.source_info_text, claims))
            if vm is None:
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
               "n_claims": len(claim_rows), **sig,
               "run_config": run_config, "claims": claim_rows}
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
                flag = "  <-- majority NOT supported" if c["majority"] not in ("SUPPORTED",) else ""
                votes = f"({c['n_votes']}v)" if c["n_votes"] < len(judges) else ""
                print(f"      [{c['majority']:<12} dis={c['disagreement']:.2f}]{votes} "
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
        print(f"  [!] RUN INCOMPLETE — {len(rows)}/{len(samples)} scored so far. "
              f"Re-run the same command after quota resets; it resumes.")
    if args.batch_claims and stats.get("fallbacks"):
        print(f"  [!] {stats['fallbacks']} judge replies needed per-claim fallback.")

    # Per-judge empty-vote diagnostics (drop-vote visibility).
    empty_by_judge, total_by_judge = Counter(), Counter()
    for r in rows:
        for c in r.get("claims", []):
            for jname, v in c["verdicts"].items():
                total_by_judge[jname] += 1
                if v == "EMPTY":
                    empty_by_judge[jname] += 1
    if sum(empty_by_judge.values()):
        print("  empty (dropped) votes per judge:")
        for jname in sorted(total_by_judge):
            e, t = empty_by_judge[jname], total_by_judge[jname]
            warn = "  <-- high; panel effectively shrinks" if t and e / t > 0.1 else ""
            print(f"    {jname:<48} {e:>5}/{t:<6} ({(e/t if t else 0):.1%}){warn}")

    scored = [r for r in rows if r["n_claims"] > 0]
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