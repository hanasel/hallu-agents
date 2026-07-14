"""Validate the LLM judge by blind hand-labelling a sample of its grades.

The LLM judge is now load-bearing for every AUROC, so we need to know how often
it agrees with a human. This tool samples graded answers from a *judged* pilot
file, shows you each (question, gold, answer) WITHOUT the judge's verdict, asks
you to grade it, then reveals the comparison and reports:
  - overall agreement rate (human vs judge)
  - a 3x3 confusion matrix (correct / incorrect / unclear)
  - Cohen's kappa (chance-corrected agreement)
  - agreement restricted to the shared-bias false-negative questions, which
    anchor the recall-ceiling claim (the judge MUST get these right).

Run on the k=1 judged file, where each stored answer maps 1:1 to a judge grade:
    python scripts/validate_judge.py --file outputs/pilot_results.judged.jsonl --n 24

Your labels are saved to outputs/judge_validation.json (resumable). Quit any
time with 'q' — progress is saved.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["correct", "incorrect", "unclear"]
G2S = {True: "correct", False: "incorrect", None: "unclear"}
IN2G = {"c": True, "i": False, "u": None}


def section(t: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {t}")
    print("=" * 70)


def load_incorrect_refs(uids) -> Dict[str, List[str]]:
    """Best-effort load of reference incorrect answers for richer context."""
    try:
        from data import load_truthfulqa
        return {s.uid: s.incorrect_answers for s in load_truthfulqa(n=None)}
    except Exception:
        return {}


def build_items(rows) -> List[dict]:
    """One item per (question, agent), tagging shared-bias-FN questions."""
    items = []
    for r in rows:
        sbfn = bool(r.get("panel_agrees") and r.get("panel_majority_wrong"))
        for name, answer in r["responses"].items():
            items.append({
                "uid": r["uid"], "category": r.get("category", "?"),
                "question": r["question"], "gold": r["correct_answer"],
                "agent": name, "answer": answer,
                "judge": r["grades"].get(name), "shared_bias": sbfn,
            })
    return items


def sample_items(items, n, seed) -> List[dict]:
    """Prioritise shared-bias-FN items, then fill with a random spread."""
    rng = random.Random(seed)
    sb = [it for it in items if it["shared_bias"]]
    rest = [it for it in items if not it["shared_bias"]]
    rng.shuffle(sb); rng.shuffle(rest)
    chosen = sb[:n] + rest[: max(0, n - len(sb[:n]))]
    rng.shuffle(chosen)
    return chosen[:n]


def cohens_kappa(pairs: List[Tuple[Optional[bool], Optional[bool]]]) -> Optional[float]:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    hc = Counter(G2S[a] for a, _ in pairs)
    jc = Counter(G2S[b] for _, b in pairs)
    pe = sum((hc[c] / n) * (jc[c] / n) for c in CLASSES)
    return None if pe == 1 else (po - pe) / (1 - pe)


def summarize(pairs: List[Tuple[Optional[bool], Optional[bool]]]) -> dict:
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    conf = {h: {j: 0 for j in CLASSES} for h in CLASSES}
    for h, j in pairs:
        conf[G2S[h]][G2S[j]] += 1
    return {"n": n, "agree": agree,
            "rate": agree / n if n else float("nan"),
            "kappa": cohens_kappa(pairs), "conf": conf}


def print_summary(s: dict, title: str) -> None:
    print(f"\n  {title}: {s['agree']}/{s['n']} agree = "
          f"{s['rate']:.0%}" + (f", kappa={s['kappa']:.2f}" if s['kappa'] is not None else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="outputs/pilot_results.judged.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="outputs/judge_validation.json")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"No such file: {path} (run regrade.py first)")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    items = build_items(rows)
    chosen = sample_items(items, args.n, args.seed)
    refs = load_incorrect_refs({it["uid"] for it in chosen})

    # Resume: reuse any already-saved human labels for these (uid, agent) keys.
    save_path = Path(args.save)
    prior = {}
    if save_path.exists():
        for rec in json.loads(save_path.read_text()):
            prior[(rec["uid"], rec["agent"])] = rec["human"]

    section(f"Blind judge validation — {len(chosen)} items from {path.name}")
    print("  For each answer, grade it vs the gold WITHOUT seeing the judge.")
    print("  Enter: [c]orrect  [i]ncorrect  [u]nclear  [s]kip  [q]uit\n")

    results = []
    for k, it in enumerate(chosen, 1):
        key = (it["uid"], it["agent"])
        if key in prior:
            human = prior[key]
        else:
            print("-" * 66)
            tag = "  [SHARED-BIAS FN]" if it["shared_bias"] else ""
            print(f"  ({k}/{len(chosen)}) {it['uid']} [{it['category']}]{tag}")
            print(f"    Q:    {it['question']}")
            print(f"    gold: {it['gold']}")
            inc = refs.get(it["uid"])
            if inc:
                print(f"    (labelled-wrong examples: {'; '.join(inc[:3])})")
            print(f"    answer [{it['agent'].split('/')[-1]}]: {str(it['answer']).strip()[:300]}")
            ans = ""
            while ans not in ("c", "i", "u", "s", "q"):
                ans = input("    your grade [c/i/u/s/q]: ").strip().lower()
            if ans == "q":
                print("  (quitting; progress saved)")
                break
            if ans == "s":
                continue
            human = IN2G[ans]
        results.append({**{kk: it[kk] for kk in ("uid", "agent", "shared_bias", "judge")},
                        "human": human})

    # Persist (merge with prior for anything not re-labelled).
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, indent=2))

    pairs = [(r["human"], r["judge"]) for r in results]
    if not pairs:
        print("\n  No labels recorded.")
        return

    section("Results — human vs LLM judge")
    s = summarize(pairs)
    print_summary(s, "Overall")
    print("\n  confusion (rows = you, cols = judge):")
    print(f"    {'':<12}" + "".join(f"{c:>12}" for c in CLASSES))
    for h in CLASSES:
        print(f"    {h:<12}" + "".join(f"{s['conf'][h][j]:>12}" for j in CLASSES))

    sb_pairs = [(r["human"], r["judge"]) for r in results if r["shared_bias"]]
    if sb_pairs:
        ss = summarize(sb_pairs)
        print_summary(ss, "Shared-bias FN subset (ceiling anchor)")
        wrong_both = sum(h is False and j is False for h, j in sb_pairs)
        print(f"    both you AND judge graded WRONG: {wrong_both}/{len(sb_pairs)} "
              f"(these must be wrong for the ceiling claim to hold)")

    # Flag disagreements for eyeballing.
    disagree = [r for r in results if r["human"] != r["judge"]]
    if disagree:
        section(f"Disagreements ({len(disagree)}) — inspect these")
        for r in disagree:
            print(f"    {r['uid']} [{r['agent'].split('/')[-1]}]: "
                  f"you={G2S[r['human']]}  judge={G2S[r['judge']]}")

    kappa = s["kappa"]
    verdict = ("strong" if kappa and kappa > 0.8 else
               "moderate" if kappa and kappa > 0.6 else "weak/uncertain")
    print(f"\n  Judge reliability: {s['rate']:.0%} agreement, kappa "
          f"{kappa:.2f} ({verdict})." if kappa is not None else "")
    print(f"  Saved -> {save_path}\n")


if __name__ == "__main__":
    main()