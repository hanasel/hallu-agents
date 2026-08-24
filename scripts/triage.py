"""Triage the SimpleQA pilot results file — before trusting any of its numbers.

Reads outputs/simpleqa_pilot_results.jsonl and answers four questions the
pilot's own summary doesn't:

  1. WHO is producing the empty responses, and with what finish_reason?
  2. Do the headline numbers survive if you drop every row containing an
     empty response?
  3. Are the Part B findings (tie-broken / rescued / shared-bias FN) built
     on rows where the "dissenting" agent simply returned nothing?
  4. Do the two measures actually *rank* questions differently, and does
     either predict incorrectness (AUROC)? Comparing their raw means is
     apples-to-oranges; ranking and AUROC are what the RQ needs.

No dependencies beyond the standard library.

    python triage_simpleqa_pilot.py outputs/simpleqa_pilot_results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from statistics import mean


def load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def h(title):
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


def is_empty(text):
    return not (text or "").strip()


# --------------------------------------------------------------------------
# Rank / AUROC helpers (no sklearn, so this runs anywhere)
# --------------------------------------------------------------------------

def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties counted as 0.5."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/simpleqa_pilot_results.jsonl"
    rows = load(path)
    agent_names = list(rows[0]["responses"].keys())
    print(f"Loaded {len(rows)} rows, {len(agent_names)} agents from {path}")

    # -- 1. Empty census ---------------------------------------------------
    h("1. Empty responses — who, and why")
    per_agent = Counter()
    finish = defaultdict(Counter)
    for r in rows:
        for name, text in r["responses"].items():
            if is_empty(text):
                per_agent[name] += 1
                finish[name][r.get("finish_reasons", {}).get(name)] += 1
    if not per_agent:
        print("  None. Good.")
    for name in agent_names:
        n = per_agent[name]
        bar = "#" * n
        fr = ", ".join(f"{k}={v}" for k, v in finish[name].items()) or "-"
        print(f"  {name.split('/')[-1]:<34} {n:>3}/{len(rows)}  {bar}")
        if n:
            print(f"  {'':<34}     finish_reason: {fr}")
    print("\n  finish_reason='length' => truncated: raise max_tokens (note this")
    print("  changes the cache key, so it re-bills every question).")
    print("  finish_reason='stop' with empty text => the model emitted only")
    print("  reasoning tokens, or a content filter fired.")

    # -- 2. Headline stats on the clean subset ----------------------------
    h("2. Headline numbers, all rows vs empty-free rows")
    clean = [r for r in rows if r.get("empty_count", 0) == 0]
    print(f"  rows with no empty response: {len(clean)} / {len(rows)}")
    if not clean:
        print("  Nothing survives. Fix the token budget and re-run.")
        return

    def block(label, rs):
        cc = Counter(r["n_clusters"] for r in rs)
        print(f"  {label:<12} n={len(rs):<4} "
              f"mean_jac={mean(r['jaccard'] for r in rs):.3f}  "
              f"mean_sem={mean(r['semantic_entropy'] for r in rs):.3f}  "
              f"clusters=" + ",".join(f"{k}->{cc[k]}" for k in sorted(cc)))
        g = Counter()
        for r in rs:
            for v in r["grades"].values():
                g[{True: "correct", False: "incorrect", None: "unclear"}[v]] += 1
        tot = sum(g.values()) or 1
        print(f"  {'':<12} grades: " + ", ".join(
            f"{k}={g[k]} ({g[k]/tot:.0%})" for k in ("correct", "incorrect", "unclear")))

    block("ALL", rows)
    block("EMPTY-FREE", clean)
    print("\n  If mean_sem drops and the cluster spread shifts left, the")
    print("  disagreement signal was partly measuring truncation, not knowledge.")

    # -- 3. Are the Part B findings artifacts? -----------------------------
    h("3. Part B rows — is the 'dissent' just an empty answer?")

    def empties_in(r):
        return [n for n, t in r["responses"].items() if is_empty(t)]

    tie = [r for r in rows if r.get("n_clusters_same_family") == 1 and r["n_clusters"] > 1]
    rescued = [r for r in tie if r.get("both_llamas_wrong")]
    fn = [r for r in rows if r.get("panel_agrees") and r.get("panel_majority_wrong")]

    for label, subset in (("tie-broken", tie), ("rescued", rescued),
                          ("shared-bias FN", fn)):
        dirty = [r for r in subset if empties_in(r)]
        print(f"  {label:<16} {len(subset):>3} rows, "
              f"{len(dirty)} contain >=1 empty response "
              f"({len(subset) - len(dirty)} clean)")
        for r in dirty[:4]:
            print(f"      {r['uid']}: empty from {[n.split('/')[-1] for n in empties_in(r)]}")

    print("\n  Any 'rescued' row whose dissenter was empty is an artifact, not")
    print("  a cross-family model catching a shared Llama bias.")

    # -- 4. Do the measures rank differently? Do they predict error? -------
    h("4. Ranking agreement and detection performance")
    jac = [r["jaccard"] for r in clean]
    sem = [r["semantic_entropy"] for r in clean]
    print(f"  Spearman(jaccard, semantic_entropy) on empty-free rows: "
          f"{spearman(jac, sem):+.3f}")
    print("  Near +1.0 means semantic entropy is re-ranking nothing on this")
    print("  dataset — expected for short-form answers, and a finding in itself.")

    print("\n  AUROC — does disagreement predict that the answer is wrong?")
    label_maj = [bool(r.get("panel_majority_wrong")) for r in clean]
    for name, scores in (("jaccard", jac), ("semantic_entropy", sem)):
        a = auroc(scores, label_maj)
        print(f"    {name:<20} vs panel-majority-wrong : "
              f"{a:.3f}" if a is not None else f"    {name:<20} : n/a (one class only)")

    print("\n  Per-agent (label = that agent graded incorrect; 'unclear' dropped):")
    for agent in agent_names:
        pairs = [(r, r["grades"].get(agent)) for r in clean]
        pairs = [(r, g) for r, g in pairs if g is not None]
        if len(pairs) < 10:
            print(f"    {agent.split('/')[-1]:<34} too few graded rows ({len(pairs)})")
            continue
        labs = [g is False for _, g in pairs]
        a_j = auroc([r["jaccard"] for r, _ in pairs], labs)
        a_s = auroc([r["semantic_entropy"] for r, _ in pairs], labs)
        base = sum(labs) / len(labs)
        fmt = lambda v: f"{v:.3f}" if v is not None else " n/a "
        print(f"    {agent.split('/')[-1]:<34} n={len(pairs):<3} "
              f"err_rate={base:.0%}  AUROC jac={fmt(a_j)} sem={fmt(a_s)}")

    print("\n  AUROC ~0.50 means the measure carries no signal for that agent.")
    print("  This is the number the thesis needs, not the mean disagreement.\n")


if __name__ == "__main__":
    main()