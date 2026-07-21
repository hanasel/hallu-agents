"""Near-free RAGTruth probe: is there ANY signal in inter-response disagreement?

This is the zero-API-cost first look at RAGTruth described in the Stage-3 plan.
It uses the six corpus responses that already ship with RAGTruth (GPT-4,
GPT-3.5, Llama-2-7/13/70B, Mistral-7B) AS the panel — no agent querying, no
keys, no cost — and asks whether lexical disagreement among those six
responses carries any information about which of them hallucinated.

It is deliberately the cheapest possible test. It answers one question before
we spend anything on the real cross-family panel or build claim-level
machinery: does disagreement predict hallucination in the RAG setting at all?

Two framings, matching the two levels RAGTruth is evaluated at:

  Per-response (the outlier hypothesis)
      For each response, dissent = mean pairwise Jaccard disagreement against
      the other five. Does a response that stands apart from its siblings tend
      to be the hallucinated one? -> response-level AUC-ROC / AUC-PR.

  Per-source (the context-difficulty hypothesis)
      For each source, mean pairwise disagreement over all six responses. Does
      a source on which the six models diverge tend to be one where more of
      them hallucinate? -> Spearman(mean disagreement, #hallucinated of 6).

Crucial caveat, measured not assumed
------------------------------------
The six corpus models differ wildly in verbosity and capability. Token-set
Jaccard will partly track *style/length*, not faithfulness, and the weakest
model may be a perpetual lexical outlier regardless of whether it hallucinated.
Section E quantifies exactly this confound (dissent-vs-length correlation;
per-model dissent vs per-model hallucination rate) so we don't over-read a
positive result. A weak or confounded signal here is itself the finding: it is
the empirical argument for the cross-family panel + semantic/claim-level
measures, rather than lexical disagreement over heterogeneous corpus models.

Run from the project root:
    python scripts/rgt_corpus_disagreement.py
    python scripts/rgt_corpus_disagreement.py --task qa --out outputs/rgt_corpus_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
from scipy.stats import spearmanr                                      # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score     # noqa: E402

from data import load_ragtruth_by_source                              # noqa: E402
# Reuse the EXACT tokenisation the project's Jaccard measure uses, so this
# probe and the real answer-level measure agree by construction.
from disagreement.answer_level import _tokenise                        # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def jaccard_disagreement(a: set, b: set) -> float:
    """1 - Jaccard similarity of two token sets. Both-empty -> agree (0.0)."""
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def main() -> None:
    ap = argparse.ArgumentParser(description="Near-free RAGTruth corpus-disagreement probe.")
    ap.add_argument("--task", default="qa",
                    choices=["qa", "summarization", "data2txt"],
                    help="RAGTruth task type to analyse (default: qa)")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--out", default="outputs/rgt_corpus_disagreement.jsonl")
    args = ap.parse_args()

    section(f"Loading RAGTruth {args.task.upper()} [{args.split}] grouped by source")
    grouped = load_ragtruth_by_source(split=args.split, task_types=[args.task])
    # Keep only complete sources (all six model responses present).
    grouped = {sid: rs for sid, rs in grouped.items() if len(rs) == 6}
    print(f"  {len(grouped)} sources with all 6 model responses "
          f"({len(grouped) * 6} responses).")

    # ------------------------------------------------------------------ #
    # Score every source and response
    # ------------------------------------------------------------------ #
    per_response: List[dict] = []   # one row per (source, model) response
    per_source: List[dict] = []     # one row per source
    rows_out: List[dict] = []

    for sid, responses in grouped.items():
        # Stable model order for reproducibility.
        responses = sorted(responses, key=lambda r: r.source_model)
        texts = [r.response for r in responses]
        toks = [_tokenise(t) for t in texts]
        labels = [r.is_hallucinated for r in responses]
        lengths = [len(t.split()) for t in texts]

        n = len(responses)
        # Pairwise Jaccard disagreement matrix.
        D = np.zeros((n, n))
        for i, j in combinations(range(n), 2):
            d = jaccard_disagreement(toks[i], toks[j])
            D[i, j] = D[j, i] = d

        # Per-response dissent = mean disagreement vs the other five.
        dissent = D.sum(axis=1) / (n - 1)
        # Per-source aggregate = mean over all unique pairs.
        source_disagreement = float(D[np.triu_indices(n, k=1)].mean())
        n_hallucinated = int(sum(labels))

        for idx, r in enumerate(responses):
            per_response.append({
                "source_id": sid,
                "model": r.source_model,
                "dissent": float(dissent[idx]),
                "is_hallucinated": bool(labels[idx]),
                "n_spans": len(r.hallucination_spans),
                "length_tokens": lengths[idx],
            })

        # Is the single most-dissenting response the hallucinated one?
        max_idx = int(np.argmax(dissent))
        per_source.append({
            "source_id": sid,
            "disagreement": source_disagreement,
            "n_hallucinated": n_hallucinated,
            "any_hallucinated": n_hallucinated > 0,
            "max_dissent_is_hallucinated": bool(labels[max_idx]),
        })

        rows_out.append({
            "source_id": sid,
            "question": responses[0].question,
            "disagreement": source_disagreement,
            "n_hallucinated": n_hallucinated,
            "per_model": {
                r.source_model: {
                    "dissent": float(dissent[i]),
                    "is_hallucinated": bool(labels[i]),
                    "length_tokens": lengths[i],
                } for i, r in enumerate(responses)
            },
        })

    # Persist per-source rows for downstream inspection.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    # A. Base rates
    # ------------------------------------------------------------------ #
    section("A. Base rates")
    resp_labels = np.array([r["is_hallucinated"] for r in per_response], dtype=float)
    n_resp = len(resp_labels)
    print(f"  responses                        : {n_resp}")
    print(f"  response-level hallucination rate: {resp_labels.mean():.1%}")
    src_any = np.array([s["any_hallucinated"] for s in per_source], dtype=float)
    print(f"  sources with >=1 hallucination   : {src_any.mean():.1%}")
    nh = np.array([s["n_hallucinated"] for s in per_source])
    print(f"  mean #hallucinated of 6 per source: {nh.mean():.2f}")

    # ------------------------------------------------------------------ #
    # B. Per-response: does dissent predict hallucination?
    # ------------------------------------------------------------------ #
    section("B. Per-response — does dissent predict hallucination?")
    dissent_arr = np.array([r["dissent"] for r in per_response])
    if resp_labels.min() == resp_labels.max():
        print("  (degenerate: all responses same label; skipping AUC)")
    else:
        auc = roc_auc_score(resp_labels, dissent_arr)
        ap_score = average_precision_score(resp_labels, dissent_arr)
        base = resp_labels.mean()
        print(f"  AUC-ROC (dissent -> hallucinated): {auc:.3f}   (0.5 = no signal)")
        print(f"  AUC-PR                           : {ap_score:.3f}   "
              f"(base rate {base:.3f})")
        mh = dissent_arr[resp_labels == 1].mean()
        mc = dissent_arr[resp_labels == 0].mean()
        print(f"  mean dissent | hallucinated      : {mh:.3f}")
        print(f"  mean dissent | clean             : {mc:.3f}")
        print(f"  separation (hallu - clean)       : {mh - mc:+.3f}")

    # ------------------------------------------------------------------ #
    # C. Per-source: does disagreement predict how many hallucinate?
    # ------------------------------------------------------------------ #
    section("C. Per-source — does disagreement predict the hallucination count?")
    dis = np.array([s["disagreement"] for s in per_source])
    rho, p = spearmanr(dis, nh)
    print(f"  Spearman(mean disagreement, #hallucinated): rho={rho:+.3f}  p={p:.3g}")
    rho2, p2 = spearmanr(dis, src_any)
    print(f"  Spearman(mean disagreement, any-hallucinated): rho={rho2:+.3f}  p={p2:.3g}")

    # ------------------------------------------------------------------ #
    # D. Outlier test
    # ------------------------------------------------------------------ #
    section("D. Outlier test — is the most-dissenting response the hallucinated one?")
    # Restrict to sources that have BOTH a hallucinated and a clean response,
    # otherwise the question is ill-posed (nothing to pick out).
    mixed = [s for s in per_source if 0 < s["n_hallucinated"] < 6]
    if mixed:
        hit = np.mean([s["max_dissent_is_hallucinated"] for s in mixed])
        # Chance = fraction of responses that are hallucinated within mixed sources.
        chance = np.mean([
            s["n_hallucinated"] / 6 for s in mixed
        ])
        print(f"  mixed sources (both clean & hallucinated present): {len(mixed)}")
        print(f"  max-dissent response is hallucinated : {hit:.1%}")
        print(f"  chance baseline (avg hallu fraction) : {chance:.1%}")
        print(f"  lift over chance                     : {hit - chance:+.1%}")
    else:
        print("  no mixed sources; outlier test not applicable")

    # ------------------------------------------------------------------ #
    # E. Confound diagnostics — is 'dissent' just length / the weak model?
    # ------------------------------------------------------------------ #
    section("E. Confounds — is dissent really about faithfulness?")
    lengths_arr = np.array([r["length_tokens"] for r in per_response])
    rho_len, p_len = spearmanr(dissent_arr, lengths_arr)
    print(f"  Spearman(dissent, response length): rho={rho_len:+.3f}  p={p_len:.3g}")
    print("  (large |rho| => dissent is largely a style/length artefact)\n")

    print("  Per-model: mean dissent vs hallucination rate")
    print(f"    {'model':<24}{'mean dissent':>14}{'hallu rate':>13}{'n':>6}")
    models = sorted({r["model"] for r in per_response})
    for m in models:
        rows_m = [r for r in per_response if r["model"] == m]
        md = np.mean([r["dissent"] for r in rows_m])
        hr = np.mean([r["is_hallucinated"] for r in rows_m])
        print(f"    {m:<24}{md:>14.3f}{hr:>12.1%}{len(rows_m):>6}")
    # Within-model AUC: the decisive control. If dissent predicts hallucination
    # ONLY because weak models both dissent more and hallucinate more, then
    # within a single model's own responses dissent should carry no signal
    # (AUC ~ 0.5). If it survives here, dissent tracks per-instance faithfulness,
    # not just which model produced the response.
    print("\n  Within-model AUC (controls for model identity):")
    print(f"    {'model':<24}{'AUC-ROC':>10}{'n_hallu':>10}")
    for m in models:
        rows_m = [r for r in per_response if r["model"] == m]
        y = np.array([r["is_hallucinated"] for r in rows_m], dtype=float)
        x = np.array([r["dissent"] for r in rows_m])
        if y.min() == y.max():
            print(f"    {m:<24}{'n/a':>10}{int(y.sum()):>10}")
        else:
            print(f"    {m:<24}{roc_auc_score(y, x):>10.3f}{int(y.sum()):>10}")
    print("    (AUC ~0.5 within every model => pooled signal is composition leakage)")

    print("\n  If the highest-dissent model is simply the highest-hallucination")
    print("  model, per-response AUC may be real signal. If the highest-dissent")
    print("  model is just the most verbose/idiosyncratic one (e.g. a small")
    print("  model) irrespective of its hallucination rate, the signal is a")
    print("  composition artefact — motivating semantic/claim-level measures.")

    section("Done")
    print(f"  Per-source rows written to: {out_path}\n")


if __name__ == "__main__":
    main()