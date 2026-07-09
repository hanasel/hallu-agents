"""A/B two (or more) NLI models over identical agent responses.

Agent responses are cached, so this changes ONLY the NLI backend — isolating
NLI quality from every other moving part. It answers: does a bigger MNLI model
fix the near-miss errors the base model made (e.g. merging "Waugh" with
"Wright", or splitting "computer program" from "language model")?

For each question it reports the cluster sizes + semantic entropy under each
model, flags questions whose clustering CHANGED, and for those prints the
pairwise NLI labels side by side so you can see which judgement flipped.

Runs in the pilot's default (relaxed) mode; pass --strict to compare under
strict clustering instead. Uses the same concise panel as the pilot.

    python scripts/ab_nli.py --n 5
    python scripts/ab_nli.py --n 20 --models cross-encoder/nli-deberta-v3-base \\
                                              cross-encoder/nli-deberta-v3-large
    python scripts/ab_nli.py --n 5 --strict

Requires: GROQ_API_KEY + torch + sentence-transformers. First use of each NLI
model downloads its weights (base ~0.4GB, large ~1.6GB).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import load_truthfulqa                                      # noqa: E402
from agents import query_agents                                       # noqa: E402
from agents.panels import cross_family_panel, CROSS_FAMILY            # noqa: E402
from disagreement import SemanticEntropyDisagreement, CrossEncoderNLI  # noqa: E402

SHORT_ANSWER_SYSTEM = (
    "Answer the question directly in a single short sentence. "
    "State your best factual answer plainly, with no preamble, no explanation, "
    "and no hedging."
)

DEFAULT_MODELS = [
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-large",
]


def section(t: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {t}")
    print("=" * 74)


def short(name: str) -> str:
    return name.split("/")[-1]


def pairwise_labels(nli, texts, question, agent_short) -> Dict[Tuple[str, str], str]:
    """Directional NLI label for every ordered pair, question-conditioned."""
    cond = [f"{question} {t}".strip() for t in texts]
    out: Dict[Tuple[str, str], str] = {}
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i == j:
                continue
            out[(agent_short[i], agent_short[j])] = nli.predict(cond[i], cond[j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B NLI models over cached responses.")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--strict", action="store_true",
                    help="compare under strict clustering (default: relaxed)")
    ap.add_argument("--single-linkage", action="store_true",
                    help="single-linkage clustering (default: complete-linkage)")
    args = ap.parse_args()

    section(f"Loading {args.n} questions + querying agents (cached)")
    samples = load_truthfulqa(n=args.n, seed=args.seed)
    agents = cross_family_panel(system_prompt=SHORT_ANSWER_SYSTEM)
    agent_short = [short(a.name) for a in agents]

    # Query once; reuse the identical responses for every NLI model.
    per_q_texts: List[List[str]] = []
    for s in samples:
        responses = query_agents(agents, s.prompt)
        per_q_texts.append([r.text for r in responses])
    print(f"  {len(samples)} questions, {len(agents)} agents "
          f"({', '.join(agent_short)})")
    linkage = "single" if args.single_linkage else "complete"
    print(f"  clustering: {'strict' if args.strict else 'relaxed'} + {linkage}-linkage")

    # For each model: compute clusters + pairwise labels for every question,
    # then free the model before loading the next (keeps memory bounded).
    results: Dict[str, dict] = {}
    for model_name in args.models:
        section(f"Scoring with {model_name}")
        nli = CrossEncoderNLI(model_name=model_name)
        measure = SemanticEntropyDisagreement(nli=nli, strict_entailment=args.strict,
                                              linkage=linkage)
        rows = []
        for s, texts in zip(samples, per_q_texts):
            r = measure.score(texts, question=s.question)
            labels = pairwise_labels(nli, texts, s.question, agent_short)
            rows.append({
                "uid": s.uid,
                "sizes": r.details["cluster_sizes"],
                "n_clusters": r.details["n_clusters"],
                "sem": r.score,
                "labels": labels,
            })
            print(f"  {s.uid}  sizes={r.details['cluster_sizes']} sem={r.score:.2f}")
        mean_sem = sum(x["sem"] for x in rows) / len(rows)
        results[model_name] = {"rows": rows, "mean_sem": mean_sem}
        del nli, measure
        gc.collect()

    # -- side-by-side comparison (first model = baseline) -------------------
    base_name = args.models[0]
    section("Comparison (baseline = " + short(base_name) + ")")
    header = f"  {'question':<20}" + "".join(f"{short(m):>28}" for m in args.models)
    print(header)
    changed_uids = []
    for i, s in enumerate(samples):
        cells = ""
        base_sizes = results[base_name]["rows"][i]["sizes"]
        differs = False
        for m in args.models:
            row = results[m]["rows"][i]
            cells += f"{str(row['sizes']) + ' ' + format(row['sem'], '.2f'):>28}"
            if row["sizes"] != base_sizes:
                differs = True
        flag = "  <- CHANGED" if differs else ""
        if differs:
            changed_uids.append(s.uid)
        print(f"  {s.uid:<20}{cells}{flag}")

    for m in args.models:
        print(f"  mean semantic entropy [{short(m):>26}] : {results[m]['mean_sem']:.3f}")

    # -- explain the changes: which pairwise judgement flipped --------------
    if changed_uids:
        section("Why they differ — pairwise label flips on changed questions")
        for i, s in enumerate(samples):
            if s.uid not in changed_uids:
                continue
            print(f"\n  {s.uid} [{s.category}]")
            print(f"    Q: {s.question[:84]}")
            for a_s, t in zip(agent_short, per_q_texts[i]):
                print(f"      {a_s:>22}: {t.strip()[:96]}")
            base_labels = results[base_name]["rows"][i]["labels"]
            for m in args.models[1:]:
                mlabels = results[m]["rows"][i]["labels"]
                flips = [(pair, base_labels[pair], mlabels[pair])
                         for pair in base_labels if base_labels[pair] != mlabels[pair]]
                print(f"    {short(base_name)} -> {short(m)} label flips:")
                if not flips:
                    print("      (no pairwise flips — cluster change came from a "
                          "representative-order effect)")
                for (p, hyp), old, new in flips:
                    print(f"      {p:>22} -> {hyp:<22} {old} => {new}")
    else:
        section("No clustering differences between the models on these questions")
        print("  Either the base model already handled them, or n is too small to "
              "hit the near-miss cases. Try a larger --n.")

    print()


if __name__ == "__main__":
    main()