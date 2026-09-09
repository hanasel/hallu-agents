"""Scan results files for corrupted or degenerate agent responses.

Motivated by 16 rows found in outputs/truthfulqa_results.jsonl whose cached
completions were garbled --- e.g.

    '! \\'\\\\Stringestero｢ : ؛ assemb...'

Responses like this pass every existing guard: finish_reason is 'stop', the
text is non-empty, and no exception is raised. They are then clustered and
graded as if they were answers.

This script flags suspicious responses by several cheap signals and prints the
worst offenders for manual review. It does NOT modify anything --- the output
is a list to inspect, not a verdict.

    python scripts/scan_corrupt_responses.py outputs/*.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Characters expected in an English answer: basic Latin, common punctuation,
# curly quotes, dashes, and the accented Latin letters that appear in proper
# nouns (Hübschle, Erdoğan, François).
_OK_CHAR = re.compile(r"[A-Za-z0-9\s\.,;:!\?'\"\-–—()\[\]/%&$£€°#*_+=<>|~`^@\\]")
_WORD = re.compile(r"[A-Za-z']+")

# Script blocks that should not appear in an English answer at all.
_FOREIGN_SCRIPT = re.compile(
    r"[\u0590-\u05FF"      # Hebrew
    r"\u0600-\u06FF"       # Arabic
    r"\u0700-\u074F"       # Syriac
    r"\u0900-\u097F"       # Devanagari
    r"\u3000-\u303F"       # CJK punctuation  (｢ lives here)
    r"\u3040-\u30FF"       # Kana
    r"\u4E00-\u9FFF"       # CJK ideographs
    r"\uAC00-\uD7AF"       # Hangul
    r"]"
)

STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "was", "were", "are", "be", "been", "by", "with", "as", "that",
        "this", "it", "its", "from", "which", "what", "who", "when", "where"}


def content_words(s: str) -> set:
    return {w.lower() for w in _WORD.findall(s or "") if w.lower() not in STOP
            and len(w) > 2}


def odd_char_ratio(s: str) -> float:
    """Fraction of characters that are neither expected ASCII nor accented Latin."""
    if not s:
        return 0.0
    odd = 0
    for ch in s:
        if _OK_CHAR.match(ch):
            continue
        # Accented Latin (é, ü, ğ, ç ...) is fine in proper nouns.
        if "LATIN" in unicodedata.name(ch, ""):
            continue
        odd += 1
    return odd / len(s)


def longest_token_run(s: str) -> int:
    """Longest run of one repeated token — catches degenerate loops."""
    toks = (s or "").split()
    if not toks:
        return 0
    best = run = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, run)
    return best


def scan(path: Path):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    flagged = []
    total = 0

    for r in rows:
        q = r.get("question", "")
        qwords = content_words(q)
        for agent, text in (r.get("responses") or {}).items():
            total += 1
            t = (text or "").strip()
            if not t:
                continue

            reasons = []
            if _FOREIGN_SCRIPT.search(t):
                reasons.append("foreign-script")
            ocr = odd_char_ratio(t)
            if ocr > 0.05:
                reasons.append(f"odd-chars={ocr:.0%}")
            run = longest_token_run(t)
            if run >= 5:
                reasons.append(f"token-loop×{run}")

            # A factual answer normally echoes some of the question's content
            # words. Near-zero overlap on a long answer is suspicious; short
            # answers ("1986.", "Equilux") legitimately share nothing, so only
            # flag when the response is long enough for the signal to mean
            # something.
            tw = content_words(t)
            if qwords and len(t.split()) >= 8:
                overlap = len(qwords & tw) / len(qwords)
                if overlap == 0.0:
                    reasons.append("no-question-overlap")

            # Very high type/token ratio on a long string suggests word salad.
            toks = t.split()
            if len(toks) >= 12 and len(set(toks)) / len(toks) > 0.97:
                reasons.append("no-repeated-words")

            if reasons:
                flagged.append((len(reasons), r.get("uid"), agent, reasons, t))

    flagged.sort(key=lambda x: -x[0])
    print("=" * 78)
    print(f"  {path.name}: {len(flagged)} flagged of {total} responses "
          f"({len(flagged)/max(total,1):.2%})")
    print("=" * 78)

    by_agent = Counter(a.split("/")[-1] for _, _, a, _, _ in flagged)
    if by_agent:
        print("  flagged by agent:")
        for a, k in by_agent.most_common():
            print(f"    {a:<30} {k}")

    by_reason = Counter(x for _, _, _, rs, _ in flagged for x in
                        (r.split("=")[0].split("×")[0] for r in rs))
    if by_reason:
        print("  flagged by reason:")
        for rsn, k in by_reason.most_common():
            print(f"    {rsn:<30} {k}")

    print(f"\n  worst {min(25, len(flagged))} for manual review:\n")
    for n, uid, agent, reasons, t in flagged[:25]:
        print(f"  [{n}] {uid}  {agent.split('/')[-1]}  {', '.join(reasons)}")
        print(f"      {t[:160]!r}\n")
    return flagged


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print(__doc__)
        sys.exit(1)
    for p in paths:
        scan(p)