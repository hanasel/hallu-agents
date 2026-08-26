# check_grader_fix.py
import json, re, sys
sys.path.insert(0, '.')
from evaluation.grading import _extract_mc_letter as new_extract
from data import load_truthfulqa

OLD_LABEL = re.compile(
    r'\banswer\s*(?:is)?\s*:?\s*[\s"\'*(\[]*([A-Za-z])(?![A-Za-z])', re.IGNORECASE)
OLD_BARE = re.compile(r'^[\s"\'*(\[]*([A-Za-z])[\s"\')\]*.,:;!]*$')

def old_extract(t):
    t = (t or '').strip()
    if not t:
        return None
    m = OLD_LABEL.search(t) or OLD_BARE.match(t)
    return m.group(1).upper() if m else None

gold = {s.uid: s.correct_letter for s in load_truthfulqa(n=790, seed=0)}
diff_letter = diff_grade = total = 0
for line in open('outputs/truthfulqa_results_mc.jsonl'):
    r = json.loads(line)
    g = gold.get(r['uid'])
    for n, t in r['responses'].items():
        total += 1
        o, nw = old_extract(t), new_extract(t)
        if o == nw:
            continue
        diff_letter += 1
        og = None if o is None else (o == g)
        ng = None if nw is None else (nw == g)
        if og != ng:
            diff_grade += 1
            print(f"{r['uid']} {n.split('/')[-1]:<22} {o}->{nw} gold={g} grade {og}->{ng}")

print(f"\n{diff_letter} extractions changed, {diff_grade} grades changed, "
      f"out of {total} responses")