import json, statistics, collections, sys

rows = [json.loads(l) for l in open(sys.argv[1])]
CORE = [n for n in rows[0]['responses']]        # adjust if pool > core

def auroc(s, l):
    p = [x for x, y in zip(s, l) if y]; n = [x for x, y in zip(s, l) if not y]
    if min(len(p), len(n)) < 10: return None
    return sum((a > b) + 0.5 * (a == b) for a in p for b in n) / (len(p) * len(n))

data = []
for r in rows:
    p = r['panels']['core']
    if p['semantic_entropy'] is None: continue
    g = [v for v in (r.get('grades_judge') or {}).values() if v is not None]
    if not g: continue
    data.append((r, p['semantic_entropy'], p['jaccard'],
                 sum(x is False for x in g) > len(g) / 2))

for field in ('question_type', 'category'):
    b = collections.defaultdict(list)
    for r, s, j, l in data: b[r.get(field)].append((s, j, l))
    print(f'\n--- by {field} ---')
    for k in sorted(b, key=lambda k: -len(b[k]))[:12]:
        v = b[k]
        a = auroc([x[0] for x in v], [x[2] for x in v])
        aj = auroc([x[1] for x in v], [x[2] for x in v])
        print(f'{str(k):<28} n={len(v):<5} hall={sum(x[2] for x in v)/len(v):.0%} '
              f'D={statistics.mean(x[0] for x in v):.3f} '
              f'sem={a if a is None else round(a,3)} jac={aj if aj is None else round(aj,3)}')
# shared-bias examples for Section 6.3
print('\n--- shared-bias examples ---')
for r in rows:
    p = r['panels']['core']
    if not p.get('cluster_sizes'): continue
    sizes = p['cluster_sizes']; n = sum(sizes)
    if max(sizes) <= n / 2: continue
    cid = sizes.index(max(sizes))
    mem = [a for a, c in p['cluster_of'].items() if c == cid]
    g = [(r.get('grades_judge') or {}).get(a) for a in mem]
    g = [x for x in g if x is not None]
    if not g or any(x is True for x in g): continue
    if sum(x is False for x in g) <= len(g) / 2: continue
    print(f"{r['uid']} {len(mem)}/{n} sem={p['semantic_entropy']:.2f}")
    print(f"   Q: {r['question'][:100]}")
    print(f"   gold: {str(r.get('correct_answer'))[:70]}")
    print(f"   said: {(r['responses'][mem[0]] or '').strip()[:90]}")