"""Optimality gaps of the solved policies: the fast-informed upper bound (FIB)
minus the exact in-model value of the solved policy, for every policy family,
in objective units, relative to the objective, and in deaths-per-100 000
equivalents (the objective is -E[CRC deaths] - lambda E[colonoscopies], so
1e-5 objective units = 1 death per 100 000 at fixed volume). Also the cap
600 -> 1500 sensitivity of both bounds at the same price.

python -m dp.gap_table
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import RES
from .sweep import policy_path

FAMILIES = {
    'headline (c6bhi, cap 1500, 4 rounds, 300 rollouts)': [
        ('c6bhi', 'death', l) for l in (0.001561, 0.00069, 0.000525)] + [
        ('c6bhi', 'inc', l) for l in (0.005125, 0.002264, 0.001724)],
    'observed class (c6bhiobs, cap 1500)': [('c6bhiobs', 'death', 0.001561), ('c6bhiobs', 'inc', 0.005125)],
    'lambda grid (c6b, cap 600, 6 rounds, 150 rollouts)': [
        ('c6b', 'death', l) for l in (0.0004, 0.000525, 0.00069, 0.000905, 0.001189, 0.001561, 0.00205, 0.002691,
                                      0.003534, 0.00464, 0.006093)] + [
        ('c6b', 'inc', l) for l in (0.001, 0.001313, 0.001724, 0.002264, 0.002972, 0.003903, 0.005125, 0.006729,
                                    0.008835, 0.011601, 0.015232)],
    'ablation (cap 600, 3 rounds)': [('ablpooled', 'death', 0.001561), ('ablnomem', 'death', 0.001561),
                                     ('ablc1', 'death', 0.001561)],
}


def meta_of(tag, obj, lam, sex):
    p = policy_path(obj, lam, sex, tag)
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return json.loads(str(z['meta']))


def main():
    rows = []
    for fam, items in FAMILIES.items():
        for tag, obj, lam in items:
            r = dict(family=fam, tag=tag, objective=obj, lam=lam)
            ok = True
            for sex in (1, 2):
                m = meta_of(tag, obj, lam, sex)
                if m is None:
                    ok = False
                    break
                ev = m['eval']
                r[f'obj_{sex}'] = m['objective']; r[f'ub_{sex}'] = m['fib_upper']; r[f'gap_{sex}'] = m['gap']
                r[f'rel_{sex}'] = m['gap'] / abs(m['objective'])
                r[f'death_{sex}'] = ev['death']; r[f'colos_{sex}'] = ev['colos']
                r[f'cap_{sex}'] = m.get('cap')
                r[f'rounds_run_{sex}'] = len(m.get('history', [])) - 1
                hist = m.get('history', [])
                if hist:
                    objs = [h['objective'] for h in hist]
                    r[f'flat_after_round_{sex}'] = int(np.argmax(objs))
                    r[f'improvement_after_round0_{sex}'] = max(objs) - objs[0]
            if not ok:
                continue
            r['gap_pooled'] = 0.5 * (r['gap_1'] + r['gap_2'])
            r['obj_pooled'] = 0.5 * (r['obj_1'] + r['obj_2'])
            r['rel_pooled'] = r['gap_pooled'] / abs(r['obj_pooled'])
            r['gap_deaths_per_100k'] = r['gap_pooled'] * 1e5
            rows.append(r)
    # cap sensitivity: c6b (600) vs c6bhi (1500) at the same lambda
    cap = []
    for obj, lam in (('death', 0.001561), ('death', 0.00069), ('death', 0.000525),
                     ('inc', 0.005125), ('inc', 0.002264), ('inc', 0.001724)):
        lo = [meta_of('c6b', obj, lam, s) for s in (1, 2)]; hi = [meta_of('c6bhi', obj, lam, s) for s in (1, 2)]
        if any(m is None for m in lo + hi):
            continue
        cap.append(dict(objective=obj, lam=lam,
                        lb_600=0.5 * (lo[0]['objective'] + lo[1]['objective']),
                        lb_1500=0.5 * (hi[0]['objective'] + hi[1]['objective']),
                        ub_600=0.5 * (lo[0]['fib_upper'] + lo[1]['fib_upper']),
                        ub_1500=0.5 * (hi[0]['fib_upper'] + hi[1]['fib_upper']),
                        death_600=0.5 * (lo[0]['eval']['death'] + lo[1]['eval']['death']),
                        death_1500=0.5 * (hi[0]['eval']['death'] + hi[1]['eval']['death']),
                        colos_600=0.5 * (lo[0]['eval']['colos'] + lo[1]['eval']['colos']),
                        colos_1500=0.5 * (hi[0]['eval']['colos'] + hi[1]['eval']['colos'])))
    out = dict(rows=rows, cap_sensitivity=cap)
    with open(os.path.join(RES, 'fib_gaps.json'), 'w') as f:
        json.dump(out, f, indent=1)
    L = ['# FIB optimality gaps of the solved policies', '',
         '| family | objective | lambda | in-model deaths /100k (pooled) | colos | lower bound (policy value) | FIB upper bound | gap | gap / |objective| | gap in deaths /100k | rounds run (m/f) | best round (m/f) |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['family']} | {r['objective']} | {r['lam']:g} | {0.5 * (r['death_1'] + r['death_2']) * 1e5:.0f} | "
                 f"{0.5 * (r['colos_1'] + r['colos_2']):.3f} | {r['obj_pooled']:.6f} | {0.5 * (r['ub_1'] + r['ub_2']):.6f} | "
                 f"{r['gap_pooled']:.5f} | {100 * r['rel_pooled']:.1f} % | {r['gap_deaths_per_100k']:.0f} | "
                 f"{r['rounds_run_1']}/{r['rounds_run_2']} | {r['flat_after_round_1']}/{r['flat_after_round_2']} |")
    L += ['', '## Cap sensitivity (600 vs 1500 belief points per key), sex-pooled', '',
          '| objective | lambda | lower bound 600 | lower bound 1500 | d LB | FIB 600 | FIB 1500 | d FIB | deaths /100k 600 -> 1500 | colos 600 -> 1500 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for c in cap:
        L.append(f"| {c['objective']} | {c['lam']:g} | {c['lb_600']:.6f} | {c['lb_1500']:.6f} | {c['lb_1500'] - c['lb_600']:+.2e} | "
                 f"{c['ub_600']:.6f} | {c['ub_1500']:.6f} | {c['ub_1500'] - c['ub_600']:+.2e} | "
                 f"{c['death_600'] * 1e5:.1f} -> {c['death_1500'] * 1e5:.1f} | {c['colos_600']:.3f} -> {c['colos_1500']:.3f} |")
    md = os.path.join(RES, 'fib_gaps.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', md)


if __name__ == '__main__':
    main()
