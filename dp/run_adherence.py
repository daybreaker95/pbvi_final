"""Adherence scenarios: each due invitation is attended with probability
alpha (per-person independent draws from a hook-local rng).

  q10y_a{a}         fixed 10-y, missed slot LOST (classic programme)
  q10y_recall_a{a}  fixed 10-y with annual re-invitation until attended
  dp_a{a}           DP policy lam=0.001561 (same policy as the headline;
                    a no-show yields no observation and the policy simply
                    re-plans -- no re-solving for the adherence level)

python -m dp.run_adherence --n 200000 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os

from .common import RES
from .engine_runner import run_arm, aggregate, efficiency, paired_diff
from .evaluate import summary_table

POL_M = os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex1.npz')
POL_F = os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex2.npz')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--alphas', type=float, nargs='*', default=[0.7, 0.5, 0.3])
    a = ap.parse_args()
    arms = {'none': {'kind': 'none'},
            'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            'dp_death_lam0.001561_q10y': {'kind': 'policy', 'policy_male': POL_M, 'policy_female': POL_F,
                                          'observed_class': False}}
    for al in a.alphas:
        t = f'{al:g}'.replace('.', '')
        arms[f'q10y_a{t}'] = {'kind': 'fixed', 'ages': [50, 60, 70], 'adherence': al}
        arms[f'q10y_recall_a{t}'] = {'kind': 'fixed', 'ages': [50, 60, 70], 'adherence': al, 'recall': True}
        arms[f'dp_a{t}'] = {'kind': 'policy', 'policy_male': POL_M, 'policy_female': POL_F,
                            'observed_class': False, 'adherence': al}
    res, paths = {}, {}
    for tag, arm in arms.items():
        print(f'== arm {tag}', flush=True)
        paths[tag] = run_arm(arm, tag, a.n, chunk=50_000, workers=a.workers)
    base = aggregate(paths['none'])
    for tag in arms:
        r = aggregate(paths[tag])
        if tag != 'none':
            r.update(efficiency(r, base))
            d, se = paired_diff(paths[tag], paths['none'], 'crc_death')
            r['paired_death_diff_vs_none'] = d; r['paired_death_diff_se'] = se
        res[tag] = r
    print('\n=== adherence scenarios (n={:,}) ===\n'.format(a.n) + summary_table(res), flush=True)
    out = os.path.join(RES, f'eval_adherence_c6b_n{a.n}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
