"""Lambda sweep: solve the dp policy for a grid of colonoscopy shadow prices
(per objective), record the exact in-model frontier, and save every policy.

python -m dp.sweep --kernels results/dp/kernels_c3.npz --objective death --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .common import RES
from .model import METRICS

POL_DIR = os.path.join(RES, 'policies')
os.makedirs(POL_DIR, exist_ok=True)

OBJECTIVES = {
    'death': dict(death=1.0),
    'inc': dict(inc=1.0),
    'ly': dict(ly=1.0),
    'combo': dict(death=1.0, inc=0.25),     # one CRC death ~ four diagnoses
}
DEFAULT_GRIDS = {
    'death': list(np.round(np.geomspace(4e-4, 8e-3, 12), 6)),
    'inc': list(np.round(np.geomspace(1e-3, 2e-2, 12), 6)),
    'ly': list(np.round(np.geomspace(6e-3, 1.2e-1, 12), 5)),
    'combo': list(np.round(np.geomspace(5e-4, 1e-2, 12), 6)),
}


def policy_path(objective, lam, sex, tag):
    return os.path.join(POL_DIR, f'{tag}_{objective}_lam{lam:.6g}_sex{sex}.npz')


def _solve_one(job):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from .solver import solve_policy
    objective, lam, sex, kernels, tag, cap, rounds, rollouts = (
        job['objective'], job['lam'], job['sex'], job['kernels'], job['tag'], job['cap'], job['rounds'], job['rollouts'])
    out = policy_path(objective, lam, sex, tag)
    if os.path.exists(out) and not job.get('force'):
        from .solver import load_policy
        p = load_policy(out)
        return dict(obj=objective, lam=lam, sex=sex, path=out, **p.meta['eval'], gap=p.meta.get('gap'), cached=True)
    t0 = time.time()
    pol = solve_policy(sex, kernels, weights=OBJECTIVES[objective], lam=lam, cap=cap, rounds=rounds,
                       rollouts=rollouts, verbose=False, class_known_roots=job.get('class_known_roots', False),
                       root_priors=job.get('root_priors'),
                       root_beliefs=job.get('root_beliefs'), root_weights=job.get('root_weights'))
    pol.meta['solve_sec'] = time.time() - t0
    pol.save(out)
    return dict(obj=objective, lam=lam, sex=sex, path=out, **pol.meta['eval'], gap=pol.meta['gap'],
                solve_sec=pol.meta['solve_sec'])


def run_sweep(kernels, objective, lams=None, tag='c3', workers=4, cap=600, rounds=6, rollouts=150,
              class_known_roots=False, force=False, root_priors=None,
              root_beliefs=None, root_weights=None):
    lams = lams if lams is not None else DEFAULT_GRIDS[objective]
    jobs = [dict(objective=objective, lam=float(l), sex=s, kernels=kernels, tag=tag, cap=cap, rounds=rounds,
                 rollouts=rollouts, class_known_roots=class_known_roots, force=force,
                 root_priors=root_priors, root_beliefs=root_beliefs, root_weights=root_weights)
            for l in lams for s in (1, 2)]
    res = []
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            res.append(_solve_one(j)); print(f'  {res[-1]["obj"]} lam={res[-1]["lam"]:.6g} sex={res[-1]["sex"]} colos={res[-1]["colos"]:.3f} ({time.time() - t0:.0f}s)', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_solve_one, j) for j in jobs]
            for f in as_completed(futs):
                r = f.result(); res.append(r)
                print(f'  {r["obj"]} lam={r["lam"]:.6g} sex={r["sex"]} colos={r["colos"]:.3f} death={r["death"] * 1e5:.0f} '
                      f'inc={r["inc"] * 1e5:.0f} ({time.time() - t0:.0f}s)', flush=True)
    res.sort(key=lambda r: (r['lam'], r['sex']))
    if not tag.endswith('cal'):
        out = os.path.join(RES, f'sweep_{tag}_{objective}.json')
        with open(out, 'w') as f:
            json.dump(dict(kernels=kernels, objective=objective, rows=res), f, indent=1)
        print('saved', out)
    return res


def pooled_rows(res, frac_female):
    """Combine per-sex in-model results into population rows per lambda."""
    by = {}
    for r in res:
        by.setdefault(r['lam'], {})[r['sex']] = r
    rows = []
    for lam, d in sorted(by.items()):
        if 1 in d and 2 in d:
            row = dict(lam=lam)
            for k in list(METRICS) + ['colos', 'objective']:
                row[k] = (1 - frac_female) * d[1][k] + frac_female * d[2][k]
            row['path_male'] = d[1]['path']; row['path_female'] = d[2]['path']
            rows.append(row)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--objective', default='death', choices=list(OBJECTIVES))
    ap.add_argument('--lams', type=float, nargs='*', default=None)
    ap.add_argument('--tag', default='c3')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cap', type=int, default=600)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--rollouts', type=int, default=150)
    ap.add_argument('--class-known-roots', action='store_true')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    run_sweep(a.kernels, a.objective, a.lams, tag=a.tag, workers=a.workers, cap=a.cap, rounds=a.rounds,
              rollouts=a.rollouts, class_known_roots=a.class_known_roots, force=a.force)
