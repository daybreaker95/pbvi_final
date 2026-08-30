"""End-to-end driver (every step is cached / resumable):

  1. kernels   : estimate all kernels from the two cohorts        -> results/dp/kernels_<tag>.npz
  2. fixed     : exhaustive in-model fixed-schedule search         -> results/dp/fixed_search_<tag>.json
  3. sweep     : lambda sweeps per objective (policies + frontier) -> results/dp/sweep_<tag>_<obj>.json
  4. baseline  : engine arms none / q10y / q5y at N_HEAD           -> results/dp/runs/...
  5. grid      : engine evaluation of every sweep policy at N_GRID
  6. headline  : engine evaluation at N_HEAD of policies matched to q10y / q5y volume
                 + best in-model fixed schedules at the same volumes
  7. report    : tables (results/dp/report_<tag>.md / .json)

python -m dp.run_pipeline --tag c3 --steps kernels,fixed,sweep,baseline,grid,headline,report
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from .common import RES, ROOT, DEFAULT_CLASS_CUTS, risk_thresholds, load_settings_pool

FRAC_FEMALE = 0.5
N_HEAD = 1_000_000
N_GRID = 200_000


def kernels_path(tag):
    return os.path.join(RES, f'kernels_{tag}.npz')


def step_kernels(tag, cuts):
    out = kernels_path(tag)
    if os.path.exists(out):
        print('kernels cached', out); return out
    cmd = [sys.executable, '-m', 'dp.estimate_kernels', '--tag', tag, '--cuts', *[str(c) for c in cuts]]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def step_fixed(tag):
    from .fixed_search import run_search
    out = os.path.join(RES, f'fixed_search_{tag}.json')
    if os.path.exists(out):
        print('fixed search cached', out); return json.load(open(out))['rows']
    return run_search(kernels_path(tag), out_path=out, frac_female=FRAC_FEMALE)


def step_sweep(tag, objectives, workers, cap, rounds):
    from .sweep import run_sweep, pooled_rows
    out = {}
    for obj in objectives:
        res = run_sweep(kernels_path(tag), obj, tag=tag, workers=workers, cap=cap, rounds=rounds)
        out[obj] = pooled_rows(res, FRAC_FEMALE)
    return out


def arm_policy(row, observed_class=False, thr=None):
    a = dict(kind='policy', policy_male=row['path_male'], policy_female=row['path_female'],
             observed_class=observed_class)
    if observed_class:
        a['class_thr'] = [float(t) for t in thr]
    return a


def step_engine(arms, n, workers, label):
    from .evaluate import evaluate_arms, summary_table
    res, paths = evaluate_arms(arms, n, chunk=50_000, workers=workers)
    print(f'\n=== {label} (n={n:,}) ===\n' + summary_table(res), flush=True)
    out = os.path.join(RES, f'eval_{label}_n{n}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='c3')
    ap.add_argument('--cuts', type=float, nargs='*', default=list(DEFAULT_CLASS_CUTS))
    ap.add_argument('--steps', default='kernels,fixed,sweep,baseline,grid,headline,report')
    ap.add_argument('--objectives', default='death,inc,ly')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--solver-workers', type=int, default=4)
    ap.add_argument('--cap', type=int, default=600)
    ap.add_argument('--cap-hi', type=int, default=1500)
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--n-head', type=int, default=N_HEAD)
    ap.add_argument('--n-grid', type=int, default=N_GRID)
    a = ap.parse_args()
    steps = a.steps.split(',')
    objectives = a.objectives.split(',')
    tag = a.tag
    t0 = time.time()

    if 'kernels' in steps:
        step_kernels(tag, a.cuts)
    if 'fixed' in steps:
        step_fixed(tag)
    sweeps = {}
    if 'sweep' in steps:
        sweeps = step_sweep(tag, objectives, a.solver_workers, a.cap, a.rounds)
        print(f'[sweep done {time.time() - t0:.0f}s]', flush=True)
    if 'baseline' in steps:
        arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
                'q5y': {'kind': 'fixed', 'ages': [50, 55, 60, 65, 70, 75]}}
        step_engine(arms, a.n_head, a.workers, f'baseline_{tag}')
    if 'grid' in steps:
        from .sweep import pooled_rows
        for obj in objectives:
            sw = json.load(open(os.path.join(RES, f'sweep_{tag}_{obj}.json')))
            rows = pooled_rows(sw['rows'], FRAC_FEMALE)
            arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
                    'q5y': {'kind': 'fixed', 'ages': [50, 55, 60, 65, 70, 75]}}
            for r in rows:
                arms[f'{tag}_{obj}_lam{r["lam"]:.6g}'] = arm_policy(r)
            step_engine(arms, a.n_grid, a.workers, f'grid_{tag}_{obj}')
    if 'headline' in steps:
        from .sweep import pooled_rows, OBJECTIVES, policy_path, run_sweep
        thr = risk_thresholds(load_settings_pool(), tuple(a.cuts))
        fixed_rows = json.load(open(os.path.join(RES, f'fixed_search_{tag}.json')))['rows']
        arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
                'q5y': {'kind': 'fixed', 'ages': [50, 55, 60, 65, 70, 75]},
                'rule_52_10_5_3_3': {'kind': 'rule', 'start': 52, 'intervals': [10, 5, 3, 3]}}
        base = json.load(open(os.path.join(RES, f'eval_baseline_{tag}_n{a.n_head}.json')))
        picks = {}          # obj -> {'lams': [...], 'obs_lam': lam}
        for obj in objectives:
            grid = json.load(open(os.path.join(RES, f'eval_grid_{tag}_{obj}_n{a.n_grid}.json')))
            sw = json.load(open(os.path.join(RES, f'sweep_{tag}_{obj}.json')))
            rows = {f'{tag}_{obj}_lam{r["lam"]:.6g}': r for r in pooled_rows(sw['rows'], FRAC_FEMALE)}
            sel = {}
            for ref in ('q10y', 'q5y'):
                vol = base[ref]['colos_per_person']
                cands = [(grid[k]['colos_per_person'], k) for k in rows if k in grid]
                below = [c for c in cands if c[0] <= vol]
                ks = set()
                if below:
                    ks.add(max(below)[1])                                  # conservative: volume <= comparator
                ks.add(min(cands, key=lambda c: abs(c[0] - vol))[1])       # nearest overall
                for k in ks:
                    sel.setdefault(rows[k]['lam'], set()).add(ref)
            q10y_lams = [l for l, r in sel.items() if 'q10y' in r]
            picks[obj] = dict(sel=sel, obs_lam=max(q10y_lams) if q10y_lams else max(sel))
        # solve every needed high-cap policy in parallel (cached if present)
        for obj in objectives:
            lams = sorted(picks[obj]['sel'])
            run_sweep(kernels_path(tag), obj, lams=lams, tag=tag + 'hi', workers=a.solver_workers,
                      cap=a.cap_hi, rounds=4, rollouts=300)
            run_sweep(kernels_path(tag), obj, lams=[picks[obj]['obs_lam']], tag=tag + 'hiobs',
                      workers=a.solver_workers, cap=a.cap_hi, rounds=4, rollouts=300, class_known_roots=True)
        for obj in objectives:
            for lam, refs in sorted(picks[obj]['sel'].items()):
                hi = {sex: policy_path(obj, lam, sex, tag + 'hi') for sex in (1, 2)}
                nm = f'dp_{obj}_lam{lam:.6g}_' + '_'.join(sorted(refs))
                arms[nm] = dict(kind='policy', policy_male=hi[1], policy_female=hi[2], observed_class=False)
            lam = picks[obj]['obs_lam']
            ho = {sex: policy_path(obj, lam, sex, tag + 'hiobs') for sex in (1, 2)}
            arms[f'dp_{obj}_lam{lam:.6g}_obsclass'] = dict(kind='policy', policy_male=ho[1], policy_female=ho[2],
                                                           observed_class=True, class_thr=[float(t) for t in thr])
        # best in-model fixed schedules at q10y-like and q5y-like volumes
        for ref, lo_, hi_ in (('q10y', 2.2, 2.75), ('q5y', 4.5, 5.2)):
            sel = [r for r in fixed_rows if lo_ <= r['colos'] <= hi_]
            if sel:
                best = min(sel, key=lambda r: r['death'])
                arms[f'bestfixed_{ref}_' + '_'.join(str(x) for x in best['ages'])] = {'kind': 'fixed', 'ages': best['ages']}
        step_engine(arms, a.n_head, a.workers, f'headline_{tag}')
    if 'report' in steps:
        from .report import write_report
        write_report(tag, a.n_head, a.n_grid, objectives)
    print(f'[pipeline done {time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
