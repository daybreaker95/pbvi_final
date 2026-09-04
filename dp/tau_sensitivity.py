"""tau-support sensitivity.

The open-ended top tau group (13+ years since the last colonoscopy) pools
every person-year with tau >= 13. In the randomised-schedule cohort the
intervals are 1-20 years and screening continues to age 80, so at the
decision ages (40-80) the group contains only tau 13-20 person-years; beyond
age 80, where no colonoscopy is offered, tau keeps growing and the group is
increasingly made of persons whose screening simply stopped. This script
re-estimates the kernels with every WAIT person-year at tau > 20 excluded
(dp.estimate_kernels --tau-max 20), re-solves the headline price on them,
and deploys the resulting policy in the engine on the headline arm's chunk
seeds.

python -m dp.estimate_kernels --tag c6b_tau20 --tau-max 20 --cuts 0.5 0.8 0.95 0.965 0.98 \
       --screen-runs results/dp/runs/screen_random_q,results/dp/runs/screen_random_q2
python -m dp.tau_sensitivity --solve --workers 2
python -m dp.tau_sensitivity --deploy --n 200000 --workers 4
python -m dp.tau_sensitivity --report --n 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES
from .sweep import run_sweep, policy_path

LAM = 0.001561
KERNELS = os.path.join(RES, 'kernels_c6b_tau20.npz')
BASE = os.path.join(RES, 'kernels_c6b.npz')
TAG = 'c6btau20hi'
HEADLINE_ARM = 'dp_death_lam0.001561_q10y'


def solve(workers):
    run_sweep(KERNELS, 'death', lams=[LAM], tag=TAG, workers=workers, cap=1500, rounds=4, rollouts=300)


def deploy(n, workers):
    from .evaluate import evaluate_arms, summary_table
    arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            HEADLINE_ARM: {'kind': 'policy',
                           'policy_male': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex1.npz'),
                           'policy_female': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex2.npz'),
                           'observed_class': False},
            f'tau20_lam{LAM:g}': {'kind': 'policy', 'policy_male': policy_path('death', LAM, 1, TAG),
                                  'policy_female': policy_path('death', LAM, 2, TAG), 'observed_class': False}}
    res, _ = evaluate_arms(arms, n, chunk=50_000, workers=workers)
    print(summary_table(res), flush=True)
    out = os.path.join(RES, f'eval_tau_sensitivity_n{n}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out)


def report(n):
    from .solver import load_policy
    from .report import policy_typical_paths
    from .validate import model_predictions
    from .engine_runner import load_chunk, arm_dir
    a = np.load(BASE, allow_pickle=True); b = np.load(KERNELS, allow_pickle=True)
    y0 = int(a['y0'])
    dT = np.abs(b['Tw'] - a['Tw']).max(axis=(0, 1, 2, 4, 5))       # by age
    dE = np.abs(b['Ew'] - a['Ew']).max(axis=(0, 1, 2, 4, 5))
    out = dict(lam=LAM, kernels=KERNELS,
               kernel_diff=dict(max_dTw_ages_40_80=float(dT[:41].max()), max_dTw_ages_81_99=float(dT[41:].max()),
                                max_dEw_ages_40_80=float(dE[:41].max()), max_dEw_ages_81_99=float(dE[41:].max()),
                                rows_changed=int((np.abs(b['Tw'] - a['Tw']).max(axis=-1) > 0).sum()),
                                rows_changed_ages_40_80=int((np.abs(b['Tw'] - a['Tw']).max(axis=-1)[:, :, :, :41] > 0).sum()),
                                person_years_excluded=int(a['W_rowcounts'].sum() - b['W_rowcounts'].sum()),
                                person_years_total=int(a['W_rowcounts'].sum())))
    pb, _ = model_predictions(BASE); pt, _ = model_predictions(KERNELS)
    out['table1_model'] = {k: dict(base_death=pb[k]['death'] * 1e5, tau20_death=pt[k]['death'] * 1e5,
                                   base_inc=pb[k]['inc'] * 1e5, tau20_inc=pt[k]['inc'] * 1e5) for k in pb}
    rows = {}
    for name, tag in (('headline', 'c6bhi'), ('tau20', TAG)):
        rows[name] = {}
        for s in (1, 2):
            pol = load_policy(policy_path('death', LAM, s, tag))
            ev = pol.meta['eval']
            rows[name][s] = dict(objective=pol.meta['objective'], death=ev['death'], inc=ev['inc'], colos=ev['colos'],
                                 gap=pol.meta['gap'], paths=policy_typical_paths(pol))
    out['in_model'] = rows
    out['pooled'] = {nm: {k: 0.5 * (r[1][k] + r[2][k]) for k in ('objective', 'death', 'inc', 'colos', 'gap')} for nm, r in rows.items()}
    # cross-evaluation: each policy scored on the other kernel set (exact)
    from .model import ReducedPOMDP
    cross = {}
    for pname, tag in (('headline', 'c6bhi'), ('tau20', TAG)):
        for kname, kp in (('base', BASE), ('tau20', KERNELS)):
            vals = []
            for s in (1, 2):
                pol = load_policy(policy_path('death', LAM, s, tag))
                m = ReducedPOMDP(s, kp, lam=LAM)
                from .model import evaluate_policy
                r = evaluate_policy(m, None, action_batch=pol.best_action_batch)
                vals.append(r)
            cross[f'policy {pname} on kernels {kname}'] = {k: 0.5 * (vals[0][k] + vals[1][k]) for k in ('death', 'inc', 'colos', 'objective')}
    out['cross_evaluation'] = cross
    ev_path = os.path.join(RES, f'eval_tau_sensitivity_n{n}.json')
    if os.path.exists(ev_path):
        eng = json.load(open(ev_path))
        def per(tag):
            o = {}
            for p in sorted(glob.glob(os.path.join(arm_dir(tag), 'seed*_n50000.npz')))[:n // 50_000]:
                d = load_chunk(p); dy = d['death_year'].astype(int)
                o[d['summary']['seed']] = dict(death=(d['crc_death'] & (dy >= 40)).sum() / len(dy) * 1e5,
                                               dx=d['diagnosed'].sum() / len(dy) * 1e5, colos=d['n_policy_colo'].sum() / len(dy))
            return o
        H = per(HEADLINE_ARM); T = per(f'tau20_lam{LAM:g}')
        seeds = sorted(set(H) & set(T))
        for k in ('death', 'dx', 'colos'):
            d = np.array([T[s][k] - H[s][k] for s in seeds])
            eng[f'tau20_lam{LAM:g}'][f'paired_vs_headline_{k}'] = dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), n_chunks=len(d))
        out['engine'] = eng
    path = os.path.join(RES, 'tau_sensitivity.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    kd = out['kernel_diff']
    L = ['# tau-support sensitivity (WAIT person-years at tau > 20 excluded)', '',
         f"- person-years excluded: {kd['person_years_excluded']:,} of {kd['person_years_total']:,} ({100 * kd['person_years_excluded'] / kd['person_years_total']:.2f} %)",
         f"- kernel rows changed: {kd['rows_changed']:,} (of which at decision ages 40-80: {kd['rows_changed_ages_40_80']:,}); max |dT| ages 40-80 = {kd['max_dTw_ages_40_80']:.4f}, ages 81-99 = {kd['max_dTw_ages_81_99']:.4f}",
         '', '## Table 1 predictions (deaths / dx per 100k)', '']
    for k, v in out['table1_model'].items():
        L.append(f"- {k}: base {v['base_death']:.1f} / {v['base_inc']:.1f}; tau20 {v['tau20_death']:.1f} / {v['tau20_inc']:.1f}")
    L += ['', '## Headline price re-solved on the tau20 kernels (in-model, pooled)', '']
    for nm, p in out['pooled'].items():
        L.append(f"- {nm}: deaths {p['death'] * 1e5:.1f}, dx {p['inc'] * 1e5:.0f}, colos {p['colos']:.3f}, objective {p['objective']:.6f}, gap {p['gap']:.5f}")
    L += ['', '## Cross-evaluation (exact, pooled)', '']
    for k, v in cross.items():
        L.append(f"- {k}: deaths {v['death'] * 1e5:.1f}, colos {v['colos']:.3f}, objective {v['objective']:.6f}")
    L += ['', '## Screening ages along canonical paths (male / female)', '']
    for nm, r in rows.items():
        L.append(f'### {nm}')
        for pth in r[1]['paths']:
            L.append(f"- {pth}: {r[1]['paths'][pth]} / {r[2]['paths'][pth]}")
        L.append('')
    if 'engine' in out:
        L += [f'## Engine (n = {n:,}, paired chunks)', '']
        for tag, r in out['engine'].items():
            extra = ''
            if 'paired_vs_headline_death' in r:
                pv = r['paired_vs_headline_death']; pc = r['paired_vs_headline_colos']
                extra = f"; paired vs headline: deaths {pv['mean']:+.1f} +- {pv['se']:.1f}, colos {pc['mean']:+.3f}"
            L.append(f"- {tag}: colos {r['colos_per_person']:.3f}, deaths {r['crc_death_per_100k']:.1f} ({r['crc_death_se']:.1f}){extra}")
    md = path.replace('.json', '.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', path, md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', action='store_true')
    ap.add_argument('--deploy', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.solve:
        solve(a.workers)
    if a.deploy:
        deploy(a.n, a.workers)
    if a.report:
        report(a.n)


if __name__ == '__main__':
    main()
