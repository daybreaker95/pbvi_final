"""The finite-discrimination baseline risk-score scenario.

For each target AUC the shadow price is bisected so that the policy's exact
in-model colonoscopy volume matches the latent-class headline policy's. That
matters: at a fixed shadow price a better score both retargets AND shrinks
volume, so a fixed-lambda comparison would confound the two. Matching volume
makes "what does a score of this quality buy" a question with one answer.

Arms:
  score_aucXX    policy solved and deployed with the score (matched volume)
  score_inf      sigma -> infinity control: an uninformative score, which must
                 reproduce the latent-class arm (wiring check)
  scorefixed_XX  the same score used to pick a per-band FIXED schedule
                 (the 2 x 2 that separates "information" from "adaptivity")

python -m dp.run_riskscore --n 200000 --workers 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import RES
from .engine_runner import run_arm, aggregate, efficiency, paired_diff
from .evaluate import summary_table
from .model import ReducedPOMDP, evaluate_fixed
from .riskscore import (
    beliefs_for_scores, calibrate, root_beliefs, root_band_edges, score_grid, CACHE,
    CONTROL_SIGMA, control_auc,
)
from .solver import solve_policy, load_policy
from .sweep import policy_path

KERNELS = os.path.join(RES, 'kernels_c6b.npz')
POL = os.path.join(RES, 'policies')
TABLES = os.path.join(RES, 'score_tables')
os.makedirs(TABLES, exist_ok=True)
FRAC_FEMALE = 0.5
N_CELLS = 2048
N_ROOTS = 14

# the volume to match: the latent-class headline policy's exact in-model
# pooled colonoscopy count (engine volume 2.289 per person)
TARGET_VOLUME = None       # computed at run time from the headline policies


def headline_volume():
    v = []
    for sex in (1, 2):
        p = load_policy(os.path.join(POL, f'c6bhi_death_lam0.001561_sex{sex}.npz'))
        v.append(p.meta['eval']['colos'])
    return (1 - FRAC_FEMALE) * v[0] + FRAC_FEMALE * v[1]


def _sweep(sigma, lams, tag, cap, rounds, workers, n_class=6, rollouts=200, n_roots=N_ROOTS):
    """Solve the given lambdas for both sexes in parallel with the score-band
    beliefs as the deployment roots; return {lam: pooled in-model metrics}."""
    from .sweep import run_sweep
    B, W, _ = root_beliefs(sigma, n_class, n_roots)
    res = run_sweep(KERNELS, 'death', lams=[float(l) for l in lams], tag=tag, workers=workers,
                    cap=cap, rounds=rounds, rollouts=rollouts,
                    root_beliefs=[b.tolist() for b in B], root_weights=W.tolist())
    by = {}
    for r in res:
        by.setdefault(r['lam'], {})[r['sex']] = r
    out = {}
    for lam, d in by.items():
        if 1 in d and 2 in d:
            out[lam] = {k: (1 - FRAC_FEMALE) * d[1][k] + FRAC_FEMALE * d[2][k]
                        for k in ('death', 'inc', 'comp', 'ly', 'colos', 'objective')}
    return out


def match_volume(sigma, target, tag, workers, probe=(0.0008, 0.0016, 0.0032),
                 cap=400, rounds=2, verbose=True):
    """lambda whose in-model pooled volume equals `target`, by log-log
    interpolation through a small parallel probe grid (the volume-price curve
    is smooth and monotone, so three points suffice; a sequential bisection
    would cost four times as many solves)."""
    ev = _sweep(sigma, probe, tag + 'cal', cap, rounds, workers, n_roots=6)
    pts = sorted((v['colos'], l) for l, v in ev.items() if v['colos'] > 0)
    if verbose:
        for c, l in pts:
            print(f'    probe lam={l:.6g} -> colos {c:.3f}', flush=True)
    if len(pts) < 2:
        return probe[len(probe) // 2]
    x = np.log([l for _, l in pts]); y = np.log([c for c, _ in pts])
    order = np.argsort(y)
    lam = float(np.exp(np.interp(np.log(target), y[order], x[order])))
    return float(f'{lam:.6g}')


def build_deployment(sigma, n_class, tag):
    """Belief table on a fine equal-mass score grid + the reporting bands."""
    path = os.path.join(TABLES, f'{tag}.npz')
    cell_edges, mids = score_grid(sigma, N_CELLS)
    if not os.path.exists(path):
        beliefs = beliefs_for_scores(mids, sigma, n_class)
        np.savez_compressed(path, beliefs=beliefs, cell_edges=cell_edges, mids=mids)
    _, _, band_edges_ = root_beliefs(sigma, n_class, N_ROOTS)
    return path, cell_edges, band_edges_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--aucs', type=float, nargs='*', default=[0.55, 0.60, 0.65, 0.70])
    ap.add_argument('--cap', type=int, default=1200)
    ap.add_argument('--rounds', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    cal = json.load(open(CACHE))
    n_class = len(cal['cuts']) + 1
    target = headline_volume()
    print(f'matching in-model volume {target:.3f} colonoscopies/person '
          f'(the latent-class headline policy)', flush=True)

    levels = []
    for t in a.aucs:
        key = f'{t:.2f}'
        if key not in cal['sigmas']:
            print(f'  skipping AUC {key}: above the ceiling {cal["ceiling_auc"]:.3f}'); continue
        levels.append((key, cal['sigmas'][key]['sigma'], cal['sigmas'][key]['auc']))
    levels.append(('ceiling', 0.0, cal['ceiling_auc']))
    levels.append(('uninformative', CONTROL_SIGMA, control_auc()))   # sigma -> inf control

    arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            'dp_death_lam0.001561_q10y': {
                'kind': 'policy',
                'policy_male': os.path.join(POL, 'c6bhi_death_lam0.001561_sex1.npz'),
                'policy_female': os.path.join(POL, 'c6bhi_death_lam0.001561_sex2.npz'),
                'observed_class': False},
            'dp_death_lam0.001561_obsclass': {
                'kind': 'policy',
                'policy_male': os.path.join(POL, 'c6bhiobs_death_lam0.001561_sex1.npz'),
                'policy_female': os.path.join(POL, 'c6bhiobs_death_lam0.001561_sex2.npz'),
                'observed_class': True,
                'class_thr': list(np.load(KERNELS, allow_pickle=True)['thr'])}}
    meta = {}
    for key, sigma, auc_v in levels:
        tag = f'score{key.replace(".", "")}'
        print(f'\n== AUC {key} (sigma {sigma:.3f}, achieved AUC {auc_v:.3f})', flush=True)
        lam = match_volume(sigma, target, tag, a.workers)
        print(f'  matched lambda = {lam:.6g}; solving at cap {a.cap}', flush=True)
        ev = _sweep(sigma, [lam], tag, a.cap, a.rounds, a.workers)[lam]
        print(f'  in-model: colos {ev["colos"]:.3f} deaths {ev["death"] * 1e5:.0f} '
              f'dx {ev["inc"] * 1e5:.0f}  ({time.time() - t0:.0f}s)', flush=True)
        tbl, cell_edges, band_edges_ = build_deployment(sigma, n_class, tag)
        arms[f'score_{key}'] = {
            'kind': 'policy',
            'policy_male': policy_path('death', lam, 1, tag),
            'policy_female': policy_path('death', lam, 2, tag),
            'observed_class': False,
            'score_sigma': sigma, 'score_belief_table': tbl,
            'score_cell_edges': [float(x) for x in cell_edges],
            'score_band_edges': [float(x) for x in band_edges_]}
        meta[key] = dict(sigma=sigma, auc=auc_v, lam=lam, in_model=ev)

    print(f'\n== engine evaluation, n = {a.n:,} per arm', flush=True)
    paths = {}
    for tag, arm in arms.items():
        print(f'-- {tag}', flush=True)
        paths[tag] = run_arm(arm, tag if not tag.startswith('score_') else f'{tag}_n{a.n}',
                             a.n, chunk=50_000, workers=a.workers, verbose=False)
    base = aggregate(paths['none'])
    res = {}
    for tag in arms:
        r = aggregate(paths[tag])
        if tag != 'none':
            r.update(efficiency(r, base))
            d, se = paired_diff(paths[tag], paths['q10y'], 'crc_death')
            r['paired_death_vs_q10y'] = d; r['paired_death_vs_q10y_se'] = se
        res[tag] = r
    print('\n' + summary_table(res), flush=True)
    out = os.path.join(RES, f'eval_riskscore_n{a.n}.json')
    with open(out, 'w') as f:
        json.dump({'arms': res, 'levels': meta, 'target_volume': target}, f, indent=1)
    print('saved', out, f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
