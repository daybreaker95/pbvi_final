"""The other half of the 2 x 2: give the SAME baseline risk score to a FIXED
programme.

For every score band, the best fixed schedule is chosen from the same 2 112
candidates the unstratified search uses, evaluated against that band's own
age-40 belief. A single price lambda selects each band's schedule, and lambda
is tuned so the population colonoscopy volume matches the adaptive arms'. The
contrast between this arm and the adaptive score arm is the part of the
benefit that requires responding to findings rather than merely knowing a
baseline risk.

python -m dp.score_fixed --auc 0.60 --n 200000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import RES
from .fixed_search import candidate_schedules
from .model import ReducedPOMDP, evaluate_fixed
from .riskscore import root_beliefs, score_grid, CACHE
from .run_riskscore import KERNELS, FRAC_FEMALE, N_ROOTS, headline_volume, TABLES

FRAC = FRAC_FEMALE


def band_tables(sigma, n_class=6, n_roots=N_ROOTS):
    """For each (sex, band): every candidate schedule's exact in-model deaths
    and colonoscopy count under that band's age-40 belief."""
    B, W, edges = root_beliefs(sigma, n_class, n_roots)
    cands = candidate_schedules()
    out = {}
    for sex in (1, 2):
        m = ReducedPOMDP(sex, KERNELS)
        rows = np.zeros((len(B), len(cands), 2))
        for bi, b in enumerate(B):
            for ci, ages in enumerate(cands):
                r = evaluate_fixed(m, ages, b0=b)
                rows[bi, ci] = (r['death'], r['colos'])
        out[sex] = rows
    return out, np.asarray(W), cands, edges


def select(tables, W, cands, lam):
    """Per-band argmin of (deaths + lambda * colonoscopies); returns the chosen
    schedule per (sex, band) and the pooled population totals."""
    chosen = {}
    death = colos = 0.0
    for sex, rows in tables.items():
        w = (1 - FRAC) if sex == 1 else FRAC
        idx = np.argmin(rows[:, :, 0] + lam * rows[:, :, 1], axis=1)
        chosen[sex] = [cands[i] for i in idx]
        death += w * float((W * rows[np.arange(len(W)), idx, 0]).sum())
        colos += w * float((W * rows[np.arange(len(W)), idx, 1]).sum())
    return chosen, death, colos


def match(tables, W, cands, target, lo=1e-5, hi=1e-1, iters=40):
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        _, _, c = select(tables, W, cands, mid)
        if c > target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def frontier(tables, W, cands, n_lam=600, lo=3e-4, hi=1e-2):
    """The price-volume envelope of the band-stratified fixed programme.

    A single price cannot land on an arbitrary volume: `select` is an argmin
    over a discrete candidate set, so volume moves in steps. The reported cell
    must therefore be read off this envelope at the target volume, exactly as
    the adaptive arms are read off theirs in dp.score_frontier."""
    pts = {}
    for lam in np.exp(np.linspace(np.log(lo), np.log(hi), n_lam)):
        _, d, c = select(tables, W, cands, float(lam))
        pts[round(c, 9)] = (d, float(lam))
    c = np.array(sorted(pts))
    return c, np.array([pts[x][0] for x in c]), np.array([pts[x][1] for x in c])


def read_at(c, v, target):
    if target < c.min() or target > c.max():
        return float('nan'), False
    return float(np.interp(target, c, v)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auc', type=float, default=0.60)
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()
    cal = json.load(open(CACHE))
    key = f'{a.auc:.2f}'
    sigma = cal['sigmas'][key]['sigma']
    target = headline_volume()
    print(f'AUC {key} (sigma {sigma:.3f}); matching volume {target:.3f}', flush=True)
    tables, W, cands, edges = band_tables(sigma)
    print(f'  evaluated {len(cands)} schedules x {len(W)} bands x 2 sexes '
          f'({time.time() - t0:.0f}s)', flush=True)
    lam = match(tables, W, cands, target)
    chosen, death, colos = select(tables, W, cands, lam)
    print(f'  lambda {lam:.6g} -> in-model colos {colos:.3f}, deaths {death * 1e5:.0f}', flush=True)
    fc, fv, fl = frontier(tables, W, cands)
    at_target, ok = read_at(fc, fv, target)
    k = int(np.searchsorted(fc, target))
    print(f'  price steps bracketing the target: {fc[k - 1]:.4f} -> {fv[k - 1] * 1e5:.1f}, '
          f'{fc[k]:.4f} -> {fv[k] * 1e5:.1f}', flush=True)
    print(f'  READ AT {target:.4f}: {at_target * 1e5:.1f} deaths /100k'
          f"{'' if ok else '  (EXTRAPOLATED)'}", flush=True)

    # the like-for-like score-blind comparator: the SAME band machinery with an
    # uninformative score, so the no-score fixed cell of the 2 x 2 differs from
    # the score cell only in the information the bands carry
    print('  building the sigma -> infinity comparator', flush=True)
    t60, W60, c60, _ = band_tables(60.0)
    f60c, f60v, _ = frontier(t60, np.asarray(W60), c60)
    blind, ok60 = read_at(f60c, f60v, target)
    print(f'  no-score fixed at {target:.4f}: {blind * 1e5:.1f} deaths /100k'
          f"{'' if ok60 else '  (EXTRAPOLATED)'}  ({time.time() - t0:.0f}s)", flush=True)
    for sex in (1, 2):
        print(f'  {"male" if sex == 1 else "female"} schedules by band:')
        for bi, ages in enumerate(chosen[sex]):
            print(f'    band {bi + 1:2d} (share {W[bi]:.3f}): {list(ages)}')
    # deploy: the engine hook needs one age list per (sex, band), matching the
    # sex-specific selection that produced the in-model numbers above
    cell_edges, _ = score_grid(sigma, 2048)
    from .engine_runner import run_arm, aggregate, efficiency
    arm = {'kind': 'band_fixed', 'score_sigma': sigma,
           'score_cell_edges': [float(x) for x in cell_edges],
           'score_band_edges': [float(x) for x in edges],
           'band_ages': [list(x) for x in chosen[1]],
           'band_ages_female': [list(x) for x in chosen[2]]}
    tag = f'scorefixed_{key}_n{a.n}'
    paths = run_arm(arm, tag, a.n, chunk=50_000, workers=a.workers, verbose=False)
    r = aggregate(paths)
    out = os.path.join(RES, f'eval_scorefixed_{key}_n{a.n}.json')
    with open(out, 'w') as f:
        json.dump({'auc': a.auc, 'sigma': sigma, 'lam': lam,
                   'target_volume': target,
                   'in_model': {'death': death, 'colos': colos},
                   'in_model_at_target': {'death': at_target, 'in_range': bool(ok)},
                   'no_score_at_target': {'death': blind, 'in_range': bool(ok60)},
                   'frontier': [[float(x), float(y)] for x, y in zip(fc, fv)],
                   'frontier_no_score': [[float(x), float(y)] for x, y in zip(f60c, f60v)],
                   'schedules_male': [list(x) for x in chosen[1]],
                   'schedules_female': [list(x) for x in chosen[2]],
                   'band_weights': W.tolist(), 'engine': r}, f, indent=1)
    print(f"engine: colos {r['colos_per_person']:.3f} deaths {r['crc_death_per_100k']:.1f} "
          f"dx {r['incidence_per_100k']:.1f}", flush=True)
    print('saved', out, f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
