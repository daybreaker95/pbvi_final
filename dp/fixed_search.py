"""Exhaustive in-model search over FIXED colonoscopy schedules.

Produces, per sex-pooled population, every candidate schedule's exact
in-model metrics (deaths, diagnoses, complication deaths, life-years,
colonoscopies) so that a fair fixed-schedule efficiency frontier can be
drawn next to the adaptive policy's frontier.

Candidate families
  * equal-interval: start in 45..65, interval 3..15, last screen <= 80
  * free schedules of k <= 3 colonoscopies on a 2-year grid 44..80 and
    k = 4 on a 3-year grid
"""
from __future__ import annotations

import itertools
import json
import os
import time

import numpy as np

from .common import RES
from .model import ReducedPOMDP, evaluate_fixed, METRICS


def candidate_schedules():
    cands = set()
    for start in range(45, 66):
        for iv in range(3, 16):
            ages = tuple(range(start, 81, iv))
            cands.add(ages)
    grid2 = list(range(44, 81, 2))
    for k in (1, 2, 3):
        for c in itertools.combinations(grid2, k):
            cands.add(tuple(c))
    grid3 = list(range(44, 81, 3))
    for c in itertools.combinations(grid3, 4):
        cands.add(tuple(c))
    cands.add((50, 60, 70)); cands.add((50, 55, 60, 65, 70, 75)); cands.add(())
    return sorted(cands, key=lambda t: (len(t), t))


def pooled_fixed_eval(models, frac_female, ages):
    """Sex-pooled metrics of a fixed schedule (models: {1: m, 2: m})."""
    r1 = evaluate_fixed(models[1], ages); r2 = evaluate_fixed(models[2], ages)
    out = {}
    for k in list(METRICS) + ['colos']:
        out[k] = (1 - frac_female) * r1[k] + frac_female * r2[k]
    return out


def run_search(kernels_npz, out_path=None, frac_female=0.5, verbose=True):
    models = {s: ReducedPOMDP(s, kernels_npz) for s in (1, 2)}
    cands = candidate_schedules()
    rows = []
    t0 = time.time()
    for i, ages in enumerate(cands):
        r = pooled_fixed_eval(models, frac_female, ages)
        rows.append(dict(ages=list(ages), n=len(ages), **r))
        if verbose and (i + 1) % 500 == 0:
            print(f'  {i + 1}/{len(cands)} ({time.time() - t0:.0f}s)', flush=True)
    base = next(r for r in rows if r['n'] == 0)
    for r in rows:
        c = r['colos']
        r['deaths_averted_per_colo'] = (base['death'] - r['death']) / c if c > 0 else 0.0
        r['cases_averted_per_colo'] = (base['inc'] - r['inc']) / c if c > 0 else 0.0
        r['lyg_per_colo'] = (r['ly'] - base['ly']) / c if c > 0 else 0.0
    out_path = out_path or os.path.join(RES, 'fixed_search.json')
    with open(out_path, 'w') as f:
        json.dump(dict(kernels=kernels_npz, frac_female=frac_female, rows=rows), f)
    if verbose:
        print('saved', out_path, f'({len(rows)} schedules, {time.time() - t0:.0f}s)')
    return rows


def frontier(rows, metric='death', bins=np.arange(0.5, 8.51, 0.5)):
    """Best fixed schedule (lowest metric) per colonoscopy-volume bin."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if lo <= r['colos'] < hi]
        if sel:
            out.append(min(sel, key=lambda r: r[metric]))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--frac-female', type=float, default=0.5)
    a = ap.parse_args()
    rows = run_search(a.kernels, frac_female=a.frac_female)
    for r in frontier(rows, 'death'):
        print(f"colos={r['colos']:.2f} deaths/100k={r['death'] * 1e5:.0f} inc/100k={r['inc'] * 1e5:.0f} ages={r['ages']}")
