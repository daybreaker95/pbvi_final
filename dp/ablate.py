"""Model-structure ablation: how much does each modelling choice matter?

Re-estimates the kernels from the SAME cached engine cohorts under coarser
model structures and asks each variant to predict, as a cohort model, what
the engine does under no screening / 10-yearly / 5-yearly colonoscopy. The
variants differ only in the abstraction, never in the data:

  full          11 clinical states, (tau x last finding) memory, 6 risk classes
  pooled        polyp stages pooled to early/advanced (7 clinical states)
  nomem         no memory conditioning (one natural-history kernel per age)
  pooled_nomem  both coarsenings
  c1 / c3       1 or 3 latent risk classes instead of 6

python -m dp.ablate --out results/dp/ablation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from . import estimate_kernels as EK
from .common import (
    RES, RUNS, NX, MAP18_TO_EXIT, DEFAULT_CLASS_CUTS, risk_thresholds, load_settings_pool,
)
from .model import ReducedPOMDP, evaluate_fixed

SCHEDULES = {'none': [], 'q10y': [50, 60, 70], 'q5y': [50, 55, 60, 65, 70, 75]}

# 18-state -> pooled 7-state clinical axis (N, EarlyAdenoma, AdvAdenoma, U I-IV)
MAP_POOLED = np.full(18, -1, dtype=np.int8)
MAP_POOLED[0] = 0
for s in (1, 2, 3, 4):
    MAP_POOLED[s] = 1
for s in (5, 6):
    MAP_POOLED[s] = 2
for k, s in enumerate((7, 8, 9, 10)):
    MAP_POOLED[s] = 3 + k

VARIANTS = {
    'full':         dict(pool=False, memory=True,  cuts=(0.5, 0.8, 0.95, 0.965, 0.98)),
    'pooled':       dict(pool=True,  memory=True,  cuts=(0.5, 0.8, 0.95, 0.965, 0.98)),
    'nomem':        dict(pool=False, memory=False, cuts=(0.5, 0.8, 0.95, 0.965, 0.98)),
    'pooled_nomem': dict(pool=True,  memory=False, cuts=(0.5, 0.8, 0.95, 0.965, 0.98)),
    'c1':           dict(pool=False, memory=True,  cuts=()),
    'c3':           dict(pool=False, memory=True,  cuts=(0.5, 0.9)),
}


def _set_axis(pool: bool):
    """Point the estimator at a coarser (or the full) clinical axis."""
    nc = 7 if pool else 11
    mp = MAP_POOLED if pool else np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1], dtype=np.int8)
    EK.NC = nc
    EK.MAP18_TO_CLIN = mp
    EK.NT_W = nc + NX
    EK.NT_K = EK.NO4 * nc + NX
    EK._TO_CODE = np.where(mp >= 0, mp, nc + MAP18_TO_EXIT).astype(np.int16)
    return nc


_TRUE_MEM_INDEX = EK.mem_index


def _set_memory(on: bool):
    if on:
        EK.mem_index = _TRUE_MEM_INDEX
    else:
        EK.mem_index = lambda tau, ol: np.zeros(np.shape(tau), dtype=np.int64)


def estimate_variant(name, nh_paths, sc_paths, force=False):
    cfg = VARIANTS[name]
    out = os.path.join(RES, f'kernels_abl_{name}.npz')
    if os.path.exists(out) and not force:
        print(f'  [{name}] cached', flush=True)
        return out
    t0 = time.time()
    nc = _set_axis(cfg['pool'])
    _set_memory(cfg['memory'])
    cuts = cfg['cuts']
    n_class = len(cuts) + 1
    thr = risk_thresholds(load_settings_pool(), cuts) if cuts else np.zeros(0)
    print(f'  [{name}] estimating: {nc} clinical states, memory={cfg["memory"]}, {n_class} classes', flush=True)
    W1, K1, occ40, dx1, n1 = EK.accumulate(nh_paths, thr, n_class, with_logs=False, verbose=False)
    W2, K2, _, dx2, n2 = EK.accumulate(sc_paths, thr, n_class, with_logs=True, verbose=False)
    W = W1 + W2; Kc = K1 + K2; dx = dx1 + dx2
    Tw, Ew, K, X, fw, fk = EK.finalize(W, Kc)
    b0 = occ40.reshape(2, n_class * nc).astype(float); b0 /= b0.sum(axis=1, keepdims=True)
    Vd, Vl = EK.smooth_dx(dx)
    np.savez_compressed(out, Tw=Tw, Ew=Ew, K=K, X=X, b0=b0, thr=thr, cuts=np.array(cuts),
                        n_class=n_class, nc=nc, y0=EK.Y0, y1=EK.Y1, bands=np.array(EK.BANDS),
                        Vexit_death=Vd, Vexit_ly=Vl, n_nh=n1, n_screen=n2)
    print(f'  [{name}] saved ({time.time() - t0:.0f}s)', flush=True)
    return out


def engine_reference():
    """Engine outcomes on the model's own population -- persons alive and
    undiagnosed at the age-40 decision epoch (the start of the second quarter
    of age 40, the snapshot from which the initial belief is estimated) -- per
    100 000. The mask is read from the never-screened quarterly cohort, whose
    first chunks share the headline arms' seeds (identical persons; no policy
    colonoscopy precedes age 40), so it transfers person-by-person to every
    arm. Falls back to the per-person arrays (death_year >= 41 and no diagnosis
    before 41) for chunks without a quarterly recording."""
    from .common import MAP18_TO_CLIN
    masks = {}
    for p in sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', '*.npz'))):
        z = np.load(p, allow_pickle=True)
        masks[json.loads(str(z['summary']))['seed']] = MAP18_TO_CLIN[z['qr'][157]] >= 0
    ref = {}
    for nm, tag in (('none', 'none'), ('q10y', 'q10y'), ('q5y', 'q5y')):
        N = D = I = C = 0
        for p in sorted(glob.glob(os.path.join(RUNS, tag, '*.npz'))):
            d = np.load(p, allow_pickle=True)
            seed = json.loads(str(d['summary']))['seed']
            if seed in masks:
                m = masks[seed]
            else:
                m = (d['death_year'] >= 41) & ((d['dx_year'] == 0) | (d['dx_year'] > 40))
            N += m.sum(); D += d['crc_death'][m].sum(); I += d['diagnosed'][m].sum()
            C += d['n_policy_colo'][m].sum()
        ref[nm] = dict(death=D / N * 1e5, inc=I / N * 1e5, colos=C / N, n=int(N),
                       mask='age-40 decision snapshot' if masks else 'per-person arrays')
    return ref


def predict(kernels, frac_female=0.5):
    m = {s: ReducedPOMDP(s, kernels) for s in (1, 2)}
    out = {}
    for nm, ages in SCHEDULES.items():
        r = [evaluate_fixed(m[s], ages) for s in (1, 2)]
        out[nm] = {k: (1 - frac_female) * r[0][k] + frac_female * r[1][k] for k in r[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--out', default=os.path.join(RES, 'ablation.json'))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    nh = sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', '*.npz')))
    sc = sorted(sum((glob.glob(os.path.join(RUNS, d, '*.npz'))
                     for d in ('screen_random_q', 'screen_random_q2')), []))
    print(f'cohorts: {len(nh)} never-screened chunks, {len(sc)} screened chunks', flush=True)
    ref = engine_reference()
    print(f"engine reference (n={ref['none']['n']:,}): " +
          ', '.join(f"{k} {v['death']:.0f}/{v['inc']:.0f}" for k, v in ref.items()), flush=True)
    rows = {}
    for name in a.variants:
        kern = estimate_variant(name, nh, sc, force=a.force)
        pr = predict(kern)
        rows[name] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in pr.items()}
    # report
    hdr = f"{'variant':14s}" + ''.join(f"{s + ' death':>14s}{s + ' inc':>13s}" for s in SCHEDULES)
    print('\n' + hdr)
    print(f"{'ENGINE':14s}" + ''.join(f"{ref[s]['death']:14.0f}{ref[s]['inc']:13.0f}" for s in SCHEDULES))
    for name, pr in rows.items():
        line = f'{name:14s}'
        for s in SCHEDULES:
            line += f"{pr[s]['death'] * 1e5:9.0f}({100 * (pr[s]['death'] * 1e5 / ref[s]['death'] - 1):+4.0f}%)"
            line += f"{pr[s]['inc'] * 1e5:8.0f}({100 * (pr[s]['inc'] * 1e5 / ref[s]['inc'] - 1):+4.0f}%)"
        print(line)
    print(f"\n{'variant':14s} {'q10y death red.':>16s} {'q10y inc red.':>14s} {'q5y death red.':>15s} {'q5y inc red.':>13s}")
    er = {s: (100 * (1 - ref[s]['death'] / ref['none']['death']), 100 * (1 - ref[s]['inc'] / ref['none']['inc']))
          for s in ('q10y', 'q5y')}
    print(f"{'ENGINE':14s} {er['q10y'][0]:15.1f}% {er['q10y'][1]:13.1f}% {er['q5y'][0]:14.1f}% {er['q5y'][1]:12.1f}%")
    for name, pr in rows.items():
        r = {s: (100 * (1 - pr[s]['death'] / pr['none']['death']), 100 * (1 - pr[s]['inc'] / pr['none']['inc']))
             for s in ('q10y', 'q5y')}
        print(f"{name:14s} {r['q10y'][0]:15.1f}% {r['q10y'][1]:13.1f}% {r['q5y'][0]:14.1f}% {r['q5y'][1]:12.1f}%")
    with open(a.out, 'w') as f:
        json.dump(dict(engine=ref, variants=rows), f, indent=1)
    print('\nsaved', a.out)


if __name__ == '__main__':
    main()
