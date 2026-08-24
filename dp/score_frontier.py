"""Complete each AUC level's in-model frontier and read it at a common volume.

A single interpolated shadow price does not land every level on the same
colonoscopy volume (the price-volume curve shifts with the belief set), and an
unmatched volume would confound "the score retargets" with "the score buys a
smaller programme". So each level gets a small frontier solved at the
reporting budget, and the answer is read off it at the latent-class policy's
volume.

python -m dp.score_frontier --workers 5
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import RES
from .riskscore import CACHE, CONTROL_SIGMA, control_auc
from .run_riskscore import KERNELS, FRAC_FEMALE, N_ROOTS, headline_volume, _sweep

OUT = os.path.join(RES, 'score_frontier.json')


def frontier(sigma, tag, lams, cap, rounds, workers):
    ev = _sweep(sigma, lams, tag, cap, rounds, workers)
    return sorted(({'lam': l, **v} for l, v in ev.items()), key=lambda r: r['colos'])


def read_at(rows, target, key='death'):
    """Interpolate a metric at the target colonoscopy volume."""
    c = np.array([r['colos'] for r in rows]); v = np.array([r[key] for r in rows])
    if target < c.min() or target > c.max():
        return float('nan'), False
    return float(np.interp(target, c, v)), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--cap', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--lams', type=float, nargs='*',
                    default=[0.0011, 0.0014, 0.0018, 0.0023])
    a = ap.parse_args()
    t0 = time.time()
    cal = json.load(open(CACHE))
    target = headline_volume()
    levels = [(k, cal['sigmas'][k]['sigma'], cal['sigmas'][k]['auc'])
              for k in ('0.55', '0.60', '0.65', '0.70') if k in cal['sigmas']]
    levels += [('ceiling', 0.0, cal['ceiling_auc']),
               ('uninformative', CONTROL_SIGMA, control_auc())]
    out = {'target_volume': target, 'levels': {}}
    for key, sigma, auc_v in levels:
        tag = f'score{key.replace(".", "")}'
        print(f'== AUC {key} (sigma {sigma:.3f})', flush=True)
        rows = frontier(sigma, tag, a.lams, a.cap, a.rounds, a.workers)
        for r in rows:
            print(f"   lam={r['lam']:.6g} colos={r['colos']:.3f} death={r['death'] * 1e5:.0f} "
                  f"dx={r['inc'] * 1e5:.0f}", flush=True)
        d, ok = read_at(rows, target, 'death')
        i, _ = read_at(rows, target, 'inc')
        print(f"   at volume {target:.3f}: deaths {d * 1e5:.0f}, dx {i * 1e5:.0f}"
              f"{'' if ok else '  (EXTRAPOLATED - not reported)'}  ({time.time() - t0:.0f}s)", flush=True)
        out['levels'][key] = dict(sigma=sigma, auc=auc_v, rows=rows,
                                  death_at_target=d, inc_at_target=i, in_range=bool(ok))
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print('\nsaved', OUT)
    print(f"\n{'level':10s} {'AUC':>6s} {'deaths at matched volume':>26s} {'vs no score':>12s}")
    base = out['levels']['uninformative']['death_at_target']
    for key, d in out['levels'].items():
        v = d['death_at_target']
        print(f"{key:10s} {d['auc']:6.3f} {v * 1e5:26.0f} {(v - base) * 1e5:+12.0f}")


if __name__ == '__main__':
    main()
