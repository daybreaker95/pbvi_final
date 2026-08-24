"""Real-engine evaluation of screening arms with paired chunk seeds.

Arms are described by JSON-able dicts (see dp.engine_runner.build_hook):
  {'kind': 'none'}
  {'kind': 'fixed', 'ages': [50, 60, 70]}
  {'kind': 'policy', 'policy_male': ..., 'policy_female': ..., 'observed_class': False, 'class_thr': [...]}

python -m dp.evaluate --n 200000 --workers 6 --arms arms.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .common import RES
from .engine_runner import run_arm, aggregate, efficiency, paired_diff, arm_dir


def evaluate_arms(arms: dict, n_total: int, chunk=50_000, workers=6, verbose=True, age_min=40, age_max=80):
    """arms: tag -> arm dict. Returns tag -> rates (+ efficiency vs 'none')."""
    paths = {}
    for tag, arm in arms.items():
        if verbose:
            print(f'== arm {tag}: {arm}', flush=True)
        paths[tag] = run_arm(arm, tag, n_total, chunk=chunk, workers=workers, verbose=verbose,
                             age_min=age_min, age_max=age_max)
    # all arms must share the same number of chunks (paired seeds)
    out = {}
    base = aggregate(paths['none']) if 'none' in paths else None
    for tag, pth in paths.items():
        r = aggregate(pth)
        if base is not None and tag != 'none':
            r.update(efficiency(r, base))
            d, se = paired_diff(pth, paths['none'], 'crc_death')
            r['paired_death_diff_vs_none'] = d; r['paired_death_diff_se'] = se
            d, se = paired_diff(pth, paths['none'], 'diagnosed')
            r['paired_inc_diff_vs_none'] = d; r['paired_inc_diff_se'] = se
        out[tag] = r
    return out, paths


def summary_table(results: dict):
    lines = [f"{'arm':28s} {'colos':>6s} {'death/100k':>10s} {'±se':>5s} {'inc/100k':>9s} {'±se':>5s} "
             f"{'dRed%':>6s} {'iRed%':>6s} {'d/1000c':>8s} {'i/1000c':>8s} {'LYG/1000':>8s} {'comp':>5s}"]
    for tag, r in results.items():
        lines.append(f"{tag:28s} {r['colos_per_person']:6.3f} {r['crc_death_per_100k']:10.1f} {r['crc_death_se']:5.1f} "
                     f"{r['incidence_per_100k']:9.1f} {r['incidence_se']:5.1f} "
                     f"{r.get('death_reduction_pct', 0):6.1f} {r.get('incidence_reduction_pct', 0):6.1f} "
                     f"{r.get('deaths_averted_per_1000_colos', 0):8.3f} {r.get('cases_averted_per_1000_colos', 0):8.3f} "
                     f"{r.get('lyg_per_1000', 0):8.1f} {r['comp_death_per_100k']:5.1f}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True, help='JSON file: {tag: arm dict}')
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--chunk', type=int, default=50_000)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    arms = json.load(open(a.arms))
    res, _ = evaluate_arms(arms, a.n, chunk=a.chunk, workers=a.workers)
    print(summary_table(res))
    out = a.out or os.path.join(RES, f'eval_{os.path.splitext(os.path.basename(a.arms))[0]}_n{a.n}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out)


if __name__ == '__main__':
    main()
