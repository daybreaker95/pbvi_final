"""jeon_lambda_sweep_inrepo.py
==============================
Lambda (colo_penalty_qaly) grid search on THIS repo's own per-patient CMOST
engine (env/cmost_individual.CRCEngine), the in-repo counterpart of
tests/jeon_lambda_sweep_real_engine.py.

Same reason as tests/jeon_policy_inrepo_eval.py: the real-engine sweep needs
NumberCrunching_policy.py from the separate `cmost_experiment_final` tree,
which upstream CMOST does not ship. Numbers here are this engine's and are NOT
drop-in comparable with results/jeon_lambda_sweep_real_engine.json -- compare
lambdas to each other, and to the no_screen / q10y / q5y reference rows this
script computes on the SAME engine and cohort.

Lambda is a QALY shadow price on each colonoscopy: raising it buys fewer, more
selectively targeted colonoscopies. The point of the sweep is the frontier
between colonoscopy volume and mortality reduction, plus -- since the risk
class is an observed input here -- how the high/low allocation itself shifts as
the budget tightens.

Every row reuses the SAME cohort and the same engine seed, and
deterministic_natural_death pairs the competing-mortality clock, so
lambda-to-lambda differences come only from the screening decisions, not from
resampled patients.

Run alone, not alongside other heavy jobs (see
tests/nhic_lambda_sweep_real_engine.py's docstring: ~15x CPU-contention
slowdown was observed when that rule was violated).

Run: python tests/jeon_lambda_sweep_inrepo.py -n 200000
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from jeon_policy_inrepo_eval import (
    build_cohort, train_policy, run_arm, summarize, RISK_NPZ,
)

RES = os.path.join(PBVI_ROOT, 'results')
LAMBDAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]


def efficiency(row, ns):
    """Mortality reduction vs the no_screen baseline, and the marginal value of
    the colonoscopies spent getting it. per_colo_death_avoided_per_100k is the
    AVERAGE efficiency of this row's whole colonoscopy volume. The MARGINAL
    counterpart -- what the extra colonoscopies beyond the next-tighter lambda
    actually buy, which is what decides whether loosening the budget one more
    step is worth it -- is added after the sweep as
    marginal_per_colo_death_avoided_per_100k."""
    colo = row['colo_per_person']
    avoided = ns['crc_death_pct'] - row['crc_death_pct']
    row['crc_deaths_avoided_pct_pts'] = avoided
    row['mortality_reduction_pct'] = 100 * avoided / ns['crc_death_pct'] if ns['crc_death_pct'] else float('nan')
    row['per_colo_death_avoided_per_100k'] = 1000 * avoided / colo if colo > 1e-9 else float('nan')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--high-frac', type=float, default=0.20)
    ap.add_argument('--lambdas', type=str, default=','.join(str(x) for x in LAMBDAS))
    ap.add_argument('--latent-risk', action='store_true',
                    help='do NOT tell the agent its risk class (pre-observed-risk behaviour)')
    ap.add_argument('--tag', type=str, default='inrepo')
    a = ap.parse_args()
    observe_risk = not a.latent_risk
    lambdas = [float(x) for x in a.lambdas.split(',') if x.strip()]

    t_start = time.time()
    print(f'building cohort n={a.n:,} (Jeon-2018 profiles -> CMOST individual_risk) ...',
          flush=True)
    params, sex, mapped_risk, risk_class, thr = build_cohort(a.n, a.seed, a.high_frac)
    if not observe_risk:
        print('  (--latent-risk: that label is NOT given to the agent)', flush=True)

    common = dict(age_min=a.age_min, age_max=a.age_max, observe_risk=observe_risk)

    # ---- reference rows on the same engine + cohort -----------------------
    refs = {}
    for name, ages in (('no_screen', ()), ('q10y', (50, 60, 70)),
                       ('q5y', (50, 55, 60, 65, 70, 75))):
        print(f'--- reference: {name} ---', flush=True)
        t0 = time.time()
        m = run_arm(params, sex, mapped_risk, risk_class, a.seed,
                    'no_screen' if name == 'no_screen' else 'fixed',
                    screen_ages=ages, **common)
        refs[name] = summarize(m, risk_class)
        print(f'  colo={refs[name]["colo_per_person"]:.3f}  '
              f'crc_death={refs[name]["crc_death_pct"]:.3f}%  ({time.time()-t0:.0f}s)',
              flush=True)
    ns = refs['no_screen']
    for name in ('q10y', 'q5y'):
        efficiency(refs[name], ns)

    # ---- the sweep --------------------------------------------------------
    rows = []
    out_path = os.path.join(RES, f'jeon_lambda_sweep_{a.tag}.json')
    for lam in lambdas:
        print(f'--- lambda={lam:.4f} ---', flush=True)
        t0 = time.time()
        pomdps, solvers, gaps = {}, {}, {}
        for s, nm in ((1, 'male'), (2, 'female')):
            p, sol = train_policy(s, a.age_min, a.age_max, lam, observe_risk)
            pomdps[s], solvers[s] = p, sol
            gaps[nm] = sol.gap_history[-1]['gap']
        m = run_arm(params, sex, mapped_risk, risk_class, a.seed, 'policy',
                    pomdps=pomdps, solvers=solvers, **common)
        row = summarize(m, risk_class)
        row.update(lam=lam, gap_male=gaps['male'], gap_female=gaps['female'],
                   elapsed_sec=time.time() - t0)
        efficiency(row, ns)
        rows.append(row)
        print(f'  colo={row["colo_per_person"]:.3f} '
              f'(hi={row["colo_per_person_high_risk"]:.3f} '
              f'lo={row["colo_per_person_low_risk"]:.3f})  '
              f'crc_death={row["crc_death_pct"]:.3f}%  '
              f'mort_red={row["mortality_reduction_pct"]:.1f}%  '
              f'({time.time()-t0:.0f}s, total {time.time()-t_start:.0f}s)', flush=True)
        with open(out_path, 'w') as f:
            json.dump({'engine': 'env/cmost_individual.CRCEngine (in-repo)',
                       'n': a.n, 'seed': a.seed, 'high_frac': a.high_frac,
                       'risk_threshold': thr, 'observe_risk': observe_risk,
                       'risk_npz': RISK_NPZ, 'age_min': a.age_min, 'age_max': a.age_max,
                       'references': refs, 'sweep': rows,
                       'elapsed_sec': time.time() - t_start}, f, indent=2)

    # marginal efficiency BETWEEN adjacent lambdas (sorted by volume ascending):
    # how many deaths the extra colonoscopies of the looser budget actually buy.
    by_vol = sorted(rows, key=lambda r: r['colo_per_person'])
    for prev, cur in zip(by_vol, by_vol[1:]):
        dc = cur['colo_per_person'] - prev['colo_per_person']
        dd = prev['crc_death_pct'] - cur['crc_death_pct']
        cur['marginal_per_colo_death_avoided_per_100k'] = 1000 * dd / dc if dc > 1e-9 else float('nan')

    print()
    cols = ['lam', 'colo_per_person', 'colo_per_person_high_risk', 'colo_per_person_low_risk',
            'crc_death_pct', 'crc_death_pct_high_risk', 'crc_death_pct_low_risk',
            'mortality_reduction_pct', 'per_colo_death_avoided_per_100k']
    def short(c):
        return (c.replace('colo_per_person', 'colo').replace('crc_death_pct', 'death')
                 .replace('_high_risk', '_hi').replace('_low_risk', '_lo')
                 .replace('mortality_reduction_pct', 'mort_red%')
                 .replace('per_colo_death_avoided_per_100k', 'per_colo/100k'))
    print(''.join(f'{short(c):>16s}' for c in cols))
    for r in rows:
        print(''.join(f'{r[c]:16.4f}' for c in cols))
    print()
    for name in ('no_screen', 'q10y', 'q5y'):
        r = refs[name]
        print(f'{name:>10s}  colo={r["colo_per_person"]:.3f}  death={r["crc_death_pct"]:.3f}%  '
              f'mort_red={r.get("mortality_reduction_pct", 0.0):.1f}%  '
              f'per_colo={r.get("per_colo_death_avoided_per_100k", float("nan")):.1f}')

    # Two different questions, two different rows -- reported side by side
    # because picking on average efficiency alone is a trap: it always favours
    # the tightest budget (the colonoscopies that survive are the highest-yield
    # ones) even where total mortality is getting worse.
    eff = max(rows, key=lambda r: r['per_colo_death_avoided_per_100k'])
    low = min(rows, key=lambda r: r['crc_death_pct'])
    print(f'\nHighest average per-colonoscopy efficiency: lambda={eff["lam"]}  '
          f'per_colo={eff["per_colo_death_avoided_per_100k"]:.1f}/100k  '
          f'colo={eff["colo_per_person"]:.3f}  mort_red={eff["mortality_reduction_pct"]:.1f}%')
    print(f'Lowest CRC mortality:                       lambda={low["lam"]}  '
          f'death={low["crc_death_pct"]:.3f}%  '
          f'colo={low["colo_per_person"]:.3f}  mort_red={low["mortality_reduction_pct"]:.1f}%')
    print('The marginal column is what decides where to stop, but it divides '
          'small mortality differences by small volume differences -- check it '
          'is not just noise before reading a frontier off it.')

    with open(out_path, 'w') as f:
        json.dump({'engine': 'env/cmost_individual.CRCEngine (in-repo)',
                   'n': a.n, 'seed': a.seed, 'high_frac': a.high_frac,
                   'risk_threshold': thr, 'observe_risk': observe_risk,
                   'risk_npz': RISK_NPZ, 'age_min': a.age_min, 'age_max': a.age_max,
                   'references': refs, 'sweep': rows,
                   'elapsed_sec': time.time() - t_start}, f, indent=2)
    print(f'Saved {out_path}  (total {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
