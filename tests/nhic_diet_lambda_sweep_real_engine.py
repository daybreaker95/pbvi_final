"""nhic_diet_lambda_sweep_real_engine.py
=========================================
Lambda (colo_penalty_qaly) grid search through the REAL NumberCrunching_policy
engine, using the age-excluded, 12-factor NHIC+diet risk score (top-20%
within each sex; transitions_9state_sex_risk_nhicdiet20pct.npz) -- see
nhic_diet_4way_eval.py for the lambda=0 baseline this sweep extends, and
nhic_lambda_sweep_real_engine.py for the 5-factor precursor this replaces.

IMPORTANT: run this alone, not alongside other heavy background jobs (see
nhic_lambda_sweep_real_engine.py's docstring for the ~15x CPU-contention
slowdown observed when this rule was violated previously).

Run: python3 tests/nhic_diet_lambda_sweep_real_engine.py
"""
import os
import sys
import json
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

import cmost_4way_eval as C4
from nhic_diet_4way_eval import build_nhic_population, NHIC_NPZ

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

AGE_MIN = 40
AGE_MAX = 80
HIGH_FRAC = 0.20
N = 200_000
SEED = 999
LAMBDAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]


def run_one(lam, t_start):
    print(f'--- lambda={lam:.3f} ---', flush=True)
    np.random.seed(SEED)
    p = C4.BNH.prepare_simulation_params(N)
    risk_class, sex_arr = build_nhic_population(p, SEED, HIGH_FRAC)

    pomdp_m, solver_m = C4.train_policy(sex=1, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=lam)
    pomdp_f, solver_f = C4.train_policy(sex=2, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=lam)
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=SEED)

    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p, N, SEED, hook, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    elapsed = time.time() - t0
    print(f'  colo={stats["avg_colonoscopies_per_person"]:.3f}  '
          f'death={stats["crc_death_per_100k"]:.1f}  inc={stats["incidence_per_100k"]:.1f}  '
          f'life_years={stats["life_years"]:.4f}  '
          f'({elapsed:.0f}s, total {time.time()-t_start:.0f}s)', flush=True)
    return dict(lam=lam, age_min=AGE_MIN, age_max=AGE_MAX, risk_npz=NHIC_NPZ,
                high_frac=HIGH_FRAC, seed=SEED, elapsed_sec=elapsed, **stats)


def main():
    t_start = time.time()
    rows = []
    out_path = os.path.join(RES, 'nhic_diet_lambda_sweep_real_engine.json')

    print('--- no_screen baseline ---', flush=True)
    np.random.seed(SEED)
    p0 = C4.BNH.prepare_simulation_params(N)
    build_nhic_population(p0, SEED, HIGH_FRAC)
    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p0, N, SEED, None, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    ns_stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    print(f'  death={ns_stats["crc_death_per_100k"]:.1f}  inc={ns_stats["incidence_per_100k"]:.1f}  '
          f'({time.time()-t0:.0f}s)', flush=True)
    baseline = dict(scenario='no_screen', seed=SEED, **ns_stats)
    with open(os.path.join(RES, 'nhic_diet_lambda_sweep_baseline.json'), 'w') as f:
        json.dump(baseline, f, indent=2)

    for lam in LAMBDAS:
        row = run_one(lam, t_start)
        colo = row['avg_colonoscopies_per_person']
        death_avtd = ns_stats['crc_death_per_100k'] - row['crc_death_per_100k']
        row['mortality_reduction_pct'] = death_avtd / ns_stats['crc_death_per_100k'] * 100
        row['per_colo_death_avoided'] = death_avtd / colo if colo > 1e-9 else float('nan')
        rows.append(row)
        with open(out_path, 'w') as f:
            json.dump(rows, f, indent=2)

    best = max(rows, key=lambda r: r['per_colo_death_avoided'])
    print(f'\nBest lambda by per-colo death-avoided efficiency: '
          f'lambda={best["lam"]}  per_colo={best["per_colo_death_avoided"]:.1f}  '
          f'mort_red={best["mortality_reduction_pct"]:.1f}%  colo={best["avg_colonoscopies_per_person"]:.3f}')
    print(f'Saved {out_path}  (total {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
