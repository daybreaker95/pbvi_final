"""indivrisk_lambda_sweep_real_engine.py
=========================================
Lambda (colo_penalty_qaly) grid search through the REAL
NumberCrunching_policy engine, with the risk label taken straight from
CMOST's own `individual_risk` (top 20% -> "high_risk", rest -> "low_risk")
and DISCLOSED to the agent at t=0 -- see indivrisk_4way_eval.py's docstring
for both points, and jeon_lambda_sweep_real_engine.py for the
composite-score / latent-risk sweep this one is the counterpart to.

Same grid, N, seed and age window as the Jeon sweep (lambda 0.000-0.009,
N=200,000, seed=999, ages 40-80) so the two are directly comparable row by
row; trains against transitions_9state_sex_risk_top20pct.npz.

IMPORTANT: run this alone, not alongside other heavy background jobs (a
~15x CPU-contention slowdown was observed when four of these ran in
parallel previously).

Run: python3 tests/indivrisk_lambda_sweep_real_engine.py
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
from indivrisk_4way_eval import classify_individual_risk, train_all, RISK_NPZ

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

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
    risk_class, thr = classify_individual_risk(p, HIGH_FRAC)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    solvers = train_all(AGE_MIN, AGE_MAX, lam, RISK_NPZ)
    hook = C4.RiskDisclosedEngineHook(solvers, risk_class, sex_arr, seed=SEED)

    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p, N, SEED, hook, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    elapsed = time.time() - t0
    print(f'  colo={stats["avg_colonoscopies_per_person"]:.3f}  '
          f'death={stats["crc_death_per_100k"]:.1f}  inc={stats["incidence_per_100k"]:.1f}  '
          f'life_years={stats["life_years"]:.4f}  '
          f'({elapsed:.0f}s, total {time.time()-t_start:.0f}s)', flush=True)
    return dict(lam=lam, age_min=AGE_MIN, age_max=AGE_MAX, risk_npz=RISK_NPZ,
                risk_def='cmost_individual_risk_top20pct_disclosed',
                risk_threshold=thr, high_frac=HIGH_FRAC, seed=SEED,
                elapsed_sec=elapsed, **stats)


def main():
    t_start = time.time()
    rows = []
    out_path = os.path.join(RES, 'indivrisk_lambda_sweep_real_engine.json')

    print('--- no_screen baseline ---', flush=True)
    np.random.seed(SEED)
    p0 = C4.BNH.prepare_simulation_params(N)
    classify_individual_risk(p0, HIGH_FRAC)
    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p0, N, SEED, None, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    ns_stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    print(f'  death={ns_stats["crc_death_per_100k"]:.1f}  inc={ns_stats["incidence_per_100k"]:.1f}  '
          f'({time.time()-t0:.0f}s)', flush=True)
    baseline = dict(scenario='no_screen', seed=SEED, **ns_stats)
    with open(os.path.join(RES, 'indivrisk_lambda_sweep_baseline.json'), 'w') as f:
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
