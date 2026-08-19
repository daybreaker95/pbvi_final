"""lambda_sweep_real_engine_15_20.py
=====================================
Extends lambda_sweep_real_engine.py (which covered top-10% only) to also
cover top-15% and top-20%, requested directly against the real
NumberCrunching_policy engine (no synthetic pre-screening this time),
N=300,000 -- same age window (50-75) and lambda grid as the top-10% run,
so all three cutoffs are directly comparable against each other and
against the existing q10y/no_screen/q5y N=1,000,000 reference rows.

Run: python3 tests/lambda_sweep_real_engine_15_20.py
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

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

RISK_CONFIGS = {
    'top15': dict(npz=os.path.join(RES, 'transitions_9state_sex_risk_top15pct.npz'),
                   threshold=3.765004235854324),
    'top20': dict(npz=os.path.join(RES, 'transitions_9state_sex_risk_top20pct.npz'),
                   threshold=3.6169798111920985),
}

AGE_MIN = 50
AGE_MAX = 75
N = 300_000
SEED = 999
LAMBDAS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]


def run_one(risk_name, cfg, lam, t_start):
    print(f'--- {risk_name}  lambda={lam:.3f} ---', flush=True)
    np.random.seed(SEED)
    p = C4.BNH.prepare_simulation_params(N)
    risk_class = (np.asarray(p['individual_risk']) >= cfg['threshold']).astype(int)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    pomdp_m, solver_m = C4.train_policy(sex=1, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=cfg['npz'], colo_penalty_qaly=lam)
    pomdp_f, solver_f = C4.train_policy(sex=2, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=cfg['npz'], colo_penalty_qaly=lam)
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=SEED)

    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p, N, SEED, hook, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    elapsed = time.time() - t0
    print(f'  colo={stats["avg_colonoscopies_per_person"]:.3f}  '
          f'death={stats["crc_death_per_100k"]:.1f}  inc={stats["incidence_per_100k"]:.1f}  '
          f'({elapsed:.0f}s, total {time.time()-t_start:.0f}s)', flush=True)
    return dict(risk_def=risk_name, lam=lam, age_min=AGE_MIN, age_max=AGE_MAX,
                risk_npz=cfg['npz'], risk_threshold=cfg['threshold'], seed=SEED,
                elapsed_sec=elapsed, **stats)


def main():
    t_start = time.time()
    rows = []
    out_path = os.path.join(RES, 'lambda_sweep_real_engine_15_20.json')
    for risk_name, cfg in RISK_CONFIGS.items():
        for lam in LAMBDAS:
            row = run_one(risk_name, cfg, lam, t_start)
            rows.append(row)
            with open(out_path, 'w') as f:
                json.dump(rows, f, indent=2)
    print(f'\nSaved {out_path}  (total {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
