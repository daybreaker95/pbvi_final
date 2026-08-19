"""lambda_sweep_real_engine.py
==============================
Re-does the lambda grid search (lambda_sweep_fine.py) through the REAL
NumberCrunching_policy engine instead of the fast synthetic belief-
propagation simulator -- the synthetic sweep's ranking of lambda values is
not guaranteed to match the real engine's (already confirmed to diverge
for the top-10%-alone, lambda=0 case: synthetic showed a clear win over
q10y, the real engine showed a near-tie). Rather than trust the synthetic
ranking and validate only the single "winning" candidate, this runs the
WHOLE lambda grid through the real engine at a moderate N (200,000,
~7-8min/point based on the ~2200s/1,000,000 scaling already observed for
this engine) -- cheap enough to cover the full grid, expensive enough to
be trustworthy, same two-stage-of-rigor pattern used everywhere else in
this project.

Risk definition: top-10% only (results/transitions_9state_sex_risk.npz,
threshold 3.974249328225908) -- the synthetic sweep already showed top-25%
structurally weaker across the whole valid (colo >= q10y) lambda range, so
it's not worth doubling real-engine cost to re-confirm that here.

no_screen/q10y/q5y are NOT re-run -- they don't depend on the trained
policy at all, and final_algorithm/csv/02_policy_vs_schedules_final.csv's
own N=1,000,000 real-engine numbers for those three rows are already the
best available reference (same seed=999 convention used here).

Run: python3 tests/lambda_sweep_real_engine.py
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

RISK_NPZ = os.path.join(RES, 'transitions_9state_sex_risk.npz')
RISK_THRESHOLD = 3.974249328225908  # top-10%-high
AGE_MIN = 50
AGE_MAX = 75
N = 200_000
SEED = 999
LAMBDAS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]


def run_one(lam, t_start):
    print(f'--- lambda={lam:.3f} ---', flush=True)
    np.random.seed(SEED)
    p = C4.BNH.prepare_simulation_params(N)
    risk_class = (np.asarray(p['individual_risk']) >= RISK_THRESHOLD).astype(int)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    pomdp_m, solver_m = C4.train_policy(sex=1, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam)
    pomdp_f, solver_f = C4.train_policy(sex=2, age_min=AGE_MIN, age_max=AGE_MAX,
                                         sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam)
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=SEED)

    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(
        p, N, SEED, hook, hook_age_max=AGE_MAX, hook_age_min=AGE_MIN)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, N)
    elapsed = time.time() - t0
    print(f'  colo={stats["avg_colonoscopies_per_person"]:.3f}  '
          f'death={stats["crc_death_per_100k"]:.1f}  inc={stats["incidence_per_100k"]:.1f}  '
          f'({elapsed:.0f}s, total {time.time()-t_start:.0f}s)', flush=True)
    return dict(lam=lam, age_min=AGE_MIN, age_max=AGE_MAX, risk_npz=RISK_NPZ,
                risk_threshold=RISK_THRESHOLD, seed=SEED, elapsed_sec=elapsed, **stats)


def main():
    t_start = time.time()
    rows = []
    out_path = os.path.join(RES, 'lambda_sweep_real_engine.json')
    for lam in LAMBDAS:
        row = run_one(lam, t_start)
        rows.append(row)
        with open(out_path, 'w') as f:
            json.dump(rows, f, indent=2)
    print(f'\nSaved {out_path}  (total {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
