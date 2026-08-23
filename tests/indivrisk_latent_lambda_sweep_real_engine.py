"""indivrisk_latent_lambda_sweep_real_engine.py
===============================================
DISCLOSURE ABLATION for indivrisk_lambda_sweep_real_engine.py.

Identical in every respect to that sweep -- same population draw (labels =
CMOST `individual_risk` top 20%), same transitions_9state_sex_risk_top20pct
.npz, same lambda grid, N, seed and age window -- EXCEPT that the risk label
is LATENT rather than disclosed:

    disclosed (indivrisk_lambda_sweep_real_engine.py)
        RiskDisclosedEngineHook: individual starts at
        initial_belief(risk_class=r), routed to a policy trained from that
        risk-pure root. One policy per (sex x risk class).

    latent (THIS script)
        SexAwareEngineHook: every individual starts at the same
        frac_high population-prior belief and must infer its own risk
        class from colonoscopy findings. One policy per sex.

So the row-by-row difference between the two output files isolates the
value of telling the agent "high_risk"/"low_risk" up front, with nothing
else varying.

Why this ablation is the one that matters: the earlier jeon_* sweep differs
from the indivrisk_* sweep in BOTH the risk definition and the disclosure,
which looked like two confounded changes -- but it is not. jeon_elbow_
analysis.bucket_map_to_individual_risk assigns everyone labeled "high" an
individual_risk drawn from CMOST's OWN top-20% sub-pool, so the Jeon label
is a noiseless indicator of the risk stratum too, exactly like the native
top-20% label here. Both pipelines hand the agent the same single bit with
the same (zero) misclassification rate, and the two npz files encode the
same stratification strength (male Normal->EarlyPolyp high/low hazard
ratio 7.26 vs 7.02). The risk-definition change is therefore close to a
no-op, leaving disclosure as the only real variable -- which is what this
script measures directly instead of by cross-pipeline inference.

Run: python3 tests/indivrisk_latent_lambda_sweep_real_engine.py
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
from indivrisk_4way_eval import classify_individual_risk, RISK_NPZ

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

AGE_MIN = 40
AGE_MAX = 80
HIGH_FRAC = 0.20
N = 200_000
SEED = 999
LAMBDAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]


def run_one(lam, t_start):
    print(f'--- lambda={lam:.3f} (LATENT risk) ---', flush=True)
    np.random.seed(SEED)
    p = C4.BNH.prepare_simulation_params(N)
    risk_class, thr = classify_individual_risk(p, HIGH_FRAC)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    # one policy per sex, risk class left latent (known_risk_class=None)
    pomdp_m, solver_m = C4.train_policy(sex=1, age_min=AGE_MIN, age_max=AGE_MAX,
                                        sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam)
    pomdp_f, solver_f = C4.train_policy(sex=2, age_min=AGE_MIN, age_max=AGE_MAX,
                                        sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam)
    print(f'  trained male gap={solver_m.gap_history[-1]["gap"]:.4f}  '
          f'female gap={solver_f.gap_history[-1]["gap"]:.4f}', flush=True)
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f,
                                 risk_class, sex_arr, seed=SEED)

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
                risk_def='cmost_individual_risk_top20pct_latent',
                risk_threshold=thr, high_frac=HIGH_FRAC, seed=SEED,
                elapsed_sec=elapsed, **stats)


def main():
    t_start = time.time()
    rows = []
    out_path = os.path.join(RES, 'indivrisk_latent_lambda_sweep_real_engine.json')

    # no_screen baseline is identical to the disclosed sweep's (same
    # population draw, no policy involved) -- reuse it instead of re-running.
    ns_stats = json.load(open(os.path.join(RES, 'indivrisk_lambda_sweep_baseline.json')))
    print(f'reusing no_screen baseline: death={ns_stats["crc_death_per_100k"]:.1f}  '
          f'inc={ns_stats["incidence_per_100k"]:.1f}', flush=True)

    for lam in LAMBDAS:
        row = run_one(lam, t_start)
        colo = row['avg_colonoscopies_per_person']
        death_avtd = ns_stats['crc_death_per_100k'] - row['crc_death_per_100k']
        row['mortality_reduction_pct'] = death_avtd / ns_stats['crc_death_per_100k'] * 100
        row['per_colo_death_avoided'] = death_avtd / colo if colo > 1e-9 else float('nan')
        rows.append(row)
        with open(out_path, 'w') as f:
            json.dump(rows, f, indent=2)

    print(f'\nSaved {out_path}  (total {time.time()-t_start:.0f}s)')


if __name__ == '__main__':
    main()
