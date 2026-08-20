"""nhic_diet_4way_eval.py
====================
Real-engine (NumberCrunching_policy.py) 4-way comparison of
no_screen / q10y / q5y / PBVI-policy, with high-risk classification driven
by the age-excluded, 12-factor NHIC+diet risk score (tests/
nhic_elbow_analysis_diet.py: Shin 2014's 5 factors + Jeon 2018's 7 dietary
quartile factors) mapped onto CMOST's own individual_risk pool.

Counterpart to nhic_4way_eval.py, which used the age-INCLUDED 5-factor NHIC
score only. This version trains against
transitions_9state_sex_risk_nhicdiet20pct.npz (built by
transitions/estimate_transitions_9state_nhic_diet_risk.py), which used the
SAME assign_profiles/percentile_map_to_individual_risk procedure.

Run: python3 tests/nhic_diet_4way_eval.py --scenario policy -n 1000000
"""
import os
import sys
import io
import json
import time
import argparse
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CMOST_PY = os.path.join(PBVI_ROOT, 'cmost_engine')  # self-contained real-engine copy
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, CMOST_PY)
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9
from pomdp.fivi import FiVI

import build_natural_history_transition_matrix as BNH
from NumberCrunching_policy import NumberCrunching_policy

from cmost_4way_eval import (
    SexAwareEngineHook, FixedScheduleHook, train_policy, run_cohort, summarize,
)
from nhic_elbow_analysis_diet import assign_profiles, percentile_map_to_individual_risk

RES = os.path.join(PBVI_ROOT, 'results')
NHIC_NPZ = os.path.join(RES, 'transitions_9state_sex_risk_nhicdiet20pct.npz')


def build_nhic_population(p, seed, high_frac=0.20):
    """Overrides p['individual_risk'] in place with NHIC+diet-mapped values
    for the exact population already drawn into p, returns per-sex-
    thresholded risk_class array. See nhic_4way_eval.build_nhic_population
    for the rationale on using prof['sex'] as the authoritative per-person
    sex label throughout (kept identical here)."""
    n = len(p['individual_risk'])
    indiv_risk_pool = np.asarray(p['individual_risk'], float)

    print(f'  assigning {n:,} NHIC+diet (12-factor) virtual profiles ...', flush=True)
    prof = assign_profiles(n, seed)
    composite_rr = prof['composite_rr']
    prof_sex = prof['sex']
    mapped_risk = percentile_map_to_individual_risk(prof_sex, composite_rr, indiv_risk_pool)

    p['individual_risk'] = mapped_risk

    risk_class = np.zeros(n, dtype=int)
    male = prof_sex == 1
    for mask in (male, ~male):
        idx = np.where(mask)[0]
        thr = np.quantile(mapped_risk[idx], 1 - high_frac)
        risk_class[idx] = (mapped_risk[idx] >= thr).astype(int)

    print(f'  frac_high|male={risk_class[male].mean():.3f}  '
          f'frac_high|female={risk_class[~male].mean():.3f}', flush=True)
    return risk_class, prof_sex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['no_screen', 'q10y', 'q5y', 'policy'])
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--high-frac', type=float, default=0.20)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--tag', type=str, default='nhicdiet20pct')
    a = ap.parse_args()

    t0 = time.time()
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)

    print('Overriding individual_risk with NHIC+diet-mapped values ...', flush=True)
    risk_class, sex_arr_nhic = build_nhic_population(p, a.seed, a.high_frac)

    if a.scenario == 'no_screen':
        hook = None
    elif a.scenario == 'q10y':
        hook = FixedScheduleHook([50, 60, 70])
    elif a.scenario == 'q5y':
        hook = FixedScheduleHook([50, 55, 60, 65, 70, 75])
    else:  # policy
        print(f'training FiVI policy (male) age={a.age_min}-{a.age_max} '
              f'risk_npz={NHIC_NPZ} lam={a.lam} ...', flush=True)
        pomdp_m, solver_m = train_policy(sex=1, age_min=a.age_min, age_max=a.age_max,
                                          sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=a.lam)
        print(f'  male gap={solver_m.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        print('training FiVI policy (female)...', flush=True)
        pomdp_f, solver_f = train_policy(sex=2, age_min=a.age_min, age_max=a.age_max,
                                          sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=a.lam)
        print(f'  female gap={solver_f.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        hook = SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr_nhic, seed=a.seed)

    print(f'[{a.scenario}] running N={a.n:,} ...', flush=True)
    t1 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = run_cohort(
        p, a.n, a.seed, hook, hook_age_max=a.age_max, hook_age_min=a.age_min)
    stats = summarize(sr, ncr, money, number, tumor_record, death_year, a.n)
    print(f'[{a.scenario}] done ({time.time()-t1:.0f}s): {stats}', flush=True)

    suffix = f'_{a.tag}' if a.tag else ''
    out_path = os.path.join(RES, f'nhic_diet_4way_{a.scenario}{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'scenario': a.scenario, 'n': a.n, 'seed': a.seed,
                    'age_min': a.age_min, 'age_max': a.age_max,
                    'high_frac': a.high_frac, 'lam': a.lam,
                    'elapsed_sec': time.time() - t0, **stats}, f, indent=2)
    print('saved', out_path, flush=True)


if __name__ == '__main__':
    main()
