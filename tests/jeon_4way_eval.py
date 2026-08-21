"""jeon_4way_eval.py
====================
Real-engine (NumberCrunching_policy.py) 4-way comparison of
no_screen / q10y / q5y / PBVI-policy, with high-risk classification driven by
the Jeon-2018-only, 12-factor risk score (tests/jeon_elbow_analysis.py),
mapped onto CMOST's individual_risk pool via bucket_map_to_individual_risk
(binary high/low bucket draw, not continuous rank-preserving).

Counterpart to nhic_diet_4way_eval.py (which mixed NHIC-Korean core factors
with Jeon-Western diet factors, continuous rank mapping). Trains against
transitions_9state_sex_risk_jeon20pct.npz (built by transitions/
estimate_transitions_9state_jeon_risk.py).

Run: python3 tests/jeon_4way_eval.py --scenario policy -n 1000000
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
from jeon_elbow_analysis import assign_profiles, bucket_map_to_individual_risk
from pomdp.model_v2 import RISK_LABELS, risk_class_from_individual_risk

RES = os.path.join(PBVI_ROOT, 'results')
NHIC_NPZ = os.path.join(RES, 'transitions_9state_sex_risk_jeon20pct.npz')


def build_nhic_population(p, seed, high_frac=0.20):
    """Overrides p['individual_risk'] in place with Jeon-2018-only,
    bucket-mapped values for the exact population already drawn into p,
    returns per-sex-thresholded risk_class array. Function name kept as
    build_nhic_population for drop-in compatibility with
    nhic_lambda_sweep_real_engine.py's calling convention (only the import
    line changes between sibling _4way_eval modules)."""
    n = len(p['individual_risk'])
    indiv_risk_pool = np.asarray(p['individual_risk'], float)

    print(f'  assigning {n:,} Jeon-2018-only (12-factor) virtual profiles ...', flush=True)
    prof = assign_profiles(n, seed)
    composite_rr = prof['composite_rr']
    prof_sex = prof['sex']
    rng = np.random.default_rng(seed + 3)
    mapped_risk, risk_class_bool = bucket_map_to_individual_risk(
        prof_sex, composite_rr, indiv_risk_pool, high_frac, rng)

    p['individual_risk'] = mapped_risk
    risk_class = risk_class_bool.astype(int)

    # The label handed to the agent is EXACTLY the spec: the top high_frac of
    # the cohort by CMOST individual_risk is high_risk, the rest low_risk.
    #
    # That is very nearly -- but not exactly -- the composite-score bucket the
    # transition matrices were estimated under. The bucket mapping draws the
    # high bucket from CMOST's own top-high_frac sub-pool and the low bucket
    # from the rest, so the two ranges would be disjoint if the pool were
    # continuous. It is not: the pool holds ~476 distinct values, and the value
    # sitting on the (1-high_frac) cut appears ~21 times, straddling the split.
    # People who drew exactly that value therefore land in either bucket while
    # a single >= cut puts them all on one side. It affects ~0.06% of a cohort,
    # reported below rather than silently ignored -- if it ever grows beyond a
    # rounding-level share, the donor-pool split and the label have genuinely
    # diverged and the high/low dynamics stop matching the label.
    risk_class, thr = risk_class_from_individual_risk(mapped_risk, high_frac=high_frac)
    n_disagree = int((risk_class != risk_class_bool.astype(int)).sum())
    print(f'  risk label: individual_risk >= {thr:.6f} -> {RISK_LABELS[1]} '
          f'({risk_class.mean():.4f} of cohort), else {RISK_LABELS[0]}; '
          f'GIVEN TO the agent', flush=True)
    print(f'  (differs from the composite-score bucket on {n_disagree}/{n} = '
          f'{100 * n_disagree / n:.3f}% -- the tied value on the pool cut)', flush=True)
    if n_disagree > 0.01 * n:
        raise RuntimeError(
            f'label and composite-score bucket disagree on {100*n_disagree/n:.2f}% of the '
            f'cohort -- far beyond the tied-boundary-value level, so the top-'
            f'{high_frac:.0%}-by-individual_risk label no longer describes the split '
            f'the transition matrices were estimated under')

    male = prof_sex == 1
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
    ap.add_argument('--tag', type=str, default='jeon20pct')
    ap.add_argument('--latent-risk', action='store_true',
                     help='do NOT tell the agent its risk class -- start every belief '
                          'from the population prior (pre-observed-risk behaviour)')
    a = ap.parse_args()

    t0 = time.time()
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)

    print('Overriding individual_risk with Jeon-2018-only bucket-mapped values ...', flush=True)
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
                                          sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=a.lam,
                                          observe_risk=not a.latent_risk)
        print(f'  male gap={solver_m.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        print('training FiVI policy (female)...', flush=True)
        pomdp_f, solver_f = train_policy(sex=2, age_min=a.age_min, age_max=a.age_max,
                                          sex_risk_npz=NHIC_NPZ, colo_penalty_qaly=a.lam,
                                          observe_risk=not a.latent_risk)
        print(f'  female gap={solver_f.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)
        hook = SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class,
                                   sex_arr_nhic, seed=a.seed, observe_risk=not a.latent_risk)

    print(f'[{a.scenario}] running N={a.n:,} ...', flush=True)
    t1 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = run_cohort(
        p, a.n, a.seed, hook, hook_age_max=a.age_max, hook_age_min=a.age_min)
    stats = summarize(sr, ncr, money, number, tumor_record, death_year, a.n)
    print(f'[{a.scenario}] done ({time.time()-t1:.0f}s): {stats}', flush=True)

    suffix = f'_{a.tag}' if a.tag else ''
    out_path = os.path.join(RES, f'jeon_4way_{a.scenario}{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'scenario': a.scenario, 'n': a.n, 'seed': a.seed,
                    'age_min': a.age_min, 'age_max': a.age_max,
                    'high_frac': a.high_frac, 'lam': a.lam,
                    'observe_risk': not a.latent_risk,
                    'elapsed_sec': time.time() - t0, **stats}, f, indent=2)
    print('saved', out_path, flush=True)


if __name__ == '__main__':
    main()
