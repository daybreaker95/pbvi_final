"""nhic_4way_eval.py
====================
Real-engine (NumberCrunching_policy.py, the original MATLAB-ported CMOST)
4-way comparison of no_screen / q10y / q5y / PBVI-policy, with high-risk
classification driven by the NHIC (Shin et al. 2014) risk score mapped onto
CMOST's own individual_risk pool -- NOT CMOST's native random individual_risk.

This is the real-engine counterpart to tests/nhic_elbow_analysis.py (which
used the fast env/cmost_individual.py CRCEngine for natural-history-only
diagnostics). Here, ALL FOUR scenarios (including no_screen/q10y/q5y) share
ONE population whose individual_risk has been overridden with NHIC-mapped
values, so the underlying disease dynamics -- not just the classification
label -- reflect real Korean risk-factor-informed risk, matching the
methodology established across the whole NHIC risk-mapping analysis.

The PBVI policy is trained against transitions_9state_sex_risk_nhic20pct.npz
(built by transitions/estimate_transitions_9state_nhic_risk.py), which used
the SAME assign_profiles/percentile_map_to_individual_risk procedure -- so
the risk_class routing at evaluation time and the transition dynamics the
policy was trained on are consistent by construction.

Run: python3 tests/nhic_4way_eval.py -n 1000000
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
CMOST_PY = os.path.abspath(os.path.join(
    PBVI_ROOT, '..', '..', '..', '..', 'cmost_experiment_final', 'CMOST_experiment', 'python'))
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
from nhic_elbow_analysis import assign_profiles, percentile_map_to_individual_risk

RES = os.path.join(PBVI_ROOT, 'results')
NHIC_NPZ = os.path.join(RES, 'transitions_9state_sex_risk_nhic20pct.npz')


def build_nhic_population(p, seed, high_frac=0.20):
    """Overrides p['individual_risk'] in place with NHIC-mapped values for
    the exact population already drawn into p (same gender_arr, same n), and
    returns the per-sex-thresholded risk_class array."""
    n = len(p['individual_risk'])
    sex_arr = np.asarray(p['gender_arr']).astype(int)  # 1=male, 2=female
    indiv_risk_pool = np.asarray(p['individual_risk'], float)  # native pool, used as the mapping TARGET distribution

    print(f'  assigning {n:,} NHIC-based virtual profiles ...', flush=True)
    prof = assign_profiles(n, seed, np.asarray(p['life_table'], float))
    # assign_profiles draws its OWN sex draw internally; we need risk mapped
    # against THIS cohort's actual sex_arr (from prepare_simulation_params),
    # not a second independent draw -- recompute composite_rr using prof's
    # per-person factors but sex_arr's sex labels are what matters for
    # routing, and assign_profiles' own `sex` field is statistically
    # equivalent (same n, same per-sex prevalence) but not index-aligned to
    # sex_arr. Since composite_rr's sex-specific HR terms were computed
    # using prof['sex'], and mapping is done per-sex-group anyway, using
    # prof['sex'] consistently for both the composite score AND the mapping
    # (rather than mixing in sex_arr) keeps every person's own factors and
    # own sex label together -- so we override sex_arr's ROLE by just using
    # prof['sex'] as the authoritative per-person sex for the NHIC-risk
    # pipeline. p['gender_arr'] is left untouched (still used elsewhere,
    # e.g. mortality/cost accounting is gender-symmetric in this model).
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
    ap.add_argument('--tag', type=str, default='nhic20pct')
    a = ap.parse_args()

    t0 = time.time()
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)

    print('Overriding individual_risk with NHIC-mapped values ...', flush=True)
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
    out_path = os.path.join(RES, f'nhic_4way_{a.scenario}{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'scenario': a.scenario, 'n': a.n, 'seed': a.seed,
                    'age_min': a.age_min, 'age_max': a.age_max,
                    'high_frac': a.high_frac, 'lam': a.lam,
                    'elapsed_sec': time.time() - t0, **stats}, f, indent=2)
    print('saved', out_path, flush=True)


if __name__ == '__main__':
    main()
