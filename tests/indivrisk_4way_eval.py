"""indivrisk_4way_eval.py
=========================
Real-engine (NumberCrunching_policy.py) 4-way comparison of
no_screen / q10y / q5y / PBVI-policy, where the risk label handed to the
agent comes straight from CMOST's OWN `individual_risk` parameter:

    individual_risk >= 80th population percentile  ->  "high_risk"  (r=1)
    otherwise                                      ->  "low_risk"   (r=0)

Two things differ from jeon_4way_eval.py:

 1. RISK DEFINITION. jeon_4way_eval.py derives the label from the Jeon-2018
    12-factor composite score and then bucket-maps it back onto CMOST's
    individual_risk pool. Here the label IS individual_risk's own top-20%
    cut -- no composite score, no mapping step. The population 80th
    percentile of the real engine's pool is 3.6169798111920985, which is
    exactly the risk_threshold already stored in
    transitions_9state_sex_risk_top20pct.npz (built by transitions/
    estimate_transitions_9state_sex_risk.py --high_frac 0.20), so the
    policy is trained and evaluated on the same cut. The threshold is
    still recomputed from the actual drawn cohort at run time and checked
    against the npz's stored value -- a silent mismatch there would route
    individuals to the wrong risk block.

 2. RISK IS DISCLOSED, NOT INFERRED. SexAwareEngineHook starts every
    individual from the same frac_high population-prior belief and lets
    the agent infer risk from findings. RiskDisclosedEngineHook instead
    starts each individual at initial_belief(risk_class=r) -- all mass in
    their own block -- and routes them to a policy trained from that same
    risk-pure root (one per (sex, risk class), four in total). T and O are
    block-diagonal in risk class, so the belief never leaves that block:
    this is the "the agent is told high_risk / low_risk" information
    structure exactly, not an approximation.

Run: python3 tests/indivrisk_4way_eval.py --scenario policy -n 1000000
"""
import os
import sys
import json
import time
import argparse
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.join(PBVI_ROOT, 'cmost_engine'))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from cmost_4way_eval import (
    BNH, RiskDisclosedEngineHook, FixedScheduleHook,
    SurveillanceAwareFixedScheduleHook, train_policy, run_cohort, summarize,
)

RES = os.path.join(PBVI_ROOT, 'results')
RISK_NPZ = os.path.join(RES, 'transitions_9state_sex_risk_top20pct.npz')


def classify_individual_risk(p, high_frac=0.20, risk_npz=RISK_NPZ, verbose=True):
    """Top-`high_frac` of CMOST's own individual_risk -> 1 (high_risk),
    rest -> 0 (low_risk). Threshold taken from THIS cohort's empirical
    quantile, then cross-checked against the training npz's own stored
    risk_threshold (they agree to <1e-9 because individual_risk is drawn
    from the same discrete pool in both places)."""
    risk = np.asarray(p['individual_risk'], float)
    thr = float(np.quantile(risk, 1.0 - high_frac))
    risk_class = (risk >= thr).astype(int)
    if verbose:
        z = np.load(risk_npz, allow_pickle=True)
        thr_train = float(z['risk_threshold'])
        print(f'  individual_risk top-{high_frac:.0%} threshold = {thr:.10f}  '
              f'(training npz: {thr_train:.10f}, delta={thr - thr_train:+.2e})', flush=True)
        if abs(thr - thr_train) > 1e-6:
            print('  WARNING: threshold mismatch vs training npz -- risk_class '
                  'routing will not line up with the policy that was trained.', flush=True)
        sex = np.asarray(p['gender_arr']).astype(int)
        print(f'  frac_high overall={risk_class.mean():.4f}  '
              f'|male={risk_class[sex == 1].mean():.4f}  '
              f'|female={risk_class[sex == 2].mean():.4f}', flush=True)
    return risk_class, thr


def train_all(age_min, age_max, lam, risk_npz=RISK_NPZ, t0=None):
    """One FiVI policy per (sex, disclosed risk class) -- 4 in total,
    ~5s each, negligible next to the cohort run."""
    solvers = {}
    for sex in (1, 2):
        for r in (0, 1):
            tag = f'{"male" if sex == 1 else "female"}/{"high" if r else "low"}_risk'
            pom, sol = train_policy(sex=sex, age_min=age_min, age_max=age_max,
                                    sex_risk_npz=risk_npz, colo_penalty_qaly=lam,
                                    known_risk_class=r)
            gap = sol.gap_history[-1]['gap']
            el = f'  ({time.time() - t0:.1f}s)' if t0 else ''
            print(f'  trained {tag:<20s} gap={gap:.4f}{el}', flush=True)
            solvers[(sex, r)] = (pom, sol)
    return solvers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=['no_screen', 'q10y', 'q5y', 'policy'])
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--high-frac', type=float, default=0.20)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--tag', type=str, default='indivrisk20pct')
    ap.add_argument('--surveillance', action='store_true',
                    help='turn CMOST post-polypectomy/post-cancer surveillance ON. '
                         'Intended for the no_screen/q10y/q5y comparators, so they '
                         'represent real guideline practice (screening schedule PLUS '
                         'the follow-up it triggers) rather than screening alone. The '
                         'POMDP world model has no surveillance concept, so the policy '
                         'arm is normally left surveillance-off -- see README 5c.')
    a = ap.parse_args()

    t0 = time.time()
    # Single seeded draw of p, shared by hook construction AND the sim --
    # see run_cohort's docstring for why this must not be two draws.
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)

    print('Classifying risk from CMOST individual_risk ...', flush=True)
    risk_class, thr = classify_individual_risk(p, a.high_frac)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    if a.scenario == 'no_screen':
        hook = None
    elif a.scenario == 'q10y':
        hook = (SurveillanceAwareFixedScheduleHook([50, 60, 70], min_gap=10)
                if a.surveillance else FixedScheduleHook([50, 60, 70]))
    elif a.scenario == 'q5y':
        hook = (SurveillanceAwareFixedScheduleHook([50, 55, 60, 65, 70, 75], min_gap=5)
                if a.surveillance else FixedScheduleHook([50, 55, 60, 65, 70, 75]))
    else:  # policy
        print(f'training FiVI policies age={a.age_min}-{a.age_max} '
              f'risk_npz={RISK_NPZ} lam={a.lam} (risk DISCLOSED) ...', flush=True)
        solvers = train_all(a.age_min, a.age_max, a.lam, RISK_NPZ, t0)
        hook = RiskDisclosedEngineHook(solvers, risk_class, sex_arr, seed=a.seed)

    print(f'[{a.scenario}] running N={a.n:,} ...', flush=True)
    t1 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = run_cohort(
        p, a.n, a.seed, hook, hook_age_max=a.age_max, hook_age_min=a.age_min,
        surveillance=a.surveillance)
    stats = summarize(sr, ncr, money, number, tumor_record, death_year, a.n)
    print(f'[{a.scenario}] done ({time.time()-t1:.0f}s): {stats}', flush=True)

    suffix = f'_{a.tag}' if a.tag else ''
    out_path = os.path.join(RES, f'indivrisk_4way_{a.scenario}{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'scenario': a.scenario, 'n': a.n, 'seed': a.seed,
                   'age_min': a.age_min, 'age_max': a.age_max,
                   'risk_def': 'cmost_individual_risk_top20pct_disclosed',
                   'risk_npz': RISK_NPZ, 'risk_threshold': thr,
                   'high_frac': a.high_frac, 'lam': a.lam,
                   'surveillance': bool(a.surveillance),
                   'elapsed_sec': time.time() - t0, **stats}, f, indent=2)
    print('saved', out_path, flush=True)


if __name__ == '__main__':
    main()
