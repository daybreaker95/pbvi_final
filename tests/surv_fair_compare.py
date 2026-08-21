"""surv_fair_compare.py
=======================
One arm of the "is the PBVI schedule more efficient than a fixed 10y/5y
schedule?" comparison, run through the real CMOST engine
(cmost_engine/NumberCrunching_policy.py), with post-diagnosis follow-up
either fully OFF or fully ON.

WHY BOTH SURVEILLANCE STREAMS MOVE TOGETHER
-------------------------------------------
CMOST has two follow-up streams, both gated by their own flag:

  Polyp_Surveillance   fires 3y after an advanced adenoma, 5y after any
                       adenoma, then every >=5y, keyed on Last_Polyp /
                       Last_AdvPolyp (NumberCrunching_policy.py ~line 1221)
  Cancer_Surveillance  fires 1y, 4y, then every >=5y after a resection,
                       keyed on Last_Cancer (~line 1236)

Both are triggered BY A FINDING, so both scale with how often an arm looks:
q5y removes more adenomas than q10y and therefore earns more post-
polypectomy follow-up, and it also diagnoses more cancers and therefore
earns more post-resection follow-up. Neither stream is exogenous.

That is why "cancer surveillance off, polyp surveillance on" is not the
fair split it looks like. It leaves the larger and more endogenous of the
two streams switched on, so the colonoscopy-budget axis that lambda prices
stops being a quantity any arm's decision rule controls -- and it matches
no real-world practice pattern either (nobody surveils adenomas but not
resections). The two coherent choices are:

  PRIMARY      both OFF. Every colonoscopy in the model is one the decision
               rule chose, so the budget axis is exactly "colonoscopies
               spent" and the POMDP's world model matches the simulator it
               is evaluated in. Isolates the scheduling rule.
  SENSITIVITY  both ON, identically in every arm. Total volume is now
               screening + follow-up, and arms are compared on that total.
               This is guideline practice, and it is the conservative test
               for the PBVI arm, whose world model has no surveillance
               concept and so is running mis-specified.

Note what neither flag touches: survival after diagnosis. CMOST draws
Detected_MortTime from stage at diagnosis alone, identically in every arm,
so "the process after cancer diagnosis" in the treatment/prognosis sense is
already symmetric. The flags only add colonoscopies.

KEEPING THE ARMS SYMMETRIC WITH SURVEILLANCE ON
-----------------------------------------------
Surveillance fires immediately before the policy decision inside the same
q==1 slot, so an arm that ignores it gets charged two colonoscopies in a
year where a coordinated arm is charged one. Both arm types are told about
external scopes through the new policy_hook.note_colonoscopy callback:

  fixed arms   min_gap = the schedule's own interval, i.e. a surveillance
               scope substitutes for one screening round (guideline
               convention: follow-up resets the screening clock)
  policy arm   screen_min_gap=1 (never two scopes in one year) plus
               react_to_external_colo, which folds the follow-up scope's
               finding into the belief. The policy's clock is its belief,
               not an interval, so it gets the minimal version of the rule.

RISK LABEL
----------
high_risk = top 20% of CMOST's own individual_risk, DISCLOSED to the agent
at t=0 (all initial belief mass on that risk block; the risk axis is
block-diagonal in T, so it stays there). This is a perfectly discriminating
label -- a real risk score is noisier, so any advantage measured here is an
upper bound on what a deployable score would deliver. The repo's Jeon-2018
composite score bucket-maps onto this same individual_risk pool, so it
reproduces nearly the same partition (~0.06% of a cohort differs).

Run one arm per process:
    python tests/surv_fair_compare.py --arm no_screen -n 500000
    python tests/surv_fair_compare.py --arm q10y      -n 500000 --surveillance
    python tests/surv_fair_compare.py --arm policy --lam 0.002 -n 500000
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
    BNH, SexAwareEngineHook, FixedScheduleHook, train_policy, run_cohort, summarize,
)

RES = os.path.join(PBVI_ROOT, 'results', 'survfair')
RISK_NPZ = os.path.join(PBVI_ROOT, 'results', 'transitions_9state_sex_risk_top20pct.npz')

SCHEDULES = {'q10y': ([50, 60, 70], 10), 'q5y': ([50, 55, 60, 65, 70, 75], 5)}


def classify_individual_risk(p, high_frac=0.20, verbose=True):
    """Top-`high_frac` of CMOST's own individual_risk -> 1 (high_risk).

    The threshold is taken from THIS cohort's empirical quantile and then
    cross-checked against the risk_threshold stored in the npz the policy
    trains on: a mismatch there would route individuals to the wrong risk
    block of the belief, silently, with no error."""
    risk = np.asarray(p['individual_risk'], float)
    thr = float(np.quantile(risk, 1.0 - high_frac))
    risk_class = (risk >= thr).astype(int)
    if verbose:
        thr_train = float(np.load(RISK_NPZ, allow_pickle=True)['risk_threshold'])
        print(f'  individual_risk top-{high_frac:.0%} threshold={thr:.10f} '
              f'(training npz {thr_train:.10f}, delta={thr - thr_train:+.2e})', flush=True)
        if abs(thr - thr_train) > 1e-6:
            print('  WARNING: threshold mismatch vs training npz', flush=True)
        print(f'  frac_high={risk_class.mean():.4f}', flush=True)
    return risk_class, thr


def build_hook(arm, lam, surveillance, risk_class, sex_arr, age_min, age_max, seed, t0):
    if arm == 'no_screen':
        return None
    if arm in SCHEDULES:
        ages, interval = SCHEDULES[arm]
        return FixedScheduleHook(ages, min_gap=interval if surveillance else None)
    # policy: one FiVI policy per sex, each expanded from BOTH risk-degenerate
    # start beliefs (observe_risk), each individual started on their own block
    print(f'training FiVI policies age={age_min}-{age_max} lam={lam} ...', flush=True)
    pm, sm = train_policy(sex=1, age_min=age_min, age_max=age_max,
                          sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam, observe_risk=True)
    print(f'  male   gap={sm.gap_history[-1]["gap"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    pf, sf = train_policy(sex=2, age_min=age_min, age_max=age_max,
                          sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam, observe_risk=True)
    print(f'  female gap={sf.gap_history[-1]["gap"]:.4f} ({time.time()-t0:.0f}s)', flush=True)
    return SexAwareEngineHook(pm, sm, pf, sf, risk_class, sex_arr, seed=seed,
                              observe_risk=True, screen_min_gap=1,
                              react_to_external_colo=bool(surveillance))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['no_screen', 'q10y', 'q5y', 'policy'])
    ap.add_argument('-n', type=int, default=500_000)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--high-frac', type=float, default=0.20)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--surveillance', action='store_true',
                    help='turn BOTH Polyp_Surveillance and Cancer_Surveillance on '
                         '(the sensitivity analysis); default is both off (primary)')
    ap.add_argument('--out', type=str, default=None)
    a = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    # ONE seeded draw of p, shared by hook construction and the simulation --
    # two draws would give the hook a risk_class/sex_arr for individual z that
    # does not describe the z the engine actually simulates (see run_cohort).
    np.random.seed(a.seed)
    p = BNH.prepare_simulation_params(a.n)
    risk_class, thr = classify_individual_risk(p, a.high_frac)
    sex_arr = np.asarray(p['gender_arr']).astype(int)

    hook = build_hook(a.arm, a.lam, a.surveillance, risk_class, sex_arr,
                      a.age_min, a.age_max, a.seed, t0)

    surv = 'surv' if a.surveillance else 'nosurv'
    label = a.arm if a.arm != 'policy' else f'policy_lam{a.lam:+.4f}'
    print(f'[{surv}/{label}] running N={a.n:,} ...', flush=True)
    t1 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = run_cohort(
        p, a.n, a.seed, hook, hook_age_max=a.age_max, hook_age_min=a.age_min,
        surveillance=a.surveillance)
    stats = summarize(sr, ncr, money, number, tumor_record, death_year, a.n,
                      risk_class=risk_class)
    print(f'[{surv}/{label}] done ({time.time()-t1:.0f}s) '
          f'colo={stats["avg_colonoscopies_per_person"]:.3f} '
          f'crc_death={stats["crc_death_per_100k"]:.1f} '
          f'ly={stats["life_years"]:.4f}', flush=True)

    out_path = a.out or os.path.join(
        RES, f'{surv}_{label}_n{a.n//1000}k_seed{a.seed}.json')
    with open(out_path, 'w') as f:
        json.dump({'arm': a.arm, 'lam': a.lam, 'surveillance': bool(a.surveillance),
                   'n': a.n, 'seed': a.seed, 'age_min': a.age_min, 'age_max': a.age_max,
                   'risk_def': 'cmost_individual_risk_top20pct_disclosed',
                   'risk_threshold': thr, 'high_frac': a.high_frac,
                   'risk_npz': RISK_NPZ, 'elapsed_sec': time.time() - t0,
                   **stats}, f, indent=2)
    print('saved', out_path, flush=True)


if __name__ == '__main__':
    main()
