"""compute_real_qaly.py
========================
Computes REAL (undiscounted) QALYs from the actual CMOST engine's own
annual state_recorder trajectory (sr), for no-screen vs the trained
policy -- something cmost_4way_eval.py's summarize() never did (only raw
clinical metrics: CRC deaths, incidence, stage%, cost, life_years). This
is the FIRST time the real engine's realization of the actual reward
objective (QALY) has been checked, as opposed to raw life-years.

Method: for each person i and year y=40..100, map sr[y-1,i] (18-state code)
to a utility value using the SAME age_weight/PHASE_UTIL_DECR tables the
POMDP reward uses:
  0-6   (Normal/EarlyPolyp/AdvPolyp)      -> age_weight(y)
  7-10  (U1-4, undetected cancer stage k) -> u_continuing(y, k)
  11-14 (D1-4, detected cancer stage k)   -> u_initial(y,k) in the person's
                                              FIRST year with a D-code,
                                              u_continuing(y,k) afterward
  death year (first year sr shows 15/16/17) -> half-cycle blend: 0.5 *
    living_utility_that_year + 0.5 * (u_term_crc if code==15 else
    u_term_other), matching the age+0.5 half-cycle convention already
    used elsewhere in this codebase (e.g. build_T_det*'s death_crc_age
    accumulation). No contribution for years after death.
Undiscounted (gamma=1 for reporting -- matches how life_years is already
reported undiscounted elsewhere; the POMDP's own gamma=0.97 is a training-
time-only Bellman discount, not part of the reported clinical metrics).

Run: python compute_real_qaly.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from pomdp.model_v2 import age_weight, u_initial, u_continuing, u_term_crc, u_term_other
import cmost_4way_eval as C4

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
YEARS = np.arange(1, 101)
AGE_W = np.array([age_weight(y) for y in YEARS])                     # (100,)
U_CONT = np.array([[u_continuing(y, k) for k in range(4)] for y in YEARS])   # (100,4)
U_INIT = np.array([[u_initial(y, k) for k in range(4)] for y in YEARS])      # (100,4)
U_TCRC = np.array([[u_term_crc(y, k) for k in range(4)] for y in YEARS])     # (100,4)
U_TOTH = np.array([[u_term_other(y, k) for k in range(4)] for y in YEARS])   # (100,4)


def qaly_from_sr(sr, discount=1.0):
    """sr: (100, n) int8, 18-state annual snapshot. Returns qaly (n,) float
    -- sum of per-year utility, ages 40..100 (or until death), each year's
    utility multiplied by discount**(age-40). discount=1.0 (default) is the
    original undiscounted behavior; discount=0.97 matches the POMDP's own
    training-time gamma (approx. the standard 3%/year health-economics
    convention: 1/1.03 = 0.9709), so training and reporting use the same
    discount rate."""
    n = sr.shape[1]
    qaly = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    # first year (1-based) each person shows a DETECTED code (11-14), else -1
    is_det = (sr >= 11) & (sr <= 14)
    first_det_row = np.where(is_det.any(axis=0), is_det.argmax(axis=0), -1)  # 0-based row idx

    for yi in range(39, 100):  # 0-based row for age 40..100
        y = yi + 1
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        s = sr[yi, idx]
        u = np.full(len(idx), AGE_W[yi])  # default: Normal/Polyp background
        is_u = (s >= 7) & (s <= 10)
        if is_u.any():
            k = s[is_u] - 7
            u[is_u] = U_CONT[yi, k]
        is_d = (s >= 11) & (s <= 14)
        if is_d.any():
            k = s[is_d] - 11
            just_dx = first_det_row[idx[is_d]] == yi
            u_d = np.where(just_dx, U_INIT[yi, k], U_CONT[yi, k])
            u[is_d] = u_d
        is_dead = s >= 15
        if is_dead.any():
            crc_mask = s[is_dead] == 15
            # terminal utility doesn't have a stage argument in u_term_* signature
            # here (it's per-stage in the POMDP because DET_CA_STAGES carries
            # its own stage k) -- for someone dying straight from an U/D state,
            # use that state's own stage; Normal/Polyp dying of "other" has no
            # stage, so fall back to stage-I decrement (smallest, background-
            # dominated) as a floor approximation.
            prev_s = sr[yi - 1, idx[is_dead]] if yi > 0 else np.zeros(is_dead.sum(), dtype=np.int8)
            stage_k = np.zeros(is_dead.sum(), dtype=int)
            has_stage = (prev_s >= 7) & (prev_s <= 14)
            stage_k[has_stage] = np.where(prev_s[has_stage] <= 10, prev_s[has_stage] - 7, prev_s[has_stage] - 11)
            term_u = np.where(crc_mask, U_TCRC[yi, stage_k], U_TOTH[yi, stage_k])
            u[is_dead] = 0.5 * u[is_dead] + 0.5 * term_u
            alive[idx[is_dead]] = False
        qaly[idx] += u * (discount ** (yi - 39))
    return qaly


def run(scenario, n, seed):
    # Single seeded p draw, shared by hook construction (risk_class/sex_arr)
    # AND the simulation itself -- see cmost_4way_eval.run_cohort's
    # docstring: this used to be two separate, uncorrelated draws (~50%
    # per-individual mismatch), so the policy hook's risk_class/sex label
    # for individual z did not correspond to individual z's actual
    # simulated risk/sex.
    np.random.seed(seed)
    p = C4.BNH.prepare_simulation_params(n)
    if scenario == 'no_screen':
        hook = None
    elif scenario == 'q10y':
        hook = C4.FixedScheduleHook([50, 60, 70])
    elif scenario == 'q5y':
        hook = C4.FixedScheduleHook([50, 55, 60, 65, 70, 75])
    else:  # policy -- one FiVI policy per sex, routed via CMOST's own
        # gender_arr (1=male, 2=female), matching CRCScreeningPOMDP9's own
        # sex=1/2 convention.
        pomdp_m, solver_m = C4.train_policy(sex=1)
        pomdp_f, solver_f = C4.train_policy(sex=2)
        risk_class = (np.asarray(p['individual_risk']) >= C4.RISK_THRESHOLD).astype(int)
        sex_arr = np.asarray(p['gender_arr']).astype(int)
        hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=seed)
    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(p, n, seed, hook)
    qaly = qaly_from_sr(sr)
    qaly_disc = qaly_from_sr(sr, discount=0.97)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, n)
    print(f'[{scenario}] n={n:,} qaly_mean={qaly.mean():.4f} qaly_disc3pct_mean={qaly_disc.mean():.4f} '
          f'({time.time()-t0:.0f}s)', flush=True)
    return {
        'qaly_mean': float(qaly.mean()), 'qaly_sd': float(qaly.std()),
        'qaly_disc3pct_mean': float(qaly_disc.mean()), 'qaly_disc3pct_sd': float(qaly_disc.std()),
        'life_years': stats['life_years'], 'life_years_disc3pct': stats['life_years_disc3pct'],
        'cost_per_person_usd': stats['cost_per_person_usd'],
        'cost_per_person_disc3pct_usd': stats['cost_per_person_disc3pct_usd'],
    }


def main():
    N = 300_000
    SEED = 999
    out = {}
    for scenario in ['no_screen', 'q10y', 'q5y', 'policy']:
        r = run(scenario, N, SEED)
        out[scenario] = {'n': N, **r}
    for scenario in ['q10y', 'q5y', 'policy']:
        out[f'delta_qaly_{scenario}_vs_noscreen'] = out[scenario]['qaly_mean'] - out['no_screen']['qaly_mean']
        out[f'delta_qaly_disc3pct_{scenario}_vs_noscreen'] = (
            out[scenario]['qaly_disc3pct_mean'] - out['no_screen']['qaly_disc3pct_mean'])
        out[f'delta_lyg_disc3pct_{scenario}_vs_noscreen'] = (
            out[scenario]['life_years_disc3pct'] - out['no_screen']['life_years_disc3pct'])
        out[f'delta_cost_disc3pct_{scenario}_vs_noscreen'] = (
            out[scenario]['cost_per_person_disc3pct_usd'] - out['no_screen']['cost_per_person_disc3pct_usd'])
        print(f"Delta QALY ({scenario} - no_screen) = {out[f'delta_qaly_{scenario}_vs_noscreen']:+.4f}  "
              f"disc3pct={out[f'delta_qaly_disc3pct_{scenario}_vs_noscreen']:+.4f}")
    with open(os.path.join(RES, 'real_qaly_compare.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', os.path.join(RES, 'real_qaly_compare.json'))


if __name__ == '__main__':
    main()
