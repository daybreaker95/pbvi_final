"""
build_T_det_from_real_cmost.py
=================================
Rebuilds model_v2.py's T_detected (post-diagnosis annual mortality by
stage-at-diagnosis) DIRECTLY from the real CMOST engine, replacing the
CRCEngine+no-screening-only version (T_detected_tauphase.npz, built by
build_T_detected_from_tauphase.py in an earlier session).

Root cause found this session (tests/verify_T_det_vs_real_cmost.py, n=20,000
smoke test): the old T_det drastically UNDER-predicts CRC-specific mortality
for early stages (Stage I 5-year CRC-death: real ~13% vs T_det ~4.5%; Stage
II: real ~26% vs T_det ~8%). Two likely compounding causes:
  (a) data source -- CRCEngine, not the real (MATLAB-ported) engine.
  (b) methodology -- the old build propagated a PURE NO-SCREENING cohort to
      get the tau (quarters-since-diagnosis) mixture at each age. But under
      no-screening, Stage I/II diagnoses are RARE and unrepresentative (real
      Stage I/II detection is overwhelmingly screening-driven) -- so the old
      table's early-stage rows were estimated from a tiny, biased sample.

Fix: run a REAL CMOST cohort under a schedule that actually produces good
stage I-IV coverage (q5y -- screening AND symptom detection both occur), then
build T_det EMPIRICALLY and directly as a life-table: for each stage k and
calendar age a, "at risk" = patients diagnosed at stage k who are alive and
already diagnosed by the start of year a (dx_year <= a <= death_year); among
them, what fraction die of CRC / other causes / survive THIS year. This
naturally marginalizes over the REAL tau-mixture at each age -- no separate
tau-phase quarterly-matrix machinery needed, since we already have the
patient-level (dx_year, dx_stage, death_year, death_cause) tuples directly
from CMOST's own TumorRecord + DeathYear/DeathCause outputs.

Corrects for the DeathYear=0 survivor bug (found this session) by treating
DeathYear==0 as MAXY+1 / cause=0 (alive/censored).

Run: python build_T_det_from_real_cmost.py -n 1000000
"""
import os
import sys
import io
import argparse
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.join(PBVI_ROOT, 'tests'))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import cmost_4way_eval as C4

RES = os.path.join(PBVI_ROOT, 'results')
MAXY = 100
STAGE_CODES = {7: 0, 8: 1, 9: 2, 10: 3}
STAGE_NAMES = ['I', 'II', 'III', 'IV']


def run_and_extract(n, seed):
    print(f'[1] running real-CMOST q5y cohort (n={n:,}) ...', flush=True)
    hook = C4.FixedScheduleHook([50, 55, 60, 65, 70, 75])
    p = C4.BNH.prepare_simulation_params(n)
    p['flag']['Polyp_Surveillance'] = False
    p['flag']['Cancer_Surveillance'] = False
    sr = np.zeros((100, n), dtype=np.int8)
    ncr = np.zeros(n, dtype=np.int32)
    args_list = [p['p'], p['stage_variables'], p['location'], p['cost'], p['cost_stage'],
                 p['risc'], p['flag'], p['special_text'], p['female'], p['sensitivity'],
                 p['screening_test'], p['screening_preference'], p['age_progression'],
                 p['new_polyp'], p['colonoscopy_likelyhood'], p['individual_risk'],
                 p['risk_dist'], p['gender_arr'], p['life_table'], p['mortality_matrix'],
                 p['location_matrix'], p['stage_duration'], p['tx1'],
                 p['direct_cancer_rate'], p['direct_cancer_speed'], p['dwell_speed']]
    np.random.seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = C4.NumberCrunching_policy(*args_list, state_recorder=sr, policy_hook=hook,
                                         n_colo_recorder=ncr,
                                         policy_hook_age_min=40, policy_hook_age_max=85)
    DeathCause = out[2]
    DeathYear = out[4].copy()
    TumorRecord = out[12]

    survivor = DeathYear == 0
    DeathYear_corr = DeathYear.copy()
    DeathYear_corr[survivor] = MAXY + 1
    DeathCause_corr = DeathCause.copy()
    DeathCause_corr[survivor] = 0

    Stage = TumorRecord['Stage']
    PatNum = TumorRecord['PatientNumber']
    nrows = Stage.shape[0]

    dx_stage = np.zeros(n, dtype=np.int16)
    dx_year = np.zeros(n, dtype=np.int16)
    for yi in range(nrows):
        row_stage = Stage[yi]
        row_pat = PatNum[yi]
        m = (row_stage != 0) & (row_pat != 0)
        if not m.any():
            continue
        pats = row_pat[m].astype(np.int64) - 1
        stgs = row_stage[m].astype(np.int16)
        first_time = dx_stage[pats] == 0
        sel_p = pats[first_time]
        sel_s = stgs[first_time]
        dx_stage[sel_p] = sel_s
        dx_year[sel_p] = yi + 1

    return dx_stage, dx_year, DeathYear_corr, DeathCause_corr


def build_life_table(dx_stage, dx_year, death_year, death_cause):
    """Direct empirical life-table T_det[age-1][k] = [p_stay, p_crc, p_oth]."""
    T_det = np.zeros((MAXY, 4, 3))
    counts_used = np.zeros((MAXY, 4), dtype=np.int64)
    for code, k in STAGE_CODES.items():
        mask = dx_stage == code
        dxy = dx_year[mask]
        dy = death_year[mask]
        dc = death_cause[mask]
        for a in range(1, MAXY + 1):
            at_risk = (dxy <= a) & (dy >= a)
            n_risk = int(at_risk.sum())
            counts_used[a - 1, k] = n_risk
            if n_risk == 0:
                T_det[a - 1, k] = [1.0, 0.0, 0.0] if a == 1 else T_det[a - 2, k]
                continue
            died_crc = at_risk & (dy == a) & (dc == 2)
            died_oth = at_risk & (dy == a) & (dc == 1)
            p_crc = float(died_crc.sum()) / n_risk
            p_oth = float(died_oth.sum()) / n_risk
            p_stay = max(0.0, 1.0 - p_crc - p_oth)
            T_det[a - 1, k] = [p_stay, p_crc, p_oth]
    return T_det, counts_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=13579)
    args = ap.parse_args()

    dx_stage, dx_year, death_year, death_cause = run_and_extract(args.n, args.seed)

    n_dx_total = int((dx_stage != 0).sum())
    print(f'  total diagnosed: {n_dx_total:,}', flush=True)
    for code, k in STAGE_CODES.items():
        print(f'  stage {STAGE_NAMES[k]}: n_dx={(dx_stage==code).sum():,}', flush=True)

    print('[2] building empirical life-table T_det ...', flush=True)
    T_det, counts_used = build_life_table(dx_stage, dx_year, death_year, death_cause)

    out = os.path.join(RES, 'T_detected_realcmost.npz')
    np.savez_compressed(out, T_detected=T_det, stage_names=np.array(STAGE_NAMES),
                         n_patients=args.n, at_risk_counts=counts_used)
    print(f'Saved {out}', flush=True)

    print('\nPreview: T_det[age][stage] = [stay, CRC-death, other-death]  (n_at_risk)')
    for age in (55, 65, 75):
        for k, name in enumerate(STAGE_NAMES):
            print(f'  age {age} stage {name}: {np.round(T_det[age-1,k], 4)}  '
                  f'(n={counts_used[age-1,k]})')


if __name__ == '__main__':
    main()
