"""
verify_T_det_vs_real_cmost.py
================================
model_v2.py's T_detected (post-diagnosis annual mortality by stage-at-
diagnosis) was built EARLIER (a prior session) via build_T_detected_from_
tauphase.py, using CRCEngine's already-validated 97-state tau-phase machinery
-- it has never been directly checked against the REAL CMOST engine's own
post-diagnosis survival, unlike the undetected side (this session's tau-phase
work). Root-causing the persistent matrix-vs-real ΔLYG gap (matrix predicts
~25.5 years gained per averted CRC death; real CMOST implies ~5.9) starts
here: does T_det's stage-conditional post-diagnosis survival curve match
what real CMOST actually produces?

Method: run a real-CMOST cohort under a mixed-detection schedule (q5y, so we
get both screen-detected early-stage and symptom-detected late-stage cases
in reasonable numbers), link each TumorRecord diagnosis event to that
patient's eventual DeathYear/DeathCause (out[4]/out[2]), and build empirical
cumulative CRC-death-by-N-years-post-diagnosis curves per stage. Compare
against T_det's own forward-propagated prediction for the same stage/age.

Corrects for the DeathYear=0 survivor bug found this session (patients alive
at end of study keep DeathYear=0) by treating DeathYear==0 as MAXY+1.

Run: python verify_T_det_vs_real_cmost.py -n 500000
"""
import os
import sys
import io
import argparse
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
import cmost_4way_eval as C4

RES = os.path.join(PBVI_ROOT, 'results')
MAXY = 100
STAGE_CODES = {7: 0, 8: 1, 9: 2, 10: 3}   # TumorRecord stage code -> 0-based k
STAGE_NAMES = ['I', 'II', 'III', 'IV']
HORIZONS = (1, 3, 5, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=500_000)
    ap.add_argument('--seed', type=int, default=2468)
    args = ap.parse_args()
    n = args.n

    print(f'[1] running real-CMOST q5y cohort (n={n:,}) for stage-linked diagnosis/death data ...', flush=True)
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
    np.random.seed(args.seed)
    with contextlib.redirect_stdout(io.StringIO()):
        out = C4.NumberCrunching_policy(*args_list, state_recorder=sr, policy_hook=hook,
                                         n_colo_recorder=ncr,
                                         policy_hook_age_min=40, policy_hook_age_max=85)
    DeathCause = out[2]
    DeathYear = out[4].copy()
    TumorRecord = out[12]
    print('  done, extracting diagnosis-linked survival data ...', flush=True)

    # DeathYear=0 survivor-bug correction (this session's finding)
    survivor = DeathYear == 0
    DeathYear_corr = DeathYear.copy()
    DeathYear_corr[survivor] = MAXY + 1
    DeathCause_corr = DeathCause.copy()
    DeathCause_corr[survivor] = 0   # 0 = alive/censored

    Stage = TumorRecord['Stage']       # (100, cols)
    PatNum = TumorRecord['PatientNumber']  # (100, cols), 1-based, 0 = empty slot
    nrows, ncols = Stage.shape

    # first diagnosis per patient (there should be at most one primary in this model,
    # but guard against duplicates by keeping the earliest row)
    dx_stage = np.zeros(n, dtype=np.int16)     # 0 = never diagnosed
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

    print('\n=== Real CMOST: empirical cumulative CRC-death-by-N-years-post-diagnosis, by stage ===')
    crc_hdrs = [('%dy_crc_pct' % h) for h in HORIZONS]
    oth_hdrs = [('%dy_oth_pct' % h) for h in HORIZONS]
    print(('{:<8}{:>8}' + '{:>12}' * len(HORIZONS) * 2).format('stage', 'n_dx', *crc_hdrs, *oth_hdrs))
    empirical = {}
    for code, k in STAGE_CODES.items():
        mask = dx_stage == code
        n_dx = int(mask.sum())
        if n_dx == 0:
            continue
        dyrs = DeathYear_corr[mask] - dx_year[mask]
        dcause = DeathCause_corr[mask]
        row_crc, row_oth = [], []
        for h in HORIZONS:
            crc_h = float(((dyrs <= h) & (dcause == 2)).mean() * 100)
            oth_h = float(((dyrs <= h) & (dcause == 1)).mean() * 100)
            row_crc.append(crc_h); row_oth.append(oth_h)
        empirical[k] = (n_dx, row_crc, row_oth)
        print(f"{STAGE_NAMES[k]:<8}{n_dx:>8}" + ''.join(f'{v:>10.2f}' for v in row_crc) + ''.join(f'{v:>10.2f}' for v in row_oth))

    # ---- matrix-side T_det prediction, propagated from the SAME dx ages ----
    T_det = np.load(os.path.join(RES, 'T_detected_tauphase.npz'))['T_detected']  # (MAXY,4,3)
    print('\n=== Matrix (T_det) prediction: same stage, weighted by REAL dx-age distribution ===')
    print(('{:<8}{:>8}' + '{:>12}' * len(HORIZONS) * 2).format('stage', 'n_dx', *crc_hdrs, *oth_hdrs))
    for code, k in STAGE_CODES.items():
        mask = dx_stage == code
        n_dx = int(mask.sum())
        if n_dx == 0:
            continue
        ages = dx_year[mask].astype(int)
        # average the per-patient forward-propagated curve over their true dx-age distribution
        crc_by_h = np.zeros(len(HORIZONS))
        oth_by_h = np.zeros(len(HORIZONS))
        uniq_ages, counts = np.unique(ages, return_counts=True)
        for a0, cnt in zip(uniq_ages, counts):
            alive, crc, oth = 1.0, 0.0, 0.0
            h_idx = 0
            for step in range(1, max(HORIZONS) + 1):
                age = min(a0 + step - 1, MAXY)
                p_stay, p_crc, p_oth = T_det[age - 1, k]
                crc += alive * p_crc
                oth += alive * p_oth
                alive *= p_stay
                if h_idx < len(HORIZONS) and step == HORIZONS[h_idx]:
                    crc_by_h[h_idx] += crc * cnt
                    oth_by_h[h_idx] += oth * cnt
                    h_idx += 1
        crc_by_h = crc_by_h / n_dx * 100
        oth_by_h = oth_by_h / n_dx * 100
        print(f"{STAGE_NAMES[k]:<8}{n_dx:>8}" + ''.join(f'{v:>10.2f}' for v in crc_by_h) + ''.join(f'{v:>10.2f}' for v in oth_by_h))


if __name__ == '__main__':
    main()
