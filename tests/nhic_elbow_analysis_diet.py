"""nhic_elbow_analysis_diet.py
================================
Extends nhic_elbow_analysis_noage.py with 7 dietary quartile factors
(fiber, calcium, folate, processed meat, red meat, fruit, vegetable) from
Jeon et al. 2018 (Gastroenterology 154(8):2152-2164.e19), Supplementary
Table S4 -- the same E-score source paper cited by Archambault et al. 2022.

Rationale: the NHIC (Shin 2014) 5-factor score alone gives only 69/16
distinct composite_rr values (male/female), causing severe tie-breaking
when percentile-mapped onto CMOST's 500-slot individual_risk pool. Jeon's
7 dietary factors are coded as sex-/study-specific quartiles (0,1,2,3,
self-normalizing to 25% each within any population by construction -- see
Jeon's own "Harmonization for E-score" section, which recodes to percentile
for exactly this between-population portability reason). Adding them
multiplies the achievable combination count by 4^7=16,384, resolving the
tie-breaking problem without needing Korean-specific quartile cutpoints
(quartile membership is self-normalizing; only the log-OR weights are
imported from the literature, not the absolute intake cutoffs).

Dietary log-OR weights (per-quartile, treated as continuous 0..3 exactly as
Jeon's own analysis did) are taken VERBATIM from Table S4, sex-specific.
Family history, BMI, glucose, cholesterol, alcohol are UNCHANGED from
nhic_elbow_analysis_noage.py (NHIC/Shin 2014, Korean cohort).

Run: python3 tests/nhic_elbow_analysis_diet.py [--n N]
"""
import os
import sys
import time
import json
import argparse
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from env.params import build_params
from env.cmost_individual import CRCEngine

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

# ---------------------------------------------------------------------------
# NHIC (Shin 2014), age-excluded -- identical to nhic_elbow_analysis_noage.py
# ---------------------------------------------------------------------------
BMI_HR = {'m': 1.13, 'f': 1.16}
GLUCOSE_HR = {'m': 1.10, 'f': 1.21}
CHOL_HR = {0: 1.00, 1: 1.10, 2: 1.16}
FAMILY_HISTORY_HR = {'m': 1.22, 'f': 1.18}
ALCOHOL_HR_M = {'none': 1.00, 'mild': 1.10, 'moderate': 1.21, 'heavy': 1.26}
ALCOHOL_HR_F = {'none': 1.00, 'mild': 1.00, 'moderate': 1.48, 'heavy': 1.48}

BMI_OBESE_PREV = {'m': 0.496, 'f': 0.277}
GLUCOSE_HIGH_PREV = {'m': 0.113, 'f': 0.120}
FAMILY_HISTORY_PREV = {'m': 0.175, 'f': 0.175}
ALCOHOL_PREV = {'none': 0.548, 'mild': 0.267, 'moderate': 0.110, 'heavy': 0.075}
CHOL_MEAN, CHOL_SD = 190.0, 38.7

# ---------------------------------------------------------------------------
# Jeon 2018, Table S4 -- per-quartile log(OR), sex-specific. Quartile 0 is
# the lowest-risk quartile by construction (footnote 3: "assigned values
# 0,1,2,3 in the order of increasing risk marginally"), so this table's sign
# convention already points toward risk regardless of the underlying
# biological direction.
# ---------------------------------------------------------------------------
DIET_BETA = {
    'm': {'fiber': 0.045, 'calcium': 0.043, 'folate': -0.020, 'processed_meat': 0.060,
          'red_meat': 0.056, 'fruit': -0.033, 'vegetable': 0.097},
    'f': {'fiber': 0.009, 'calcium': 0.063, 'folate': -0.023, 'processed_meat': -0.052,
          'red_meat': 0.163, 'fruit': -0.003, 'vegetable': 0.045},
}
DIET_FACTORS = ['fiber', 'calcium', 'folate', 'processed_meat', 'red_meat', 'fruit', 'vegetable']


def assign_profiles(n, seed):
    rng = np.random.default_rng(seed)
    sex = np.where(rng.random(n) < 0.5, 1, 2)   # 1=male, 2=female
    is_m = sex == 1

    bmi_obese = np.empty(n, dtype=bool)
    bmi_obese[is_m] = rng.random(int(is_m.sum())) < BMI_OBESE_PREV['m']
    bmi_obese[~is_m] = rng.random(int((~is_m).sum())) < BMI_OBESE_PREV['f']

    glucose_high = np.empty(n, dtype=bool)
    glucose_high[is_m] = rng.random(int(is_m.sum())) < GLUCOSE_HIGH_PREV['m']
    glucose_high[~is_m] = rng.random(int((~is_m).sum())) < GLUCOSE_HIGH_PREV['f']

    family_hist = rng.random(n) < FAMILY_HISTORY_PREV['m']

    tc = rng.normal(CHOL_MEAN, CHOL_SD, size=n)
    chol_cat = np.where(tc <= 200, 0, np.where(tc <= 239, 1, 2))

    alc_names = np.array(['none', 'mild', 'moderate', 'heavy'])
    alc_p = np.array([ALCOHOL_PREV[a] for a in alc_names])
    alcohol_cat = rng.choice(alc_names, size=n, p=alc_p)

    # 7 dietary quartiles, independent uniform 0..3 each (self-normalizing --
    # see module docstring). Real-world correlation between e.g. red meat and
    # fiber intake is not modeled; flagged as a limitation.
    diet_q = {f: rng.integers(0, 4, size=n) for f in DIET_FACTORS}

    log_rr = np.zeros(n)
    log_rr[is_m & bmi_obese] += np.log(BMI_HR['m'])
    log_rr[~is_m & bmi_obese] += np.log(BMI_HR['f'])
    log_rr[is_m & glucose_high] += np.log(GLUCOSE_HR['m'])
    log_rr[~is_m & glucose_high] += np.log(GLUCOSE_HR['f'])
    for c, hr in CHOL_HR.items():
        m = is_m & (chol_cat == c)
        log_rr[m] += np.log(hr)
    log_rr[is_m & family_hist] += np.log(FAMILY_HISTORY_HR['m'])
    log_rr[~is_m & family_hist] += np.log(FAMILY_HISTORY_HR['f'])
    for a, hr in ALCOHOL_HR_M.items():
        m = is_m & (alcohol_cat == a)
        log_rr[m] += np.log(hr)
    for a, hr in ALCOHOL_HR_F.items():
        m = (~is_m) & (alcohol_cat == a)
        log_rr[m] += np.log(hr)
    for f in DIET_FACTORS:
        log_rr[is_m] += DIET_BETA['m'][f] * diet_q[f][is_m]
        log_rr[~is_m] += DIET_BETA['f'][f] * diet_q[f][~is_m]

    composite_rr = np.exp(log_rr)
    out = dict(sex=sex, bmi_obese=bmi_obese, glucose_high=glucose_high,
               family_hist=family_hist, chol_cat=chol_cat, alcohol_cat=alcohol_cat,
               composite_rr=composite_rr)
    out.update({f'diet_{f}': diet_q[f] for f in DIET_FACTORS})
    return out


def percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool):
    mapped = np.empty(len(composite_rr))
    for s in (1, 2):
        idx = np.where(sex == s)[0]
        ranks = composite_rr[idx].argsort().argsort()
        pct = (ranks + 0.5) / len(idx)
        mapped[idx] = np.quantile(indiv_risk_pool, pct)
    return mapped


def simulate_natural_history(sex, mapped_risk, seed, eng):
    n = len(sex)
    crc_death = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient(gender=int(sex[i]), individual_risk=float(mapped_risk[i]))
        for y in range(1, 101):
            for q in (1, 2, 3, 4):
                if not pt.alive:
                    break
                eng._step_quarter(pt, y, q)
            if not pt.alive:
                break
        if (not pt.alive) and pt.death_cause == 2:
            crc_death[i] = True
        if (i + 1) % 100_000 == 0:
            el = time.time() - t0
            print(f'    {i+1:,}/{n:,}  ({el:.0f}s, {1000*el/(i+1):.4f} ms/pt, '
                  f'eta={el/(i+1)*(n-i-1):.0f}s)', flush=True)
    return crc_death


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=20260819)
    args = ap.parse_args()
    N = args.n
    SEED = args.seed

    print('Building params ...', flush=True)
    params = build_params('CMOST13', n_patients=200_000, seed=SEED + 2)
    indiv_risk_pool = np.asarray(params['individual_risk'], float)
    print(f'  individual_risk pool: n={len(indiv_risk_pool)}  unique={len(np.unique(indiv_risk_pool))}',
          flush=True)

    print(f'Assigning {N:,} NHIC+diet (age-excluded, 12-factor) virtual profiles ...', flush=True)
    prof = assign_profiles(N, SEED)
    sex, composite_rr = prof['sex'], prof['composite_rr']
    male = sex == 1
    print(f'  composite_rr (male):   mean={composite_rr[male].mean():.4f}  '
          f'p90={np.quantile(composite_rr[male],0.9):.4f}  '
          f'unique={len(np.unique(composite_rr[male]))}', flush=True)
    print(f'  composite_rr (female): mean={composite_rr[~male].mean():.4f}  '
          f'p90={np.quantile(composite_rr[~male],0.9):.4f}  '
          f'unique={len(np.unique(composite_rr[~male]))}', flush=True)

    mapped_risk = percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool)

    print(f'Simulating natural history (no screening), N={N:,} ...', flush=True)
    eng = CRCEngine(params, rng=np.random.default_rng(SEED + 1))
    crc_death = simulate_natural_history(sex, mapped_risk, SEED, eng)
    print(f'  overall CRC death rate: {crc_death.mean()*100:.3f}%', flush=True)

    print('\nCutoff sweep (1%p granularity, within-sex percentile of mapped_risk):', flush=True)
    rows = []
    baseline_rate = crc_death.mean()
    for pct in range(1, 51):
        frac = pct / 100.0
        is_high = np.zeros(N, dtype=bool)
        for s in (1, 2):
            idx = np.where(sex == s)[0]
            thr = np.quantile(mapped_risk[idx], 1 - frac)
            is_high[idx] = mapped_risk[idx] >= thr
        d_hi = crc_death[is_high].mean() if is_high.any() else float('nan')
        d_lo = crc_death[~is_high].mean() if (~is_high).any() else float('nan')
        rr = d_hi / d_lo if d_lo > 0 else float('nan')
        rows.append(dict(cutoff_pct=pct, n_high=int(is_high.sum()), frac_high=float(is_high.mean()),
                          death_rate_high_pct=float(d_hi * 100), death_rate_low_pct=float(d_lo * 100),
                          rr=float(rr)))
        print(f'  top{pct:3d}%  n_high={is_high.sum():7,d}  death_high={d_hi*100:.4f}%  '
              f'death_low={d_lo*100:.4f}%  RR={rr:.3f}', flush=True)

    out = dict(n=N, seed=SEED, baseline_death_rate_pct=float(baseline_rate * 100), sweep=rows)
    out_path = os.path.join(RES, 'nhic_elbow_sweep_diet.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)


if __name__ == '__main__':
    main()
