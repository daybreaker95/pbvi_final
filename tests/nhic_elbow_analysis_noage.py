"""nhic_elbow_analysis_noage.py
================================
Age-excluded variant of nhic_elbow_analysis.py.

Rationale: CMOST's own engine already separates age-dependent risk from
individual-level risk -- env/cmost_individual.py's _step_quarter computes
`polyp_rate = pt.individual_risk * self.new_polyp[yi]`, where `new_polyp[yi]`
is CMOST's own age(year)-indexed onset-rate curve. `individual_risk` is
therefore meant to be an age-INDEPENDENT lifelong multiplier layered on top
of age-driven risk that CMOST already models elsewhere. Mapping an
age-INCLUSIVE NHIC score onto individual_risk double-counts age: once via
CMOST's native new_polyp[yi], again via the age-weighted composite score
itself (which additionally required an arbitrary "classification age"
sampled from a range that didn't match the simulation's own age window --
a symptom of this same conflation, not a separate bug).

This version drops NHIC's age term (HR 1.11/yr men, 1.08/yr women) entirely.
Removing age also removes the need for any "classification age" sampling.

Because NHIC's own women's Colorectum (overall) model only retained age,
glucose, and family history in stepwise selection (BMI/cholesterol/alcohol
were not statistically significant for women at that outcome), dropping age
would leave women with only glucose x family_history = 4 composite-score
values -- far coarser than CMOST's own 500-slot individual_risk pool and
prone to severe boundary tie-breaking (worse than the earlier KCS problem).
To fix this without inventing new risk factors, we borrow the SAME cohort's
(Shin et al. 2014) own subsite-specific coefficients for women where the
overall-Colorectum model is blank: BMI from the Right-colon submodel (the
only subsite where BMI was retained for women, HR 1.16) and alcohol from the
Rectum submodel (the only subsite where alcohol was retained for women, HR
1.00/1.48 for a 3-tier <15 / >=15 g/day split -- coarser than men's 4-tier
split because that is how NHIC's own women's Rectum table reports it).
Cholesterol has no retained coefficient for women in ANY subsite in the
source paper, so it is omitted entirely for women (not approximated).

Caveat carried forward explicitly (not resolved by this script): removing
age necessarily reduces real-world discrimination, since age is normally the
single strongest CRC risk predictor. NHIC's own reported C-statistics
(0.762-0.786 men, 0.678-0.763 women) are for the FULL model including age;
there is no equivalent age-excluded C-statistic available (would require the
original patient-level data, which we do not have). This script cannot
demonstrate that the age-excluded score retains meaningful real-world
discrimination -- see conversation notes on why an AUC/RR check against
CMOST's own simulated outcome cannot answer that question either (it is
circular: CMOST's simulated outcome depends only on the assigned
individual_risk value, not on which real-world characteristics produced that
rank). The RR-vs-cutoff sweep below tells us where the elbow sits GIVEN this
score's ranking, not whether the ranking itself is realistic.

Run: python3 tests/nhic_elbow_analysis_noage.py [--n N]
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
# NHIC HR table, AGE TERM REMOVED. Men: Colorectum (overall) column as-is.
# Women: Colorectum column for glucose/family_history; BMI borrowed from
# Right-colon submodel; alcohol borrowed from Rectum submodel (3-tier, not
# 4-tier -- that is how the source table splits it for women). Cholesterol:
# no subsite has a retained coefficient for women -- omitted, not imputed.
# ---------------------------------------------------------------------------
BMI_HR = {'m': 1.13, 'f': 1.16}                       # f: borrowed, Right colon
GLUCOSE_HR = {'m': 1.10, 'f': 1.21}                    # both: Colorectum (overall)
CHOL_HR = {0: 1.00, 1: 1.10, 2: 1.16}                  # men only; 0<=200,1=201-239,2>=240
FAMILY_HISTORY_HR = {'m': 1.22, 'f': 1.18}             # both: Colorectum (overall)
ALCOHOL_HR_M = {'none': 1.00, 'mild': 1.10, 'moderate': 1.21, 'heavy': 1.26}   # 4-tier
ALCOHOL_HR_F = {'none': 1.00, 'mild': 1.00, 'moderate': 1.48, 'heavy': 1.48}   # f: borrowed,
                                                        # Rectum, 3-tier (<15g/d=1.00,
                                                        # >=15g/d=1.48) collapsed onto the
                                                        # same 4-category prevalence buckets
                                                        # (moderate+heavy both >=15g/d)

# ---------------------------------------------------------------------------
# Prevalences (unchanged from the age-included version; population-average,
# not age-conditioned -- consistent with there being no classification age
# left to condition on).
# ---------------------------------------------------------------------------
BMI_OBESE_PREV = {'m': 0.496, 'f': 0.277}
GLUCOSE_HIGH_PREV = {'m': 0.113, 'f': 0.120}
FAMILY_HISTORY_PREV = {'m': 0.175, 'f': 0.175}
ALCOHOL_PREV = {'none': 0.548, 'mild': 0.267, 'moderate': 0.110, 'heavy': 0.075}
CHOL_MEAN, CHOL_SD = 190.0, 38.7


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

    family_hist = rng.random(n) < FAMILY_HISTORY_PREV['m']  # same for both sexes

    tc = rng.normal(CHOL_MEAN, CHOL_SD, size=n)
    chol_cat = np.where(tc <= 200, 0, np.where(tc <= 239, 1, 2))

    alc_names = np.array(['none', 'mild', 'moderate', 'heavy'])
    alc_p = np.array([ALCOHOL_PREV[a] for a in alc_names])
    alcohol_cat = rng.choice(alc_names, size=n, p=alc_p)

    log_rr = np.zeros(n)
    log_rr[is_m & bmi_obese] += np.log(BMI_HR['m'])
    log_rr[~is_m & bmi_obese] += np.log(BMI_HR['f'])
    log_rr[is_m & glucose_high] += np.log(GLUCOSE_HR['m'])
    log_rr[~is_m & glucose_high] += np.log(GLUCOSE_HR['f'])
    for c, hr in CHOL_HR.items():           # men only -- women have no term at all
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

    composite_rr = np.exp(log_rr)
    return dict(sex=sex, bmi_obese=bmi_obese, glucose_high=glucose_high,
                family_hist=family_hist, chol_cat=chol_cat, alcohol_cat=alcohol_cat,
                composite_rr=composite_rr)


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

    print(f'Assigning {N:,} NHIC-based (age-excluded) virtual profiles ...', flush=True)
    prof = assign_profiles(N, SEED)
    sex, composite_rr = prof['sex'], prof['composite_rr']
    for name, arr in (('bmi_obese', prof['bmi_obese']), ('glucose_high', prof['glucose_high']),
                       ('family_hist', prof['family_hist'])):
        print(f'  observed {name}: {arr.mean():.3f}', flush=True)
    print(f'  observed chol tiers (<=200/201-239/>=240): '
          f'{(prof["chol_cat"]==0).mean():.3f}/{(prof["chol_cat"]==1).mean():.3f}/{(prof["chol_cat"]==2).mean():.3f}',
          flush=True)
    for a in ('none', 'mild', 'moderate', 'heavy'):
        print(f'  observed alcohol={a}: {(prof["alcohol_cat"]==a).mean():.3f}', flush=True)
    male = sex == 1
    print(f'  composite_rr (male):   mean={composite_rr[male].mean():.3f}  '
          f'p90={np.quantile(composite_rr[male],0.9):.3f}  '
          f'unique={len(np.unique(composite_rr[male]))}', flush=True)
    print(f'  composite_rr (female): mean={composite_rr[~male].mean():.3f}  '
          f'p90={np.quantile(composite_rr[~male],0.9):.3f}  '
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
    out_path = os.path.join(RES, 'nhic_elbow_sweep_noage.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)


if __name__ == '__main__':
    main()
