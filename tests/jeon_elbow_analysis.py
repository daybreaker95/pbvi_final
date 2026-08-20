"""jeon_elbow_analysis.py
=========================
All risk-factor HRs sourced from ONE paper -- Jeon et al. 2018 (Gastroenterology
154(8):2152-2164.e19, "Determining Risk of Colorectal Cancer and Starting Age
of Screening Based on Lifestyle, Environmental, and Genetic Factors") -- instead
of mixing NHIC (Shin 2014, Korean cohort) for BMI/glucose/cholesterol/family
history/alcohol with Jeon 2018 (GECCO/CORECT, mostly US + Germany/Australia/
Israel/Canada) for diet only, as tests/nhic_elbow_analysis_diet.py did.
Rationale (user request): source consistency -- a single Western-cohort paper
throughout, rather than a Korean-cohort core mixed with a Western-cohort diet
extension.

Factors and source (all Jeon 2018):
  - BMI            Table S4, continuous per 5kg/m2 -- approximated here as a
                    single obese(>=25)-vs-not step assuming ~1 unit (5kg/m2)
                    average gap between groups. This is a judgment call, not
                    a value taken verbatim from the table.
  - Diabetes        Table S4, binary yes/no (replaces NHIC's continuous
                    glucose>=126mg/dL threshold -- Jeon has no lab-glucose
                    variable, only self-reported diabetes diagnosis)
  - Alcohol         Table S4, 3-tier: <1g/day (abstinent) / 1-28g/day
                    (reference) / >28g/day (heavy) -- NOTE this reference
                    category is DIFFERENT from NHIC's (NHIC's reference was
                    "none"; Jeon's is "light-to-moderate", so abstainers
                    carry a HR>1 here, a J-shaped-curve artifact common in
                    alcohol-cancer epidemiology, not a data error)
  - Family history  Table S7 (a separate term alongside E-score in Jeon's own
                    model, not one of the 19 E-score factors), binary
  - 7 dietary       Table S4, unchanged from nhic_elbow_analysis_diet.py
    factors         (fiber, calcium, folate, processed meat, red meat,
                    fruit, vegetable), per-quartile (0,1,2,3), self-
                    normalizing 25% each within any population by
                    construction (Jeon's own quartile-recoding rationale)

Cholesterol is DROPPED entirely -- Jeon 2018 has no cholesterol variable.

Prevalences: obesity, diabetes-proxy, and family-history prevalences reuse
the Korean values from nhic_elbow_analysis_noage.py (population prevalence is
independent of which paper's HR estimate is used) since no better
Jeon-cohort-matched Korean data source was pulled for this pass. Alcohol
prevalence is a coarse remap of NHIC's 4-tier buckets onto Jeon's 3-tier
scheme (<1g/day := NHIC's 'none'; 1-28g/day := NHIC's 'mild'+'moderate';
>28g/day := NHIC's 'heavy') -- an approximation, not a Korean-specific
measurement of Jeon's exact g/day cutoffs.

Mapping onto CMOST's individual_risk pool uses bucket_map_to_individual_risk
(binary high/low bucket draw), NOT the continuous rank-preserving
percentile_map_to_individual_risk used by earlier NHIC scripts -- see that
function's docstring for the rationale (user wants the agent to see only a
high/low label, not a fine-grained rank).

Run: python3 tests/jeon_elbow_analysis.py [--n N] [--high_frac F]
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
# Jeon et al. 2018, Table S4 + Table S7 -- see module docstring for exact
# provenance of each value.
# ---------------------------------------------------------------------------
BMI_HR = {'m': 1.254, 'f': 1.113}          # approximated from per-5kg/m2 continuous coef.
DIABETES_HR = {'m': 1.108, 'f': 1.391}
ALCOHOL_HR = {                              # reference = 1-28 g/day
    'abstinent': {'m': 1.083, 'f': 1.127},
    'moderate':  {'m': 1.000, 'f': 1.000},
    'heavy':     {'m': 1.428, 'f': 1.107},
}
FAMILY_HISTORY_HR = {'m': 1.67, 'f': 1.46}  # Table S7, separate term

# Prevalences re-sourced to US/Western population statistics, matching the
# Jeon 2018 (Western-cohort) HRs above -- these previously reused Korean
# NHIC-cohort prevalences, a source mismatch.
#   BMI>=25, diabetes, alcohol: self-computed, weighted (WTMEC2YR),
#              sex-specific, from RAW NHANES August 2021-August 2023
#              microdata (DEMO_L/BMX_L/DIQ_L/ALQ_L, downloaded directly
#              from wwwn.cdc.gov as XPT files) -- ONE single survey cycle
#              for all three, not a patchwork of different years/surveys.
#              Diabetes = DIQ010==1 (doctor-diagnosed), matching Jeon's own
#              "self-reported type 2 diabetes" variable definition.
#              Alcohol g/day per respondent = (ALQ121 frequency category
#              converted to days/week) x (ALQ130 drinks on a drinking day)
#              x 14g ethanol/standard US drink, then bucketed at Jeon's
#              exact <1 / 1-28 / >28 g/day boundaries.
#   Family history: NHANES has no family-history-of-cancer question in any
#              cycle (confirmed by inspecting the Aug2021-2023 Medical
#              Conditions file, MCQ_L -- no such variable exists), so this
#              stays at Jeon 2018's own validation-cohort controls value
#              (Table S3), cross-checked against independent HPFS (men,
#              9.4%) / NHS (women, 10.0%) cohort estimates and found
#              consistent.
BMI_OBESE_PREV = {'m': 0.736, 'f': 0.693}
DIABETES_PREV = {'m': 0.120, 'f': 0.107}
FAMILY_HISTORY_PREV = {'m': 0.100, 'f': 0.130}
ALCOHOL_PREV_M = {'abstinent': 0.275, 'moderate': 0.599, 'heavy': 0.126}
ALCOHOL_PREV_F = {'abstinent': 0.424, 'moderate': 0.533, 'heavy': 0.043}

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

    diabetes = np.empty(n, dtype=bool)
    diabetes[is_m] = rng.random(int(is_m.sum())) < DIABETES_PREV['m']
    diabetes[~is_m] = rng.random(int((~is_m).sum())) < DIABETES_PREV['f']

    family_hist = np.empty(n, dtype=bool)
    family_hist[is_m] = rng.random(int(is_m.sum())) < FAMILY_HISTORY_PREV['m']
    family_hist[~is_m] = rng.random(int((~is_m).sum())) < FAMILY_HISTORY_PREV['f']

    alc_names = np.array(['abstinent', 'moderate', 'heavy'])
    alcohol_cat = np.empty(n, dtype=alc_names.dtype)
    alc_p_m = np.array([ALCOHOL_PREV_M[a] for a in alc_names])
    alc_p_f = np.array([ALCOHOL_PREV_F[a] for a in alc_names])
    alcohol_cat[is_m] = rng.choice(alc_names, size=int(is_m.sum()), p=alc_p_m)
    alcohol_cat[~is_m] = rng.choice(alc_names, size=int((~is_m).sum()), p=alc_p_f)

    diet_q = {f: rng.integers(0, 4, size=n) for f in DIET_FACTORS}

    log_rr = np.zeros(n)
    log_rr[is_m & bmi_obese] += np.log(BMI_HR['m'])
    log_rr[~is_m & bmi_obese] += np.log(BMI_HR['f'])
    log_rr[is_m & diabetes] += np.log(DIABETES_HR['m'])
    log_rr[~is_m & diabetes] += np.log(DIABETES_HR['f'])
    log_rr[is_m & family_hist] += np.log(FAMILY_HISTORY_HR['m'])
    log_rr[~is_m & family_hist] += np.log(FAMILY_HISTORY_HR['f'])
    for a in alc_names:
        m_male = is_m & (alcohol_cat == a)
        log_rr[m_male] += np.log(ALCOHOL_HR[a]['m'])
        m_fem = (~is_m) & (alcohol_cat == a)
        log_rr[m_fem] += np.log(ALCOHOL_HR[a]['f'])
    for f in DIET_FACTORS:
        log_rr[is_m] += DIET_BETA['m'][f] * diet_q[f][is_m]
        log_rr[~is_m] += DIET_BETA['f'][f] * diet_q[f][~is_m]

    composite_rr = np.exp(log_rr)
    out = dict(sex=sex, bmi_obese=bmi_obese, diabetes=diabetes,
               family_hist=family_hist, alcohol_cat=alcohol_cat,
               composite_rr=composite_rr)
    out.update({f'diet_{f}': diet_q[f] for f in DIET_FACTORS})
    return out


def percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool):
    """Continuous rank-preserving mapping (used ONLY for the diagnostic elbow
    sweep below -- the actual pipeline uses bucket_map_to_individual_risk).
    Cheap to sweep many cutoffs from a single simulation run because the
    mapped_risk value does not itself depend on any cutoff choice."""
    mapped = np.empty(len(composite_rr))
    for s in (1, 2):
        idx = np.where(sex == s)[0]
        ranks = composite_rr[idx].argsort().argsort()
        pct = (ranks + 0.5) / len(idx)
        mapped[idx] = np.quantile(indiv_risk_pool, pct)
    return mapped


def bucket_map_to_individual_risk(sex, composite_rr, indiv_risk_pool, high_frac, rng):
    """Binary bucket mapping: the top high_frac (by composite_rr, within sex)
    draws (with replacement) from CMOST's own top high_frac sub-pool; the
    rest draws from CMOST's remaining (1-high_frac) sub-pool. Unlike
    percentile_map_to_individual_risk (used by earlier NHIC scripts), this
    does NOT rank-match individuals 1:1 within a bucket -- it only encodes a
    binary high/low label, which is all the downstream POMDP agent is meant
    to observe. The donor-pool split fraction is the SAME high_frac used for
    the classification threshold (option "A" from conversation: a single
    fraction drives both which people are "high" and which CMOST sub-range
    they draw from)."""
    n = len(composite_rr)
    mapped = np.empty(n)
    is_high = np.zeros(n, dtype=bool)
    sorted_pool = np.sort(np.asarray(indiv_risk_pool, float))
    n_pool = len(sorted_pool)
    cut_idx = int(round(n_pool * (1 - high_frac)))
    low_pool = sorted_pool[:cut_idx]
    high_pool = sorted_pool[cut_idx:]
    for s in (1, 2):
        idx = np.where(sex == s)[0]
        thr = np.quantile(composite_rr[idx], 1 - high_frac)
        hi_mask = composite_rr[idx] >= thr
        hi_idx, lo_idx = idx[hi_mask], idx[~hi_mask]
        is_high[hi_idx] = True
        mapped[hi_idx] = rng.choice(high_pool, size=len(hi_idx), replace=True)
        mapped[lo_idx] = rng.choice(low_pool, size=len(lo_idx), replace=True)
    return mapped, is_high


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
    ap.add_argument('--high_frac', type=float, default=0.20)
    ap.add_argument('--sweep', action='store_true',
                     help='Diagnostic RR-vs-cutoff sweep (1-50%%) using continuous '
                          'percentile mapping -- cheap, one simulation, many cutoffs. '
                          'Finds the elbow BEFORE committing to a high_frac for the '
                          'bucket-mapped pipeline.')
    args = ap.parse_args()
    N, SEED, HIGH_FRAC = args.n, args.seed, args.high_frac

    print('Building params ...', flush=True)
    params = build_params('CMOST13', n_patients=200_000, seed=SEED + 2)
    indiv_risk_pool = np.asarray(params['individual_risk'], float)
    print(f'  individual_risk pool: n={len(indiv_risk_pool)}  unique={len(np.unique(indiv_risk_pool))}',
          flush=True)

    print(f'Assigning {N:,} Jeon-2018-only (12-factor) virtual profiles ...', flush=True)
    prof = assign_profiles(N, SEED)
    sex, composite_rr = prof['sex'], prof['composite_rr']
    male = sex == 1
    print(f'  composite_rr (male):   mean={composite_rr[male].mean():.4f}  '
          f'p90={np.quantile(composite_rr[male],0.9):.4f}  '
          f'unique={len(np.unique(composite_rr[male]))}', flush=True)
    print(f'  composite_rr (female): mean={composite_rr[~male].mean():.4f}  '
          f'p90={np.quantile(composite_rr[~male],0.9):.4f}  '
          f'unique={len(np.unique(composite_rr[~male]))}', flush=True)

    if args.sweep:
        mapped_risk = percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool)
        print(f'Simulating natural history (no screening), N={N:,} ...', flush=True)
        eng = CRCEngine(params, rng=np.random.default_rng(SEED + 1))
        crc_death = simulate_natural_history(sex, mapped_risk, SEED, eng)
        print(f'  overall CRC death rate: {crc_death.mean()*100:.3f}%', flush=True)

        print('\nCutoff sweep (1%p granularity, within-sex percentile of mapped_risk):', flush=True)
        rows = []
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
        out = dict(n=N, seed=SEED, baseline_death_rate_pct=float(crc_death.mean() * 100), sweep=rows)
        out_path = os.path.join(RES, 'jeon_elbow_sweep.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved {out_path}', flush=True)
        return

    rng = np.random.default_rng(SEED + 3)
    mapped_risk, is_high = bucket_map_to_individual_risk(sex, composite_rr, indiv_risk_pool, HIGH_FRAC, rng)
    print(f'  frac_high|male={is_high[male].mean():.3f}  frac_high|female={is_high[~male].mean():.3f}', flush=True)

    print(f'Simulating natural history (no screening), N={N:,} ...', flush=True)
    eng = CRCEngine(params, rng=np.random.default_rng(SEED + 1))
    crc_death = simulate_natural_history(sex, mapped_risk, SEED, eng)
    print(f'  overall CRC death rate: {crc_death.mean()*100:.3f}%', flush=True)

    d_hi = crc_death[is_high].mean()
    d_lo = crc_death[~is_high].mean()
    rr = d_hi / d_lo if d_lo > 0 else float('nan')
    print(f'  death_high={d_hi*100:.4f}%  death_low={d_lo*100:.4f}%  RR={rr:.3f}  (high_frac={HIGH_FRAC})', flush=True)

    out = dict(n=N, seed=SEED, high_frac=HIGH_FRAC,
               baseline_death_rate_pct=float(crc_death.mean() * 100),
               death_rate_high_pct=float(d_hi * 100), death_rate_low_pct=float(d_lo * 100), rr=float(rr))
    out_path = os.path.join(RES, 'jeon_elbow_bucket_check.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)


if __name__ == '__main__':
    main()
