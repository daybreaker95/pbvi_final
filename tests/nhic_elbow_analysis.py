"""nhic_elbow_analysis.py
=========================
Replaces the earlier (KCS-based) risk-factor composite score with the NHIC
model (Shin et al. 2014, PLOS ONE, "Risk Prediction Model for Colorectal
Cancer: National Health Insurance Corporation Study, Korea", Tables 1 & 2).

Why NHIC over KCS: NHIC has TRUE continuous age (HR 1.11/yr men, 1.08/yr
women) instead of KCS's 2-bucket age term, a much larger cohort
(846,559m+479,449f development), and higher C-statistics (0.762-0.786 men,
0.678-0.763 women vs KCS's 0.681). See Tables 1 (men) / 2 (women),
Colorectum (C18-C20) column, for every HR used below.

Men's colorectum model uses: age(cont.), BMI>=25, glucose>=126,
cholesterol(<=200/201-239/>=240), family history of cancer, alcohol
(0/1-14.9/15-24.9/>=25 g/day). [height, meat consumption also in the paper's
model but omitted here -- no Korean population prevalence data collected for
them yet; see conversation notes.]

Women's colorectum model uses: age(cont.), glucose>=126, family history of
cancer only -- Table 2 has no BMI, cholesterol, or alcohol row for the
Colorectum column in women (those are blank/dash there; alcohol and BMI only
appear in the Rectum/Right-colon submodels for women).

Prevalence sources:
  - BMI>=25, family history, diabetes(glucose proxy): reused from
    risk_factor_auc_validation.py's RISK_FACTORS table (same underlying
    KNHANES/KDCA figures used for the earlier KCS work).
  - Alcohol 4-tier (none/mild/moderate/heavy): Jeon et al. 2023, JAMA
    Network Open (jeon_2023_oi_221551_1674659687.27701.pdf), Table 1,
    2009 baseline, Korean NHIS cohort age>=40, N=3,933,382: none=54.8%,
    mild(<15g/d)=26.7%, moderate(15-29.9g/d)=11.0%, heavy(>=30g/d)=7.5%.
    (Men only -- women's colorectum model doesn't use alcohol.)
  - Cholesterol: no clean 3-tier Korean population breakdown found despite
    extensive search (medi-105-e48127.pdf / Yoo et al. 2026 gives only the
    population MEAN trend, not a categorical split). Approximated as
    Normal(mean=190, sd=38.7) and then bucketed into NHIC's own tiers
    (<=200 / 201-239 / >=240) -- mean from Yoo 2026 (2022-2023 overall
    population), sd from a KNHANES-based CVD-mortality cohort study
    (PMC6739006, baseline mean+-sd = 200.4+-38.7). FIRST-PASS APPROXIMATION,
    per explicit user instruction to "do it with the normal approx first and
    just look at the results."

"Classification age": each simulated individual gets ONE fixed age (drawn
from a life-table-weighted stationary distribution over 50-75, i.e. the age
at which they'd hypothetically get risk-scored in a clinic) used only to
evaluate the NHIC age term -- this is separate from the age variable inside
the full lifetime natural-history simulation, exactly as was done for the
KCS-based elbow analysis.

Run: python3 tests/nhic_elbow_analysis.py [--n N] [--quick]
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
# NHIC HR table (Shin et al. 2014, Table 1 = men, Table 2 = women; Colorectum
# C18-C20 column only)
# ---------------------------------------------------------------------------
AGE_HR_PER_YEAR = {'m': 1.11, 'f': 1.08}
BMI_HR = 1.13                 # men only; women's colorectum model has no BMI row
GLUCOSE_HR = {'m': 1.10, 'f': 1.21}
CHOL_HR = {0: 1.00, 1: 1.10, 2: 1.16}   # men only; 0<=200, 1=201-239, 2>=240
FAMILY_HISTORY_HR = {'m': 1.22, 'f': 1.18}
ALCOHOL_HR = {'none': 1.00, 'mild': 1.10, 'moderate': 1.21, 'heavy': 1.26}  # men only

# ---------------------------------------------------------------------------
# Prevalences
# ---------------------------------------------------------------------------
BMI_OBESE_PREV = {'m': 0.496, 'f': 0.277}          # KNHANES 2022 (Asian BMI>=25)
GLUCOSE_HIGH_PREV = {'m': 0.113, 'f': 0.120}        # KDCA diabetes prevalence proxy
FAMILY_HISTORY_PREV = {'m': 0.175, 'f': 0.175}      # Bayes back-calc (CRC-specific; NHIC
                                                     # def is "any cancer" -- likely a
                                                     # lower-bound approximation here)
ALCOHOL_PREV = {'none': 0.548, 'mild': 0.267, 'moderate': 0.110, 'heavy': 0.075}  # Jeon 2023
CHOL_MEAN, CHOL_SD = 190.0, 38.7

AGE_MIN, AGE_MAX = 50, 75


def life_table_age_weights(life_table, sex_idx):
    """Stationary-population weight per integer age in [AGE_MIN, AGE_MAX],
    proportional to survivorship l(age) = prod_{a=0}^{age-1}(1-q(a)) computed
    from CMOST's own annual all-cause mortality life table."""
    q = np.clip(np.asarray(life_table[:100, sex_idx], float), 0.0, 1.0)
    surv = np.r_[1.0, np.cumprod(1.0 - q)]  # surv[age] = P(alive at age)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    w = surv[ages]
    return ages, w / w.sum()


def assign_profiles(n, seed, life_table):
    rng = np.random.default_rng(seed)
    sex = np.where(rng.random(n) < 0.5, 1, 2)   # 1=male, 2=female (CMOST convention)
    is_m = sex == 1

    # Draw an integer "birth year" from the life-table-weighted stationary
    # population, then jitter uniformly within that year (age, age+1) --
    # NHIC's own Cox model treats age as a genuinely continuous covariate;
    # sampling only the integer year (as an earlier version of this script
    # did) needlessly collapsed the age term back down to 26 discrete
    # values, which for the women's model (only age+glucose+family_history)
    # left just 26*2*2=104 distinct composite-score values -- coarser than
    # CMOST's own 500-slot individual_risk pool and prone to the same
    # boundary tie-breaking problem the KCS score had. Jittering restores
    # the life-table's population weighting (still much denser near 50
    # than near 75) while making the age term -- and therefore the
    # composite score -- continuous.
    ages_m, w_m = life_table_age_weights(life_table, 0)
    ages_f, w_f = life_table_age_weights(life_table, 1)
    age = np.empty(n, dtype=float)
    age[is_m] = rng.choice(ages_m, size=int(is_m.sum()), p=w_m) + rng.random(int(is_m.sum()))
    age[~is_m] = rng.choice(ages_f, size=int((~is_m).sum()), p=w_f) + rng.random(int((~is_m).sum()))

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
    log_rr[is_m] += np.log(AGE_HR_PER_YEAR['m']) * (age[is_m] - AGE_MIN)
    log_rr[~is_m] += np.log(AGE_HR_PER_YEAR['f']) * (age[~is_m] - AGE_MIN)
    log_rr[is_m & bmi_obese] += np.log(BMI_HR)
    log_rr[is_m & glucose_high] += np.log(GLUCOSE_HR['m'])
    log_rr[~is_m & glucose_high] += np.log(GLUCOSE_HR['f'])
    for c, hr in CHOL_HR.items():
        m = is_m & (chol_cat == c)
        log_rr[m] += np.log(hr)
    log_rr[is_m & family_hist] += np.log(FAMILY_HISTORY_HR['m'])
    log_rr[~is_m & family_hist] += np.log(FAMILY_HISTORY_HR['f'])
    for a, hr in ALCOHOL_HR.items():
        m = is_m & (alcohol_cat == a)
        log_rr[m] += np.log(hr)

    composite_rr = np.exp(log_rr)
    return dict(sex=sex, age=age, bmi_obese=bmi_obese, glucose_high=glucose_high,
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
        if (i + 1) % 50_000 == 0:
            el = time.time() - t0
            print(f'    {i+1:,}/{n:,}  ({el:.0f}s, {1000*el/(i+1):.4f} ms/pt, '
                  f'eta={el/(i+1)*(n-i-1):.0f}s)', flush=True)
    return crc_death


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--seed', type=int, default=20260812)
    args = ap.parse_args()
    N = args.n
    SEED = args.seed

    print(f'Building params / life table ...', flush=True)
    params = build_params('CMOST13', n_patients=200_000, seed=SEED + 2)
    life_table = np.asarray(params['life_table'], float)
    indiv_risk_pool = np.asarray(params['individual_risk'], float)
    print(f'  individual_risk pool: n={len(indiv_risk_pool)}  unique={len(np.unique(indiv_risk_pool))}  '
          f'mean={indiv_risk_pool.mean():.3f}', flush=True)

    print(f'Assigning {N:,} NHIC-based virtual profiles ...', flush=True)
    prof = assign_profiles(N, SEED, life_table)
    sex, composite_rr = prof['sex'], prof['composite_rr']
    for name, arr in (('bmi_obese', prof['bmi_obese']), ('glucose_high', prof['glucose_high']),
                       ('family_hist', prof['family_hist'])):
        print(f'  observed {name}: {arr.mean():.3f}', flush=True)
    print(f'  observed chol tiers (<=200/201-239/>=240): '
          f'{(prof["chol_cat"]==0).mean():.3f}/{(prof["chol_cat"]==1).mean():.3f}/{(prof["chol_cat"]==2).mean():.3f}',
          flush=True)
    for a in ('none', 'mild', 'moderate', 'heavy'):
        print(f'  observed alcohol={a}: {(prof["alcohol_cat"]==a).mean():.3f}', flush=True)
    print(f'  composite_rr: mean={composite_rr.mean():.3f}  p90={np.quantile(composite_rr,0.9):.3f}  '
          f'p99={np.quantile(composite_rr,0.99):.3f}', flush=True)

    mapped_risk = percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool)

    print(f'Simulating natural history (no screening), N={N:,} ...', flush=True)
    eng = CRCEngine(params, rng=np.random.default_rng(SEED + 1))
    crc_death = simulate_natural_history(sex, mapped_risk, SEED, eng)
    print(f'  overall CRC death rate: {crc_death.mean()*100:.3f}%', flush=True)

    # -------- cutoff sweep: RR(top X%) vs (bottom (100-X)%), 1%p granularity --------
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
    out_path = os.path.join(RES, 'nhic_elbow_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)

    # simple elbow heuristic: point of max curvature drop in RR-vs-cutoff (largest
    # second derivative of RR wrt cutoff -- same style used for the KCS elbow)
    rr_arr = np.array([r['rr'] for r in rows])
    d1 = np.diff(rr_arr)
    d2 = np.diff(d1)
    elbow_idx = int(np.argmax(np.abs(d2))) + 2  # +2 to align with 2nd-diff offset
    print(f'Approx elbow at cutoff = top {rows[elbow_idx]["cutoff_pct"]}%  (RR={rows[elbow_idx]["rr"]:.3f})')


if __name__ == '__main__':
    main()
