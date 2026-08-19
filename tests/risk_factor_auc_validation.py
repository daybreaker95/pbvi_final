"""risk_factor_auc_validation.py
=================================
Validates a Korean-epidemiology-based composite CRC risk score against
simulated natural-history outcomes, using the SAME logic as nihms864876.pdf
(Pandya et al. 2017, CVD PREDICT model validation): rank each individual by
a risk score, compare that ranking to whether they actually experienced the
outcome (there: observed NHANES mortality vs model-simulated mortality;
here: no "observed" arm exists for Korean CRC individuals, so this checks
model-simulated CRC death against the risk score alone -- i.e. "does the
risk score we're about to use for the top-10% cutoff actually discriminate
who develops/dies of CRC in this engine's own dynamics").

Pipeline
--------
1. Build a synthetic Korean-population cohort of N virtual clinical
   profiles: sex + 6 binary risk factors (family history, obesity, current
   smoking, physical inactivity, diabetes, high-risk drinking), each drawn
   Bernoulli at its own sex-specific Korean prevalence.
2. Composite relative risk = product of each present factor's HR
   (log-linear/multiplicative combination, the same functional form
   Framingham-style risk equations use -- see CRC_HR table below for every
   prevalence/HR value and its source).
3. Within each sex separately (individual_risk is drawn independent of
   gender in CMOST -- see estimate_transitions_9state_sex_risk.py's own
   docstring -- and the risk-class transition matrices assume an even ~10%
   split WITHIN each sex, so ranking must not pool across sexes), convert
   the composite relative risk to a percentile, then map that percentile
   onto CMOST's own individual_risk empirical distribution (quantile-
   quantile matching) -- this preserves CMOST's calibrated overall risk
   distribution while making WHO ends up high-risk driven by real factors.
4. Run each profile's OWN natural history (no screening at all) through
   env/cmost_individual.py's CRCEngine -- the same per-patient engine used
   to originally estimate the risk-stratified transition matrices, and the
   one CMOST built individual_risk for in the first place (it enters
   directly into the per-quarter polyp/cancer hazard).
5. AUC(composite risk score, simulated CRC death) -- the nihms864876-style
   validation. Also reports the observed CRC death rate in the (now
   risk-factor-driven) top-10% vs bottom-90% bucket, and compares against
   CMOST's own un-mapped individual_risk top-10%/bottom-90% split as a
   baseline for how much discriminative power the mapping ADDS.

Run: python3 tests/risk_factor_auc_validation.py
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from env.params import build_params
from env.cmost_individual import CRCEngine

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

# ---------------------------------------------------------------------------
# Risk factor table -- prevalence (Korean general population, sex-stratified
# where available) and colorectal-cancer HR, with sources noted inline.
# All factors modeled as binary present/absent.
# ---------------------------------------------------------------------------
# name: (prevalence_male, prevalence_female, HR, note)
RISK_FACTORS = {
    'family_history': (0.175, 0.175,
                        1.53,
                        'any first-degree relative; prevalence back-calculated via '
                        'Bayes rule from p(FH|patient)=25% & HR=1.53 (no direct Korean '
                        'general-population survey found)'),
    'obesity': (0.496, 0.277,
                1.25,
                'KNHANES 2022 obesity prevalence (Asian BMI>=25 cutoff); HR from '
                'Korean cohort (women BMI>=25 HR=1.25; used as single representative value)'),
    'current_smoker': (0.324, 0.063,
                        1.093,
                        'KDCA 2023 current-smoking rate; HR from Korean cohort (borderline)'),
    'physical_inactive': (1 - 0.544, 1 - 0.504,
                           1 / 0.67,
                           'KNHANES 2023 aerobic-activity rate -> inactive = 1-rate; '
                           'protective HR=0.67 (men, high intensity) inverted to a '
                           'risk-increasing HR for the "inactive" indicator'),
    'diabetes': (0.113, 0.120,
                 1.51,
                 'KDCA age-specific diabetes prevalence (M 40s/F 50s, used as a stand-in '
                 'for overall adult prevalence); HR from Korean Multi-center Cancer '
                 'Cohort (high fasting glucose vs normal)'),
    'high_risk_drinking': (0.199, 0.077,
                            2.24,
                            'KDCA 2023 high-risk-drinking rate; HR from Korean Multi-center '
                            'Cancer Cohort (>=30g/day)'),
}

N = 100_000
SEED = 20260809


def assign_profiles(n, seed):
    rng = np.random.default_rng(seed)
    sex = np.where(rng.random(n) < 0.5, 1, 2)  # 1=male, 2=female (CMOST convention)
    factors = {}
    for name, (p_m, p_f, hr, _note) in RISK_FACTORS.items():
        p = np.where(sex == 1, p_m, p_f)
        factors[name] = (rng.random(n) < p)
    log_rr = np.zeros(n)
    for name, (_p_m, _p_f, hr, _note) in RISK_FACTORS.items():
        log_rr += factors[name] * np.log(hr)
    composite_rr = np.exp(log_rr)
    return sex, factors, composite_rr


def percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool, rng):
    """Within each sex, rank by composite_rr -> percentile -> quantile of
    indiv_risk_pool (CMOST's own reference distribution, sex-independent)."""
    mapped = np.empty(len(composite_rr))
    for s in (1, 2):
        idx = np.where(sex == s)[0]
        ranks = composite_rr[idx].argsort().argsort()  # 0..len-1
        pct = (ranks + 0.5) / len(idx)  # avoid exact 0/1
        mapped[idx] = np.quantile(indiv_risk_pool, pct)
    return mapped


def simulate_natural_history(sex, mapped_risk, seed):
    params = build_params('CMOST13', n_patients=20000, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
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
        if (i + 1) % 20000 == 0:
            el = time.time() - t0
            print(f'    {i+1}/{n}  ({el:.0f}s, {1000*el/(i+1):.3f} ms/pt)', flush=True)
    return crc_death


def auc(score, outcome):
    """Mann-Whitney U based AUC -- avoids requiring sklearn."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank = sums / counts
    ranks = avg_rank[inv]
    n1 = outcome.sum()
    n0 = len(outcome) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r1 = ranks[outcome].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n0))


def roc_points(score, outcome):
    """Standard ROC curve points (fpr, tpr, threshold), thresholds swept
    over every distinct score value, descending -- no sklearn dependency."""
    order = np.argsort(-score)
    s_sorted = score[order]
    y_sorted = outcome[order]
    n1 = int(outcome.sum())
    n0 = len(outcome) - n1
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    # collapse to one point per distinct threshold (last occurrence of each tie group)
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tpr = tp[distinct] / n1
    fpr = fp[distinct] / n0
    thr = s_sorted[distinct]
    # prepend (0,0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thr = np.r_[np.inf, thr]
    return fpr, tpr, thr


def main():
    print('Assigning virtual clinical profiles ...', flush=True)
    sex, factors, composite_rr = assign_profiles(N, SEED)
    for name, (p_m, p_f, hr, note) in RISK_FACTORS.items():
        print(f'  {name:20s} HR={hr:.3f}  prevalence(M/F)={p_m:.3f}/{p_f:.3f}  observed={factors[name].mean():.3f}',
              flush=True)

    print('Loading CMOST individual_risk reference pool ...', flush=True)
    params = build_params('CMOST13', n_patients=200_000, seed=SEED + 2)
    indiv_risk_pool = np.asarray(params['individual_risk'], float)
    print(f'  pool: n={len(indiv_risk_pool)}  mean={indiv_risk_pool.mean():.3f}  '
          f'p90={np.quantile(indiv_risk_pool, 0.9):.3f}', flush=True)

    rng = np.random.default_rng(SEED + 3)
    mapped_risk = percentile_map_to_individual_risk(sex, composite_rr, indiv_risk_pool, rng)
    thr = {s: np.quantile(mapped_risk[sex == s], 0.9) for s in (1, 2)}
    is_high = np.array([mapped_risk[i] >= thr[sex[i]] for i in range(N)])
    print(f'  mapped top-10% thresholds: male={thr[1]:.3f}  female={thr[2]:.3f}  '
          f'frac_high(M/F)={is_high[sex==1].mean():.3f}/{is_high[sex==2].mean():.3f}', flush=True)

    print(f'Simulating natural history (no screening), N={N:,} ...', flush=True)
    crc_death = simulate_natural_history(sex, mapped_risk, SEED)
    print(f'  overall CRC death rate: {crc_death.mean()*100:.3f}%', flush=True)

    auc_composite = auc(composite_rr, crc_death)
    auc_mapped = auc(mapped_risk, crc_death)
    print(f'\nAUC(composite risk-factor score, simulated CRC death) = {auc_composite:.4f}')
    print(f'AUC(mapped individual_risk, simulated CRC death)      = {auc_mapped:.4f}  '
          f'(differs slightly from the above -- percentile mapping is rank-preserving '
          f'WITHIN each sex only, not globally, so pooled-population AUC can shift a bit; '
          f'this second number is the one that reflects what the simulator/policy actually sees)')

    print(f'\nCRC death rate, top-10%(risk-factor-mapped) vs bottom-90%:')
    print(f'  top-10%:    {crc_death[is_high].mean()*100:.3f}%  (n={is_high.sum()})')
    print(f'  bottom-90%: {crc_death[~is_high].mean()*100:.3f}%  (n={(~is_high).sum()})')
    rr_top_vs_bottom = crc_death[is_high].mean() / crc_death[~is_high].mean()
    print(f'  relative risk top vs bottom: {rr_top_vs_bottom:.2f}x')

    out_path = os.path.join(RES, 'risk_factor_auc_validation.json')
    import json
    with open(out_path, 'w') as f:
        json.dump({
            'n': N, 'seed': SEED,
            'auc_composite': auc_composite, 'auc_mapped': auc_mapped,
            'crc_death_rate_pct': float(crc_death.mean() * 100),
            'crc_death_rate_top10_pct': float(crc_death[is_high].mean() * 100),
            'crc_death_rate_bottom90_pct': float(crc_death[~is_high].mean() * 100),
            'rr_top_vs_bottom': float(rr_top_vs_bottom),
            'risk_factors': {k: dict(hr=v[2], prevalence_male=v[0], prevalence_female=v[1])
                              for k, v in RISK_FACTORS.items()},
        }, f, indent=2)
    print('saved', out_path)

    # -------- plotting data exports --------
    import csv

    raw_path = os.path.join(RES, 'risk_factor_auc_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'sex', 'composite_rr', 'mapped_individual_risk', 'is_top10pct', 'crc_death'])
        for i in range(N):
            w.writerow([i, int(sex[i]), f'{composite_rr[i]:.6f}', f'{mapped_risk[i]:.6f}',
                        int(is_high[i]), int(crc_death[i])])
    print('saved', raw_path, f'({N:,} rows -- one per simulated individual)')

    for score_name, score_arr in (('composite', composite_rr), ('mapped', mapped_risk)):
        fpr, tpr, thr = roc_points(score_arr, crc_death)
        roc_path = os.path.join(RES, f'risk_factor_auc_roc_{score_name}.csv')
        with open(roc_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['fpr', 'tpr', 'threshold'])
            for x, y, t in zip(fpr, tpr, thr):
                w.writerow([f'{x:.6f}', f'{y:.6f}', f'{t:.6f}' if np.isfinite(t) else 'inf'])
        print('saved', roc_path, f'({len(fpr)} points, AUC={auc(score_arr, crc_death):.4f})')


if __name__ == '__main__':
    main()
