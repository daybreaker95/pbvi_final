"""
compare_9state_vs_cmost_sex.py
================================
Same as compare_9state_vs_cmost.py, but the matrix side uses the SEX x RISK
cross-stratified matrices (transitions_9state_sex_risk.npz, from
estimate_transitions_9state_sex_risk.py) instead of the sex-pooled ones --
i.e. exactly what pomdp/model_v2.py consumes when sex=1/2 is set.

Matrix-side population mix: male-low/high and female-low/high combined at
their TRUE simulated shares (frac_female, frac_high_male, frac_high_female
from the npz -- all ~0.50/0.25/0.25, i.e. "50:50" as requested, but using
the exact simulated values rather than a hardcoded 0.5).

CMOST side is UNCHANGED (default random gender per CMOST's own
fraction_female, ~50/50) -- this script only changes how the matrix side is
assembled.

Run: python compare_9state_vs_cmost_sex.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse, csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
from compare_9state_vs_cmost import (
    load_detection, run_microsim, summarize_microsim,
    run_cohort9_annual, combine_sexmix, print_row, RES,
)


def load_sex_risk():
    z = np.load(os.path.join(RES, 'transitions_9state_sex_risk.npz'), allow_pickle=True)
    ages = z['ages']
    out = {}
    for tag in ('male_low', 'male_high', 'female_low', 'female_high'):
        out[f'P_{tag}'] = {int(a): z[f'P_undetected_{tag}'][i] for i, a in enumerate(ages)}
        out[f'd_{tag}'] = {int(a): z[f'd_symp_{tag}'][i] for i, a in enumerate(ages)}
    out['frac_female'] = float(z['frac_female'])
    out['frac_high_male'] = float(z['frac_high_male'])
    out['frac_high_female'] = float(z['frac_high_female'])
    return out


def run_cohort9_sexmix(sr, T_det, d_EA, d_AA, d_pc, screen_ages):
    res = {}
    for tag in ('male_low', 'male_high', 'female_low', 'female_high'):
        res[tag] = run_cohort9_annual(sr[f'P_{tag}'], sr[f'd_{tag}'], T_det, d_EA, d_AA, d_pc, screen_ages)
    male = combine_sexmix(res['male_high'], res['male_low'],
                          w_male=sr['frac_high_male'], w_female=1 - sr['frac_high_male'])
    female = combine_sexmix(res['female_high'], res['female_low'],
                            w_male=sr['frac_high_female'], w_female=1 - sr['frac_high_female'])
    # combine_sexmix only reads '_raw' from its two args regardless of the
    # w_male/w_female arg *names* -- reuse it again for the sex combination
    return combine_sexmix(male, female, w_male=1 - sr['frac_female'], w_female=sr['frac_female'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=555)
    args = ap.parse_args()
    n = args.n

    sr = load_sex_risk()
    T_det_pooled = np.load(os.path.join(RES, 'T_detected_tauphase.npz'))['T_detected']
    d_EA, d_AA, d_pc = load_detection()
    print(f"sex-risk mix: frac_female={sr['frac_female']:.3f} "
          f"frac_high_male={sr['frac_high_male']:.3f} frac_high_female={sr['frac_high_female']:.3f}")

    eng = CRCEngine(build_params('CMOST13', 20000, args.seed), rng=np.random.default_rng(0))
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}

    rows_out = []
    for name, sa in scenarios.items():
        print(f"[microsim] {name} (n={n}) ...", flush=True)
        t0 = time.time()
        m = run_microsim(eng, n, sa, seed=args.seed)
        cm = summarize_microsim(m, n)
        # NOTE: T_det is reused pooled (CRC-hazard is gender-independent);
        # only the undetected-world dynamics (P_undet/d_symp) are sex-split.
        # The other-death competing-hazard sex split lives in T_det per
        # pomdp/model_v2.py's splice, but for THIS aggregate comparison the
        # pooled T_det's other-death (already the mixed life table) is the
        # correct thing to combine against a mixed CMOST population anyway.
        mx = run_cohort9_sexmix(sr, T_det_pooled, d_EA, d_AA, d_pc, sa)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

        print(f"\n================ {name} ================")
        print(f"{'metric':<24}{'CMOST':>14}{'9-state sex-mix':>16}{'diff':>10}")
        print('-' * 64)
        pr = lambda label, a, b, fmt='%.2f': print_row(label, a, b, fmt, rows_out, name)
        pr('Avg age at death', cm['avg_age_all'], mx['avg_age_all'])
        pr('Avg age at CRC death', cm['avg_age_crc'], mx['avg_age_crc'])
        pr('CRC deaths /100k', cm['crc_deaths_100k'], mx['crc_deaths_100k'], '%.0f')
        pr('Incidence /100k', cm['incid_100k'], mx['incid_100k'], '%.0f')
        pr('Screen-detected /100k', cm['screen_100k'], mx['screen_100k'], '%.0f')
        pr('Symptom-detected /100k', cm['symptom_100k'], mx['symptom_100k'], '%.0f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (all)', cm['stage_all_pct'][k], mx['stage_all_pct'][k], '%.1f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (screen)', cm['stage_scr_pct'][k], mx['stage_scr_pct'][k], '%.1f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (symptom)', cm['stage_sym_pct'][k], mx['stage_sym_pct'][k], '%.1f')

    out_csv = os.path.join(RES, 'compare_9state_vs_cmost_sex.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'metric', 'CMOST', 'matrix9state', 'diff_pct'])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out_csv}")


if __name__ == '__main__':
    main()
