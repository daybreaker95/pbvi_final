"""
compare_model_v2_vs_cmost.py
=============================
Same CMOST-direct-vs-matrix comparison as compare_9state_vs_cmost.py, but
propagates the cohort through pomdp/model_v2.py's OWN assembled M_undet
matrices (the 13-state undetected+detected axis, WAIT/SCREEN summed over
observations) instead of re-deriving a separate cohort-propagation from the
raw npz files. Purpose: compare_9state_vs_cmost.py validates the underlying
transition DATA (T_undetected/d_symp/T_detected); this script validates
that model_v2.py's ASSEMBLY of that data into the new 13-state POMDP
(detected Ca-stage states, risk-class blocks, screen/self-presentation
routing) doesn't introduce a wiring bug that the row-sum-to-1 checks alone
wouldn't catch.

Run: python compare_model_v2_vs_cmost.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse, csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
from pomdp.model_v2 import (
    CRCScreeningPOMDP9, NORMAL, CA_STAGES,
    CRC_DEATH, OTHER_DEATH, WAIT, SCREEN, NO,
)
from compare_9state_vs_cmost import run_microsim, summarize_microsim, print_row, RES

MAXY = 100


def run_cohort_modelv2(pomdp: CRCScreeningPOMDP9, screen_ages):
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(pomdp.NS)
    for r in range(pomdp.n_risk):
        w = pomdp.frac_high if r == 1 else (1 - pomdp.frac_high)
        dist[pomdp.fidx(r, NORMAL)] = w

    inc_scr = np.zeros(4); inc_sym = np.zeros(4)
    death_crc_age = 0.0
    death_oth_age = 0.0
    prev_crc = prev_oth = 0.0
    crc_idx = [pomdp.fidx(r, CRC_DEATH) for r in range(pomdp.n_risk)]
    oth_idx = [pomdp.fidx(r, OTHER_DEATH) for r in range(pomdp.n_risk)]

    d_pc = pomdp.d_PC_stage
    for age in range(pomdp.age_min, MAXY + 1):
        # (see caller: pomdp is built with age_min=1 so this loop matches
        # CMOST's own age-1 start -- natural history accumulates for
        # decades before any screen_ages entry becomes possible)
        a = SCREEN if age in screen_ages else WAIT
        M = pomdp.M[min(age, pomdp.life_max)][a]

        # Stage-at-diagnosis attribution is computed INDEPENDENTLY here
        # (mirrors model_v2._undet_MR's own p_diag_vec logic) rather than
        # read off M[O_CANCER]'s destination columns: _route_diag() now
        # splits same-year deaths straight into CRC_DEATH/OTHER_DEATH,
        # which would mix different source stages k together in those two
        # columns and make the stage breakdown unrecoverable from M alone.
        for r in range(pomdp.n_risk):
            dd = pomdp._d_symp[r]
            dsy = dd[min(max(age, min(dd)), max(dd))]
            for k, s in enumerate(CA_STAGES):
                src = pomdp.fidx(r, s)
                mass_here = dist[src]
                if mass_here <= 0:
                    continue
                p_diag_vec = np.asarray(dsy[s], dtype=float)
                if a == SCREEN:
                    inc_scr[k] += mass_here * d_pc[k]
                    miss_vec = (1 - d_pc[k]) * p_diag_vec
                    for kk in range(4):
                        if miss_vec[kk] > 0:
                            inc_sym[kk] += mass_here * miss_vec[kk]
                else:
                    for kk in range(4):
                        if p_diag_vec[kk] > 0:
                            inc_sym[kk] += mass_here * p_diag_vec[kk]

        Tfull = sum(M[o] for o in range(NO))
        nxt = dist @ Tfull

        cur_crc = nxt[crc_idx].sum()
        cur_oth = nxt[oth_idx].sum()
        age_c = age + 0.5
        death_crc_age += (cur_crc - prev_crc) * age_c
        death_oth_age += (cur_oth - prev_oth) * age_c
        prev_crc, prev_oth = cur_crc, cur_oth
        dist = nxt

    living = 1.0 - dist[crc_idx].sum() - dist[oth_idx].sum()
    death_oth_age += living * MAXY
    tot_crc = dist[crc_idx].sum()
    tot_oth = dist[oth_idx].sum() + living
    incid = inc_scr.sum() + inc_sym.sum()
    return {
        'avg_age_all': (death_oth_age + death_crc_age) / max(tot_oth + tot_crc, 1e-9),
        'avg_age_crc': death_crc_age / max(tot_crc, 1e-9),
        'crc_deaths_100k': 1e5 * tot_crc,
        'incid_100k': 1e5 * incid,
        'screen_100k': 1e5 * inc_scr.sum(),
        'symptom_100k': 1e5 * inc_sym.sum(),
        'stage_all_pct': 100 * (inc_scr + inc_sym) / max(incid, 1e-9),
        'stage_scr_pct': 100 * inc_scr / max(inc_scr.sum(), 1e-9),
        'stage_sym_pct': 100 * inc_sym / max(inc_sym.sum(), 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=777)
    args = ap.parse_args()
    n = args.n

    # age_min=1 (NOT the real screening-decision age_min=40) so M_undet
    # covers the full age-1..100 range CMOST itself simulates -- otherwise
    # the cohort would start "clean" at 40, missing decades of pre-40
    # natural-history buildup that CMOST's microsim actually has by then.
    pomdp = CRCScreeningPOMDP9(age_min=1, age_max=85, life_max=MAXY, gamma=0.97)
    d_EA, d_AA, d_pc = pomdp.d_EA, pomdp.d_AA, pomdp.d_PC_stage
    print(f"n_risk={pomdp.n_risk} NC={pomdp.NC} NS={pomdp.NS} T_det_source={pomdp._T_det_source}")

    eng = CRCEngine(build_params('CMOST13', 20000, args.seed), rng=np.random.default_rng(0))
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}

    rows_out = []
    for name, sa in scenarios.items():
        print(f"[microsim] {name} (n={n}) ...", flush=True)
        t0 = time.time()
        m = run_microsim(eng, n, sa, seed=args.seed)
        cm = summarize_microsim(m, n)
        mx = run_cohort_modelv2(pomdp, sa)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

        print(f"\n================ {name} ================")
        print(f"{'metric':<24}{'CMOST':>14}{'model_v2':>14}{'diff':>10}")
        print('-' * 62)
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

    out_csv = os.path.join(RES, 'compare_model_v2_vs_cmost.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'metric', 'CMOST', 'matrix9state', 'diff_pct'])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out_csv}")


if __name__ == '__main__':
    main()
