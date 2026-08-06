"""
compare_cost_lyg_matrix_vs_cmost.py
======================================
Extends the matrix-vs-real-CMOST validation (compare_realengine_matrix_only.py,
CRC deaths/incidence/stage% only) with the two metrics it never covered:
per-person COST and avg-age-at-death (LYG). For no_screen/q10y/q5y (fixed
schedules), deterministic cohort propagation -- same loop structure as
compare_model_v2_vs_cmost.run_cohort_modelv2 -- but also accumulates cost
using the SAME cost-only reward mirror built for tests/compute_matrix_cost.py
(build_Rcost), so cost tracks exactly what the current (post-fix) reward
function would charge, with no MC noise.

Real-side numbers are read directly from the FINAL N=1,000,000
results/cmost_4way_{scenario}.json files (T_det/d_symp real-CMOST rebuild +
cost bugs fixed + DeathYear=0 correction all already baked in).

Run: python compare_cost_lyg_matrix_vs_cmost.py
"""
import os
import sys
import json
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, CA_STAGES, CRC_DEATH, OTHER_DEATH, WAIT, SCREEN, NO
from compare_model_v2_vs_cmost import MAXY
from compute_matrix_cost import build_Rcost

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def run_cohort_with_cost(pomdp, screen_ages, Rcost):
    """Deterministic cohort propagation (age 1..MAXY, matching
    run_cohort_modelv2's own age_min=1 convention for full lifetime
    coverage), tracking disease-state mass AND expected cost simultaneously.
    Returns avg_age_all (all-cause avg age at death) and cost_per_person."""
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(pomdp.NS)
    for r in range(pomdp.n_risk):
        w = pomdp.frac_high if r == 1 else (1 - pomdp.frac_high)
        dist[pomdp.fidx(r, NORMAL)] = w

    death_crc_age = death_oth_age = 0.0
    prev_crc = prev_oth = 0.0
    crc_idx = [pomdp.fidx(r, CRC_DEATH) for r in range(pomdp.n_risk)]
    oth_idx = [pomdp.fidx(r, OTHER_DEATH) for r in range(pomdp.n_risk)]
    cost_total = 0.0

    for age in range(pomdp.age_min, MAXY + 1):
        a = SCREEN if age in screen_ages else WAIT
        M = pomdp.M[min(age, pomdp.life_max)][a]
        Rc = Rcost[min(age, pomdp.life_max)][a]
        cost_total += float(dist @ Rc)

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
    avg_age_all = (death_oth_age + death_crc_age) / max(tot_oth + tot_crc, 1e-9)
    return avg_age_all, cost_total


def main():
    pomdp = CRCScreeningPOMDP9(age_min=1, age_max=85, life_max=MAXY, gamma=0.97)
    print(f"n_risk={pomdp.n_risk} frac_high={pomdp.frac_high:.4f}")
    Rcost = build_Rcost(pomdp)

    scenarios = {
        'No screening': ([], 'cmost_4way_no_screen.json'),
        'Screen q10y (50,60,70)': ([50, 60, 70], 'cmost_4way_q10y.json'),
        'Screen q5y (50-75)': ([50, 55, 60, 65, 70, 75], 'cmost_4way_q5y.json'),
    }

    rows = []
    print(f"\n{'scenario':<26}{'metric':<20}{'real':>12}{'matrix':>12}{'gap':>10}")
    print('-' * 80)
    for name, (sa, fname) in scenarios.items():
        avg_age, cost_pp = run_cohort_with_cost(pomdp, sa, Rcost)
        real = json.load(open(os.path.join(RES, fname)))

        real_age = real['avg_age_at_death']
        gap_age = (real_age - avg_age) / avg_age * 100
        rows.append([name, 'avg_age_at_death', round(real_age, 3), round(avg_age, 3), f"{gap_age:+.1f}%"])
        print(f"{name:<26}{'avg_age_at_death':<20}{real_age:>12.3f}{avg_age:>12.3f}{gap_age:>+9.1f}%")

        real_cost = real['cost_per_person_usd']
        gap_cost = (real_cost - cost_pp) / cost_pp * 100
        rows.append([name, 'cost_per_person_usd', round(real_cost, 2), round(cost_pp, 2), f"{gap_cost:+.1f}%"])
        print(f"{name:<26}{'cost_per_person_usd':<20}{real_cost:>12.2f}{cost_pp:>12.2f}{gap_cost:>+9.1f}%")

    out_csv = os.path.join(RES, 'compare_cost_lyg_matrix_vs_cmost.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'metric', 'real_cmost', 'matrix', 'gap_pct'])
        w.writerows(rows)
    print(f"\nSaved {out_csv}")


if __name__ == '__main__':
    main()
