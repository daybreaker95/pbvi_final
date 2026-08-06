"""matrix_self_consistency_4metric.py
=====================================
Study-1 credibility check: does the matrix's OWN self-consistent prediction
of the trained (age-declining-utility) policy's clinical/cost benefit
(vs no-screening) match what the same policy actually produces when driven
through the real CMOST engine (cmost_4way_eval.py --scenario policy, N=1M)?

Reuses gather_fivi_full_trajectory.simulate_fivi (LYG/mortality/incidence)
and compute_matrix_cost.simulate_cost (cost) -- both driven by the SAME
CRN scheme (seed, age), same trained policy, same N -- so all four metrics
come from internally-consistent matrix-only MC, directly comparable to the
real-CMOST cmost_4way_{no_screen,policy}.json numbers already on disk.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import cmost_4way_eval as C4
from gather_fivi_full_trajectory import simulate_fivi
from compute_matrix_cost import build_Rcost, simulate_cost

MC_N = 1_000_000
SEED = 42
RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def summarize(r):
    crc = r['death_cause'] == 2
    dx = r['dx_stage']
    n = len(dx)
    return {
        'avg_age_all': float(r['death_age'].mean()),
        'crc_100k': 1e5 * crc.mean(),
        'incid_100k': 1e5 * (dx > 0).mean(),
    }


def main():
    t0 = time.time()
    print('training policy (current age-declining-utility matrix)...', flush=True)
    pomdp, solver = C4.train_policy()
    print(f'  gap={solver.gap_history[-1]["gap"]:.4f}  ({time.time()-t0:.1f}s)', flush=True)

    Rcost = build_Rcost(pomdp)

    t0 = time.time()
    r_nos = simulate_fivi(pomdp, solver, MC_N, seed=SEED, policy=False)
    r_pol = simulate_fivi(pomdp, solver, MC_N, seed=SEED, policy=True)
    s_nos, s_pol = summarize(r_nos), summarize(r_pol)
    print(f'clinical MC done ({time.time()-t0:.0f}s)', flush=True)

    t0 = time.time()
    cost_nos = simulate_cost(pomdp, solver, Rcost, MC_N, seed=SEED, policy=False)
    cost_pol = simulate_cost(pomdp, solver, Rcost, MC_N, seed=SEED, policy=True)
    print(f'cost MC done ({time.time()-t0:.0f}s)', flush=True)

    life_years_nos = s_nos['avg_age_all'] - pomdp.age_min
    life_years_pol = s_pol['avg_age_all'] - pomdp.age_min
    dlyg_days = (life_years_pol - life_years_nos) * 365.25
    mort_red = (s_nos['crc_100k'] - s_pol['crc_100k']) / s_nos['crc_100k'] * 100
    incid_red = (s_nos['incid_100k'] - s_pol['incid_100k']) / s_nos['incid_100k'] * 100
    cost_nos_pp = float(cost_nos.mean())
    cost_pol_pp = float(cost_pol.mean())
    cost_pct = (cost_pol_pp - cost_nos_pp) / cost_nos_pp * 100

    matrix_result = {
        'n': MC_N, 'seed': SEED,
        'crc_100k_no_screen': s_nos['crc_100k'], 'crc_100k_policy': s_pol['crc_100k'],
        'incid_100k_no_screen': s_nos['incid_100k'], 'incid_100k_policy': s_pol['incid_100k'],
        'cost_pp_no_screen': cost_nos_pp, 'cost_pp_policy': cost_pol_pp,
        'life_years_no_screen': life_years_nos, 'life_years_policy': life_years_pol,
        'mortality_reduction_pct': mort_red, 'incidence_reduction_pct': incid_red,
        'cost_change_pct': cost_pct, 'dlyg_days': dlyg_days,
    }
    with open(os.path.join(RES, 'matrix_self_consistency_4metric.json'), 'w') as f:
        json.dump(matrix_result, f, indent=2)

    real_nos = json.load(open(os.path.join(RES, 'cmost_4way_no_screen.json')))
    real_pol = json.load(open(os.path.join(RES, 'cmost_4way_policy.json')))
    real_mort_red = (real_nos['crc_death_per_100k'] - real_pol['crc_death_per_100k']) / real_nos['crc_death_per_100k'] * 100
    real_incid_red = (real_nos['incidence_per_100k'] - real_pol['incidence_per_100k']) / real_nos['incidence_per_100k'] * 100
    real_cost_pct = (real_pol['cost_per_person_usd'] - real_nos['cost_per_person_usd']) / real_nos['cost_per_person_usd'] * 100
    real_dlyg_days = (real_pol['life_years'] - real_nos['life_years']) * 365.25

    print(f"\n{'metric':<28}{'matrix (self-pred)':>20}{'real CMOST':>14}{'gap':>10}")
    print('-' * 72)
    rows = [
        ('Mortality reduction %', mort_red, real_mort_red),
        ('Incidence reduction %', incid_red, real_incid_red),
        ('Cost change %', cost_pct, real_cost_pct),
        ('dLYG (days)', dlyg_days, real_dlyg_days),
    ]
    for name, mval, rval in rows:
        gap = mval - rval
        print(f"{name:<28}{mval:>20.2f}{rval:>14.2f}{gap:>+10.2f}")


if __name__ == '__main__':
    main()
