"""n_sweep_convergence.py
=========================
N-sweep convergence check (like Krijkamp et al. Figure 6 / this project's
earlier m3_convergence.py, but on the CURRENT final model): for q5y vs
no-screen (CRN, same seed per N), track cost, QALY, ICER, INMB as N grows
from 10 to 300,000 -- do these stabilize well before the N=1,000,000
already anchored by the final 4-way results, i.e. would N=10,000,000 be
expected to change anything (per 1/sqrt(N) MCSE theory)?

Reuses cmost_4way_eval.run_cohort (for sr/Money) and
compute_real_qaly.qaly_from_sr (for the QALY side) -- same CRN seed for
no_screen and q5y at each N so the DELTA (not the absolute levels) is the
precision-relevant quantity, matching this project's established CRN
convention.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

import cmost_4way_eval as C4
from compute_real_qaly import qaly_from_sr

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
N_GRID = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
SEED = 999
WTP = 100000  # matches model_v2.WTP_DEFAULT (Neumann/Cohen/Weinstein 2014)


def eval_scenario(n, seed, hook):
    # neither no_screen (hook=None) nor q5y (FixedScheduleHook) ever
    # consult risk_class/sex, so a single seeded p per call is both
    # correct and sufficient here (unlike the 'policy' scenario in
    # cmost_4way_eval.py, which needs p shared with hook construction).
    np.random.seed(seed)
    p = C4.BNH.prepare_simulation_params(n)
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(p, n, seed, hook)
    cost_pp = float(money['AllCost'].sum()) / n
    qaly_pp = float(qaly_from_sr(sr).mean())
    return cost_pp, qaly_pp


def main():
    hook_q5y = C4.FixedScheduleHook([50, 55, 60, 65, 70, 75])
    rows = []
    for n in N_GRID:
        t0 = time.time()
        cost_nos, qaly_nos = eval_scenario(n, SEED, None)
        cost_q5, qaly_q5 = eval_scenario(n, SEED, hook_q5y)
        dcost = cost_q5 - cost_nos
        dqaly = qaly_q5 - qaly_nos
        icer = dcost / dqaly if abs(dqaly) > 1e-9 else float('nan')
        inmb = WTP * dqaly - dcost
        rows.append(dict(n=n, cost_nos=cost_nos, cost_q5=cost_q5, dcost=dcost,
                          qaly_nos=qaly_nos, qaly_q5=qaly_q5, dqaly=dqaly,
                          icer=icer, inmb=inmb))
        print(f"N={n:>8,}  dCost=${dcost:>9.2f}  dQALY={dqaly:>8.5f}  "
              f"ICER=${icer:>12,.0f}  INMB=${inmb:>10.2f}  ({time.time()-t0:.0f}s)", flush=True)

    out = os.path.join(RES, 'n_sweep_convergence_q5y.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print('saved', out)


if __name__ == '__main__':
    main()
