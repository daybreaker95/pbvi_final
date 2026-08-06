"""
estimate_stratified.py
======================

Estimate the 6-state transition matrices SEPARATELY for a low- and a high-
adenoma-risk subpopulation, so the POMDP can carry a latent risk class and infer
it from observed colonoscopy findings (a patient in whom adenomas keep being
found is probably high-risk, and should be screened sooner -- the mechanism of
personalised, history-dependent screening).

Like estimate_transitions.py, the dynamics are estimated on a QUARTER-year step
and the consumed annual matrix is formed by composition P_year = P_quarter^4,
which removes the annual-discretization artifacts.

Risk class is defined from CMOST's per-individual adenoma-risk multiplier
(IndividualRisk): the top `high_frac` (default 25%, matching Zaika 2024's high-
risk stratum) are 'high', the rest 'low'.

Outputs results/transitions_stratified.npz:
  P_low, P_high            (A,6,6)  class-specific ANNUAL (composed) matrices
  Pq_low, Pq_high          (A,6,6)  class-specific one-QUARTER matrices
  prev_low, prev_high      (A+1,6)  prevalence among living by age, per class
  ages, risk_threshold, frac_high
  lifetime_inc_low/high    scalar   lifetime clinical CRC incidence per class
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import CRCEngine, DEAD_OTHER, N_STATES
from transitions.estimate_transitions import (
    tally_quarterly, tally_transitions, normalise, compose_annual, NQ,
    AGE_MIN, AGE_MAX,
)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def main(n=100000, seed=2468, settings='CMOST13', high_frac=0.25):
    params = build_params(settings, n_patients=500, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))

    # risk threshold = (1-high_frac) quantile of the IndividualRisk pool
    pool = np.asarray(params['individual_risk'], float)
    thr = float(np.quantile(pool, 1 - high_frac))
    print(f"risk threshold (IndividualRisk {1-high_frac:.0%} quantile) = {thr:.4f}")

    qpaths = np.full((n, NQ + 2), DEAD_OTHER, dtype=np.int8)
    risk_hi = np.zeros(n, dtype=bool)
    diagnosed = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        risk_hi[i] = pt.individual_risk >= thr
        dead = False
        for y in range(1, 101):
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                qpaths[i, t] = eng.clinical_state(pt)
                if not pt.alive:
                    dead = True
                    break
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    dead = True
                    break
            if dead:
                break
        diagnosed[i] = len(pt.det_stage) > 0
        if (i + 1) % 25000 == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")
    print(f"done {n} in {time.time()-t0:.0f}s; frac high = {risk_hi.mean():.3f}")

    # yearly snapshots (for prevalence)
    ypaths = np.full((n, 101), DEAD_OTHER, dtype=np.int8)
    for y in range(1, 101):
        ypaths[:, y] = qpaths[:, (y - 1) * 4 + 1]

    def stratum(mask):
        ages, counts_q = tally_quarterly(qpaths[mask])
        Pq, _, _ = normalise(counts_q)
        P = compose_annual(Pq)
        _, _, alive, prev = tally_transitions(ypaths[mask])
        return ages, P, Pq, prev, alive

    ages, P_lo, Pq_lo, prev_lo, alive_lo = stratum(~risk_hi)
    _, P_hi, Pq_hi, prev_hi, alive_hi = stratum(risk_hi)

    inc_lo = 100 * diagnosed[~risk_hi].mean()
    inc_hi = 100 * diagnosed[risk_hi].mean()
    print(f"lifetime clinical CRC incidence: low={inc_lo:.2f}%  high={inc_hi:.2f}%  "
          f"(ratio {inc_hi/max(inc_lo,1e-9):.2f}x)")

    np.savez_compressed(
        os.path.join(RES, 'transitions_stratified.npz'),
        ages=ages, P_low=P_lo, P_high=P_hi, Pq_low=Pq_lo, Pq_high=Pq_hi,
        prev_low=prev_lo, prev_high=prev_hi,
        risk_threshold=thr, frac_high=float(risk_hi.mean()),
        lifetime_inc_low=inc_lo, lifetime_inc_high=inc_hi)
    print(f"Saved {os.path.join(RES, 'transitions_stratified.npz')}")

    for age in (55, 65):
        ai = age - AGE_MIN
        print(f"\n age {age}  AA->EarlyCa (annual):  low={100*P_lo[ai,2,3]:.2f}%  "
              f"high={100*P_hi[ai,2,3]:.2f}%   Norm->EA: low={100*P_lo[ai,0,1]:.2f}% "
              f"high={100*P_hi[ai,0,1]:.2f}%")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    main(n=n)
