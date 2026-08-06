"""
estimate_transitions_9state_stratified.py
==========================================
9-state T_undetected/d_symp, split by adenoma-risk class (low/high), same
methodology as transitions/estimate_stratified.py (7-state) but reusing
estimate_transitions_9state.py's 4-exit-sink annual composition so each
class's matrix carries correct per-stage detection attribution.

Risk class definition (identical convention to estimate_stratified.py):
CMOST's per-individual `individual_risk` (fixed for life) -- the top
`high_frac` (default 25%, matching Zaika 2024's high-risk stratum) are
'high', the rest 'low'.

Does NOT touch transitions_9state.npz (the pooled/unstratified version) --
writes to transitions_9state_stratified.npz.

Run: python estimate_transitions_9state_stratified.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
from env.params import build_params
from env.cmost_individual import CRCEngine
from env.state9 import clinical_state9, N_STATES9, STATE9_NAMES, OTHER_DEATH
from estimate_transitions_9state import (
    tally, normalise_quarter, compose_annual, MAXY, NQ, AGE_MIN, AGE_MAX, CA_STAGES,
)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def simulate_with_risk(n, seed, settings='CMOST13'):
    """Same as estimate_transitions_9state.simulate(), plus per-patient
    individual_risk (for the low/high split) and a diagnosed flag (for the
    lifetime-incidence sanity check)."""
    params = build_params(settings, n_patients=20000, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
    qstate = np.full((n, NQ + 2), OTHER_DEATH, dtype=np.int8)
    first_detect_q = np.full(n, -1, dtype=np.int32)
    risk = np.zeros(n, dtype=float)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        risk[i] = pt.individual_risk
        was_detected = False
        for y in range(1, MAXY + 1):
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                s, det = clinical_state9(pt)
                qstate[i, t] = s
                if det and not was_detected:
                    first_detect_q[i] = t
                    was_detected = True
                if not pt.alive:
                    break
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    s2, det2 = clinical_state9(pt)
                    qstate[i, t + 1] = s2
                    if det2 and not was_detected:
                        first_detect_q[i] = t + 1
                        was_detected = True
                    break
            if not pt.alive:
                break
        if (i + 1) % 25000 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{n}  ({el:.0f}s, {1000*el/(i+1):.3f} ms/pt)")
    print(f"  done {n} in {time.time()-t0:.0f}s")
    return qstate, first_detect_q, risk


def estimate_for_mask(qstate, fdq, mask):
    counts_stay, counts_detect, n_row = tally(qstate[mask], fdq[mask])
    Pq_stay, dq = normalise_quarter(counts_stay, counts_detect, n_row)
    P_year, d_year = compose_annual(Pq_stay, dq)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    return Pq_stay[ages], dq[ages], P_year[ages], d_year[ages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=97531)
    ap.add_argument('--high_frac', type=float, default=0.25)
    args = ap.parse_args()

    print(f"Simulating n={args.n} (no screening, quarterly, 9-state, with individual_risk) ...")
    qstate, fdq, risk = simulate_with_risk(args.n, args.seed)

    thr = float(np.quantile(risk, 1 - args.high_frac))
    risk_hi = risk >= thr
    print(f"risk threshold (IndividualRisk {1-args.high_frac:.0%} quantile) = {thr:.4f}")
    print(f"frac high = {risk_hi.mean():.3f}")

    inc_lo = 100 * (fdq[~risk_hi] >= 0).mean()
    inc_hi = 100 * (fdq[risk_hi] >= 0).mean()
    print(f"lifetime diagnosed-CRC incidence (no screening): low={inc_lo:.2f}%  "
          f"high={inc_hi:.2f}%  (ratio {inc_hi/max(inc_lo,1e-9):.2f}x)")

    print("Estimating LOW-risk class matrix ...")
    Pq_lo, dq_lo, P_lo, d_lo = estimate_for_mask(qstate, fdq, ~risk_hi)
    print("Estimating HIGH-risk class matrix ...")
    Pq_hi, dq_hi, P_hi, d_hi = estimate_for_mask(qstate, fdq, risk_hi)

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    out = os.path.join(RES, 'transitions_9state_stratified.npz')
    np.savez_compressed(
        out, ages=ages,
        P_undetected_low=P_lo, d_symp_low=d_lo, Pq_stay_low=Pq_lo, dq_quarter_low=dq_lo,
        P_undetected_high=P_hi, d_symp_high=d_hi, Pq_stay_high=Pq_hi, dq_quarter_high=dq_hi,
        risk_threshold=thr, frac_high=float(risk_hi.mean()),
        lifetime_inc_low=inc_lo, lifetime_inc_high=inc_hi,
        state_names=np.array(STATE9_NAMES), n_patients=args.n,
    )
    print(f"Saved {out}")

    print("\nPreview (age 55, 65): stay-self / d_symp total, low vs high, per Ca stage")
    for age in (55, 65):
        ai = age - AGE_MIN
        for si, name in enumerate(['Ca I', 'Ca II', 'Ca III', 'Ca IV']):
            s = CA_STAGES[si]
            tot_lo = d_lo[ai, s].sum(); tot_hi = d_hi[ai, s].sum()
            print(f"  age {age} {name:7s}: low stay={100*P_lo[ai,s,s]:5.1f}% d_symp={100*tot_lo:5.2f}%   "
                  f"high stay={100*P_hi[ai,s,s]:5.1f}% d_symp={100*tot_hi:5.2f}%")


if __name__ == '__main__':
    main()
