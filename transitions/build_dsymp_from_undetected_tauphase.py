"""
build_dsymp_from_undetected_tauphase.py
=========================================
Builds an improved, NON-phase-dimensioned P_undetected[age]/d_symp[age] pair
for pomdp/model_v2.py's 9-state undetected-world dynamics, by marginalizing
the tau-phased matrix from verify_undetected_tauphase.py -- same pattern
build_T_detected_from_tauphase.py already used for the post-diagnosis side
(see that file's docstring): use the fully-validated, phase-aware quarterly
dynamics as a CALIBRATION tool, without exposing "quarters since cancer
onset" as a belief-tracked POMDP dimension.

Why this is needed (see verify_undetected_tauphase.py's docstring): CMOST
draws a cancer focus's symptom-presentation time AND its I->II->III->IV
stage-progression schedule as ONE-TIME deterministic draws at onset, so
"probability of detection this year" is a function of how long the focus has
already been undetected, not just its current stage -- something a plain
memoryless d_symp[age][stage] table cannot represent. n=100,000 validation
(verify_undetected_tauphase.py) confirmed the tau-aware cohort reproduces
CMOST's own no-screening incidence/stage distribution far more closely than
the current matrix (incidence -0.24% vs +8.3%, Stage I -0.12pp vs +25.2pp).

Method, per risk class (low/high, same 25% split convention as
estimate_transitions_9state_stratified.py):
  1. Simulate a no-screening cohort (quarterly), recording BOTH the tau-
     augmented state path (verify_undetected_tauphase.ext_state) and
     individual_risk.
  2. Estimate the tau-augmented quarterly transition matrix per risk mask
     (verify_undetected_tauphase.estimate_Tq).
  3. Propagate a fresh no-screening cohort through it year by year. At each
     age, for each 9-state SOURCE category (Normal/EarlyPolyp/AdvPolyp,
     undetected Ca I..IV):
       a. Redirect each "just-entered-detected-stage-k" transition to an
          absorbing per-stage sink (same 4-exit-sink trick as
          estimate_transitions_9state.compose_annual) so within-year
          detection timing doesn't get conflated with subsequent
          already-diagnosed dynamics.
       b. For the undetected Ca-stage categories, weight-average the
          resulting year-end row using the COHORT'S OWN, already-propagated
          tau-occupancy within that stage (not a uniform average) -- this is
          the step that actually captures "time since onset": patients who
          have been undetected for 3 years have a materially different
          detection profile over the next year than patients who just
          became Ca-I this quarter, and the cohort's realistic tau-mixture
          (built up from realistic onset-time arrivals) reflects this.
       c. Collapse the year-end row back onto the plain 9-state axis
          (P_year[src,dst]) and read off the 4 sink columns (d_year[src,k]).
     Then advance the REAL (un-redirected) cohort forward one year to get
     next age's starting tau-occupancy.

Output: results/transitions_9state_stratified_tauphase.npz -- schema-
identical to the currently-active transitions_9state_stratified.npz
(P_undetected_low/high, d_symp_low/high, ages, risk_threshold, frac_high,
...), a drop-in swap for pomdp/model_v2.py.

Run: python build_dsymp_from_undetected_tauphase.py -n 400000
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
from env.params import build_params
from env.cmost_individual import CRCEngine
import verify_undetected_tauphase as VUT
from env.state9 import (
    NORMAL, EARLY_POLYP, ADV_POLYP, CA_I, CRC_DEATH, OTHER_DEATH,
    N_STATES9, STATE9_NAMES,
)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
MAXY, AGE_MIN, AGE_MAX = VUT.MAXY, VUT.AGE_MIN, VUT.AGE_MAX
N, EA, AA = VUT.N, VUT.EA, VUT.AA
NCTAU_U, NCTAU_D = VUT.NCTAU_U, VUT.NCTAU_D
cidx_u, cidx_d = VUT.cidx_u, VUT.cidx_d
NS = VUT.NS
DOTH, DCRC = VUT.DOTH, VUT.DCRC


def run_microsim_with_risk(n, seed):
    """Same as verify_undetected_tauphase.run_microsim_q(no screening), plus
    per-patient individual_risk for the low/high split."""
    eng = CRCEngine(build_params('CMOST13', 500, seed), rng=np.random.default_rng(0))
    eng.rng = np.random.default_rng(seed)
    paths = np.full((n, VUT.NQ + 2), -1, dtype=np.int16)
    risk = np.zeros(n, dtype=float)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        risk[i] = pt.individual_risk
        for y in range(1, MAXY + 1):
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                paths[i, t] = VUT.ext_state(pt, y + (q - 1) / 4.0)
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    dstate = DCRC if pt.death_cause == 2 else DOTH
                    paths[i, t + 1:] = dstate
                    break
            if not pt.alive:
                break
        if (i + 1) % 50000 == 0:
            print(f"    microsim {i+1}/{n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  done {n} in {time.time()-t0:.0f}s", flush=True)
    return paths, risk


def propagate_and_extract(T):
    """No-screening propagation through the tau-augmented quarterly matrix T
    (MAXY+1, NS, NS); returns P_year (MAXY,9,9), d_year (MAXY,9,4)."""
    dist = np.zeros(NS); dist[N] = 1.0
    P_year = np.zeros((MAXY, N_STATES9, N_STATES9))
    d_year = np.zeros((MAXY, N_STATES9, 4))
    NSX = NS + 4
    SINK0 = NS
    u_idx = [[cidx_u(k, tau) for tau in range(NCTAU_U)] for k in range(4)]

    def fill_row(age, src9, row):
        P_year[age - 1, src9, NORMAL] = row[N]
        P_year[age - 1, src9, EARLY_POLYP] = row[EA]
        P_year[age - 1, src9, ADV_POLYP] = row[AA]
        for k in range(4):
            P_year[age - 1, src9, CA_I + k] = row[u_idx[k]].sum()
        P_year[age - 1, src9, CRC_DEATH] = 0.0     # unreachable pre-diagnosis
        P_year[age - 1, src9, OTHER_DEATH] = row[DOTH]
        for k in range(4):
            d_year[age - 1, src9, k] = row[SINK0 + k]

    for age in range(1, MAXY + 1):
        Ty = T[min(age, MAXY)]
        Ty_aug = np.zeros((NSX, NSX))
        Ty_aug[:NS, :NS] = Ty
        for k in range(4):
            c0 = cidx_d(k, 0)
            Ty_aug[:NS, SINK0 + k] = Ty[:NS, c0]   # redirect fresh-detection mass to sink
            Ty_aug[:NS, c0] = 0.0                  # sever the original (now-unreachable) column
        for k in range(4):
            Ty_aug[SINK0 + k, SINK0 + k] = 1.0
        Ty4_aug = np.linalg.matrix_power(Ty_aug, 4)

        fill_row(age, NORMAL, Ty4_aug[N])
        fill_row(age, EARLY_POLYP, Ty4_aug[EA])
        fill_row(age, ADV_POLYP, Ty4_aug[AA])
        for k in range(4):
            tau_idx = u_idx[k]
            mass = dist[tau_idx].sum()
            if mass > 1e-12:
                w = dist[tau_idx] / mass
                row = w @ Ty4_aug[tau_idx]
            else:
                row = Ty4_aug[cidx_u(k, 0)]   # no cohort mass yet -> fresh-onset fallback
            fill_row(age, CA_I + k, row)

        for _qq in (1, 2, 3, 4):
            dist = dist @ Ty
    return P_year, d_year


def estimate_for_mask(paths, mask):
    T = VUT.estimate_Tq(paths[mask])
    P_year, d_year = propagate_and_extract(T)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    return P_year[ages - 1], d_year[ages - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=400_000)
    ap.add_argument('--seed', type=int, default=31415)
    ap.add_argument('--high_frac', type=float, default=0.25)
    args = ap.parse_args()

    print(f"[1] no-screening microsim (n={args.n:,}, tau-augmented paths, quarterly) ...", flush=True)
    paths, risk = run_microsim_with_risk(args.n, args.seed)

    thr = float(np.quantile(risk, 1 - args.high_frac))
    risk_hi = risk >= thr
    print(f"risk threshold (individual_risk {1-args.high_frac:.0%} quantile) = {thr:.6f}", flush=True)
    print(f"frac high = {risk_hi.mean():.4f}", flush=True)

    print("[2] estimating LOW-risk tau-phased quarterly matrix + marginalizing ...", flush=True)
    t0 = time.time()
    P_lo, d_lo = estimate_for_mask(paths, ~risk_hi)
    print(f"    done ({time.time()-t0:.0f}s)", flush=True)

    print("[3] estimating HIGH-risk tau-phased quarterly matrix + marginalizing ...", flush=True)
    t0 = time.time()
    P_hi, d_hi = estimate_for_mask(paths, risk_hi)
    print(f"    done ({time.time()-t0:.0f}s)", flush=True)

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    out = os.path.join(RES, 'transitions_9state_stratified_tauphase.npz')
    np.savez_compressed(
        out, ages=ages,
        P_undetected_low=P_lo, d_symp_low=d_lo,
        P_undetected_high=P_hi, d_symp_high=d_hi,
        risk_threshold=thr, frac_high=float(risk_hi.mean()),
        state_names=np.array(STATE9_NAMES), n_patients=args.n,
        source='verify_undetected_tauphase (tau-since-onset marginalized)',
    )
    print(f"Saved {out}", flush=True)

    print("\nPreview (age 55, 65): d_symp (self-present this year, by dest stage) for undetected Ca states")
    for age in (55, 65):
        ai = age - AGE_MIN
        for k, name in enumerate(['Ca I', 'Ca II', 'Ca III', 'Ca IV']):
            s = CA_I + k
            print(f"  age {age} {name:7s} LOW : stay={100*P_lo[ai,s].sum():5.1f}%  "
                  f"d_symp={np.round(100*d_lo[ai,s],3).tolist()}")
            print(f"  age {age} {name:7s} HIGH: stay={100*P_hi[ai,s].sum():5.1f}%  "
                  f"d_symp={np.round(100*d_hi[ai,s],3).tolist()}")


if __name__ == '__main__':
    main()
