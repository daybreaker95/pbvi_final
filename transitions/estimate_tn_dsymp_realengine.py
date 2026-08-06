"""Re-estimate T_undetected/d_symp DIRECTLY from the real CMOST engine
(NumberCrunching_100000.py + calculate_sub.py's quarterly_recorder hook),
bypassing CRCEngine entirely -- unlike estimate_transitions_9state.py
(which simulates via env.cmost_individual.CRCEngine, a from-scratch
reimplementation), this drives the ACTUAL MATLAB-ported engine and just
converts its 18-state quarterly snapshot into our 9-state axis.

Root-cause context: direct comparison showed CRCEngine's own no-screening
incidence is ~3.6% below the real engine's (4544 vs 4708/100k, N=10M vs
N=1M) despite the symptom-timing FORMULA being byte-identical between the
two codebases (same SojournMatrix construction, same symptime = onset_time
+ sojourn convention) -- so the discrepancy must live upstream (onset-rate
application, RNG consumption order, or some other implementation nuance we
did not fully trace) rather than in the detection-delay mechanism itself.
Rather than keep diffing two 1000+ line codebases, this re-estimates the
transition matrix from the real engine's own dynamics directly, matching
the same pattern used earlier for T_detected (tau-phase marginalization).

18-state -> 9-state (state9) collapse:
  0(Normal)->0, 1-4(P1-4,early)->1, 5-6(P5-6,adv)->2, 7-10(U1-4)->3-6(CA_I-IV),
  11-14(D1-4)->3-6(CA_I-IV) [state9 doesn't carry a detected flag in the
  VALUE -- detection is tracked separately via first_detect_q], 15(Dead_CRC)->7,
  16-17(Dead_Comp/Other)->8.
Since this is a pure natural-history (no screening) cohort, ANY entry into
D1-D4 is by definition a symptom-detection event (no other pathway exists).

Risk stratification: uses the REAL engine's own IndividualRisk array (not
CRCEngine's) with a threshold recomputed as ITS OWN top-25% quantile --
more correct than reusing CRCEngine's threshold (3.5278), which was only an
approximate transfer to begin with (see report section 06 limitations).
"""
import os
import sys
import io
import time
import argparse
import contextlib
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PBVI_ROOT = os.path.abspath(os.path.join(_THIS, '..'))
CMOST_PY = os.path.abspath(os.path.join(
    PBVI_ROOT, '..', '..', '..', '..', 'cmost_experiment_final', 'CMOST_experiment', 'python'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, CMOST_PY)

from transitions.estimate_transitions_9state import (
    tally, normalise_quarter, compose_annual, MAXY, NQ, AGE_MIN, AGE_MAX,
)
from env.state9 import N_STATES9, STATE9_NAMES, CRC_DEATH, OTHER_DEATH

# env.cmost_individual (imported transitively above) inserts its OWN parent
# dir at sys.path[0] as a side effect -- that dir happens to also contain a
# stale legacy calculate_sub.py without quarterly_recorder support, which
# would silently shadow the real one. Re-assert CMOST_PY's priority right
# before importing it.
sys.path.insert(0, CMOST_PY)
from build_natural_history_transition_matrix import load_cmost13
from calculate_sub import calculate_sub

RES = os.path.join(PBVI_ROOT, 'results')

# 18-state (NumberCrunching_100000's _pomdp_state_idx) -> our 9-state (state9)
MAP18TO9 = np.array([
    0,             # 0 Normal
    1, 1, 1, 1,    # 1-4 P1-4 -> EARLY_POLYP
    2, 2,          # 5-6 P5-6 -> ADV_POLYP
    3, 4, 5, 6,    # 7-10 U1-4 -> CA_I..IV
    3, 4, 5, 6,    # 11-14 D1-4 -> CA_I..IV (detection tracked separately)
    7,             # 15 Dead_CRC -> CRC_DEATH
    8, 8,          # 16-17 Dead_Comp/Other -> OTHER_DEATH
], dtype=np.int8)
DET18_LO, DET18_HI = 11, 14   # D1..D4 range in the 18-state scheme


def run_chunk(n, seed):
    """Pure natural-history (no screening, no surveillance) cohort via the
    REAL engine, quarterly-recorded. Returns (qr[400,n] int8 18-state,
    individual_risk[n] float)."""
    np.random.seed(seed)
    settings = load_cmost13()
    settings['Number_patients'] = n
    settings['Screening']['Mode'] = 'off'
    settings['Polyp_Surveillance'] = 'off'
    settings['Cancer_Surveillance'] = 'off'
    settings['SpecialText'] = ''
    settings['SpecialFlag'] = 'off'
    handles = {'Variables': settings}
    qr = np.zeros((NQ, n), dtype=np.int8)
    with contextlib.redirect_stdout(io.StringIO()):
        handles, bm = calculate_sub(handles, quarterly_recorder=qr)
    risk = np.asarray(handles['data']['IndividualRisk'], dtype=float)
    return qr, risk


def build_qstate_fdq(qr9):
    """qr9: (NQ, n) int8, 9-state, quarter-start snapshots (qi=0..399, i.e.
    t=1..400 in estimate_transitions_9state.py's 1-based convention).
    Returns qstate(n, NQ+2), first_detect_q(n,) matching tally()'s inputs."""
    NQt, n = qr9.shape
    qstate = np.full((n, NQ + 2), OTHER_DEATH, dtype=np.int8)
    qstate[:, 1:NQt + 1] = qr9.T
    # pad t=NQt+1 (boundary slot tally() may touch) with the last recorded state
    qstate[:, NQt + 1] = qr9[-1]

    is_det18 = None  # placeholder, computed by caller before calling this
    return qstate


def first_detect_from_18state(qr18):
    """qr18: (NQ, n) int8, RAW 18-state (pre-collapse). Detection = first
    quarter (1-based t) where state is in D1..D4 (11..14) -- the only
    detection pathway in a pure no-screening natural-history run."""
    NQt, n = qr18.shape
    is_det = (qr18 >= DET18_LO) & (qr18 <= DET18_HI)   # (NQ, n)
    any_det = is_det.any(axis=0)
    first_q0 = np.where(any_det, is_det.argmax(axis=0), -1)  # 0-based qi, or -1
    fdq = np.where(any_det, first_q0 + 1, -1).astype(np.int32)  # -> 1-based t
    return fdq


def estimate_for_mask(qstate, fdq, mask):
    counts_stay, counts_detect, n_row = tally(qstate[mask], fdq[mask])
    Pq_stay, dq = normalise_quarter(counts_stay, counts_detect, n_row)
    P_year, d_year = compose_annual(Pq_stay, dq)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    return Pq_stay[ages], dq[ages], P_year[ages], d_year[ages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--high_frac', type=float, default=0.25)
    args = ap.parse_args()

    print(f"Simulating n={args.n:,} via REAL CMOST engine (quarterly, no screening) ...", flush=True)
    t0 = time.time()
    n_chunks = max(1, args.n // args.chunk)
    qr18_chunks, risk_chunks = [], []
    for i in range(n_chunks):
        tC = time.time()
        qr18, risk = run_chunk(args.chunk, args.seed + i)
        qr18_chunks.append(qr18)
        risk_chunks.append(risk)
        print(f"  chunk {i+1}/{n_chunks} done ({time.time()-tC:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    qr18 = np.concatenate(qr18_chunks, axis=1)   # (NQ, n_total)
    risk = np.concatenate(risk_chunks)
    del qr18_chunks, risk_chunks
    n_total = qr18.shape[1]

    fdq = first_detect_from_18state(qr18)
    qr9 = MAP18TO9[qr18]
    qstate = build_qstate_fdq(qr9)
    del qr18, qr9

    frac_ever_detected = float((fdq >= 0).mean())
    print(f"  fraction ever diagnosed (no screening): {frac_ever_detected:.4f}", flush=True)

    thr = float(np.quantile(risk, 1 - args.high_frac))
    risk_hi = risk >= thr
    print(f"risk threshold (REAL engine IndividualRisk {1-args.high_frac:.0%} quantile) = {thr:.4f}", flush=True)
    print(f"frac high = {risk_hi.mean():.3f}", flush=True)

    inc_lo = 100 * (fdq[~risk_hi] >= 0).mean()
    inc_hi = 100 * (fdq[risk_hi] >= 0).mean()
    print(f"lifetime diagnosed-CRC incidence (no screening): low={inc_lo:.2f}%  "
          f"high={inc_hi:.2f}%  (ratio {inc_hi/max(inc_lo,1e-9):.2f}x)", flush=True)

    print("Estimating LOW-risk class matrix ...", flush=True)
    Pq_lo, dq_lo, P_lo, d_lo = estimate_for_mask(qstate, fdq, ~risk_hi)
    print("Estimating HIGH-risk class matrix ...", flush=True)
    Pq_hi, dq_hi, P_hi, d_hi = estimate_for_mask(qstate, fdq, risk_hi)

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    out = os.path.join(RES, 'transitions_9state_stratified_realengine.npz')
    np.savez_compressed(
        out, ages=ages,
        P_undetected_low=P_lo, d_symp_low=d_lo, Pq_stay_low=Pq_lo, dq_quarter_low=dq_lo,
        P_undetected_high=P_hi, d_symp_high=d_hi, Pq_stay_high=Pq_hi, dq_quarter_high=dq_hi,
        risk_threshold=thr, frac_high=float(risk_hi.mean()),
        lifetime_inc_low=inc_lo, lifetime_inc_high=inc_hi,
        state_names=np.array(STATE9_NAMES), n_patients=n_total,
    )
    print(f"Saved {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == '__main__':
    main()
