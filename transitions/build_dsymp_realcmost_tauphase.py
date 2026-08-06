"""
build_dsymp_realcmost_tauphase.py
====================================
Rebuilds P_undetected/d_symp (pre-diagnosis dynamics) using the SAME
tau-phase methodology as this morning's verify_undetected_tauphase.py /
build_dsymp_from_undetected_tauphase.py (which used CRCEngine), but sourced
DIRECTLY from the real CMOST engine (NumberCrunching_100000 + calculate_sub's
quarterly_recorder, same pattern as estimate_tn_dsymp_realengine.py and
today's build_T_det_realcmost_tauphase.py).

Rationale: this morning's CRCEngine-vs-real-engine data-source ablation
(before tau-phase existed) showed data source barely mattered for the
undetected side -- tau-phase (methodology) was the dominant fix. Today's
T_det work found the OPPOSITE for the post-diagnosis side (data source was
dominant). Redoing the undetected side with real-CMOST data now closes the
loop methodologically even though the result may not move much -- using
CRCEngine anywhere in the final pipeline is no longer defensible once a
real-CMOST-sourced alternative exists and is cheap enough to build.

No-screening only (tau_u, the undetected-side clock, is diagnosis-route-
independent in the same way tau_d was -- see build_T_det_realcmost_tauphase.py
for why no-screening-only sourcing is fine here too: onset time doesn't
depend on how/when future screening would happen).

18-state -> verify_undetected_tauphase's 209-state tau-augmented layout:
  0 Normal -> N
  1-4 P1-4 (early adenoma) -> EA
  5-6 P5-6 (adv adenoma) -> AA
  7-10 U1-4 (undetected cancer I-IV) -> cidx_u(k, tau_u), tau_u = quarters
        since first entering ANY cancer state (7-10 or 11-14 if somehow
        observed there first at quarterly granularity)
  11-14 D1-4 (detected cancer I-IV) -> cidx_d(k, tau_d), tau_d = quarters
        since first entering a D-state (diagnosis)
  15 Dead_CRC -> DCRC
  16-17 Dead_Comp/Other -> DOTH

Run: python build_dsymp_realcmost_tauphase.py -n 1000000
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
sys.path.insert(0, _THIS)
sys.path.insert(0, CMOST_PY)

import verify_undetected_tauphase as VUT
from build_dsymp_from_undetected_tauphase import propagate_and_extract
from estimate_tn_dsymp_realengine import run_chunk, first_detect_from_18state, DET18_LO, DET18_HI
from env.state9 import STATE9_NAMES

RES = os.path.join(PBVI_ROOT, 'results')
NQ = VUT.NQ  # 400
N, EA, AA = VUT.N, VUT.EA, VUT.AA
cidx_u, cidx_d = VUT.cidx_u, VUT.cidx_d
NCTAU_U, TAU_U_MAX = VUT.NCTAU_U, VUT.TAU_U_MAX
NCTAU_D, TAU_D_MAX = VUT.NCTAU_D, VUT.TAU_D_MAX
NS = VUT.NS
DOTH, DCRC = VUT.DOTH, VUT.DCRC
UDET_LO, UDET_HI = 7, 10  # U1-U4 in the 18-state scheme


def first_onset_from_18state(qr18):
    """First quarter (1-based t) where state is in U1..U4 or D1..D4 (7..14)
    -- i.e. first entry into ANY cancer state, undetected or detected. Under
    no-screening the only path into D-states is via a prior U-state, so this
    always captures true onset (or is within 1 quarter of it at worst, if a
    transition lands mid-quarter-boundary)."""
    NQt, n = qr18.shape
    is_ca = (qr18 >= UDET_LO) & (qr18 <= DET18_HI)
    any_ca = is_ca.any(axis=0)
    first_q0 = np.where(any_ca, is_ca.argmax(axis=0), -1)
    foq = np.where(any_ca, first_q0 + 1, -1).astype(np.int32)
    return foq


def map_to_vut209(qr18, foq, fdq):
    """qr18: (NQ, n) int16 raw 18-state. foq/fdq: (n,) 1-based first-onset /
    first-diagnosis quarter or -1. Returns qstate (n, NQ+2) int16 in VUT's
    209-state numbering."""
    NQt, n = qr18.shape
    PRE = np.full(18, -1, dtype=np.int32)
    PRE[0] = N
    PRE[1:5] = EA
    PRE[5:7] = AA
    PRE[15] = DCRC
    PRE[16] = DOTH; PRE[17] = DOTH

    qstate = np.full((n, NQ + 2), DOTH, dtype=np.int16)
    for t in range(1, NQt + 1):
        s18 = qr18[t - 1].astype(np.int64)
        mapped = PRE[s18].copy()
        is_u = (s18 >= UDET_LO) & (s18 <= UDET_HI)
        if is_u.any():
            k = (s18[is_u] - UDET_LO).astype(np.int64)
            tau = t - foq[is_u]
            tau = np.clip(tau, 0, TAU_U_MAX)
            mapped[is_u] = np.array([cidx_u(kk, tt) for kk, tt in zip(k, tau)], dtype=np.int32)
        is_d = (s18 >= DET18_LO) & (s18 <= DET18_HI)
        if is_d.any():
            k = (s18[is_d] - DET18_LO).astype(np.int64)
            tau = t - fdq[is_d]
            tau = np.clip(tau, 0, TAU_D_MAX)
            mapped[is_d] = np.array([cidx_d(kk, tt) for kk, tt in zip(k, tau)], dtype=np.int32)
        qstate[:, t] = mapped.astype(np.int16)
    qstate[:, NQt + 1] = qstate[:, NQt]
    return qstate


def estimate_for_mask(qstate, mask, T_pooled_doth=None):
    """T_pooled_doth: optional (MAXY+1, NS) OTHER_DEATH column from an
    UNSTRATIFIED (pooled) estimate_Tq run. OTHER_DEATH is driven purely by
    LifeTable[age,gender] in the CMOST engine -- it does NOT depend on
    IndividualRisk at all -- so re-estimating it separately per risk
    stratum is both unnecessary and, worse, biased: the high-risk stratum's
    "still Normal at this age" subpopulation shrinks fast with age (most
    high-risk people have already progressed to a polyp/cancer state by
    then), so its per-stratum OTHER_DEATH tabulation is drawn from an
    increasingly small, selected sub-sample and comes out systematically
    LOW (confirmed empirically: age-80 Normal->OtherDeath ratio high-risk/
    low-risk = 0.89, growing worse with age -- see the LYG/cost matrix-vs-
    real-CMOST investigation this traced back from). Fix: substitute the
    pooled (whole-cohort) OTHER_DEATH probability into every row, rescaling
    the row's other entries so it still sums to 1 -- i.e. keep the
    stratified CRC-relevant transition shape, just correct the one
    component that was never supposed to vary by risk stratum."""
    T = VUT.estimate_Tq(qstate[mask])
    if T_pooled_doth is not None:
        for y in range(T.shape[0]):
            for s in range(NS):
                if s in (DOTH, DCRC):
                    continue
                old_doth = T[y, s, DOTH]
                new_doth = T_pooled_doth[y, s]
                rest = 1.0 - old_doth
                new_rest = 1.0 - new_doth
                if rest > 1e-12:
                    T[y, s, :] *= (new_rest / rest)
                T[y, s, DOTH] = new_doth
    P_year, d_year = propagate_and_extract(T)
    ages = np.arange(1, 100)
    return P_year[ages - 1], d_year[ages - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=24680)
    ap.add_argument('--high_frac', type=float, default=0.25)
    args = ap.parse_args()

    print(f'[1] real-CMOST no-screening quarterly cohort, n={args.n:,} ...', flush=True)
    t0 = time.time()
    n_chunks = max(1, args.n // args.chunk)
    qr18_chunks, risk_chunks = [], []
    for i in range(n_chunks):
        tC = time.time()
        qr18, risk = run_chunk(args.chunk, args.seed + i)
        qr18_chunks.append(qr18)
        risk_chunks.append(risk)
        print(f'  chunk {i+1}/{n_chunks} done ({time.time()-tC:.0f}s, total {time.time()-t0:.0f}s)', flush=True)
    qr18 = np.concatenate(qr18_chunks, axis=1)
    risk = np.concatenate(risk_chunks)
    del qr18_chunks, risk_chunks
    n_total = qr18.shape[1]

    print('[2] first-onset / first-diagnosis quarters + mapping to 209-state ...', flush=True)
    foq = first_onset_from_18state(qr18)
    fdq = first_detect_from_18state(qr18)
    frac_ca = float((foq >= 0).mean())
    frac_dx = float((fdq >= 0).mean())
    print(f'  ever-cancer: {frac_ca:.4f}  ever-diagnosed (no-screening): {frac_dx:.4f}', flush=True)

    qstate = map_to_vut209(qr18, foq, fdq)
    del qr18

    thr = float(np.quantile(risk, 1 - args.high_frac))
    risk_hi = risk >= thr
    print(f'risk threshold = {thr:.6f}  frac_high = {risk_hi.mean():.4f}', flush=True)

    print('[2.5] estimating POOLED (risk-independent) matrix for OTHER_DEATH ...', flush=True)
    tA = time.time()
    T_pooled = VUT.estimate_Tq(qstate)
    T_pooled_doth = T_pooled[:, :, DOTH].copy()
    del T_pooled
    print(f'    done ({time.time()-tA:.0f}s)', flush=True)

    print('[3] estimating LOW-risk tau-phased matrix + marginalizing ...', flush=True)
    tA = time.time()
    P_lo, d_lo = estimate_for_mask(qstate, ~risk_hi, T_pooled_doth)
    print(f'    done ({time.time()-tA:.0f}s)', flush=True)

    print('[4] estimating HIGH-risk tau-phased matrix + marginalizing ...', flush=True)
    tA = time.time()
    P_hi, d_hi = estimate_for_mask(qstate, risk_hi, T_pooled_doth)
    print(f'    done ({time.time()-tA:.0f}s)', flush=True)

    ages = np.arange(1, 100)
    out = os.path.join(RES, 'transitions_9state_stratified_realcmost_tauphase.npz')
    np.savez_compressed(
        out, ages=ages,
        P_undetected_low=P_lo, d_symp_low=d_lo,
        P_undetected_high=P_hi, d_symp_high=d_hi,
        risk_threshold=thr, frac_high=float(risk_hi.mean()),
        state_names=np.array(STATE9_NAMES), n_patients=n_total,
        source='real CMOST (NumberCrunching_100000), no-screening, tau-phase',
    )
    print(f'Saved {out}  ({time.time()-t0:.0f}s total)', flush=True)

    print('\nPreview (age 55, 65): d_symp for undetected Ca states')
    for age in (55, 65):
        ai = age - 1
        for k, name in enumerate(['Ca I', 'Ca II', 'Ca III', 'Ca IV']):
            s = 3 + k
            print(f'  age {age} {name:7s} LOW : stay={100*P_lo[ai,s].sum():5.1f}%  '
                  f'd_symp={np.round(100*d_lo[ai,s],3).tolist()}')
            print(f'  age {age} {name:7s} HIGH: stay={100*P_hi[ai,s].sum():5.1f}%  '
                  f'd_symp={np.round(100*d_hi[ai,s],3).tolist()}')


if __name__ == '__main__':
    main()
