"""
build_T_det_realcmost_tauphase.py
====================================
Rebuilds T_det EXACTLY the same way the original (already-validated)
build_T_detected_from_tauphase.py did -- 97-state tau-phased quarterly
matrix (verify_screening_tauphase.py's machinery), estimate_Tq +
propagate_and_extract, UNCHANGED -- except the input data comes from the
REAL CMOST engine (NumberCrunching_100000 + calculate_sub's quarterly_
recorder, same pattern as estimate_tn_dsymp_realengine.py) instead of
CRCEngine.

This replaces the simplified life-table attempt (build_T_det_from_real_
cmost.py), which showed a spurious old-age under-mortality artifact
(~1.6% of the population stuck "detected but never resolved" by age 100)
-- likely from pooling at-risk patients across very different diagnosis
ages without properly composing the quarterly hazard. Reusing the proper
tau-phase quarterly-matrix-power composition (Tx^4, same as the
already-validated undetected-side and original T_det pipelines) avoids
that failure mode by construction.

18-state (NumberCrunching_100000's _pomdp_state_idx) -> VST's 97-state:
  0 Normal -> VST.N
  1-4 P1-4 (early adenoma) -> VST.EA
  5-6 P5-6 (adv adenoma) -> VST.AA
  7-10 U1-4 (undetected cancer I-IV) -> VST.P1..P4 (undetected, no tau)
  11-14 D1-4 (detected cancer I-IV) -> VST.cidx(k, tau_d), tau_d = quarters
        since first entering ANY D-state (stage assumed fixed post-dx,
        matching how det_stage/TumorRecord['Stage'] work elsewhere)
  15 Dead_CRC -> VST.DCRC
  16-17 Dead_Comp/Other -> VST.DOTH

Run: python build_T_det_realcmost_tauphase.py -n 1000000
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

import verify_screening_tauphase as VST
from build_T_detected_from_tauphase import propagate_and_extract
from estimate_tn_dsymp_realengine import run_chunk, first_detect_from_18state, DET18_LO, DET18_HI

RES = os.path.join(PBVI_ROOT, 'results')
NQ = VST.NQ  # 400


def map_to_vst97(qr18, fdq):
    """qr18: (NQ, n) int16 raw 18-state. fdq: (n,) 1-based first-diagnosis
    quarter or -1. Returns qstate (n, NQ+2) int8 in VST's 97-state numbering,
    matching estimate_Tq's expected (n, NQ+2) 1-based-t layout."""
    NQt, n = qr18.shape
    PRE = np.full(18, -1, dtype=np.int16)
    PRE[0] = VST.N
    PRE[1:5] = VST.EA
    PRE[5:7] = VST.AA
    PRE[7] = VST.P1; PRE[8] = VST.P2; PRE[9] = VST.P3; PRE[10] = VST.P4
    PRE[15] = VST.DCRC
    PRE[16] = VST.DOTH; PRE[17] = VST.DOTH

    qstate = np.full((n, NQ + 2), VST.DOTH, dtype=np.int8)
    for t in range(1, NQt + 1):
        s18 = qr18[t - 1].astype(np.int64)
        mapped = PRE[s18].copy()
        is_det = (s18 >= DET18_LO) & (s18 <= DET18_HI)
        if is_det.any():
            k = (s18[is_det] - DET18_LO).astype(np.int64)
            tau = t - fdq[is_det]
            tau = np.clip(tau, 0, VST.TAU_MAX)
            mapped[is_det] = (VST.C_BASE + k * VST.NCTAU + tau).astype(np.int8)
        qstate[:, t] = mapped
    qstate[:, NQt + 1] = qstate[:, NQt]
    return qstate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1_000_000)
    ap.add_argument('--chunk', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=97531)
    args = ap.parse_args()

    print(f'[1] real-CMOST no-screening quarterly cohort, n={args.n:,} ...', flush=True)
    t0 = time.time()
    n_chunks = max(1, args.n // args.chunk)
    qr18_chunks = []
    for i in range(n_chunks):
        tC = time.time()
        qr18, risk = run_chunk(args.chunk, args.seed + i)
        qr18_chunks.append(qr18)
        print(f'  chunk {i+1}/{n_chunks} done ({time.time()-tC:.0f}s, total {time.time()-t0:.0f}s)', flush=True)
    qr18 = np.concatenate(qr18_chunks, axis=1)
    del qr18_chunks
    n_total = qr18.shape[1]

    print('[2] extracting first-diagnosis quarter + mapping to 97-state ...', flush=True)
    fdq = first_detect_from_18state(qr18)
    n_dx = int((fdq >= 0).sum())
    print(f'  total diagnosed: {n_dx:,} / {n_total:,}', flush=True)
    for k, name in enumerate(['I', 'II', 'III', 'IV']):
        dxq = fdq[fdq >= 0]
        dxstage = qr18[dxq - 1, np.where(fdq >= 0)[0]] - DET18_LO
        print(f'  stage {name}: n_dx={(dxstage==k).sum():,}', flush=True)

    qstate = map_to_vst97(qr18, fdq)
    del qr18

    print('[3] estimating tau-phased quarterly matrix (VST.estimate_Tq) ...', flush=True)
    T = VST.estimate_Tq(qstate)
    del qstate

    print('[4] propagating no-screening cohort + marginalizing tau (build_T_detected_from_tauphase logic) ...', flush=True)
    T_det = propagate_and_extract(T)

    out = os.path.join(RES, 'T_detected_realcmost_tauphase.npz')
    np.savez_compressed(out, T_detected=T_det, stage_names=np.array(['I', 'II', 'III', 'IV']),
                         n_patients=n_total, n_diagnosed=n_dx)
    print(f'Saved {out}  ({time.time()-t0:.0f}s total)', flush=True)

    print('\nPreview: T_det[age][stage] = [stay, CRC-death, other-death]')
    for age in (55, 65, 75, 85, 95):
        for k, name in enumerate(['I', 'II', 'III', 'IV']):
            print(f'  age {age} stage {name}: {np.round(T_det[age-1,k], 4)}')


if __name__ == '__main__':
    main()
