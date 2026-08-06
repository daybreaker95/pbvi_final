"""
build_T_detected_from_tauphase.py
==================================
Builds a MORE ACCURATE T_detected[age][stage] (annual: self / CRC-death /
other-death) for pomdp/model_v2.py's 9-state POMDP, by reusing the ALREADY-
VALIDATED 97-state tau-phase machinery (verify_screening_tauphase.py) as a
CALIBRATION TOOL -- not as an active POMDP dimension.

Method: propagate a no-screening cohort through the 97-state QUARTERLY
transition matrix (same estimation as verify_screening_tauphase.py), which
naturally carries the REALISTIC mix of tau (quarters-since-diagnosis) at
every age, for every stage. At each age, marginalize (occupancy-weight)
across tau to get the population-average CRC-death / other-death / stay
hazard for that stage -- this captures the front-loaded timing's AGGREGATE
effect at each age WITHOUT exposing tau as a belief-tracked state.

Output: results/T_detected_tauphase.npz
  T_detected (99,4,3) -- [age-1][stage 0=I..3=IV] = [p_stay, p_crc, p_other]

This REPLACES the naive same-year mortality_matrix sampling previously used
in pomdp/model_v2.py's _build_T_detected().

Run: python build_T_detected_from_tauphase.py -n 100000
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
from env.params import build_params
from env.cmost_individual import CRCEngine
import verify_screening_tauphase as VST

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
MAXY, NS = VST.MAXY, VST.NS
NCTAU, DOTH, DCRC, N = VST.NCTAU, VST.DOTH, VST.DCRC, VST.N
cidx = VST.cidx


def propagate_and_extract(T, pooled_other_death=True):
    """No-screening 97-state propagation; at the start of each age-year,
    marginalize occupancy across tau to get an annual {stay,crc,oth} hazard
    per stage, then step the full dist forward one year (4 quarters).

    pooled_other_death: OTHER_DEATH is driven purely by LifeTable[age,
    gender] in the CMOST engine -- it does not depend on cancer stage OR
    tau (time-since-diagnosis) at all. Estimating it separately per
    (stage,tau) tau-clock state starves old-age/late-stage cells (few
    survivors reach them) of sample, producing noisy per-cell rates (empty
    cells go to 0). Instead pool p_oth_q across ALL clinical (stage,tau)
    states at that age (mass-weighted by the SAME propagating `dist` used
    for p_crc_q), and use that single pooled value for every stage k --
    p_crc_q stays fully stage-specific (that genuinely varies by stage)."""
    dist = np.zeros(NS); dist[N] = 1.0
    T_det = np.zeros((MAXY, 4, 3))
    for age in range(1, MAXY + 1):
        Ty = T[min(age, MAXY)]
        if pooled_other_death:
            all_clin_idx = [cidx(kk, tau) for kk in range(4) for tau in range(NCTAU)]
            mass_all = dist[all_clin_idx].sum()
            p_oth_pooled = (sum(dist[i] * Ty[i, DOTH] for i in all_clin_idx) / mass_all
                             if mass_all > 1e-12 else 0.0)
        for k in range(4):
            tau_idx = [cidx(k, tau) for tau in range(NCTAU)]
            mass = dist[tau_idx].sum()
            if mass > 1e-12:
                p_crc_q = sum(dist[i] * Ty[i, DCRC] for i in tau_idx) / mass
                if pooled_other_death:
                    p_oth_q = p_oth_pooled
                else:
                    p_oth_q = sum(dist[i] * Ty[i, DOTH] for i in tau_idx) / mass
                p_stay_q = max(0.0, 1 - p_crc_q - p_oth_q)
                Q = np.array([[p_stay_q, p_crc_q, p_oth_q], [0, 1, 0], [0, 0, 1]])
                T_det[age - 1, k] = np.linalg.matrix_power(Q, 4)[0]
            else:
                T_det[age - 1, k] = [1.0, 0.0, 0.0]
        for _qq in (1, 2, 3, 4):
            dist = dist @ Ty
    return T_det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=24681)
    args = ap.parse_args()

    eng = CRCEngine(build_params('CMOST13', 20000, args.seed), rng=np.random.default_rng(0))

    print(f"[1] no-screening microsim (n={args.n}, quarterly, 97-state layout) ...")
    _, paths = VST.run_microsim_q(eng, args.n, [], record_paths=True, seed=args.seed)
    print("[2] estimating tau-phased quarterly transition matrix ...")
    T = VST.estimate_Tq(paths)

    print("[3] propagating no-screening cohort + marginalizing tau per age/stage ...")
    T_det = propagate_and_extract(T)

    out = os.path.join(RES, 'T_detected_tauphase.npz')
    np.savez_compressed(out, T_detected=T_det, stage_names=np.array(['I', 'II', 'III', 'IV']))
    print(f"Saved {out}")

    print("\nPreview: T_detected[age][stage] = [stay, CRC-death, other-death]")
    for age in (45, 55, 65, 75):
        for k, name in enumerate(['I', 'II', 'III', 'IV']):
            print(f"  age {age} stage {name}: {np.round(T_det[age-1,k], 4)}")


if __name__ == '__main__':
    main()
