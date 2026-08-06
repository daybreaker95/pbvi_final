"""
estimate_transitions.py
=======================

Empirically estimate the age-stratified 6-state transition matrices of the
CMOST natural history by Monte-Carlo simulation of a large no-screening cohort.

The dynamics are estimated on a QUARTER-year step (matching CMOST's internal
quarterly event loop) and the annual transition matrix that the POMDP consumes
is formed by composition,  P_year[age] = P_quarter[age]^4.  Estimating quarterly
and composing removes the annual-discretization artifacts of a directly-estimated
1-year matrix (which never observes cancers that arise, are diagnosed and kill
within a single year, and so undercounts CRC incidence and the stage-IV share).
See transitions/SCREENING_VALIDATION.md for the quantified effect.

For every simulated patient we record the discretised clinical state
    Normal, Early Adenoma, Advanced Adenoma, Preclinical Cancer,
    Clinical Cancer, Dead
at every quarter of life and tally the one-quarter transitions
    Nq[age, s, s']  =  # patient-quarters observed going from s to s' within `age`.

The maximum-likelihood per-quarter estimator is the row-normalised count, and the
consumed annual matrix is its 4th matrix power.  Multinomial standard errors and a
patient-level bootstrap are provided for the (annual, directly-estimated) matrix
for backward-compatible uncertainty reporting.

Outputs (results/):
  transitions_cmost13.npz
     P             (A,6,6)   ANNUAL transition matrix consumed downstream
                             ( = P_quarter^4 ; quarterly-composed )
     P_quarter     (A,6,6)   MLE one-QUARTER transition probabilities
     P_annual_direct (A,6,6) directly-estimated 1-year matrix (for reference)
     counts        (A,6,6)   yearly-snapshot transition counts (compat)
     counts_quarter(A,6,6)   one-quarter transition counts
     SE            (A,6,6)   multinomial SE of the direct annual matrix
     n_row         (A,6)     yearly samples per (age,from-state)
     prevalence    (A+1,6)   fraction of living cohort in each state by age
     alive         (A+1,)    # alive at each age
     ages          (A,)      the from-ages
     band_edges, P_band, counts_band, SE_band   (5-year pooled, direct annual)
     state_names, n_patients
  state_paths_cmost13.npz
     paths         (n,101)   yearly-snapshot state paths (compat)
     qpaths        (n,401)   quarterly state paths
  transitions_bootstrap_cmost13.npz  (if --bootstrap)
     P_lo, P_hi    (B,6,6)   percentile CI bounds (direct annual, band-pooled)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import CRCEngine, DIAGNOSED, DEAD_OTHER, N_STATES, STATE_NAMES

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
AGE_MIN = 20     # first from-age recorded
AGE_MAX = 99     # last from-age recorded (transition into AGE_MAX+1)
NQ = 100 * 4     # quarters over a 100-year horizon


def simulate_quarterly_paths(n_patients: int, seed: int, settings='CMOST13'):
    """Simulate n patients quarter-by-quarter; return quarterly + yearly paths.

    Returns
    -------
    qpaths : (n, NQ+2) int8    6-state at quarter index t=(y-1)*4+q, t=1..400
    paths  : (n, 101)  int8    yearly snapshot (state at start of year y = q1)
             entries after death are DEAD.
    """
    params = build_params(settings, n_patients=500, seed=seed)
    rng = np.random.default_rng(seed + 1)
    eng = CRCEngine(params, rng=rng)

    qpaths = np.full((n_patients, NQ + 2), DEAD_OTHER, dtype=np.int8)
    t0 = time.time()
    for i in range(n_patients):
        pt = eng.new_patient()
        dead = False
        for y in range(1, 101):
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                qpaths[i, t] = eng.clinical_state(pt)   # state at quarter start
                if not pt.alive:
                    dead = True
                    break
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    # remaining quarters stay DEAD (array pre-filled)
                    dead = True
                    break
            if dead:
                break
        if (i + 1) % 20000 == 0:
            el = time.time() - t0
            print(f"  simulated {i+1}/{n_patients}  ({el:.0f}s, "
                  f"{1000*el/(i+1):.2f} ms/pt)")
    print(f"  done: {n_patients} patients in {time.time()-t0:.0f}s")

    # yearly snapshot = state at the first quarter of each year
    paths = np.full((n_patients, 101), DEAD_OTHER, dtype=np.int8)
    for y in range(1, 101):
        paths[:, y] = qpaths[:, (y - 1) * 4 + 1]
    return qpaths, paths


def simulate_state_paths(n_patients: int, seed: int, settings='CMOST13'):
    """Backward-compatible yearly-path simulator (quarter-accurate underneath)."""
    _, paths = simulate_quarterly_paths(n_patients, seed, settings)
    return paths


def tally_quarterly(qpaths: np.ndarray):
    """Tally one-QUARTER transition counts, grouped by the from-quarter's year."""
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    A = len(ages)
    counts = np.zeros((A, N_STATES, N_STATES), dtype=np.int64)
    for ai, age in enumerate(ages):
        for q in (1, 2, 3, 4):
            t = (age - 1) * 4 + q
            s_from = qpaths[:, t]
            s_to = qpaths[:, t + 1]
            for s in range(N_STATES):
                m = s_from == s
                if not m.any():
                    continue
                tos = s_to[m]
                for sp in range(N_STATES):
                    counts[ai, s, sp] += np.count_nonzero(tos == sp)
    return ages, counts


def compose_annual(P_quarter: np.ndarray):
    """Annual matrix as the 4th matrix power of the per-quarter matrix (per age)."""
    P_year = np.empty_like(P_quarter)
    for ai in range(P_quarter.shape[0]):
        P_year[ai] = np.linalg.matrix_power(P_quarter[ai], 4)
    return P_year


def tally_transitions(paths: np.ndarray):
    """Tally one-year transition counts by from-age from a yearly state-path array."""
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    A = len(ages)
    counts = np.zeros((A, N_STATES, N_STATES), dtype=np.int64)
    alive = np.zeros(A + 1, dtype=np.int64)
    prevalence = np.zeros((A + 1, N_STATES), dtype=np.float64)

    for ai, age in enumerate(ages):
        s_from = paths[:, age]
        s_to = paths[:, age + 1]
        for s in range(N_STATES):
            m = s_from == s
            if not m.any():
                continue
            tos = s_to[m]
            for sp in range(N_STATES):
                counts[ai, s, sp] = np.count_nonzero(tos == sp)

    # prevalence among the living, at each age. NOTE: DIAGNOSED conflates
    # "alive with a diagnosed cancer" and "dead of CRC" (CRC death always
    # follows a prior diagnosis in this engine, and the active POMDP doesn't
    # need to tell them apart -- see clinical_state() docstring), so "living"
    # here undercounts true deaths by however many are CRC deaths.
    for k, age in enumerate(np.arange(AGE_MIN, AGE_MAX + 2)):
        col = paths[:, age]
        living = col != DEAD_OTHER
        alive_n = np.count_nonzero(living)
        alive[k] = alive_n
        if alive_n:
            for s in range(N_STATES):
                prevalence[k, s] = np.count_nonzero(col[living] == s) / alive_n
    return ages, counts, alive, prevalence


def normalise(counts: np.ndarray):
    """Row-normalise counts -> (P, SE, n_row)."""
    A = counts.shape[0]
    P = np.zeros_like(counts, dtype=np.float64)
    SE = np.zeros_like(counts, dtype=np.float64)
    n_row = counts.sum(axis=2)              # (A,6)
    for ai in range(A):
        for s in range(N_STATES):
            n = n_row[ai, s]
            if n > 0:
                p = counts[ai, s] / n
                P[ai, s] = p
                SE[ai, s] = np.sqrt(np.clip(p * (1 - p), 0, None) / n)
            else:
                # no observations: identity row (state persists) as a prior
                P[ai, s, s] = 1.0
    # DIAGNOSED and DEAD_OTHER are absorbing by construction
    for absorb in (DIAGNOSED, DEAD_OTHER):
        P[:, absorb, :] = 0.0
        P[:, absorb, absorb] = 1.0
    return P, SE, n_row


def band_pool(ages, counts, width=5):
    """Pool transition counts into `width`-year age bands."""
    edges = list(range(int(ages[0]), int(ages[-1]) + 2, width))
    B = len(edges) - 1
    counts_band = np.zeros((B, N_STATES, N_STATES), dtype=np.int64)
    for bi in range(B):
        lo, hi = edges[bi], edges[bi + 1]
        mask = (ages >= lo) & (ages < hi)
        counts_band[bi] = counts[mask].sum(axis=0)
    P_band, SE_band, _ = normalise(counts_band)
    return np.array(edges), P_band, SE_band, counts_band


def bootstrap_ci(paths, n_boot=200, alpha=0.05, seed=0):
    """Patient-level non-parametric bootstrap CI for band-pooled (direct annual) P."""
    rng = np.random.default_rng(seed)
    n = paths.shape[0]
    boots = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ages, counts, _, _ = tally_transitions(paths[idx])
        _, P_band, _, _ = band_pool(ages, counts)
        boots.append(P_band)
        if (b + 1) % 50 == 0:
            print(f"    bootstrap {b+1}/{n_boot}")
    boots = np.array(boots)                # (n_boot, B, 6, 6)
    lo = np.percentile(boots, 100 * alpha / 2, axis=0)
    hi = np.percentile(boots, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--n_patients', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--settings', default='CMOST13')
    ap.add_argument('--bootstrap', type=int, default=0,
                    help='number of bootstrap resamples (0 = skip)')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Estimating QUARTERLY transitions: n={args.n_patients}, settings={args.settings}")
    qpaths, paths = simulate_quarterly_paths(args.n_patients, args.seed, args.settings)

    # quarterly estimate + annual composition (the consumed matrix)
    ages, counts_q = tally_quarterly(qpaths)
    P_quarter, _, _ = normalise(counts_q)
    P = compose_annual(P_quarter)

    # yearly-snapshot direct annual estimate (reference + prevalence/alive + CI)
    _, counts, alive, prevalence = tally_transitions(paths)
    P_annual_direct, SE, n_row = normalise(counts)
    edges, P_band, SE_band, counts_band = band_pool(ages, counts)

    out = os.path.join(RESULTS_DIR, f'transitions_{args.settings.lower()}.npz')
    np.savez_compressed(
        out, P=P, P_quarter=P_quarter, P_annual_direct=P_annual_direct,
        counts=counts, counts_quarter=counts_q, SE=SE, n_row=n_row,
        prevalence=prevalence, alive=alive, ages=ages,
        band_edges=edges, P_band=P_band, counts_band=counts_band, SE_band=SE_band,
        state_names=np.array(STATE_NAMES), n_patients=args.n_patients,
    )
    print(f"Saved {out}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, f'state_paths_{args.settings.lower()}.npz'),
        paths=paths, qpaths=qpaths)

    if args.bootstrap > 0:
        print(f"Bootstrapping ({args.bootstrap} resamples) ...")
        lo, hi = bootstrap_ci(paths, n_boot=args.bootstrap, seed=args.seed)
        np.savez_compressed(
            os.path.join(RESULTS_DIR, f'transitions_bootstrap_{args.settings.lower()}.npz'),
            P_lo=lo, P_hi=hi, band_edges=edges)
        print("Saved bootstrap CI")

    # quick console preview at a few ages (annual, composed)
    print("\nPreview -- composed ANNUAL transition rows at selected ages (%):")
    for age in (50, 60, 65, 70):
        ai = age - AGE_MIN
        print(f"\n age {age}:  quarter n_row={counts_q[ai].sum(axis=1)}")
        for s in range(N_STATES):
            row = " ".join(f"{100*P[ai,s,sp]:6.2f}" for sp in range(N_STATES))
            print(f"   {STATE_NAMES[s][:18]:18s} -> {row}")


if __name__ == '__main__':
    main()
