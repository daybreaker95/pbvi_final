"""
estimate_transitions_9state_jeon_risk.py
==========================================
Like estimate_transitions_9state_nhic_diet_risk.py, but:
  1. ALL risk-factor HRs come from Jeon et al. 2018 alone (tests/
     jeon_elbow_analysis.py), not a Korean-NHIC-core + Jeon-diet mix.
  2. Mapping onto CMOST's individual_risk pool uses bucket_map_to_
     individual_risk (binary high/low bucket draw) instead of the
     continuous rank-preserving percentile_map_to_individual_risk -- the
     agent only needs to see a high/low label, not a fine-grained rank.

Writes transitions_9state_sex_risk_jeon20pct.npz.

Run: python estimate_transitions_9state_jeon_risk.py -n 200000 --high_frac 0.20
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

from env.params import build_params
from env.cmost_individual import CRCEngine
from env.state9 import clinical_state9, N_STATES9, STATE9_NAMES, OTHER_DEATH
from estimate_transitions_9state import (
    tally, normalise_quarter, compose_annual, MAXY, NQ, AGE_MIN, AGE_MAX, CA_STAGES,
)
from jeon_elbow_analysis import assign_profiles, bucket_map_to_individual_risk

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


def simulate_with_jeon_risk(n, seed, params, high_frac):
    indiv_risk_pool = np.asarray(params['individual_risk'], float)
    print(f'  individual_risk pool: n={len(indiv_risk_pool)}  unique={len(np.unique(indiv_risk_pool))}', flush=True)

    print(f'  assigning {n:,} Jeon-2018-only (12-factor) virtual profiles ...', flush=True)
    prof = assign_profiles(n, seed)
    sex_np, composite_rr = prof['sex'], prof['composite_rr']
    male = sex_np == 1
    print(f'  composite_rr (male):   mean={composite_rr[male].mean():.4f}  unique={len(np.unique(composite_rr[male]))}', flush=True)
    print(f'  composite_rr (female): mean={composite_rr[~male].mean():.4f}  unique={len(np.unique(composite_rr[~male]))}', flush=True)

    rng = np.random.default_rng(seed + 3)
    mapped_risk, risk_hi_by_prof = bucket_map_to_individual_risk(sex_np, composite_rr, indiv_risk_pool, high_frac, rng)

    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
    qstate = np.full((n, NQ + 2), OTHER_DEATH, dtype=np.int8)
    first_detect_q = np.full(n, -1, dtype=np.int32)
    gender = np.zeros(n, dtype=np.int8)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient(gender=int(sex_np[i]), individual_risk=float(mapped_risk[i]))
        gender[i] = pt.gender
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
            print(f"    {i+1}/{n}  ({el:.0f}s, {1000*el/(i+1):.3f} ms/pt)", flush=True)
    print(f"  done {n} in {time.time()-t0:.0f}s", flush=True)

    # NOTE: risk_hi is defined by prof's sex labels (index-aligned with
    # composite_rr / mapped_risk), same convention as risk_hi_by_prof above.
    return qstate, first_detect_q, gender, mapped_risk, risk_hi_by_prof


def estimate_for_mask(qstate, fdq, mask):
    counts_stay, counts_detect, n_row = tally(qstate[mask], fdq[mask])
    Pq_stay, dq = normalise_quarter(counts_stay, counts_detect, n_row)
    P_year, d_year = compose_annual(Pq_stay, dq)
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    return P_year[ages], d_year[ages]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=24680)
    ap.add_argument('--high_frac', type=float, default=0.20)
    ap.add_argument('--tag', type=str, default='jeon20pct')
    args = ap.parse_args()

    print(f'Building params / life table ...', flush=True)
    params = build_params('CMOST13', n_patients=20000, seed=args.seed)

    print(f"Simulating n={args.n} (no screening, quarterly, 9-state, Jeon-2018 bucket risk) ...", flush=True)
    qstate, fdq, gender, mapped_risk, risk_hi = simulate_with_jeon_risk(
        args.n, args.seed, params, args.high_frac)

    male = gender == 1
    print(f"frac_female={float((~male).mean()):.3f}  "
          f"frac_high|male={float(risk_hi[male].mean()):.3f}  "
          f"frac_high|female={float(risk_hi[~male].mean()):.3f}", flush=True)

    combos = {
        'male_low': male & ~risk_hi, 'male_high': male & risk_hi,
        'female_low': ~male & ~risk_hi, 'female_high': ~male & risk_hi,
    }
    out = {}
    for name, mask in combos.items():
        print(f"Estimating {name} (n={mask.sum()}) ...", flush=True)
        P, d = estimate_for_mask(qstate, fdq, mask)
        out[f'P_undetected_{name}'] = P
        out[f'd_symp_{name}'] = d
        inc = 100 * (fdq[mask] >= 0).mean()
        print(f"  lifetime diagnosed-CRC incidence (no screening) = {inc:.2f}%", flush=True)

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    outpath = os.path.join(RES, f'transitions_9state_sex_risk_{args.tag}.npz')
    np.savez_compressed(
        outpath, ages=ages,
        high_frac=args.high_frac,
        frac_female=float((~male).mean()),
        frac_high_male=float(risk_hi[male].mean()),
        frac_high_female=float(risk_hi[~male].mean()),
        state_names=np.array(STATE9_NAMES), n_patients=args.n,
        **out,
    )
    print(f'Saved {outpath}', flush=True)


if __name__ == '__main__':
    main()
