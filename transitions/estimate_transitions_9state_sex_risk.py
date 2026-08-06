"""
estimate_transitions_9state_sex_risk.py
========================================
9-state T_undetected/d_symp, cross-stratified by SEX and adenoma-risk class
(4 combinations: male-low, male-high, female-low, female-high).

CMOST bakes sex into the disease dynamics themselves, not just competing
mortality: polyp onset rate (new_polyp_female multiplier), adenoma
progression speed (gender_progression[stage, gender]), and de-novo cancer
rate (direct_cancer_rate[gender, age]) all differ by sex in
env/cmost_individual.py's _step_quarter(). So this -- unlike T_detected's
CRC-hazard, which IS gender-independent (mortality_matrix has no gender
axis) -- genuinely needs separate estimation, not just a reward/cost tweak.

individual_risk itself is sampled independently of gender (calculate_sub.py
draws it with no gender conditioning), so ONE population-wide top-25%
threshold is used for both sexes (verified empirically below: frac_high
comes out ~0.25 within each sex).

Does NOT touch transitions_9state.npz / transitions_9state_stratified.npz --
writes to transitions_9state_sex_risk.npz.

Run: python estimate_transitions_9state_sex_risk.py -n 200000
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


def simulate_with_sex_risk(n, seed, settings='CMOST13'):
    params = build_params(settings, n_patients=20000, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
    qstate = np.full((n, NQ + 2), OTHER_DEATH, dtype=np.int8)
    first_detect_q = np.full(n, -1, dtype=np.int32)
    gender = np.zeros(n, dtype=np.int8)
    risk = np.zeros(n, dtype=float)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        gender[i] = pt.gender
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
    return qstate, first_detect_q, gender, risk


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
    ap.add_argument('--high_frac', type=float, default=0.25)
    args = ap.parse_args()

    print(f"Simulating n={args.n} (no screening, quarterly, 9-state, sex+individual_risk) ...")
    qstate, fdq, gender, risk = simulate_with_sex_risk(args.n, args.seed)

    thr = float(np.quantile(risk, 1 - args.high_frac))
    risk_hi = risk >= thr
    male = gender == 1
    print(f"risk threshold (pop-wide {1-args.high_frac:.0%} quantile) = {thr:.4f}")
    print(f"frac_female={float((~male).mean()):.3f}  "
          f"frac_high|male={float(risk_hi[male].mean()):.3f}  "
          f"frac_high|female={float(risk_hi[~male].mean()):.3f}")

    combos = {
        'male_low': male & ~risk_hi, 'male_high': male & risk_hi,
        'female_low': ~male & ~risk_hi, 'female_high': ~male & risk_hi,
    }
    out = {}
    for name, mask in combos.items():
        print(f"Estimating {name} (n={mask.sum()}) ...")
        P, d = estimate_for_mask(qstate, fdq, mask)
        out[f'P_undetected_{name}'] = P
        out[f'd_symp_{name}'] = d
        inc = 100 * (fdq[mask] >= 0).mean()
        print(f"  lifetime diagnosed-CRC incidence (no screening) = {inc:.2f}%")

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    outpath = os.path.join(RES, 'transitions_9state_sex_risk.npz')
    np.savez_compressed(
        outpath, ages=ages, risk_threshold=thr,
        frac_female=float((~male).mean()),
        frac_high_male=float(risk_hi[male].mean()),
        frac_high_female=float(risk_hi[~male].mean()),
        state_names=np.array(STATE9_NAMES), n_patients=args.n,
        **out,
    )
    print(f"Saved {outpath}")


if __name__ == '__main__':
    main()
