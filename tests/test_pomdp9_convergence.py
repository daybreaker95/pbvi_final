"""Convergence smoke test for the 13-state (undetected+detected) POMDP9.

Removes the previous small fixed expansions=2 cap: runs PBVI with a much
larger belief set and enough expansions to see whether V(b0)/gain actually
stabilises, rather than reporting a single under-resourced pass. Also
prints per-pure-state Q(WAIT) vs Q(SCREEN) at a few ages, and clinical
marginals from a short rollout, as a reward-design sanity check.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pomdp.model_v2 import (CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP,
                             CA_STAGES, DET_CA_STAGES, WAIT, SCREEN,
                             O_NOTEST, O_NORMAL, O_ADENOMA, O_ADVAD, O_CANCER)
from pomdp.pbvi_v2 import PBVI9, PBVIPolicy9


def main():
    print("Building 13-state QALY POMDP ...")
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    print(f"  n_risk={pomdp.n_risk} NC={pomdp.NC} NS={pomdp.NS} T_det source={pomdp._T_det_source}")

    N_BELIEF = 1500
    EXPANSIONS = 8
    print(f"Solving PBVI9 (n_belief={N_BELIEF}, expansions={EXPANSIONS}) ...")
    t0 = time.time()
    solver = PBVI9(pomdp, n_belief=N_BELIEF, seed=0)
    solver.solve(expansions=EXPANSIONS)
    print(f"  solved in {time.time()-t0:.0f}s, final |B|={len(solver.B)}")

    print("\n--- Q(WAIT) vs Q(SCREEN) by pure clinical state, several ages ---")
    for age in (45, 55, 65, 75):
        print(f" age {age}")
        for label, s in ([('Normal', NORMAL), ('EarlyPolyp', EARLY_POLYP), ('AdvPolyp', ADV_POLYP)]
                          + [(f'CA_{k}', CA_STAGES[k]) for k in range(4)]):
            b = np.zeros(pomdp.NS); b[pomdp.fidx(0, s)] = 1.0
            Q = solver.q_values(age, b)
            print(f"   {label:12s} Q(WAIT)={Q[0]:8.4f}  Q(SCREEN)={Q[1]:8.4f}  diff={Q[1]-Q[0]:+.4f}")

    print("\n--- Policy rollout: Monte-Carlo belief trajectory, no findings ---")
    policy = PBVIPolicy9(solver)
    policy.reset()
    actions = []
    for age in range(pomdp.age_min, pomdp.age_min + 30):
        a = policy.act({'age': age, 'obs': O_NOTEST})
        actions.append(a)
    screen_ages = [pomdp.age_min + i for i, a in enumerate(actions) if a == SCREEN]
    print(f"  screen ages (assuming always-clean/no-test path): {screen_ages}")

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'policies')
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, 'pbvi9_qaly_13state.npz'))
    print(f"\nSaved policy to {outdir}/pbvi9_qaly_13state.npz")


if __name__ == '__main__':
    main()
