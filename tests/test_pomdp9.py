"""End-to-end smoke test for the 9-state, budget-free QALY POMDP
(pomdp/model_v2.py + pomdp/pbvi_v2.py). Separate from tests/test_pomdp.py,
which stays on the original 7-state/budget-aware pipeline."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pomdp.model_v2 import (CRCScreeningPOMDP9, NORMAL, WAIT, SCREEN,
                             O_NOTEST, O_NORMAL, O_ADENOMA, O_ADVAD, O_CANCER)
from pomdp.pbvi_v2 import PBVI9, PBVIPolicy9


def next_screen_age(solver, pomdp, b, start):
    for age in range(start + 1, pomdp.age_max + 1):
        if solver.best_action(age, b) == SCREEN:
            return age
        bn = pomdp.belief_update(b, age, WAIT, O_NOTEST)
        if bn is not None:
            b = bn
    return None


def main():
    print("Building 9-state QALY POMDP (no budget) ...")
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    print(f"  n_risk={pomdp.n_risk}  NS={pomdp.NS}  NC={pomdp.NC}  T_det source={pomdp._T_det_source}")
    print(f"  d_EA={pomdp.d_EA:.3f} d_AA={pomdp.d_AA:.3f} d_PC_stage={np.round(pomdp.d_PC_stage,3).tolist()}")

    print("Solving PBVI9 ...")
    solver = PBVI9(pomdp, n_belief=350, seed=0)
    solver.solve(expansions=2)

    print("\n--- Personalisation on screening findings ---")
    age0 = 55
    b_at = pomdp.initial_belief()
    for age in range(pomdp.age_min, age0):
        b_at = pomdp.belief_update(b_at, age, WAIT, O_NOTEST)
    scenarios = {
        'clean colonoscopy (Normal)': O_NORMAL,
        'low-risk polyp found': O_ADENOMA,
        'ADVANCED polyp found': O_ADVAD,
    }
    for label, obs in scenarios.items():
        b = pomdp.belief_update(b_at, age0, SCREEN, obs)
        if b is None:
            print(f"  screen@{age0}, {label:28s}: observation not reachable")
            continue
        cm = pomdp.clinical_marginal(b)
        nxt = next_screen_age(solver, pomdp, b.copy(), start=age0)
        print(f"  screen@{age0}, {label:28s}: belief[N,EarlyP,AdvP]="
              f"[{cm[0]:.2f},{cm[1]:.2f},{cm[2]:.3f}] -> next screen age {nxt}")

    print("\n--- Policy rollout smoke test (belief-tracking end to end) ---")
    policy = PBVIPolicy9(solver)
    policy.reset()
    obs = {'age': pomdp.age_min, 'obs': O_NOTEST}
    actions = []
    for age in range(pomdp.age_min, pomdp.age_min + 15):
        obs = {'age': age, 'obs': O_NOTEST}
        a = policy.act(obs)
        actions.append(a)
    print(f"  first 15 actions from age {pomdp.age_min} (0=WAIT,1=SCREEN): {actions}")

    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'policies')
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, 'pbvi9_qaly.npz'))
    print(f"\nSaved policy to {outdir}/pbvi9_qaly.npz")


if __name__ == '__main__':
    main()
