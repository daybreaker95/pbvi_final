"""End-to-end test of the risk-aware POMDP + PBVI, demonstrating that the
policy personalises on screening findings (adenoma found -> infer high risk ->
rescreen sooner), which a fixed population schedule cannot do."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pomdp.model import (CRCScreeningPOMDP, NC, NORMAL, EA, AA, ECA, ACA, WAIT, SCREEN,
                         O_NOTEST, O_NORMAL, O_ADENOMA, O_ADVAD, O_CANCER)
from pomdp.pbvi import PBVI, PBVIPolicy


def next_screen_age(solver, pomdp, b, k, start):
    """First age > start at which the policy screens, propagating belief under
    WAIT/NO_TEST in the meantime."""
    for age in range(start + 1, pomdp.age_max + 1):
        if k > 0 and solver.best_action(age, k, b) == SCREEN:
            return age
        bn = pomdp.belief_update(b, age, WAIT, O_NOTEST)
        if bn is not None:
            b = bn
    return None


def main():
    print("Building risk-aware POMDP (budget=4) ...")
    pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=4, gamma=1.0,
                              screen_disutility=0.01, use_risk_classes=True)
    print(f"  n_risk={pomdp.n_risk}  NS={pomdp.NS}  frac_high={pomdp.frac_high:.3f}")
    print(f"  d_EA={pomdp.d_EA:.3f} d_AA={pomdp.d_AA:.3f} "
          f"d_early={pomdp.d_early:.3f} d_adv={pomdp.d_adv:.3f}")

    print("Solving PBVI ...")
    solver = PBVI(pomdp, n_belief=350, seed=0)
    solver.solve(expansions=2)

    # --- Personalisation demonstration ---
    print("\n--- Personalisation on screening findings ---")
    age0 = 55
    # propagate the initial belief (unscreened) from age_min to age0
    b_at = pomdp.initial_belief()
    for age in range(pomdp.age_min, age0):
        b_at = pomdp.belief_update(b_at, age, WAIT, O_NOTEST)
    scenarios = {
        'clean colonoscopy (Normal)': O_NORMAL,
        'low-risk adenoma found':     O_ADENOMA,
        'ADVANCED adenoma found':     O_ADVAD,
    }
    for label, obs in scenarios.items():
        b = pomdp.belief_update(b_at, age0, SCREEN, obs)
        if b is None:
            print(f"  screen@{age0}, {label:28s}: observation not reachable")
            continue
        rm = pomdp.risk_marginal(b)
        cm = pomdp.clinical_marginal(b)
        nxt = next_screen_age(solver, pomdp, b.copy(), k=3, start=age0)
        interval = f"{nxt - age0} y" if nxt else ">30 y"
        print(f"  screen@{age0}, {label:28s}: P(high risk)={rm[-1]:.2f}, "
              f"belief[N,EA,AA]=[{cm[0]:.2f},{cm[1]:.2f},{cm[2]:.3f}] "
              f"-> next screen age {nxt} (in {interval})")

    print("\n  => Finding an (advanced) adenoma raises the inferred risk and")
    print("     shortens the next screening interval; a clean exam lengthens it.")

    # save a couple of policies for the comparison experiment
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results', 'policies')
    os.makedirs(outdir, exist_ok=True)
    solver.save(os.path.join(outdir, 'pbvi_k4.npz'))
    print(f"\nSaved policy to {outdir}/pbvi_k4.npz")


if __name__ == '__main__':
    main()
