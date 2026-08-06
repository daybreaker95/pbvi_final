"""Throttle add_reachable_beliefs to add FEW points per pass (small
n_rollouts), over MANY passes, to show the belief-set growth / value
trajectory at finer granularity -- and confirm it lands on the same final
value as the coarse (big n_rollouts, few passes) runs."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL
from pomdp.pbvi_v2 import PBVI9

p = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
solver = PBVI9(p, n_belief=50, seed=0)   # tiny initial seed set on purpose
solver.Vnat = p.natural_value()
b0 = p.initial_belief()
vnat = solver.Vnat[p.age_min][NORMAL]

EXPANSIONS = 30
print(f"initial |B| = {len(solver.B)}")
print(f"{'pass':>4} {'|B|':>6} {'V(b0)':>10} {'gain':>10}")
for it in range(EXPANSIONS + 1):
    for age in range(p.age_max, p.age_min - 1, -1):
        solver.Gamma[age] = solver._backup(age)
    v = solver.value(p.age_min, b0)
    print(f"{it:>4} {len(solver.B):>6} {v:>10.5f} {v-vnat:>10.5f}")
    if it < EXPANSIONS:
        solver.add_reachable_beliefs(n_rollouts=8, horizon=45, max_belief=10000)

print(f"\nfinal |B| = {len(solver.B)}")
