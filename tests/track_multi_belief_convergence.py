"""Track V(age_min, b) across PBVI passes for SEVERAL test beliefs (not
just b0), to check whether refinement is happening elsewhere in belief
space even where b0 itself looks flat."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP, CA_STAGES
from pomdp.pbvi_v2 import PBVI9

p = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
solver = PBVI9(p, n_belief=1500, seed=0)
solver.Vnat = p.natural_value()

test = {'b0 (pure Normal)': p.initial_belief()}
for label, s in [('pure EarlyPolyp', EARLY_POLYP), ('pure AdvPolyp', ADV_POLYP)] + \
                [(f'pure CA_{k}', CA_STAGES[k]) for k in range(4)]:
    b = np.zeros(p.NS); b[p.fidx(0, s)] = 1.0
    test[label] = b
# a mixed, more "realistic" belief: mostly normal with a small tail
mix = np.zeros(p.NS)
mix[p.fidx(0, NORMAL)] = 0.9
mix[p.fidx(0, EARLY_POLYP)] = 0.07
mix[p.fidx(0, ADV_POLYP)] = 0.02
mix[p.fidx(0, CA_STAGES[0])] = 0.01
test['mixed (90N/7EP/2AP/1CA_I)'] = mix

EXPANSIONS = 10
history = {k: [] for k in test}
for it in range(EXPANSIONS + 1):
    for age in range(p.age_max, p.age_min - 1, -1):
        solver.Gamma[age] = solver._backup(age)
    for label, b in test.items():
        history[label].append(solver.value(p.age_min, b))
    print(f"pass {it}: |B|={len(solver.B)}  " +
          "  ".join(f"{k}={history[k][-1]:.4f}" for k in test))
    if it < EXPANSIONS:
        solver.add_reachable_beliefs(n_rollouts=200, horizon=45, max_belief=6000)

print("\n--- summary: first vs last pass ---")
for label in test:
    h = history[label]
    print(f"{label:28s} pass0={h[0]:.4f}  last={h[-1]:.4f}  delta={h[-1]-h[0]:+.4f}")
