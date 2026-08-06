"""Gathers PBVI9 convergence + reward-trend + clinical-outcome data into a
single JSON for visualization. Does not modify pbvi_v2.py/model_v2.py --
replicates solve()'s loop here just to capture per-pass history that
solve() only prints."""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP, CA_STAGES, WAIT, SCREEN
from pomdp.pbvi_v2 import PBVI9
from mc_eval_policy9 import simulate, summarize

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'convergence_viz_data.json')


def main():
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    solver = PBVI9(pomdp, n_belief=1500, seed=0)
    solver.Vnat = pomdp.natural_value()

    EXPANSIONS = 10
    passes = []
    for it in range(EXPANSIONS + 1):
        for age in range(pomdp.age_max, pomdp.age_min - 1, -1):
            solver.Gamma[age] = solver._backup(age)
        b0 = pomdp.initial_belief()
        v = solver.value(pomdp.age_min, b0)
        vnat = solver.Vnat[pomdp.age_min][NORMAL]
        passes.append({'pass': it, 'n_belief': len(solver.B), 'V_policy': v,
                        'V_noscreen': vnat, 'gain': v - vnat})
        print(f"pass {it}: |B|={len(solver.B)} V={v:.4f} gain={v-vnat:.4f}")
        if it < EXPANSIONS:
            solver.add_reachable_beliefs(n_rollouts=200, horizon=45, max_belief=6000)

    # Q(WAIT) vs Q(SCREEN) trend by age, for representative pure states
    reward_trend = {}
    labels = {'Normal': NORMAL, 'EarlyPolyp': EARLY_POLYP, 'AdvPolyp': ADV_POLYP,
              'CA_I': CA_STAGES[0], 'CA_II': CA_STAGES[1], 'CA_III': CA_STAGES[2], 'CA_IV': CA_STAGES[3]}
    ages = list(range(pomdp.age_min, pomdp.age_max + 1))
    for label, s in labels.items():
        qw, qs = [], []
        for age in ages:
            b = np.zeros(pomdp.NS); b[pomdp.fidx(0, s)] = 1.0
            Q = solver.q_values(age, b)
            qw.append(Q[WAIT]); qs.append(Q[SCREEN])
        reward_trend[label] = {'ages': ages, 'Q_WAIT': qw, 'Q_SCREEN': qs,
                                'diff': [a - b for a, b in zip(qs, qw)]}

    # clinical MC outcomes: no-screen vs policy
    r_pol = simulate(pomdp, solver, 30000, seed=1, policy=True)
    r_nos = simulate(pomdp, solver, 30000, seed=1, policy=False)
    s_pol, s_nos = summarize(r_pol), summarize(r_nos)

    data = {
        'passes': passes,
        'reward_trend': reward_trend,
        'clinical': {
            'no_screen': {'avg_age_all': s_nos['avg_age_all'], 'avg_age_crc': s_nos['avg_age_crc'],
                          'crc_100k': s_nos['crc_100k'], 'incid_100k': s_nos['incid_100k'],
                          'stage_pct': s_nos['stage_pct'].tolist()},
            'policy': {'avg_age_all': s_pol['avg_age_all'], 'avg_age_crc': s_pol['avg_age_crc'],
                       'crc_100k': s_pol['crc_100k'], 'incid_100k': s_pol['incid_100k'],
                       'stage_pct': s_pol['stage_pct'].tolist()},
        },
    }
    with open(OUT, 'w') as f:
        json.dump(data, f)
    print(f"\nSaved -> {OUT}")


if __name__ == '__main__':
    main()
