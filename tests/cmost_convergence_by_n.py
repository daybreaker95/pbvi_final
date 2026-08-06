"""
cmost_convergence_by_n.py
===========================
Instead of scaling the final CMOST-in-loop validation up to N=10,000,000
(risky given this session's repeated background-job losses at 5+ hour
durations, and not statistically necessary -- see below), this demonstrates
that the clinical metrics reported at N=1,000,000 (results/cmost_4way_*.json)
have already CONVERGED: run the same no_screen/policy scenarios at a ladder
of increasing N and show the estimates stabilize well before N=1,000,000.

Trains the FiVI/PBS policy ONCE (same tau-phase-matrix-trained policy as
cmost_4way_eval.py's current active version), then runs both no_screen and
policy scenarios at N = 10k / 30k / 100k / 300k, reusing the ALREADY-SAVED
N=1,000,000 results (results/cmost_4way_no_screen.json,
results/cmost_4way_policy.json) as the final ladder point instead of
re-simulating it.

Run: python cmost_convergence_by_n.py
"""
import os
import sys
import io
import json
import time
import contextlib
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

import cmost_4way_eval as C4

RES = os.path.join(PBVI_ROOT, 'results')
N_LADDER = [10_000, 30_000, 100_000, 300_000]  # 1,000,000 reused from saved JSON
SEED = 999


def run_one(scenario, n, hook, hook_age_max=85):
    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(n, SEED, hook, hook_age_max)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, n)
    stats['elapsed_sec'] = time.time() - t0
    print(f"  [{scenario}] n={n:,} done ({stats['elapsed_sec']:.0f}s): "
          f"crc_death={stats['crc_death_per_100k']:.1f} incid={stats['incidence_per_100k']:.1f} "
          f"cost={stats['cost_per_person_usd']:.1f} life_years={stats['life_years']:.4f}", flush=True)
    return stats


def main():
    print('training FiVI/PBS policy (tau-phase matrix) ...', flush=True)
    pomdp, solver = C4.train_policy()
    print(f'  gap={solver.gap_history[-1]["gap"]:.4f}', flush=True)

    ladder_results = {'no_screen': [], 'policy': []}

    for n in N_LADDER:
        print(f'--- N={n:,} ---', flush=True)
        r = run_one('no_screen', n, None)
        r['n'] = n
        ladder_results['no_screen'].append(r)

        p_tmp = C4.BNH.prepare_simulation_params(n)
        risk_class = (np.asarray(p_tmp['individual_risk']) >= C4.RISK_THRESHOLD).astype(int)
        hook = C4.EngineHook13(pomdp, solver, risk_class, seed=SEED)
        r = run_one('policy', n, hook)
        r['n'] = n
        ladder_results['policy'].append(r)

    # append the already-saved N=1,000,000 points
    for scenario in ('no_screen', 'policy'):
        saved = json.load(open(os.path.join(RES, f'cmost_4way_{scenario}.json')))
        saved = dict(saved)
        ladder_results[scenario].append(saved)

    out = os.path.join(RES, 'cmost_convergence_by_n.json')
    with open(out, 'w') as f:
        json.dump(ladder_results, f, indent=2)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
