"""
run_comparison.py
=================

Head-to-head comparison, in the TRUE CMOST gym environment, of

  * No screening
  * Guideline colonoscopy (every 10 y, 45-75)
  * Zaika 2024 optimal FIXED population schedules (1-4 colonoscopies)
  * PBVI adaptive, history-dependent POMDP policy (1-4 colonoscopy budget)

The PBVI policy tracks a belief over (latent risk class x clinical state) and
therefore individualises timing on realised colonoscopy findings.  We report,
for each policy, mean life-years, CRC incidence/mortality, mean number of
screening colonoscopies, life-years gained (LYG) vs no screening, and LYG per
colonoscopy, using per-patient common random numbers for variance reduction.

Produces:
  results/comparison_table.csv, results/comparison.json
  paper/figures/efficiency_frontier.png
  paper/figures/pbvi_adaptivity.png
"""

from __future__ import annotations

import os
import sys
import json
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import CRCEngine
from env.crc_env import CRCScreeningEnv, EnvConfig, NoScreening, FixedAgeSchedule, UniformInterval
from pomdp.model import CRCScreeningPOMDP
from pomdp.pbvi import PBVI, PBVIPolicy

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(os.path.join(RES, 'policies'), exist_ok=True)
os.makedirs(PAPER_FIG, exist_ok=True)

# Zaika 2024 optimal fixed colonoscopy schedules (life-years objective),
# read from the paper's abstract/results (ages of the 1-4 colonoscopies).
ZAIKA_LYG = {
    1: [55],
    2: [49, 64],
    3: [45, 57, 69],
    4: [40, 51, 61, 72],
}


def solve_pbvi(budget, disutility=0.0, gamma=1.0, cache=True, verbose=False,
               n_belief=700, expansions=4):
    """Solve the budget-`budget` PBVI policy.

    Hyperparameters were tuned for the quarterly-composed natural-history model:
    no screen-disutility (the budget cap already limits colonoscopy use), a larger
    belief set (700) and more belief-expansion passes (4) noticeably improve the
    mid-budget policies (e.g. budget 3 LYG 85 -> 91, matching the best fixed
    schedule at fewer colonoscopies) relative to the pre-rebuild (1 expansion,
    350 beliefs, disutility 0.01) settings."""
    path = os.path.join(RES, 'policies', f'pbvi_k{budget}.npz')
    pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=budget,
                              gamma=gamma, screen_disutility=disutility,
                              use_risk_classes=True)
    solver = PBVI(pomdp, n_belief=n_belief, seed=0)
    if cache and os.path.exists(path):
        solver.solve(expansions=0, verbose=False)   # build Vnat + shapes
        solver.load(path)
    else:
        solver.solve(expansions=expansions, verbose=verbose)
        solver.save(path)
    return pomdp, solver


def find_best_fixed_greedy(N, ref, cfg, n_search=2500, seed=4242,
                           grid=range(46, 79, 3)):
    """Greedy forward search for a strong FIXED N-colonoscopy schedule *in this
    environment* (fair, re-optimized baseline vs Zaika's published ages, which
    were tuned on a different CMOST calibration). Maximizes LYG per 1000."""
    chosen = []
    for _ in range(N):
        best_age, best_lyg = None, -1e9
        for age in grid:
            if any(abs(age - c) < 5 for c in chosen):
                continue
            sched = sorted(chosen + [age])
            a = eval_policy(FixedAgeSchedule(sched), n_search, seed, cfg)
            lyg = 1000 * (ref['ly_lost_to_crc'].mean() - a['ly_lost_to_crc'].mean())
            if lyg > best_lyg:
                best_lyg, best_age = lyg, age
        if best_age is None:
            break
        chosen = sorted(chosen + [best_age])
    return chosen


def eval_policy(policy, n, seed, config, settings='CMOST13'):
    params = build_params(settings, n_patients=500, seed=777)
    eng = CRCEngine(params, rng=np.random.default_rng(0))
    env = CRCScreeningEnv(eng, config)
    rows = []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + i)
        rows.append(env.rollout(policy))
    a = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    return a


def summarize(a, ref):
    n = len(a['life_years'])
    lost = a['ly_lost_to_crc'].mean()
    return {
        'life_years': float(a['life_years'].mean()),
        'crc_incidence': float(100 * a['ever_clinical'].mean()),
        'crc_mortality': float(100 * a['crc_death'].mean()),
        'mean_colo': float(a['n_colonoscopies'].mean()),
        'ly_lost_to_crc': float(lost),
        'LYG_per_1000': float(1000 * (ref['ly_lost_to_crc'].mean() - lost)),
        'inc_reduction': float(100 * (ref['ever_clinical'].mean() - a['ever_clinical'].mean())),
        'mort_reduction': float(100 * (ref['crc_death'].mean() - a['crc_death'].mean())),
        'LYG_per_colo': float(1000 * (ref['ly_lost_to_crc'].mean() - lost)
                              / max(a['n_colonoscopies'].mean(), 1e-9)),
    }


def main(n=12000, seed=90210):
    # decisions begin at 40 (POMDP age_min); natural history is warmed up 1-39.
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)

    print("Reference (no screening) ...", flush=True)
    ref = eval_policy(NoScreening(), n, seed, cfg)

    print("Re-optimizing best FIXED schedules in CMOST13 (greedy) ...", flush=True)
    best_fixed = {}
    for K in (1, 2, 3, 4):
        best_fixed[K] = find_best_fixed_greedy(K, ref, cfg)
        print(f"  best fixed x{K}: {best_fixed[K]}", flush=True)
    json.dump(best_fixed, open(os.path.join(RES, 'best_fixed_schedules.json'), 'w'))

    print("Solving PBVI policies for budgets 1-4 ...", flush=True)
    solvers = {}
    for K in (1, 2, 3, 4):
        pomdp, solver = solve_pbvi(K, cache=False, verbose=False)
        solvers[K] = (pomdp, solver)

    policies = {'Guideline q10y 45-75': UniformInterval(10, 45, 75)}
    for K in (1, 2, 3, 4):
        policies[f'Best fixed x{K}'] = FixedAgeSchedule(best_fixed[K])
    for K in (1, 2, 3, 4):
        policies[f'Zaika fixed x{K}'] = FixedAgeSchedule(ZAIKA_LYG[K])
    for K in (1, 2, 3, 4):
        policies[f'PBVI adaptive x{K}'] = PBVIPolicy(solvers[K][1])

    print(f"\nEvaluating {len(policies)} policies on n={n} paired patients ...", flush=True)
    raw = {'No screening': ref}
    for name, pol in policies.items():
        raw[name] = eval_policy(pol, n, seed, cfg)
        print(f"  done: {name}", flush=True)
        partial = {nm: summarize(a, ref) for nm, a in raw.items()}
        json.dump(partial, open(os.path.join(RES, 'comparison.json'), 'w'), indent=2)

    results = {name: summarize(a, ref) for name, a in raw.items()}

    # ---- table ----
    cols = ['mean_colo', 'life_years', 'crc_incidence', 'crc_mortality',
            'LYG_per_1000', 'inc_reduction', 'mort_reduction', 'LYG_per_colo']
    with open(os.path.join(RES, 'comparison_table.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy'] + cols)
        for name, r in results.items():
            w.writerow([name] + [f"{r[c]:.3f}" for c in cols])
    with open(os.path.join(RES, 'comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 96)
    print(f"{'policy':<24}{'colo':>6}{'lifeYrs':>9}{'CRCinc%':>8}{'CRCmort%':>9}"
          f"{'LYG/1k':>8}{'incRed':>8}{'mortRed':>8}{'LYG/colo':>9}")
    print("-" * 96)
    for name, r in results.items():
        print(f"{name:<24}{r['mean_colo']:>6.2f}{r['life_years']:>9.3f}"
              f"{r['crc_incidence']:>8.2f}{r['crc_mortality']:>9.2f}"
              f"{r['LYG_per_1000']:>8.1f}{r['inc_reduction']:>8.2f}"
              f"{r['mort_reduction']:>8.2f}{r['LYG_per_colo']:>9.1f}")

    make_frontier_figure(results)
    make_adaptivity_figure(raw, solvers)
    print(f"\nSaved comparison_table.csv, comparison.json and figures.")
    return results, raw


def make_frontier_figure(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def pts(prefix, metric):
        return [(results[f'{prefix} x{K}']['mean_colo'],
                 results[f'{prefix} x{K}'][metric]) for K in (1, 2, 3, 4)]

    ax1.plot(*zip(*pts('Best fixed', 'LYG_per_1000')), 'o-', color='#E45756',
             label='Best fixed schedule (re-optimized)')
    ax1.plot(*zip(*pts('Zaika fixed', 'LYG_per_1000')), 'D--', color='#F58518',
             alpha=0.7, label='Zaika 2024 published ages')
    ax1.plot(*zip(*pts('PBVI adaptive', 'LYG_per_1000')), 's-', color='#4C78A8',
             label='PBVI adaptive (POMDP)')
    g = results['Guideline q10y 45-75']
    ax1.plot(g['mean_colo'], g['LYG_per_1000'], '^', color='green', ms=10, label='Guideline q10y')
    ax1.set_xlabel('mean screening colonoscopies / person')
    ax1.set_ylabel('life-years gained per 1000')
    ax1.set_title('Efficiency frontier: LYG vs colonoscopy use')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.plot(*zip(*pts('Best fixed', 'mort_reduction')), 'o-', color='#E45756',
             label='Best fixed')
    ax2.plot(*zip(*pts('PBVI adaptive', 'mort_reduction')), 's-', color='#4C78A8',
             label='PBVI adaptive')
    ax2.set_xlabel('mean screening colonoscopies / person')
    ax2.set_ylabel('CRC mortality reduction (abs. %)')
    ax2.set_title('CRC mortality reduction vs colonoscopy use')
    ax2.grid(alpha=0.3); ax2.legend()
    fig.suptitle('Adaptive POMDP screening vs fixed population schedules (true CMOST outcomes)')
    fig.savefig(os.path.join(PAPER_FIG, 'efficiency_frontier.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def make_adaptivity_figure(raw, solvers):
    """Show that PBVI allocates more colonoscopies to patients who turn out to
    develop adenomas/cancer (i.e., it personalises), unlike the fixed schedule."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    K = 3
    pbvi = raw[f'PBVI adaptive x{K}']
    zk = raw[f'Best fixed x{K}']
    # distribution of colonoscopies used
    bins = np.arange(0, K + 2) - 0.5
    ax1.hist(pbvi['n_colonoscopies'], bins=bins, alpha=0.7, color='#4C78A8',
             label='PBVI adaptive', density=True)
    ax1.hist(zk['n_colonoscopies'], bins=bins, alpha=0.5, color='#E45756',
             label='Best fixed', density=True)
    ax1.set_xlabel('screening colonoscopies used'); ax1.set_ylabel('fraction of patients')
    ax1.set_title(f'Colonoscopy allocation (budget {K})')
    ax1.legend(); ax1.grid(alpha=0.3)

    # colonoscopies used vs whether the patient developed CRC (true outcome)
    got_crc = pbvi['ever_clinical'] > 0.5
    ax2.bar([0, 1],
            [pbvi['n_colonoscopies'][~got_crc].mean(), pbvi['n_colonoscopies'][got_crc].mean()],
            color=['#4C78A8', '#E45756'], alpha=0.8)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['no CRC', 'developed CRC'])
    ax2.set_ylabel('mean colonoscopies (PBVI)')
    ax2.set_title('PBVI directs screening toward higher-risk individuals')
    ax2.grid(alpha=0.3)
    fig.savefig(os.path.join(PAPER_FIG, 'pbvi_adaptivity.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    main(n=n)
