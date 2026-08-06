"""
frontier_sweep.py -- trace a smooth PBVI efficiency frontier by sweeping the
per-colonoscopy penalty (soft budget), and overlay Zaika's fixed-schedule
points, all evaluated in the true CMOST environment.

Higher penalty -> the adaptive policy screens fewer, higher-yield individuals,
so each (mean colonoscopies, LYG) pair is a point on the personalized frontier.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.crc_env import EnvConfig, NoScreening, FixedAgeSchedule
from pomdp.model import CRCScreeningPOMDP
from pomdp.pbvi import PBVI, PBVIPolicy
from experiments.run_comparison import eval_policy, summarize, ZAIKA_LYG

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))


def main(n=15000, seed=90210):
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)
    ref = eval_policy(NoScreening(), n, seed, cfg)

    # PBVI frontier: sweep the per-colonoscopy disutility (soft budget)
    lambdas = [0.6, 0.4, 0.28, 0.2, 0.14, 0.09, 0.05, 0.02]
    pbvi_pts = []
    for lam in lambdas:
        pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=8, gamma=1.0,
                                  screen_disutility=lam, use_risk_classes=True)
        solver = PBVI(pomdp, n_belief=350, seed=0)
        solver.solve(expansions=2, verbose=False)
        a = eval_policy(PBVIPolicy(solver), n, seed, cfg)
        s = summarize(a, ref)
        pbvi_pts.append((lam, s['mean_colo'], s['LYG_per_1000'], s['mort_reduction']))
        print(f"  lambda={lam:.2f}: colo={s['mean_colo']:.2f} "
              f"LYG/1k={s['LYG_per_1000']:.1f} mortRed={s['mort_reduction']:.2f}")

    # Zaika points
    zk_pts = []
    for K in (1, 2, 3, 4):
        a = eval_policy(FixedAgeSchedule(ZAIKA_LYG[K]), n, seed, cfg)
        s = summarize(a, ref)
        zk_pts.append((K, s['mean_colo'], s['LYG_per_1000'], s['mort_reduction']))

    json.dump({'pbvi': pbvi_pts, 'zaika': zk_pts},
              open(os.path.join(RES, 'frontier.json'), 'w'), indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    pc = [p[1] for p in pbvi_pts]; py = [p[2] for p in pbvi_pts]
    zc = [p[1] for p in zk_pts]; zy = [p[2] for p in zk_pts]
    ax1.plot(pc, py, 's-', color='#4C78A8', label='PBVI adaptive (soft budget)')
    ax1.plot(zc, zy, 'o-', color='#E45756', label='Zaika fixed schedules')
    ax1.set_xlabel('mean screening colonoscopies / person')
    ax1.set_ylabel('life-years gained per 1000')
    ax1.set_title('Personalized vs fixed screening: LYG frontier')
    ax1.grid(alpha=0.3); ax1.legend()

    pm = [p[3] for p in pbvi_pts]; zm = [p[3] for p in zk_pts]
    ax2.plot(pc, pm, 's-', color='#4C78A8', label='PBVI adaptive')
    ax2.plot(zc, zm, 'o-', color='#E45756', label='Zaika fixed')
    ax2.set_xlabel('mean screening colonoscopies / person')
    ax2.set_ylabel('CRC mortality reduction (abs. %)')
    ax2.set_title('CRC mortality reduction frontier')
    ax2.grid(alpha=0.3); ax2.legend()
    fig.suptitle('Efficiency frontier in the true CMOST environment')
    fig.savefig(os.path.join(PAPER_FIG, 'frontier_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved frontier.json and frontier_sweep.png")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    main(n=n)
