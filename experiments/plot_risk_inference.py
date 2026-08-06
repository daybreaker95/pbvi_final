"""
plot_risk_inference.py -- figure of the personalization MECHANISM: how a
colonoscopy finding updates the inferred risk class and the recommended next
interval. This is the clean, unconfounded demonstration of history-dependence.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model import (CRCScreeningPOMDP, WAIT, SCREEN,
                         O_NORMAL, O_ADENOMA, O_ADVAD, O_NOTEST, NORMAL, EA, AA)
from pomdp.pbvi import PBVI

PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))


def main():
    p = CRCScreeningPOMDP(age_min=40, age_max=85, budget=3, gamma=1.0,
                          screen_disutility=0.008, use_risk_classes=True)
    solver = PBVI(p, n_belief=350, seed=0); solver.solve(expansions=1, verbose=False)

    ages = list(range(45, 76, 5))
    findings = [('clean exam', O_NORMAL, '#4C78A8'),
                ('adenoma', O_ADENOMA, '#F58518'),
                ('advanced adenoma', O_ADVAD, '#E45756')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, obs, col in findings:
        phigh = []
        for age in ages:
            b = p.initial_belief()
            for a in range(p.age_min, age):
                bn = p.belief_update(b, a, WAIT, O_NOTEST); b = bn if bn is not None else b
            bs = p.belief_update(b, age, SCREEN, obs)
            phigh.append(p.risk_marginal(bs)[-1] if bs is not None else np.nan)
        ax1.plot(ages, phigh, 'o-', color=col, label=label)
    ax1.axhline(p.frac_high, ls=':', color='gray', label=f'prior ({p.frac_high:.2f})')
    ax1.set_xlabel('age at colonoscopy'); ax1.set_ylabel('inferred P(high risk) after exam')
    ax1.set_title('Findings update the inferred risk class')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    # belief re-accumulation of adenoma risk after each finding at age 55:
    # a finding raises inferred risk, so the belief re-develops adenoma faster
    age0 = 55
    b0 = p.initial_belief()
    for a in range(p.age_min, age0):
        bn = p.belief_update(b0, a, WAIT, O_NOTEST); b0 = bn if bn is not None else b0
    horizon = list(range(age0, 81))
    for label, obs, col in findings:
        b = p.belief_update(b0, age0, SCREEN, obs)
        traj = []
        for a in horizon:
            cm = p.clinical_marginal(b)
            traj.append(100 * (cm[EA] + cm[AA]))   # P(adenoma present) in belief
            bn = p.belief_update(b, a, WAIT, O_NOTEST); b = bn if bn is not None else b
        ax2.plot(horizon, traj, 'o-', ms=3, color=col, label=label)
    ax2.set_xlabel('age'); ax2.set_ylabel('belief P(adenoma present) %')
    ax2.set_title(f'Post-finding adenoma-belief re-accumulation (screen @ {age0})')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle('POMDP personalization mechanism (belief-based, history-dependent)')
    fig.savefig(os.path.join(PAPER_FIG, 'pbvi_risk_inference.png'), dpi=150, bbox_inches='tight')
    print('saved pbvi_risk_inference.png')


if __name__ == '__main__':
    main()
