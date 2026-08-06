"""
prs_targeting.py
================

Does a strong baseline risk factor (a polygenic / genetic risk score, PRS) turn
ON mortality-targeting -- i.e. actually REDUCE CRC mortality in the truly
high-risk class, not merely save colonoscopies?

Background (`results_risk_factors.md`).  Baseline family-history + prior-adenoma
factors reach only a modest discrimination (AUC ~ 0.67).  At that AUC, and at the
mid cost c = 0.06, the payoff was efficiency (same mortality, fewer colonoscopies)
rather than a high-risk mortality drop.  Two things were limiting: (a) the risk
factors were too weakly discriminating, and (b) the per-colonoscopy cost was high
enough that even identified high-risk patients got only ~2.5 colonoscopies -- not
the intensive surveillance that cuts their mortality.

This experiment removes both limits:
  * a PRS-style baseline signal whose discrimination we push to AUC = 0.80-0.90
    (what a polygenic score + family history can plausibly reach), plus an ORACLE
    (AUC -> 1.0) upper bound;
  * a LOW per-colonoscopy cost (c = 0.02-0.03) that lets the policy give an
    identified high-risk patient 5-6 lifetime colonoscopies (true surveillance).

We evaluate in the true CMOST environment (matched adherence, paired CRN) and
report outcomes BY TRUE RISK CLASS: colonoscopy use and CRC mortality in the
high- and low-risk classes, total colonoscopies, and life-years gained.  The
fixed population schedules (x3, x4) give the reference high-risk-class mortality.

Hypothesis: as baseline AUC rises past ~0.8, PBVI concentrates enough colonoscopies
on the true high-risk class to push their CRC mortality BELOW the fixed schedule,
while total colonoscopy use falls (the low-risk majority is de-escalated).

Run:  python experiments/prs_targeting.py [n]     (default n = 25000)
Outputs: results/prs_targeting.json, results/prs_targeting.csv,
         paper/figures/prs_targeting.png
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

from env.crc_env import EnvConfig
from experiments.risk_factors import (solve_pbvi_cost, eval_frontier, summ,
                                       BEST_FIXED)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(PAPER_FIG, exist_ok=True)

# baseline discrimination -> Gaussian class-mean separation d = sqrt(2)*Phi^{-1}(AUC).
# 0.67 ~ family history + prior adenoma (results_risk_factors.md); 0.80/0.90 ~ PRS
# (+ family history); "oracle" (d large) ~ perfect risk knowledge (upper bound).
AUC_TO_D = {0.50: 0.0, 0.67: 0.622, 0.80: 1.190, 0.90: 1.812, 1.00: 8.0}
COSTS = [0.03, 0.02]                 # low cost -> intensive surveillance enabled


def by_class(arr, ref_lost, ref_lost_hi, ref_lost_lo):
    hi = arr['true_high']
    return {
        'total_colo': float(arr['n_colo'].mean()),
        'overall': summ(arr, ref_lost),
        'high': summ(arr, ref_lost_hi, hi),
        'low': summ(arr, ref_lost_lo, ~hi),
        'colo_high': float(arr['n_colo'][hi].mean()),
        'colo_low': float(arr['n_colo'][~hi].mean()),
    }


def main(n=25000, seed=90210):
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)
    z = np.load(os.path.join(RES, 'transitions_stratified.npz'), allow_pickle=True)
    thr = float(z['risk_threshold'])

    # reference (no screening) + per-class LYG denominators
    ref = eval_frontier('fixed', n, seed, cfg, thr, sched=[])
    hi = ref['true_high']
    ref_lost = ref['ly_lost'].mean()
    ref_lost_hi = ref['ly_lost'][hi].mean()
    ref_lost_lo = ref['ly_lost'][~hi].mean()
    ref_mort_hi = 100 * ref['crc_death'][hi].mean()
    ref_mort_lo = 100 * ref['crc_death'][~hi].mean()
    print(f"no-screening: overall {100*ref['crc_death'].mean():.2f}%  "
          f"high-class {ref_mort_hi:.2f}%  low-class {ref_mort_lo:.2f}%  "
          f"(high-class frac {hi.mean():.3f})", flush=True)

    # fixed population references (their high-class mortality is the benchmark)
    fixed = {}
    for K in (1, 2, 3, 4):
        arr = eval_frontier('fixed', n, seed, cfg, thr, sched=BEST_FIXED[K])
        fixed[K] = by_class(arr, ref_lost, ref_lost_hi, ref_lost_lo)
        print(f"  Fixed x{K}: total colo {fixed[K]['total_colo']:.2f}  "
              f"high-class mort {fixed[K]['high']['crc_mortality']:.2f}%  "
              f"low-class mort {fixed[K]['low']['crc_mortality']:.2f}%", flush=True)

    # PBVI cost-based + PRS-strength personalized prior
    grid = {}
    for c in COSTS:
        p, solver = solve_pbvi_cost(c)
        for auc, d in AUC_TO_D.items():
            arr = eval_frontier('pbvi_person', n, seed, cfg, thr, solver=solver, p=p,
                                signal='gauss', d=d)
            grid[(c, auc)] = by_class(arr, ref_lost, ref_lost_hi, ref_lost_lo)
            g = grid[(c, auc)]
            print(f"  c={c} AUC={auc:.2f}: total {g['total_colo']:.2f} | "
                  f"high colo {g['colo_high']:.2f} mort {g['high']['crc_mortality']:.2f}% | "
                  f"low colo {g['colo_low']:.2f} mort {g['low']['crc_mortality']:.2f}% | "
                  f"overall {g['overall']['crc_mortality']:.2f}%", flush=True)

    out = {
        'config': {'n': n, 'seed': seed, 'risk_threshold': thr, 'costs': COSTS,
                   'aucs': list(AUC_TO_D), 'ref_mort_high': ref_mort_hi,
                   'ref_mort_low': ref_mort_lo,
                   'ref_mort_overall': 100 * float(ref['crc_death'].mean())},
        'fixed': {f'x{K}': v for K, v in fixed.items()},
        'grid': {f'c{c}|AUC{auc:.2f}': v for (c, auc), v in grid.items()},
    }
    with open(os.path.join(RES, 'prs_targeting.json'), 'w') as f:
        json.dump(out, f, indent=2)
    _write_csv(grid, fixed)
    _make_figure(grid, fixed, out)
    _print_report(grid, fixed, out)
    print("\nSaved results/prs_targeting.json, results/prs_targeting.csv, "
          "paper/figures/prs_targeting.png")
    return out


def _write_csv(grid, fixed):
    with open(os.path.join(RES, 'prs_targeting.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy', 'cost', 'AUC', 'total_colo', 'colo_high', 'colo_low',
                    'mort_high', 'mort_high_se', 'mort_low', 'mort_overall',
                    'LYG1000_high'])
        for K, v in fixed.items():
            w.writerow([f'Fixed x{K}', '', '', f"{v['total_colo']:.3f}",
                        f"{v['colo_high']:.3f}", f"{v['colo_low']:.3f}",
                        f"{v['high']['crc_mortality']:.3f}",
                        f"{v['high']['crc_mortality_se']:.3f}",
                        f"{v['low']['crc_mortality']:.3f}",
                        f"{v['overall']['crc_mortality']:.3f}",
                        f"{v['high']['LYG_per_1000']:.1f}"])
        for (c, auc), v in grid.items():
            w.writerow([f'PBVI c={c}', c, f"{auc:.2f}", f"{v['total_colo']:.3f}",
                        f"{v['colo_high']:.3f}", f"{v['colo_low']:.3f}",
                        f"{v['high']['crc_mortality']:.3f}",
                        f"{v['high']['crc_mortality_se']:.3f}",
                        f"{v['low']['crc_mortality']:.3f}",
                        f"{v['overall']['crc_mortality']:.3f}",
                        f"{v['high']['LYG_per_1000']:.1f}"])


def _make_figure(grid, fixed, out):
    aucs = out['config']['aucs']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {0.03: '#4C78A8', 0.02: '#54A24B'}

    # panel 1: high-class CRC mortality vs AUC (targeting turns on)
    ax = axes[0]
    for c in out['config']['costs']:
        y = [grid[(c, a)]['high']['crc_mortality'] for a in aucs]
        e = [grid[(c, a)]['high']['crc_mortality_se'] for a in aucs]
        ax.errorbar(aucs, y, yerr=e, fmt='s-', color=colors[c], capsize=3,
                    label=f'PBVI cost c={c}')
    ax.axhline(fixed[3]['high']['crc_mortality'], ls='--', color='#E45756',
               label=f"Fixed x3 ({fixed[3]['high']['crc_mortality']:.2f}%)")
    ax.axhline(fixed[4]['high']['crc_mortality'], ls=':', color='#B279A2',
               label=f"Fixed x4 ({fixed[4]['high']['crc_mortality']:.2f}%)")
    ax.axhline(out['config']['ref_mort_high'], ls='-', color='gray', alpha=0.5,
               label=f"No screening ({out['config']['ref_mort_high']:.2f}%)")
    ax.axvline(0.67, ls=':', color='green', alpha=0.6)
    ax.text(0.67, ax.get_ylim()[1], ' FH+PA', color='green', va='top', fontsize=8)
    ax.set_xlabel('baseline risk-factor discrimination (AUC)')
    ax.set_ylabel('CRC mortality, TRUE high-risk class (%)')
    ax.set_title('Mortality-targeting turns on at AUC $\\geq$ 0.8')
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    # panel 2: colonoscopy allocation by true class vs AUC (c=0.03)
    ax = axes[1]
    c0 = 0.03
    hi = [grid[(c0, a)]['colo_high'] for a in aucs]
    lo = [grid[(c0, a)]['colo_low'] for a in aucs]
    ax.plot(aucs, hi, 's-', color='#E45756', label='true high-risk class')
    ax.plot(aucs, lo, 'o-', color='#4C78A8', label='true low-risk class')
    ax.set_xlabel('baseline risk-factor discrimination (AUC)')
    ax.set_ylabel('mean colonoscopies / person')
    ax.set_title(f'Surveillance concentrates on high-risk (c={c0})')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # panel 3: high-class mortality vs TOTAL colonoscopy use (frontier)
    ax = axes[2]
    fx = sorted((fixed[K]['total_colo'], fixed[K]['high']['crc_mortality']) for K in fixed)
    ax.plot([x for x, _ in fx], [y for _, y in fx], 'o-', color='#E45756',
            label='Fixed population')
    for c in out['config']['costs']:
        pts = sorted((grid[(c, a)]['total_colo'], grid[(c, a)]['high']['crc_mortality'])
                     for a in aucs)
        ax.plot([x for x, _ in pts], [y for _, y in pts], 's-', color=colors[c],
                label=f'PBVI+PRS c={c}')
    ax.set_xlabel('TOTAL mean colonoscopies / person')
    ax.set_ylabel('CRC mortality, high-risk class (%)')
    ax.set_title('High-risk mortality at matched total use')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle('A strong PRS turns on mortality-targeting: PBVI cuts high-risk-class '
                 'CRC mortality below fixed schedules (true CMOST)', fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PAPER_FIG, 'prs_targeting.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _print_report(grid, fixed, out):
    print("\n" + "=" * 90)
    print("PRS mortality-targeting  [outcomes by TRUE risk class]")
    print("=" * 90)
    print(f"  no-screening high-class mortality = {out['config']['ref_mort_high']:.2f}%")
    print(f"  Fixed x3: high-class {fixed[3]['high']['crc_mortality']:.2f}% "
          f"@ total {fixed[3]['total_colo']:.2f} ;  "
          f"Fixed x4: high-class {fixed[4]['high']['crc_mortality']:.2f}% "
          f"@ total {fixed[4]['total_colo']:.2f}")
    for c in out['config']['costs']:
        print(f"\n  cost c = {c}")
        print(f"  {'AUC':>5}{'total':>8}{'colo_hi':>9}{'colo_lo':>9}"
              f"{'mort_hi':>9}{'mort_lo':>9}{'mort_all':>9}{'LYG_hi':>8}")
        for a in out['config']['aucs']:
            v = grid[(c, a)]
            print(f"  {a:>5.2f}{v['total_colo']:>8.2f}{v['colo_high']:>9.2f}"
                  f"{v['colo_low']:>9.2f}{v['high']['crc_mortality']:>9.2f}"
                  f"{v['low']['crc_mortality']:>9.2f}{v['overall']['crc_mortality']:>9.2f}"
                  f"{v['high']['LYG_per_1000']:>8.0f}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    main(n=n)
