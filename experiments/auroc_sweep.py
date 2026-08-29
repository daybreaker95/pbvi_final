"""
auroc_sweep.py
==============

How much risk-classifier discrimination does risk-stratified colonoscopy
screening actually need, and which combination of risk factors / emerging tests
would supply it?

`prs_targeting.py` answered a coarse version of the first question with four
points (AUC 0.50 / 0.67 / 0.80 / 0.90) plus an ORACLE at AUC -> 1.0, and found
that high-risk-class mortality-targeting "turns on" somewhere around 0.8.  Two
things were left open:

  1. the grid was too coarse to locate the turn-on, and the oracle arm is not a
     reachable operating point -- no assay will ever perfectly observe a lifelong
     latent adenoma-risk class;
  2. an AUC is not a screening programme.  Nothing said which *combination* of
     risk factors and tests puts a programme at a given AUROC, nor what the
     marginal value of adding one more modality is.

This experiment sweeps discrimination on a fine grid up to a deliberate CEILING
of AUROC = 0.85 -- an ambitious but defensible target for a combined
questionnaire + PRS + biomarker panel -- and runs it in two tracks:

  TRACK A (continuous).  An abstract calibrated risk score at each AUROC in
  0.50 ... 0.85.  This is the response curve: what does the POMDP do with
  exactly this much information, at each colonoscopy budget?

  TRACK B (panels).  Nine named risk-factor / risk-test COMBINATIONS from
  `risk_panels.py`, from family history alone (AUROC ~ 0.60) through today's
  clinical baseline (FH + prior adenoma, ~0.67), a lifestyle/environmental
  E-score, current and next-generation PRS, and emerging assays used as risk
  stratifiers (faecal microbiome, quantitative f-Hb, multi-target stool DNA,
  blood methylated cfDNA), ending at ~0.85.  Each panel's AUROC is MEASURED from
  the simulated patients, not assumed, and each rung adds exactly one modality,
  so the ladder reads as the marginal value of the next test.

Both tracks are run at four per-colonoscopy costs (c = 0.02, 0.03, 0.06, 0.10),
i.e. four budget regimes, because `results_risk_factors.md` showed the payoff of
discrimination is budget-dependent: a tight budget converts information into
saved colonoscopies, a loose one into high-risk-class mortality.

Every arm is evaluated in the true CMOST environment with matched adherence and
paired common random numbers (the same patients, the same marker draws), and is
reported BY TRUE RISK CLASS.  Fixed population schedules x1..x4 and no-screening
give the references.

Run:   python experiments/auroc_sweep.py [n] [--workers W]
       (default n = 50000, workers = 5)
Outputs: results/auroc_sweep.json,
         results/auroc_sweep_gauss.csv, results/auroc_sweep_panels.csv,
         paper/figures/auroc_sweep.png
"""

from __future__ import annotations

import os
import sys
import json
import csv
import time
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.crc_env import EnvConfig
from experiments.risk_factors import (solve_pbvi_cost, eval_frontier, summ,
                                      empirical_auc, BEST_FIXED)
from experiments import risk_panels as RP

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(PAPER_FIG, exist_ok=True)

# Discrimination grid.  Dense from 0.70 up, because prs_targeting.py located the
# mortality-targeting turn-on somewhere in 0.67-0.80 and the point of this sweep
# is to resolve it.  0.85 is the ceiling: no oracle arm.
AUC_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85]
COSTS = [0.02, 0.03, 0.06, 0.10]
C_MAIN = 0.03                  # surveillance-enabling cost used in the figures

_SOLVERS = {}                  # per-worker-process cache: cost -> (pomdp, solver)


def _solver_for(c):
    if c not in _SOLVERS:
        _SOLVERS[c] = solve_pbvi_cost(c)
    return _SOLVERS[c]


def by_class(arr, ref_lost, ref_lost_hi, ref_lost_lo):
    """Outcome bundle for one arm, split by TRUE risk class."""
    hi = arr['true_high']
    return {
        'total_colo': float(arr['n_colo'].mean()),
        'colo_high': float(arr['n_colo'][hi].mean()),
        'colo_low': float(arr['n_colo'][~hi].mean()),
        'overall': summ(arr, ref_lost),
        'high': summ(arr, ref_lost_hi, hi),
        'low': summ(arr, ref_lost_lo, ~hi),
        'auc_realized': float(empirical_auc(arr['p_high'], hi)),
        'frac_flagged': float(np.mean(arr['p_high'] > RP.PRIOR)),
    }


# ---------------------------------------------------------------------------
# one arm = one (policy, signal) configuration; runs in a worker process
# ---------------------------------------------------------------------------
def run_arm(task):
    n, seed, thr, spec = task['n'], task['seed'], task['thr'], task['spec']
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)
    t0 = time.time()

    if spec['kind'] == 'fixed':
        arr = eval_frontier('fixed', n, seed, cfg, thr, sched=spec['sched'])
    else:
        p, solver = _solver_for(spec['cost'])
        post = (RP.gauss_fn(spec['auc']) if spec['kind'] == 'gauss'
                else RP.make_panel_fn(spec['panel']))
        arr = eval_frontier('pbvi_person', n, seed, cfg, thr, solver=solver, p=p,
                            post_fn=post)

    return task['key'], {
        'n_colo': arr['n_colo'], 'crc_death': arr['crc_death'],
        'ever_clinical': arr['ever_clinical'], 'ly_lost': arr['ly_lost'],
        'true_high': arr['true_high'], 'p_high': arr['p_high'],
        'secs': time.time() - t0,
    }


def _tasks(n, seed, thr):
    """Reference arms first (the no-screen arm defines the LYG denominators)."""
    out = [{'n': n, 'seed': seed, 'thr': thr, 'key': 'ref|no_screen',
            'spec': {'kind': 'fixed', 'sched': []}}]
    for K in (1, 2, 3, 4):
        out.append({'n': n, 'seed': seed, 'thr': thr, 'key': f'fixed|x{K}',
                    'spec': {'kind': 'fixed', 'sched': BEST_FIXED[K]}})
    for c in COSTS:
        for a in AUC_GRID:
            out.append({'n': n, 'seed': seed, 'thr': thr,
                        'key': f'gauss|{c}|{a:.3f}',
                        'spec': {'kind': 'gauss', 'cost': c, 'auc': a}})
    for c in COSTS:
        for name in RP.PANEL_NAMES:
            out.append({'n': n, 'seed': seed, 'thr': thr,
                        'key': f'panel|{c}|{name}',
                        'spec': {'kind': 'panel', 'cost': c, 'panel': name}})
    return out


# ---------------------------------------------------------------------------
def main(n=50000, seed=90210, workers=5):
    z = np.load(os.path.join(RES, 'transitions_stratified.npz'), allow_pickle=True)
    thr = float(z['risk_threshold'])

    # Solve (and cache to disk) every cost-based policy SERIALLY first: workers
    # would otherwise race to write the same results/policies/*.npz.
    for c in COSTS:
        t = time.time()
        solve_pbvi_cost(c)
        print(f"solved cost-based PBVI c={c} ({time.time()-t:.1f}s)", flush=True)

    tasks = _tasks(n, seed, thr)
    print(f"AUROC sweep: {len(tasks)} arms x n={n} patients, {workers} workers", flush=True)
    print(f"  Track A: {len(AUC_GRID)} AUROC levels {AUC_GRID[0]}-{AUC_GRID[-1]} "
          f"x {len(COSTS)} costs", flush=True)
    print(f"  Track B: {len(RP.PANEL_NAMES)} risk-factor panels x {len(COSTS)} costs",
          flush=True)

    raw = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (key, res) in enumerate(ex.map(run_arm, tasks), 1):
            raw[key] = res
            print(f"  [{i:>3}/{len(tasks)}] {key:<28} "
                  f"colo={res['n_colo'].mean():5.2f} "
                  f"mort={100*res['crc_death'].mean():5.2f}%  "
                  f"({res['secs']:.0f}s, elapsed {time.time()-t0:.0f}s)", flush=True)

    # LYG denominators from the no-screening arm
    ref = raw['ref|no_screen']
    hi = ref['true_high']
    ref_lost = ref['ly_lost'].mean()
    ref_lost_hi = ref['ly_lost'][hi].mean()
    ref_lost_lo = ref['ly_lost'][~hi].mean()
    ref_mort = {'overall': 100 * float(ref['crc_death'].mean()),
                'high': 100 * float(ref['crc_death'][hi].mean()),
                'low': 100 * float(ref['crc_death'][~hi].mean()),
                'frac_high': float(hi.mean())}
    print(f"\nno screening: overall {ref_mort['overall']:.2f}%  "
          f"high-class {ref_mort['high']:.2f}%  low-class {ref_mort['low']:.2f}%  "
          f"(high-class frac {ref_mort['frac_high']:.3f})", flush=True)

    def pack(key):
        return by_class(raw[key], ref_lost, ref_lost_hi, ref_lost_lo)

    fixed = {K: pack(f'fixed|x{K}') for K in (1, 2, 3, 4)}
    gauss = {(c, a): pack(f'gauss|{c}|{a:.3f}') for c in COSTS for a in AUC_GRID}
    panels = {(c, nm): pack(f'panel|{c}|{nm}') for c in COSTS for nm in RP.PANEL_NAMES}

    out = {
        'config': {
            'n': n, 'seed': seed, 'risk_threshold': thr, 'costs': COSTS,
            'auc_grid': AUC_GRID, 'auc_ceiling': AUC_GRID[-1], 'c_main': C_MAIN,
            'panels': [{'name': nm, 'desc': RP.PANEL_DESC[nm],
                        'markers': RP.PANEL_BY_NAME[nm],
                        'auc_realized': panels[(C_MAIN, nm)]['auc_realized']}
                       for nm in RP.PANEL_NAMES],
            'markers': {k: {'spec': list(v), 'label': RP.MARKER_LABEL[k]}
                        for k, v in RP.MARKERS.items()},
            'ref_mort': ref_mort,
        },
        'fixed': {f'x{K}': v for K, v in fixed.items()},
        'gauss': {f'c{c}|AUC{a:.3f}': v for (c, a), v in gauss.items()},
        'panels': {f'c{c}|{nm}': v for (c, nm), v in panels.items()},
    }
    with open(os.path.join(RES, 'auroc_sweep.json'), 'w') as f:
        json.dump(out, f, indent=2)

    _write_csvs(gauss, panels, fixed)
    _make_figure(gauss, panels, fixed, out)
    _print_report(gauss, panels, fixed, out)
    print("\nSaved results/auroc_sweep.json, results/auroc_sweep_gauss.csv, "
          "results/auroc_sweep_panels.csv, paper/figures/auroc_sweep.png")
    return out


# ---------------------------------------------------------------------------
_COLS = ['total_colo', 'colo_high', 'colo_low', 'mort_high', 'mort_high_se',
         'mort_low', 'mort_overall', 'mort_overall_se', 'LYG1000_overall',
         'LYG1000_high', 'incidence_overall']


def _row(v):
    return [f"{v['total_colo']:.3f}", f"{v['colo_high']:.3f}", f"{v['colo_low']:.3f}",
            f"{v['high']['crc_mortality']:.3f}", f"{v['high']['crc_mortality_se']:.3f}",
            f"{v['low']['crc_mortality']:.3f}",
            f"{v['overall']['crc_mortality']:.3f}",
            f"{v['overall']['crc_mortality_se']:.3f}",
            f"{v['overall']['LYG_per_1000']:.1f}", f"{v['high']['LYG_per_1000']:.1f}",
            f"{v['overall']['crc_incidence']:.3f}"]


def _write_csvs(gauss, panels, fixed):
    with open(os.path.join(RES, 'auroc_sweep_gauss.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'cost', 'AUROC_target', 'AUROC_realized'] + _COLS)
        for K, v in fixed.items():
            w.writerow([f'Fixed x{K}', '', '', ''] + _row(v))
        for (c, a), v in sorted(gauss.items()):
            w.writerow([f'PBVI c={c}', c, f'{a:.3f}',
                        f"{v['auc_realized']:.4f}"] + _row(v))
    with open(os.path.join(RES, 'auroc_sweep_panels.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['panel', 'description', 'n_markers', 'markers', 'cost',
                    'AUROC_realized'] + _COLS)
        for (c, nm), v in sorted(panels.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            mk = RP.PANEL_BY_NAME[nm]
            w.writerow([nm, RP.PANEL_DESC[nm], len(mk), '+'.join(mk), c,
                        f"{v['auc_realized']:.4f}"] + _row(v))


# ---------------------------------------------------------------------------
def _make_figure(gauss, panels, fixed, out):
    aucs = out['config']['auc_grid']
    colors = {0.02: '#54A24B', 0.03: '#4C78A8', 0.06: '#F58518', 0.10: '#9D7660'}
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.6))

    # (a) high-risk-class CRC mortality vs AUROC, per budget
    ax = axes[0, 0]
    for c in COSTS:
        y = [gauss[(c, a)]['high']['crc_mortality'] for a in aucs]
        e = [gauss[(c, a)]['high']['crc_mortality_se'] for a in aucs]
        ax.errorbar(aucs, y, yerr=e, fmt='o-', ms=4, capsize=2, color=colors[c],
                    label=f'PBVI c={c}')
    for K, ls, col in ((3, '--', '#E45756'), (4, ':', '#B279A2')):
        ax.axhline(fixed[K]['high']['crc_mortality'], ls=ls, color=col, lw=1.2,
                   label=f"Fixed x{K} ({fixed[K]['high']['crc_mortality']:.2f}%)")
    ax.axvline(0.67, ls=':', color='green', alpha=0.7)
    ax.annotate('FH+PA\n(today)', (0.67, ax.get_ylim()[1]), color='green',
                fontsize=7.5, ha='center', va='top')
    ax.set_xlabel('risk-classifier discrimination (AUROC)')
    ax.set_ylabel('CRC mortality, TRUE high-risk class (%)')
    ax.set_title('(a) High-risk-class mortality vs discrimination')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (b) colonoscopy reallocation by true class vs AUROC
    ax = axes[0, 1]
    for c in (C_MAIN, 0.06):
        ls = '-' if c == C_MAIN else '--'
        ax.plot(aucs, [gauss[(c, a)]['colo_high'] for a in aucs], 's' + ls,
                ms=4, color='#E45756', label=f'true high class (c={c})')
        ax.plot(aucs, [gauss[(c, a)]['colo_low'] for a in aucs], 'o' + ls,
                ms=4, color='#4C78A8', label=f'true low class (c={c})')
    ax.set_xlabel('risk-classifier discrimination (AUROC)')
    ax.set_ylabel('mean colonoscopies / person')
    ax.set_title('(b) Colonoscopy reallocation by true class')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (c) total colonoscopy use vs AUROC (information buys volume back)
    ax = axes[0, 2]
    for c in COSTS:
        ax.plot(aucs, [gauss[(c, a)]['total_colo'] for a in aucs], 'o-', ms=4,
                color=colors[c], label=f'c={c}')
    ax.set_xlabel('risk-classifier discrimination (AUROC)')
    ax.set_ylabel('TOTAL mean colonoscopies / person')
    ax.set_title('(c) Total colonoscopy use falls as AUROC rises')
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    # (d) efficiency frontier: overall mortality vs total colonoscopies
    ax = axes[1, 0]
    fx = sorted((fixed[K]['total_colo'], fixed[K]['overall']['crc_mortality'])
                for K in fixed)
    ax.plot([x for x, _ in fx], [y for _, y in fx], 'o-', color='#E45756',
            label='Fixed population x1-x4')
    for c in COSTS:
        pts = [(gauss[(c, a)]['total_colo'], gauss[(c, a)]['overall']['crc_mortality'])
               for a in aucs]
        ax.plot([x for x, _ in pts], [y for _, y in pts], '.-', ms=5,
                color=colors[c], label=f'PBVI c={c}, AUROC 0.50->0.85')
        ax.scatter([pts[-1][0]], [pts[-1][1]], marker='*', s=90, color=colors[c],
                   zorder=5)
    ax.set_xlabel('TOTAL mean colonoscopies / person')
    ax.set_ylabel('overall CRC mortality (%)')
    ax.set_title('(d) Efficiency frontier (star = AUROC 0.85)')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # (e) panel ladder: realized AUROC per rung
    ax = axes[1, 1]
    names = out['config']['panels']
    y = np.arange(len(names))
    vals = [p['auc_realized'] for p in names]
    ax.barh(y, vals, color='#4C78A8', height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([p['name'] for p in names], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 0.90)
    ax.axvline(0.85, ls='--', color='#E45756', lw=1.2, label='sweep ceiling 0.85')
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f'{v:.3f}', va='center', fontsize=7)
    ax.set_xlabel('measured AUROC for the latent high-risk class')
    ax.set_title('(e) Risk-factor panels: what each combination reaches')
    ax.grid(alpha=0.3, axis='x'); ax.legend(fontsize=7)

    # (f) panel ladder outcomes at the main cost
    ax = axes[1, 2]
    pm = [panels[(C_MAIN, p['name'])] for p in names]
    ax.plot(vals, [v['high']['crc_mortality'] for v in pm], 's-', color='#E45756',
            label='high-risk class mortality')
    ax.plot(vals, [v['overall']['crc_mortality'] for v in pm], 'o-', color='#4C78A8',
            label='overall mortality')
    ga = [gauss[(C_MAIN, a)]['high']['crc_mortality'] for a in aucs]
    ax.plot(aucs, ga, '-', color='#E45756', alpha=0.35, lw=3,
            label='Track A (abstract score), high class')
    for p, v in zip(names, pm):
        ax.annotate(p['name'].split()[0], (p['auc_realized'],
                                           v['high']['crc_mortality']),
                    fontsize=6.5, xytext=(0, 6), textcoords='offset points',
                    ha='center')
    ax.set_xlabel('measured panel AUROC')
    ax.set_ylabel('CRC mortality (%)')
    ax.set_title(f'(f) Panels land on the Track-A curve (c={C_MAIN})')
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle('How much risk discrimination does risk-stratified screening need? '
                 f"AUROC swept to {out['config']['auc_ceiling']:.2f} across four "
                 'colonoscopy budgets, with named risk-factor / risk-test panels '
                 f"(true CMOST, n={out['config']['n']:,})", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PAPER_FIG, 'auroc_sweep.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
def _print_report(gauss, panels, fixed, out):
    cfg = out['config']
    print("\n" + "=" * 100)
    print("TRACK A -- abstract risk score, AUROC 0.50 -> %.2f  [by TRUE risk class]"
          % cfg['auc_ceiling'])
    print("=" * 100)
    print(f"  no screening: high-class {cfg['ref_mort']['high']:.2f}%  "
          f"overall {cfg['ref_mort']['overall']:.2f}%")
    for K in (1, 2, 3, 4):
        v = fixed[K]
        print(f"  Fixed x{K}: total {v['total_colo']:.2f}  high-class "
              f"{v['high']['crc_mortality']:.2f}%  overall "
              f"{v['overall']['crc_mortality']:.2f}%")
    for c in COSTS:
        print(f"\n  cost c = {c}")
        print(f"  {'AUROC':>6}{'total':>8}{'colo_hi':>9}{'colo_lo':>9}"
              f"{'mort_hi':>9}{'+-':>6}{'mort_lo':>9}{'mort_all':>9}{'LYG/1k':>8}")
        for a in cfg['auc_grid']:
            v = gauss[(c, a)]
            print(f"  {a:>6.3f}{v['total_colo']:>8.2f}{v['colo_high']:>9.2f}"
                  f"{v['colo_low']:>9.2f}{v['high']['crc_mortality']:>9.2f}"
                  f"{v['high']['crc_mortality_se']:>6.2f}"
                  f"{v['low']['crc_mortality']:>9.2f}"
                  f"{v['overall']['crc_mortality']:>9.2f}"
                  f"{v['overall']['LYG_per_1000']:>8.0f}")

    print("\n" + "=" * 100)
    print(f"TRACK B -- named risk-factor / risk-test panels (c = {C_MAIN})")
    print("=" * 100)
    print(f"  {'panel':<17}{'AUROC':>7}{'total':>8}{'colo_hi':>9}{'colo_lo':>9}"
          f"{'mort_hi':>9}{'+-':>6}{'mort_all':>9}   description")
    for p in cfg['panels']:
        v = panels[(C_MAIN, p['name'])]
        print(f"  {p['name']:<17}{p['auc_realized']:>7.3f}{v['total_colo']:>8.2f}"
              f"{v['colo_high']:>9.2f}{v['colo_low']:>9.2f}"
              f"{v['high']['crc_mortality']:>9.2f}"
              f"{v['high']['crc_mortality_se']:>6.2f}"
              f"{v['overall']['crc_mortality']:>9.2f}   {p['desc']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('n', nargs='?', type=int, default=50000)
    ap.add_argument('--workers', type=int, default=5,
                    help='process pool size (keep low: other CMOST jobs may be running)')
    ap.add_argument('--seed', type=int, default=90210)
    a = ap.parse_args()
    main(n=a.n, seed=a.seed, workers=a.workers)
