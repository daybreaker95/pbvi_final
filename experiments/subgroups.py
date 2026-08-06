"""
subgroups.py
============

WHERE does the PBVI adaptive policy actually help?  (matched perfect adherence)

The pooled comparison shows PBVI ~ best-fixed on LYG.  The mechanistic reason is
that a colonoscopy is informative about the latent risk class only ~13% of the
time (when it finds an adenoma); the other ~87% look like the population prior, so
PBVI mimics the fixed schedule for them.  This script tests the flip side of that
argument -- PBVI's value should concentrate where personalization has something to
work with:

  PART 1 -- PATIENT SUBGROUPS (matched adherence, budgets K=2,3).
    Stratify the population by POLICY-INDEPENDENT traits, so the PBVI-vs-fixed
    contrast within a stratum is paired and fair:
      * latent adenoma-risk (CMOST IndividualRisk) -- quartiles, and the top-25%
        "high-risk class" the POMDP tries to infer;
      * natural-history phenotype -- whether, under NO screening, the patient ever
        develops clinical CRC ("has disease to find").
    Hypothesis: PBVI directs MORE colonoscopies to (and lowers mortality in) the
    high-risk / would-develop-CRC subgroup, and FEWER colonoscopies to the
    low-risk subgroup (same outcome, less screening) -- a reallocation that nets
    out to a small pooled difference but is real within strata.

  PART 2 -- SITUATION: SCREENING BUDGET / SURVEILLANCE REGIME (matched adherence).
    Budget K in {1,2,3,4,6}.  Adaptivity has more room when there are more
    decisions to personalize, so PBVI's colonoscopy-EFFICIENCY edge (CRC-mortality
    reduction per colonoscopy) should widen with budget, and PBVI should reach a
    given mortality with fewer colonoscopies than the fixed schedule.

Endpoints: CRC mortality (stable), CRC incidence, realized colonoscopies, LYG/1000,
and mortality-reduction per colonoscopy.  Paired common random numbers.

Run:  python experiments/subgroups.py [n]        (default n = 40000)
Outputs: results/subgroups.json, results/subgroups_strata.csv,
         results/subgroups_budget.csv, paper/figures/subgroups.png
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
from env.crc_env import (CRCScreeningEnv, EnvConfig, NoScreening, FixedAgeSchedule)
from pomdp.model import CRCScreeningPOMDP
from pomdp.pbvi import PBVI, PBVIPolicy
from experiments.run_comparison import find_best_fixed_greedy

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(PAPER_FIG, exist_ok=True)

BEST_FIXED = json.load(open(os.path.join(RES, 'best_fixed_schedules.json')))
BEST_FIXED = {int(k): v for k, v in BEST_FIXED.items()}


def load_pbvi(budget, n_belief=700):
    pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=budget,
                              gamma=1.0, screen_disutility=0.0, use_risk_classes=True)
    solver = PBVI(pomdp, n_belief=n_belief, seed=0)
    path = os.path.join(RES, 'policies', f'pbvi_k{budget}.npz')
    solver.solve(expansions=0, verbose=False)
    if os.path.exists(path):
        solver.load(path)
    else:
        solver.solve(expansions=4, verbose=False)
        solver.save(path)
    return solver


def eval_pop(policy, n, seed, cfg):
    """Paired roll-out; returns per-patient outcome arrays (incl individual_risk)."""
    params = build_params('CMOST13', n_patients=500, seed=777)
    eng = CRCEngine(params, rng=np.random.default_rng(0))
    env = CRCScreeningEnv(eng, cfg)
    rows = []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + i)
        rows.append(env.rollout(policy))
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def stratum_stats(arr, ref, mask):
    """CRC outcomes for `arr` on a subgroup `mask`, LYG vs `ref` on the same mask."""
    n = int(mask.sum())
    d = arr['crc_death'][mask]
    inc = arr['ever_clinical'][mask]
    colo = arr['n_colonoscopies'][mask]
    lost = arr['ly_lost_to_crc'][mask]
    ref_lost = ref['ly_lost_to_crc'][mask].mean()
    ref_mort = 100 * ref['crc_death'][mask].mean()
    lyg = 1000 * (ref_lost - lost.mean())
    mort = 100 * d.mean()
    return {
        'n': n,
        'mean_colo': float(colo.mean()),
        'crc_mortality': float(mort),
        'crc_mortality_se': float(100 * d.std(ddof=1) / np.sqrt(max(n, 1))),
        'crc_incidence': float(100 * inc.mean()),
        'mort_reduction': float(ref_mort - mort),
        'LYG_per_1000': float(lyg),
        'mortred_per_colo': float((ref_mort - mort) / max(colo.mean(), 1e-9)),
    }


def paired_mort_diff(arr_a, arr_b, mask):
    da = 100 * arr_a['crc_death'][mask]
    db = 100 * arr_b['crc_death'][mask]
    diff = da - db
    n = len(diff)
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(max(n, 1)))


# ---------------------------------------------------------------------------
def main(n=40000, seed=90210):
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)
    z = np.load(os.path.join(RES, 'transitions_stratified.npz'), allow_pickle=True)
    risk_thr = float(z['risk_threshold'])

    print("No screening (reference + natural-history labels) ...", flush=True)
    ref = eval_pop(NoScreening(), n, seed, cfg)
    ir = ref['individual_risk']
    # policy-independent strata
    q = np.quantile(ir, [0.25, 0.5, 0.75])
    risk_q = np.digitize(ir, q)                 # 0..3  (Q1..Q4)
    high_class = ir >= risk_thr                 # top-25% class the POMDP infers
    nh_crc = ref['ever_clinical'] > 0.5         # would develop CRC without screening
    print(f"  high-risk class fraction = {high_class.mean():.3f}; "
          f"would-develop-CRC = {nh_crc.mean():.3f}", flush=True)

    # =====================================================================
    # PART 1: subgroup stratification at matched adherence
    # =====================================================================
    strata = {}          # (K, stratum_label) -> {'fixed':..,'pbvi':..,'ref':..,'paired':..}
    raw = {}
    for K in (2, 3, 4):
        solver = load_pbvi(K)
        print(f"\n[Part 1] K={K}: evaluating best-fixed {BEST_FIXED[K]} and PBVI ...", flush=True)
        fixed = eval_pop(FixedAgeSchedule(BEST_FIXED[K]), n, seed, cfg)
        pbvi = eval_pop(PBVIPolicy(solver), n, seed, cfg)
        raw[K] = {'fixed': fixed, 'pbvi': pbvi}

        masks = {
            'all': np.ones(n, dtype=bool),
            'risk_Q1': risk_q == 0, 'risk_Q2': risk_q == 1,
            'risk_Q3': risk_q == 2, 'risk_Q4': risk_q == 3,
            'low_class': ~high_class, 'high_class': high_class,
            'NH_no_crc': ~nh_crc, 'NH_would_crc': nh_crc,
        }
        for lab, m in masks.items():
            md, se = paired_mort_diff(pbvi, fixed, m)
            strata[(K, lab)] = {
                'ref': stratum_stats(ref, ref, m),
                'fixed': stratum_stats(fixed, ref, m),
                'pbvi': stratum_stats(pbvi, ref, m),
                'paired_pbvi_minus_fixed_mort': {'d_pp': md, 'se': se},
            }
        s = strata[(K, 'high_class')]
        print(f"  high-risk class: fixed mort {s['fixed']['crc_mortality']:.2f}% "
              f"(colo {s['fixed']['mean_colo']:.2f})  vs  PBVI {s['pbvi']['crc_mortality']:.2f}% "
              f"(colo {s['pbvi']['mean_colo']:.2f})  diff "
              f"{s['paired_pbvi_minus_fixed_mort']['d_pp']:+.2f}"
              f"+/-{s['paired_pbvi_minus_fixed_mort']['se']:.2f} pp", flush=True)

    # =====================================================================
    # PART 2: budget / surveillance sweep
    # =====================================================================
    print("\n[Part 2] budget/surveillance sweep ...", flush=True)
    budgets = [1, 2, 3, 4, 6]
    n2 = min(n, 20000)
    budget_rows = {}
    for K in budgets:
        if K in BEST_FIXED:
            bf = BEST_FIXED[K]
        else:
            print(f"  greedy best-fixed for K={K} ...", flush=True)
            bf = find_best_fixed_greedy(K, ref, cfg)
            BEST_FIXED[K] = bf
        solver = load_pbvi(K)
        fixed = eval_pop(FixedAgeSchedule(bf), n2, seed, cfg)
        pbvi = eval_pop(PBVIPolicy(solver), n2, seed, cfg)
        refK = {k: v[:n2] for k, v in ref.items()}
        allm = np.ones(n2, dtype=bool)
        budget_rows[K] = {
            'schedule': bf,
            'fixed': stratum_stats(fixed, refK, allm),
            'pbvi': stratum_stats(pbvi, refK, allm),
        }
        f, p = budget_rows[K]['fixed'], budget_rows[K]['pbvi']
        print(f"  K={K}: fixed mort {f['crc_mortality']:.2f}% @ {f['mean_colo']:.2f} colo "
              f"(eff {f['mortred_per_colo']:.3f})  |  PBVI mort {p['crc_mortality']:.2f}% "
              f"@ {p['mean_colo']:.2f} colo (eff {p['mortred_per_colo']:.3f})", flush=True)

    # ---- outputs ----
    out = {
        'config': {'n': n, 'n_budget': n2, 'seed': seed, 'risk_threshold': risk_thr,
                   'best_fixed': BEST_FIXED},
        'strata': {f'K{K}|{lab}': v for (K, lab), v in strata.items()},
        'budget': {f'K{K}': v for K, v in budget_rows.items()},
    }
    with open(os.path.join(RES, 'subgroups.json'), 'w') as f:
        json.dump(out, f, indent=2)
    _write_csvs(strata, budget_rows)
    _make_figure(strata, budget_rows, raw, ref, risk_q)
    _print_report(strata, budget_rows)
    print("\nSaved results/subgroups.json, results/subgroups_strata.csv, "
          "results/subgroups_budget.csv, paper/figures/subgroups.png")
    return out


# ---------------------------------------------------------------------------
def _write_csvs(strata, budget_rows):
    cols = ['n', 'mean_colo', 'crc_mortality', 'crc_mortality_se', 'crc_incidence',
            'mort_reduction', 'LYG_per_1000', 'mortred_per_colo']
    with open(os.path.join(RES, 'subgroups_strata.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'stratum', 'policy'] + cols
                   + ['paired_pbvi_minus_fixed_mort_pp', 'se'])
        for (K, lab), v in strata.items():
            for pol in ('ref', 'fixed', 'pbvi'):
                extra = ['', '']
                if pol == 'pbvi':
                    extra = [f"{v['paired_pbvi_minus_fixed_mort']['d_pp']:.4f}",
                             f"{v['paired_pbvi_minus_fixed_mort']['se']:.4f}"]
                w.writerow([K, lab, pol] + [f"{v[pol][c]:.4f}" for c in cols] + extra)
    with open(os.path.join(RES, 'subgroups_budget.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'schedule', 'policy'] + cols)
        for K, v in budget_rows.items():
            for pol in ('fixed', 'pbvi'):
                w.writerow([K, ';'.join(map(str, v['schedule'])), pol]
                           + [f"{v[pol][c]:.4f}" for c in cols])


def _make_figure(strata, budget_rows, raw, ref, risk_q):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # panel 1: colonoscopy allocation by risk quartile (does PBVI target high-risk?)
    ax = axes[0]
    K = 4
    labs = ['risk_Q1', 'risk_Q2', 'risk_Q3', 'risk_Q4']
    fx = [strata[(K, l)]['fixed']['mean_colo'] for l in labs]
    pb = [strata[(K, l)]['pbvi']['mean_colo'] for l in labs]
    x = np.arange(4); w = 0.38
    ax.bar(x - w/2, fx, w, label='Best fixed', color='#E45756', alpha=0.85)
    ax.bar(x + w/2, pb, w, label='PBVI adaptive', color='#4C78A8', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(['Q1\n(low)', 'Q2', 'Q3', 'Q4\n(high)'])
    ax.set_xlabel('individual adenoma-risk quartile')
    ax.set_ylabel('mean screening colonoscopies')
    ax.set_title(f'PBVI does NOT target high-risk (K={K})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # panel 2: CRC mortality by risk class: no-screen / fixed / PBVI
    ax = axes[1]
    groups = [('low_class', 'Low-risk\n(bottom 75%)'), ('high_class', 'High-risk\n(top 25%)')]
    x = np.arange(len(groups)); w = 0.27
    for j, (pol, color, lab) in enumerate([('ref', '#BAB0AC', 'No screening'),
                                           ('fixed', '#E45756', 'Best fixed'),
                                           ('pbvi', '#4C78A8', 'PBVI adaptive')]):
        vals = [strata[(K, g)][pol]['crc_mortality'] for g, _ in groups]
        err = [strata[(K, g)][pol]['crc_mortality_se'] for g, _ in groups]
        ax.bar(x + (j - 1) * w, vals, w, yerr=err, capsize=3, color=color, alpha=0.85, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel('CRC mortality (%)')
    ax.set_title(f'PBVI ~ fixed within each risk class (K={K})')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # panel 3: efficiency vs budget (mortality reduction per colonoscopy)
    ax = axes[2]
    Ks = sorted(budget_rows)
    effx = [budget_rows[K]['fixed']['mortred_per_colo'] for K in Ks]
    effp = [budget_rows[K]['pbvi']['mortred_per_colo'] for K in Ks]
    ax.plot(Ks, effx, 'o-', color='#E45756', label='Best fixed')
    ax.plot(Ks, effp, 's-', color='#4C78A8', label='PBVI adaptive')
    ax.set_xlabel('screening budget K (lifetime colonoscopies)')
    ax.set_ylabel('CRC-mortality reduction per colonoscopy (pp)')
    ax.set_title('Efficiency vs budget (surveillance regime)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle('Where PBVI helps at matched adherence: not by risk-targeting, '
                 'but by colonoscopy efficiency at larger budgets (true CMOST)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PAPER_FIG, 'subgroups.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _print_report(strata, budget_rows):
    print("\n" + "=" * 96)
    print("PART 1 -- subgroup stratification (matched adherence)")
    print("=" * 96)
    for K in (2, 3):
        print(f"\n  Budget K = {K}   [mean colonoscopies | CRC mortality % | LYG/1000]")
        print(f"  {'stratum':<14}{'n':>7} | {'no-screen':>10} | {'best-fixed':>22} | "
              f"{'PBVI adaptive':>22} | {'PBVI-fixed mort':>16}")
        for lab in ['all', 'risk_Q1', 'risk_Q2', 'risk_Q3', 'risk_Q4',
                    'low_class', 'high_class', 'NH_no_crc', 'NH_would_crc']:
            v = strata[(K, lab)]
            r, fx, pb = v['ref'], v['fixed'], v['pbvi']
            d = v['paired_pbvi_minus_fixed_mort']
            sig = '*' if abs(d['d_pp']) > 2 * d['se'] else ' '
            print(f"  {lab:<14}{fx['n']:>7d} | {r['crc_mortality']:>9.2f}% | "
                  f"{fx['mean_colo']:>5.2f} {fx['crc_mortality']:>6.2f} {fx['LYG_per_1000']:>7.1f} | "
                  f"{pb['mean_colo']:>5.2f} {pb['crc_mortality']:>6.2f} {pb['LYG_per_1000']:>7.1f} | "
                  f"{d['d_pp']:>+7.2f}+/-{d['se']:.2f}{sig}")

    print("\n" + "=" * 96)
    print("PART 2 -- budget / surveillance sweep (matched adherence, whole population)")
    print("=" * 96)
    print(f"  {'K':>3} | {'best-fixed: colo  mort%  eff(pp/colo)':>40} | "
          f"{'PBVI: colo  mort%  eff(pp/colo)':>40}")
    for K in sorted(budget_rows):
        f, p = budget_rows[K]['fixed'], budget_rows[K]['pbvi']
        print(f"  {K:>3} | {f['mean_colo']:>10.2f}{f['crc_mortality']:>8.2f}"
              f"{f['mortred_per_colo']:>12.3f}     | "
              f"{p['mean_colo']:>8.2f}{p['crc_mortality']:>8.2f}{p['mortred_per_colo']:>12.3f}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    main(n=n)
