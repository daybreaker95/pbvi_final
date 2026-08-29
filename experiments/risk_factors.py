"""
risk_factors.py
===============

Can baseline risk factors (family history, prior-adenoma history) let the POMDP
IDENTIFY and TARGET high-risk patients -- the thing it could not do from clean
colonoscopies alone?

Two design changes vs the base model, both required for genuine risk-targeting:

 (1) PERSONALIZED PRIOR.  Age is already fully observed (the POMDP conditions on
     it directly).  We add two baseline observations correlated with the latent
     adenoma-risk class: family history (FH) and prior-adenoma history (PA).  Each
     patient's initial belief is set to the Bayes posterior P(high | FH, PA)
     instead of the flat population prior 0.25.  The discrimination of this signal
     is summarized by its AUC; we sweep AUC from 0.50 (uninformative = the base
     model) to 0.90 (strong), and anchor where a concrete FH+PA model lands.

 (2) COST-BASED BUDGET.  The base model caps colonoscopies at K PER PERSON, which
     structurally forbids giving MORE screens to high-risk people -- "targeting"
     could only shift timing, and age dominates timing (verified: changing the
     initial risk belief under a hard cap moves mean colonoscopies by <0.1).  We
     replace the hard cap with a per-colonoscopy disutility c (life-year cost);
     PBVI then chooses screen QUANTITY per person from the value trade-off, so a
     high-risk belief buys more colonoscopies (verified: c=0.06 gives 0.8 colos at
     P(high)=0.05 vs 4.2 at P(high)=0.90).

Comparisons (true CMOST, matched adherence, paired CRN):
  * Fixed population schedule (no risk info)                 -- best-fixed K=1..4
  * Risk-stratified fixed (baseline signal -> 2 tiers)       -- clinical risk strat.
  * PBVI cost-based + FLAT prior (no baseline info)          -- isolates change (2)
  * PBVI cost-based + PERSONALIZED prior (baseline signal)   -- full risk-targeting

We report the CRC-mortality vs colonoscopy-use efficiency frontier, colonoscopy
REALLOCATION and mortality by TRUE risk class, and how targeting strengthens with
the discrimination (AUC) of the baseline risk factors.

Run:  python experiments/risk_factors.py [n]     (default n = 20000)
Outputs: results/risk_factors.json, results/risk_factors_frontier.csv,
         results/risk_factors_auc.csv, paper/figures/risk_factors.png
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
from env.crc_env import (CRCScreeningEnv, EnvConfig, NoScreening,
                         FixedAgeSchedule, WAIT, SCREEN)
from pomdp.model import CRCScreeningPOMDP, NORMAL
from pomdp.pbvi import PBVI, PBVIPolicy

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(os.path.join(RES, 'policies'), exist_ok=True)
os.makedirs(PAPER_FIG, exist_ok=True)

BEST_FIXED = json.load(open(os.path.join(RES, 'best_fixed_schedules.json')))
BEST_FIXED = {int(k): v for k, v in BEST_FIXED.items()}

# AUC -> Gaussian class-mean separation d = sqrt(2) * Phi^{-1}(AUC)
AUC_TO_D = {0.50: 0.0, 0.60: 0.358, 0.70: 0.742, 0.80: 1.190, 0.90: 1.812}

# concrete baseline risk-factor model (class-conditional probabilities)
FH_HIGH, FH_LOW = 0.30, 0.10       # P(family history + | high / low risk class)
PA_HIGH, PA_LOW = 0.25, 0.06       # P(prior adenoma + | high / low risk class)


def solve_pbvi_cost(c, budget=8, expansions=4, n_belief=700):
    """Cost-based PBVI (per-colonoscopy disutility c, effectively uncapped)."""
    path = os.path.join(RES, 'policies', f'pbvi_cost_c{c:.3f}.npz')
    pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=budget,
                              gamma=1.0, screen_disutility=c, use_risk_classes=True)
    solver = PBVI(pomdp, n_belief=n_belief, seed=0)
    solver.solve(expansions=0, verbose=False)
    if os.path.exists(path) and _cache_matches(path, pomdp.NS):
        solver.load(path)
    else:
        solver.solve(expansions=expansions, verbose=False)
        solver.save(path)
    return pomdp, solver


def _cache_matches(path, NS):
    """True iff a saved alpha-vector cache has this model's hidden-state count.

    The state space has changed since some caches under results/policies/ were
    written (a 6- vs 7-clinical-state block); loading one of those silently
    produces alpha vectors of the wrong width and the first q_values() call dies
    in a matmul.  Check the width and re-solve instead (a solve costs ~2 s)."""
    try:
        z = np.load(path, allow_pickle=True)
        key = next(k for k in z.files if k.startswith('a_'))
        return z[key].shape[1] == NS
    except Exception:
        return False


def belief_ph(p, ph):
    b = np.zeros(p.NS)
    b[p.fidx(0, NORMAL)] = 1.0 - ph
    b[p.fidx(1, NORMAL)] = ph
    return b


def gaussian_posterior(true_high, d, rng, prior=0.25):
    """Baseline risk score s ~ N(+-d/2, 1); return P(high | s)."""
    s = rng.normal(d / 2 if true_high else -d / 2, 1.0)
    logit = np.log(prior / (1 - prior)) + d * s
    return 1.0 / (1.0 + np.exp(-logit))


def concrete_posterior(true_high, rng, prior=0.25):
    """P(high | FH, PA) from the concrete family-history + prior-adenoma model."""
    fh = rng.random() < (FH_HIGH if true_high else FH_LOW)
    pa = rng.random() < (PA_HIGH if true_high else PA_LOW)
    lh = prior * (FH_HIGH if fh else 1 - FH_HIGH) * (PA_HIGH if pa else 1 - PA_HIGH)
    ll = (1 - prior) * (FH_LOW if fh else 1 - FH_LOW) * (PA_LOW if pa else 1 - PA_LOW)
    return lh / (lh + ll)


def empirical_auc(scores, labels):
    """AUC via the rank (Mann-Whitney) estimator."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum(); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return (ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ---------------------------------------------------------------------------
def eval_frontier(kind, n, seed, cfg, thr, *, solver=None, p=None,
                  signal='gauss', d=0.0, sched=None, sched_hi=None, sched_lo=None,
                  tau=None, sig_seed=555, post_fn=None):
    """Roll out one policy configuration; return per-patient arrays incl true class.
    kinds: 'fixed' (sched), 'stratified' (sched_hi/lo, tau on baseline p_high),
           'pbvi_flat' (solver, flat prior), 'pbvi_person' (solver, personalized).

    post_fn, if given, overrides `signal`: any callable (true_high, rng) ->
    P(high | baseline observations).  `experiments/risk_panels.py` uses this to
    plug in named risk-factor/risk-test panels."""
    params = build_params('CMOST13', n_patients=500, seed=777)
    eng = CRCEngine(params, rng=np.random.default_rng(0))
    env = CRCScreeningEnv(eng, cfg)
    fixed_pol = FixedAgeSchedule(sched) if sched is not None else None
    fixed_hi = FixedAgeSchedule(sched_hi) if sched_hi is not None else None
    fixed_lo = FixedAgeSchedule(sched_lo) if sched_lo is not None else None
    pbvi_pol = PBVIPolicy(solver) if solver is not None else None

    rows = []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + i)
        obs, info = env.reset()
        true_high = env.pt.individual_risk >= thr
        srng = np.random.default_rng(sig_seed + i)
        if post_fn is not None:
            ph = post_fn(bool(true_high), srng)
        elif signal == 'gauss':
            ph = gaussian_posterior(true_high, d, srng)
        else:
            ph = concrete_posterior(true_high, srng)

        if kind == 'fixed':
            pol = fixed_pol; pol.reset()
        elif kind == 'stratified':
            pol = fixed_hi if ph >= tau else fixed_lo; pol.reset()
        else:
            pol = pbvi_pol; pol.reset()
            pol.b = belief_ph(p, ph if kind == 'pbvi_person' else p.frac_high)

        while not env._done:
            obs, r, term, trunc, info = env.step(pol.act(obs))

        pt = env.pt
        crc_death = pt.death_cause in (2, 3)
        age_death = pt.death_time if pt.death_time > 0 else cfg.max_age
        ly_lost = max(0.0, pt.other_death_time - age_death) if crc_death else 0.0
        rows.append((env.n_colo, int(pt.death_cause == 2), int(pt.ever_clinical),
                     ly_lost, bool(true_high), float(ph)))
    a = np.array(rows, dtype=float)
    return {'n_colo': a[:, 0], 'crc_death': a[:, 1], 'ever_clinical': a[:, 2],
            'ly_lost': a[:, 3], 'true_high': a[:, 4].astype(bool), 'p_high': a[:, 5]}


def summ(arr, ref_lost, mask=None):
    if mask is None:
        mask = np.ones(len(arr['crc_death']), dtype=bool)
    n = int(mask.sum())
    d = arr['crc_death'][mask]
    return {
        'n': n,
        'mean_colo': float(arr['n_colo'][mask].mean()),
        'crc_mortality': float(100 * d.mean()),
        'crc_mortality_se': float(100 * d.std(ddof=1) / np.sqrt(max(n, 1))),
        'crc_incidence': float(100 * arr['ever_clinical'][mask].mean()),
        'LYG_per_1000': float(1000 * (ref_lost - arr['ly_lost'][mask].mean())),
    }


# ---------------------------------------------------------------------------
def main(n=20000, seed=90210):
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)
    z = np.load(os.path.join(RES, 'transitions_stratified.npz'), allow_pickle=True)
    thr = float(z['risk_threshold'])
    prior_high = float(z['frac_high'])

    # reference (no screening) + true-class labels + concrete-model AUC
    ref = eval_frontier('fixed', n, seed, cfg, thr, sched=[])   # empty schedule = no screen
    ref_lost = ref['ly_lost'].mean()
    ref_lost_hi = ref['ly_lost'][ref['true_high']].mean()
    ref_lost_lo = ref['ly_lost'][~ref['true_high']].mean()
    # concrete FH+PA discrimination
    srng = np.random.default_rng(9)
    labels = ref['true_high']
    scores = np.array([concrete_posterior(h, srng) for h in labels])
    auc_concrete = empirical_auc(scores, labels)
    print(f"no-screen CRC mortality={100*ref['crc_death'].mean():.2f}%  "
          f"high-class={100*ref['crc_death'][labels].mean():.2f}%  "
          f"low-class={100*ref['crc_death'][~labels].mean():.2f}%", flush=True)
    print(f"concrete FH+prior-adenoma model discrimination: AUC={auc_concrete:.3f}", flush=True)

    costs = [0.03, 0.06, 0.10]
    solvers = {}
    for c in costs:
        print(f"solving cost-based PBVI c={c} ...", flush=True)
        solvers[c] = solve_pbvi_cost(c)

    # =====================================================================
    # PART A: efficiency frontier (concrete realistic risk factors)
    # =====================================================================
    print("\n[Part A] efficiency frontier ...", flush=True)
    frontier = {}

    def record(label, arr):
        frontier[label] = {
            'all': summ(arr, ref_lost),
            'high': summ(arr, ref_lost_hi, arr['true_high']),
            'low': summ(arr, ref_lost_lo, ~arr['true_high']),
            'colo_high': float(arr['n_colo'][arr['true_high']].mean()),
            'colo_low': float(arr['n_colo'][~arr['true_high']].mean()),
        }
        s = frontier[label]['all']
        print(f"  {label:<34} colo={s['mean_colo']:.2f} "
              f"(hi {frontier[label]['colo_high']:.2f}/lo {frontier[label]['colo_low']:.2f}) "
              f"CRCmort={s['crc_mortality']:.2f}%", flush=True)

    for K in (1, 2, 3, 4):
        record(f'Fixed population x{K}',
               eval_frontier('fixed', n, seed, cfg, thr, sched=BEST_FIXED[K]))

    # risk-stratified fixed: flag anyone whose baseline posterior exceeds the
    # population prior (i.e. has a positive family history or prior adenoma) ->
    # intensive tier.  (The concrete posterior is discrete, so a quantile cut is
    # degenerate; this "any positive risk factor" rule is the clinical analogue.)
    ph_all = np.array([concrete_posterior(h, np.random.default_rng(555 + i))
                       for i, h in enumerate(ref['true_high'])])
    tau = prior_high
    print(f"  risk-stratified flag: baseline posterior > {tau:.3f} -> "
          f"{100*np.mean(ph_all > tau):.1f}% of patients in intensive tier", flush=True)
    for hi_s, lo_s, lab in ((BEST_FIXED[4], BEST_FIXED[2], 'RiskStrat fixed 4/2'),
                            (BEST_FIXED[4], BEST_FIXED[1], 'RiskStrat fixed 4/1')):
        record(lab, eval_frontier('stratified', n, seed, cfg, thr, signal='concrete',
                                  sched_hi=hi_s, sched_lo=lo_s, tau=tau))

    for c in costs:
        p, solver = solvers[c]
        record(f'PBVI cost c={c} flat prior',
               eval_frontier('pbvi_flat', n, seed, cfg, thr, solver=solver, p=p))
        record(f'PBVI cost c={c} +FH/PA (AUC{auc_concrete:.2f})',
               eval_frontier('pbvi_person', n, seed, cfg, thr, solver=solver, p=p,
                             signal='concrete'))

    # =====================================================================
    # PART B: discrimination (AUC) sweep at fixed cost
    # =====================================================================
    print("\n[Part B] discrimination sweep (c=0.06) ...", flush=True)
    c_star = 0.06
    p, solver = solvers[c_star]
    aucsweep = {}
    for auc, d in AUC_TO_D.items():
        arr = eval_frontier('pbvi_person', n, seed, cfg, thr, solver=solver, p=p,
                            signal='gauss', d=d)
        aucsweep[auc] = {
            'all': summ(arr, ref_lost),
            'high': summ(arr, ref_lost_hi, arr['true_high']),
            'low': summ(arr, ref_lost_lo, ~arr['true_high']),
            'colo_high': float(arr['n_colo'][arr['true_high']].mean()),
            'colo_low': float(arr['n_colo'][~arr['true_high']].mean()),
        }
        s = aucsweep[auc]
        print(f"  AUC={auc:.2f}: colo hi/lo {s['colo_high']:.2f}/{s['colo_low']:.2f}  "
              f"high-class mort {s['high']['crc_mortality']:.2f}%  "
              f"overall mort {s['all']['crc_mortality']:.2f}% @ {s['all']['mean_colo']:.2f} colo",
              flush=True)

    out = {
        'config': {'n': n, 'seed': seed, 'risk_threshold': thr, 'costs': costs,
                   'auc_concrete': auc_concrete, 'c_star': c_star, 'tau': tau,
                   'fh': [FH_HIGH, FH_LOW], 'pa': [PA_HIGH, PA_LOW],
                   'ref_mort': 100 * float(ref['crc_death'].mean())},
        'frontier': frontier,
        'auc_sweep': {f'{a:.2f}': v for a, v in aucsweep.items()},
    }
    with open(os.path.join(RES, 'risk_factors.json'), 'w') as f:
        json.dump(out, f, indent=2)
    _write_csvs(frontier, aucsweep)
    _make_figure(frontier, aucsweep, out)
    _print_report(frontier, aucsweep, out)
    print("\nSaved results/risk_factors.json, results/risk_factors_frontier.csv, "
          "results/risk_factors_auc.csv, paper/figures/risk_factors.png")
    return out


def _write_csvs(frontier, aucsweep):
    with open(os.path.join(RES, 'risk_factors_frontier.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['policy', 'mean_colo', 'colo_high', 'colo_low', 'crc_mortality',
                    'crc_mort_high', 'crc_mort_low', 'LYG_per_1000'])
        for lab, v in frontier.items():
            w.writerow([lab, f"{v['all']['mean_colo']:.3f}", f"{v['colo_high']:.3f}",
                        f"{v['colo_low']:.3f}", f"{v['all']['crc_mortality']:.3f}",
                        f"{v['high']['crc_mortality']:.3f}", f"{v['low']['crc_mortality']:.3f}",
                        f"{v['all']['LYG_per_1000']:.1f}"])
    with open(os.path.join(RES, 'risk_factors_auc.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['AUC', 'colo_high', 'colo_low', 'crc_mort_high', 'crc_mort_low',
                    'crc_mort_all', 'mean_colo'])
        for a, v in aucsweep.items():
            w.writerow([f"{a:.2f}", f"{v['colo_high']:.3f}", f"{v['colo_low']:.3f}",
                        f"{v['high']['crc_mortality']:.3f}", f"{v['low']['crc_mortality']:.3f}",
                        f"{v['all']['crc_mortality']:.3f}", f"{v['all']['mean_colo']:.3f}"])


def _make_figure(frontier, aucsweep, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # panel 1: efficiency frontier CRC mortality vs mean colonoscopies
    ax = axes[0]
    def pts(pred):
        xs = [(v['all']['mean_colo'], v['all']['crc_mortality'])
              for lab, v in frontier.items() if pred(lab)]
        return sorted(xs)
    for pred, style, color, lab in [
        (lambda l: l.startswith('Fixed population'), 'o-', '#E45756', 'Fixed population'),
        (lambda l: l.startswith('RiskStrat'), 'D', '#F58518', 'Risk-stratified fixed'),
        (lambda l: 'flat prior' in l, 's--', '#9D7660', 'PBVI cost, flat prior'),
        (lambda l: '+FH/PA' in l, '^-', '#4C78A8', 'PBVI cost, +FH/PA prior')]:
        p = pts(pred)
        if p:
            ax.plot([x for x, _ in p], [y for _, y in p], style, color=color, label=lab, ms=7)
    ax.axhline(out['config']['ref_mort'], ls=':', color='gray', label='No screening')
    ax.set_xlabel('mean colonoscopies / person'); ax.set_ylabel('CRC mortality (%)')
    ax.set_title('Efficiency frontier (lower-left better)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # panel 2: colonoscopy reallocation by true class vs AUC
    ax = axes[1]
    aucs = sorted(aucsweep)
    hi = [aucsweep[a]['colo_high'] for a in aucs]
    lo = [aucsweep[a]['colo_low'] for a in aucs]
    ax.plot(aucs, hi, 's-', color='#E45756', label='true high-risk class')
    ax.plot(aucs, lo, 'o-', color='#4C78A8', label='true low-risk class')
    ax.axvline(out['config']['auc_concrete'], ls=':', color='green',
               label=f"FH+PA (AUC={out['config']['auc_concrete']:.2f})")
    ax.set_xlabel('baseline risk-factor discrimination (AUC)')
    ax.set_ylabel('mean colonoscopies / person')
    ax.set_title('PBVI now targets high-risk (cost budget)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # panel 3: high-class CRC mortality vs AUC
    ax = axes[2]
    mh = [aucsweep[a]['high']['crc_mortality'] for a in aucs]
    ml = [aucsweep[a]['low']['crc_mortality'] for a in aucs]
    ax.plot(aucs, mh, 's-', color='#E45756', label='true high-risk class')
    ax.plot(aucs, ml, 'o-', color='#4C78A8', label='true low-risk class')
    ax.axvline(out['config']['auc_concrete'], ls=':', color='green',
               label=f"FH+PA (AUC={out['config']['auc_concrete']:.2f})")
    ax.set_xlabel('baseline risk-factor discrimination (AUC)')
    ax.set_ylabel('CRC mortality (%)')
    ax.set_title('Mortality by true class vs AUC (c=0.06)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle('Baseline risk factors + cost-based budget let PBVI target high-risk '
                 '(true CMOST, matched adherence)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PAPER_FIG, 'risk_factors.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _print_report(frontier, aucsweep, out):
    print("\n" + "=" * 94)
    print("PART A -- efficiency frontier [mean colo (hi/lo) | CRC mort % (hi/lo class)]")
    print("=" * 94)
    print(f"  {'policy':<36}{'colo':>6}{'hi':>6}{'lo':>6}{'mort%':>7}{'hiMort':>8}{'loMort':>8}{'LYG/1k':>8}")
    for lab, v in frontier.items():
        print(f"  {lab:<36}{v['all']['mean_colo']:>6.2f}{v['colo_high']:>6.2f}"
              f"{v['colo_low']:>6.2f}{v['all']['crc_mortality']:>7.2f}"
              f"{v['high']['crc_mortality']:>8.2f}{v['low']['crc_mortality']:>8.2f}"
              f"{v['all']['LYG_per_1000']:>8.1f}")
    print("\n" + "=" * 94)
    print("PART B -- discrimination (AUC) sweep, cost-based PBVI c=0.06")
    print("=" * 94)
    print(f"  {'AUC':>5}{'colo_hi':>9}{'colo_lo':>9}{'mort_hi':>9}{'mort_lo':>9}"
          f"{'mort_all':>9}{'colo_all':>9}")
    for a in sorted(aucsweep):
        v = aucsweep[a]
        print(f"  {a:>5.2f}{v['colo_high']:>9.2f}{v['colo_low']:>9.2f}"
              f"{v['high']['crc_mortality']:>9.2f}{v['low']['crc_mortality']:>9.2f}"
              f"{v['all']['crc_mortality']:>9.2f}{v['all']['mean_colo']:>9.2f}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n=n)
