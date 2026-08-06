"""
nonadherence.py
===============

The value of a PERSONALIZED (PBVI) adaptive policy under IMPERFECT screening
adherence.

Motivation
----------
Under *perfect* adherence the re-optimized best-fixed population schedule and the
PBVI adaptive policy are statistically indistinguishable (see
`results/comparison.json`).  But real patients miss recommended colonoscopies.

  * A FIXED population schedule recommends screening only at preset ages.  A
    missed slot is lost forever -- there is nothing in the policy that reacts to
    the fact that the patient did not show up.
  * The PBVI ADAPTIVE policy conditions on (age, realized colonoscopy history,
    remaining budget).  A missed colonoscopy performs no test (no information) and
    consumes no budget, so at the next epoch the belief is essentially unchanged
    and the policy simply re-recommends screening -- it *re-plans*.  (Verified:
    forcing every screen to be missed leaves the budget full and the policy keeps
    re-inviting the patient.)

We quantify whether this robustness translates into better outcomes for people
who do not adhere to the recommended schedule.

Experiments
-----------
(1) ADHERENCE SWEEP.  Per-visit attendance probability alpha in {1.0, .75, .5,
    .25}; budgets K in {2, 3}; three policies:
      - "Fixed (no recall)": FixedAgeSchedule at the re-optimized best-fixed ages
        (the strong fixed comparator; misses are permanently lost).
      - "Fixed + recall":     same ages, but the program re-invites the patient
        every year until the colonoscopy is completed, capped at K (isolates the
        *retry/recall* effect from personalization).
      - "PBVI adaptive":      belief-tracking POMDP policy, budget K.

(2) NON-ADHERENT SUBGROUP.  A heterogeneous population in which a fraction of
    patients are chronically low-adherence.  We report outcomes *within the
    low-adherence subgroup* -- the people who mostly do NOT get screened on
    schedule -- under each policy.  The subgroup membership is fixed per patient
    index (independent of policy) so the comparison is paired.

Endpoints.  Primary = CRC mortality (Monte-Carlo-stable, SE ~0.05-0.09 pp).
Secondary = CRC incidence, realized mean screening colonoscopies, life-years
gained per 1000 (LYG; noisy) and LYG per colonoscopy.  Paired common random
numbers (engine RNG and adherence RNG both seeded per patient index).

Run:  python experiments/nonadherence.py [n]      (default n = 20000)
Outputs: results/nonadherence.json, results/nonadherence_sweep.csv,
         results/nonadherence_subgroup.csv, paper/figures/nonadherence.png
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
from pomdp.model import CRCScreeningPOMDP
from pomdp.pbvi import PBVI, PBVIPolicy

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
os.makedirs(PAPER_FIG, exist_ok=True)

# Re-optimized best-fixed schedules (results/best_fixed_schedules.json)
BEST_FIXED = {2: [58, 70], 3: [52, 58, 70]}


# ---------------------------------------------------------------------------
# A fixed schedule with an annual recall/reminder until the colonoscopy is done
# ---------------------------------------------------------------------------
class FixedRecallPolicy:
    """Screen starting at each scheduled age; if the patient does not attend,
    re-invite every year until they do, then move on to the next scheduled age.
    At most `len(ages)` colonoscopies, with a minimum spacing `min_gap` between
    completed colonoscopies.  Isolates the recall/retry effect from the
    personalization in PBVI (no belief, no risk inference)."""

    def __init__(self, ages, min_gap=5):
        self.ages = sorted(int(a) for a in ages)
        self.min_gap = int(min_gap)

    def reset(self):
        self.j = 0              # index of the next colonoscopy owed
        self.last_done = -999
        self._cur_age = None

    def act(self, obs):
        age = int(obs['age'])
        self._cur_age = age
        if self.j >= len(self.ages):
            return WAIT
        due = max(self.ages[self.j], self.last_done + self.min_gap)
        return SCREEN if age >= due else WAIT

    def notify(self, a_real):
        if a_real == SCREEN:
            self.last_done = self._cur_age
            self.j += 1


# ---------------------------------------------------------------------------
# Adherence-aware rollout
# ---------------------------------------------------------------------------
def rollout_adh(env, policy, alpha_fn, rng_adh):
    """One episode where each SCREEN recommendation is attended with a
    patient/visit-specific probability.  `alpha_fn(age) -> prob` gives the
    attendance probability; `rng_adh` supplies the Bernoulli draws.  The realized
    (possibly overridden) action is fed back to the policy so belief-tracking /
    recall policies react to what actually happened (no test, budget preserved)."""
    obs, info = env.reset()
    if hasattr(policy, 'reset'):
        policy.reset()
    n_reco = 0
    n_attended = 0
    while not env._done:
        a = policy.act(obs)
        a_real = a
        if a == SCREEN:
            n_reco += 1
            if rng_adh.random() < alpha_fn(int(obs['age'])):
                n_attended += 1
            else:
                a_real = WAIT           # patient did not show up
        # inform the policy of the realized action
        if hasattr(policy, '_last_a'):      # PBVIPolicy belief/budget tracking
            policy._last_a = a_real
        if hasattr(policy, 'notify'):       # FixedRecallPolicy
            policy.notify(a_real)
        obs, r, term, trunc, info = env.step(a_real)

    pt = env.pt
    crc_death = pt.death_cause in (2, 3)
    age_death = pt.death_time if pt.death_time > 0 else env.cfg.max_age
    ly_lost = max(0.0, pt.other_death_time - age_death) if crc_death else 0.0
    return {
        'life_years': age_death,
        'ly_lost_to_crc': ly_lost,
        'crc_death': int(pt.death_cause == 2),
        'colo_death': int(pt.death_cause == 3),
        'ever_clinical': int(pt.ever_clinical),
        'n_colonoscopies': env.n_colo,
        'n_recommended': n_reco,
        'n_attended': n_attended,
    }


# ---------------------------------------------------------------------------
def load_pbvi(budget, n_belief=700):
    pomdp = CRCScreeningPOMDP(age_min=40, age_max=85, budget=budget,
                              gamma=1.0, screen_disutility=0.0,
                              use_risk_classes=True)
    solver = PBVI(pomdp, n_belief=n_belief, seed=0)
    path = os.path.join(RES, 'policies', f'pbvi_k{budget}.npz')
    solver.solve(expansions=0, verbose=False)      # build Vnat + shapes
    if os.path.exists(path):
        solver.load(path)
    else:                                          # solve from scratch if no cache
        solver.solve(expansions=4, verbose=False)
        solver.save(path)
    return solver


def eval_adh(policy_factory, n, seed, cfg, alpha_fn, adh_seed,
             subgroup_fn=None):
    """Roll out a freshly built policy over n paired patients under an adherence
    model.  Returns a dict of per-patient outcome arrays (+ subgroup mask)."""
    params = build_params('CMOST13', n_patients=500, seed=777)
    eng = CRCEngine(params, rng=np.random.default_rng(0))
    env = CRCScreeningEnv(eng, cfg)
    policy = policy_factory()
    rows = []
    sub = []
    for i in range(n):
        eng.rng = np.random.default_rng(seed + i)          # engine CRN
        rng_adh = np.random.default_rng(adh_seed + i)      # adherence CRN
        a_i = (lambda age, i=i: alpha_fn(age, i))
        rows.append(rollout_adh(env, policy, a_i, rng_adh))
        sub.append(subgroup_fn(i) if subgroup_fn else True)
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    arr['_subgroup'] = np.array(sub, dtype=bool)
    return arr


# ---------------------------------------------------------------------------
def summarize(arr, ref_lost, mask=None):
    if mask is None:
        mask = np.ones(len(arr['crc_death']), dtype=bool)
    n = int(mask.sum())
    d = arr['crc_death'][mask]
    inc = arr['ever_clinical'][mask]
    colo = arr['n_colonoscopies'][mask]
    lost = arr['ly_lost_to_crc'][mask]
    reco = arr['n_recommended'][mask]
    att = arr['n_attended'][mask]
    lyg = 1000.0 * (ref_lost - lost.mean())
    return {
        'n': n,
        'crc_mortality': float(100 * d.mean()),
        'crc_mortality_se': float(100 * d.std(ddof=1) / np.sqrt(n)),
        'crc_incidence': float(100 * inc.mean()),
        'mean_colo': float(colo.mean()),
        'mean_recommended': float(reco.mean()),
        'attendance_rate': float(att.sum() / max(reco.sum(), 1e-9)),
        'LYG_per_1000': float(lyg),
        'LYG_per_colo': float(lyg / max(colo.mean(), 1e-9)),
        'ly_lost_mean': float(lost.mean()),
    }


def paired_mort_diff(arr_a, arr_b, mask=None):
    """Paired difference in CRC mortality (percentage points), a - b, with SE."""
    if mask is None:
        mask = np.ones(len(arr_a['crc_death']), dtype=bool)
    da = 100 * arr_a['crc_death'][mask]
    db = 100 * arr_b['crc_death'][mask]
    diff = da - db
    n = len(diff)
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------------------
def main(n=20000, seed=90210, adh_seed=13):
    cfg = EnvConfig(start_age=40, stop_age=90, budget=None, discount=0.0)

    # reference: no screening (for LYG denominator), no adherence relevance
    print("Reference (no screening) ...", flush=True)
    ref = eval_adh(lambda: NoScreening(), n, seed, cfg,
                   alpha_fn=lambda age, i: 1.0, adh_seed=adh_seed)
    ref_lost = ref['ly_lost_to_crc'].mean()
    ref_mort = 100 * ref['crc_death'].mean()
    print(f"  no-screening CRC mortality = {ref_mort:.2f}%%", flush=True)

    # pre-load PBVI solvers (cached)
    solvers = {K: load_pbvi(K) for K in (2, 3)}

    def make_policy(kind, K):
        if kind == 'fixed':
            return FixedAgeSchedule(BEST_FIXED[K])
        if kind == 'recall':
            return FixedRecallPolicy(BEST_FIXED[K])
        if kind == 'pbvi':
            return PBVIPolicy(solvers[K])
        raise ValueError(kind)

    KINDS = [('fixed', 'Fixed (no recall)'),
             ('recall', 'Fixed + recall'),
             ('pbvi', 'PBVI adaptive')]

    # =====================================================================
    # Experiment 1: homogeneous per-visit adherence sweep
    # =====================================================================
    ALPHAS = [1.0, 0.75, 0.5, 0.25]
    sweep = {}       # (K, alpha, kind) -> summary
    raw_sweep = {}   # (K, alpha, kind) -> arr  (for paired diffs)
    print("\n=== Experiment 1: adherence sweep ===", flush=True)
    for K in (2, 3):
        for alpha in ALPHAS:
            afn = (lambda age, i, al=alpha: al)
            for kind, label in KINDS:
                arr = eval_adh(lambda k=kind, KK=K: make_policy(k, KK),
                               n, seed, cfg, alpha_fn=afn, adh_seed=adh_seed)
                s = summarize(arr, ref_lost)
                sweep[(K, alpha, kind)] = s
                raw_sweep[(K, alpha, kind)] = arr
                print(f"  K={K} alpha={alpha:.2f} {label:<18} "
                      f"colo={s['mean_colo']:.2f} attend={s['attendance_rate']:.2f} "
                      f"CRCmort={s['crc_mortality']:.2f}% LYG/1k={s['LYG_per_1000']:.1f}",
                      flush=True)

    # =====================================================================
    # Experiment 2: heterogeneous population + non-adherent subgroup
    # =====================================================================
    # 40% of patients are chronically low-adherence (attend 20% of invitations),
    # 60% are high-adherence (attend 90%).  Class fixed per patient index.
    F_LOW, A_LOW, A_HIGH = 0.40, 0.20, 0.90

    def is_low(i):
        return np.random.default_rng(777_000 + i).random() < F_LOW

    def hetero_alpha(age, i):
        return A_LOW if is_low(i) else A_HIGH

    print("\n=== Experiment 2: heterogeneous population, non-adherent subgroup "
          f"({int(F_LOW*100)}%% low-adherence @ {A_LOW:.0%}, "
          f"rest @ {A_HIGH:.0%}) ===", flush=True)
    subgroup = {}      # (K, kind, grp) -> summary
    raw_sub = {}       # (K, kind) -> arr
    for K in (2, 3):
        for kind, label in KINDS:
            arr = eval_adh(lambda k=kind, KK=K: make_policy(k, KK),
                           n, seed, cfg, alpha_fn=hetero_alpha,
                           adh_seed=adh_seed, subgroup_fn=is_low)
            raw_sub[(K, kind)] = arr
            low = arr['_subgroup']
            for grp, mask in (('all', None), ('low_adh', low), ('high_adh', ~low)):
                subgroup[(K, kind, grp)] = summarize(arr, ref_lost, mask)
            s = subgroup[(K, kind, 'low_adh')]
            print(f"  K={K} {label:<18} [LOW-adh subgroup] "
                  f"colo={s['mean_colo']:.2f} CRCmort={s['crc_mortality']:.2f}% "
                  f"LYG/1k={s['LYG_per_1000']:.1f}", flush=True)

    # ---- write outputs ----
    out = {
        'config': {'n': n, 'seed': seed, 'adh_seed': adh_seed,
                   'best_fixed': BEST_FIXED, 'ref_mortality': ref_mort,
                   'ref_lost': ref_lost, 'alphas': ALPHAS,
                   'hetero': {'f_low': F_LOW, 'a_low': A_LOW, 'a_high': A_HIGH}},
        'sweep': {f'K{K}|a{alpha}|{kind}': sweep[(K, alpha, kind)]
                  for (K, alpha, kind) in sweep},
        'subgroup': {f'K{K}|{kind}|{grp}': subgroup[(K, kind, grp)]
                     for (K, kind, grp) in subgroup},
    }
    # paired PBVI - fixed(no recall) mortality differences
    out['paired_pbvi_minus_fixed'] = {}
    for K in (2, 3):
        for alpha in ALPHAS:
            m, se = paired_mort_diff(raw_sweep[(K, alpha, 'pbvi')],
                                     raw_sweep[(K, alpha, 'fixed')])
            out['paired_pbvi_minus_fixed'][f'K{K}|a{alpha}'] = {'d_mort_pp': m, 'se': se}
    out['paired_pbvi_minus_fixed_lowadh'] = {}
    for K in (2, 3):
        low = raw_sub[(K, 'pbvi')]['_subgroup']
        m, se = paired_mort_diff(raw_sub[(K, 'pbvi')], raw_sub[(K, 'fixed')], mask=low)
        out['paired_pbvi_minus_fixed_lowadh'][f'K{K}'] = {'d_mort_pp': m, 'se': se}

    with open(os.path.join(RES, 'nonadherence.json'), 'w') as f:
        json.dump(out, f, indent=2)

    _write_sweep_csv(sweep, ALPHAS)
    _write_subgroup_csv(subgroup)
    _make_figure(sweep, subgroup, ref_mort, ALPHAS)

    _print_report(sweep, subgroup, out, ref_mort, ALPHAS)
    print("\nSaved results/nonadherence.json, "
          "results/nonadherence_sweep.csv, results/nonadherence_subgroup.csv, "
          "paper/figures/nonadherence.png")
    return out


# ---------------------------------------------------------------------------
def _write_sweep_csv(sweep, ALPHAS):
    cols = ['crc_mortality', 'crc_mortality_se', 'crc_incidence', 'mean_colo',
            'attendance_rate', 'LYG_per_1000', 'LYG_per_colo']
    with open(os.path.join(RES, 'nonadherence_sweep.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'alpha', 'policy'] + cols)
        for (K, alpha, kind), s in sweep.items():
            w.writerow([K, alpha, kind] + [f"{s[c]:.4f}" for c in cols])


def _write_subgroup_csv(subgroup):
    cols = ['n', 'crc_mortality', 'crc_mortality_se', 'crc_incidence',
            'mean_colo', 'attendance_rate', 'LYG_per_1000', 'LYG_per_colo']
    with open(os.path.join(RES, 'nonadherence_subgroup.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['budget', 'policy', 'group'] + cols)
        for (K, kind, grp), s in subgroup.items():
            w.writerow([K, kind, grp] + [f"{s[c]:.4f}" for c in cols])


def _make_figure(sweep, subgroup, ref_mort, ALPHAS):
    kinds = [('fixed', 'Fixed (no recall)', '#E45756', 'o-'),
             ('recall', 'Fixed + recall', '#F58518', 'D--'),
             ('pbvi', 'PBVI adaptive', '#4C78A8', 's-')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # panels 1-2: CRC mortality vs adherence, budgets 2 and 3
    for ax, K in zip(axes[:2], (2, 3)):
        for kind, label, color, style in kinds:
            y = [sweep[(K, a, kind)]['crc_mortality'] for a in ALPHAS]
            ax.plot(ALPHAS, y, style, color=color, label=label)
        ax.axhline(ref_mort, ls=':', color='gray', label='No screening')
        ax.set_xlabel('per-visit adherence  $\\alpha$')
        ax.set_ylabel('CRC mortality (%)')
        ax.set_title(f'Budget K = {K}: mortality vs adherence')
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # panel 3: non-adherent subgroup, CRC mortality by policy (grouped bars)
    ax = axes[2]
    labels = [lab for _, lab, _, _ in kinds]
    x = np.arange(len(labels))
    w = 0.35
    for j, K in enumerate((2, 3)):
        vals = [subgroup[(K, kind, 'low_adh')]['crc_mortality'] for kind, *_ in kinds]
        err = [subgroup[(K, kind, 'low_adh')]['crc_mortality_se'] for kind, *_ in kinds]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=err, capsize=3,
               label=f'K = {K}', color=['#4C78A8', '#72B7B2'][j], alpha=0.85)
    ax.axhline(ref_mort, ls=':', color='gray', label='No screening')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8)
    ax.set_ylabel('CRC mortality (%)')
    ax.set_title('Non-adherent subgroup\n(chronic low adherence)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)

    fig.suptitle('Personalized (PBVI) vs fixed screening under imperfect adherence '
                 '(true CMOST outcomes)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(PAPER_FIG, 'nonadherence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _print_report(sweep, subgroup, out, ref_mort, ALPHAS):
    print("\n" + "=" * 92)
    print("EXPERIMENT 1 -- adherence sweep (CRC mortality %, realized colonoscopies)")
    print("=" * 92)
    for K in (2, 3):
        print(f"\n  Budget K = {K}   (no-screening CRC mortality = {ref_mort:.2f}%)")
        print(f"  {'alpha':>6} | {'Fixed(no recall)':>22} | {'Fixed + recall':>22} | "
              f"{'PBVI adaptive':>22}")
        print(f"  {'':>6} | {'colo   mort%':>22} | {'colo   mort%':>22} | {'colo   mort%':>22}")
        print("  " + "-" * 82)
        for a in ALPHAS:
            cells = []
            for kind in ('fixed', 'recall', 'pbvi'):
                s = sweep[(K, a, kind)]
                cells.append(f"{s['mean_colo']:>5.2f}  {s['crc_mortality']:>5.2f}")
            print(f"  {a:>6.2f} | {cells[0]:>22} | {cells[1]:>22} | {cells[2]:>22}")
        print("  paired PBVI - Fixed(no recall) mortality diff (pp, - = PBVI better):")
        for a in ALPHAS:
            d = out['paired_pbvi_minus_fixed'][f'K{K}|a{a}']
            sig = '*' if abs(d['d_mort_pp']) > 2 * d['se'] else ' '
            print(f"     alpha={a:.2f}:  {d['d_mort_pp']:+.3f} +/- {d['se']:.3f} pp {sig}")

    print("\n" + "=" * 92)
    print("EXPERIMENT 2 -- NON-ADHERENT subgroup (chronic low adherence)")
    print("=" * 92)
    for K in (2, 3):
        print(f"\n  Budget K = {K}")
        print(f"  {'policy':<20}{'group':<10}{'n':>7}{'colo':>7}{'attend':>8}"
              f"{'CRCmort%':>10}{'CRCinc%':>9}{'LYG/1k':>8}")
        print("  " + "-" * 79)
        for kind, label in (('fixed', 'Fixed(no recall)'),
                            ('recall', 'Fixed + recall'),
                            ('pbvi', 'PBVI adaptive')):
            for grp in ('all', 'low_adh', 'high_adh'):
                s = subgroup[(K, kind, grp)]
                print(f"  {label:<20}{grp:<10}{s['n']:>7d}{s['mean_colo']:>7.2f}"
                      f"{s['attendance_rate']:>8.2f}{s['crc_mortality']:>10.2f}"
                      f"{s['crc_incidence']:>9.2f}{s['LYG_per_1000']:>8.1f}")
        d = out['paired_pbvi_minus_fixed_lowadh'][f'K{K}']
        sig = '*' if abs(d['d_mort_pp']) > 2 * d['se'] else ' '
        print(f"  -> LOW-adherence subgroup, paired PBVI - Fixed(no recall) mortality: "
              f"{d['d_mort_pp']:+.3f} +/- {d['se']:.3f} pp {sig}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n=n)
