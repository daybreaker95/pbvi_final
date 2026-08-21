"""jeon_policy_inrepo_eval.py
=============================
4-way comparison (no_screen / q10y / q5y / PBVI-policy) driven by THIS repo's
own per-patient CMOST engine (env/cmost_individual.CRCEngine) instead of the
CMOST-in-the-loop harness that tests/jeon_4way_eval.py uses.

Why this exists: jeon_4way_eval.py needs NumberCrunching_policy.py and
build_natural_history_transition_matrix.py, which belong to the separate
`cmost_experiment_final` experiment tree -- upstream CMOST has no policy_hook,
so a plain CMOST checkout cannot run it (env/params.find_cmost_experiment_python
says so explicitly). env/cmost_individual.py, by contrast, re-implements every
quarterly CMOST event from the same calculate_sub.prepare_parameters bundle,
and is already what the transition-estimation and elbow-analysis scripts run
on. So the policy CAN be evaluated end-to-end here -- but the numbers are this
engine's, not NumberCrunching's, and are not drop-in comparable with
results/jeon_4way_*.json.

Everything else matches jeon_4way_eval.py: Jeon-2018 12-factor profiles,
bucket-mapped onto CMOST's individual_risk pool, high_risk = top --high-frac by
individual_risk, that label GIVEN to the agent (belief starts on its own risk
block), one FiVI policy per sex trained on
transitions_9state_sex_risk_jeon20pct.npz.

All four arms reuse the SAME cohort (identical gender/individual_risk per
person, identical engine seed) so the comparison is paired, and
deterministic_natural_death pairs the competing-mortality clock across arms.

Run: python tests/jeon_policy_inrepo_eval.py -n 20000
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings

PBVI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PBVI_ROOT)
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from env.params import build_params
from env.cmost_individual import CRCEngine
from pomdp.model_v2 import (CRCScreeningPOMDP9, RISK_LABELS, WAIT, SCREEN,
                            O_NOTEST, O_CANCER, risk_class_from_individual_risk)
from pomdp.fivi import FiVI
from jeon_elbow_analysis import assign_profiles, bucket_map_to_individual_risk

RES = os.path.join(PBVI_ROOT, 'results')
RISK_NPZ = os.path.join(RES, 'transitions_9state_sex_risk_jeon20pct.npz')
MAXY = 100


def train_policy(sex, age_min, age_max, lam, observe_risk):
    pomdp = CRCScreeningPOMDP9(age_min=age_min, age_max=age_max, gamma=0.97, sex=sex,
                               sex_risk_npz=RISK_NPZ, colo_penalty_qaly=lam,
                               observe_risk=observe_risk)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=True, dbbu_interval=1)
    return pomdp, solver


def build_cohort(n, seed, high_frac):
    """Jeon-2018 profiles bucket-mapped onto CMOST's individual_risk pool,
    exactly as jeon_4way_eval.build_nhic_population does."""
    params = build_params('CMOST13', n_patients=20000, seed=seed)
    pool = np.asarray(params['individual_risk'], float)
    prof = assign_profiles(n, seed)
    sex = prof['sex'].astype(int)
    mapped_risk, bucket_high = bucket_map_to_individual_risk(
        sex, prof['composite_rr'], pool, high_frac, np.random.default_rng(seed + 3))

    # The label handed to the agent is EXACTLY the spec: the top high_frac of
    # the cohort by CMOST individual_risk is high_risk, the rest low_risk.
    #
    # That is very nearly -- but not exactly -- the composite-score bucket the
    # transition matrices were estimated under. The bucket mapping draws the
    # high bucket from CMOST's own top-high_frac sub-pool and the low bucket
    # from the rest, so the two ranges would be disjoint if the pool were
    # continuous. It is not: the pool holds ~476 distinct values, and the value
    # sitting on the (1-high_frac) cut appears ~21 times, straddling the split.
    # People who drew exactly that value therefore land in either bucket while
    # a single >= cut puts them all on one side. It affects ~0.06% of a cohort,
    # reported below rather than silently ignored -- if it ever grows beyond a
    # rounding-level share, the donor-pool split and the label have genuinely
    # diverged and the high/low dynamics stop matching the label.
    risk_class, thr = risk_class_from_individual_risk(mapped_risk, high_frac=high_frac)
    n_disagree = int((risk_class != bucket_high.astype(int)).sum())
    print(f'  risk label: individual_risk >= {thr:.6f} -> {RISK_LABELS[1]} '
          f'({risk_class.mean():.4f} of cohort), else {RISK_LABELS[0]}; '
          f'GIVEN TO the agent', flush=True)
    print(f'  (differs from the composite-score bucket on {n_disagree}/{n} = '
          f'{100 * n_disagree / n:.3f}% -- the tied value on the pool cut)', flush=True)
    if n_disagree > 0.01 * n:
        raise RuntimeError(
            f'label and composite-score bucket disagree on {100*n_disagree/n:.2f}% of the '
            f'cohort -- far beyond the tied-boundary-value level, so the top-'
            f'{high_frac:.0%}-by-individual_risk label no longer describes the split '
            f'the transition matrices were estimated under')
    return params, sex, mapped_risk, risk_class, thr


def run_arm(params, sex, mapped_risk, risk_class, seed, arm,
            pomdps=None, solvers=None, screen_ages=(), age_min=40, age_max=80,
            observe_risk=True):
    """One arm over the whole cohort. arm: 'no_screen' | 'fixed' | 'policy'."""
    n = len(sex)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
    eng.deterministic_natural_death = True     # pair the competing-mortality clock
    screen_ages = set(int(x) for x in screen_ages)

    death_age = np.empty(n)
    death_cause = np.zeros(n, np.int8)
    dx_stage = np.zeros(n, np.int8)            # 7..10 CMOST stage, 0 = never diagnosed
    dx_route = np.zeros(n, np.int8)            # 1 = screen-detected, 2 = symptomatic
    n_colo = np.zeros(n, np.int32)

    b0 = {}
    if arm == 'policy':
        for s in (1, 2):
            p = pomdps[s]
            if observe_risk and p.n_risk > 1:
                b0[s] = np.stack([p.initial_belief(risk_class=r) for r in range(p.n_risk)])
            else:
                b0[s] = np.stack([p.initial_belief(), p.initial_belief()])

    t0 = time.time()
    for i in range(n):
        g = int(sex[i])
        pt = eng.new_patient(gender=g, individual_risk=float(mapped_risk[i]))
        p = solver = None
        b = None
        if arm == 'policy':
            p, solver = pomdps[g], solvers[g]
            b = b0[g][int(risk_class[i]) if (observe_risk and p.n_risk > 1) else 0].copy()

        for y in range(1, MAXY + 1):
            o = O_NOTEST
            a = WAIT
            if age_min <= y <= age_max and pt.alive and not pt.ever_clinical:
                if arm == 'policy':
                    a = int(solver.best_action(y, b))
                elif arm == 'fixed':
                    a = SCREEN if y in screen_ages else WAIT
                if a == SCREEN:
                    was = pt.ever_clinical
                    o = int(eng.colonoscopy(pt, y, 1, 'Scre'))
                    n_colo[i] += 1
                    if pt.ever_clinical and not was:
                        dx_route[i] = 1

            symp_before = pt.ever_clinical
            for q in (1, 2, 3, 4):
                if not pt.alive:
                    break
                eng._step_quarter(pt, y, q)
            if pt.ever_clinical and not symp_before and dx_route[i] == 0:
                dx_route[i] = 2
                if o == O_NOTEST:
                    o = O_CANCER      # a cancer surfacing clinically is observed

            if arm == 'policy' and age_min <= y <= age_max:
                bn = p.belief_update(b, y, a, o)
                if bn is not None:
                    b = bn
            if not pt.alive:
                break

        death_age[i] = pt.death_time if not pt.alive else MAXY
        death_cause[i] = pt.death_cause
        if pt.det_stage:
            dx_stage[i] = int(pt.det_stage[0])
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            print(f'    {i+1}/{n}  ({el:.0f}s, {1000*el/(i+1):.2f} ms/pt)', flush=True)

    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage,
                dx_route=dx_route, n_colo=n_colo)


def summarize(m, risk_class):
    crc = m['death_cause'] == 2
    diagnosed = m['dx_stage'] > 0
    early = np.isin(m['dx_stage'], (7, 8))
    hi = risk_class == 1
    return {
        'crc_death_pct': float(100 * crc.mean()),
        'crc_death_pct_high_risk': float(100 * crc[hi].mean()),
        'crc_death_pct_low_risk': float(100 * crc[~hi].mean()),
        'avg_age_at_death': float(m['death_age'].mean()),
        'life_years_from_40': float(m['death_age'].mean() - 40.0),
        'colo_per_person': float(m['n_colo'].mean()),
        'colo_per_person_high_risk': float(m['n_colo'][hi].mean()),
        'colo_per_person_low_risk': float(m['n_colo'][~hi].mean()),
        'incidence_pct': float(100 * diagnosed.mean()),
        'pct_dx_stage_I_II': float(100 * early[diagnosed].mean()) if diagnosed.any() else 0.0,
        'pct_screen_detected': float(100 * (m['dx_route'] == 1)[diagnosed].mean())
                                if diagnosed.any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=999)
    ap.add_argument('--age-min', type=int, default=40)
    ap.add_argument('--age-max', type=int, default=80)
    ap.add_argument('--high-frac', type=float, default=0.20)
    ap.add_argument('--lam', type=float, default=0.0)
    ap.add_argument('--latent-risk', action='store_true',
                    help='do NOT tell the agent its risk class (pre-observed-risk behaviour)')
    ap.add_argument('--scenarios', type=str, default='no_screen,q10y,q5y,policy')
    ap.add_argument('--tag', type=str, default='inrepo')
    a = ap.parse_args()
    observe_risk = not a.latent_risk
    scenarios = [s.strip() for s in a.scenarios.split(',') if s.strip()]

    t0 = time.time()
    print(f'building cohort n={a.n:,} (Jeon-2018 profiles -> CMOST individual_risk) ...',
          flush=True)
    params, sex, mapped_risk, risk_class, thr = build_cohort(a.n, a.seed, a.high_frac)
    if not observe_risk:
        print('  (--latent-risk: that label is NOT given to the agent -- every belief '
              'starts from the population prior)', flush=True)

    pomdps = solvers = None
    if 'policy' in scenarios:
        pomdps, solvers = {}, {}
        for s, name in ((1, 'male'), (2, 'female')):
            p, sol = train_policy(s, a.age_min, a.age_max, a.lam, observe_risk)
            pomdps[s], solvers[s] = p, sol
            gh = sol.gap_history[-1]
            per = [round(g, 4) for g in gh.get('gap_per_start', [])]
            print(f'  FiVI {name}: gap={gh["gap"]:.4f} per_start={per} '
                  f'({time.time()-t0:.1f}s)', flush=True)

    arms = {'no_screen': ('no_screen', ()),
            'q10y': ('fixed', (50, 60, 70)),
            'q5y': ('fixed', (50, 55, 60, 65, 70, 75)),
            'policy': ('policy', ())}
    out = {}
    for sc in scenarios:
        arm, ages = arms[sc]
        print(f'[{sc}] running n={a.n:,} ...', flush=True)
        t1 = time.time()
        m = run_arm(params, sex, mapped_risk, risk_class, a.seed, arm,
                    pomdps=pomdps, solvers=solvers, screen_ages=ages,
                    age_min=a.age_min, age_max=a.age_max, observe_risk=observe_risk)
        out[sc] = summarize(m, risk_class)
        print(f'[{sc}] done ({time.time()-t1:.0f}s)', flush=True)

    hdr = ['crc_death_pct', 'crc_death_pct_high_risk', 'crc_death_pct_low_risk',
           'colo_per_person', 'colo_per_person_high_risk', 'colo_per_person_low_risk',
           'incidence_pct', 'pct_dx_stage_I_II', 'life_years_from_40']
    print()
    print(f'{"metric":30s}' + ''.join(f'{s:>13s}' for s in scenarios))
    for k in hdr:
        print(f'{k:30s}' + ''.join(f'{out[s][k]:13.3f}' for s in scenarios))

    path = os.path.join(RES, f'jeon_policy_{a.tag}.json')
    with open(path, 'w') as f:
        json.dump({'engine': 'env/cmost_individual.CRCEngine (in-repo)',
                   'n': a.n, 'seed': a.seed, 'high_frac': a.high_frac,
                   'risk_threshold': thr, 'observe_risk': observe_risk,
                   'lam': a.lam, 'age_min': a.age_min, 'age_max': a.age_max,
                   'elapsed_sec': time.time() - t0, 'scenarios': out}, f, indent=2)
    print('\nsaved', path, flush=True)


if __name__ == '__main__':
    main()
