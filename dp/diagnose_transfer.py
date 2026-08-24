"""Model-vs-engine transfer diagnostic for an adaptive policy.

Runs the policy in the engine (quarterly recording + screening log), then
compares, per FIRST-screen finding group (normal / adenoma / multi / advad /
cancer), engine vs model: group share, CRC deaths, diagnoses, colonoscopies.
Also compares the distribution of screening ages.

python -m dp.diagnose_transfer --male results/dp/policies/c3_death_lam0.002076_sex1.npz --female ... --n 100000
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from .common import RUNS, MAP18_TO_CLIN, SCREEN, WAIT, O_NOTEST, O_EXIT, SCREEN_OBS, OBS_NAMES
from .engine_runner import run_arm, aggregate
from .model import METRICS
from .solver import load_policy


def model_by_first_obs(policy=None, prune=1e-10, model=None, action_fn=None):
    """Propagate the policy tree, tagging nodes by the first colonoscopy
    observation; returns per-tag expectations (share, death, inc, colos).
    Either a Policy or (model, action_fn) may be given."""
    m = policy.model if policy is not None else model
    act = policy.best_action if policy is not None else action_fn
    nodes = {(m.initial_memory(), 'none'): [np.asarray(m.initial_belief(), float)]}
    acc = {}

    def add(tag, k, v):
        acc.setdefault(tag, {'death': 0.0, 'inc': 0.0, 'colos': 0.0, 'share': 0.0})
        acc[tag][k] += v
    for y in range(m.age_min, m.age_max + 1):
        nxt = {}
        for ((tau, ol), tag), lst in nodes.items():
            key = (y, tau, ol)
            for u in lst:
                mass = u.sum()
                if mass <= prune:
                    continue
                a = act(y, tau, ol, u / mass)
                add(tag, 'death', float(u @ m.flow['death'][key][a]))
                add(tag, 'inc', float(u @ m.flow['inc'][key][a]))
                if a == SCREEN:
                    add(tag, 'colos', mass)
                if a == SCREEN and tag == 'none':
                    # exits at this first colonoscopy (cancer found / comp death)
                    xo = m.Xs(y, tau, ol)
                    add('first_exit', 'share', float((u @ xo).sum()))
                    add('first_exit', 'death', float(u @ (xo @ m.Vx['death'][y])))
                    add('first_exit', 'inc', float(u @ (xo @ m.Vx['inc'][y])))
                    add('first_exit', 'colos', mass)
                    # the flow above already charged everything to 'none'; move first-screen flows
                    acc['none']['death'] -= float(u @ m.flow['death'][key][a])
                    acc['none']['inc'] -= float(u @ m.flow['inc'][key][a])
                    acc['none']['colos'] -= mass
                    for o, M in m.M[key][a].items():
                        v = u @ M
                        t2 = OBS_NAMES[o]
                        # post-screen mass of this observation branch (alive after the screen)
                        Ko = m.Ks(y, tau, ol)[o]
                        w = u @ Ko
                        add(t2, 'share', float(w.sum()))
                        add(t2, 'colos', float(w.sum()))
                        # natural-history exits during the screen year for this branch
                        olo = [i for i, oo in enumerate(SCREEN_OBS) if oo == o][0]
                        E0 = m.Ew(y, 0, olo)
                        add(t2, 'death', float(w @ (E0 @ m.Vx['death'][y])))
                        add(t2, 'inc', float(w @ (E0 @ m.Vx['inc'][y])))
                        if v.sum() > prune:
                            nxt.setdefault((m.succ[key][a][o], t2), []).append(v)
                else:
                    for o, M in m.M[key][a].items():
                        v = u @ M
                        if v.sum() > prune:
                            nxt.setdefault((m.succ[key][a][o], tag), []).append(v)
        nodes = nxt
    Vn = {k: m.natural_value(k) for k in ('death', 'inc')}
    for ((tau, ol), tag), lst in nodes.items():
        U = np.sum(lst, axis=0)
        add(tag, 'death', float(U @ Vn['death'][(m.age_max + 1, tau, ol)]))
        add(tag, 'inc', float(U @ Vn['inc'][(m.age_max + 1, tau, ol)]))
    acc['none']['share'] = 1.0 - sum(v['share'] for k, v in acc.items() if k != 'none')
    return acc


def engine_by_first_obs(paths, sex_filter=None):
    out = {}
    tot_n = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        n = len(d['sex'])
        sex = d['sex']
        lz, ly, lo = d['log_z'], d['log_y'], d['log_obs']
        first_obs = np.full(n, -1, dtype=np.int8)
        first_y = np.full(n, 0, dtype=np.int16)
        order = np.lexsort((ly, lz))
        seen = np.zeros(n, bool)
        for i in order:
            z = lz[i]
            if not seen[z]:
                seen[z] = True; first_obs[z] = lo[i]; first_y[z] = ly[i]
        sel = np.ones(n, bool) if sex_filter is None else (sex == sex_filter)
        tot_n += sel.sum()
        for tag_code, tag in [(-1, 'none'), (O_EXIT, 'first_exit')] + [(o, OBS_NAMES[o]) for o in SCREEN_OBS]:
            g = sel & (first_obs == tag_code)
            r = out.setdefault(tag, {'n': 0, 'death': 0, 'inc': 0, 'colos': 0})
            r['n'] += int(g.sum()); r['death'] += int(d['crc_death'][g].sum())
            r['inc'] += int(d['diagnosed'][g].sum()); r['colos'] += int(d['n_policy_colo'][g].sum())
    return out, tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--male'); ap.add_argument('--female')
    ap.add_argument('--rule', default=None, help='start,iv_normal,iv_adenoma,iv_multi,iv_advad e.g. 52,10,5,3,3')
    ap.add_argument('--kernels', default=None)
    ap.add_argument('--n', type=int, default=100_000); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--tag', default='diag_policy')
    a = ap.parse_args()
    if a.rule:
        vals = [int(v) for v in a.rule.split(',')]
        arm = {'kind': 'rule', 'start': vals[0], 'intervals': vals[1:5]}
        from .model import ReducedPOMDP
        from .hooks import rule_action_fn
        models = {s: ReducedPOMDP(s, a.kernels) for s in (1, 2)}
        fn = rule_action_fn(vals[0], tuple(vals[1:5]))
    else:
        arm = {'kind': 'policy', 'policy_male': a.male, 'policy_female': a.female, 'observed_class': False}
        pols = {1: load_policy(a.male), 2: load_policy(a.female)}
    paths = run_arm(arm, a.tag, a.n, chunk=50_000, workers=a.workers, quarterly=True)
    for sex in (1, 2):
        acc = model_by_first_obs(pols[sex]) if not a.rule else model_by_first_obs(model=models[sex], action_fn=fn)
        eng, n = engine_by_first_obs(paths, sex_filter=sex)
        print(f'\n=== sex {sex}: first-screen finding groups (share %, deaths/100k of whole cohort, inc/100k, colos/person of whole cohort)')
        print(f"{'group':12s} {'share_m':>8s} {'share_e':>8s} {'death_m':>8s} {'death_e':>8s} {'inc_m':>8s} {'inc_e':>8s} {'colos_m':>8s} {'colos_e':>8s}")
        for tag in ['none', 'first_exit'] + [OBS_NAMES[o] for o in SCREEN_OBS]:
            mm = acc.get(tag, {'share': 0, 'death': 0, 'inc': 0, 'colos': 0}); ee = eng.get(tag, {'n': 0, 'death': 0, 'inc': 0, 'colos': 0})
            print(f"{tag:12s} {mm['share'] * 100:8.2f} {ee['n'] / n * 100:8.2f} {mm['death'] * 1e5:8.0f} {ee['death'] / n * 1e5:8.0f} "
                  f"{mm['inc'] * 1e5:8.0f} {ee['inc'] / n * 1e5:8.0f} {mm['colos']:8.3f} {ee['colos'] / n:8.3f}")
    # screening-age histogram
    ages = np.concatenate([np.load(p, allow_pickle=True)['log_y'] for p in paths])
    h = np.bincount(ages, minlength=81)[40:81]
    print('\nengine screening-age histogram (per 1000 persons):', ' '.join(f'{y}:{h[i] / a.n * 1000:.0f}' for i, y in enumerate(range(40, 81)) if h[i] > 0))
    r = aggregate(paths)
    print('engine totals:', {k: round(v, 3) for k, v in r.items() if k in ('colos_per_person', 'crc_death_per_100k', 'incidence_per_100k')})


if __name__ == '__main__':
    main()
