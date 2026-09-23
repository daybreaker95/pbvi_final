"""Heterogeneous-adherence experiments E1-E4 (engine, n = 200 000 per arm).

Every scenario is a per-person attendance process (dp/adherence.py) applied
to the same three programmes as Table 10: the fixed 10-yearly programme
(slot lost), the same with annual recall, and the headline adaptive policy
unchanged (it re-plans after a no-show). All scenarios share the population
mean attendance of the homogeneous alpha = 0.7 reference unless stated.

  E1  same mean, growing heterogeneity      beta(mean .7, kappa 5 / 2 / 1 / 0.5)
  E2  state-dependent ("habit") attendance  markov(p_first .5, p_aa .85/.90, p_ma .25/.15)
  E3  adherence-aware policy                truth = 2-class mixture (30 % @ .15, 70 % @ .90);
                                            arms add the policy re-solved with the latent
                                            adherence class (dp/adherence_model.py, lambda grid),
                                            and that policy deployed under the misspecified
                                            E1 (kappa = 1) and E2 truths
  E4  missing-not-at-random                 beta(kappa 1) with Gaussian-copula correlation
                                            rho = 0.2 / 0.4 between risk rank and NON-adherence

Outcomes per arm: the Table-10 rates plus outcomes within propensity
subgroups (low < 0.3, mid, high >= 0.7, and the latent class for E3),
paired across arms because the propensities depend on the chunk seed only.

python -m dp.run_adherence_hetero --n 200000 --workers 5 [--experiments e1,e2,e3,e4]
python -m dp.run_adherence_hetero --solve-aware --solver-workers 3     (E3 policies first)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .common import RES
from .engine_runner import job_list, run_jobs, aggregate, efficiency, paired_diff, load_chunk
from .evaluate import summary_table
from .adherence import propensities, propensity_groups, spec_mean

POL_M = os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex1.npz')
POL_F = os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex2.npz')
KERNELS = os.path.join(RES, 'kernels_c6b.npz')
AGES = [50, 60, 70]

MEAN = 0.7
SCENARIOS = {
    'homog_a07':    dict(kind='homog', alpha=MEAN),
    'e1_beta_k5':   dict(kind='beta', mean=MEAN, kappa=5.0),
    'e1_beta_k2':   dict(kind='beta', mean=MEAN, kappa=2.0),
    'e1_beta_k1':   dict(kind='beta', mean=MEAN, kappa=1.0),
    'e1_beta_k05':  dict(kind='beta', mean=MEAN, kappa=0.5),
    'e2_markov_a':  dict(kind='markov', p_first=0.5, p_aa=0.85, p_ma=0.25),
    'e2_markov_b':  dict(kind='markov', p_first=0.5, p_aa=0.90, p_ma=0.15),
    'e3_mix':       dict(kind='mixture', probs=[0.15, 0.90], weights=[0.3, 0.7]),
    'e4_beta_k1_rho02': dict(kind='beta', mean=MEAN, kappa=1.0, rho=0.2),
    'e4_beta_k1_rho04': dict(kind='beta', mean=MEAN, kappa=1.0, rho=0.4),
}
EXPERIMENTS = {
    'e1': ['homog_a07', 'e1_beta_k5', 'e1_beta_k2', 'e1_beta_k1', 'e1_beta_k05'],
    'e2': ['e2_markov_a', 'e2_markov_b'],
    'e3': ['e3_mix'],
    'e4': ['e4_beta_k1_rho02', 'e4_beta_k1_rho04'],
}
# adherence-aware policy: the latent class model it reasons about (= the E3 truth)
AWARE_ADH = dict(probs=[0.15, 0.90], prior=[0.3, 0.7])
AWARE_LAMS = [0.001561, 0.00205, 0.002691]
# the aware policies are also deployed (misspecified) under these truths
AWARE_MISSPEC = ['e1_beta_k1', 'e2_markov_a']


def aware_path(lam, sex):
    return os.path.join(RES, 'policies', f'c6bhi_adh_death_lam{lam:.6g}_sex{sex}.npz')


def _solve_one(job):
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    from .adherence_model import solve_adherence_policy
    lam, sex = job['lam'], job['sex']
    out = aware_path(lam, sex)
    if os.path.exists(out):
        return dict(lam=lam, sex=sex, cached=True)
    t0 = time.time()
    print(f'  [solve lam={lam:.6g} sex={sex} start]', flush=True)
    pol = solve_adherence_policy(sex, KERNELS, AWARE_ADH, weights=dict(death=1.0), lam=lam,
                                 cap=job['cap'], rounds=job['rounds'], rollouts=job['rollouts'],
                                 verbose=True, out=out)
    ev = pol.meta['eval']
    return dict(lam=lam, sex=sex, solve_sec=time.time() - t0, invites=ev['colos'], death=ev['death'],
                objective=pol.meta['objective'], gap=pol.meta['gap'])


def solve_aware(workers=3, cap=1500, rounds=4, rollouts=300, lams=AWARE_LAMS):
    jobs = [dict(lam=l, sex=s, cap=cap, rounds=rounds, rollouts=rollouts) for l in lams for s in (1, 2)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(_solve_one, j) for j in jobs]):
            r = f.result()
            print(f'  aware lam={r["lam"]:.6g} sex={r["sex"]} ' + ('cached' if r.get('cached') else
                  f'invites={r["invites"]:.3f} death={r["death"] * 1e5:.0f} obj={r["objective"]:.6f} '
                  f'gap={r["gap"]:.6f} ({r["solve_sec"]:.0f}s)') + f'  [{time.time() - t0:.0f}s]', flush=True)


def arms_for(scen: str, aware_lams=()):
    spec = SCENARIOS[scen]
    tag_a = 'a07' if scen == 'homog_a07' else scen        # reuse the cached Table-10 arms
    arms = {
        f'q10y_{tag_a}': dict(kind='fixed', ages=AGES, adherence=spec),
        f'q10y_recall_{tag_a}': dict(kind='fixed', ages=AGES, adherence=spec, recall=True),
        f'dp_{tag_a}': dict(kind='policy', policy_male=POL_M, policy_female=POL_F,
                            observed_class=False, adherence=spec),
    }
    if scen == 'homog_a07':                               # legacy float spec -> identical cache tags
        for a in arms.values():
            a['adherence'] = MEAN
    for lam in aware_lams:
        arms[f'dpaware_lam{lam:.6g}_{scen}'] = dict(kind='policy', policy_male=aware_path(lam, 1),
                                                    policy_female=aware_path(lam, 2), observed_class=False,
                                                    adherence=spec)
    return arms


# ---------------------------------------------------------------------------
# subgroup outcomes by adherence propensity (paired: propensity depends on the
# chunk seed and the population only, not on the arm)
# ---------------------------------------------------------------------------
GROUP_NAMES = {'beta': ['low<0.3', 'mid', 'high>=0.7'], 'mixture': ['class_low', 'class_high']}


def _groups(spec, d):
    """Per-person group index for one chunk file (None for kinds with no propensity)."""
    kind = spec['kind']
    n, seed = len(d['risk']), d['summary']['seed']
    if kind == 'beta':
        p, _ = propensities(spec, n, seed, d['risk'].astype(float))
        if 'adh_p' in d:
            assert np.allclose(p, d['adh_p'], atol=1e-5), 'propensity mismatch vs saved hook arrays'
        return propensity_groups(p), GROUP_NAMES['beta']
    if kind == 'mixture':
        _, cls = propensities(spec, n, seed, d['risk'].astype(float))
        return cls, GROUP_NAMES['mixture']
    return None, None


def subgroup_rates(paths, spec, base_paths=None):
    acc = {}
    for i, pth in enumerate(paths):
        d = load_chunk(pth)
        g, names = _groups(spec, d)
        if g is None:
            return None
        dy = d['death_year'].astype(int)
        dead = d['crc_death'] & (dy >= 40)
        base_dead = None
        if base_paths is not None:
            b = load_chunk(base_paths[i])
            assert b['summary']['seed'] == d['summary']['seed']
            base_dead = b['crc_death'] & (b['death_year'].astype(int) >= 40)
        for k, nm in enumerate(names):
            sel = g == k
            r = acc.setdefault(nm, dict(n=0, deaths=0, colos=0, invited=0, attended=0, never=0, base_deaths=0, diff=[]))
            r['n'] += int(sel.sum()); r['deaths'] += int(dead[sel].sum())
            r['colos'] += int(d['n_policy_colo'][sel].sum())
            if 'adh_invited' in d:
                r['invited'] += int(d['adh_invited'][sel].sum()); r['attended'] += int(d['adh_attended'][sel].sum())
                r['never'] += int(((d['adh_attended'][sel] == 0) & (d['adh_invited'][sel] > 0)).sum())
            if base_dead is not None:
                r['base_deaths'] += int(base_dead[sel].sum())
                r['diff'].append((dead[sel].sum() - base_dead[sel].sum()) / sel.sum() * 1e5)
    out = {}
    for nm, r in acc.items():
        n = max(r['n'], 1)
        o = dict(share=r['n'] / sum(x['n'] for x in acc.values()), n=r['n'],
                 crc_death_per_100k=r['deaths'] / n * 1e5, colos_per_person=r['colos'] / n,
                 invites_per_person=r['invited'] / n, attendance=r['attended'] / max(r['invited'], 1),
                 never_attended_share=r['never'] / n)
        if r['diff']:
            o['crc_death_per_100k_none'] = r['base_deaths'] / n * 1e5
            o['death_reduction_pct'] = 100 * (1 - r['deaths'] / max(r['base_deaths'], 1))
            o['paired_death_diff_vs_none'] = float(np.mean(r['diff']))
            o['paired_death_diff_se'] = float(np.std(r['diff'], ddof=1) / np.sqrt(len(r['diff']))) if len(r['diff']) > 1 else float('nan')
        out[nm] = o
    return out


def never_share(paths):
    """Share of persons who were invited at least once and never attended."""
    n = never = 0
    for pth in paths:
        d = load_chunk(pth)
        if 'adh_invited' not in d:
            return None
        n += len(d['risk']); never += int(((d['adh_attended'] == 0) & (d['adh_invited'] > 0)).sum())
    return never / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--experiments', default='e1,e2,e3,e4')
    ap.add_argument('--solve-aware', action='store_true')
    ap.add_argument('--solver-workers', type=int, default=3)
    ap.add_argument('--cap', type=int, default=1500)
    a = ap.parse_args()
    if a.solve_aware:
        solve_aware(workers=a.solver_workers, cap=a.cap)
        return
    exps = a.experiments.split(',')
    have_aware = [l for l in AWARE_LAMS if all(os.path.exists(aware_path(l, s)) for s in (1, 2))]
    scen_arms = {}
    for e in exps:
        for scen in EXPERIMENTS[e]:
            lams = have_aware if (scen == 'e3_mix' or scen in AWARE_MISSPEC) else ()
            scen_arms[scen] = arms_for(scen, lams)
    all_arms = {'none': dict(kind='none')}
    for arms in scen_arms.values():
        all_arms.update(arms)
    # one job queue for everything (better worker utilisation than arm by arm)
    jobs = []
    for tag, arm in all_arms.items():
        jobs += job_list(arm, tag, a.n, 50_000)
    t0 = time.time()
    print(f'[hetero] {len(all_arms)} arms, {len(jobs)} chunk jobs, workers={a.workers}', flush=True)
    run_jobs(jobs, workers=a.workers)
    print(f'[hetero] engine done in {time.time() - t0:.0f}s', flush=True)

    paths = {tag: [j['out_path'] for j in job_list(arm, tag, a.n, 50_000)] for tag, arm in all_arms.items()}
    base = aggregate(paths['none'])
    out_path = os.path.join(RES, f'eval_adherence_hetero_n{a.n}.json')
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res['none'] = dict(rates=base)
    for scen, arms in scen_arms.items():
        spec = SCENARIOS[scen]
        block = dict(spec=spec, mean_attendance=spec_mean(spec), arms={})
        rates = {}
        for tag in arms:
            r = aggregate(paths[tag]); r.update(efficiency(r, base))
            d, se = paired_diff(paths[tag], paths['none'], 'crc_death')
            r['paired_death_diff_vs_none'] = d; r['paired_death_diff_se'] = se
            r['never_attended_share'] = never_share(paths[tag])
            rates[tag] = r
            block['arms'][tag] = dict(arm=arms[tag], rates=r,
                                      subgroups=subgroup_rates(paths[tag], spec, base_paths=paths['none']))
        res[scen] = block
        print(f'\n=== {scen} {spec} (mean attendance {spec_mean(spec):.3f}) ===\n' + summary_table(rates), flush=True)
        for tag, r in rates.items():
            sg = block['arms'][tag]['subgroups']
            if sg:
                print('   ' + tag + ' | ' + '  '.join(f"{k}: {v['crc_death_per_100k']:.0f}/100k "
                                                    f"(-{v['death_reduction_pct']:.0f}%, {v['colos_per_person']:.2f} colos)"
                                                    for k, v in sg.items()), flush=True)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out_path)


if __name__ == '__main__':
    main()
