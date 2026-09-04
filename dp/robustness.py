"""Solver robustness at the headline price (lambda = 0.001561, mortality
objective, cap 1500 / 4 rounds / 300 rollouts as for the headline policies).

Variants re-solve the same MOMDP with different epsilon-greedy rollout seeds
and a different reference screening propensity (the propensity that
generates the initial reachable closure), and 'base' re-solves the exact
headline configuration (seed 0, p_ref 0.12) to check that the solve is
deterministic and to compute the belief-set coverage diagnostic
(PBVISolver.density_diagnostic). Every variant is then deployed in the
engine on the headline arms' first chunk seeds, so the engine result can be
compared with the headline arm population-paired.

python -m dp.robustness --solve --workers 2
python -m dp.robustness --deploy --n 200000 --workers 4
python -m dp.robustness --report --n 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES
from .sweep import run_sweep, policy_path

LAM = 0.001561
KERNELS = os.path.join(RES, 'kernels_c6b.npz')
VARIANTS = {
    'base': dict(p_ref=0.12, seed=0, density=True),
    'seed1': dict(p_ref=0.12, seed=1),
    'seed2': dict(p_ref=0.12, seed=2),
    'pref006': dict(p_ref=0.06, seed=0),
    'pref025': dict(p_ref=0.25, seed=0),
}
HEADLINE_ARM = 'dp_death_lam0.001561_q10y'


def tag_of(name):
    return f'c6bhi_rob_{name}'


def solve(workers):
    for name, cfg in VARIANTS.items():
        print(f'== variant {name}: {cfg}', flush=True)
        run_sweep(KERNELS, 'death', lams=[LAM], tag=tag_of(name), workers=workers, cap=1500, rounds=4,
                  rollouts=300, p_ref=cfg['p_ref'], seed=cfg['seed'], density=cfg.get('density', False))


def deploy(n, workers):
    from .evaluate import evaluate_arms, summary_table
    arms = {'none': {'kind': 'none'}, 'q10y': {'kind': 'fixed', 'ages': [50, 60, 70]},
            HEADLINE_ARM: {'kind': 'policy',
                           'policy_male': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex1.npz'),
                           'policy_female': os.path.join(RES, 'policies', 'c6bhi_death_lam0.001561_sex2.npz'),
                           'observed_class': False}}
    # the headline policy re-deployed under a new tag: reproduces the cached arm
    # bit-for-bit (same seeds, same policy) and records the deployment counters
    arms[HEADLINE_ARM + '_rerun'] = dict(arms[HEADLINE_ARM])
    for name in VARIANTS:
        arms[f'rob_{name}_lam{LAM:g}'] = {'kind': 'policy',
                                          'policy_male': policy_path('death', LAM, 1, tag_of(name)),
                                          'policy_female': policy_path('death', LAM, 2, tag_of(name)),
                                          'observed_class': False}
    res, _ = evaluate_arms(arms, n, chunk=50_000, workers=workers)
    print(summary_table(res), flush=True)
    out = os.path.join(RES, f'eval_robustness_n{n}.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print('saved', out)


def report(n):
    from .solver import load_policy
    from .report import policy_typical_paths
    from .engine_runner import load_chunk, arm_dir
    head = {s: load_policy(os.path.join(RES, 'policies', f'c6bhi_death_lam{LAM:.6g}_sex{s}.npz')) for s in (1, 2)}
    rows = {}
    for name in ['headline'] + list(VARIANTS):
        rows[name] = {}
        for s in (1, 2):
            pol = head[s] if name == 'headline' else load_policy(policy_path('death', LAM, s, tag_of(name)))
            ev = pol.meta['eval']
            rows[name][s] = dict(objective=pol.meta['objective'], death=ev['death'], inc=ev['inc'], colos=ev['colos'],
                                 gap=pol.meta['gap'], paths=policy_typical_paths(pol),
                                 solver=pol.meta.get('solver'), density=pol.meta.get('density'))
            if name != 'headline':
                # alpha-set identity check against the headline policy
                same_keys = set(pol.alphas) == set(head[s].alphas)
                same = same_keys and all(pol.alphas[k].shape == head[s].alphas[k].shape and
                                         np.allclose(pol.alphas[k], head[s].alphas[k]) for k in pol.alphas)
                rows[name][s]['alphas_identical_to_headline'] = bool(same)
                rows[name][s]['paths_identical_to_headline'] = bool(
                    rows[name][s]['paths'] == policy_typical_paths(head[s]))
    # pooled in-model numbers
    pooled = {}
    for name, r in rows.items():
        pooled[name] = {k: 0.5 * (r[1][k] + r[2][k]) for k in ('objective', 'death', 'inc', 'colos', 'gap')}
    # engine
    eng = {}
    ev_path = os.path.join(RES, f'eval_robustness_n{n}.json')
    if os.path.exists(ev_path):
        eng = json.load(open(ev_path))
        # paired vs the headline arm on the same chunks, from per-chunk arrays
        def per(tag):
            out = {}
            for p in sorted(glob.glob(os.path.join(arm_dir(tag), 'seed*_n50000.npz')))[:n // 50_000]:
                d = load_chunk(p); dy = d['death_year'].astype(int)
                out[d['summary']['seed']] = dict(death=(d['crc_death'] & (dy >= 40)).sum() / len(dy) * 1e5,
                                                 dx=d['diagnosed'].sum() / len(dy) * 1e5,
                                                 colos=d['n_policy_colo'].sum() / len(dy),
                                                 crc_death_vec=d['crc_death'], colo_vec=d['n_policy_colo'],
                                                 counters=d['summary'].get('hook_counters'))
            return out
        H = per(HEADLINE_ARM)
        for tag in eng:
            if tag in ('none', 'q10y'):
                continue
            P = per(tag)
            seeds = sorted(set(H) & set(P))
            dd = np.array([P[s]['death'] - H[s]['death'] for s in seeds])
            eng[tag]['paired_vs_headline_death'] = dict(mean=float(dd.mean()), se=float(dd.std(ddof=1) / np.sqrt(len(dd))))
            eng[tag]['identical_outcomes_to_headline'] = bool(all(
                np.array_equal(P[s]['crc_death_vec'], H[s]['crc_death_vec']) and
                np.array_equal(P[s]['colo_vec'], H[s]['colo_vec']) for s in seeds))
            cs = [P[s]['counters'] for s in seeds if P[s]['counters']]
            if cs:
                eng[tag]['hook_counters'] = {k: int(sum(c[k] for c in cs)) for k in cs[0]}
    out = dict(lam=LAM, variants=VARIANTS, in_model=rows, pooled=pooled, engine=eng)
    path = os.path.join(RES, 'robustness_solver.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    L = ['# Solver robustness at lambda = 0.001561 (mortality objective, cap 1500)', '',
         '| variant | p_ref | seed | in-model deaths /100k (pooled) | colos | objective | FIB gap | alphas = headline (m/f) | paths = headline (m/f) |',
         '|---|---|---|---|---|---|---|---|---|']
    for name, p in pooled.items():
        cfg = VARIANTS.get(name, dict(p_ref=0.12, seed=0))
        r = rows[name]
        ai = '/'.join('yes' if r[s].get('alphas_identical_to_headline') else 'no' for s in (1, 2)) if name != 'headline' else '-'
        pi = '/'.join('yes' if r[s].get('paths_identical_to_headline') else 'no' for s in (1, 2)) if name != 'headline' else '-'
        L.append(f"| {name} | {cfg['p_ref']} | {cfg['seed']} | {p['death'] * 1e5:.1f} | {p['colos']:.3f} | {p['objective']:.6f} | {p['gap']:.5f} | {ai} | {pi} |")
    L += ['', '## Screening ages along canonical paths (male / female)', '']
    for name, r in rows.items():
        L.append(f'### {name}')
        for pth in r[1]['paths']:
            L.append(f"- {pth}: {r[1]['paths'][pth]} / {r[2]['paths'][pth]}")
        L.append('')
    if eng:
        L += [f'## Engine (n = {n:,}, paired with the headline arm)', '',
              '| arm | colos | CRC deaths /100k (SE) | paired vs headline arm | identical outcomes | counters |', '|---|---|---|---|---|---|']
        for tag, r in eng.items():
            pv = r.get('paired_vs_headline_death')
            L.append(f"| {tag} | {r['colos_per_person']:.3f} | {r['crc_death_per_100k']:.1f} ({r['crc_death_se']:.1f}) | "
                     f"{'' if pv is None else f'{pv['mean']:+.1f} +- {pv['se']:.1f}'} | {r.get('identical_outcomes_to_headline', '')} | {r.get('hook_counters', '')} |")
    for name in ('base',):
        d = rows[name][1].get('density')
        if d:
            L += ['', '## Belief-set coverage (base re-solve, male / female)', '']
            for s in (1, 2):
                d = rows[name][s]['density']
                L.append(f"- sex {s}: {d['n_points_total']:,} belief points over {d['n_keys']} keys; on-policy weighted mean L1 "
                         f"{d['on_policy']['mean']:.4f} (p95 {d['on_policy']['p95']:.4f}, max {d['on_policy']['max']:.3f}); "
                         f"one-step deviations: mean {d['deviation']['mean']:.4f}, p95 {d['deviation']['p95']:.4f}, max {d['deviation']['max']:.3f}, "
                         f"mass within 0.01 / 0.05 / 0.10: {100 * d['deviation']['frac_le_001']:.1f} / {100 * d['deviation']['frac_le_005']:.1f} / {100 * d['deviation']['frac_le_010']:.1f} %")
    md = path.replace('.json', '.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', path, md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--solve', action='store_true')
    ap.add_argument('--deploy', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.solve:
        solve(a.workers)
    if a.deploy:
        deploy(a.n, a.workers)
    if a.report:
        report(a.n)


if __name__ == '__main__':
    main()
