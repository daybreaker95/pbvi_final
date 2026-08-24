"""Assemble tables / policy descriptions from the pipeline outputs.

write_report(tag, n_head, n_grid, objectives) -> results/dp/report_<tag>.md (+ .json)
"""
from __future__ import annotations

import json
import os

import numpy as np

from .common import RES, SCREEN, WAIT, O_NORMAL, O_ADENOMA, O_MULTI, O_ADVAD, O_NOTEST, OBS_NAMES
from .kernels import TAU_NEVER


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def policy_typical_paths(policy, max_age=80):
    """Describe a policy by the screening ages along canonical observation
    paths: all-normal, and 'finding X at the first screen then normal'."""
    m = policy.model
    out = {}

    def run(path_obs):
        b = m.initial_belief(); tau, ol = m.initial_memory()
        ages = []; k = 0
        for y in range(m.age_min, max_age + 1):
            a = policy.best_action(y, tau, ol, b)
            if a == SCREEN:
                ages.append(y)
                o = path_obs[k] if k < len(path_obs) else O_NORMAL
                k += 1
            else:
                o = O_NOTEST
            M, (tau, ol) = m.step(y, tau, ol, a, o)
            v = b @ M
            if v.sum() <= 0:
                break
            b = v / v.sum()
        return ages

    out['all_normal'] = run([])
    for o, nm in ((O_ADENOMA, 'adenoma_then_normal'), (O_MULTI, 'multi_then_normal'), (O_ADVAD, 'advad_then_normal')):
        out[nm] = run([o])
    out['adenoma_adenoma'] = run([O_ADENOMA, O_ADENOMA])
    out['advad_advad'] = run([O_ADVAD, O_ADVAD])
    return out


def policy_intervals_by_class(policy, max_age=80):
    """Screening ages along the all-normal path when the class is KNOWN."""
    m = policy.model
    out = {}
    for c in range(m.n_class):
        b = m.initial_belief(class_known=c); tau, ol = m.initial_memory()
        ages = []
        for y in range(m.age_min, max_age + 1):
            a = policy.best_action(y, tau, ol, b)
            if a == SCREEN:
                ages.append(y)
            M, (tau, ol) = m.step(y, tau, ol, a, O_NORMAL if a == SCREEN else O_NOTEST)
            v = b @ M
            if v.sum() <= 0:
                break
            b = v / v.sum()
        out[f'class{c}'] = ages
    return out


def matched_volume_table(grid: dict, base: dict, tag_prefix: str):
    """Linear interpolation of the policy frontier at the exact q10y / q5y
    engine volumes, with SEs taken conservatively (max of the neighbours)."""
    pts = sorted((r['colos_per_person'], r['crc_death_per_100k'], r['incidence_per_100k'],
                  r.get('crc_death_se', 0.0), r.get('incidence_se', 0.0))
                 for k, r in grid.items() if k.startswith(tag_prefix) and r['colos_per_person'] > 0)
    lines = ['| comparator | colos | comparator deaths | policy deaths (interp.) | comparator dx | policy dx (interp.) |',
             '|---|---|---|---|---|---|']
    out = {}
    for ref in ('q10y', 'q5y'):
        if ref not in base:
            continue
        v = base[ref]['colos_per_person']
        lo = [q for q in pts if q[0] <= v]
        hi = [q for q in pts if q[0] >= v]
        if not lo or not hi:
            continue
        p0, p1 = max(lo), min(hi)
        w = 0.0 if p1[0] == p0[0] else (v - p0[0]) / (p1[0] - p0[0])
        d = p0[1] + w * (p1[1] - p0[1]); i = p0[2] + w * (p1[2] - p0[2])
        dse = max(p0[3], p1[3]); ise = max(p0[4], p1[4])
        lines.append(f"| {ref} | {v:.3f} | {base[ref]['crc_death_per_100k']:.0f} +- {base[ref]['crc_death_se']:.0f} | "
                     f"{d:.0f} +- {dse:.0f} | {base[ref]['incidence_per_100k']:.0f} +- {base[ref]['incidence_se']:.0f} | {i:.0f} +- {ise:.0f} |")
        out[ref] = dict(vol=v, policy_death=d, policy_death_se=dse, policy_inc=i, policy_inc_se=ise)
    return '\n'.join(lines), out


def md_table(results: dict, keys, headers):
    lines = ['| arm | ' + ' | '.join(headers) + ' |', '|---|' + '---|' * len(headers)]
    for tag, r in results.items():
        vals = []
        for k, fmt in keys:
            v = r.get(k)
            vals.append(fmt.format(v) if isinstance(v, (int, float)) else str(v))
        lines.append(f'| {tag} | ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


KEYS = [('colos_per_person', '{:.3f}'), ('crc_death_per_100k', '{:.1f}'), ('crc_death_se', '{:.1f}'),
        ('incidence_per_100k', '{:.1f}'), ('incidence_se', '{:.1f}'), ('death_reduction_pct', '{:.1f}'),
        ('incidence_reduction_pct', '{:.1f}'), ('deaths_averted_per_1000_colos', '{:.3f}'),
        ('cases_averted_per_1000_colos', '{:.3f}'), ('lyg_per_1000', '{:.1f}'), ('comp_death_per_100k', '{:.1f}')]
HEADERS = ['colos/person', 'CRC death/100k', 'SE', 'incidence/100k', 'SE', 'death red. %', 'inc red. %',
           'deaths averted /1000 colos', 'cases averted /1000 colos', 'LYG/1000', 'comp death/100k']


def write_report(tag, n_head, n_grid, objectives):
    parts = [f'# dp pipeline report ({tag})\n']
    head = _load(os.path.join(RES, f'eval_headline_{tag}_n{n_head}.json'))
    base = _load(os.path.join(RES, f'eval_baseline_{tag}_n{n_head}.json'))
    if head:
        parts.append(f'## Headline engine evaluation (n={n_head:,} per arm, paired chunk seeds)\n')
        parts.append(md_table(head, KEYS, HEADERS) + '\n')
    elif base:
        parts.append(f'## Baseline engine evaluation (n={n_head:,})\n')
        parts.append(md_table(base, KEYS, HEADERS) + '\n')
    for obj in objectives:
        grid = _load(os.path.join(RES, f'eval_grid_{tag}_{obj}_n{n_grid}.json'))
        if grid:
            parts.append(f'## Lambda grid, objective = {obj} (engine, n={n_grid:,} per arm)\n')
            parts.append(md_table(grid, KEYS, HEADERS) + '\n')
            tbl, _ = matched_volume_table(grid, grid, f'{tag}_{obj}_lam')
            parts.append(f'### Frontier interpolated at the comparator volumes ({obj})\n')
            parts.append(tbl + '\n')
        sw = _load(os.path.join(RES, f'sweep_{tag}_{obj}.json'))
        if sw:
            from .sweep import pooled_rows
            rows = pooled_rows(sw['rows'], 0.5)
            parts.append(f'## In-model frontier, objective = {obj} (exact, sex-pooled)\n')
            parts.append('| lambda | colos | death/100k | inc/100k | LY | objective |\n|---|---|---|---|---|---|')
            for r in rows:
                parts.append(f"| {r['lam']:.6g} | {r['colos']:.3f} | {r['death'] * 1e5:.0f} | {r['inc'] * 1e5:.0f} | {r['ly']:.3f} | {r['objective']:.6f} |")
            parts.append('')
    fx = _load(os.path.join(RES, f'fixed_search_{tag}.json'))
    if fx:
        from .fixed_search import frontier
        parts.append('## In-model best fixed schedules by volume (deaths objective)\n')
        parts.append('| colos | death/100k | inc/100k | ages |\n|---|---|---|---|')
        for r in frontier(fx['rows'], 'death'):
            parts.append(f"| {r['colos']:.2f} | {r['death'] * 1e5:.0f} | {r['inc'] * 1e5:.0f} | {r['ages']} |")
        parts.append('')
    # policy descriptions for headline policies
    if head:
        from .solver import load_policy
        from .engine_runner import arm_dir, load_chunk
        import glob
        parts.append('## Policy structure (screening ages along canonical observation paths)\n')
        for arm_tag, r in head.items():
            chunks = sorted(glob.glob(os.path.join(arm_dir(arm_tag), '*.npz')))
            if not chunks:
                continue
            arm = load_chunk(chunks[0])['summary']['arm']
            if arm.get('kind') != 'policy':
                continue
            parts.append(f'### {arm_tag}\n')
            for sex, key in ((1, 'policy_male'), (2, 'policy_female')):
                pol = load_policy(arm[key])
                paths = policy_typical_paths(pol)
                byc = policy_intervals_by_class(pol)
                ev = pol.meta.get('eval', {})
                parts.append(f'*{"male" if sex == 1 else "female"}* - in-model: deaths {ev.get("death", 0) * 1e5:.0f}/100k, '
                             f'dx {ev.get("inc", 0) * 1e5:.0f}/100k, colos {ev.get("colos", 0):.3f}, FIB gap {pol.meta.get("gap", float("nan")):.5f}\n')
                parts.append('| observation path | screening ages |\n|---|---|')
                for nm, ages in list(paths.items()) + [(f'class {c} known, all normal', v) for c, v in enumerate(byc.values())]:
                    parts.append(f'| {nm} | {ages} |')
                parts.append('')
            ages_all, obs_all, n_tot, nscr = [], [], 0, []
            for c in chunks:
                d = load_chunk(c)
                if 'log_y' not in d:
                    continue
                n_tot += len(d['sex']); ages_all.append(d['log_y']); obs_all.append(d['log_obs']); nscr.append(d['n_policy_colo'])
            if n_tot:
                ages_all = np.concatenate(ages_all); obs_all = np.concatenate(obs_all); nscr = np.concatenate(nscr)
                h = np.bincount(ages_all, minlength=81)[40:81] / n_tot
                parts.append('engine: colonoscopies per 1000 persons by age: ' + ', '.join(f'{y}:{h[i] * 1000:.0f}' for i, y in enumerate(range(40, 81)) if h[i] * 1000 >= 5))
                parts.append('')
                bc = np.bincount(nscr)
                parts.append('engine: number of colonoscopies per person: ' + ', '.join(f'{k}:{v / n_tot * 100:.1f}%' for k, v in enumerate(bc) if v / n_tot >= 0.001))
                parts.append('')
                ob = np.bincount(obs_all, minlength=6) / max(len(obs_all), 1)
                parts.append('engine: findings per colonoscopy: ' + ', '.join(f'{OBS_NAMES[i]}:{ob[i] * 100:.1f}%' for i in range(6) if ob[i] > 0))
                parts.append('')
    out_md = os.path.join(RES, f'report_{tag}.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print('saved', out_md)
    return out_md
