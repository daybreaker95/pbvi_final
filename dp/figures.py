"""Figures for the dp pipeline (matplotlib, PNG into paper/figures/dp_*.png).

python -m dp.figures --tag c3 --n-head 1000000 --n-grid 200000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import RES, ROOT, SCREEN_OBS, OBS_NAMES

FIG = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG, exist_ok=True)


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def frontier_figure(tag, n_head, n_grid, objectives):
    base = _load(os.path.join(RES, f'eval_baseline_{tag}_n{n_head}.json')) or {}
    head = _load(os.path.join(RES, f'eval_headline_{tag}_n{n_head}.json')) or {}
    fixed = _load(os.path.join(RES, f'fixed_search_{tag}.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = {'death': 'C0', 'inc': 'C2', 'ly': 'C4', 'combo': 'C5'}
    for obj in objectives:
        grid = _load(os.path.join(RES, f'eval_grid_{tag}_{obj}_n{n_grid}.json'))
        if not grid:
            continue
        pts = [(r['colos_per_person'], r['crc_death_per_100k'], r['incidence_per_100k'], r.get('crc_death_se', 0), r.get('incidence_se', 0))
               for k, r in grid.items() if k.startswith(f'{tag}_{obj}_lam')]
        pts.sort()
        if not pts:
            continue
        x = [p[0] for p in pts]
        axes[0].errorbar(x, [p[1] for p in pts], yerr=[p[3] for p in pts], fmt='o-', ms=4, color=colors.get(obj, 'C0'),
                         label=f'adaptive DP policy ({obj} objective), engine n={n_grid // 1000}k')
        axes[1].errorbar(x, [p[2] for p in pts], yerr=[p[4] for p in pts], fmt='o-', ms=4, color=colors.get(obj, 'C0'),
                         label=f'adaptive DP policy ({obj} objective)')
    src = head or base
    for tagk, mk, lab in (('q10y', 's', 'fixed 10-y (50/60/70)'), ('q5y', 'D', 'fixed 5-y (50..75)')):
        if tagk in src:
            r = src[tagk]
            axes[0].errorbar([r['colos_per_person']], [r['crc_death_per_100k']], yerr=[r['crc_death_se']], fmt=mk, ms=9, color='k', label=lab)
            axes[1].errorbar([r['colos_per_person']], [r['incidence_per_100k']], yerr=[r['incidence_se']], fmt=mk, ms=9, color='k', label=lab)
    for k, r in src.items():
        if k.startswith('bestfixed'):
            axes[0].plot([r['colos_per_person']], [r['crc_death_per_100k']], '^', ms=9, color='C3')
            axes[1].plot([r['colos_per_person']], [r['incidence_per_100k']], '^', ms=9, color='C3')
    if any(k.startswith('bestfixed') for k in src):
        axes[0].plot([], [], '^', ms=9, color='C3', label='best fixed schedule (in-model search, engine-verified)')
    for k, r in src.items():
        if k.endswith('_match_q10y') or k.endswith('_match_q5y'):
            axes[0].plot([r['colos_per_person']], [r['crc_death_per_100k']], '*', ms=14, color='C1')
            axes[1].plot([r['colos_per_person']], [r['incidence_per_100k']], '*', ms=14, color='C1')
    if any(k.endswith('_match_q10y') for k in src):
        axes[0].plot([], [], '*', ms=14, color='C1', label=f'DP policy at matched volume (engine n={n_head // 1000}k)')
    if fixed:
        rows = fixed['rows']
        # in-model fixed frontier as a faint reference
        from .fixed_search import frontier
        fr = frontier(rows, 'death')
        axes[0].plot([r['colos'] for r in fr], [r['death'] * 1e5 for r in fr], ':', color='gray', label='best fixed schedules, in-model (all volumes)')
        fr = frontier(rows, 'inc')
        axes[1].plot([r['colos'] for r in fr], [r['inc'] * 1e5 for r in fr], ':', color='gray')
    if 'none' in src:
        axes[0].axhline(src['none']['crc_death_per_100k'], color='gray', lw=0.8, ls='--')
        axes[1].axhline(src['none']['incidence_per_100k'], color='gray', lw=0.8, ls='--')
    axes[0].set_xlabel('colonoscopies per person (ages 40-80)'); axes[0].set_ylabel('CRC deaths per 100,000')
    axes[1].set_xlabel('colonoscopies per person (ages 40-80)'); axes[1].set_ylabel('CRC diagnoses per 100,000')
    axes[0].set_title('Mortality efficiency frontier (real CMOST engine)'); axes[1].set_title('Incidence efficiency frontier')
    axes[0].legend(fontsize=7, loc='upper right')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, f'dp_frontier_{tag}.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def per_colo_figure(tag, n_head):
    src = _load(os.path.join(RES, f'eval_headline_{tag}_n{n_head}.json')) or _load(os.path.join(RES, f'eval_baseline_{tag}_n{n_head}.json'))
    if not src:
        return None
    arms = [k for k in src if k != 'none']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in ((axes[0], 'deaths_averted_per_1000_colos', 'CRC deaths averted per 1000 colonoscopies'),
                         (axes[1], 'cases_averted_per_1000_colos', 'CRC diagnoses averted per 1000 colonoscopies')):
        vals = [src[a].get(key, 0) for a in arms]
        ax.barh(range(len(arms)), vals, color=['k' if a in ('q10y', 'q5y') else ('C3' if a.startswith('bestfixed') else 'C1') for a in arms])
        ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=7)
        ax.set_xlabel(lab); ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    out = os.path.join(FIG, f'dp_per_colonoscopy_{tag}.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def policy_figure(policy_paths: dict, tag, label):
    """Screening ages along canonical observation paths, per sex."""
    from .solver import load_policy
    from .report import policy_typical_paths, policy_intervals_by_class
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sex in zip(axes, (1, 2)):
        pol = load_policy(policy_paths[sex])
        paths = policy_typical_paths(pol)
        byc = policy_intervals_by_class(pol)
        rows = list(paths.items()) + [(f'class {c} known (all normal)', v) for c, v in enumerate(byc.values())]
        for i, (nm, ages) in enumerate(rows):
            ax.plot(ages, [i] * len(ages), 'o-', color='C1' if 'known' not in nm else 'C0')
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
        ax.set_xlim(40, 81); ax.set_xlabel('age'); ax.set_title(f'{"male" if sex == 1 else "female"}: {label}')
        ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    out = os.path.join(FIG, f'dp_policy_{tag}_{label}.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


def validation_figure(tag):
    v = _load(os.path.join(RES, 'validation.json'))
    if not v or 'occupancy' not in v:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ages = list(range(40, 100))
    for ax, key in zip(axes, ('q10y_sex1', 'none_sex1')):
        if key not in v['occupancy']:
            continue
        e = np.array(v['occupancy'][key]['engine']); m = np.array(v['occupancy'][key]['model'])
        for idx, lab, c in ((slice(1, 5), 'early adenoma (P1-4)', 'C0'), (slice(5, 7), 'advanced adenoma (P5-6)', 'C3'), (slice(7, 11), 'undetected cancer', 'C2')):
            ax.plot(ages, e[:, idx].sum(axis=1) * 100, '-', color=c, label=f'engine: {lab}')
            ax.plot(ages, m[:, idx].sum(axis=1) * 100, '--', color=c, label=f'model: {lab}')
        ax.set_yscale('log'); ax.set_xlabel('age'); ax.set_ylabel('% of age-40 cohort (decision time)')
        ax.set_title(f'{key.split("_")[0]} schedule, male'); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    out = os.path.join(FIG, f'dp_validation_{tag}.png')
    fig.savefig(out, dpi=160); plt.close(fig)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='c3'); ap.add_argument('--n-head', type=int, default=1_000_000)
    ap.add_argument('--n-grid', type=int, default=200_000); ap.add_argument('--objectives', default='death,inc,ly')
    a = ap.parse_args()
    print(frontier_figure(a.tag, a.n_head, a.n_grid, a.objectives.split(',')))
    print(per_colo_figure(a.tag, a.n_head))
    print(validation_figure(a.tag))
