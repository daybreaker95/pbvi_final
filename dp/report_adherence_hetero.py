"""Tables + figure for the heterogeneous-adherence experiments
(results/dp/eval_adherence_hetero_n<N>.json from dp/run_adherence_hetero.py).

python -m dp.report_adherence_hetero --n 200000
-> results/dp/report_adherence_hetero_n<N>.md, paper/figures/dp_adherence_hetero.png
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .common import RES, ROOT
from .run_adherence_hetero import SCENARIOS, EXPERIMENTS, AWARE_LAMS, AWARE_ADH

FIG = os.path.join(ROOT, 'paper', 'figures')

LABEL = {
    'homog_a07': 'homogeneous, α = 0.7 (Table 10)',
    'e1_beta_k5': 'Beta κ = 5', 'e1_beta_k2': 'Beta κ = 2', 'e1_beta_k1': 'Beta κ = 1', 'e1_beta_k05': 'Beta κ = 0.5',
    'e2_markov_a': 'Markov .5/.85/.25', 'e2_markov_b': 'Markov .5/.90/.15',
    'e3_mix': 'mixture 30 % @ .15, 70 % @ .90',
    'e4_beta_k1_rho02': 'Beta κ = 1, ρ = 0.2', 'e4_beta_k1_rho04': 'Beta κ = 1, ρ = 0.4',
}


def arm_kind(tag):
    if tag.startswith('dpaware'):
        return 'aware'
    if tag.startswith('dp_'):
        return 'dp'
    if tag.startswith('q10y_recall'):
        return 'recall'
    return 'fixed'


def cell(r):
    return f"{r['colos_per_person']:.2f} → {r['crc_death_per_100k']:.0f} ({r['death_reduction_pct']:.0f} %)"


def pick_matched_aware(block, ref_tag):
    """The aware arm whose attended volume is nearest to (and not above by
    more than 3 %) the unchanged policy's volume in the same scenario."""
    ref = block['arms'][ref_tag]['rates']['colos_per_person']
    cands = [(t, a['rates']['colos_per_person']) for t, a in block['arms'].items() if arm_kind(t) == 'aware']
    if not cands:
        return None
    ok = [c for c in cands if c[1] <= ref * 1.03] or cands
    return min(ok, key=lambda c: abs(c[1] - ref))[0]


def write(n):
    res = json.load(open(os.path.join(RES, f'eval_adherence_hetero_n{n}.json')))
    none = res['none']['rates']['crc_death_per_100k']
    L = [f'# Heterogeneous adherence (engine, n = {n:,} per arm)\n',
         f'No screening: {none:.0f} CRC deaths / 100 000. Cells: colonoscopies per person → CRC deaths per 100 000 '
         f'(mortality reduction). Fixed = 10-yearly at 50/60/70, slot lost if missed; recall = annual re-invitation '
         f'until attended; adaptive = headline policy (λ = 0.001561) unchanged, re-planning after a no-show.\n']

    def table(scens, title, extra_cols=()):
        L.append(f'\n## {title}\n')
        hdr = '| scenario | mean attend. | never-attended share | fixed | fixed + recall | adaptive (unchanged) |'
        hdr += ''.join(f' {c} |' for c in extra_cols)
        L.append(hdr); L.append('|' + '---|' * (6 + len(extra_cols)))
        for s in scens:
            if s not in res:
                continue
            b = res[s]
            tags = {arm_kind(t): t for t in b['arms'] if arm_kind(t) != 'aware'}
            never = b['arms'][tags['dp']]['rates'].get('never_attended_share')
            row = (f"| {LABEL.get(s, s)} | {b['mean_attendance']:.2f} | "
                   f"{'' if never is None else f'{100 * never:.0f} %'} | "
                   f"{cell(b['arms'][tags['fixed']]['rates'])} | {cell(b['arms'][tags['recall']]['rates'])} | "
                   f"{cell(b['arms'][tags['dp']]['rates'])} |")
            for c in extra_cols:
                if c == 'adaptive (adherence-aware, matched volume)':
                    t = pick_matched_aware(b, tags['dp'])
                    row += f" {cell(b['arms'][t]['rates']) if t else '—'} |"
            L.append(row)

    def subgroup_table(scen, title):
        b = res.get(scen)
        if not b:
            return
        L.append(f'\n### {title}: outcomes within adherence subgroups\n')
        L.append('CRC deaths / 100 000 within the subgroup (mortality reduction vs the same persons unscreened); '
                 'colonoscopies per person in the subgroup.\n')
        groups = None
        rows = []
        for t, a in b['arms'].items():
            sg = a.get('subgroups')
            if not sg:
                continue
            groups = list(sg)
            rows.append((t, sg))
        if not rows:
            return
        L.append('| arm | ' + ' | '.join(f"{g} ({100 * rows[0][1][g]['share']:.0f} % of pop.)" for g in groups) + ' |')
        L.append('|---|' + '---|' * len(groups))
        for t, sg in rows:
            L.append(f'| {t} | ' + ' | '.join(
                f"{sg[g]['crc_death_per_100k']:.0f} ({sg[g]['death_reduction_pct']:.0f} %), {sg[g]['colos_per_person']:.2f} c"
                for g in groups) + ' |')

    table(EXPERIMENTS['e1'], 'E1 — same mean attendance, growing between-person heterogeneity (Beta propensities)')
    subgroup_table('e1_beta_k1', 'E1, Beta κ = 1')
    table(EXPERIMENTS['e2'], 'E2 — state-dependent (habitual) attendance: P(attend | first / after attending / after missing)')
    table(EXPERIMENTS['e3'] + ['e1_beta_k1', 'e2_markov_a'],
          'E3 — adherence-aware policy (latent 2-class adherence in the hidden state)',
          extra_cols=('adaptive (adherence-aware, matched volume)',))
    if 'e3_mix' in res:
        b = res['e3_mix']
        L.append('\nAware-policy λ grid under the mixture truth (λ per attended colonoscopy):\n')
        L.append('| λ | colonoscopies | invitations / person | CRC deaths / 100k | deaths averted / 1000 colos |')
        L.append('|---|---|---|---|---|')
        for t, a in b['arms'].items():
            if arm_kind(t) != 'aware':
                continue
            r = a['rates']
            inv = np.mean([v['invites_per_person'] for v in a['subgroups'].values()]) if a.get('subgroups') else float('nan')
            inv = sum(v['invites_per_person'] * v['share'] for v in a['subgroups'].values()) if a.get('subgroups') else float('nan')
            L.append(f"| {t.split('_lam')[1].split('_')[0]} | {r['colos_per_person']:.2f} | {inv:.2f} | "
                     f"{r['crc_death_per_100k']:.0f} | {r['deaths_averted_per_1000_colos']:.2f} |")
        subgroup_table('e3_mix', 'E3, mixture truth')
    table(['e1_beta_k1'] + EXPERIMENTS['e4'], 'E4 — non-adherence correlated with CRC risk (missing not at random)',
          extra_cols=('adaptive (adherence-aware, matched volume)',))
    subgroup_table('e4_beta_k1_rho04', 'E4, ρ = 0.4')
    out = os.path.join(RES, f'report_adherence_hetero_n{n}.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('wrote', out)
    return res


def figure(res, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    none = res['none']['rates']['crc_death_per_100k']
    style = {'fixed': ('tab:gray', 'fixed 10-y'), 'recall': ('tab:blue', 'fixed + recall'),
             'dp': ('tab:red', 'adaptive'), 'aware': ('tab:green', 'adaptive, adherence-aware')}
    # (a) E1: deaths vs kappa
    ax = axes[0]
    scens = EXPERIMENTS['e1']
    x = np.arange(len(scens))
    for k in ('fixed', 'recall', 'dp'):
        ys = [next(a['rates']['crc_death_per_100k'] for t, a in res[s]['arms'].items() if arm_kind(t) == k)
              for s in scens if s in res]
        ax.plot(x[:len(ys)], ys, 'o-', color=style[k][0], label=style[k][1])
    ax.set_xticks(x); ax.set_xticklabels(['κ=∞', 'κ=5', 'κ=2', 'κ=1', 'κ=0.5'])
    ax.set_xlabel('between-person heterogeneity (Beta κ), mean attendance 0.7')
    ax.set_ylabel('CRC deaths / 100 000'); ax.axhline(none, color='k', ls=':', lw=1); ax.legend(fontsize=8)
    ax.set_title('E1: heterogeneity at fixed mean')
    # (b) efficiency frontier across all scenarios
    ax = axes[1]
    for s, b in res.items():
        if s == 'none':
            continue
        for t, a in b['arms'].items():
            k = arm_kind(t); r = a['rates']
            ax.scatter(r['colos_per_person'], r['crc_death_per_100k'], color=style[k][0], s=22,
                       marker='s' if s.startswith('e4') else ('^' if s.startswith('e2') else 'o'), alpha=.8)
    for k, (c, l) in style.items():
        ax.scatter([], [], color=c, label=l)
    ax.axhline(none, color='k', ls=':', lw=1)
    ax.set_xlabel('colonoscopies per person (attended)'); ax.set_ylabel('CRC deaths / 100 000')
    ax.set_title('all scenarios (○ Beta/mixture, △ Markov, □ MNAR)'); ax.legend(fontsize=8)
    # (c) subgroup: E3 mixture, deaths within low-adherence class
    ax = axes[2]
    b = res.get('e3_mix')
    if b:
        tags = [t for t in b['arms'] if b['arms'][t].get('subgroups')]
        groups = list(b['arms'][tags[0]]['subgroups'])
        w = 0.8 / len(tags)
        for i, t in enumerate(tags):
            sg = b['arms'][t]['subgroups']
            ax.bar(np.arange(len(groups)) + i * w, [sg[g]['death_reduction_pct'] for g in groups], w,
                   color=style[arm_kind(t)][0], alpha=.55 + .45 * (arm_kind(t) != 'aware'),
                   label=t.replace('_e3_mix', ''))
        ax.set_xticks(np.arange(len(groups)) + w * (len(tags) - 1) / 2); ax.set_xticklabels(groups)
        ax.set_ylabel('mortality reduction within class (%)'); ax.legend(fontsize=6)
        ax.set_title('E3: who benefits (mixture truth)')
    fig.tight_layout()
    out = os.path.join(FIG, 'dp_adherence_hetero.png')
    fig.savefig(out, dpi=150)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000)
    a = ap.parse_args()
    res = write(a.n)
    figure(res, a.n)


if __name__ == '__main__':
    main()
