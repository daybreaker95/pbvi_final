"""Tables and a figure for the finite-discrimination risk-score scenario.

Reports, in this order of priority:
  1. by OBSERVED score band  - what a programme can actually act on;
  2. by TRUE risk class      - the mechanism;
  3. the joint (true class x score band) cells - what a noisy score gets
     wrong, which is the whole point of the scenario and is invisible to a
     perfect-classifier arm;
  4. the score's own calibration: AUC and the risk gradient it induces, next
     to published CRC risk-score gradients.

python -m dp.report_riskscore --n 200000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES, RUNS, risk_class_of
from .engine_runner import aggregate, paired_diff
from .riskscore import CACHE, load_atoms, _population_scores, root_band_edges

KERNELS = os.path.join(RES, 'kernels_c6b.npz')
GROUPS = {'low (class 0, 50 %)': [0], 'mid (classes 1-2, 45 %)': [1, 2],
          'high (classes 3-5, 5 %)': [3, 4, 5]}


def arm_paths(tag):
    return sorted(glob.glob(os.path.join(RUNS, tag, '*.npz')))


def per_person(tag, thr):
    """Pooled per-person arrays of one arm."""
    out = {k: [] for k in ('crc_death', 'diagnosed', 'colos', 'cls', 'band', 'comp')}
    for p in arm_paths(tag):
        d = np.load(p, allow_pickle=True)
        dy = d['death_year'].astype(int)
        out['crc_death'].append(d['crc_death'] & (dy >= 40))
        out['comp'].append(d['comp_death'] & (dy >= 40))
        out['diagnosed'].append(d['diagnosed'])
        out['colos'].append(d['n_policy_colo'])
        out['cls'].append(risk_class_of(d['risk'].astype(float), thr))
        out['band'].append(d['score_band'] if 'score_band' in d.files
                           else np.full(len(dy), -1, dtype=np.int16))
    return {k: np.concatenate(v) for k, v in out.items()}


def by_group(pp, key, groups):
    rows = []
    for name, ks in groups.items():
        m = np.isin(pp[key], ks)
        n = int(m.sum())
        rows.append(dict(group=name, n=n, colos=float(pp['colos'][m].mean()),
                         death=float(pp['crc_death'][m].mean() * 1e5),
                         inc=float(pp['diagnosed'][m].mean() * 1e5)))
    return rows


def gradient_table(cal, n_bands=10, n_mc=2_000_000):
    """The risk gradient each score induces: lifetime-CRC relative risk by
    score decile, which is what a clinical reader can compare with published
    scores (AUC alone is rank-invariant and hides the tail magnitude)."""
    counts, pool, cls = load_atoms()
    n_i = counts.sum(axis=(0, 2))
    # lifetime CRC risk per atom, from the never-screened cohort
    risk_by_atom = {}
    from .riskscore import nearest_atom
    num = np.zeros(len(pool)); den = np.zeros(len(pool))
    for p in sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', '*.npz'))):
        d = np.load(p, allow_pickle=True)
        ai = nearest_atom(d['risk'].astype(float), pool)
        np.add.at(num, ai, d['diagnosed'].astype(float))
        np.add.at(den, ai, 1.0)
    rate = np.divide(num, np.maximum(den, 1))
    base = num.sum() / den.sum()
    rows = []
    levels = ([('uninformative', cal['control'])] if 'control' in cal else []) + \
        list(cal['sigmas'].items()) + [('ceiling', dict(sigma=0.0, auc=cal['ceiling_auc']))]
    for key, d in levels:
        sigma = d['sigma']
        rng = np.random.default_rng(7)
        p_atom = n_i / n_i.sum()
        idx = rng.choice(len(pool), size=n_mc, p=p_atom)
        S = np.log(pool[idx]) + (sigma * rng.standard_normal(n_mc) if sigma > 0 else 0.0)
        q = np.quantile(S, np.arange(1, n_bands) / n_bands)
        b = np.searchsorted(q, S, side='right')
        rr = [rate[idx[b == k]].mean() / base for k in range(n_bands)]
        rows.append(dict(level=key, sigma=sigma, auc=d['auc'],
                         rr_top=float(rr[-1]), rr_bottom=float(rr[0]),
                         rr_ratio=float(rr[-1] / max(rr[0], 1e-9)), rr_deciles=[float(x) for x in rr]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200_000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    thr = np.load(KERNELS, allow_pickle=True)['thr']
    cal = json.load(open(CACHE))
    ev = json.load(open(os.path.join(RES, f'eval_riskscore_n{a.n}.json')))
    arms = ev['arms']; levels = ev['levels']

    lines = ['# Finite-discrimination risk score (engine, n = {:,} per arm)\n'.format(a.n)]
    lines.append('Every score arm is solved at the shadow price that matches the '
                 'latent-class policy\'s in-model colonoscopy volume '
                 f'({ev["target_volume"]:.3f} per person), so the levels differ in '
                 'information and not in budget.\n')

    lines.append('## Score calibration and the risk gradient it induces\n')
    lines.append('| level | sigma | AUC (within sex, lifetime CRC) | RR top decile | RR bottom decile | top/bottom |')
    lines.append('|---|---|---|---|---|---|')
    for g in gradient_table(cal):
        lines.append(f"| {g['level']} | {g['sigma']:.2f} | {g['auc']:.3f} | {g['rr_top']:.2f} | "
                     f"{g['rr_bottom']:.2f} | {g['rr_ratio']:.1f} |")
    lines.append('')

    lines.append('## Overall engine outcomes\n')
    lines.append('| arm | AUC | colos/person | CRC deaths /100k (SE) | CRC dx /100k (SE) | '
                 'deaths averted /1000 colos | paired vs q10y |')
    lines.append('|---|---|---|---|---|---|---|')
    order = ['none', 'q10y', 'dp_death_lam0.001561_q10y'] + \
            [k for k in arms if k.startswith('score_')] + ['dp_death_lam0.001561_obsclass']
    for k in order:
        if k not in arms:
            continue
        r = arms[k]
        lv = levels.get(k[len('score_'):], {}) if k.startswith('score_') else {}
        auc = f"{lv.get('auc', float('nan')):.3f}" if lv else ('-' if k != 'dp_death_lam0.001561_obsclass' else 'class known')
        pd_ = (f"{r['paired_death_vs_q10y']:+.1f} +- {r['paired_death_vs_q10y_se']:.1f}"
               if 'paired_death_vs_q10y' in r else '-')
        lines.append(f"| {k} | {auc} | {r['colos_per_person']:.3f} | "
                     f"{r['crc_death_per_100k']:.1f} ({r['crc_death_se']:.1f}) | "
                     f"{r['incidence_per_100k']:.1f} ({r['incidence_se']:.1f}) | "
                     f"{r.get('deaths_averted_per_1000_colos', float('nan')):.2f} | {pd_} |")
    lines.append('')

    # per-arm breakdowns
    score_arms = [k for k in arms if k.startswith('score_')]
    for k in ['q10y', 'dp_death_lam0.001561_q10y'] + score_arms + ['dp_death_lam0.001561_obsclass']:
        tag = f'{k}_n{a.n}' if k.startswith('score_') else k
        if not arm_paths(tag):
            continue
        pp = per_person(tag, thr)
        lines.append(f'### {k}: by true risk class\n')
        lines.append('| group | colos | CRC deaths /100k | CRC dx /100k |')
        lines.append('|---|---|---|---|')
        for r in by_group(pp, 'cls', GROUPS):
            lines.append(f"| {r['group']} | {r['colos']:.2f} | {r['death']:.0f} | {r['inc']:.0f} |")
        lines.append('')
        if pp['band'].max() >= 0:
            nb = int(pp['band'].max()) + 1
            lines.append(f'### {k}: by observed score band\n')
            lines.append('| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |')
            lines.append('|---|---|---|---|---|---|')
            for b in range(nb):
                m = pp['band'] == b
                if m.sum() == 0:
                    continue
                lines.append(f"| {b + 1} | {m.mean():.3f} | {pp['colos'][m].mean():.2f} | "
                             f"{pp['crc_death'][m].mean() * 1e5:.0f} | {pp['diagnosed'][m].mean() * 1e5:.0f} | "
                             f"{np.isin(pp['cls'][m], [3, 4, 5]).mean():.3f} |")
            lines.append('')
            # the misclassification cells
            hi = np.isin(pp['cls'], [3, 4, 5])
            low_band = pp['band'] <= nb // 2
            a1 = hi & low_band
            a2 = (pp['cls'] == 0) & (pp['band'] >= nb - 2)
            if a1.sum() == 0 or a2.sum() == 0:
                lines.append('')
                continue
            lines.append(f'*Misclassified cells*: truly high-risk people in the lower half of the '
                         f'score ({a1.mean() * 100:.2f} % of the population) receive '
                         f'{pp["colos"][a1].mean():.2f} colonoscopies and die of CRC at '
                         f'{pp["crc_death"][a1].mean() * 1e5:.0f} per 100 000; truly low-risk people '
                         f'in the top two bands ({a2.mean() * 100:.2f} %) receive '
                         f'{pp["colos"][a2].mean():.2f} colonoscopies for a CRC mortality of '
                         f'{pp["crc_death"][a2].mean() * 1e5:.0f}.\n')
    out = a.out or os.path.join(RES, f'report_riskscore_n{a.n}.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines[:40]))
    print('...\nsaved', out)


if __name__ == '__main__':
    main()
