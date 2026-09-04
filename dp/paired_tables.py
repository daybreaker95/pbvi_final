"""Generating script for the manuscript's engine tables (Tables 1, 4, 5, 6,
the life-year and complication-death contrasts), recomputed from the cached
per-chunk engine output under results/dp/runs/.

Statistics. Arms share chunk seeds, so every contrast is population-paired.
Two standard errors are reported for each paired difference: the chunk-level
SE (20 chunk pairs, ddof = 1, the manuscript's convention, with a t(19) 95 %
interval) and the person-level SE (the chunks contain the same persons in the
same order, so per-person outcome differences are available; SE = sd / sqrt(N)).
Holm-adjusted two-sided p-values (t with 19 df) are given for the family of
death contrasts of Table 5.

python -m dp.paired_tables
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

from .common import RES, RUNS, MAP18_TO_CLIN
from .engine_runner import load_chunk, aggregate, efficiency, arm_dir, class_thresholds

HEAD = ['none', 'q10y', 'bestfixed_q10y_54_64_74', 'dp_death_lam0.001561_q10y', 'dp_inc_lam0.005125_q10y',
        'rule_52_10_5_3_3', 'q5y', 'bestfixed_q5y_48_54_60_66_72_78', 'dp_death_lam0.00069_q5y',
        'dp_inc_lam0.001724_q5y', 'dp_death_lam0.001561_obsclass']
PAIRS = [('dp_death_lam0.001561_q10y', 'q10y'), ('dp_death_lam0.001561_q10y', 'bestfixed_q10y_54_64_74'),
         ('dp_death_lam0.00069_q5y', 'q5y'), ('dp_death_lam0.00069_q5y', 'bestfixed_q5y_48_54_60_66_72_78'),
         ('dp_death_lam0.001561_obsclass', 'q10y'), ('rule_52_10_5_3_3', 'q10y'), ('bestfixed_q10y_54_64_74', 'q10y')]
CUTS = (0.5, 0.8, 0.95, 0.965, 0.98)


def load_arm(tag, n_chunks=None):
    out = {}
    for p in sorted(glob.glob(os.path.join(arm_dir(tag), 'seed*_n50000.npz')))[:n_chunks]:
        d = load_chunk(p)
        out[d['summary']['seed']] = d
    return out


def per_chunk(d):
    dy = d['death_year'].astype(int)
    n = len(dy)
    return dict(n=n, death=(d['crc_death'] & (dy >= 40)).sum() / n * 1e5,
                comp=(d['comp_death'] & (dy >= 40)).sum() / n * 1e5, dx=d['diagnosed'].sum() / n * 1e5,
                colos=d['n_policy_colo'].sum() / n, ly=np.maximum(dy - 1 - 40, 0).sum() / n * 1e3)


def person_vec(d, key):
    dy = d['death_year'].astype(int)
    return {'death': (d['crc_death'] & (dy >= 40)).astype(float) * 1e5,
            'comp': (d['comp_death'] & (dy >= 40)).astype(float) * 1e5,
            'dx': d['diagnosed'].astype(float) * 1e5,
            'colos': d['n_policy_colo'].astype(float),
            'ly': np.maximum(dy - 1 - 40, 0).astype(float) * 1e3}[key]


def paired(A, B, key):
    seeds = sorted(set(A) & set(B))
    dc = np.array([per_chunk(A[s])[key] - per_chunk(B[s])[key] for s in seeds])
    k = len(dc)
    se = dc.std(ddof=1) / np.sqrt(k)
    tcrit = stats.t.ppf(0.975, k - 1)
    pv = np.concatenate([person_vec(A[s], key) - person_vec(B[s], key) for s in seeds])
    se_p = pv.std(ddof=1) / np.sqrt(len(pv))
    t = dc.mean() / se if se > 0 else float('inf')
    return dict(mean=float(dc.mean()), se_chunk=float(se), n_chunks=k, ci95_t=[float(dc.mean() - tcrit * se), float(dc.mean() + tcrit * se)],
                t=float(t), p_two_sided=float(2 * stats.t.sf(abs(t), k - 1)), se_person=float(se_p), n_persons=int(len(pv)),
                ratio_to_se=float(abs(dc.mean()) / se) if se > 0 else float('inf'))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx]); adj[idx] = min(1.0, run)
    return adj.tolist()


def main():
    arms = {t: load_arm(t) for t in HEAD}
    n_chunks = len(arms['none'])
    s0 = sorted(arms['none'])[0]
    for t in HEAD[1:]:
        assert np.array_equal(arms['none'][s0]['risk'], arms[t][s0]['risk']), t
    out = dict(n_chunks=n_chunks)
    # ---- Table 4
    paths = {t: [os.path.join(arm_dir(t), f'seed{s}_n50000.npz') for s in sorted(arms[t])] for t in HEAD}
    base = aggregate(paths['none'])
    t4 = {}
    for t in HEAD:
        r = aggregate(paths[t])
        if t != 'none':
            r.update(efficiency(r, base))
        t4[t] = r
    out['table4'] = t4
    # ---- Table 5 + complication + LY contrasts
    t5 = {}
    for a, b in PAIRS:
        t5[f'{a} - {b}'] = {k: paired(arms[a], arms[b], k) for k in ('colos', 'death', 'dx', 'comp', 'ly')}
    pd = [t5[k]['death']['p_two_sided'] for k in t5]
    for k, adj in zip(t5, holm(pd)):
        t5[k]['death']['p_holm'] = adj
    out['table5'] = t5
    for a, b in (('dp_death_lam0.001561_q10y', 'q10y'), ('dp_death_lam0.00069_q5y', 'q5y')):
        t5.setdefault('extra', {})[f'{a} - {b} (comp, LY)'] = {k: paired(arms[a], arms[b], k) for k in ('comp', 'ly')}
    # ---- Table 6 by true class
    thr = class_thresholds(CUTS)
    groups = {'low': (0,), 'mid': (1, 2), 'high': (3, 4, 5)}
    t6 = {}
    for t in ['none', 'q10y', 'q5y', 'dp_death_lam0.001561_q10y', 'dp_death_lam0.001561_obsclass']:
        row = {}
        for g, v in groups.items():
            N = D = X = C = 0
            for s in sorted(arms[t]):
                d = arms[t][s]
                cls = np.searchsorted(thr, d['risk'].astype(float), side='right')
                m = np.isin(cls, v); dy = d['death_year'].astype(int)
                N += m.sum(); D += (d['crc_death'] & (dy >= 40))[m].sum(); X += d['diagnosed'][m].sum(); C += d['n_policy_colo'][m].sum()
            row[g] = dict(n=int(N), colos=C / N, death=D / N * 1e5, dx=X / N * 1e5)
        t6[t] = row
    out['table6'] = t6
    # ---- Table 1 population definitions (engine side): the 'none' arm's first 20 chunks are the
    # first 20 chunks of the never-screened quarterly cohort (same seeds, no colonoscopy before 50),
    # so the age-40 snapshot mask transfers person-by-person to the q10y / q5y arms.
    nhq = {}
    for p in sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', 'seed*_n50000.npz'))):
        z = np.load(p, allow_pickle=True)
        seed = json.loads(str(z['summary']))['seed']
        if seed in arms['none']:
            nhq[seed] = MAP18_TO_CLIN[z['qr'][157]] >= 0
    t1 = {}
    for t in ('none', 'q10y', 'q5y'):
        for mask_name in ('snapshot_q2_age40', 'death_year_ge41_and_undiagnosed_at_40'):
            N = D = X = 0
            for s in sorted(arms[t]):
                d = arms[t][s]; dy = d['death_year'].astype(int)
                if mask_name == 'snapshot_q2_age40':
                    if s not in nhq:
                        continue
                    m = nhq[s]
                    assert np.array_equal(d['risk'], arms['none'][s]['risk'])
                else:
                    m = (dy >= 41) & ((d['dx_year'] == 0) | (d['dx_year'] > 40))
                N += m.sum(); D += (d['crc_death'] & (dy >= 40))[m].sum(); X += d['diagnosed'][m].sum()
            t1[f'{t} | {mask_name}'] = dict(n=int(N), death=D / N * 1e5, dx=X / N * 1e5)
    out['table1_engine'] = t1
    # model predictions for Table 1 (exact)
    from .validate import model_predictions
    pred, _ = model_predictions(os.path.join(RES, 'kernels_c6b.npz'))
    out['table1_model'] = {k: dict(death=v['death'] * 1e5, dx=v['inc'] * 1e5, colos=v['colos']) for k, v in pred.items()}
    # ---- derived percentages used in the prose
    q10, q5, a10, a5 = t4['q10y'], t4['q5y'], t4['dp_death_lam0.001561_q10y'], t4['dp_death_lam0.00069_q5y']
    bf10, bf5 = t4['bestfixed_q10y_54_64_74'], t4['bestfixed_q5y_48_54_60_66_72_78']
    eff = lambda r: r['deaths_averted_per_1000_colos']; effx = lambda r: r['cases_averted_per_1000_colos']
    out['prose'] = dict(
        colos_fewer_vs_q10y_pct=100 * (1 - a10['colos_per_person'] / q10['colos_per_person']),
        colos_fewer_vs_q5y_pct=100 * (1 - a5['colos_per_person'] / q5['colos_per_person']),
        colos_fewer_vs_bf10_pct=100 * (1 - a10['colos_per_person'] / bf10['colos_per_person']),
        colos_fewer_vs_bf5_pct=100 * (1 - a5['colos_per_person'] / bf5['colos_per_person']),
        mort_lower_vs_q10y_pct=100 * (1 - a10['crc_death_per_100k'] / q10['crc_death_per_100k']),
        inc_lower_vs_q10y_pct=100 * (1 - a10['incidence_per_100k'] / q10['incidence_per_100k']),
        mort_lower_vs_q5y_pct=100 * (1 - a5['crc_death_per_100k'] / q5['crc_death_per_100k']),
        inc_lower_vs_q5y_pct=100 * (1 - a5['incidence_per_100k'] / q5['incidence_per_100k']),
        eff_gain_vs_q10y_pct=100 * (eff(a10) / eff(q10) - 1), effx_gain_vs_q10y_pct=100 * (effx(a10) / effx(q10) - 1),
        eff_gain_vs_q5y_pct=100 * (eff(a5) / eff(q5) - 1), effx_gain_vs_q5y_pct=100 * (effx(a5) / effx(q5) - 1),
        bestfixed_share_of_adaptive_gain_pct=100 * (q10['crc_death_per_100k'] - bf10['crc_death_per_100k']) / (q10['crc_death_per_100k'] - a10['crc_death_per_100k']),
        model_error_q5y_death_pct=100 * (out['table1_model']['q5y']['death'] / t1['q5y | snapshot_q2_age40']['death'] - 1),
        model_error_q10y_death_pct=100 * (out['table1_model']['q10y']['death'] / t1['q10y | snapshot_q2_age40']['death'] - 1),
        model_error_none_death_pct=100 * (out['table1_model']['none']['death'] / t1['none | snapshot_q2_age40']['death'] - 1),
    )
    with open(os.path.join(RES, 'paired_tables.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    # ---- markdown
    L = [f'# Engine tables recomputed from results/dp/runs ({n_chunks} chunks of 50 000 per arm)', '',
         '## Table 4', '', '| arm | colos | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000 colos | dx averted /1000 colos | comp deaths /100k | LY from 40 |', '|---|---|---|---|---|---|---|---|']
    for t, r in t4.items():
        L.append(f"| {t} | {r['colos_per_person']:.3f} | {r['crc_death_per_100k']:.1f} ({r['crc_death_se']:.1f}) | {r['incidence_per_100k']:.1f} ({r['incidence_se']:.1f}) | "
                 f"{r.get('deaths_averted_per_1000_colos', float('nan')):.2f} | {r.get('cases_averted_per_1000_colos', float('nan')):.2f} | {r['comp_death_per_100k']:.1f} | {r['life_years_from40']:.3f} |")
    L += ['', '## Table 5 (paired; chunk SE, t(19) 95 % CI, person-level SE, Holm-adjusted p for the death contrasts)', '',
          '| contrast | d colos | d deaths: mean +- chunk SE [95 % CI] (person SE) | |d|/SE | p (Holm) | d dx: mean +- SE (person SE) | d comp deaths | d LY /1000 |',
          '|---|---|---|---|---|---|---|---|']
    for k, v in t5.items():
        if k == 'extra':
            continue
        d, x, c, ly = v['death'], v['dx'], v['comp'], v['ly']
        L.append(f"| {k} | {v['colos']['mean']:+.3f} | {d['mean']:+.1f} +- {d['se_chunk']:.1f} [{d['ci95_t'][0]:+.1f}, {d['ci95_t'][1]:+.1f}] ({d['se_person']:.1f}) | "
                 f"{d['ratio_to_se']:.1f} | {d['p_two_sided']:.1e} ({d['p_holm']:.1e}) | {x['mean']:+.1f} +- {x['se_chunk']:.1f} ({x['se_person']:.1f}) | "
                 f"{c['mean']:+.1f} +- {c['se_chunk']:.1f} | {ly['mean']:+.1f} +- {ly['se_chunk']:.1f} |")
    for k, v in t5.get('extra', {}).items():
        L.append(f"| {k} | | | | | | {v['comp']['mean']:+.1f} +- {v['comp']['se_chunk']:.1f} | {v['ly']['mean']:+.1f} +- {v['ly']['se_chunk']:.1f} |")
    L += ['', '## Table 6 (by true risk class; colos / deaths per 100k / dx per 100k)', '', '| arm | low | mid | high |', '|---|---|---|---|']
    for t, r in t6.items():
        L.append(f"| {t} | " + ' | '.join(f"{r[g]['colos']:.2f} / {r[g]['death']:.0f} / {r[g]['dx']:.0f}" for g in ('low', 'mid', 'high')) + ' |')
    L += ['', '## Table 1 population definitions (engine)', '', '| arm | mask | n | CRC deaths /100k | CRC dx /100k |', '|---|---|---|---|---|']
    for k, r in t1.items():
        L.append(f"| {k.split(' | ')[0]} | {k.split(' | ')[1]} | {r['n']:,} | {r['death']:.1f} | {r['dx']:.1f} |")
    L += ['', 'Model (exact): ' + ', '.join(f"{k}: {v['death']:.1f} / {v['dx']:.1f}" for k, v in out['table1_model'].items()), '',
          '## Prose quantities', ''] + [f'- {k}: {v:.2f}' for k, v in out['prose'].items()]
    md = os.path.join(RES, 'paired_tables.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', md)


if __name__ == '__main__':
    main()
