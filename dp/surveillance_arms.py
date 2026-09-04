"""Surveillance-augmented fixed comparators.

Every engine arm of the paper runs with CMOST's built-in polyp and cancer
surveillance switched off (dp.engine_runner.run_chunk), so the fixed
comparators of Tables 4-5 are screening-only programmes. This script adds
fixed programmes that include CMOST13's own post-polypectomy surveillance
rule (dp.hooks.FixedSurveillanceHook re-implements the engine's
`Polyp_Surveillance` block on the hook side, so that surveillance
colonoscopies are single annual decisions and are counted as programme
colonoscopies), evaluated on the same paired chunk seeds as the headline arms:

  q10y_surv : screening colonoscopy whenever 50 <= age <= 70 and >= 10 years
              since the last colonoscopy (CMOST's rolling screening
              semantics; 50/60/70 for a person without findings) + surveillance
  q5y_surv  : the same with 50 <= age <= 75 and a 5-year interval + surveillance

python -m dp.surveillance_arms --run --n 1000000 --workers 4
python -m dp.surveillance_arms --analyse --n 1000000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES
from .engine_runner import run_arm, aggregate, efficiency, arm_dir, load_chunk

ARMS = {
    'q10y_surv': dict(kind='fixed_surv', mode='rolling', start=50, end=70, interval=10),
    'q5y_surv': dict(kind='fixed_surv', mode='rolling', start=50, end=75, interval=5),
    # adaptive policy solved (cap 1500) at a price whose engine volume sits just
    # below the 10-yearly + surveillance programme's, for a paired matched-volume contrast
    'dp_death_lam0.001189_surv': dict(kind='policy',
                                      policy_male=os.path.join(RES, 'policies', 'c6bhi2_death_lam0.001189_sex1.npz'),
                                      policy_female=os.path.join(RES, 'policies', 'c6bhi2_death_lam0.001189_sex2.npz'),
                                      observed_class=False),
}
SURV = ['q10y_surv', 'q5y_surv']
MATCHED = ['dp_death_lam0.001189_surv']
REF = ['none', 'q10y', 'q5y', 'bestfixed_q10y_54_64_74', 'bestfixed_q5y_48_54_60_66_72_78',
       'dp_death_lam0.001561_q10y', 'dp_death_lam0.00069_q5y', 'dp_death_lam0.000525_q5y',
       'dp_inc_lam0.005125_q10y', 'dp_inc_lam0.002264_q5y', 'dp_inc_lam0.001724_q5y',
       'rule_52_10_5_3_3', 'dp_death_lam0.001561_obsclass']


def chunk_paths(tag, n_chunks=None):
    p = sorted(glob.glob(os.path.join(arm_dir(tag), 'seed*_n50000.npz')))
    return p[:n_chunks] if n_chunks else p


def per_chunk(d):
    dy = d['death_year'].astype(int)
    n = len(dy)
    return dict(seed=d['summary']['seed'], n=n,
                death=(d['crc_death'] & (dy >= 40)).sum() / n * 1e5,
                comp=(d['comp_death'] & (dy >= 40)).sum() / n * 1e5,
                dx=d['diagnosed'].sum() / n * 1e5,
                colos=d['n_policy_colo'].sum() / n,
                ly=np.maximum(dy - 1 - 40, 0).sum() / n * 1e3)


def paired(rows_a, rows_b, key):
    A = {r['seed']: r[key] for r in rows_a}; B = {r['seed']: r[key] for r in rows_b}
    seeds = sorted(set(A) & set(B))
    d = np.array([A[s] - B[s] for s in seeds])
    return dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan'),
                n_chunks=len(d))


def analyse(n_total, out_path):
    n_chunks = n_total // 50_000
    rows = {t: [per_chunk(load_chunk(p)) for p in chunk_paths(t, n_chunks)] for t in list(ARMS) + REF}
    rows = {t: r for t, r in rows.items() if r}
    agg = {}
    base = aggregate(chunk_paths('none', n_chunks))
    for t in rows:
        r = aggregate(chunk_paths(t, n_chunks))
        if t != 'none':
            r.update(efficiency(r, base))
        agg[t] = r
    # surveillance share
    surv = {}
    for t in SURV:
        if t not in rows:
            continue
        ns = nsurv = 0
        for p in chunk_paths(t, n_chunks):
            s = load_chunk(p)['summary']
            hc = s.get('hook_counters', {})
            ns += hc.get('n_screening_colos', 0); nsurv += hc.get('n_surveillance_colos', 0)
        surv[t] = dict(screening_colos_per_person=ns / agg[t]['n'], surveillance_colos_per_person=nsurv / agg[t]['n'],
                       surveillance_share=nsurv / max(ns + nsurv, 1))
    # paired contrasts
    contrasts = {}
    for t in SURV:
        if t not in rows:
            continue
        for ref in REF + MATCHED:
            if ref not in rows or ref == t:
                continue
            contrasts[f'{t} - {ref}'] = {k: paired(rows[t], rows[ref], k) for k in ('colos', 'death', 'dx', 'comp', 'ly')}
    # adaptive engine frontier (200k grid, death objective) interpolated at each surv arm's volume
    grid = json.load(open(os.path.join(RES, 'eval_grid_c6b_death_n200000.json')))
    pts = sorted((r['colos_per_person'], r['crc_death_per_100k'], r['incidence_per_100k'])
                 for k, r in grid.items() if k.startswith('c6b_death_lam') and r['colos_per_person'] > 0)
    c = np.array([p[0] for p in pts]); dth = np.array([p[1] for p in pts]); inc = np.array([p[2] for p in pts])
    matched = {}
    for t in SURV:
        if t not in agg:
            continue
        v = agg[t]['colos_per_person']
        matched[t] = dict(volume=v, adaptive_death_interp_200k=float(np.interp(v, c, dth)),
                          adaptive_inc_interp_200k=float(np.interp(v, c, inc)))
    out = dict(n=n_total, arms=ARMS, aggregate=agg, surveillance=surv, contrasts=contrasts, matched=matched)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=1)
    # markdown
    L = [f'# Surveillance-augmented fixed comparators (engine, n = {n_total:,} per arm, paired seeds)', '',
         '| arm | colos/person | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000 colos | dx averted /1000 colos | comp deaths /100k |',
         '|---|---|---|---|---|---|---|']
    for t, r in agg.items():
        L.append(f"| {t} | {r['colos_per_person']:.3f} | {r['crc_death_per_100k']:.1f} ({r['crc_death_se']:.1f}) | "
                 f"{r['incidence_per_100k']:.1f} ({r['incidence_se']:.1f}) | {r.get('deaths_averted_per_1000_colos', float('nan')):.2f} | "
                 f"{r.get('cases_averted_per_1000_colos', float('nan')):.2f} | {r['comp_death_per_100k']:.1f} |")
    L += ['', '## Surveillance share', '']
    for t, s in surv.items():
        L.append(f"- {t}: screening {s['screening_colos_per_person']:.3f} + surveillance {s['surveillance_colos_per_person']:.3f} "
                 f"colonoscopies per person (surveillance share {100 * s['surveillance_share']:.1f} %)")
    L += ['', '## Paired contrasts (arm - reference; mean +- SE over chunk pairs)', '',
          '| contrast | d colos | d CRC deaths /100k | d CRC dx /100k | d comp deaths /100k | d LY /1000 |', '|---|---|---|---|---|---|']
    for k, v in contrasts.items():
        L.append(f"| {k} | {v['colos']['mean']:+.3f} | {v['death']['mean']:+.1f} +- {v['death']['se']:.1f} | "
                 f"{v['dx']['mean']:+.1f} +- {v['dx']['se']:.1f} | {v['comp']['mean']:+.1f} +- {v['comp']['se']:.1f} | "
                 f"{v['ly']['mean']:+.1f} +- {v['ly']['se']:.1f} |")
    L += ['', '## Adaptive engine frontier (200k grid) interpolated at the surveillance arms\' volumes', '']
    for t, mch in matched.items():
        L.append(f"- {t}: {mch['volume']:.3f} colos -> adaptive {mch['adaptive_death_interp_200k']:.0f} deaths, "
                 f"{mch['adaptive_inc_interp_200k']:.0f} dx per 100k (vs {agg[t]['crc_death_per_100k']:.0f} / {agg[t]['incidence_per_100k']:.0f})")
    md = out_path.replace('.json', '.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print('saved', out_path, md)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--n', type=int, default=1_000_000)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.run:
        for tag in a.arms.split(','):
            print(f'== arm {tag}: {ARMS[tag]}', flush=True)
            run_arm(ARMS[tag], tag, a.n, chunk=50_000, workers=a.workers)
    if a.analyse or not a.run:
        analyse(a.n, os.path.join(RES, f'eval_surveillance_n{a.n}.json'))


if __name__ == '__main__':
    main()
