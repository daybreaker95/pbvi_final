"""Model-vs-engine validation: exact in-model predictions for no-screening /
q10y / q5y against paired real-engine runs (and decision-time occupancy
trajectories when quarterly recordings are available).

python -m dp.validate --kernels results/dp/kernels_c3.npz --runs-none nh_quarterly --runs-q10y q10y_dbg
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES, RUNS, NC, MAP18_TO_CLIN, WAIT, SCREEN
from .model import ReducedPOMDP, evaluate_fixed
from .engine_runner import aggregate

SCHEDULES = {'none': [], 'q10y': [50, 60, 70], 'q5y': [50, 55, 60, 65, 70, 75]}


def model_predictions(kernels, frac_female=0.5):
    models = {s: ReducedPOMDP(s, kernels) for s in (1, 2)}
    out = {}
    for name, ages in SCHEDULES.items():
        r1 = evaluate_fixed(models[1], ages); r2 = evaluate_fixed(models[2], ages)
        out[name] = {k: (1 - frac_female) * r1[k] + frac_female * r2[k] for k in r1}
    return out, models


def engine_summary(run_dir):
    paths = sorted(glob.glob(os.path.join(RUNS, run_dir, '*.npz')))
    return aggregate(paths) if paths else None


def occupancy_engine(run_dir, ages=range(40, 100)):
    """Decision-time (start of Q2) clinical occupancy among persons alive &
    undiagnosed at the age-40 decision, by sex: dict sex -> (len(ages), NC+1)."""
    occ = {1: np.zeros((len(ages), NC + 1)), 2: np.zeros((len(ages), NC + 1))}
    for p in sorted(glob.glob(os.path.join(RUNS, run_dir, '*.npz'))):
        d = np.load(p, allow_pickle=True)
        if 'qr' not in d.files:
            continue
        qr = d['qr']; sex = d['sex']
        ok = MAP18_TO_CLIN[qr[157]] >= 0
        for sx in (1, 2):
            m = ok & (sex == sx)
            for i, y in enumerate(ages):
                s = MAP18_TO_CLIN[qr[(y - 1) * 4 + 1][m]]
                occ[sx][i, :NC] += np.bincount(s[s >= 0], minlength=NC)
                occ[sx][i, NC] += (s < 0).sum()
    for sx in occ:
        tot = occ[sx].sum(axis=1, keepdims=True)
        occ[sx] = np.where(tot > 0, occ[sx] / np.maximum(tot, 1), np.nan)
    return occ


def occupancy_model(model, schedule, ages=range(40, 100)):
    """Model decision-time occupancy (fraction of the age-40 cohort) per age,
    clinical marginal + exited mass, for a fixed schedule."""
    s = set(schedule)
    nodes = {model.initial_memory(): model.initial_belief().copy()}
    out = np.zeros((len(ages), NC + 1))
    for i, y in enumerate(ages):
        U = sum(nodes.values())
        out[i, :NC] = model.clinical_marginal(U)
        out[i, NC] = 1.0 - U.sum()
        a = SCREEN if (y in s and y <= model.age_max) else WAIT
        nxt = {}
        for (tau, ol), u in nodes.items():
            key = (y, tau, ol)
            for o, M in model.M[key][a].items():
                nk = model.succ[key][a][o]
                nxt[nk] = nxt.get(nk, 0) + u @ M
        nodes = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernels', required=True)
    ap.add_argument('--runs-none', default='nh_quarterly')
    ap.add_argument('--runs-q10y', default=None)
    ap.add_argument('--runs-q5y', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    pred, models = model_predictions(a.kernels)
    report = {'model': pred, 'engine': {}}
    for name, rd in (('none', a.runs_none), ('q10y', a.runs_q10y), ('q5y', a.runs_q5y)):
        if rd:
            e = engine_summary(rd)
            if e:
                report['engine'][name] = e
    print(f"{'arm':6s} {'':8s} {'death/100k':>10s} {'inc/100k':>9s} {'colos':>6s} {'LY':>6s}")
    for name in SCHEDULES:
        p = pred[name]
        print(f"{name:6s} {'model':8s} {p['death'] * 1e5:10.0f} {p['inc'] * 1e5:9.0f} {p['colos']:6.3f} {p['ly']:6.2f}")
        e = report['engine'].get(name)
        if e:
            print(f"{'':6s} {'engine':8s} {e['crc_death_per_100k']:10.0f} {e['incidence_per_100k']:9.0f} "
                  f"{e['policy_colos_per_person']:6.3f} {e['life_years_from40']:6.2f}   (n={e['n']:,})")
    # occupancy comparison for q10y if quarterly data exists
    for name, rd in (('q10y', a.runs_q10y), ('q5y', a.runs_q5y), ('none', a.runs_none)):
        if not rd:
            continue
        occ_e = occupancy_engine(rd)
        if np.isnan(occ_e[1]).all():
            continue
        print(f'\n--- occupancy {name} (engine | model), % of age-40 cohort: N, P1-4, P5-6, U, exited')
        for sx in (1, 2):
            occ_m = occupancy_model(models[sx], SCHEDULES[name])
            for i, y in enumerate(range(40, 100)):
                if y in (41, 45, 49, 51, 53, 55, 58, 59, 61, 63, 65, 69, 71, 75, 79, 85, 90):
                    e = occ_e[sx][i] * 100; m = occ_m[i] * 100
                    f = lambda v: f'N {v[0]:5.1f} P1-4 {v[1:5].sum():5.1f} P5-6 {v[5:7].sum():5.2f} U {v[7:11].sum():5.2f} ex {v[11]:5.1f}'
                    print(f'sex{sx} y={y}: {f(e)} | {f(m)}')
            report.setdefault('occupancy', {})[f'{name}_sex{sx}'] = dict(engine=occ_e[sx].tolist(), model=occ_m.tolist())
    out = a.out or os.path.join(RES, 'validation.json')
    with open(out, 'w') as f:
        json.dump(report, f)
    print('saved', out)


if __name__ == '__main__':
    main()
