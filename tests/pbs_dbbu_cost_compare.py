"""Backs out raw ($) cost for the 3 solver configs without touching
model_v2.py's reward code: reward = u - cost/wtp, so running the SAME
trajectory (same true states/actions, same random substrate) through a
second POMDP instance built with wtp effectively infinite gives
reward_pure ~= u (cost term vanishes). Then:
    cost = wtp_real * (reward_pure - reward_blended)
accumulated the same (undiscounted) way cum_qaly is elsewhere in this
project. M (transition/observation dynamics) does not depend on wtp at
all -- only R does -- so it's valid to reuse the SAME action sequence
(decided by the wtp=$50k-trained policy) against both R arrays."""
import os
import sys
import json
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI, WAIT, SCREEN, NO

MAX_ITERS = 39
WTP = 50_000
WTP_HUGE = 1e12
MC_N = 500_000


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_with_cost(pomdp, pomdp_pure, solver, N, seed, policy=True):
    """pomdp: wtp=$50k (drives action choice + belief updates, M identical
    to pomdp_pure). pomdp_pure: wtp~inf, same M, R~=pure utility u."""
    rng = np.random.default_rng((seed, 0))
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    true_s = np.array([pomdp.fidx(r, NORMAL) for r in risk])
    alive = np.ones(N, dtype=bool)
    death_age = np.full(N, float(pomdp.life_max))
    death_cause = np.zeros(N, dtype=int)
    dx_stage = np.zeros(N, dtype=int)
    b0 = pomdp.initial_belief()
    belief = np.tile(b0, (N, 1))
    det_local = {s: k + 1 for k, s in enumerate(DET_CA_STAGES)}
    cum_qaly = np.zeros(N)       # blended (u - cost/wtp), undiscounted sum
    cum_pure = np.zeros(N)       # ~pure utility u, undiscounted sum

    for age in range(pomdp.age_min, pomdp.life_max + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if policy and age <= pomdp.age_max:
            actions = fivi_best_action_batch(solver, age, belief[idx])
        else:
            actions = np.zeros(len(idx), dtype=int)
        M = pomdp.M[min(age, pomdp.life_max)]
        R = pomdp.R[min(age, pomdp.life_max)]
        R_pure = pomdp_pure.R[min(age, pomdp.life_max)]
        u_age = np.random.default_rng((seed, age)).random(N)
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            cum_qaly[sub] += R[a][true_s[sub]]
            cum_pure[sub] += R_pure[a][true_s[sub]]
            s_snap = true_s[sub].copy()
            for s in np.unique(s_snap):
                grp = sub[s_snap == s]
                row = np.concatenate([M[a][o][s, :] for o in range(NO)])
                row = row / row.sum()
                cdf = np.cumsum(row)
                u = u_age[grp]
                flat = np.searchsorted(cdf, u, side='right')
                obs = flat // NS
                nxt = flat % NS
                for o in range(NO):
                    m = grp[obs == o]
                    if len(m) == 0:
                        continue
                    bn = belief[m] @ M[a][o]
                    tot = bn.sum(axis=1, keepdims=True)
                    ok = tot[:, 0] > 1e-300
                    belief[m[ok]] = bn[ok] / tot[ok]
                local_nxt = nxt % pomdp.NC
                became_crc = local_nxt == CRC_DEATH
                became_oth = local_nxt == OTHER_DEATH
                if became_crc.any():
                    g2 = grp[became_crc]
                    alive[g2] = False
                    death_age[g2] = age + 0.5
                    death_cause[g2] = 2
                if became_oth.any():
                    g2 = grp[became_oth]
                    alive[g2] = False
                    death_age[g2] = age + 0.5
                    death_cause[g2] = 1
                newly_det = np.isin(local_nxt, list(det_local.keys())) & (dx_stage[grp] == 0)
                if newly_det.any():
                    g2 = grp[newly_det]
                    stg = np.array([det_local[v] for v in local_nxt[newly_det]])
                    dx_stage[g2] = stg
                true_s[grp] = nxt
    death_age[alive] = pomdp.life_max
    cost = WTP * (cum_pure - cum_qaly)   # $ per person, undiscounted total
    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage,
                cum_qaly=cum_qaly, cost=cost)


def summarize(r):
    return {
        'crc_100k': float(100_000 * (r['death_cause'] == 2).mean()),
        'incid_100k': float(100_000 * (r['dx_stage'] > 0).mean()),
        'cum_qaly': float(r['cum_qaly'].mean()),
        'cost_per_person_usd': float(r['cost'].mean()),
    }


def run_config(use_pbs, dbbu_interval):
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP)
    pomdp_pure = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP_HUGE)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=MAX_ITERS, time_limit=None, precision=1e-9, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=use_pbs, dbbu_interval=dbbu_interval)
    r_pol = simulate_with_cost(pomdp, pomdp_pure, solver, MC_N, seed=42, policy=True)
    return pomdp, pomdp_pure, summarize(r_pol)


def main():
    configs = [
        ('no_pbs', dict(use_pbs=False, dbbu_interval=1)),
        ('pbs', dict(use_pbs=True, dbbu_interval=1)),
        ('pbs_dbbu10', dict(use_pbs=True, dbbu_interval=10)),
    ]
    results = {}
    pomdp = pomdp_pure = None
    for name, kw in configs:
        print(f'=== {name} ===', flush=True)
        pomdp, pomdp_pure, stats = run_config(**kw)
        results[name] = stats
        print(f'  {stats}', flush=True)

    r_nos = simulate_with_cost(pomdp, pomdp_pure, None, MC_N, seed=42, policy=False)
    results['no_screen'] = summarize(r_nos)
    print('no_screen baseline:', results['no_screen'], flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pbs_dbbu_cost_compare.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved', out_path)

    print('\n=== summary ===')
    print(f"{'config':<12}{'cost/person':>14}{'cum_qaly':>10}{'crc_100k':>10}{'incid_100k':>12}")
    for name in ['no_screen', 'no_pbs', 'pbs', 'pbs_dbbu10']:
        s = results[name]
        print(f"{name:<12}{s['cost_per_person_usd']:>14,.2f}{s['cum_qaly']:>10.4f}"
              f"{s['crc_100k']:>10.1f}{s['incid_100k']:>12.1f}")


if __name__ == '__main__':
    main()
