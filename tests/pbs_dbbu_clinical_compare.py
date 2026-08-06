"""Clinical-benefit comparison for the 3 solver configs (no_pbs / pbs /
pbs_dbbu10): retrains each (same wtp=$50k, seed, 39 iters) and runs an
N=500,000 MC clinical evaluation (CRN, same convention as
gather_fivi_full_trajectory.py's simulate_fivi) to get reward/cum_qaly/
life_years/crc_100k/incid_100k -- answers "does DBBU's different schedule
actually perform worse, or just differently?" """
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
MC_N = 500_000


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_fivi(pomdp, solver, N, seed, policy=True):
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
    disc_reward = np.zeros(N)
    cum_qaly = np.zeros(N)

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
        disc = pomdp.gamma ** (age - pomdp.age_min)
        u_age = np.random.default_rng((seed, age)).random(N)
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            disc_reward[sub] += disc * R[a][true_s[sub]]
            cum_qaly[sub] += R[a][true_s[sub]]
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
    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage,
                disc_reward=disc_reward, cum_qaly=cum_qaly)


def summarize(r, N):
    crc_100k = 100_000 * (r['death_cause'] == 2).mean()
    incid_100k = 100_000 * (r['dx_stage'] > 0).mean()
    life_years = float((r['death_age'] - 40).mean())
    return {
        'crc_100k': float(crc_100k), 'incid_100k': float(incid_100k),
        'reward': float(r['disc_reward'].mean()), 'cum_qaly': float(r['cum_qaly'].mean()),
        'life_years': life_years,
    }


def run_config(use_pbs, dbbu_interval):
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=MAX_ITERS, time_limit=None, precision=1e-9, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=use_pbs, dbbu_interval=dbbu_interval)
    r_pol = simulate_fivi(pomdp, solver, MC_N, seed=42, policy=True)
    return pomdp, summarize(r_pol, MC_N)


def main():
    configs = [
        ('no_pbs', dict(use_pbs=False, dbbu_interval=1)),
        ('pbs', dict(use_pbs=True, dbbu_interval=1)),
        ('pbs_dbbu10', dict(use_pbs=True, dbbu_interval=10)),
    ]
    results = {}
    pomdp = None
    for name, kw in configs:
        print(f'=== {name} ===', flush=True)
        pomdp, stats = run_config(**kw)
        results[name] = stats
        print(f'  {stats}', flush=True)

    # no-screen baseline (policy=False), same pomdp/model
    r_nos = simulate_fivi(pomdp, None, MC_N, seed=42, policy=False)
    results['no_screen'] = summarize(r_nos, MC_N)
    print('no_screen baseline:', results['no_screen'], flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pbs_dbbu_clinical_compare.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved', out_path)

    print('\n=== summary ===')
    print(f"{'config':<12}{'reward':>10}{'cum_qaly':>10}{'life_yrs':>10}{'crc_100k':>10}{'incid_100k':>12}")
    for name in ['no_screen', 'no_pbs', 'pbs', 'pbs_dbbu10']:
        s = results[name]
        print(f"{name:<12}{s['reward']:>10.4f}{s['cum_qaly']:>10.4f}{s['life_years']:>10.4f}"
              f"{s['crc_100k']:>10.1f}{s['incid_100k']:>12.1f}")


if __name__ == '__main__':
    main()
