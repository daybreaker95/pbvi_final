"""Robustness check: is PBS+DBBU(theta=10) RELIABLY better than PBS alone,
or did the single seed=0 run just get lucky? Trains both configs across
multiple FiVI solver seeds (0..9), same wtp=$50k/max_iters=39 otherwise,
and for each reports final gap + MC clinical/cost outcome (N=200,000 for
speed across 20 runs) -- so we can see whether pbs_dbbu10 consistently
wins, consistently loses, or is just noisy relative to plain pbs."""
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
MC_N = 200_000
SEEDS = list(range(10))


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_with_cost(pomdp, pomdp_pure, solver, N, seed):
    rng = np.random.default_rng((seed, 0))
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    true_s = np.array([pomdp.fidx(r, NORMAL) for r in risk])
    alive = np.ones(N, dtype=bool)
    death_cause = np.zeros(N, dtype=int)
    dx_stage = np.zeros(N, dtype=int)
    b0 = pomdp.initial_belief()
    belief = np.tile(b0, (N, 1))
    det_local = {s: k + 1 for k, s in enumerate(DET_CA_STAGES)}
    cum_qaly = np.zeros(N)
    cum_pure = np.zeros(N)

    for age in range(pomdp.age_min, pomdp.life_max + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if age <= pomdp.age_max:
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
                    death_cause[grp[became_crc]] = 2
                    alive[grp[became_crc]] = False
                if became_oth.any():
                    death_cause[grp[became_oth]] = 1
                    alive[grp[became_oth]] = False
                newly_det = np.isin(local_nxt, list(det_local.keys())) & (dx_stage[grp] == 0)
                if newly_det.any():
                    g2 = grp[newly_det]
                    stg = np.array([det_local[v] for v in local_nxt[newly_det]])
                    dx_stage[g2] = stg
                true_s[grp] = nxt
    cost = WTP * (cum_pure - cum_qaly)
    return {
        'crc_100k': float(100_000 * (death_cause == 2).mean()),
        'incid_100k': float(100_000 * (dx_stage > 0).mean()),
        'net_benefit': float(cum_qaly.mean()),
        'cost_per_person_usd': float(cost.mean()),
    }


def run_one(fivi_seed, use_pbs, dbbu_interval):
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP)
    pomdp_pure = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP_HUGE)
    solver = FiVI(pomdp, seed=fivi_seed, max_sawtooth_points=300)
    solver.solve(max_iters=MAX_ITERS, time_limit=None, precision=1e-9, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=use_pbs, dbbu_interval=dbbu_interval)
    gap = solver.gap_history[-1]['gap']
    vl = solver.gap_history[-1]['vl']
    stats = simulate_with_cost(pomdp, pomdp_pure, solver, MC_N, seed=42)
    stats.update({'gap': gap, 'vl': vl})
    return stats


def main():
    results = {'pbs': [], 'pbs_dbbu10': []}
    for fivi_seed in SEEDS:
        print(f'--- fivi_seed={fivi_seed} ---', flush=True)
        s_pbs = run_one(fivi_seed, True, 1)
        s_dbbu = run_one(fivi_seed, True, 10)
        results['pbs'].append({'seed': fivi_seed, **s_pbs})
        results['pbs_dbbu10'].append({'seed': fivi_seed, **s_dbbu})
        d_crc = s_dbbu['crc_100k'] - s_pbs['crc_100k']
        d_cost = s_dbbu['cost_per_person_usd'] - s_pbs['cost_per_person_usd']
        d_nb = s_dbbu['net_benefit'] - s_pbs['net_benefit']
        winner = 'dbbu10 lower mortality' if d_crc < 0 else ('pbs lower mortality' if d_crc > 0 else 'tie')
        print(f'  pbs:        crc_100k={s_pbs["crc_100k"]:.1f} cost=${s_pbs["cost_per_person_usd"]:,.0f} '
              f'net_benefit={s_pbs["net_benefit"]:.4f} gap={s_pbs["gap"]:.4f}', flush=True)
        print(f'  pbs_dbbu10: crc_100k={s_dbbu["crc_100k"]:.1f} cost=${s_dbbu["cost_per_person_usd"]:,.0f} '
              f'net_benefit={s_dbbu["net_benefit"]:.4f} gap={s_dbbu["gap"]:.4f}', flush=True)
        print(f'  -> Δcrc_100k={d_crc:+.1f}  Δcost=${d_cost:+,.0f}  Δnet_benefit={d_nb:+.5f}  ({winner})', flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pbs_dbbu_robustness.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved', out_path)

    print('\n=== summary across', len(SEEDS), 'seeds ===')
    crc_pbs = [r['crc_100k'] for r in results['pbs']]
    crc_dbbu = [r['crc_100k'] for r in results['pbs_dbbu10']]
    nb_pbs = [r['net_benefit'] for r in results['pbs']]
    nb_dbbu = [r['net_benefit'] for r in results['pbs_dbbu10']]
    cost_pbs = [r['cost_per_person_usd'] for r in results['pbs']]
    cost_dbbu = [r['cost_per_person_usd'] for r in results['pbs_dbbu10']]
    n_dbbu_lower_mortality = sum(1 for a, b in zip(crc_dbbu, crc_pbs) if a < b)
    n_dbbu_higher_net_benefit = sum(1 for a, b in zip(nb_dbbu, nb_pbs) if a > b)
    print(f'pbs        crc_100k: mean={np.mean(crc_pbs):.1f} std={np.std(crc_pbs):.1f} min={min(crc_pbs):.1f} max={max(crc_pbs):.1f}')
    print(f'pbs_dbbu10 crc_100k: mean={np.mean(crc_dbbu):.1f} std={np.std(crc_dbbu):.1f} min={min(crc_dbbu):.1f} max={max(crc_dbbu):.1f}')
    print(f'pbs        net_benefit: mean={np.mean(nb_pbs):.4f} std={np.std(nb_pbs):.4f}')
    print(f'pbs_dbbu10 net_benefit: mean={np.mean(nb_dbbu):.4f} std={np.std(nb_dbbu):.4f}')
    print(f'pbs        cost: mean=${np.mean(cost_pbs):,.0f} std=${np.std(cost_pbs):,.0f}')
    print(f'pbs_dbbu10 cost: mean=${np.mean(cost_dbbu):,.0f} std=${np.std(cost_dbbu):,.0f}')
    print(f'dbbu10 had LOWER mortality in {n_dbbu_lower_mortality}/{len(SEEDS)} seeds')
    print(f'dbbu10 had HIGHER net_benefit in {n_dbbu_higher_net_benefit}/{len(SEEDS)} seeds')


if __name__ == '__main__':
    main()
