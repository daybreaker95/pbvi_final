"""WTP sensitivity check: retrain FiVI (this time WITH PBS -- use_pbs=True,
n_stochastic_trajectories=5 for better belief coverage/lower-bound quality;
DBBU still left exact every iter, dbbu_interval=1, since the frozen-v_u
issue found in the main run is structural at t=1 and DBBU's caching
shortcut doesn't touch that) at wtp in {50000 (baseline), 60000, 100000},
extract age-wise SCREEN% and first-screening-age distribution for each, so
we can see how sensitive the learned schedule is to the WTP preference
parameter."""
import os
import sys
import json
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI, WAIT, SCREEN, NO


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_with_schedule(pomdp, solver, N, seed):
    rng = np.random.default_rng((seed, 0))
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    true_s = np.array([pomdp.fidx(r, NORMAL) for r in risk])
    alive = np.ones(N, dtype=bool)
    b0 = pomdp.initial_belief()
    belief = np.tile(b0, (N, 1))
    ages = list(range(pomdp.age_min, pomdp.life_max + 1))
    action_log = {a: np.full(N, -1, dtype=np.int8) for a in ages}
    for age in ages:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if age <= pomdp.age_max:
            actions = fivi_best_action_batch(solver, age, belief[idx])
        else:
            actions = np.zeros(len(idx), dtype=int)
        action_log[age][idx] = actions
        M = pomdp.M[min(age, pomdp.life_max)]
        u_age = np.random.default_rng((seed, age)).random(N)
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
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
                dead = (local_nxt == CRC_DEATH) | (local_nxt == OTHER_DEATH)
                if dead.any():
                    alive[grp[dead]] = False
                true_s[grp] = nxt
    return risk, action_log


def simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed):
    """Sex-aware counterpart of simulate_with_schedule: each individual is
    routed to their own sex-specific (pomdp, solver) pair. fraction_female
    =0.5 matches CMOST's own settings value."""
    rng = np.random.default_rng((seed, 0))
    NS, NC = pomdp_m.NS, pomdp_m.NC
    is_male = rng.random(N) >= 0.5
    risk = np.empty(N, dtype=int)
    risk[is_male] = (rng.random(int(is_male.sum())) < pomdp_m.frac_high).astype(int)
    risk[~is_male] = (rng.random(int((~is_male).sum())) < pomdp_f.frac_high).astype(int)
    true_s = np.empty(N, dtype=int)
    true_s[is_male] = np.array([pomdp_m.fidx(r, NORMAL) for r in risk[is_male]])
    true_s[~is_male] = np.array([pomdp_f.fidx(r, NORMAL) for r in risk[~is_male]])
    alive = np.ones(N, dtype=bool)
    b0_m, b0_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()
    belief = np.where(is_male[:, None], b0_m[None, :], b0_f[None, :])
    ages = list(range(pomdp_m.age_min, pomdp_m.life_max + 1))
    action_log = {a: np.full(N, -1, dtype=np.int8) for a in ages}
    for age in ages:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        actions = np.zeros(len(idx), dtype=int)
        if age <= pomdp_m.age_max:
            m_sub = is_male[idx]
            if m_sub.any():
                actions[m_sub] = fivi_best_action_batch(solver_m, age, belief[idx[m_sub]])
            if (~m_sub).any():
                actions[~m_sub] = fivi_best_action_batch(solver_f, age, belief[idx[~m_sub]])
        action_log[age][idx] = actions
        u_age = np.random.default_rng((seed, age)).random(N)
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            for male_flag, pomdp in ((True, pomdp_m), (False, pomdp_f)):
                sx_mask = is_male[sub] if male_flag else ~is_male[sub]
                sub2 = sub[sx_mask]
                if len(sub2) == 0:
                    continue
                M = pomdp.M[min(age, pomdp.life_max)]
                s_snap = true_s[sub2].copy()
                for s in np.unique(s_snap):
                    grp = sub2[s_snap == s]
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
                    local_nxt = nxt % NC
                    dead = (local_nxt == CRC_DEATH) | (local_nxt == OTHER_DEATH)
                    if dead.any():
                        alive[grp[dead]] = False
                    true_s[grp] = nxt
    return risk, action_log


def run_for_wtp(wtp, N=500_000, seed=42):
    pomdp_m = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, wtp=wtp, sex=1)
    pomdp_f = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, wtp=wtp, sex=2)
    solver_m = FiVI(pomdp_m, seed=0, max_sawtooth_points=300)
    solver_f = FiVI(pomdp_f, seed=0, max_sawtooth_points=300)
    solver_m.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                    n_stochastic_trajectories=5, use_pbs=True, dbbu_interval=1)
    solver_f.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                    n_stochastic_trajectories=5, use_pbs=True, dbbu_interval=1)
    gap = 0.5 * solver_m.gap_history[-1]['gap'] + 0.5 * solver_f.gap_history[-1]['gap']
    risk, action_log = simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed)
    ages = sorted(action_log.keys())
    age_stats = []
    for age in ages:
        a = action_log[age]
        seen = a >= 0
        if seen.sum() == 0:
            continue
        screen_pct = float((a[seen] == SCREEN).mean() * 100)
        if screen_pct > 1.0:
            age_stats.append({'age': age, 'screen_pct_all': screen_pct})
    # first age where population-wide SCREEN% crosses 50%
    first_major = next((r['age'] for r in age_stats if r['screen_pct_all'] >= 50), None)
    return {'wtp': wtp, 'gap': gap, 'gap_male': solver_m.gap_history[-1]['gap'],
            'gap_female': solver_f.gap_history[-1]['gap'],
            'age_stats': age_stats, 'first_major_screen_age': first_major}


def main():
    results = {}
    for wtp in (50_000, 75_000, 100_000, 125_000, 150_000):
        print(f'=== WTP=${wtp:,} ===', flush=True)
        r = run_for_wtp(wtp)
        results[str(wtp)] = r
        print(f'  gap={r["gap"]:.4f}  first_major_screen_age={r["first_major_screen_age"]}', flush=True)
        for a in r['age_stats']:
            print(f"    age={a['age']}: {a['screen_pct_all']:.1f}%", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'wtp_sensitivity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', out_path)


if __name__ == '__main__':
    main()
