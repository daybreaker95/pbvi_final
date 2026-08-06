"""Retrains FiVI to convergence (fast -- pure backup only, no MC eval per
iteration) on the final pure 13-state model, then extracts:
  1. age-wise SCREEN recommendation % (population-weighted, via MC simulation
     with common-random-numbers, same convention as gather_fivi_full_trajectory.py)
  2. a handful of example individual screening schedules (exact ages where
     SCREEN was chosen), for low-risk and high-risk individuals separately.
"""
import os
import sys
import json
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI, WAIT, SCREEN, NO


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed):
    """Sex-aware counterpart of simulate_with_schedule: each individual is
    routed to their own sex-specific (pomdp, solver) pair. Records the
    action taken at every age for every (still-alive) person, so we can
    derive age-wise SCREEN% and pull out individual schedules."""
    rng = np.random.default_rng((seed, 0))
    NS, NC = pomdp_m.NS, pomdp_m.NC
    is_male = rng.random(N) >= 0.5
    risk = np.empty(N, dtype=int)
    risk[is_male] = (rng.random(int(is_male.sum())) < pomdp_m.frac_high).astype(int)
    risk[~is_male] = (rng.random(int((~is_male).sum())) < pomdp_f.frac_high).astype(int)
    true_s = np.empty(N, dtype=int)
    for male_flag, pomdp in ((True, pomdp_m), (False, pomdp_f)):
        sx = is_male if male_flag else ~is_male
        for r in range(pomdp.n_risk):
            mask = sx & (risk == r)
            cnt = int(mask.sum())
            if cnt > 0:
                s_local = rng.choice(pomdp.NC, size=cnt, p=pomdp._burnin_dist[r])
                true_s[mask] = np.array([pomdp.fidx(r, int(s)) for s in s_local])
    alive = np.ones(N, dtype=bool)
    b0_m, b0_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()
    belief = np.where(is_male[:, None], b0_m[None, :], b0_f[None, :])

    ages = list(range(pomdp_m.age_min, pomdp_m.life_max + 1))
    # action_log[age] = int8 array, N long: 0=WAIT,1=SCREEN,-1=dead/not-evaluated
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
    return risk, is_male, action_log


def main():
    pomdp_m = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, sex=1)
    pomdp_f = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, sex=2)
    solver_m = FiVI(pomdp_m, seed=0, max_sawtooth_points=300)
    solver_f = FiVI(pomdp_f, seed=0, max_sawtooth_points=300)
    solver_m.solve(max_iters=100, time_limit=300, precision=1e-3, verbose=True,
                    n_stochastic_trajectories=0, use_pbs=True, dbbu_interval=1)
    solver_f.solve(max_iters=100, time_limit=300, precision=1e-3, verbose=True,
                    n_stochastic_trajectories=0, use_pbs=True, dbbu_interval=1)

    N = 500_000
    risk, is_male, action_log = simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed=42)
    ages = sorted(action_log.keys())

    # --- 1. age-wise SCREEN% (overall, and split by risk class) ---
    age_stats = []
    for age in ages:
        a = action_log[age]
        seen = a >= 0
        if seen.sum() == 0:
            continue
        screen_pct = float((a[seen] == SCREEN).mean() * 100)
        low = seen & (risk == 0)
        high = seen & (risk == 1)
        low_pct = float((a[low] == SCREEN).mean() * 100) if low.sum() else None
        high_pct = float((a[high] == SCREEN).mean() * 100) if high.sum() else None
        age_stats.append({'age': age, 'n_alive': int(seen.sum()),
                           'screen_pct_all': screen_pct,
                           'screen_pct_low': low_pct, 'screen_pct_high': high_pct})

    # --- 2. example individual schedules ---
    def schedule_for(person_idx):
        sched = []
        for age in ages:
            a = action_log[age][person_idx]
            if a == SCREEN:
                sched.append(age)
            elif a == -1:
                break
        return sched

    rng2 = np.random.default_rng(123)
    low_idx = np.where(risk == 0)[0]
    high_idx = np.where(risk == 1)[0]
    examples = {'low_risk': [], 'high_risk': []}
    for i in rng2.choice(low_idx, size=6, replace=False):
        examples['low_risk'].append({'person': int(i), 'sex': 1 if is_male[i] else 2,
                                      'screen_ages': schedule_for(i)})
    for i in rng2.choice(high_idx, size=6, replace=False):
        examples['high_risk'].append({'person': int(i), 'sex': 1 if is_male[i] else 2,
                                       'screen_ages': schedule_for(i)})

    out = {'age_stats': age_stats, 'examples': examples,
           'gap_final_male': solver_m.gap_history[-1] if solver_m.gap_history else None,
           'gap_final_female': solver_f.gap_history[-1] if solver_f.gap_history else None,
           'n_sim': N}
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'policy_schedule_extract.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', out_path)

    print('\n=== age-wise SCREEN% (전체 / low-risk / high-risk) ===')
    for r in age_stats:
        lo = f"{r['screen_pct_low']:.1f}" if r['screen_pct_low'] is not None else '-'
        hi = f"{r['screen_pct_high']:.1f}" if r['screen_pct_high'] is not None else '-'
        print(f"age={r['age']}: all={r['screen_pct_all']:.1f}%  low={lo}%  high={hi}%  (n_alive={r['n_alive']})")

    print('\n=== example schedules (low-risk) ===')
    for e in examples['low_risk']:
        print(f"  person {e['person']} (sex={e['sex']}): {e['screen_ages']}")
    print('=== example schedules (high-risk) ===')
    for e in examples['high_risk']:
        print(f"  person {e['person']} (sex={e['sex']}): {e['screen_ages']}")


if __name__ == '__main__':
    main()
