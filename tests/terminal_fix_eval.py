"""terminal_fix_eval.py
========================
Tests the human-in-the-loop terminal-year correction (pomdp/fivi.py's
best_action now reuses age_max-1's decision rule at age_max, instead of the
myopic "no more future" terminal rule) against the ORIGINAL hard-cutoff
headline: does the age-80 SCREEN% spike shrink, and how do the aggregate
mortality/incidence/cost/colonoscopy numbers change?

Two parts, same pattern as extract_policy_schedules.py / cmost_4way_eval.py:
  1. Age-wise SCREEN% (N=500,000 synthetic sim) -- direct look at age 80.
  2. Real-CMOST N=1,000,000 4-way validation -- headline comparison against
     the existing final_algorithm/csv/02_policy_vs_schedules_final.csv row.
"""
import os
import sys
import json
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRC_DEATH, OTHER_DEATH
from pomdp.fivi import WAIT, SCREEN, NO
import cmost_4way_eval as C4

RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def fivi_best_action_batch(solver, age, B):
    """Same terminal-year correction as the now-patched FiVI.best_action:
    age_max reuses age_max-1's Gamma instead of the myopic terminal one."""
    t = min(age, solver.p.age_max - 1) - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed):
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


def part1_age_wise(pomdp_m, solver_m, pomdp_f, solver_f):
    print('=== Part 1: age-wise SCREEN% (N=500,000 synthetic sim) ===', flush=True)
    N = 500_000
    risk, is_male, action_log = simulate_with_schedule_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed=42)
    ages = sorted(action_log.keys())
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
        age_stats.append({'age': age, 'screen_pct_all': screen_pct,
                           'screen_pct_low': low_pct, 'screen_pct_high': high_pct,
                           'n_alive': int(seen.sum())})
        if 70 <= age <= 85 or screen_pct > 1.0:
            print(f'  age={age}: all={screen_pct:.1f}%  low={low_pct:.1f}%  high={high_pct:.1f}%', flush=True)
    return age_stats


def part2_cmost_in_loop(pomdp_m, solver_m, pomdp_f, solver_f, n=1_000_000, seed=999):
    print(f'=== Part 2: real-CMOST N={n:,} validation ===', flush=True)
    np.random.seed(seed)
    p = C4.BNH.prepare_simulation_params(n)
    risk_class = (np.asarray(p['individual_risk']) >= C4.RISK_THRESHOLD).astype(int)
    sex_arr = np.asarray(p['gender_arr']).astype(int)
    hook = C4.SexAwareEngineHook(pomdp_m, solver_m, pomdp_f, solver_f, risk_class, sex_arr, seed=seed)
    t0 = time.time()
    _, sr, ncr, money, number, tumor_record, death_year = C4.run_cohort(p, n, seed, hook, hook_age_max=80)
    stats = C4.summarize(sr, ncr, money, number, tumor_record, death_year, n)
    print(f'  done ({time.time()-t0:.0f}s): {stats}', flush=True)
    return stats


def main():
    print('Training policies (standard hard-cutoff age_max=80, patched best_action)...', flush=True)
    t0 = time.time()
    pomdp_m, solver_m = C4.train_policy(sex=1)
    print(f'  male gap={solver_m.gap_history[-1]["gap"]:.4f} ({time.time()-t0:.1f}s)', flush=True)
    pomdp_f, solver_f = C4.train_policy(sex=2)
    print(f'  female gap={solver_f.gap_history[-1]["gap"]:.4f} ({time.time()-t0:.1f}s)', flush=True)

    age_stats = part1_age_wise(pomdp_m, solver_m, pomdp_f, solver_f)
    cmost_stats = part2_cmost_in_loop(pomdp_m, solver_m, pomdp_f, solver_f)

    out = {'age_stats': age_stats, 'cmost_4way_stats': cmost_stats,
           'gap_male': solver_m.gap_history[-1]['gap'], 'gap_female': solver_f.gap_history[-1]['gap']}
    out_path = os.path.join(RES, 'terminal_fix_eval.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('saved', out_path)


if __name__ == '__main__':
    main()
