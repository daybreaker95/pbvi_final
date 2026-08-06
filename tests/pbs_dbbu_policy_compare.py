"""Does PBS/DBBU change the EXTRACTED POLICY, not just the reported gap/v_l
numbers? Trains three configs -- identical wtp=$50,000, seed, max_iters=39,
n_stochastic_trajectories=0 -- varying only use_pbs/dbbu_interval:
  no_pbs   : use_pbs=False, dbbu_interval=1 (exact every time, i.e. off)
  pbs      : use_pbs=True,  dbbu_interval=1 (matches the earlier ablation)
  pbs_dbbu : use_pbs=True,  dbbu_interval=10 (DBBU's caching shortcut
             actually engaged this time -- paper's own tested theta=10)

For each: full gap_history (39 iters) + age-wise SCREEN% schedule extracted
via N=500,000 MC (CRN, same convention as extract_policy_schedules.py) so
the three configs' EXTRACTED POLICIES can be compared directly, not just
their value-bound numbers."""
import os
import sys
import json
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9, NORMAL, CRC_DEATH, OTHER_DEATH
from pomdp.fivi import FiVI, WAIT, SCREEN, NO

MAX_ITERS = 39
WTP = 50_000


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


def run_config(use_pbs, dbbu_interval, n_sched=500_000):
    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97, wtp=WTP)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=MAX_ITERS, time_limit=None, precision=1e-9, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=use_pbs, dbbu_interval=dbbu_interval)
    gap_history = solver.gap_history

    risk, action_log = simulate_with_schedule(pomdp, solver, n_sched, seed=42)
    ages = sorted(action_log.keys())
    age_stats = []
    for age in ages:
        a = action_log[age]
        seen = a >= 0
        if seen.sum() == 0:
            continue
        pct = float((a[seen] == SCREEN).mean() * 100)
        if pct > 1.0:
            age_stats.append({'age': age, 'screen_pct_all': pct})
    return gap_history, age_stats


def main():
    configs = [
        ('no_pbs', dict(use_pbs=False, dbbu_interval=1)),
        ('pbs', dict(use_pbs=True, dbbu_interval=1)),
        ('pbs_dbbu10', dict(use_pbs=True, dbbu_interval=10)),
    ]
    results = {}
    for name, kw in configs:
        print(f'=== {name} ({kw}) ===', flush=True)
        gh, sched = run_config(**kw)
        results[name] = {'gap_history': gh, 'schedule': sched}
        print(f'  final gap={gh[-1]["gap"]:.4f}  final vl={gh[-1]["vl"]:.5f}', flush=True)
        print(f'  schedule: {[(r["age"], round(r["screen_pct_all"],1)) for r in sched]}', flush=True)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'pbs_dbbu_policy_compare.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('\nsaved', out_path)

    print('\n=== schedule comparison (age: no_pbs / pbs / pbs_dbbu10) ===')
    all_ages = sorted(set(r['age'] for cfg in results.values() for r in cfg['schedule']))
    maps = {name: {r['age']: r['screen_pct_all'] for r in results[name]['schedule']} for name, _ in configs}
    for age in all_ages:
        vals = [f"{maps[name].get(age, 0.0):.1f}%" for name, _ in configs]
        print(f"  age={age}: {' / '.join(vals)}")


if __name__ == '__main__':
    main()
