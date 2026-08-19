"""lambda_sweep.py
==================
Explores the (risk-stratification x colonoscopy-budget-penalty) grid to find
the PBVI configuration with the best mortality-reduction-per-colonoscopy
efficiency, following the same "cheap synthetic exploration -> expensive
real-engine confirmation" two-stage pattern already used elsewhere in this
project (e.g. cmost_4way_eval.py's own exploration/validation split).

Three changes are swept simultaneously, all newly parameterized (none of
this touches the previously-published age_min=40/top-25%/lambda=0 headline
results, which remain reproducible via the same defaults):

  1. Screening age window: age_min=50 (was 40) -- matches q10y/q5y's own
     start age (both fixed schedules start at 50), so the comparison no
     longer gives PBVI a 10-year head start q10y/q5y never get to use.
  2. High-risk definition: top-25% (results/transitions_9state_sex_risk_
     BACKUP_top25pct.npz) vs top-10% (results/transitions_9state_sex_risk.npz,
     current file, threshold 3.974249328225908) -- run BOTH, side by side.
  3. colo_penalty_qaly (lambda): a fixed per-SCREEN QALY "shadow price"
     stacked on top of the existing PROC_DISUTIL/comp_disutil terms (see
     pomdp/model_v2.py's CRCScreeningPOMDP9 docstring) -- swept over a grid
     to find where mortality-reduction-per-colonoscopy peaks.

STAGE 1 (this script, cheap): for every (risk_def, lambda) combo, train both
sex policies and evaluate via a FAST synthetic belief-propagation Monte
Carlo simulation (N=300,000, drawing directly from the POMDP's own learned
transition matrices -- no real engine call), tracking CRC-specific deaths
and total colonoscopy count per person. This is NOT the validated
NumberCrunching_policy engine; it is only used here to RANK candidates
cheaply. All results are saved so the ranking is auditable.

STAGE 2 (separate follow-up script, run only for the winning candidate(s)):
full N=1,000,000 CMOST-in-loop validation via cmost_4way_eval.py's own
train_policy/run_cohort, matching the exact methodology behind
final_algorithm/csv/02_policy_vs_schedules_final.csv.

Run: python3 tests/lambda_sweep.py
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

from pomdp.model_v2 import (
    CRCScreeningPOMDP9, NORMAL, CRC_DEATH, OTHER_DEATH,
)
from pomdp.fivi import FiVI, WAIT, SCREEN, NO

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

RISK_CONFIGS = {
    'top25': dict(
        npz=os.path.join(RES, 'transitions_9state_sex_risk_BACKUP_top25pct.npz'),
        threshold=3.5150943204057215,
    ),
    'top10': dict(
        npz=os.path.join(RES, 'transitions_9state_sex_risk.npz'),
        threshold=3.974249328225908,
    ),
}

LAMBDAS = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035,
           0.004, 0.0045, 0.005, 0.006, 0.007, 0.008, 0.01]

AGE_MIN = 50
AGE_MAX = 80
N_EVAL = 300_000
EVAL_SEED = 42


def train(sex, sex_risk_npz, colo_penalty_qaly):
    pomdp = CRCScreeningPOMDP9(age_min=AGE_MIN, age_max=AGE_MAX, gamma=0.97, sex=sex,
                                sex_risk_npz=sex_risk_npz, colo_penalty_qaly=colo_penalty_qaly)
    solver = FiVI(pomdp, seed=0, max_sawtooth_points=300)
    solver.solve(max_iters=50, time_limit=300, precision=1e-3, verbose=False,
                 n_stochastic_trajectories=0, use_pbs=True, dbbu_interval=1)
    return pomdp, solver


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def synthetic_eval(pomdp_m, solver_m, pomdp_f, solver_f, N, seed, no_screen=False):
    """Belief-propagation Monte Carlo over the POMDP's own transition
    matrices (env-free, fast). Tracks CRC-specific deaths separately from
    other-cause deaths and per-person colonoscopy count. If no_screen=True,
    every action is forced to WAIT regardless of the trained policy (used
    once per risk-config to get a comparison baseline internal to this
    same fast simulator, matching the two-stage exploration/confirmation
    split described in the module docstring)."""
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
    crc_dead = np.zeros(N, dtype=bool)
    n_colo = np.zeros(N, dtype=np.int32)
    b0_m, b0_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()
    belief = np.where(is_male[:, None], b0_m[None, :], b0_f[None, :])
    ages = list(range(pomdp_m.age_min, pomdp_m.life_max + 1))
    for age in ages:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if no_screen:
            actions = np.zeros(len(idx), dtype=int)
        else:
            actions = np.zeros(len(idx), dtype=int)
            if age <= pomdp_m.age_max:
                m_sub = is_male[idx]
                if m_sub.any():
                    actions[m_sub] = fivi_best_action_batch(solver_m, age, belief[idx[m_sub]])
                if (~m_sub).any():
                    actions[~m_sub] = fivi_best_action_batch(solver_f, age, belief[idx[~m_sub]])
        screened = idx[actions == SCREEN]
        n_colo[screened] += 1
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
                    crc_d = local_nxt == CRC_DEATH
                    oth_d = local_nxt == OTHER_DEATH
                    if crc_d.any():
                        crc_dead[grp[crc_d]] = True
                        alive[grp[crc_d]] = False
                    if oth_d.any():
                        alive[grp[oth_d]] = False
                    true_s[grp] = nxt
    return {
        'crc_death_per_100k': float(crc_dead.mean() * 100_000),
        'avg_colonoscopies_per_person': float(n_colo.mean()),
    }


def main():
    out_rows = []
    out_path = os.path.join(RES, 'lambda_sweep.json')

    for risk_name, cfg in RISK_CONFIGS.items():
        print(f'=== risk definition: {risk_name} (threshold={cfg["threshold"]:.4f}) ===', flush=True)

        # no_screen reference, internal to this same fast simulator, computed
        # once per risk-config (risk-config only changes disease dynamics for
        # the high-risk subgroup's underlying transitions, not the true
        # population split itself -- but we hold it fixed per risk_name for
        # full internal consistency within the sweep).
        pomdp_m0, solver_m0 = train(1, cfg['npz'], 0.0)
        pomdp_f0, solver_f0 = train(2, cfg['npz'], 0.0)
        ns = synthetic_eval(pomdp_m0, solver_m0, pomdp_f0, solver_f0, N_EVAL, EVAL_SEED, no_screen=True)
        no_screen_death = ns['crc_death_per_100k']
        print(f'  no_screen (synthetic, this simulator): crc_death_per_100k={no_screen_death:.1f}', flush=True)

        for lam in LAMBDAS:
            t0 = time.time()
            pomdp_m, solver_m = train(1, cfg['npz'], lam)
            pomdp_f, solver_f = train(2, cfg['npz'], lam)
            res = synthetic_eval(pomdp_m, solver_m, pomdp_f, solver_f, N_EVAL, EVAL_SEED, no_screen=False)
            colo = res['avg_colonoscopies_per_person']
            death = res['crc_death_per_100k']
            reduction = no_screen_death - death
            per_colo = reduction / colo if colo > 1e-9 else float('nan')
            row = dict(risk_def=risk_name, lam=lam, colo_per_person=colo,
                       crc_death_per_100k=death, no_screen_death_per_100k=no_screen_death,
                       mortality_reduction_per_100k=reduction, per_colo_efficiency=per_colo,
                       gap_male=solver_m.gap_history[-1]['gap'], gap_female=solver_f.gap_history[-1]['gap'],
                       elapsed_sec=time.time() - t0)
            out_rows.append(row)
            print(f'  lambda={lam:.4f}  colo/person={colo:.3f}  crc_death={death:.1f}  '
                  f'per_colo_eff={per_colo:.1f}  ({row["elapsed_sec"]:.1f}s)', flush=True)
            with open(out_path, 'w') as f:
                json.dump(out_rows, f, indent=2)

    print('\n=== best per_colo_efficiency by risk_def ===')
    for risk_name in RISK_CONFIGS:
        sub = [r for r in out_rows if r['risk_def'] == risk_name]
        best = max(sub, key=lambda r: r['per_colo_efficiency'])
        print(f'  {risk_name}: lambda={best["lam"]:.4f}  colo/person={best["colo_per_person"]:.3f}  '
              f'per_colo_eff={best["per_colo_efficiency"]:.1f}  crc_death={best["crc_death_per_100k"]:.1f}')

    print(f'\nSaved full grid to {out_path}')


if __name__ == '__main__':
    main()
