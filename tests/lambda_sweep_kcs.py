"""lambda_sweep_fine.py
=======================
Follow-up to lambda_sweep.py: the coarse grid showed the naive "maximize
per-colo efficiency" objective degenerates at lambda>=0.01 (colonoscopy
volume collapses BELOW q10y's own 2.61/person, absolute CRC mortality gets
WORSE than q10y even though the ratio looks best) -- see results/
lambda_sweep.json. This script restricts to the lambda in [0.001, 0.009]
band that stayed clinically sane in the coarse sweep, and builds a full
comparison against no_screen/q10y/q5y (all re-simulated through the SAME
fast synthetic engine and the SAME risk-stratification transition matrices,
for full internal consistency -- no_screen/q10y/q5y are policy-independent
fixed schedules, but their outcome numbers still depend on which risk-def's
underlying disease-transition matrices are loaded).

Adds incidence (diagnosed-CRC) tracking on top of lambda_sweep.py's
CRC-mortality-only synthetic_eval -- diagnosed = ever entered a
DET_CA_STAGES state (screening-detected or symptom-detected, no
distinction here; matches the "incidence = diagnosed" convention used
throughout this project's real-CMOST-engine tables).

Run: python3 tests/lambda_sweep_fine.py
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

from pomdp.model_v2 import CRCScreeningPOMDP9, CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
from pomdp.fivi import FiVI, WAIT, SCREEN, NO

RES = os.path.join(os.path.dirname(__file__), '..', 'results')

RISK_CONFIGS = {
    'kcs4pt': dict(npz=os.path.join(RES, 'transitions_9state_sex_risk_kcs4pt.npz')),
    'kcs5pt': dict(npz=os.path.join(RES, 'transitions_9state_sex_risk_kcs5pt.npz')),
}

LAMBDAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
Q10Y_AGES = {50, 60, 70}
Q5Y_AGES = {50, 55, 60, 65, 70, 75}

AGE_MIN = 50
AGE_MAX = 75   # matches q10y/q5y's own end age (75) and the source report's
               # #6 recommendation (USPSTF: 76-85 is selective/C-grade) --
               # was 80 in the first pass of this sweep (results/
               # lambda_sweep_fine_BACKUP_age80.json), superseded here.
N_EVAL = 300_000
EVAL_SEED = 42
_DET_SET = set(int(x) for x in DET_CA_STAGES)


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


def synthetic_eval(pomdp_m, solver_m, pomdp_f, solver_f, N, seed,
                    mode='policy', fixed_ages=None):
    """mode: 'policy' (use solver_m/solver_f), 'no_screen' (always WAIT),
    'fixed' (SCREEN iff age in fixed_ages, else WAIT) -- solver_m/solver_f
    still supply the transition matrices (M, frac_high, burnin) either way,
    they're just not consulted for the action when mode != 'policy'."""
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
    diagnosed = np.zeros(N, dtype=bool)
    n_colo = np.zeros(N, dtype=np.int32)
    b0_m, b0_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()
    belief = np.where(is_male[:, None], b0_m[None, :], b0_f[None, :])
    ages = list(range(pomdp_m.age_min, pomdp_m.life_max + 1))
    for age in ages:
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if mode == 'no_screen':
            actions = np.zeros(len(idx), dtype=int)
        elif mode == 'fixed':
            actions = np.full(len(idx), SCREEN if age in fixed_ages else WAIT, dtype=int)
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
                    det_d = np.isin(local_nxt, list(_DET_SET))
                    if det_d.any():
                        diagnosed[grp[det_d]] = True
                    if crc_d.any():
                        crc_dead[grp[crc_d]] = True
                        alive[grp[crc_d]] = False
                    if oth_d.any():
                        alive[grp[oth_d]] = False
                    true_s[grp] = nxt
    return {
        'crc_death_per_100k': float(crc_dead.mean() * 100_000),
        'incidence_per_100k': float(diagnosed.mean() * 100_000),
        'avg_colonoscopies_per_person': float(n_colo.mean()),
    }


def main():
    rows = []
    out_path = os.path.join(RES, 'lambda_sweep_kcs.json')

    for risk_name, cfg in RISK_CONFIGS.items():
        print(f'=== {risk_name} ===', flush=True)
        pomdp_m0, solver_m0 = train(1, cfg['npz'], 0.0)
        pomdp_f0, solver_f0 = train(2, cfg['npz'], 0.0)

        ns = synthetic_eval(pomdp_m0, solver_m0, pomdp_f0, solver_f0, N_EVAL, EVAL_SEED, mode='no_screen')
        q10 = synthetic_eval(pomdp_m0, solver_m0, pomdp_f0, solver_f0, N_EVAL, EVAL_SEED, mode='fixed', fixed_ages=Q10Y_AGES)
        q5 = synthetic_eval(pomdp_m0, solver_m0, pomdp_f0, solver_f0, N_EVAL, EVAL_SEED, mode='fixed', fixed_ages=Q5Y_AGES)

        no_screen_death, no_screen_inc = ns['crc_death_per_100k'], ns['incidence_per_100k']

        def make_row(scenario, lam, res):
            death, inc, colo = res['crc_death_per_100k'], res['incidence_per_100k'], res['avg_colonoscopies_per_person']
            mort_red = (no_screen_death - death) / no_screen_death * 100 if no_screen_death else float('nan')
            inc_red = (no_screen_inc - inc) / no_screen_inc * 100 if no_screen_inc else float('nan')
            per_colo = (no_screen_death - death) / colo if colo > 1e-9 else float('nan')
            return dict(risk_def=risk_name, scenario=scenario, lam=lam, colo_per_person=colo,
                        crc_death_per_100k=death, incidence_per_100k=inc,
                        mortality_reduction_pct=mort_red, incidence_reduction_pct=inc_red,
                        per_colo_efficiency=per_colo)

        rows.append(make_row('no_screen', None, ns))
        rows.append(make_row('q10y', None, q10))
        rows.append(make_row('q5y', None, q5))
        for r in rows[-3:]:
            print(f"  {r['scenario']:10s}  colo={r['colo_per_person']:.3f}  death={r['crc_death_per_100k']:.1f}  "
                  f"inc={r['incidence_per_100k']:.1f}  mort_red={r['mortality_reduction_pct']:.1f}%  "
                  f"inc_red={r['incidence_reduction_pct']:.1f}%", flush=True)

        for lam in LAMBDAS:
            t0 = time.time()
            pomdp_m, solver_m = train(1, cfg['npz'], lam)
            pomdp_f, solver_f = train(2, cfg['npz'], lam)
            res = synthetic_eval(pomdp_m, solver_m, pomdp_f, solver_f, N_EVAL, EVAL_SEED, mode='policy')
            row = make_row(f'pbvi_lambda', lam, res)
            row['elapsed_sec'] = time.time() - t0
            rows.append(row)
            print(f"  lambda={lam:.3f}  colo={row['colo_per_person']:.3f}  death={row['crc_death_per_100k']:.1f}  "
                  f"inc={row['incidence_per_100k']:.1f}  mort_red={row['mortality_reduction_pct']:.1f}%  "
                  f"inc_red={row['incidence_reduction_pct']:.1f}%  per_colo_eff={row['per_colo_efficiency']:.1f}  "
                  f"({row['elapsed_sec']:.1f}s)", flush=True)
            with open(out_path, 'w') as f:
                json.dump(rows, f, indent=2)

    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
