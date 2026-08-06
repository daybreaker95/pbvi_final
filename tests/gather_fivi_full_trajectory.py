"""Gathers FiVI (with PBS+DBBU) training trajectory at fine checkpoint
granularity: belief-point count, vl/vu/gap, AND periodic Monte-Carlo
clinical-outcome evaluation (using the policy AS OF that checkpoint), so
we can see whether clinical metrics stabilise alongside the value bounds
-- not just at the final iteration."""
import os
import sys
import json
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import CRCScreeningPOMDP9
from pomdp.fivi import FiVI
from mc_eval_policy9 import simulate, summarize
from compute_matrix_cost import build_Rcost, simulate_cost


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_fivi(pomdp, solver, N, seed, policy=True):
    """Same structure as mc_eval_policy9.simulate(), but drives actions
    from FiVI's Gamma (fivi_best_action_batch) instead of PBVI9.q_values.

    Common Random Numbers (CRN): person i's random draw at age `a` comes
    from a substrate u_age = default_rng((seed, a)).random(N), generated
    from ONLY (seed, age) -- NOT from which action/group person i lands
    in. So person i's "dice roll" at every age is IDENTICAL across
    different calls (different checkpoints/policies) to this function,
    as long as they're still alive and at the same age -- isolating the
    checkpoint-to-checkpoint difference in avg reward to genuine policy
    differences instead of independent-sample noise. Previously this used
    a single shared rng.random(len(grp)) per (action,state) group, which
    only stayed synchronized until the first point the policy diverged."""
    from pomdp.model_v2 import CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
    from pomdp.fivi import WAIT, SCREEN, NO
    rng = np.random.default_rng((seed, 0))
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    # Draw each person's TRUE starting state (not just the agent's belief)
    # from the age_min burn-in distribution -- ~10% of a real age-40
    # population already carries a precursor lesion (or rarely an
    # undetected cancer), not the "everyone is 100% Normal" this used to
    # assume. See pomdp.model_v2._compute_burnin_dist.
    true_s = np.empty(N, dtype=int)
    for r in range(pomdp.n_risk):
        mask = risk == r
        cnt = int(mask.sum())
        if cnt > 0:
            s_local = rng.choice(pomdp.NC, size=cnt, p=pomdp._burnin_dist[r])
            true_s[mask] = np.array([pomdp.fidx(r, int(s)) for s in s_local])
    alive = np.ones(N, dtype=bool)
    death_age = np.full(N, float(pomdp.life_max))
    death_cause = np.zeros(N, dtype=int)
    dx_stage = np.zeros(N, dtype=int)
    b0 = pomdp.initial_belief()
    belief = np.tile(b0, (N, 1))
    det_local = {s: k + 1 for k, s in enumerate(DET_CA_STAGES)}
    disc_reward = np.zeros(N)   # empirical discounted return -- the "R" metric
                                 # Perseus/FiVI papers actually plot as "reward"
                                 # (avg over simulated trajectories under the
                                 # policy), NOT the analytical v_l/v_u bounds.
    cum_qaly = np.zeros(N)      # UNDISCOUNTED lifetime sum of (utility - cost/wtp) --
                                 # the ssrn-3802759-style "total QALYs" headline metric
                                 # (their paper reports "+2.06 QALY/patient" as ITS
                                 # headline, not a discounted scalar). This is what the
                                 # POMDP is NOT optimizing (gamma=0.97 discounted reward
                                 # is still the actual Bellman objective) -- cum_qaly is
                                 # purely a reporting-side derived quantity computed from
                                 # the SAME simulated trajectories.

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
        u_age = np.random.default_rng((seed, age)).random(N)   # CRN substrate for this age
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            disc_reward[sub] += disc * R[a][true_s[sub]]
            cum_qaly[sub] += R[a][true_s[sub]]
            s_snap = true_s[sub].copy()   # freeze pre-transition states -- true_s[sub] mutates
                                           # in place below (true_s[grp]=nxt), so re-reading it
                                           # live would let someone who JUST moved into state s'
                                           # this same age get double-transitioned when s' is
                                           # visited later in this loop. Under CRN this is
                                           # catastrophic (not just noisy): the double-hit reuses
                                           # the SAME u_age[person] value against a DIFFERENT
                                           # state's cdf, producing a systematically biased
                                           # outcome rather than a fresh random one.
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


def simulate_fivi_pair(pomdp_m, solver_m, pomdp_f, solver_f, N, seed, policy=True):
    """Sex-aware counterpart of simulate_fivi: same CRN/mechanics, but each
    individual is routed to their own sex-specific (pomdp, solver) pair --
    sex is observable at t=0 (drawn once, fixed for the whole trajectory
    run via the same (seed,0) rng substrate risk/burn-in already use), not
    belief-tracked like risk_class. fraction_female=0.5 matches CMOST's own
    settings value (confirmed via env/params.py's load_settings('CMOST13'))."""
    from pomdp.model_v2 import CRC_DEATH, OTHER_DEATH, DET_CA_STAGES
    from pomdp.fivi import WAIT, SCREEN, NO
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
    death_age = np.full(N, float(pomdp_m.life_max))
    death_cause = np.zeros(N, dtype=int)
    dx_stage = np.zeros(N, dtype=int)
    b0_m, b0_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()
    belief = np.where(is_male[:, None], b0_m[None, :], b0_f[None, :])
    det_local = {s: k + 1 for k, s in enumerate(DET_CA_STAGES)}
    disc_reward = np.zeros(N)
    cum_qaly = np.zeros(N)

    for age in range(pomdp_m.age_min, pomdp_m.life_max + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        actions = np.zeros(len(idx), dtype=int)
        if policy and age <= pomdp_m.age_max:
            m_sub = is_male[idx]
            if m_sub.any():
                actions[m_sub] = fivi_best_action_batch(solver_m, age, belief[idx[m_sub]])
            if (~m_sub).any():
                actions[~m_sub] = fivi_best_action_batch(solver_f, age, belief[idx[~m_sub]])
        disc = pomdp_m.gamma ** (age - pomdp_m.age_min)
        u_age = np.random.default_rng((seed, age)).random(N)   # CRN substrate, shared across sexes
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
                R = pomdp.R[min(age, pomdp.life_max)]
                disc_reward[sub2] += disc * R[a][true_s[sub2]]
                cum_qaly[sub2] += R[a][true_s[sub2]]
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
                        mm = grp[obs == o]
                        if len(mm) == 0:
                            continue
                        bn = belief[mm] @ M[a][o]
                        tot = bn.sum(axis=1, keepdims=True)
                        ok = tot[:, 0] > 1e-300
                        belief[mm[ok]] = bn[ok] / tot[ok]
                    local_nxt = nxt % NC
                    became_crc = local_nxt == CRC_DEATH
                    became_oth = local_nxt == OTHER_DEATH
                    if became_crc.any():
                        g2 = grp[became_crc]
                        alive[g2] = False; death_age[g2] = age + 0.5; death_cause[g2] = 2
                    if became_oth.any():
                        g2 = grp[became_oth]
                        alive[g2] = False; death_age[g2] = age + 0.5; death_cause[g2] = 1
                    newly_det = np.isin(local_nxt, list(det_local.keys())) & (dx_stage[grp] == 0)
                    if newly_det.any():
                        g2 = grp[newly_det]
                        stg = np.array([det_local[v] for v in local_nxt[newly_det]])
                        dx_stage[g2] = stg
                    true_s[grp] = nxt
    death_age[alive] = pomdp_m.life_max
    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage,
                disc_reward=disc_reward, cum_qaly=cum_qaly)


def main():
    pomdp_m = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, sex=1)
    pomdp_f = CRCScreeningPOMDP9(age_min=40, age_max=80, gamma=0.97, sex=2)
    solver_m = FiVI(pomdp_m, seed=0, max_sawtooth_points=300)
    solver_f = FiVI(pomdp_f, seed=0, max_sawtooth_points=300)
    b1_m, b1_f = pomdp_m.initial_belief(), pomdp_f.initial_belief()

    # PURE Algorithm 1-3 (no stochastic-trajectory diversification, no PBS)
    # -- the faithful, paper-as-written comparison, iteration cost recorded
    # in wall-clock seconds too (x-axis-as-time idea). MC_N reduced from
    # 10M to 2M vs the sex-pooled version -- two solvers now train and get
    # MC-evaluated every checkpoint, and the N-sweep already showed these
    # metrics stabilize well before 300k, so 2M keeps wall-clock reasonable
    # without sacrificing precision.
    MC_N = 2_000_000
    CHECKPOINT_EVERY = 3
    MAX_ITERS = 50
    TIME_LIMIT = 5400

    Rcost_m = build_Rcost(pomdp_m)
    Rcost_f = build_Rcost(pomdp_f)

    r_nos = simulate_fivi_pair(pomdp_m, solver_m, pomdp_f, solver_f, MC_N, seed=42, policy=False)
    s_nos = summarize(r_nos)
    reward_nos = float(r_nos['disc_reward'].mean())
    cum_qaly_nos = float(r_nos['cum_qaly'].mean())
    # cost under no-screening doesn't depend on the trained policy at all,
    # so either sex's Rcost/solver gives an equivalent WAIT-only baseline;
    # blend both sexes' own cost structure via the same simulate_fivi_pair
    # mechanics for consistency rather than picking one arbitrarily.
    cost_nos = float(simulate_cost(pomdp_m, solver_m, Rcost_m, MC_N, seed=42, policy=False).mean() * 0.5
                      + simulate_cost(pomdp_f, solver_f, Rcost_f, MC_N, seed=42, policy=False).mean() * 0.5)
    life_years_nos = float((r_nos['death_age'] - pomdp_m.age_min).mean())

    trajectory = []
    t0 = time.time()
    train_time = 0.0   # cumulative FiVI solve time (both sexes), excludes MC eval time
    for kappa in range(1, MAX_ITERS + 1):
        t_iter0 = time.time()
        for pomdp, solver in ((pomdp_m, solver_m), (pomdp_f, solver_f)):
            solver.expand(stochastic=False)   # ONE trajectory per iteration, as in the paper
            exact_ub = True                    # DBBU off (dbbu_interval=1) -- exact every time
            for t in range(solver.h, 0, -1):
                B = np.vstack([solver.eye, solver.extra_b[t]])
                alphas, acts = solver._backup_batch(B, t)   # exhaustive backup (no PBS)
                solver.Gamma[t] = (alphas, acts)
                ub_vals = solver._ub_backup_batch(B, t, exact=exact_ub)
                solver.corner_ub[t] = ub_vals[:pomdp.NS]
                if len(solver.extra_ub[t]):
                    solver.extra_ub[t] = ub_vals[pomdp.NS:]
        train_time += time.time() - t_iter0

        A1_m, _ = solver_m.Gamma[1]
        A1_f, _ = solver_f.Gamma[1]
        vl = 0.5 * float(np.max(A1_m @ b1_m)) + 0.5 * float(np.max(A1_f @ b1_f))
        vu = 0.5 * solver_m.UB(b1_m, 1) + 0.5 * solver_f.UB(b1_f, 1)
        gap = vu - vl
        n_belief = (sum(len(v) for v in solver_m.extra_ub.values())
                    + sum(len(v) for v in solver_f.extra_ub.values()))
        rec = {'iter': kappa, 'train_time': train_time, 'vl': vl, 'vu': vu, 'gap': gap, 'n_belief': n_belief}
        print(f"iter {kappa}: t={train_time:.1f}s vl={vl:.4f} vu={vu:.4f} gap={gap:.4f} n_belief={n_belief}", flush=True)

        if kappa % CHECKPOINT_EVERY == 0 or kappa == 1:
            tmc0 = time.time()
            r_pol = simulate_fivi_pair(pomdp_m, solver_m, pomdp_f, solver_f, MC_N, seed=42, policy=True)
            s_pol = summarize(r_pol)
            rec['reward'] = float(r_pol['disc_reward'].mean())   # empirical MC discounted return (CRN)
            rec['cum_qaly'] = float(r_pol['cum_qaly'].mean())    # undiscounted lifetime QALY total (CRN)
            rec['delta_qaly_vs_noscreen'] = rec['cum_qaly'] - cum_qaly_nos   # ssrn-3802759-style headline
            rec['life_years'] = float((r_pol['death_age'] - pomdp_m.age_min).mean())  # LYG metric (Zaika-comparable)
            rec['delta_lyg_vs_noscreen'] = rec['life_years'] - life_years_nos
            rec['cost'] = float(0.5 * simulate_cost(pomdp_m, solver_m, Rcost_m, MC_N, seed=42, policy=True).mean()
                                 + 0.5 * simulate_cost(pomdp_f, solver_f, Rcost_f, MC_N, seed=42, policy=True).mean())
            rec['delta_cost_vs_noscreen'] = rec['cost'] - cost_nos
            rec['mortality_reduction_pct'] = (s_nos['crc_100k'] - s_pol['crc_100k']) / s_nos['crc_100k'] * 100
            rec['incidence_reduction_pct'] = (s_nos['incid_100k'] - s_pol['incid_100k']) / s_nos['incid_100k'] * 100
            rec['clinical'] = {
                'crc_100k': s_pol['crc_100k'], 'incid_100k': s_pol['incid_100k'],
                'avg_age_all': s_pol['avg_age_all'],
                'stage_pct': s_pol['stage_pct'].tolist(),
            }
            print(f"   reward={rec['reward']:.5f}  cum_qaly={rec['cum_qaly']:.4f}  cost=${rec['cost']:,.2f}  "
                  f"Δqaly_vs_noscreen={rec['delta_qaly_vs_noscreen']:+.4f}  "
                  f"ΔLYG_vs_noscreen={rec['delta_lyg_vs_noscreen']:+.4f}  "
                  f"mort_red={rec['mortality_reduction_pct']:.1f}%  incid_red={rec['incidence_reduction_pct']:.1f}%"
                  f"  (MC eval took {time.time()-tmc0:.0f}s)", flush=True)
        trajectory.append(rec)
        if time.time() - t0 > TIME_LIMIT:
            break

    out = {
        'trajectory': trajectory,
        'no_screen_baseline': {
            'crc_100k': s_nos['crc_100k'], 'incid_100k': s_nos['incid_100k'],
            'avg_age_all': s_nos['avg_age_all'], 'stage_pct': s_nos['stage_pct'].tolist(),
            'reward': reward_nos, 'cum_qaly': cum_qaly_nos, 'life_years': life_years_nos,
            'cost': cost_nos,
        },
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'fivi_full_trajectory.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print('saved', out_path)


if __name__ == '__main__':
    main()
