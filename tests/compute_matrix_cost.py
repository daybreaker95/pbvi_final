"""
compute_matrix_cost.py
========================
Computes matrix's OWN predicted expected total cost per person (no-screening
vs trained policy), for direct comparison against real CMOST's actual
cost-per-person -- a validation matrix vs cmost.py never did (only clinical
metrics: CRC deaths, incidence, stage%). Motivated by a user question: if
the matrix's survival-timing predictions are off (LYG mismatch), its
cost predictions (which partly depend on how long someone survives in a
continuing-care state) should plausibly be off too.

model_v2.py's actual _undet_MR reward DOES include the terminal/at-death
Final_I..IV cost swap (a previously-fixed bug, see model_v2.py's
DETECTED-block cost_blend and _route_diag's cost_bump) -- but THIS file's
cost-only mirror (undet_MR_cost) was written before that fix and never
updated to match, so it silently UNDERSTATED matrix cost by omitting the
Final-cost swap entirely (2026-07-29 fix: now mirrors both the DETECTED-
block blend and the diagnosis-routing same-year-death bump).

Method: byte-for-byte mirror of pomdp.model_v2.CRCScreeningPOMDP9._undet_MR,
but every `u - cost/wtp` (or `u_k - cost_k/wtp`) blended term is replaced
by the bare `cost` (or `cost_k`) -- i.e. same diagnosis-routing weights,
same T_det/d_symp/d_PC_stage tables, just tracking dollars instead of
QALY-minus-cost/wtp. Then runs the SAME CRN MC trajectory structure as
gather_fivi_full_trajectory.simulate_fivi, accumulating cost instead of
disc_reward.

Run: python compute_matrix_cost.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
import numpy as np

from pomdp.model_v2 import (
    CRCScreeningPOMDP9, EARLY_POLYP, ADV_POLYP, CA_STAGES,
    CRC_DEATH, OTHER_DEATH, DET_CA_STAGES, N_STATES9, WAIT, SCREEN, NO,
)
from pomdp.fivi import FiVI
import cmost_4way_eval as C4


def undet_MR_cost(pomdp, age, r):
    """Cost-only mirror of pomdp._undet_MR -- returns (rw_cost, rs_cost),
    NC-vectors of expected ANNUAL dollar cost (same diagnosis-routing
    weights as the real reward, no utility/wtp blending). Now mirrors BOTH
    Final-cost swap sites in the real reward:
      (a) DETECTED block: cost_blend = (p_stay+p_oth)*cont_cost + p_crc*final_cost
      (b) diagnosis-routing (_route_diag): same-year-CRC-death adds a
          p_crc_k*final_cost[k] bump ON TOP OF the diagnosis-year init_cost."""
    Pd = pomdp._P_undet[r]
    dd = pomdp._d_symp[r]
    Tn = Pd[min(max(age, min(Pd)), max(Pd))]
    dsy = dd[min(max(age, min(dd)), max(dd))]
    d_pc = pomdp.d_PC_stage
    Tdet_row = pomdp.T_det[min(age, pomdp.life_max) - 1]  # (4,3): p_stay,p_crc,p_oth per stage

    def cost_bump(k, mass):
        """Extra $ from a same-year CRC death among `mass` newly diagnosed
        at stage k -- mirrors _route_diag's cost_bump exactly."""
        p_crc_k = Tdet_row[k, 1]
        return mass * p_crc_k * pomdp.final_cost[k]

    rw = np.zeros(pomdp.NC)
    rs = np.zeros(pomdp.NC)

    for k, ds in enumerate(DET_CA_STAGES):
        p_stay, p_crc, p_oth = Tdet_row[k]
        _, cost = pomdp._qaly_detected(age, k, just_diagnosed=False)
        cost_blend = (p_stay + p_oth) * cost + p_crc * pomdp.final_cost[k]
        rw[ds] = cost_blend
        rs[ds] = cost_blend

    for s in range(N_STATES9):
        if s in (CRC_DEATH, OTHER_DEATH):
            continue
        _, cost = pomdp._qaly_undetected(age, s, WAIT)
        rw[s] = cost
        if s in CA_STAGES:
            p_diag_vec = dsy[s]
            p_diag_total = float(p_diag_vec.sum())
            if p_diag_total > 0:
                val_avg = 0.0
                for k in range(4):
                    if p_diag_vec[k] > 0:
                        _, cost_k = pomdp._qaly_detected(age, k, just_diagnosed=True)
                        val_avg += p_diag_vec[k] * cost_k + cost_bump(k, p_diag_vec[k])
                rw[s] += val_avg - p_diag_total * cost

    for s in range(N_STATES9):
        if s in (CRC_DEATH, OTHER_DEATH):
            continue
        _, cost = pomdp._qaly_undetected(age, s, SCREEN)
        rs[s] = cost
        if s in CA_STAGES:
            i = CA_STAGES.index(s)
            _, cost_i = pomdp._qaly_detected(age, i, just_diagnosed=True)
            rs[s] += d_pc[i] * (cost_i - cost) + cost_bump(i, d_pc[i])
            p_diag_vec = (1 - d_pc[i]) * dsy[s]
            p_diag_total = float(p_diag_vec.sum())
            if p_diag_total > 0:
                val_avg = 0.0
                for k in range(4):
                    if p_diag_vec[k] > 0:
                        _, cost_k = pomdp._qaly_detected(age, k, just_diagnosed=True)
                        val_avg += p_diag_vec[k] * cost_k + cost_bump(k, p_diag_vec[k])
                rs[s] += val_avg - p_diag_total * cost

    return rw, rs


def build_Rcost(pomdp):
    Rcost = {}
    for age in range(pomdp.age_min, pomdp.life_max + 1):
        rw_full = np.zeros(pomdp.NS)
        rs_full = np.zeros(pomdp.NS)
        for r in range(pomdp.n_risk):
            rw, rs = undet_MR_cost(pomdp, age, r)
            blk = slice(r * pomdp.NC, (r + 1) * pomdp.NC)
            rw_full[blk] = rw
            rs_full[blk] = rs
        Rcost[age] = {WAIT: rw_full, SCREEN: rs_full}
    return Rcost


def fivi_best_action_batch(solver, age, B):
    t = age - solver.p.age_min + 1
    alphas, acts = solver.Gamma[t]
    if len(alphas) == 0:
        return np.zeros(len(B), dtype=int)
    idx = np.argmax(alphas @ B.T, axis=0)
    return acts[idx]


def simulate_cost(pomdp, solver, Rcost, N, seed, policy):
    rng = np.random.default_rng((seed, 0))
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    # See gather_fivi_full_trajectory.simulate_fivi -- true starting state
    # drawn from the age_min burn-in distribution, not fixed at Normal.
    true_s = np.empty(N, dtype=int)
    for r in range(pomdp.n_risk):
        mask = risk == r
        cnt = int(mask.sum())
        if cnt > 0:
            s_local = rng.choice(pomdp.NC, size=cnt, p=pomdp._burnin_dist[r])
            true_s[mask] = np.array([pomdp.fidx(r, int(s)) for s in s_local])
    alive = np.ones(N, dtype=bool)
    b0 = pomdp.initial_belief()
    belief = np.tile(b0, (N, 1))
    cum_cost = np.zeros(N)

    for age in range(pomdp.age_min, pomdp.life_max + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if policy and age <= pomdp.age_max:
            actions = fivi_best_action_batch(solver, age, belief[idx])
        else:
            actions = np.zeros(len(idx), dtype=int)
        M = pomdp.M[min(age, pomdp.life_max)]
        Rc = Rcost[min(age, pomdp.life_max)]
        u_age = np.random.default_rng((seed, age)).random(N)
        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            cum_cost[sub] += Rc[a][true_s[sub]]
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
    return cum_cost


def main():
    pomdp, solver = C4.train_policy()
    Rcost = build_Rcost(pomdp)

    MC_N = 2_000_000
    cost_nos = simulate_cost(pomdp, solver, Rcost, MC_N, seed=42, policy=False)
    cost_pol = simulate_cost(pomdp, solver, Rcost, MC_N, seed=42, policy=True)

    print(f'[matrix] no-screen: mean cost/person = ${cost_nos.mean():,.2f}')
    print(f'[matrix] policy:    mean cost/person = ${cost_pol.mean():,.2f}')
    print(f'[matrix] delta cost = ${cost_pol.mean() - cost_nos.mean():,.2f}')
    print()
    print('Real CMOST (original $1160 cost basis, N=300k, from earlier this session):')
    print('  no-screen: $2,699.13   policy: $5,276.28   delta: $2,577.15')


if __name__ == '__main__':
    main()
