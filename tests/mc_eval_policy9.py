"""
mc_eval_policy9.py
===================
Monte-Carlo clinical-outcome evaluation of a trained PBVI9 policy: N
synthetic patients, TRUE hidden state sampled forward via the SAME
per-observation matrices the POMDP itself uses (so this is internally
consistent with the trained value function, not a re-run of CMOST), with
the agent choosing WAIT/SCREEN each year from its OWN belief (batched
Q-value computation, not per-patient Python calls). Reports avg age at
death, CRC-death rate, incidence, and stage-at-diagnosis mix, under the
policy vs a fixed no-screening baseline -- used to check whether these
CLINICAL outputs (not just V(b0)) have stabilised as PBVI resources grow.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pomdp.model_v2 import (CRCScreeningPOMDP9, NORMAL,
                             CRC_DEATH, OTHER_DEATH, WAIT, SCREEN, NO)
from pomdp.pbvi_v2 import PBVI9


def batched_best_action(solver, age, B):
    p = solver.p
    if age > p.age_max:
        return np.zeros(len(B), dtype=int)
    R = p.R[age]
    A_next, _ = solver._next_gamma(age)
    Qs = np.zeros((len(B), 2))
    for a in (WAIT, SCREEN):
        Ms = solver._Mstack(age, a)                      # (NO,NS,NS)
        Bo = np.einsum('ns,ost->not', B, Ms)              # (N,NO,NS)
        scores = np.einsum('not,mt->nom', Bo, A_next)     # (N,NO,M)
        best = scores.max(axis=2).sum(axis=1)             # (N,)
        Qs[:, a] = B @ R[a] + p.gamma * best
    return Qs.argmax(axis=1)


def simulate(pomdp: CRCScreeningPOMDP9, solver, N, seed, policy=True, family_history=None):
    rng = np.random.default_rng(seed)
    NS = pomdp.NS
    risk = (rng.random(N) < pomdp.frac_high).astype(int)
    true_s = np.array([pomdp.fidx(r, NORMAL) for r in risk])
    alive = np.ones(N, dtype=bool)
    death_age = np.full(N, float(pomdp.life_max))
    death_cause = np.zeros(N, dtype=int)     # 0 none, 1 other, 2 crc
    dx_stage = np.zeros(N, dtype=int)        # 1..4, 0 none
    dx_route = np.zeros(N, dtype=int)        # 1 screen, 2 symptom
    b0 = pomdp.initial_belief(family_history=family_history)
    belief = np.tile(b0, (N, 1))

    from pomdp.model_v2 import DET_CA_STAGES
    det_local = {s: k + 1 for k, s in enumerate(DET_CA_STAGES)}

    for age in range(pomdp.age_min, pomdp.life_max + 1):
        idx = np.where(alive)[0]
        if len(idx) == 0:
            break
        if policy and age <= pomdp.age_max:
            actions = batched_best_action(solver, age, belief[idx])
        else:
            actions = np.zeros(len(idx), dtype=int)   # WAIT-only (no-screen baseline / post age_max)
        M = pomdp.M[min(age, pomdp.life_max)]

        for a in (WAIT, SCREEN):
            sub = idx[actions == a]
            if len(sub) == 0:
                continue
            s_snap = true_s[sub].copy()   # freeze pre-transition states: true_s[sub] mutates
                                           # in place below (true_s[grp]=nxt), so re-reading it
                                           # live would let someone who JUST moved into state s'
                                           # this same age get double-transitioned when s' is
                                           # visited later in this loop
            for s in np.unique(s_snap):
                grp = sub[s_snap == s]
                row = np.concatenate([M[a][o][s, :] for o in range(NO)])
                row = row / row.sum()
                cdf = np.cumsum(row)
                u = rng.random(len(grp))
                flat = np.searchsorted(cdf, u, side='right')
                obs = flat // NS
                nxt = flat % NS
                # belief update (only meaningful for policy runs, harmless otherwise)
                for o in range(NO):
                    m = grp[obs == o]
                    if len(m) == 0:
                        continue
                    bn = belief[m] @ M[a][o]
                    tot = bn.sum(axis=1, keepdims=True)
                    ok = tot[:, 0] > 1e-300
                    belief[m[ok]] = bn[ok] / tot[ok]
                # outcome bookkeeping
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
                    dx_route[g2] = 1 if a == SCREEN else 2
                true_s[grp] = nxt

    living = alive
    death_age[living] = pomdp.life_max
    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage, dx_route=dx_route)


def summarize(r):
    crc = r['death_cause'] == 2
    dx = r['dx_stage']
    n = len(dx)
    return dict(
        avg_age_all=float(r['death_age'].mean()),
        avg_age_crc=float(r['death_age'][crc].mean()) if crc.any() else float('nan'),
        crc_100k=1e5 * crc.mean(),
        incid_100k=1e5 * (dx > 0).mean(),
        stage_pct=np.array([100 * np.mean(dx[dx > 0] == k) for k in range(1, 5)]) if (dx > 0).any() else np.zeros(4),
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=20000)
    ap.add_argument('--n_belief', type=int, default=1500)
    ap.add_argument('--expansions', type=int, default=6)
    args = ap.parse_args()

    pomdp = CRCScreeningPOMDP9(age_min=40, age_max=85, gamma=0.97)
    solver = PBVI9(pomdp, n_belief=args.n_belief, seed=0)
    solver.solve(expansions=args.expansions, horizon=45, n_rollouts=200,
                 max_belief=max(6000, 4 * args.n_belief))

    print("\nSimulating policy vs no-screen baseline ...")
    r_pol = simulate(pomdp, solver, args.n, seed=1, policy=True)
    r_nos = simulate(pomdp, solver, args.n, seed=1, policy=False)
    s_pol, s_nos = summarize(r_pol), summarize(r_nos)
    print(f"{'metric':<22}{'no-screen':>12}{'policy':>12}")
    print(f"{'avg age at death':<22}{s_nos['avg_age_all']:>12.2f}{s_pol['avg_age_all']:>12.2f}")
    print(f"{'avg age at CRC death':<22}{s_nos['avg_age_crc']:>12.2f}{s_pol['avg_age_crc']:>12.2f}")
    print(f"{'CRC deaths /100k':<22}{s_nos['crc_100k']:>12.0f}{s_pol['crc_100k']:>12.0f}")
    print(f"{'incidence /100k':<22}{s_nos['incid_100k']:>12.0f}{s_pol['incid_100k']:>12.0f}")
    for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
        print(f"{'stage ' + lbl + ' % at dx':<22}{s_nos['stage_pct'][k]:>12.1f}{s_pol['stage_pct'][k]:>12.1f}")


if __name__ == '__main__':
    main()
