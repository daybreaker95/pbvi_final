"""
verify_screening_quarterly.py -- QUARTER-stepping 14-state transition-matrix
cohort vs the CMOST microsim, under no screening / q10y / q5y colonoscopy.

Same 14-state model as verify_screening_yearly.py, but the transition matrix is
estimated and propagated on a QUARTER-year step (4 steps/year). This removes the
annual-discretization artifacts (issues (1) incidence undercount and (3) stage-IV
undercount): under no screening the cohort now matches the microsim on all 8
metrics almost exactly. The residual CRC-death-timing error under screening
(issue (4)) is addressed separately in verify_screening_tauphase.py.

One quarterly 14x14 matrix T_q[age] is estimated per year of age; the cohort
applies the screening detect+treat operator at q1 of a screen year, then steps
T_q[age] four times.

Run:  python verify_screening_quarterly.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

N, EA, AA = 0, 1, 2
P1, P2, P3, P4 = 3, 4, 5, 6
C1, C2, C3, C4 = 7, 8, 9, 10
DOTH, DCRC = 11, 12
NS = 13
PSTAGE = {7: P1, 8: P2, 9: P3, 10: P4}
CSTAGE = {7: C1, 8: C2, 9: C3, 10: C4}
CLIN = (C1, C2, C3, C4)
MAXY = 100
NQ = MAXY * 4
AGE_MIN, AGE_MAX = 1, 99


def load_detect():
    z = np.load(os.path.join(RES, 'pomdp_effects.npz'), allow_pickle=True)
    return float(z['d_EA']), float(z['d_AA']), float(z['d_PC'])


def ext_state(pt):
    if pt.ever_clinical:
        return CSTAGE[int(pt.det_stage[0])]
    if pt.ca_stage:
        return PSTAGE[max(pt.ca_stage)]
    mps = pt.max_polyp_stage()
    if mps >= 5:
        return AA
    if mps >= 1:
        return EA
    return N


def run_microsim_q(eng, n, screen_ages, record_paths=False, seed=0):
    eng.rng = np.random.default_rng(seed)
    screen_ages = set(int(a) for a in screen_ages)
    paths = np.full((n, NQ + 2), -1, dtype=np.int8) if record_paths else None
    death_age = np.empty(n); death_cause = np.zeros(n, np.int8)
    ever_clin = np.zeros(n, bool); dx_stage = np.zeros(n, np.int8)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        for y in range(1, MAXY + 1):
            do_screen = y in screen_ages
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                if record_paths:
                    paths[i, t] = ext_state(pt)
                if q == 1 and do_screen and pt.alive and not pt.ever_clinical:
                    eng.colonoscopy(pt, y, 1, 'Scre')
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    dstate = DCRC if pt.death_cause == 2 else DOTH
                    if record_paths:
                        paths[i, t + 1:] = dstate
                    break
            if not pt.alive:
                break
        if pt.alive:
            death_age[i] = MAXY; death_cause[i] = 0
        else:
            death_age[i] = pt.death_time; death_cause[i] = pt.death_cause
        ever_clin[i] = pt.ever_clinical
        if pt.det_stage:
            dx_stage[i] = int(pt.det_stage[0])
        if (i + 1) % 25000 == 0:
            print(f"    microsim {i+1}/{n} ({time.time()-t0:.0f}s)")
    return metrics_micro(death_age, death_cause, ever_clin, dx_stage), paths


def metrics_micro(death_age, death_cause, ever_clin, dx_stage):
    crc = death_cause == 2
    st = np.array([np.count_nonzero(dx_stage == s) for s in (7, 8, 9, 10)], float)
    st_pct = 100 * st / st.sum() if st.sum() else st
    return {
        'avg_age_death_all': float(death_age.mean()),
        'avg_age_crc_death': float(death_age[crc].mean()) if crc.any() else float('nan'),
        'crc_deaths_100k': 1e5 * crc.mean(),
        'crc_incidence_100k': 1e5 * ever_clin.mean(),
        'stage_pct': st_pct,
    }


def estimate_Tq(paths):
    """One quarterly 14x14 matrix per year of age."""
    counts = np.zeros((MAXY + 1, NS, NS), np.int64)
    for t in range(1, NQ):
        y = (t - 1) // 4 + 1
        if y > AGE_MAX:
            break
        sf, stv = paths[:, t], paths[:, t + 1]
        for s in range(NS):
            m = sf == s
            if not m.any():
                continue
            tos = stv[m]
            for sp in range(NS):
                counts[y, s, sp] += np.count_nonzero(tos == sp)
    T = np.zeros((MAXY + 1, NS, NS))
    for y in range(MAXY + 1):
        for s in range(NS):
            tot = counts[y, s].sum()
            T[y, s] = counts[y, s] / tot if tot > 0 else np.eye(NS)[s]
        for d in (DOTH, DCRC):
            T[y, d] = 0.0; T[y, d, d] = 1.0
    return T


def screen_matrix(d_EA, d_AA, d_PC):
    S = np.eye(NS)
    S[EA, EA] = 1 - d_EA; S[EA, N] = d_EA
    S[AA, AA] = 1 - d_AA; S[AA, N] = d_AA
    for Pk, Ck in ((P1, C1), (P2, C2), (P3, C3), (P4, C4)):
        S[Pk, Pk] = 1 - d_PC; S[Pk, Ck] = d_PC
    return S


def run_cohort_q(T, screen_ages, d):
    d_EA, d_AA, d_PC = d
    S = screen_matrix(d_EA, d_AA, d_PC)
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(NS); dist[N] = 1.0
    clin_cum = np.zeros(4)
    death_oth_age = death_crc_age = 0.0
    prev_doth = prev_dcrc = 0.0
    nonclin = [s for s in range(NS) if s not in CLIN and s not in (DOTH, DCRC)]
    for age in range(AGE_MIN, MAXY + 1):
        Ty = T[min(age, AGE_MAX)]
        if age in screen_ages:
            for k, Pk in enumerate((P1, P2, P3, P4)):
                clin_cum[k] += dist[Pk] * d_PC
            dist = dist @ S
        for q in (1, 2, 3, 4):
            for k, Ck in enumerate(CLIN):
                clin_cum[k] += sum(dist[s] * Ty[s, Ck] for s in nonclin)
            nxt = dist @ Ty
            age_q = age + (q - 1) / 4.0
            death_oth_age += (nxt[DOTH] - prev_doth) * age_q
            death_crc_age += (nxt[DCRC] - prev_dcrc) * age_q
            prev_doth, prev_dcrc = nxt[DOTH], nxt[DCRC]
            dist = nxt
    living = 1.0 - dist[DOTH] - dist[DCRC]
    death_oth_age += living * MAXY
    tot_oth = dist[DOTH] + living; tot_crc = dist[DCRC]
    incid = clin_cum.sum()
    st_pct = 100 * clin_cum / incid if incid > 0 else clin_cum
    return {
        'avg_age_death_all': (death_oth_age + death_crc_age) / (tot_oth + tot_crc),
        'avg_age_crc_death': death_crc_age / tot_crc if tot_crc > 0 else float('nan'),
        'crc_deaths_100k': 1e5 * tot_crc,
        'crc_incidence_100k': 1e5 * incid,
        'stage_pct': st_pct,
    }


def print_compare(name, micro, coh):
    print(f"\n================ {name} ================")
    s_m, s_c = micro['stage_pct'], coh['stage_pct']
    rows = [
        ('Avg age at death (all)', micro['avg_age_death_all'], coh['avg_age_death_all'], '%.2f'),
        ('Avg age at CRC death',   micro['avg_age_crc_death'], coh['avg_age_crc_death'], '%.2f'),
        ('CRC deaths /100k',       micro['crc_deaths_100k'],   coh['crc_deaths_100k'],   '%.0f'),
        ('CRC incidence /100k',    micro['crc_incidence_100k'],coh['crc_incidence_100k'],'%.0f'),
        ('Stage I %', s_m[0], s_c[0], '%.1f'), ('Stage II %', s_m[1], s_c[1], '%.1f'),
        ('Stage III %', s_m[2], s_c[2], '%.1f'), ('Stage IV %', s_m[3], s_c[3], '%.1f'),
    ]
    print(f"{'metric':<24}{'microsim':>12}{'quarterly-TM':>14}{'diff':>10}")
    print('-' * 60)
    for lab, a, b, f in rows:
        print(f"{lab:<24}{f%a:>12}{f%b:>14}{b-a:>+10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20240705)
    args = ap.parse_args()
    eng = CRCEngine(build_params('CMOST13', 500, 7), rng=np.random.default_rng(0))
    d = load_detect()
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}
    micro = {}
    print(f"[1] no-screening microsim (quarterly record, n={args.n}) ...")
    micro['No screening'], paths = run_microsim_q(eng, args.n, [], True, args.seed)
    print("[2] estimating QUARTERLY transition matrices ...")
    T = estimate_Tq(paths)
    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={args.n}) ...")
        micro[name], _ = run_microsim_q(eng, args.n, sa, False, args.seed + 1)
    for name, sa in scenarios.items():
        print_compare(name, micro[name], run_cohort_q(T, sa, d))


if __name__ == '__main__':
    main()
