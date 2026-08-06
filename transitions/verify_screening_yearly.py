"""
verify_screening_yearly.py -- ANNUAL (1-year) transition-matrix cohort vs the
CMOST microsim, under no screening / q10y / q5y colonoscopy.

This is the BASELINE that exposes the annual time-discretization artifacts:
 (1) CRC incidence is systematically UNDERCOUNTED, and
 (3) Stage IV share is UNDERCOUNTED,
because cancers that arise, surface (are diagnosed) and kill within a single
year are never seen in the clinical state at a year boundary.  A snapshot
diagnostic proves the cohort exactly reproduces the microsim's annual-snapshot
incidence, i.e. the whole gap vs the (event-based) microsim truth is discretization.

14-state model:
  0 N  1 EA  2 AA  3 P1  4 P2  5 P3  6 P4 (preclinical stage I..IV)
  7 C1 8 C2 9 C3 10 C4 (clinical dx by stage)  11 OtherDead  12 CRCDead

Compare with verify_screening_quarterly.py (fixes 1 & 3) and
verify_screening_tauphase.py (also fixes CRC-death timing).

Run:  python verify_screening_yearly.py -n 100000
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
AGE_MIN, AGE_MAX = 1, 99
MAXY = 100


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


def run_microsim(eng, n, screen_ages, record_paths=False, seed=0):
    eng.rng = np.random.default_rng(seed)
    screen_ages = set(int(a) for a in screen_ages)
    paths = np.full((n, MAXY + 1), -1, dtype=np.int8) if record_paths else None
    death_age = np.empty(n); death_cause = np.zeros(n, np.int8)
    ever_clin = np.zeros(n, bool); dx_stage = np.zeros(n, np.int8)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        dstate = DOTH
        for y in range(1, MAXY + 1):
            if record_paths:
                paths[i, y] = ext_state(pt)
            eng.step_year(pt, y, screen=(y in screen_ages))
            if not pt.alive:
                dstate = DCRC if pt.death_cause == 2 else DOTH
                if record_paths:
                    paths[i, y + 1:MAXY + 1] = dstate
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
    return metrics_from_micro(death_age, death_cause, ever_clin, dx_stage), paths


def metrics_from_micro(death_age, death_cause, ever_clin, dx_stage):
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


def estimate_T(paths):
    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    A = len(ages)
    counts = np.zeros((A, NS, NS), dtype=np.int64)
    for ai, age in enumerate(ages):
        sf, stv = paths[:, age], paths[:, age + 1]
        for s in range(NS):
            m = sf == s
            if not m.any():
                continue
            tos = stv[m]
            for sp in range(NS):
                counts[ai, s, sp] = np.count_nonzero(tos == sp)
    T = np.zeros((A, NS, NS))
    for ai in range(A):
        for s in range(NS):
            tot = counts[ai, s].sum()
            T[ai, s] = counts[ai, s] / tot if tot > 0 else np.eye(NS)[s]
    for d in (DOTH, DCRC):
        T[:, d, :] = 0.0; T[:, d, d] = 1.0
    return ages, T


def screen_matrix(d_EA, d_AA, d_PC):
    S = np.eye(NS)
    S[EA, EA] = 1 - d_EA; S[EA, N] = d_EA
    S[AA, AA] = 1 - d_AA; S[AA, N] = d_AA
    for Pk, Ck in ((P1, C1), (P2, C2), (P3, C3), (P4, C4)):
        S[Pk, Pk] = 1 - d_PC; S[Pk, Ck] = d_PC
    return S


def run_cohort(ages, T, screen_ages, d):
    d_EA, d_AA, d_PC = d
    S = screen_matrix(d_EA, d_AA, d_PC)
    screen_ages = set(int(a) for a in screen_ages)
    Amap = {int(a): i for i, a in enumerate(ages)}
    dist = np.zeros(NS); dist[N] = 1.0
    clin_cum = np.zeros(4)
    death_oth_age = death_crc_age = 0.0
    prev_doth = prev_dcrc = 0.0
    clin_idx = [C1, C2, C3, C4]
    for age in range(AGE_MIN, AGE_MAX + 1):
        Ti = T[Amap.get(age, len(ages) - 1)]
        dwork = dist.copy()
        if age in screen_ages:
            new = dwork @ S
            for k, Pk in enumerate((P1, P2, P3, P4)):
                clin_cum[k] += dwork[Pk] * d_PC
            dwork = new
        nxt = dwork @ Ti
        for k, Ck in enumerate(clin_idx):
            inflow = 0.0
            for s in range(NS):
                if s in clin_idx or s in (DOTH, DCRC):
                    continue
                inflow += dwork[s] * Ti[s, Ck]
            clin_cum[k] += inflow
        death_oth_age += (nxt[DOTH] - prev_doth) * (age + 0.5)
        death_crc_age += (nxt[DCRC] - prev_dcrc) * (age + 0.5)
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
    print(f"{'metric':<24}{'microsim':>12}{'annual-TM':>14}{'diff':>10}")
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
    print(f"detection d_EA={d[0]:.3f} d_AA={d[1]:.3f} d_PC={d[2]:.3f}")
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}
    micro = {}
    print(f"\n[1] no-screening microsim (n={args.n}) ...")
    micro['No screening'], paths = run_microsim(eng, args.n, [], True, args.seed)
    print("[2] estimating 14-state ANNUAL transition matrices ...")
    ages, T = estimate_T(paths)

    # DIAGNOSTIC: annual-snapshot vs event-based incidence proves discretization cause
    clin_codes = (C1, C2, C3, C4)
    snap_ever = np.zeros(args.n, dtype=bool)
    snap_stage = np.zeros(args.n, dtype=np.int8)
    for i in range(args.n):
        row = paths[i]
        seen = np.isin(row, clin_codes)
        if seen.any():
            snap_ever[i] = True
            snap_stage[i] = 7 + clin_codes.index(int(row[np.argmax(seen)]))
    ss = np.array([np.count_nonzero(snap_stage == s) for s in (7, 8, 9, 10)], float)
    ev = micro['No screening']['crc_incidence_100k']; sn = 1e5 * snap_ever.mean()
    print("\n--- DIAGNOSTIC (no screening): root cause of incidence undercount ---")
    print(f"    event-based ever-clinical (microsim truth): {ev:.0f} /100k")
    print(f"    annual-snapshot ever-clinical             : {sn:.0f} /100k (should ~= cohort)")
    print(f"    incidence lost to within-year surface->die: {ev - sn:.0f} /100k")
    print(f"    annual-snapshot stage I/II/III/IV %       : "
          + " / ".join(f"{100*x/ss.sum():.1f}" for x in ss))

    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={args.n}) ...")
        micro[name], _ = run_microsim(eng, args.n, sa, False, args.seed + 1)
    for name, sa in scenarios.items():
        print_compare(name, micro[name], run_cohort(ages, T, sa, d))


if __name__ == '__main__':
    main()
