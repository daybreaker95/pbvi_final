"""
verify_screening_tauphase.py -- FINAL metric-complete transition-matrix cohort
vs the CMOST microsimulation, under no screening / q10y / q5y colonoscopy.

Builds an augmented, QUARTER-stepping Markov cohort from the no-screening
microsim and compares 8 clinical metrics against the microsim ground truth:
  Avg age at death (all), Avg age at CRC death, CRC deaths /100k,
  CRC incidence /100k, Stage I/II/III/IV % at diagnosis.

Two refinements over a plain 6-state annual matrix:
 (a) time-since-diagnosis PHASES: the clinical state is subdivided by tau =
     QUARTERS since diagnosis (0..TAU_MAX-1, then a 'survivor' phase). CMOST
     draws a fixed time-to-CRC-death at diagnosis (mortality_matrix), so post-
     diagnosis mortality is front-loaded (deaths within ~5y else effectively
     cured) -- non-Markovian in a plain clinical state. Adding tau restores the
     correct CRC-death TIMING/age inside the Markov cohort.
 (b) stage-specific screen detection: d_PC[stage] faithfully reproduces the
     CMOST engine cancer-detection line  rand < Colo_Detection[stage-1] AND
     reach_ok[loc]  (cancers use NO Location_ColoDetection factor).

State layout (NS states):
  0 N  1 EA  2 AA  3 P1  4 P2  5 P3  6 P4
  clinical: C_BASE + k*NCTAU + tau     k=0..3 (dx stage I..IV), tau=0..TAU_MAX
  DOTH, DCRC (absorbing)

Run:  python verify_screening_tauphase.py -n 100000
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
PSTAGE = {7: P1, 8: P2, 9: P3, 10: P4}
C_BASE = 7
TAU_MAX = 21                       # tau=0..20 active (quarters, 5y), 21 = survivor
NCTAU = TAU_MAX + 1                # 22 phases per stage
DOTH = C_BASE + 4 * NCTAU          # 95
DCRC = DOTH + 1                    # 96
NS = DCRC + 1                      # 97
MAXY = 100
NQ = MAXY * 4
AGE_MIN, AGE_MAX = 1, 99


def cidx(k, tau):
    return C_BASE + k * NCTAU + min(tau, TAU_MAX)

CLIN_ALL = set(cidx(k, tau) for k in range(4) for tau in range(NCTAU))
NONCLIN = [s for s in range(NS) if s not in CLIN_ALL and s not in (DOTH, DCRC)]


def stage_specific_dPC(eng, M=4_000_000, seed=1):
    """Per-preclinical-stage screen-detection prob, faithful to the CMOST engine
    cancer-detection line (Colo_Detection[stage-1] * P(loc >= reach))."""
    colo = eng.colo_detection
    rng = np.random.default_rng(seed)
    li = (rng.random(M) * 999).round().astype(int)
    loc = eng.location_matrix[1, li].astype(int)
    ri = (rng.random(M) * 999).round().astype(int)
    reach = eng.colo_reach_matrix[ri].astype(int)
    R = float((loc >= reach).mean())
    return np.array([colo[6] * R, colo[7] * R, colo[8] * R, colo[9] * R])


def load_adenoma_detect():
    z = np.load(os.path.join(RES, 'pomdp_effects.npz'), allow_pickle=True)
    return float(z['d_EA']), float(z['d_AA'])


def ext_state(pt, t_now):
    if pt.ever_clinical:
        k = int(pt.det_stage[0]) - 7
        # diagnosis happens inside a quarter step, so the FIRST clinical snapshot
        # (next quarter boundary) is one quarter after det_year -> shift by -1 so
        # the diagnosis-entry phase is tau=0 (consistent with the screen operator).
        tau = int(round((t_now - pt.det_year[0]) * 4)) - 1
        return cidx(k, max(0, tau))
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
                    paths[i, t] = ext_state(pt, y + (q - 1) / 4.0)
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
    """One quarterly NSxNS matrix per year of age (vectorised via bincount)."""
    counts = np.zeros((MAXY + 1, NS, NS), np.int64)
    for t in range(1, NQ):
        y = (t - 1) // 4 + 1
        if y > AGE_MAX:
            break
        sf, stv = paths[:, t].astype(np.int64), paths[:, t + 1].astype(np.int64)
        m = (sf >= 0) & (stv >= 0)
        flat = sf[m] * NS + stv[m]
        counts[y] += np.bincount(flat, minlength=NS * NS).reshape(NS, NS)
    T = np.zeros((MAXY + 1, NS, NS))
    for y in range(MAXY + 1):
        rs = counts[y].sum(axis=1)
        for s in range(NS):
            T[y, s] = counts[y, s] / rs[s] if rs[s] > 0 else np.eye(NS)[s]
        for dth in (DOTH, DCRC):
            T[y, dth] = 0.0; T[y, dth, dth] = 1.0
    return T


def screen_matrix(d_EA, d_AA, dPC):
    S = np.eye(NS)
    S[EA, EA] = 1 - d_EA; S[EA, N] = d_EA
    S[AA, AA] = 1 - d_AA; S[AA, N] = d_AA
    for k, Pk in enumerate((P1, P2, P3, P4)):
        S[Pk, Pk] = 1 - dPC[k]; S[Pk, cidx(k, 0)] = dPC[k]
    return S


def run_cohort_q(T, screen_ages, d_EA, d_AA, dPC):
    S = screen_matrix(d_EA, d_AA, dPC)
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(NS); dist[N] = 1.0
    clin_cum = np.zeros(4)
    death_oth_age = death_crc_age = 0.0
    prev_doth = prev_dcrc = 0.0
    for age in range(AGE_MIN, MAXY + 1):
        Ty = T[min(age, AGE_MAX)]
        if age in screen_ages:
            for k, Pk in enumerate((P1, P2, P3, P4)):
                clin_cum[k] += dist[Pk] * dPC[k]
            dist = dist @ S
        for q in (1, 2, 3, 4):
            for k in range(4):
                c0 = cidx(k, 0)
                clin_cum[k] += sum(dist[s] * Ty[s, c0] for s in NONCLIN)
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
    print(f"{'metric':<24}{'microsim':>12}{'a+b TM':>14}{'diff':>10}")
    print('-' * 60)
    for lab, a, b, f in rows:
        print(f"{lab:<24}{f%a:>12}{f%b:>14}{b-a:>+10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20240705)
    args = ap.parse_args()
    eng = CRCEngine(build_params('CMOST13', 500, 7), rng=np.random.default_rng(0))
    d_EA, d_AA = load_adenoma_detect()
    dPC = stage_specific_dPC(eng)
    print(f"NS={NS} states; d_EA={d_EA:.3f} d_AA={d_AA:.3f} "
          f"d_PC[I..IV]={np.round(dPC,3).tolist()}")
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}
    micro = {}
    print(f"[1] no-screening microsim (quarterly, n={args.n}) ...")
    micro['No screening'], paths = run_microsim_q(eng, args.n, [], True, args.seed)
    print("[2] estimating tau-phased QUARTERLY transition matrices ...")
    T = estimate_Tq(paths)
    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={args.n}) ...")
        micro[name], _ = run_microsim_q(eng, args.n, sa, False, args.seed + 1)
    for name, sa in scenarios.items():
        print_compare(name, micro[name], run_cohort_q(T, sa, d_EA, d_AA, dPC))


if __name__ == '__main__':
    main()
