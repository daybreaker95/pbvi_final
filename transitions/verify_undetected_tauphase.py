"""
verify_undetected_tauphase.py
==============================
Extends verify_screening_tauphase.py's already-validated tau-phase method to
the PRE-diagnosis (undetected) side of the natural history, to test the
hypothesis that our memoryless 9-state d_symp[age][stage] loses the same kind
of duration-since-event information that the original tau-phase fix restored
for POST-diagnosis mortality timing.

Root cause recap: in env/cmost_individual.py, at cancer onset (_spawn_cancer)
CMOST draws BOTH the symptom-presentation time (symptime = onset_time +
sojourn, sojourn in [0.25, 6.25]y) AND the I->II->III->IV stage-progression
schedule (ca_tstage_I/II/III) as ONE-TIME deterministic draws keyed off the
same onset clock. So "probability of symptom detection this year" is not a
function of current stage alone -- it is front-loaded/back-loaded depending
on how long the focus has already been undetected, exactly the kind of
non-Markovian timing the tau-phase trick already fixed for det_year-based
CRC-death timing (build_T_detected_from_tauphase.py).

This script adds a SECOND tau axis -- tau_u = quarters since ca_year[f] (the
onset of the currently most-advanced undetected focus) -- alongside the
EXISTING detected-side tau (quarters since det_year[0]), producing an even
more refined state layout:
  0 N  1 EA  2 AA
  undetected clinical: U_BASE + k*NCTAU_U + tau_u     k=0..3 (stage I..IV)
  detected  clinical:  D_BASE + k*NCTAU_D + tau_d      k=0..3 (stage I..IV)
  DOTH, DCRC (absorbing)

This is a VALIDATION/CALIBRATION tool only -- it is never wired into
pomdp/model_v2.py's actual 13-state POMDP (which stays 13-dimensional for
FiVI training). Its only purpose is to test, and if confirmed, to eventually
recalibrate the existing simple d_symp[age][stage] table used by the 13-state
model -- the same relationship build_T_detected_from_tauphase.py already has
to verify_screening_tauphase.py for the post-diagnosis side.

Run:  python verify_undetected_tauphase.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
import verify_screening_tauphase as VST

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

N, EA, AA = 0, 1, 2

TAU_U_MAX = 28                      # undetected-tau: quarters since cancer onset
                                     # (sojourn is drawn in [0.25, 6.25]y = [1,25]q,
                                     # so 28q/7y comfortably covers the full range)
NCTAU_U = TAU_U_MAX + 1             # 29 phases per undetected stage

TAU_D_MAX = VST.TAU_MAX             # 21 (unchanged, already-validated detected side)
NCTAU_D = VST.NCTAU                 # 22

U_BASE = 3
D_BASE = U_BASE + 4 * NCTAU_U
DOTH = D_BASE + 4 * NCTAU_D
DCRC = DOTH + 1
NS = DCRC + 1


def cidx_u(k, tau):
    return U_BASE + k * NCTAU_U + min(max(tau, 0), TAU_U_MAX)


def cidx_d(k, tau):
    return D_BASE + k * NCTAU_D + min(max(tau, 0), TAU_D_MAX)


CLIN_ALL = set(cidx_d(k, tau) for k in range(4) for tau in range(NCTAU_D))
UNDET_ALL = set(cidx_u(k, tau) for k in range(4) for tau in range(NCTAU_U))
NONCLIN = [s for s in range(NS) if s not in CLIN_ALL and s not in (DOTH, DCRC)]
# NONCLIN = {N, EA, AA} u UNDET_ALL -- every pre-diagnosis state; a landing in
# cidx_d(k, 0) from any of these is, by construction, a fresh diagnosis.

MAXY, NQ, AGE_MIN, AGE_MAX = VST.MAXY, VST.NQ, VST.AGE_MIN, VST.AGE_MAX


def ext_state(pt, t_now):
    """Same convention as verify_screening_tauphase.ext_state, but the
    undetected-cancer branch now also carries tau_u = quarters since the
    onset of the most-advanced active focus (mirrors PSTAGE[max(ca_stage)]'s
    existing "collapse to worst focus" choice -- not a new approximation)."""
    if pt.ever_clinical:
        k = int(pt.det_stage[0]) - 7
        tau_d = int(round((t_now - pt.det_year[0]) * 4)) - 1
        return cidx_d(k, max(0, tau_d))
    if pt.ca_stage:
        stage = max(pt.ca_stage)
        f = pt.ca_stage.index(stage)
        tau_u = int(round((t_now - pt.ca_year[f]) * 4))
        return cidx_u(stage - 7, tau_u)
    mps = pt.max_polyp_stage()
    if mps >= 5:
        return AA
    if mps >= 1:
        return EA
    return N


def run_microsim_q(eng, n, screen_ages, record_paths=False, seed=0):
    """Identical to VST.run_microsim_q except it calls the extended
    ext_state() above (so paths carry the new, larger NS)."""
    eng.rng = np.random.default_rng(seed)
    screen_ages = set(int(a) for a in screen_ages)
    paths = np.full((n, NQ + 2), -1, dtype=np.int16) if record_paths else None
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
    return VST.metrics_micro(death_age, death_cause, ever_clin, dx_stage), paths


def estimate_Tq(paths):
    """Same bincount tally as VST.estimate_Tq, just against the bigger NS."""
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
    for k in range(4):
        for tau in range(NCTAU_U):
            s_idx = cidx_u(k, tau)
            S[s_idx, s_idx] = 1 - dPC[k]
            S[s_idx, cidx_d(k, 0)] = dPC[k]
    return S


def run_cohort_q(T, screen_ages, d_EA, d_AA, dPC):
    S = screen_matrix(d_EA, d_AA, dPC)
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(NS); dist[N] = 1.0
    clin_cum = np.zeros(4)
    death_oth_age = death_crc_age = 0.0
    prev_doth = prev_dcrc = 0.0
    u_idx = [[cidx_u(k, tau) for tau in range(NCTAU_U)] for k in range(4)]
    for age in range(AGE_MIN, MAXY + 1):
        Ty = T[min(age, AGE_MAX)]
        if age in screen_ages:
            for k in range(4):
                clin_cum[k] += dist[u_idx[k]].sum() * dPC[k]
            dist = dist @ S
        for q in (1, 2, 3, 4):
            for k in range(4):
                c0 = cidx_d(k, 0)
                clin_cum[k] += dist[NONCLIN] @ Ty[NONCLIN, c0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=20240705)
    args = ap.parse_args()
    eng = CRCEngine(build_params('CMOST13', 500, 7), rng=np.random.default_rng(0))
    d_EA, d_AA = VST.load_adenoma_detect()
    dPC = VST.stage_specific_dPC(eng)
    print(f"NS={NS} states (undetected tau: {NCTAU_U}/stage, detected tau: {NCTAU_D}/stage); "
          f"d_EA={d_EA:.3f} d_AA={d_AA:.3f} d_PC[I..IV]={np.round(dPC,3).tolist()}")
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}
    micro = {}
    print(f"[1] no-screening microsim (quarterly, n={args.n}) ...")
    micro['No screening'], paths = run_microsim_q(eng, args.n, [], True, args.seed)
    print("[2] estimating undetected+detected tau-phased QUARTERLY transition matrices ...")
    T = estimate_Tq(paths)
    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={args.n}) ...")
        micro[name], _ = run_microsim_q(eng, args.n, sa, False, args.seed + 1)
    for name, sa in scenarios.items():
        VST.print_compare(name, micro[name], run_cohort_q(T, sa, d_EA, d_AA, dPC))


if __name__ == '__main__':
    main()
