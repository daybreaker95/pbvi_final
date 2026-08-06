"""
estimate_transitions_9state.py
===============================
NEW, separate estimation for the 9-state (full I-IV cancer stage) layout with
"detected" as CONTEXT rather than a folded-in state. Does NOT touch/overwrite
transitions_cmost13.npz or transitions_stratified.npz (the currently-active
7-state pipeline) -- writes to transitions_9state.npz instead.

Two pieces are estimated:

  T_undetected[age] (9x9, quarterly-composed annual): severity dynamics
    (Normal -> ... -> Ca IV, + Other Death) while NOT yet diagnosed. Tallied
    ONLY over quarters strictly before a patient's first diagnosis (self-
    presentation truncates their contribution here) -- this is the disease's
    natural progression, independent of detection.

  d_symp[age][stage] (4,): per-quarter-composed-to-annual probability that an
    undetected Ca-stage patient self-presents (becomes DIAGNOSED) THIS year,
    i.e. Tn_undetected's excluded "exit" mass -- used exactly like the old
    Tn[ECA,DIAG] term in pomdp/model.py's WAIT reward, just per-stage now.
    Detection always captures the patient at whatever stage state9.py reports
    that quarter (matches how CMOST's own det_stage[0] works -- no extra
    approximation beyond what the engine already does).

CRC_DEATH is intentionally unreachable in T_undetected: in this engine CRC
death always follows a prior diagnosis (mortality_matrix / det_morttime is
only set once det_stage is populated), so a fresh, never-diagnosed patient
cannot transition directly to CRC_DEATH.

Run: python estimate_transitions_9state.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
from env.state9 import clinical_state9, N_STATES9, STATE9_NAMES, CA_I, CA_IV, OTHER_DEATH, CRC_DEATH

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
AGE_MIN, AGE_MAX = 1, 99
MAXY = 100
NQ = MAXY * 4
CA_STAGES = (CA_I, CA_I + 1, CA_I + 2, CA_I + 3)   # CA_I..CA_IV


def simulate(n, seed, settings='CMOST13'):
    """Quarterly no-screening simulation. Returns:
      qstate  (n, NQ+2) int8  -- state9 index each quarter (post first-detect
                                 quarters are frozen at CA_x/CRC_DEATH/OTHER_DEATH
                                 as usual, just not used past first detection)
      first_detect_q (n,) int32 -- quarter index of first diagnosis, or -1
    """
    params = build_params(settings, n_patients=20000, seed=seed)
    eng = CRCEngine(params, rng=np.random.default_rng(seed + 1))
    qstate = np.full((n, NQ + 2), OTHER_DEATH, dtype=np.int8)
    first_detect_q = np.full(n, -1, dtype=np.int32)
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        was_detected = False
        for y in range(1, MAXY + 1):
            for q in (1, 2, 3, 4):
                t = (y - 1) * 4 + q
                s, det = clinical_state9(pt)
                qstate[i, t] = s
                if det and not was_detected:
                    first_detect_q[i] = t
                    was_detected = True
                if not pt.alive:
                    break
                eng._step_quarter(pt, y, q)
                if not pt.alive:
                    s2, det2 = clinical_state9(pt)
                    qstate[i, t + 1] = s2
                    if det2 and not was_detected:
                        first_detect_q[i] = t + 1
                        was_detected = True
                    break
            if not pt.alive:
                break
        if (i + 1) % 25000 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{n}  ({el:.0f}s, {1000*el/(i+1):.3f} ms/pt)")
    print(f"  done {n} in {time.time()-t0:.0f}s")
    return qstate, first_detect_q


def tally(qstate, first_detect_q):
    """Quarterly counts:
       counts_stay[age_y, s, s']   -- stayed-undetected transitions
       counts_detect[age_y, s]     -- (for s in CA_I..CA_IV) became-detected this quarter
    """
    n = qstate.shape[0]
    counts_stay = np.zeros((MAXY + 1, N_STATES9, N_STATES9), dtype=np.int64)
    counts_detect = np.zeros((MAXY + 1, N_STATES9), dtype=np.int64)
    n_row = np.zeros((MAXY + 1, N_STATES9), dtype=np.int64)
    for i in range(n):
        fdq = first_detect_q[i]
        # loop t = 1 .. fdq-1 inclusive (so t+1 can reach fdq, the detection quarter)
        last_t = fdq if fdq > 0 else NQ
        for t in range(1, min(last_t, NQ)):
            y = (t - 1) // 4 + 1
            s = qstate[i, t]
            sp = qstate[i, t + 1]
            n_row[y, s] += 1
            if fdq > 0 and (t + 1) == fdq:
                counts_detect[y, s] += 1
            else:
                counts_stay[y, s, sp] += 1
    return counts_stay, counts_detect, n_row


def normalise_quarter(counts_stay, counts_detect, n_row):
    A = MAXY + 1
    Pq_stay = np.zeros((A, N_STATES9, N_STATES9))
    dq = np.zeros((A, N_STATES9))
    for y in range(A):
        for s in range(N_STATES9):
            tot = n_row[y, s]
            if tot > 0:
                Pq_stay[y, s] = counts_stay[y, s] / tot
                dq[y, s] = counts_detect[y, s] / tot
            else:
                Pq_stay[y, s, s] = 1.0
        Pq_stay[y, OTHER_DEATH] = 0.0
        Pq_stay[y, OTHER_DEATH, OTHER_DEATH] = 1.0
        Pq_stay[y, CRC_DEATH] = 0.0
        Pq_stay[y, CRC_DEATH, CRC_DEATH] = 1.0
    return Pq_stay, dq


def compose_annual(Pq_stay, dq):
    """P_year = P_quarter_full^4 restricted back to the 'stayed undetected'
    block; d_year[s, k] = annual probability of exiting to "detected AT
    STAGE k" starting from state s.

    Uses FOUR separate exit sinks (one per Ca stage), not one combined sink:
    a single shared sink would attribute every mid-year detection to the
    stage the patient STARTED the year in, even if they silently progressed
    to a later stage before symptoms fired -- the same family of annual-
    discretization artifact as SCREENING_VALIDATION.md's incidence/stage
    undercount. Separate sinks let the matrix power correctly route the
    exit to whichever stage the patient is actually AT when detection fires.
    """
    A = MAXY + 1
    NEXIT = 4
    NSX = N_STATES9 + NEXIT
    EXIT0 = N_STATES9   # exit sinks are EXIT0..EXIT0+3, one per Ca stage
    P_year = np.zeros((A, N_STATES9, N_STATES9))
    d_year = np.zeros((A, N_STATES9, NEXIT))
    for y in range(A):
        Tx = np.zeros((NSX, NSX))
        Tx[:N_STATES9, :N_STATES9] = Pq_stay[y]
        for k, s in enumerate(CA_STAGES):
            Tx[s, EXIT0 + k] = dq[y, s]
        rs = Tx[:N_STATES9].sum(axis=1)
        for s in range(N_STATES9):
            if rs[s] < 1.0 - 1e-9:
                Tx[s, s] += 1.0 - rs[s]
        for k in range(NEXIT):
            Tx[EXIT0 + k, EXIT0 + k] = 1.0
        Tx4 = np.linalg.matrix_power(Tx, 4)
        P_year[y] = Tx4[:N_STATES9, :N_STATES9]
        d_year[y] = Tx4[:N_STATES9, EXIT0:EXIT0 + NEXIT]
    return P_year, d_year


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=13579)
    args = ap.parse_args()

    print(f"Simulating n={args.n} (no screening, quarterly, 9-state undetected world) ...")
    qstate, fdq = simulate(args.n, args.seed)
    frac_ever_detected = float((fdq >= 0).mean())
    print(f"  fraction ever diagnosed (no screening): {frac_ever_detected:.4f}")

    print("Tallying quarterly stay/detect counts ...")
    counts_stay, counts_detect, n_row = tally(qstate, fdq)
    Pq_stay, dq = normalise_quarter(counts_stay, counts_detect, n_row)

    print("Composing to annual (quarterly^4, with detection as an exit sink) ...")
    P_year, d_year = compose_annual(Pq_stay, dq)

    ages = np.arange(AGE_MIN, AGE_MAX + 1)
    P_year_a = P_year[ages]
    d_year_a = d_year[ages]

    out = os.path.join(RES, 'transitions_9state.npz')
    np.savez_compressed(
        out, ages=ages, P_undetected=P_year_a, d_symp=d_year_a,
        # quarterly (uncomposed) matrices too -- annual composition mis-
        # attributes "exit stage" when progression happens mid-year before
        # detection (a same-year onset->detection artifact, same family of
        # issue as SCREENING_VALIDATION.md's annual-discretization finding).
        # Cohort propagation that cares about stage-at-detection should step
        # quarter-by-quarter using Pq_stay/dq_quarter instead.
        Pq_stay=Pq_stay[ages], dq_quarter=dq[ages],
        state_names=np.array(STATE9_NAMES), n_patients=args.n,
        frac_ever_detected_no_screen=frac_ever_detected,
    )
    print(f"Saved {out}")

    print("\nPreview (age 55, 65): P_undetected diag + d_symp (self-present, by exit stage) for Ca states")
    for age in (55, 65):
        ai = age - AGE_MIN
        for si, name in enumerate(['Ca I', 'Ca II', 'Ca III', 'Ca IV']):
            s = CA_STAGES[si]
            total_exit = d_year_a[ai, s].sum()
            same_stage = d_year_a[ai, s, si]
            print(f"  age {age} {name:7s}: stay-self={100*P_year_a[ai,s,s]:5.1f}%  "
                  f"d_symp total={100*total_exit:5.2f}% (same-stage={100*same_stage:5.2f}%)  "
                  f"row-sum(stay)+total_exit={100*(P_year_a[ai,s].sum()+total_exit):5.1f}%")


if __name__ == '__main__':
    main()
