"""
compare_matlab_benchmark.py -- extended CMOST-microsim vs tau-phase transition-
matrix comparison, targeting the metric list of the legacy MATLAB reference
file CMOST13_02072016_Results_Variable_Combined.csv.

IMPORTANT SCOPE NOTE: that CSV was generated with Polyp_Surveillance='on' and
Cancer_Surveillance='on' (83k+ follow-up colonoscopies). This script does NOT
reproduce that scenario -- surveillance is OFF here (matching every other
script in transitions/), because the transition-matrix / cohort model has no
surveillance-scheduling logic at all yet. This is a same-methodology
(no_screen / q10y / q5y, surveillance off) CMOST-direct-simulation vs
tau-phase-cohort comparison, just with a wider metric set than
verify_screening_tauphase.py, run at n=1,000,000.

Metrics deliberately OMITTED with reasons:
  - Total costs               : no cost tracking in the individual engine
  - fraction direct all/right : de-novo ("direct") cancer pathway not
                                 separately flagged in Patient records
  - sojourn time, overall time: ambiguous multi-lesion index matching risk;
                                 dwell time (unambiguous, ca_dwell) is reported
                                 instead
  - Patients died of colonoscopy / years lost to colonoscopy on the MATRIX
    side: the cohort operator has no complication-death branch (CMOST-side
    numbers ARE reported; matrix side marked N/A)
  - Average age male/female on the MATRIX side: current stratified matrices
    split by risk class, not gender (CMOST-side numbers ARE reported)

Dwell time on the matrix side is an ANALYTIC approximation: it assumes the
preclinical stage states (P1-P4) are memoryless with a single age-band-
averaged (50-70) self-transition probability, and reports the implied
GEOMETRIC distribution's median/mean/quartiles. Per the whole SCREENING_
VALIDATION.md investigation, CMOST's own dwell-time draw is likely NOT
memoryless (same family of issue as the post-diagnosis mortality timing), so
a real gap here is expected, not a bug.

Run: python compare_matlab_benchmark.py -n 1000000
"""
from __future__ import annotations
import os, sys, time, argparse, csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))

from env.params import build_params
from env.cmost_individual import CRCEngine

import verify_screening_tauphase as VST  # reuse state layout + cohort machinery

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
N, EA, AA = VST.N, VST.EA, VST.AA
P1, P2, P3, P4 = VST.P1, VST.P2, VST.P3, VST.P4
C_BASE, TAU_MAX, NCTAU = VST.C_BASE, VST.TAU_MAX, VST.NCTAU
DOTH, DCRC, NS = VST.DOTH, VST.DCRC, VST.NS
cidx, NONCLIN = VST.cidx, VST.NONCLIN
MAXY, NQ, AGE_MIN, AGE_MAX = VST.MAXY, VST.NQ, VST.AGE_MIN, VST.AGE_MAX


# ---------------------------------------------------------------------------
# extended CMOST-direct microsim
# ---------------------------------------------------------------------------
def run_microsim_extended(eng, n, screen_ages, seed):
    eng.rng = np.random.default_rng(seed)
    screen_ages = set(int(a) for a in screen_ages)
    death_age = np.empty(n); death_cause = np.zeros(n, np.int8)
    other_death_time = np.empty(n)
    gender = np.empty(n, np.int8)
    ever_clin = np.zeros(n, bool)
    dx_stage = np.zeros(n, np.int8)
    dx_route = np.zeros(n, np.int8)          # 1=screening 2=symptom
    n_screen_colo = 0
    dwell_all = []
    t0 = time.time()
    for i in range(n):
        pt = eng.new_patient()
        gender[i] = pt.gender
        for y in range(1, MAXY + 1):
            do_screen = y in screen_ages
            for q in (1, 2, 3, 4):
                if q == 1 and do_screen and pt.alive and not pt.ever_clinical:
                    was_diag = pt.ever_clinical
                    eng.colonoscopy(pt, y, 1, 'Scre')
                    n_screen_colo += 1
                    if pt.ever_clinical and not was_diag:
                        dx_route[i] = 1
                eng._step_quarter(pt, y, q)
                if pt.ever_clinical and dx_route[i] == 0:
                    dx_route[i] = 2   # became clinical this quarter via symptom path
                if not pt.alive:
                    break
            if not pt.alive:
                break
        if pt.alive:
            death_age[i] = MAXY; death_cause[i] = 0
        else:
            death_age[i] = pt.death_time; death_cause[i] = pt.death_cause
        other_death_time[i] = pt.other_death_time if pt.other_death_time < 1e17 else death_age[i]
        ever_clin[i] = pt.ever_clinical
        if pt.det_stage:
            dx_stage[i] = int(pt.det_stage[0])
        if pt.ca_dwell:
            dwell_all.extend(float(d) for d in pt.ca_dwell if d > 0)
        if (i + 1) % 100000 == 0:
            el = time.time() - t0
            print(f"    microsim {i+1}/{n} ({el:.0f}s, {1000*el/(i+1):.3f} ms/pt)")
    return dict(death_age=death_age, death_cause=death_cause,
                other_death_time=other_death_time, gender=gender,
                ever_clin=ever_clin, dx_stage=dx_stage, dx_route=dx_route,
                n_screen_colo=n_screen_colo, dwell_all=np.array(dwell_all))


def q(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float('nan')


def summarize_microsim(m, n):
    crc = m['death_cause'] == 2
    colo_d = m['death_cause'] == 3
    male = m['gender'] == 1
    ly_lost_crc = np.maximum(0.0, m['other_death_time'][crc] - m['death_age'][crc])
    ly_lost_colo = np.maximum(0.0, m['other_death_time'][colo_d] - m['death_age'][colo_d])
    n_symp_colo = int(m['n_colonoscopies_total']) - m['n_screen_colo'] if 'n_colonoscopies_total' in m else None
    dx = m['dx_stage']; route = m['dx_route']
    def stage_counts(mask):
        return np.array([np.count_nonzero((dx == s) & mask) for s in (7, 8, 9, 10)])
    sc_scr = stage_counts(route == 1)
    sc_sym = stage_counts(route == 2)
    sc_all = sc_scr + sc_sym
    dwell = m['dwell_all']
    out = {
        'n_patients': n,
        'avg_age_all': float(m['death_age'].mean()),
        'avg_age_male': float(m['death_age'][male].mean()),
        'avg_age_female': float(m['death_age'][~male].mean()),
        'screening_colo': m['n_screen_colo'],
        'symptom_colo_approx': int(np.count_nonzero(route == 2)),
        'followup_colo': 0,
        'crc_deaths': int(crc.sum()),
        'years_lost_crc': float(ly_lost_crc.sum()),
        'colo_deaths': int(colo_d.sum()),
        'years_lost_colo': float(ly_lost_colo.sum()),
        'all_ca': int(m['ever_clin'].sum()),
        'detected_screening': int((route == 1).sum()),
        'detected_symptoms': int((route == 2).sum()),
        'detected_surveillance': 0,
        'detected_baseline': 0,
        'stage_screen_pct': 100 * sc_scr / sc_scr.sum() if sc_scr.sum() else sc_scr * 0.0,
        'stage_symp_pct': 100 * sc_sym / sc_sym.sum() if sc_sym.sum() else sc_sym * 0.0,
        'stage_all_pct': 100 * sc_all / sc_all.sum() if sc_all.sum() else sc_all * 0.0,
        'dwell_median': q(dwell, 50), 'dwell_mean': float(dwell.mean()) if len(dwell) else float('nan'),
        'dwell_lq': q(dwell, 25), 'dwell_uq': q(dwell, 75),
    }
    return out


# ---------------------------------------------------------------------------
# matrix / tau-phase cohort side (analytic), extended
# ---------------------------------------------------------------------------
def run_cohort_extended(T, screen_ages, d_EA, d_AA, dPC):
    S = VST.screen_matrix(d_EA, d_AA, dPC)
    screen_ages = set(int(a) for a in screen_ages)
    dist = np.zeros(NS); dist[N] = 1.0
    clin_cum_scr = np.zeros(4); clin_cum_symp = np.zeros(4)
    death_oth_age = death_crc_age = 0.0
    prev_doth = prev_dcrc = 0.0
    screen_colo_mass = 0.0
    clin_mass_idx = [cidx(k, tau) for k in range(4) for tau in range(NCTAU)]
    for age in range(AGE_MIN, MAXY + 1):
        Ty = T[min(age, AGE_MAX)]
        if age in screen_ages:
            # only alive AND not-yet-clinically-diagnosed patients are screened
            # (matches the `not pt.ever_clinical` gate used everywhere else)
            eligible_mass = dist.sum() - dist[DOTH] - dist[DCRC] - dist[clin_mass_idx].sum()
            screen_colo_mass += eligible_mass
            for k, Pk in enumerate((P1, P2, P3, P4)):
                clin_cum_scr[k] += dist[Pk] * dPC[k]
            dist = dist @ S
        for qq in (1, 2, 3, 4):
            for k in range(4):
                c0 = cidx(k, 0)
                clin_cum_symp[k] += sum(dist[s] * Ty[s, c0] for s in NONCLIN)
            nxt = dist @ Ty
            age_q = age + (qq - 1) / 4.0
            death_oth_age += (nxt[DOTH] - prev_doth) * age_q
            death_crc_age += (nxt[DCRC] - prev_dcrc) * age_q
            prev_doth, prev_dcrc = nxt[DOTH], nxt[DCRC]
            dist = nxt
    living = 1.0 - dist[DOTH] - dist[DCRC]
    death_oth_age += living * MAXY
    tot_oth = dist[DOTH] + living; tot_crc = dist[DCRC]
    incid_scr, incid_symp = clin_cum_scr.sum(), clin_cum_symp.sum()
    incid_all = incid_scr + incid_symp
    st_scr = 100 * clin_cum_scr / incid_scr if incid_scr > 0 else clin_cum_scr
    st_symp = 100 * clin_cum_symp / incid_symp if incid_symp > 0 else clin_cum_symp
    st_all = 100 * (clin_cum_scr + clin_cum_symp) / incid_all if incid_all > 0 else clin_cum_scr
    return {
        'avg_age_all': (death_oth_age + death_crc_age) / (tot_oth + tot_crc),
        'avg_age_crc_death': death_crc_age / tot_crc if tot_crc > 0 else float('nan'),
        'crc_deaths_100k': 1e5 * tot_crc,
        'screening_colo_100k': 1e5 * screen_colo_mass,
        'detected_screening_100k': 1e5 * incid_scr,
        'detected_symptoms_100k': 1e5 * incid_symp,
        'all_ca_100k': 1e5 * incid_all,
        'stage_screen_pct': st_scr, 'stage_symp_pct': st_symp, 'stage_all_pct': st_all,
    }


def analytic_dwell_geometric(T, ages_band=(50, 70)):
    """CMOST's ca_dwell = time spent as adenoma (EA/AA) before becoming cancer.
    Matrix-side analog: expected/approx-quantile absorption time out of the
    {EA,AA} transient sub-chain (age-band-averaged), starting from EA (new
    adenomas start at stage 1 = EA). Uses the fundamental matrix (I-Q)^-1 for
    the MEAN (exact, given the averaged Q), and an effective-geometric
    approximation (matched to that mean) for median/quartiles -- shape is
    still an approximation, but the target quantity (adenoma dwell, not
    preclinical-cancer dwell) and the mean are now correct.
    NOTE: this is unconditional sojourn in {EA,AA} (regression to Normal OR
    progression to cancer both end it), whereas CMOST's ca_dwell conditions on
    those that actually progressed to cancer -- a residual definitional gap.
    """
    lo, hi = ages_band
    idx = [a for a in range(lo, hi) if AGE_MIN <= a <= AGE_MAX]
    Qsum = np.zeros((2, 2))
    for a in idx:
        Ty = T[a - AGE_MIN]
        Qsum += np.array([[Ty[EA, EA], Ty[EA, AA]], [Ty[AA, EA], Ty[AA, AA]]])
    Qm = Qsum / len(idx)
    fund = np.linalg.inv(np.eye(2) - Qm)
    mean_q = float(fund[0].sum())            # expected quarters, starting in EA
    p_eff = min(max(1 - 1.0 / mean_q, 1e-6), 1 - 1e-9) if mean_q > 0 else 0.0
    def qtr(pct):
        return np.log(1 - pct) / np.log(p_eff)
    median_q, lq_q, uq_q = qtr(0.50), qtr(0.25), qtr(0.75)
    return dict(dwell_median=median_q / 4.0, dwell_mean=mean_q / 4.0,
                dwell_lq=lq_q / 4.0, dwell_uq=uq_q / 4.0, p_stay=p_eff)


# ---------------------------------------------------------------------------
def print_row(label, cm, mx, fmt='%.2f', unit=''):
    cm_s = fmt % cm if cm is not None and not (isinstance(cm, float) and np.isnan(cm)) else 'n/a'
    mx_s = fmt % mx if mx is not None and not (isinstance(mx, float) and np.isnan(mx)) else 'n/a'
    diff = ''
    if isinstance(cm, (int, float)) and isinstance(mx, (int, float)) and cm not in (0, None) and mx is not None:
        try:
            diff = f"{100*(mx-cm)/cm:+.1f}%" if abs(cm) > 1e-9 else ''
        except Exception:
            diff = ''
    print(f"{label:<28}{cm_s+unit:>16}{mx_s+unit:>16}{diff:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=1000000)
    ap.add_argument('--seed', type=int, default=20240705)
    args = ap.parse_args()
    n = args.n

    eng = CRCEngine(build_params('CMOST13', 500, 7), rng=np.random.default_rng(0))
    eng.deterministic_natural_death = True   # populate other_death_time for paired LYG/years-lost accounting
    d_EA, d_AA = VST.load_adenoma_detect()
    dPC = VST.stage_specific_dPC(eng)
    print(f"NS={NS}  n={n}  d_EA={d_EA:.3f} d_AA={d_AA:.3f} d_PC={np.round(dPC,3).tolist()}")

    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}

    print(f"[1] no-screening microsim (n={n}) -- for tau-phase matrix estimation ...")
    micro0 = run_microsim_extended(eng, n, [], seed=args.seed)
    print("[2] re-running no-screening WITH quarterly path recording for T estimation ...")
    eng.rng = np.random.default_rng(args.seed)
    _, paths = VST.run_microsim_q(eng, n, [], record_paths=True, seed=args.seed)
    T = VST.estimate_Tq(paths)
    del paths

    micro_summ = {'No screening': summarize_microsim(micro0, n)}
    cohort_summ = {'No screening': run_cohort_extended(T, [], d_EA, d_AA, dPC)}

    for name, sa in scenarios.items():
        if name == 'No screening':
            continue
        print(f"[3] microsim {name} (n={n}) ...")
        m = run_microsim_extended(eng, n, sa, seed=args.seed + 1)
        micro_summ[name] = summarize_microsim(m, n)
        cohort_summ[name] = run_cohort_extended(T, sa, d_EA, d_AA, dPC)

    dwell_geo = analytic_dwell_geometric(T)
    print(f"\n(matrix-side dwell = analytic geometric approx, age 50-70 avg p_stay={dwell_geo['p_stay']:.4f})")

    rows_out = []
    for name in scenarios:
        cm, mx = micro_summ[name], cohort_summ[name]
        print(f"\n================ {name} ================")
        print(f"{'metric':<28}{'CMOST (n='+str(n)+')':>16}{'tau-phase matrix':>16}{'diff':>10}")
        print('-' * 72)
        pairs = [
            ('Avg age at death (all)', cm['avg_age_all'], mx['avg_age_all'], '%.2f', ''),
            ('Avg age male', cm['avg_age_male'], None, '%.2f', ''),
            ('Avg age female', cm['avg_age_female'], None, '%.2f', ''),
            ('Screening colo /100k', 1e5*cm['screening_colo']/n, mx['screening_colo_100k'], '%.0f', ''),
            ('Symptom colo /100k (approx)', 1e5*cm['symptom_colo_approx']/n, mx['detected_symptoms_100k'], '%.0f', ''),
            ('Follow-up colo /100k', cm['followup_colo'], 0.0, '%.0f', ''),
            ('CRC deaths /100k', 1e5*cm['crc_deaths']/n, mx['crc_deaths_100k'], '%.0f', ''),
            ('Years lost to CRC /100k', 1e5*cm['years_lost_crc']/n, None, '%.0f', ''),
            ('Colonoscopy deaths /100k', 1e5*cm['colo_deaths']/n, None, '%.1f', ''),
            ('Years lost to colo /100k', 1e5*cm['years_lost_colo']/n, None, '%.1f', ''),
            ('All CRC cases /100k', 1e5*cm['all_ca']/n, mx['all_ca_100k'], '%.0f', ''),
            ('Detected: screening /100k', 1e5*cm['detected_screening']/n, mx['detected_screening_100k'], '%.0f', ''),
            ('Detected: symptoms /100k', 1e5*cm['detected_symptoms']/n, mx['detected_symptoms_100k'], '%.0f', ''),
            ('Stage I % (screening)', cm['stage_screen_pct'][0], mx['stage_screen_pct'][0], '%.1f', ''),
            ('Stage II % (screening)', cm['stage_screen_pct'][1], mx['stage_screen_pct'][1], '%.1f', ''),
            ('Stage III % (screening)', cm['stage_screen_pct'][2], mx['stage_screen_pct'][2], '%.1f', ''),
            ('Stage IV % (screening)', cm['stage_screen_pct'][3], mx['stage_screen_pct'][3], '%.1f', ''),
            ('Stage I % (symptoms)', cm['stage_symp_pct'][0], mx['stage_symp_pct'][0], '%.1f', ''),
            ('Stage II % (symptoms)', cm['stage_symp_pct'][1], mx['stage_symp_pct'][1], '%.1f', ''),
            ('Stage III % (symptoms)', cm['stage_symp_pct'][2], mx['stage_symp_pct'][2], '%.1f', ''),
            ('Stage IV % (symptoms)', cm['stage_symp_pct'][3], mx['stage_symp_pct'][3], '%.1f', ''),
            ('Stage I % (all)', cm['stage_all_pct'][0], mx['stage_all_pct'][0], '%.1f', ''),
            ('Stage II % (all)', cm['stage_all_pct'][1], mx['stage_all_pct'][1], '%.1f', ''),
            ('Stage III % (all)', cm['stage_all_pct'][2], mx['stage_all_pct'][2], '%.1f', ''),
            ('Stage IV % (all)', cm['stage_all_pct'][3], mx['stage_all_pct'][3], '%.1f', ''),
            ('Dwell time median (yr)', cm['dwell_median'], dwell_geo['dwell_median'], '%.2f', ''),
            ('Dwell time mean (yr)', cm['dwell_mean'], dwell_geo['dwell_mean'], '%.2f', ''),
            ('Dwell time LQ (yr)', cm['dwell_lq'], dwell_geo['dwell_lq'], '%.2f', ''),
            ('Dwell time UQ (yr)', cm['dwell_uq'], dwell_geo['dwell_uq'], '%.2f', ''),
        ]
        for label, cval, mval, fmt, unit in pairs:
            print_row(label, cval, mval, fmt, unit)
            rows_out.append({'scenario': name, 'metric': label, 'CMOST': cval, 'matrix': mval})

    out_csv = os.path.join(RES, 'matlab_benchmark_comparison.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'metric', 'CMOST', 'matrix'])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out_csv}")


if __name__ == '__main__':
    main()
