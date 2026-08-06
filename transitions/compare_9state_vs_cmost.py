"""
compare_9state_vs_cmost.py
===========================
CMOST-direct simulation vs the 9-state cohort model (env/state9.py,
transitions_9state.npz + T_detected_tauphase.npz -- i.e. exactly what
pomdp/model_v2.py consumes), under no_screen / q10y / q5y, with FULL
stage I-IV granularity (no early/advanced bucketing needed this time,
since state9 already carries stage as a first-class axis).

Mirrors transitions/compare_matlab_benchmark.py's structure and scope note:
surveillance is OFF (matches the whole 9-state estimation methodology).

Run: python compare_9state_vs_cmost.py -n 100000
"""
from __future__ import annotations
import os, sys, time, argparse, csv
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.params import build_params
from env.cmost_individual import CRCEngine
from env.state9 import (NORMAL, EARLY_POLYP, ADV_POLYP, CA_I, CA_II, CA_III, CA_IV,
                          CRC_DEATH, OTHER_DEATH, N_STATES9, clinical_state9)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CA_STAGES = (CA_I, CA_II, CA_III, CA_IV)
MAXY = 100


def load_detection():
    e = np.load(os.path.join(RES, 'pomdp_effects.npz'), allow_pickle=True)
    d_EA, d_AA = float(e['d_EA']), float(e['d_AA'])
    if 'd_PC_stage' in e.files and len(e['d_PC_stage']) == 4:
        d_pc = np.asarray(e['d_PC_stage'], float)
    else:
        d_pc = np.full(4, float(e['d_PC']))
    return d_EA, d_AA, d_pc


# ---------------------------------------------------------------------------
# CMOST-direct microsim
# ---------------------------------------------------------------------------
def run_microsim(eng, n, screen_ages, seed):
    eng.rng = np.random.default_rng(seed)
    screen_ages = set(int(a) for a in screen_ages)
    death_age = np.empty(n); death_cause = np.zeros(n, np.int8)
    dx_stage = np.zeros(n, np.int8)   # 7..10, 0 if never diagnosed
    dx_route = np.zeros(n, np.int8)   # 1=screen 2=symptom
    sojourn = np.full(n, np.nan)      # det_year - onset year of the DETECTED focus
    n_screen_colo = 0
    for i in range(n):
        pt = eng.new_patient()
        for y in range(1, MAXY + 1):
            do_screen = y in screen_ages
            for q in (1, 2, 3, 4):
                if q == 1 and do_screen and pt.alive and not pt.ever_clinical:
                    was = pt.ever_clinical
                    eng.colonoscopy(pt, y, 1, 'Scre')
                    n_screen_colo += 1
                    if pt.ever_clinical and not was:
                        dx_route[i] = 1
                eng._step_quarter(pt, y, q)
                if pt.ever_clinical and dx_route[i] == 0:
                    dx_route[i] = 2
                if not pt.alive:
                    break
            if not pt.alive:
                break
        death_age[i] = pt.death_time if not pt.alive else MAXY
        death_cause[i] = pt.death_cause
        if pt.det_stage:
            dx_stage[i] = int(pt.det_stage[0])
            sojourn[i] = pt.det_year[0] - pt.det_onset_year[0]
    return dict(death_age=death_age, death_cause=death_cause, dx_stage=dx_stage,
                dx_route=dx_route, n_screen_colo=n_screen_colo,
                sojourn=sojourn)


def summarize_microsim(m, n):
    crc = m['death_cause'] == 2
    dx = m['dx_stage']; route = m['dx_route']
    def stage_counts(mask):
        return np.array([np.count_nonzero((dx == s) & mask) for s in (7, 8, 9, 10)])
    sc_scr = stage_counts(route == 1)
    sc_sym = stage_counts(route == 2)
    diagnosed = dx > 0
    return {
        'avg_age_all': float(m['death_age'].mean()),
        'avg_age_crc': float(m['death_age'][crc].mean()) if crc.any() else float('nan'),
        'crc_deaths_100k': 1e5 * crc.mean(),
        'incid_100k': 1e5 * (dx > 0).mean(),
        'screen_100k': 1e5 * (route == 1).mean(),
        'symptom_100k': 1e5 * (route == 2).mean(),
        'stage_all_pct': 100 * (sc_scr + sc_sym) / max((sc_scr + sc_sym).sum(), 1),
        'stage_scr_pct': 100 * sc_scr / max(sc_scr.sum(), 1),
        'stage_sym_pct': 100 * sc_sym / max(sc_sym.sum(), 1),
        'avg_sojourn': float(m['sojourn'][diagnosed].mean()) if diagnosed.any() else float('nan'),
    }


# ---------------------------------------------------------------------------
# 9-state cohort propagation (exactly what pomdp/model_v2.py consumes)
# ---------------------------------------------------------------------------
def run_cohort9(Pq_stay, dq, T_det, d_EA, d_AA, d_pc, screen_ages):
    """QUARTERLY-stepped undetected-world propagation (NOT the annually-
    composed P_undetected/d_symp): composing to annual first mis-attributes
    the detection stage whenever a patient silently progresses mid-year
    before self-presenting -- the exit gets credited to the stage they
    STARTED the year in, not the (more advanced) stage they were actually
    at when detected. This is the same family of annual-discretization
    artifact documented in transitions/SCREENING_VALIDATION.md; stepping
    quarter-by-quarter with the raw quarterly matrices avoids it."""
    screen_ages = set(int(a) for a in screen_ages)
    dist_u = np.zeros(N_STATES9); dist_u[NORMAL] = 1.0
    dist_d = np.zeros(4)
    inc_scr = np.zeros(4); inc_sym = np.zeros(4)
    m_crc = m_oth_det = 0.0
    death_crc_age = 0.0
    death_oth_age = 0.0
    prev_oth_u = 0.0
    ages = sorted(Pq_stay.keys())

    for age in range(min(ages), MAXY + 1):
        Tq = Pq_stay[min(max(age, min(ages)), max(ages))]
        dq_a = dq[min(max(age, min(ages)), max(ages))]

        if age in screen_ages:
            new_u = dist_u.copy()
            new_u[NORMAL] += dist_u[EARLY_POLYP] * d_EA + dist_u[ADV_POLYP] * d_AA
            new_u[EARLY_POLYP] -= dist_u[EARLY_POLYP] * d_EA
            new_u[ADV_POLYP] -= dist_u[ADV_POLYP] * d_AA
            for k, s in enumerate(CA_STAGES):
                det = dist_u[s] * d_pc[k]
                new_u[s] -= det
                dist_d[k] += det
                inc_scr[k] += det
            dist_u = new_u

        # 4 quarterly sub-steps: symptomatic presentation attributed to
        # whatever stage the patient is truly AT that quarter
        for _qq in (1, 2, 3, 4):
            for k, s in enumerate(CA_STAGES):
                exit_mass = dist_u[s] * dq_a[s]
                dist_d[k] += exit_mass
                inc_sym[k] += exit_mass
            dist_u = dist_u @ Tq   # rows sum to 1 - dq_a[s]; no double counting

        # detected-world hazard, once per year (age+0.5 half-cycle credit)
        new_d = np.zeros(4)
        for k in range(4):
            p_stay, p_crc, p_oth = T_det[min(age, T_det.shape[0]) - 1, k]
            new_d[k] = dist_d[k] * p_stay
            m_crc += dist_d[k] * p_crc
            m_oth_det += dist_d[k] * p_oth
            death_crc_age += dist_d[k] * p_crc * (age + 0.5)
            death_oth_age += dist_d[k] * p_oth * (age + 0.5)
        dist_d = new_d

        cur_oth_u = dist_u[OTHER_DEATH]
        death_oth_age += (cur_oth_u - prev_oth_u) * (age + 0.5)
        prev_oth_u = cur_oth_u

    living = (dist_u.sum() - dist_u[OTHER_DEATH]) + dist_d.sum()
    death_oth_age += living * MAXY   # still alive at horizon -> credited at MAXY
    tot_oth = dist_u[OTHER_DEATH] + m_oth_det + living
    tot_crc = m_crc

    # raw (un-normalised, per-1-starting-population) accumulators, so callers
    # can correctly population-weight-combine BEFORE deriving rates/averages
    # (e.g. male/female 50:50 -- simple-averaging the derived % values below
    # would be wrong for ratio-type metrics like avg age or stage-mix %).
    raw = dict(inc_scr=inc_scr, inc_sym=inc_sym, m_crc=m_crc,
               death_crc_age=death_crc_age, death_oth_age=death_oth_age,
               tot_oth=tot_oth, tot_crc=tot_crc)

    avg_age_all = (death_oth_age + death_crc_age) / max(tot_oth + tot_crc, 1e-9)
    avg_age_crc = death_crc_age / max(tot_crc, 1e-9)
    incid = inc_scr.sum() + inc_sym.sum()
    return {
        'avg_age_all': avg_age_all,
        'avg_age_crc': avg_age_crc,
        'crc_deaths_100k': 1e5 * tot_crc,
        'incid_100k': 1e5 * incid,
        'screen_100k': 1e5 * inc_scr.sum(),
        'symptom_100k': 1e5 * inc_sym.sum(),
        'stage_all_pct': 100 * (inc_scr + inc_sym) / max(incid, 1e-9),
        'stage_scr_pct': 100 * inc_scr / max(inc_scr.sum(), 1e-9),
        'stage_sym_pct': 100 * inc_sym / max(inc_sym.sum(), 1e-9),
        '_raw': raw,
    }


def run_cohort9_sojourn(Pq_stay, dq, T_det, d_EA, d_AA, d_pc, screen_ages):
    """Same quarterly propagation as run_cohort9(), PLUS a companion
    accumulator tracked in parallel with the undetected Ca-stage mass:
      eacc[k] = sum(mass * age_of_first_entry_into_any_Ca_stage)
    for currently-undetected population sitting in Ca-stage k (k=0..3 <->
    CA_I..CA_IV). Entry age is a fixed per-individual attribute, so it
    propagates through the EXACT SAME within-Ca-stage sub-transition
    operator as the raw mass -- the standard "moment accumulator through a
    Markov chain" trick (rows of Tq_ca already sum to < 1 wherever mass
    leaks to symptom-detection or other-cause death, exactly mirroring how
    dist_u itself leaks there). On every detection exit (screen or
    symptom), the exiting mass is split proportionally using the CURRENT
    eacc, giving mean sojourn time (age at detection - age of first
    Ca-stage entry), weighted by actually-detected mass -- the same
    population CMOST's det_year/det_onset_year metrics are computed over.
    """
    screen_ages = set(int(a) for a in screen_ages)
    dist_u = np.zeros(N_STATES9); dist_u[NORMAL] = 1.0
    dist_d = np.zeros(4)
    inc_scr = np.zeros(4); inc_sym = np.zeros(4)
    ages = sorted(Pq_stay.keys())

    eacc = np.zeros(4)
    sojourn_num = 0.0; sojourn_den = 0.0

    def _exit_explicit(k, s, exit_mass, cur_age):
        """For exits handled by an EXPLICIT subtraction from dist_u (the
        screening operator) -- eacc must be explicitly reduced by the same
        proportion here, since nothing else discounts it."""
        nonlocal sojourn_num, sojourn_den
        if exit_mass <= 0 or dist_u[s] <= 1e-300:
            return
        frac = exit_mass / dist_u[s]
        entry_age_exit = eacc[k] * frac
        sojourn_num += exit_mass * cur_age - entry_age_exit
        sojourn_den += exit_mass
        eacc[k] -= entry_age_exit

    def _exit_implicit(k, exit_mass, dq_s, cur_age):
        """For exits handled IMPLICITLY by Tq_ca's row already summing to
        (1 - dq_a[s]) (the quarterly symptomatic-presentation hazard) --
        do NOT also subtract from eacc here, or the (1-dq_a[s]) factor
        gets applied twice (once here, once via the Tq_ca propagation right
        after) and the accumulator is under-decayed, inflating sojourn
        time ~10x. Just read off the exiting share; Tq_ca's own row-sum
        deficiency retires the rest."""
        nonlocal sojourn_num, sojourn_den
        if exit_mass <= 0:
            return
        entry_age_exit = eacc[k] * dq_s
        sojourn_num += exit_mass * cur_age - entry_age_exit
        sojourn_den += exit_mass

    for age in range(min(ages), MAXY + 1):
        Tq = Pq_stay[min(max(age, min(ages)), max(ages))]
        dq_a = dq[min(max(age, min(ages)), max(ages))]
        Tq_ca = Tq[np.ix_(CA_STAGES, CA_STAGES)]

        if age in screen_ages:
            new_u = dist_u.copy()
            new_u[NORMAL] += dist_u[EARLY_POLYP] * d_EA + dist_u[ADV_POLYP] * d_AA
            new_u[EARLY_POLYP] -= dist_u[EARLY_POLYP] * d_EA
            new_u[ADV_POLYP] -= dist_u[ADV_POLYP] * d_AA
            for k, s in enumerate(CA_STAGES):
                det = dist_u[s] * d_pc[k]
                _exit_explicit(k, s, det, age)
                new_u[s] -= det
                dist_d[k] += det
                inc_scr[k] += det
            dist_u = new_u

        for _qq in (1, 2, 3, 4):
            age_q = age + (_qq - 1) / 4.0
            for k, s in enumerate(CA_STAGES):
                exit_mass = dist_u[s] * dq_a[s]
                _exit_implicit(k, exit_mass, dq_a[s], age_q)
                dist_d[k] += exit_mass
                inc_sym[k] += exit_mass
            entry_in = np.array([
                dist_u[NORMAL] * Tq[NORMAL, s]
                + dist_u[EARLY_POLYP] * Tq[EARLY_POLYP, s]
                + dist_u[ADV_POLYP] * Tq[ADV_POLYP, s]
                for s in CA_STAGES])
            eacc = eacc @ Tq_ca + entry_in * age_q
            dist_u = dist_u @ Tq

        new_d = np.zeros(4)
        for k in range(4):
            p_stay, p_crc, p_oth = T_det[min(age, T_det.shape[0]) - 1, k]
            new_d[k] = dist_d[k] * p_stay
        dist_d = new_d

    avg_sojourn = sojourn_num / sojourn_den if sojourn_den > 1e-9 else float('nan')
    return {'avg_sojourn': avg_sojourn}


def combine_sexmix(res_male, res_female, w_male=0.5, w_female=0.5):
    """Population-weight-combine two run_cohort9() outputs (e.g. male/female
    50:50) at the RAW ACCUMULATOR level, then re-derive rates/averages --
    NOT a simple average of the already-derived % values, which would be
    wrong for ratio-type metrics (avg age, stage-mix %)."""
    rm, rf = res_male['_raw'], res_female['_raw']
    inc_scr = w_male * rm['inc_scr'] + w_female * rf['inc_scr']
    inc_sym = w_male * rm['inc_sym'] + w_female * rf['inc_sym']
    m_crc = w_male * rm['m_crc'] + w_female * rf['m_crc']
    death_crc_age = w_male * rm['death_crc_age'] + w_female * rf['death_crc_age']
    death_oth_age = w_male * rm['death_oth_age'] + w_female * rf['death_oth_age']
    tot_oth = w_male * rm['tot_oth'] + w_female * rf['tot_oth']
    tot_crc = w_male * rm['tot_crc'] + w_female * rf['tot_crc']

    avg_age_all = (death_oth_age + death_crc_age) / max(tot_oth + tot_crc, 1e-9)
    avg_age_crc = death_crc_age / max(tot_crc, 1e-9)
    incid = inc_scr.sum() + inc_sym.sum()
    return {
        'avg_age_all': avg_age_all,
        'avg_age_crc': avg_age_crc,
        'crc_deaths_100k': 1e5 * tot_crc,
        'incid_100k': 1e5 * incid,
        'screen_100k': 1e5 * inc_scr.sum(),
        'symptom_100k': 1e5 * inc_sym.sum(),
        'stage_all_pct': 100 * (inc_scr + inc_sym) / max(incid, 1e-9),
        'stage_scr_pct': 100 * inc_scr / max(inc_scr.sum(), 1e-9),
        'stage_sym_pct': 100 * inc_sym / max(inc_sym.sum(), 1e-9),
        # propagate a combined raw accumulator too, so combine_sexmix()
        # outputs can themselves be fed back into combine_sexmix() (e.g.
        # low+high within a sex, then male+female) -- chainable combination.
        '_raw': dict(inc_scr=inc_scr, inc_sym=inc_sym, m_crc=m_crc,
                     death_crc_age=death_crc_age, death_oth_age=death_oth_age,
                     tot_oth=tot_oth, tot_crc=tot_crc),
    }


def run_cohort9_annual(P_undet, d_symp, T_det, d_EA, d_AA, d_pc, screen_ages):
    """ANNUAL version of run_cohort9 (1 step/year via the 4-exit-sink-fixed
    P_undetected/d_symp, (9,4)-shaped) instead of 4 quarterly sub-steps.
    Used for the sex-stratified matrices, which were only saved in annual
    form. Mathematically equivalent to the quarterly version -- verified
    directly (max abs diff ~5e-13) when this fix was built -- so this is
    NOT a new approximation, just a cheaper equivalent computation."""
    screen_ages = set(int(a) for a in screen_ages)
    dist_u = np.zeros(N_STATES9); dist_u[NORMAL] = 1.0
    dist_d = np.zeros(4)
    inc_scr = np.zeros(4); inc_sym = np.zeros(4)
    m_crc = m_oth_det = 0.0
    death_crc_age = 0.0
    death_oth_age = 0.0
    prev_oth_u = 0.0
    ages = sorted(P_undet.keys())

    for age in range(min(ages), MAXY + 1):
        Ty = P_undet[min(max(age, min(ages)), max(ages))]
        dsy = d_symp[min(max(age, min(ages)), max(ages))]   # (9,4)

        if age in screen_ages:
            new_u = dist_u.copy()
            new_u[NORMAL] += dist_u[EARLY_POLYP] * d_EA + dist_u[ADV_POLYP] * d_AA
            new_u[EARLY_POLYP] -= dist_u[EARLY_POLYP] * d_EA
            new_u[ADV_POLYP] -= dist_u[ADV_POLYP] * d_AA
            for k, s in enumerate(CA_STAGES):
                det = dist_u[s] * d_pc[k]
                new_u[s] -= det
                dist_d[k] += det
                inc_scr[k] += det
            dist_u = new_u

        # one annual step: exit mass (any origin state, any destination
        # stage k) attributed via the full (9,4) matrix -- correctly
        # captures mid-year progression before self-presentation
        exit_mass = dist_u @ dsy   # (4,)
        inc_sym += exit_mass
        dist_d += exit_mass
        dist_u = dist_u @ Ty

        new_d = np.zeros(4)
        for k in range(4):
            p_stay, p_crc, p_oth = T_det[min(age, T_det.shape[0]) - 1, k]
            new_d[k] = dist_d[k] * p_stay
            m_crc += dist_d[k] * p_crc
            m_oth_det += dist_d[k] * p_oth
            death_crc_age += dist_d[k] * p_crc * (age + 0.5)
            death_oth_age += dist_d[k] * p_oth * (age + 0.5)
        dist_d = new_d

        cur_oth_u = dist_u[OTHER_DEATH]
        death_oth_age += (cur_oth_u - prev_oth_u) * (age + 0.5)
        prev_oth_u = cur_oth_u

    living = (dist_u.sum() - dist_u[OTHER_DEATH]) + dist_d.sum()
    death_oth_age += living * MAXY
    tot_oth = dist_u[OTHER_DEATH] + m_oth_det + living
    tot_crc = m_crc

    raw = dict(inc_scr=inc_scr, inc_sym=inc_sym, m_crc=m_crc,
               death_crc_age=death_crc_age, death_oth_age=death_oth_age,
               tot_oth=tot_oth, tot_crc=tot_crc)

    avg_age_all = (death_oth_age + death_crc_age) / max(tot_oth + tot_crc, 1e-9)
    avg_age_crc = death_crc_age / max(tot_crc, 1e-9)
    incid = inc_scr.sum() + inc_sym.sum()
    return {
        'avg_age_all': avg_age_all,
        'avg_age_crc': avg_age_crc,
        'crc_deaths_100k': 1e5 * tot_crc,
        'incid_100k': 1e5 * incid,
        'screen_100k': 1e5 * inc_scr.sum(),
        'symptom_100k': 1e5 * inc_sym.sum(),
        'stage_all_pct': 100 * (inc_scr + inc_sym) / max(incid, 1e-9),
        'stage_scr_pct': 100 * inc_scr / max(inc_scr.sum(), 1e-9),
        'stage_sym_pct': 100 * inc_sym / max(inc_sym.sum(), 1e-9),
        '_raw': raw,
    }


def print_row(label, a, b, fmt='%.2f', rows_out=None, scenario=None):
    a_s = fmt % a if not (isinstance(a, float) and np.isnan(a)) else 'n/a'
    b_s = fmt % b if not (isinstance(b, float) and np.isnan(b)) else 'n/a'
    diff = f"{100*(b-a)/a:+.1f}%" if abs(a) > 1e-9 else ''
    print(f"{label:<24}{a_s:>14}{b_s:>14}{diff:>10}")
    if rows_out is not None:
        rows_out.append({'scenario': scenario, 'metric': label,
                         'CMOST': a, 'matrix9state': b, 'diff_pct': diff})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=100000)
    ap.add_argument('--seed', type=int, default=555)
    args = ap.parse_args()
    n = args.n

    z9 = np.load(os.path.join(RES, 'transitions_9state.npz'), allow_pickle=True)
    ages9 = z9['ages']
    Pq_stay = {int(a): z9['Pq_stay'][i] for i, a in enumerate(ages9)}
    dq_quarter = {int(a): z9['dq_quarter'][i] for i, a in enumerate(ages9)}
    T_det = np.load(os.path.join(RES, 'T_detected_tauphase.npz'))['T_detected']
    d_EA, d_AA, d_pc = load_detection()

    eng = CRCEngine(build_params('CMOST13', 20000, args.seed), rng=np.random.default_rng(0))
    scenarios = {'No screening': [], 'Screen q10y (50,60,70)': [50, 60, 70],
                 'Screen q5y (50-75)': [50, 55, 60, 65, 70, 75]}

    rows_out = []
    for name, sa in scenarios.items():
        print(f"[microsim] {name} (n={n}) ...", flush=True)
        t0 = time.time()
        m = run_microsim(eng, n, sa, seed=args.seed)
        cm = summarize_microsim(m, n)
        mx = run_cohort9(Pq_stay, dq_quarter, T_det, d_EA, d_AA, d_pc, sa)
        mx.update(run_cohort9_sojourn(Pq_stay, dq_quarter, T_det, d_EA, d_AA, d_pc, sa))
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

        print(f"\n================ {name} ================")
        print(f"{'metric':<24}{'CMOST':>14}{'9-state matrix':>14}{'diff':>10}")
        print('-' * 62)
        pr = lambda label, a, b, fmt='%.2f': print_row(label, a, b, fmt, rows_out, name)
        pr('Avg age at death', cm['avg_age_all'], mx['avg_age_all'])
        pr('Avg age at CRC death', cm['avg_age_crc'], mx['avg_age_crc'])
        pr('CRC deaths /100k', cm['crc_deaths_100k'], mx['crc_deaths_100k'], '%.0f')
        pr('Incidence /100k', cm['incid_100k'], mx['incid_100k'], '%.0f')
        pr('Screen-detected /100k', cm['screen_100k'], mx['screen_100k'], '%.0f')
        pr('Symptom-detected /100k', cm['symptom_100k'], mx['symptom_100k'], '%.0f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (all)', cm['stage_all_pct'][k], mx['stage_all_pct'][k], '%.1f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (screen)', cm['stage_scr_pct'][k], mx['stage_scr_pct'][k], '%.1f')
        for k, lbl in enumerate(['I', 'II', 'III', 'IV']):
            pr(f'Stage {lbl} % (symptom)', cm['stage_sym_pct'][k], mx['stage_sym_pct'][k], '%.1f')
        pr('Sojourn time (yrs, mean)', cm['avg_sojourn'], mx['avg_sojourn'])

    out_csv = os.path.join(RES, 'compare_9state_vs_cmost.csv')
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['scenario', 'metric', 'CMOST', 'matrix9state', 'diff_pct'])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved -> {out_csv}")


if __name__ == '__main__':
    main()
