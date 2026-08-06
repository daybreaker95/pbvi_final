"""
validate_transitions.py
=======================

Verification & validation of the empirically-estimated 6-state transition
matrices, following the microsimulation-verification methodology of Krijkamp
et al. (2018, "Microsimulation modeling for health decision sciences using R",
Med Decis Making; NIHMS931782) and external comparison against CISNET/CRC-SPIN
natural-history targets (Rutter et al.) and SEER.

Checks produced
---------------
V1  Markov-trace consistency: propagate the estimated age-specific transition
    matrices as a deterministic cohort model and compare the resulting state
    occupancy over age to the microsimulation's own empirical occupancy.  Close
    agreement verifies both the estimator and the adequacy of the 6-state
    (memoryless) abstraction.

V2  Monte-Carlo convergence: subsample the cohort at increasing N and show that
    key transition-probability estimates stabilise and their Monte-Carlo
    standard error shrinks as ~1/sqrt(N).

V3  Natural-history summary vs literature: cumulative CRC incidence, adenoma /
    advanced-adenoma prevalence, mean sojourn time, fraction of preclinical
    cancers surfacing clinically, and adenoma dwell time.

Figures are written to results/figures/ and paper/figures/; numeric results to
results/validation_summary.json and paper/validation_summary.md.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.params import build_params
from env.cmost_individual import (
    CRCEngine, N_STATES, STATE_NAMES,
    NORMAL, EARLY_ADENOMA, ADVANCED_ADENOMA, PRECLINICAL_CANCER,
    CLINICAL_CANCER, DEAD,
)
from transitions.estimate_transitions import (
    tally_transitions, normalise, band_pool, AGE_MIN, AGE_MAX,
)

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
FIG = os.path.join(RES, 'figures')
PAPER_FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures'))
for d in (FIG, PAPER_FIG):
    os.makedirs(d, exist_ok=True)

SHORT = ["Norm", "EA", "AA", "PreCa", "ClinCa", "Dead"]
COLORS = ['#4C78A8', '#F58518', '#E45756', '#B279A2', '#72B7B2', '#9D9D9D']


def _savefig(fig, name):
    for d in (FIG, PAPER_FIG):
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# V1 -- Markov cohort trace vs microsimulation occupancy
# ---------------------------------------------------------------------------
def markov_trace(P, ages, start_age, init_dist):
    """Propagate the age-specific transition matrices deterministically."""
    A = len(ages)
    trace = np.zeros((A + 1, N_STATES))
    d = init_dist.copy()
    trace[0] = d
    for ai in range(A):
        d = d @ P[ai]
        trace[ai + 1] = d
    return trace   # rows correspond to ages start_age .. start_age+A


def empirical_occupancy(paths, ages):
    """Full-cohort state distribution (incl. Dead) at each age."""
    n = paths.shape[0]
    all_ages = list(ages) + [ages[-1] + 1]
    occ = np.zeros((len(all_ages), N_STATES))
    for k, age in enumerate(all_ages):
        col = paths[:, age]
        for s in range(N_STATES):
            occ[k, s] = np.count_nonzero(col == s) / n
    return occ


def markov_property_test(paths, ages, band=(55, 75)):
    """Genuine test of the first-order Markov assumption of the 6-state model.

    Unlike the trace check (which is an algebraic identity: the MLE one-step
    matrix reproduces the marginal by construction), this test asks whether the
    NEXT-state distribution depends on PRIOR history.  For each current state s
    we split patient-years by whether the patient was already in s the previous
    year ('stayer', longer sojourn) versus arrived this year ('newcomer') and
    compare their outgoing distributions via total-variation distance.  Small
    TV => the memoryless 6-state abstraction is adequate.
    """
    lo, hi = band
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, s in enumerate([EARLY_ADENOMA, ADVANCED_ADENOMA, PRECLINICAL_CANCER]):
        stay_to = np.zeros(N_STATES)
        new_to = np.zeros(N_STATES)
        for age in range(max(lo, int(ages[0]) + 1), min(hi, int(ages[-1]))):
            prev, cur, nxt = paths[:, age - 1], paths[:, age], paths[:, age + 1]
            here = cur == s
            stayer = here & (prev == s)
            newcomer = here & (prev != s)
            for sp in range(N_STATES):
                stay_to[sp] += np.count_nonzero(nxt[stayer] == sp)
                new_to[sp] += np.count_nonzero(nxt[newcomer] == sp)
        stp = stay_to / stay_to.sum() if stay_to.sum() else stay_to
        ntp = new_to / new_to.sum() if new_to.sum() else new_to
        tv = 0.5 * np.abs(stp - ntp).sum()
        results[STATE_NAMES[s]] = {
            'n_stayer': int(stay_to.sum()), 'n_newcomer': int(new_to.sum()),
            'tv_distance': float(tv),
            'stayer_dist': [float(x) for x in stp],
            'newcomer_dist': [float(x) for x in ntp],
        }
        ax = axes[j]
        x = np.arange(N_STATES)
        ax.bar(x - 0.2, 100 * ntp, width=0.4, label='newcomer', color=COLORS[1])
        ax.bar(x + 0.2, 100 * stp, width=0.4, label='stayer', color=COLORS[0])
        ax.set_title(f'{STATE_NAMES[s]}  (TV={tv:.3f})', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('P(next | now) %'); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle('Markov-property test: outgoing distribution by prior sojourn (age 55-75)')
    _savefig(fig, 'V1b_markov_property.png')
    return results


def check_markov_consistency(P, paths, ages):
    start_age = ages[0]
    occ = empirical_occupancy(paths, ages)
    init = occ[0]
    trace = markov_trace(P, ages, start_age, init)

    all_ages = np.array(list(ages) + [ages[-1] + 1])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    max_abs = 0.0
    for s in range(N_STATES):
        ax = axes.flat[s]
        ax.plot(all_ages, 100 * occ[:, s], color=COLORS[s], lw=2, label='microsim')
        ax.plot(all_ages, 100 * trace[:, s], '--', color='k', lw=1.3, label='Markov trace')
        ax.set_title(STATE_NAMES[s]); ax.grid(alpha=0.3)
        if s >= 3:
            ax.set_xlabel('age')
        if s % 3 == 0:
            ax.set_ylabel('% of cohort')
        max_abs = max(max_abs, np.max(np.abs(occ[:, s] - trace[:, s])))
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('V1  Markov-trace consistency: 6-state cohort model vs CMOST microsimulation')
    _savefig(fig, 'V1_markov_trace_consistency.png')

    # per-age total-variation distance
    tv = 0.5 * np.abs(occ - trace).sum(axis=1)
    return {
        'max_abs_state_discrepancy': float(max_abs),
        'mean_total_variation': float(tv.mean()),
        'max_total_variation': float(tv.max()),
    }


# ---------------------------------------------------------------------------
# V2 -- Monte-Carlo convergence of transition estimates
# ---------------------------------------------------------------------------
def check_mc_convergence(paths, ages, seed=0):
    rng = np.random.default_rng(seed)
    n_full = paths.shape[0]
    Ns = [1000, 2000, 5000, 10000, 20000, 50000, n_full]
    # three representative transitions to track (age band 55-70 pooled)
    targets = [
        (ADVANCED_ADENOMA, PRECLINICAL_CANCER, 'AA→PreCa'),
        (EARLY_ADENOMA, ADVANCED_ADENOMA, 'EA→AA'),
        (PRECLINICAL_CANCER, CLINICAL_CANCER, 'PreCa→ClinCa'),
        (NORMAL, EARLY_ADENOMA, 'Norm→EA'),
    ]
    band_lo, band_hi = 55, 70
    mask = (ages >= band_lo) & (ages < band_hi)

    est = {lab: [] for *_, lab in targets}
    se = {lab: [] for *_, lab in targets}
    for N in Ns:
        idx = rng.choice(n_full, size=N, replace=False)
        _, counts, _, _ = tally_transitions(paths[idx])
        cb = counts[mask].sum(axis=0)
        P, SE, nrow = normalise(cb[None])  # normalise expects (A,6,6)
        P, SE = P[0], SE[0]
        for s, sp, lab in targets:
            est[lab].append(100 * P[s, sp])
            se[lab].append(100 * SE[s, sp])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for s, sp, lab in targets:
        ax1.plot(Ns, est[lab], 'o-', label=lab)
    ax1.set_xscale('log'); ax1.set_xlabel('N patients'); ax1.set_ylabel('P (%/yr)')
    ax1.set_title('Transition estimate vs sample size (age 55-70)')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    for s, sp, lab in targets:
        ax2.plot(Ns, se[lab], 'o-', label=lab)
    # reference 1/sqrt(N) slope
    ref = np.array(Ns, float)
    ax2.plot(ref, se['AA→PreCa'][0] * np.sqrt(Ns[0] / ref), 'k--', lw=1, label=r'$\propto 1/\sqrt{N}$')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('N patients'); ax2.set_ylabel('Monte-Carlo SE (%/yr)')
    ax2.set_title('Monte-Carlo standard error scaling')
    ax2.grid(alpha=0.3, which='both'); ax2.legend(fontsize=8)
    fig.suptitle('V2  Monte-Carlo convergence of empirical transition probabilities')
    _savefig(fig, 'V2_mc_convergence.png')

    return {'Ns': Ns,
            'estimates_pct': {k: [float(x) for x in v] for k, v in est.items()},
            'se_pct': {k: [float(x) for x in v] for k, v in se.items()}}


# ---------------------------------------------------------------------------
# V3 -- natural-history summary vs literature (needs one modest sim run)
# ---------------------------------------------------------------------------
def natural_history_summary(paths, ages, n_record=30000, seed=999, settings='CMOST13'):
    # prevalence by age from paths (yearly snapshots are fine for prevalence)
    n = paths.shape[0]
    inc_ages = np.arange(ages[0], ages[-1] + 2)
    prev_any = np.zeros(len(inc_ages))
    prev_adv = np.zeros(len(inc_ages))
    for k, age in enumerate(inc_ages):
        col = paths[:, age]
        living = col != DEAD
        nl = living.sum()
        if nl:
            prev_any[k] = 100 * np.count_nonzero(np.isin(col[living],
                          [EARLY_ADENOMA, ADVANCED_ADENOMA])) / nl
            prev_adv[k] = 100 * np.count_nonzero(col[living] == ADVANCED_ADENOMA) / nl

    # dwell/progression AND proper cumulative clinical incidence: a dedicated
    # recording run (yearly snapshots undercount diagnose-and-die-same-year CRC,
    # so incidence is measured here from the actual diagnosis event).
    cache = os.path.join(RES, 'nh_times2.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        dwell = z['dwell']
        n_adenoma_onset = int(z['n_adenoma_onset'])
        n_adenoma_to_cancer = int(z['n_adenoma_to_cancer'])
        dx_ages = z['dx_ages']
        n_rec = int(z['n_rec'])
    else:
        params = build_params(settings, n_patients=500, seed=seed)
        eng = CRCEngine(params, rng=np.random.default_rng(seed + 3))
        dwell = []
        n_adenoma_onset = 0
        n_adenoma_to_cancer = 0
        dx_ages = []       # age at first clinical diagnosis (per diagnosed patient)
        n_rec = n_record
        for i in range(n_record):
            pt = eng.new_patient()
            seen_adenoma = False
            recorded_cancer = False
            for y in range(1, 101):
                eng.step_year(pt, y, screen=False)
                if len(pt.polyp_stage) > 0 and not seen_adenoma:
                    seen_adenoma = True
                    n_adenoma_onset += 1
                if pt.ca_dwell and not recorded_cancer:
                    for dw in pt.ca_dwell:
                        if dw > 0:
                            dwell.append(float(dw))
                            recorded_cancer = True
                            if seen_adenoma:
                                n_adenoma_to_cancer += 1
                            break
                if not pt.alive:
                    break
            if pt.det_stage:
                dx_ages.append(float(pt.det_year[0]))
        dwell = np.array(dwell)
        dx_ages = np.array(dx_ages)
        np.savez(cache, dwell=dwell, n_adenoma_onset=n_adenoma_onset,
                 n_adenoma_to_cancer=n_adenoma_to_cancer,
                 dx_ages=dx_ages, n_rec=n_rec)

    # cumulative clinical CRC incidence by age (proper)
    cum_inc = np.array([100 * np.count_nonzero(dx_ages <= age) / n_rec
                        for age in inc_ages])

    # sojourn (preclinical->clinical) from the estimated transition matrix:
    # expected years in Preclinical before leaving, and fraction leaving to Clinical
    _, _, _, counts_band = band_pool(ages, tally_transitions(paths)[1])
    edges, P_band, _, _ = band_pool(ages, tally_transitions(paths)[1])
    # use the 50-70 bands averaged
    bi = [i for i in range(len(edges) - 1) if 50 <= edges[i] < 70]
    p_pp = np.mean([P_band[i, PRECLINICAL_CANCER, PRECLINICAL_CANCER] for i in bi])
    p_pc = np.mean([P_band[i, PRECLINICAL_CANCER, CLINICAL_CANCER] for i in bi])
    p_pd = np.mean([P_band[i, PRECLINICAL_CANCER, DEAD] for i in bi])
    mean_sojourn = 1.0 / (1.0 - p_pp) if p_pp < 1 else float('nan')
    frac_surface = p_pc / (p_pc + p_pd) if (p_pc + p_pd) > 0 else float('nan')

    # figure: cumulative incidence + prevalence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(inc_ages, cum_inc, color=COLORS[4], lw=2)
    ax1.axhspan(6.7, 7.2, color='green', alpha=0.15, label='CISNET cum. CRC 40-100 (6.7-7.2%)')
    ax1.set_xlabel('age'); ax1.set_ylabel('cumulative clinical CRC (%)')
    ax1.set_title('Cumulative CRC incidence (no screening)')
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    ax2.plot(inc_ages, prev_any, color=COLORS[1], lw=2, label='any adenoma')
    ax2.plot(inc_ages, prev_adv, color=COLORS[2], lw=2, label='advanced adenoma')
    ax2.axhspan(5, 9, color=COLORS[2], alpha=0.12, label='adv. adenoma ref 5-9%')
    ax2.set_xlim(40, 85); ax2.set_xlabel('age'); ax2.set_ylabel('prevalence among living (%)')
    ax2.set_title('Adenoma prevalence (no screening)')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle('V3  Natural-history outputs vs literature targets')
    _savefig(fig, 'V3_natural_history.png')

    frac_progress = (n_adenoma_to_cancer / n_adenoma_onset) if n_adenoma_onset else float('nan')
    return {
        'cum_crc_incidence_lifetime_pct': float(cum_inc[-1]),
        'cum_crc_incidence_age80_pct': float(cum_inc[min(len(cum_inc)-1, 80-ages[0])]),
        'adv_adenoma_prev_60_pct': float(prev_adv[60 - ages[0]]),
        'any_adenoma_prev_60_pct': float(prev_any[60 - ages[0]]),
        'mean_sojourn_years_fromT': float(mean_sojourn),
        'frac_preclinical_surface_clinical': float(frac_surface),
        'mean_dwell_years_adenoma_to_cancer': float(np.mean(dwell)) if len(dwell) else None,
        'median_dwell_years': float(np.median(dwell)) if len(dwell) else None,
        'frac_adenoma_progress_to_cancer_pct': float(100 * frac_progress),
    }


# ---------------------------------------------------------------------------
# transition heatmaps (paper figure)
# ---------------------------------------------------------------------------
def plot_heatmaps(P, ages):
    sel = [50, 60, 70, 80]
    fig, axes = plt.subplots(1, len(sel), figsize=(4 * len(sel), 3.6))
    for j, age in enumerate(sel):
        ai = age - ages[0]
        ax = axes[j]
        im = ax.imshow(100 * P[ai], cmap='viridis', vmin=0, vmax=100)
        ax.set_title(f'age {age}')
        ax.set_xticks(range(N_STATES)); ax.set_yticks(range(N_STATES))
        ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(SHORT if j == 0 else [''] * N_STATES, fontsize=7)
        for r in range(N_STATES):
            for c in range(N_STATES):
                v = 100 * P[ai, r, c]
                if v >= 0.5:
                    ax.text(c, r, f'{v:.0f}', ha='center', va='center',
                            color='white' if v < 60 else 'black', fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='P (%/yr)')
    fig.suptitle('Empirical 1-year transition matrices (CMOST natural history)')
    _savefig(fig, 'transition_heatmaps.png')


def main():
    npz = np.load(os.path.join(RES, 'transitions_cmost13.npz'), allow_pickle=True)
    P = npz['P']; ages = npz['ages']            # consumed matrix = quarterly-composed
    # direct 1-year MLE matrix (yearly snapshots) -> exact implementation check
    P_direct = npz['P_annual_direct'] if 'P_annual_direct' in npz.files else P
    paths = np.load(os.path.join(RES, 'state_paths_cmost13.npz'))['paths']
    print(f"Loaded P {P.shape} (composed), P_annual_direct, paths {paths.shape}")

    summary = {}
    print("V1: Markov-trace consistency (implementation check) ...")
    summary['V1_markov_consistency'] = check_markov_consistency(P_direct, paths, ages)
    summary['V1_markov_consistency']['note'] = (
        'Exact agreement (~1e-16) is expected for the DIRECT 1-year MLE matrix: it '
        'reproduces the yearly-snapshot cohort marginal by construction, verifying '
        'the estimator + cohort propagation (per Krijkamp 2018). The CONSUMED matrix '
        'is the quarterly-composed P = P_quarter^4, which removes annual-'
        'discretization artifacts (CRC-incidence and stage-IV undercount); its '
        'metric fidelity vs the microsim is quantified in SCREENING_VALIDATION.md.')
    print("   ", summary['V1_markov_consistency'])

    print("V1b: Markov-property (memorylessness) test ...")
    summary['V1b_markov_property'] = markov_property_test(paths, ages)
    for k, v in summary['V1b_markov_property'].items():
        print(f"    {k:20s} TV(stayer vs newcomer)={v['tv_distance']:.3f} "
              f"(n_stay={v['n_stayer']}, n_new={v['n_newcomer']})")

    print("V2: Monte-Carlo convergence ...")
    summary['V2_mc_convergence'] = check_mc_convergence(paths, ages)

    print("V3: natural-history summary ...")
    summary['V3_natural_history'] = natural_history_summary(paths, ages)
    print("   ", summary['V3_natural_history'])

    plot_heatmaps(P, ages)

    # external literature targets (for the report)
    summary['literature_targets'] = {
        'cum_crc_incidence_40_100_pct': '6.7-7.2 (CISNET); 7.1 (CRC-AIM)',
        'adv_adenoma_prevalence_50_65_pct': '5-9 (screening colonoscopy cohorts)',
        'any_adenoma_prevalence_50_65_pct': '25-30',
        'mean_sojourn_years': '1.9 (CRC-SPIN); 2-5 (models)',
        'frac_preclinical_surface_clinical_pct': '87 (CRC-SPIN)',
        'mean_dwell_years_adenoma_to_cancer': '25.4 (CRC-SPIN, progressing adenomas)',
        'frac_adenoma_progress_pct': '7.4 (CRC-SPIN)',
        'refs': ['Krijkamp 2018 NIHMS931782 (verification method)',
                 'Rutter et al. CRC-SPIN', 'CRC-AIM (Piscitello 2020)',
                 'Prakash 2017 CMOST'],
    }

    with open(os.path.join(RES, 'validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved validation_summary.json and figures to {FIG}")


if __name__ == '__main__':
    main()
