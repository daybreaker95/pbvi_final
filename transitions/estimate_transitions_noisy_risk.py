"""estimate_transitions_noisy_risk.py
=====================================
Build (sex x risk-class) transition matrices for a risk label that is a
NOISY observation of CMOST's individual_risk, rather than individual_risk
itself.

WHY
---
Everywhere else in this repo the agent's high/low label is the top 20% of
CMOST's own `individual_risk`. That is an oracle in the strict sense:
`prepare_simulation_params` draws exactly three things per person --
individual_risk, gender, and screening_preference (all zeros) -- and the
engine's only use of it is `PolypRate = PolypRate * IndividualRisk`. So
individual_risk x sex IS the complete person-level risk heterogeneity in
CMOST, and a label that reveals it exactly is a risk score that cannot be
improved on. No deployable score is that good.

The Jeon-2018 composite in tests/jeon_elbow_analysis.py does NOT test this,
because `bucket_map_to_individual_risk` ASSIGNS individual_risk from CMOST's
top-`high_frac` sub-pool to whoever the composite ranks in the top
`high_frac`. The composite score and the individual_risk cut are therefore
the same partition by construction, not two things that happen to agree.

MODEL OF A REAL SCORE
---------------------
Risk scores are built log-additively from hazard ratios, so the natural
corruption is in log space:

    S_i = log(individual_risk_i) + sigma * eps_i,    eps ~ N(0, 1)

and the programme labels the top `high_frac` by S. sigma=0 recovers the
oracle. sigma is calibrated to a target c-statistic for lifetime CRC death,
which is the quantity the epidemiological literature actually reports
(Jeon et al. 2018 reach ~0.63 for their full environmental + genetic model).

The simulation is run ONCE and the labelling applied afterwards, so every
sigma re-tallies the same cohort and the same state paths -- differences
between sigmas are the label, not Monte-Carlo noise.

Run: python transitions/estimate_transitions_noisy_risk.py -n 400000
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np

from env.state9 import N_STATES9, STATE9_NAMES, CRC_DEATH
from estimate_transitions_9state import AGE_MIN, AGE_MAX
from estimate_transitions_9state_sex_risk import simulate_with_sex_risk, estimate_for_mask

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CACHE = os.path.join(RES, 'noisy_risk_cohort.npz')


def crc_death_from_qstate(qstate):
    return (qstate == CRC_DEATH).any(axis=1)


def c_statistic(score, event):
    """Probability a random case scores above a random non-case (ties at
    half), computed by rank rather than by the O(n^2) pair count."""
    score = np.asarray(score, float)
    event = np.asarray(event, bool)
    n1, n0 = int(event.sum()), int((~event).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), float)
    sr = score[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # average rank, 1-based
        i = j + 1
    return float((ranks[event].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def label_for_sigma(risk, sigma, high_frac, rng):
    """Top-`high_frac` by the noisy score. The threshold is taken on the
    SCORE, which is what a programme ranking on its own score would do --
    not on individual_risk, which it cannot see."""
    s = np.log(np.asarray(risk, float))
    if sigma > 0:
        s = s + sigma * rng.standard_normal(len(s))
    thr = float(np.quantile(s, 1.0 - high_frac))
    return s, (s >= thr), thr


def calibrate_sigma(risk, crc_death, target_c, high_frac, seed=7, tol=2e-4):
    """Bisect on sigma for a target c-statistic. Monotone: more noise can
    only move the score toward chance."""
    rng = np.random.default_rng(seed)
    sd = float(np.std(np.log(risk)))
    lo, hi = 0.0, 40.0 * max(sd, 1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s, _, _ = label_for_sigma(risk, mid, high_frac, np.random.default_rng(seed))
        c = c_statistic(s, crc_death)
        if abs(c - target_c) < tol:
            return mid, c
        if c > target_c:      # too discriminating -> add noise
            lo = mid
        else:
            hi = mid
    return mid, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=400_000)
    ap.add_argument('--seed', type=int, default=24680)
    ap.add_argument('--high_frac', type=float, default=0.20)
    ap.add_argument('--target-c', type=float, nargs='*', default=[0.63, 0.58],
                    help='c-statistics for lifetime CRC death to calibrate to. '
                         '0.63 is roughly Jeon 2018 environmental+genetic; 0.58 is '
                         'a lifestyle-only score. sigma=0 (the oracle) is always '
                         'included as the reference arm.')
    ap.add_argument('--noise-seed', type=int, default=7)
    ap.add_argument('--recompute', action='store_true')
    a = ap.parse_args()

    if os.path.exists(CACHE) and not a.recompute:
        z = np.load(CACHE, allow_pickle=True)
        if int(z['n']) == a.n and int(z['seed']) == a.seed:
            print(f'reusing cached cohort {CACHE}')
            qstate, fdq, gender, risk = z['qstate'], z['fdq'], z['gender'], z['risk']
        else:
            os.remove(CACHE)
    if not os.path.exists(CACHE) or a.recompute:
        print(f'Simulating n={a.n} (no screening, quarterly, 9-state) ...', flush=True)
        t0 = time.time()
        qstate, fdq, gender, risk = simulate_with_sex_risk(a.n, a.seed)
        np.savez_compressed(CACHE, qstate=qstate, fdq=fdq, gender=gender, risk=risk,
                            n=a.n, seed=a.seed)
        print(f'  cached {CACHE} ({time.time()-t0:.0f}s)', flush=True)

    crc_death = crc_death_from_qstate(qstate)
    male = gender == 1
    print(f'\ncohort n={len(risk):,}  lifetime CRC death {crc_death.mean()*100:.2f}%  '
          f'sd(log individual_risk)={np.std(np.log(risk)):.3f}')

    arms = [('oracle', 0.0)]
    for tc in a.target_c:
        sig, got = calibrate_sigma(risk, crc_death, tc, a.high_frac, a.noise_seed)
        arms.append((f'c{tc:.2f}'.replace('.', ''), sig))
        print(f'  target c={tc:.2f} -> sigma={sig:.4f} (achieved c={got:.4f})')

    print(f"\n{'label':<10}{'sigma':>8}{'c-stat':>9}{'RR(hi/lo)':>11}"
          f"{'sens':>8}{'PPV':>8}{'CRCdeath hi':>13}{'lo':>9}")
    summary = {}
    true_hi = risk >= np.quantile(risk, 1 - a.high_frac)
    for name, sigma in arms:
        s, hi, thr = label_for_sigma(risk, sigma, a.high_frac,
                                     np.random.default_rng(a.noise_seed))
        c = c_statistic(s, crc_death)
        dh, dl = crc_death[hi].mean(), crc_death[~hi].mean()
        sens = (hi & true_hi).sum() / max(true_hi.sum(), 1)
        ppv = (hi & true_hi).sum() / max(hi.sum(), 1)
        print(f'{name:<10}{sigma:>8.4f}{c:>9.4f}{dh/dl:>11.2f}{sens:>8.3f}{ppv:>8.3f}'
              f'{dh*1e5:>13.0f}{dl*1e5:>9.0f}')

        combos = {'male_low': male & ~hi, 'male_high': male & hi,
                  'female_low': ~male & ~hi, 'female_high': ~male & hi}
        out = {}
        for cname, mask in combos.items():
            P, d = estimate_for_mask(qstate, fdq, mask)
            out[f'P_undetected_{cname}'] = P
            out[f'd_symp_{cname}'] = d

        # risk_threshold is stored on the SCORE scale here, not on
        # individual_risk -- surv_fair_compare's oracle cross-check does not
        # apply, so the noisy driver carries sigma instead and re-derives the
        # cut from its own cohort.
        path = os.path.join(RES, f'transitions_9state_noisyrisk_{name}.npz')
        np.savez_compressed(
            path, ages=np.arange(AGE_MIN, AGE_MAX + 1),
            risk_threshold=thr, score_scale='log_individual_risk_plus_gaussian',
            sigma=sigma, c_statistic=c, noise_seed=a.noise_seed,
            rr_high_low=dh / dl, sensitivity_vs_true_top=sens, ppv_vs_true_top=ppv,
            frac_female=float((~male).mean()),
            frac_high_male=float(hi[male].mean()),
            frac_high_female=float(hi[~male].mean()),
            state_names=np.array(STATE9_NAMES), n_patients=len(risk),
            high_frac=a.high_frac, **out)
        summary[name] = path
        print(f'           saved {os.path.basename(path)}')

    print('\n' + '\n'.join(f'{k}: {v}' for k, v in summary.items()))


if __name__ == '__main__':
    main()
