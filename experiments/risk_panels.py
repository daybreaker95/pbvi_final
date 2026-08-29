"""
risk_panels.py
==============

Risk-factor / risk-test PANELS: named combinations of baseline markers a patient
could plausibly be characterised by *before* the first colonoscopy, and the
discrimination (AUROC for the latent high-adenoma-risk class) each combination
reaches.

Why this exists.  `risk_factors.py` sweeps discrimination as an abstract knob (a
Gaussian score whose AUC we dial), and `prs_targeting.py` picks three points on
that knob (0.67 / 0.80 / 0.90 plus an oracle).  Neither says *which* real
combination of risk factors and tests would put a screening programme at a given
AUROC.  This module supplies that mapping: each marker is a separate observation
with its own class-conditional distribution, markers are combined by naive Bayes
into a calibrated posterior P(high | markers), and the resulting panel AUROC is
*measured*, not assumed.

Two kinds of marker:

  * BINARY -- P(+ | high), P(+ | low).  Family history and prior-adenoma history,
    exactly the two factors already used in `risk_factors.py` (FH 0.30/0.10,
    PA 0.25/0.06), kept identical so the ladder's second rung reproduces that
    experiment's AUC ~ 0.67 anchor.

  * CONTINUOUS -- a Gaussian score separated by d = sqrt(2) * Phi^{-1}(AUC)
    between the classes, i.e. parameterised directly by its own stand-alone
    AUROC.  Its log-likelihood ratio is d * s, so independent continuous markers
    combine as d_total^2 = sum_i d_i^2.

IMPORTANT -- what these AUROCs mean.  The target label here is CMOST's *latent
lifelong adenoma-risk class* (individual_risk above/below the top-25% cut), NOT
prevalent cancer.  Published discrimination figures for these assays are mostly
for *detecting existing neoplasia*, a different (and generally easier) label.
The per-marker numbers below are therefore deliberately conservative SCENARIO
ANCHORS -- "a test of this kind, used as a risk stratifier, reaching this much
risk-class discrimination" -- not reproductions of any single reported statistic.
The scientific content of the experiment is the *sweep* (`auroc_sweep.py`
Track A); the panels locate plausible, nameable points on it.

Ladder rationale (each rung adds one modality on top of the previous):

  P1  family history alone                      -- what a questionnaire gives free
  P2  + prior-adenoma history                   -- today's clinical baseline
  P3  + lifestyle/environmental E-score         -- Jeon et al. 2018 E-score factors
  P4  + PRS, current generation (~140 loci)     -- Jeon 2018 / Thomas 2020 era
  P5  PRS upgraded to genome-wide/multi-ancestry -- the near-term PRS improvement
  P6  + faecal microbiome signature
  P7  + quantitative faecal haemoglobin (f-Hb) used as a continuous stratifier
        rather than a positive/negative FIT
  P8  + multi-target stool DNA/RNA panel
  P9  + blood-based methylated cell-free DNA score

P9 lands at AUROC ~ 0.85, the ceiling this study sweeps to.
"""

from __future__ import annotations

import math

PRIOR = 0.25          # population fraction in the high-risk class (frac_high)


def _erfinv(y):
    """Inverse error function: rational initial guess + Newton polish."""
    if y <= -1.0 or y >= 1.0:
        raise ValueError('erfinv domain')
    a = 0.147
    ln1my2 = math.log(1.0 - y * y)
    t1 = 2.0 / (math.pi * a) + ln1my2 / 2.0
    x = math.copysign(math.sqrt(max(0.0, math.sqrt(t1 * t1 - ln1my2 / a) - t1)), y)
    for _ in range(3):
        err = math.erf(x) - y
        x -= err / (2.0 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def auc_to_d(auc):
    """Gaussian class-mean separation reproducing a given stand-alone AUROC:
    d = sqrt(2) * Phi^{-1}(auc), with Phi^{-1}(p) = sqrt(2) * erfinv(2p-1)."""
    return 2.0 * _erfinv(2.0 * auc - 1.0)


# --- marker menu ------------------------------------------------------------
# ('bin', P(+|high), P(+|low))  |  ('gauss', stand-alone AUROC)
MARKERS = {
    'fh':       ('bin', 0.30, 0.10),   # family history of CRC, 1st-degree relative
    'pa':       ('bin', 0.25, 0.06),   # personal history of adenoma
    'env':      ('gauss', 0.58),       # lifestyle/environmental E-score
    'prs_now':  ('gauss', 0.62),       # PRS, ~140 established CRC loci
    'prs_next': ('gauss', 0.68),       # PRS, genome-wide / multi-ancestry
    'microb':   ('gauss', 0.59),       # faecal microbiome signature
    'fit_q':    ('gauss', 0.65),       # quantitative faecal haemoglobin (f-Hb)
    'mtsdna':   ('gauss', 0.66),       # multi-target stool DNA/RNA panel
    'cfdna':    ('gauss', 0.655),      # blood methylated cell-free DNA score
}

MARKER_LABEL = {
    'fh': 'family history (1st-degree)',
    'pa': 'prior-adenoma history',
    'env': 'lifestyle/environmental E-score',
    'prs_now': 'PRS, ~140 loci',
    'prs_next': 'PRS, genome-wide/multi-ancestry',
    'microb': 'faecal microbiome signature',
    'fit_q': 'quantitative faecal haemoglobin',
    'mtsdna': 'multi-target stool DNA/RNA',
    'cfdna': 'blood methylated cfDNA score',
}

# Canonical draw order.  Every panel draws EVERY marker's variate in this order
# and ignores the ones it does not use, so a given patient sees common random
# numbers across panels (paired comparison between rungs of the ladder).
MARKER_ORDER = ['fh', 'pa', 'env', 'prs_now', 'prs_next', 'microb',
                'fit_q', 'mtsdna', 'cfdna']

PANELS = [
    ('P1 FH', 'family history alone',
     ['fh']),
    ('P2 FH+PA', "+ prior-adenoma history (today's clinical baseline)",
     ['fh', 'pa']),
    ('P3 +ENV', '+ lifestyle/environmental E-score',
     ['fh', 'pa', 'env']),
    ('P4 +PRS-140', '+ PRS, current generation (~140 loci)',
     ['fh', 'pa', 'env', 'prs_now']),
    ('P5 PRS-GW', 'PRS upgraded to genome-wide / multi-ancestry',
     ['fh', 'pa', 'env', 'prs_next']),
    ('P6 +microbiome', '+ faecal microbiome signature',
     ['fh', 'pa', 'env', 'prs_next', 'microb']),
    ('P7 +FIT f-Hb', '+ quantitative faecal haemoglobin as a stratifier',
     ['fh', 'pa', 'env', 'prs_next', 'microb', 'fit_q']),
    ('P8 +mt-sDNA', '+ multi-target stool DNA/RNA panel',
     ['fh', 'pa', 'env', 'prs_next', 'microb', 'fit_q', 'mtsdna']),
    ('P9 +cfDNA', '+ blood methylated cell-free DNA score',
     ['fh', 'pa', 'env', 'prs_next', 'microb', 'fit_q', 'mtsdna', 'cfdna']),
]
PANEL_NAMES = [name for name, _, _ in PANELS]
PANEL_BY_NAME = {name: markers for name, _, markers in PANELS}
PANEL_DESC = {name: desc for name, desc, _ in PANELS}

_D = {k: auc_to_d(v[1]) for k, v in MARKERS.items() if v[0] == 'gauss'}


def panel_posterior(true_high, markers, rng, prior=PRIOR):
    """Naive-Bayes P(high | panel readings) for one patient."""
    use = set(markers)
    logit = math.log(prior / (1.0 - prior))
    for key in MARKER_ORDER:
        spec = MARKERS[key]
        if spec[0] == 'bin':
            p1, p0 = spec[1], spec[2]
            pos = rng.random() < (p1 if true_high else p0)
            if key in use:
                logit += math.log(p1 / p0) if pos else math.log((1 - p1) / (1 - p0))
        else:
            d = _D[key]
            s = rng.normal(d / 2 if true_high else -d / 2, 1.0)
            if key in use:
                logit += d * s
    return 1.0 / (1.0 + math.exp(-logit))


def make_panel_fn(panel_name, prior=PRIOR):
    """Posterior function for a named panel, for eval_frontier(post_fn=...)."""
    markers = PANEL_BY_NAME[panel_name]

    def fn(true_high, rng):
        return panel_posterior(true_high, markers, rng, prior)
    return fn


def gauss_fn(auc, prior=PRIOR):
    """Track-A signal: one abstract score at exactly this AUROC."""
    d = auc_to_d(auc)

    def fn(true_high, rng):
        s = rng.normal(d / 2 if true_high else -d / 2, 1.0)
        logit = math.log(prior / (1.0 - prior)) + d * s
        return 1.0 / (1.0 + math.exp(-logit))
    return fn


if __name__ == '__main__':
    import os
    import sys
    import numpy as np
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from experiments.risk_factors import empirical_auc

    n = 200000
    lab = np.random.default_rng(0).random(n) < PRIOR
    rngs = [np.random.default_rng(555 + i) for i in range(n)]

    print(f"{'marker':<10}{'stand-alone AUROC':>19}   description")
    for key in MARKER_ORDER:
        sc = np.array([panel_posterior(bool(h), [key], r)
                       for h, r in zip(lab, (np.random.default_rng(555 + i)
                                             for i in range(n)))])
        print(f"{key:<10}{empirical_auc(sc, lab):>19.4f}   {MARKER_LABEL[key]}")

    print(f"\n{'panel':<17}{'#':>3}{'AUROC':>9}   description")
    for name, desc, mk in PANELS:
        sc = np.array([panel_posterior(bool(h), mk, r) for h, r in zip(lab, rngs)])
        print(f"{name:<17}{len(mk):>3}{empirical_auc(sc, lab):>9.4f}   {desc}")
        rngs = [np.random.default_rng(555 + i) for i in range(n)]
