"""A baseline risk score of finite discrimination, observed once at age 40.

The perfect-classifier arm of the main analysis (the risk class revealed
exactly) is an upper bound no real instrument reaches. Here the programme
instead observes

    S = log(individual_risk) + sigma * N(0, 1)

drawn once per person, and starts each person's belief from the posterior
over latent risk classes implied by their score band. sigma is calibrated
so that S attains a target discrimination for lifetime CRC, measured the
way risk models are usually reported: the AUC of S for separating persons
who are diagnosed with CRC over their lifetime (absent screening) from
those who are not.

The AUC has a ceiling below 1 even at sigma = 0, because knowing CMOST's
risk multiplier exactly still leaves the disease stochastic; that ceiling
(about 0.75) is itself a reportable quantity.

python -m dp.riskscore --calibrate      # sigma for each target AUC
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .common import RES, RUNS, risk_class_of, load_settings_pool, risk_thresholds

CACHE = os.path.join(RES, 'riskscore_calibration.json')
# the uninformative control's noise: large enough to swamp the risk pool's own
# spread (log-risk ranges over about 3 units), finite so the solver stays well
# conditioned. It leaves a measurable trace of signal - see `control_auc`.
CONTROL_SIGMA = 60.0


# ---------------------------------------------------------------------------
def auc(score, y):
    """Rank-based AUC (probability a random case outranks a random control)."""
    score = np.asarray(score, float)
    y = np.asarray(y, bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), float)
    s_sorted = score[order]
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # average ranks within ties
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_cohort(max_chunks=None, outcome='diagnosed'):
    """Never-screened cohort: (individual_risk, outcome, sex)."""
    risk, out, sex = [], [], []
    paths = sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', '*.npz')))
    if max_chunks:
        paths = paths[:max_chunks]
    for p in paths:
        d = np.load(p, allow_pickle=True)
        risk.append(d['risk'].astype(float)); out.append(d[outcome]); sex.append(d['sex'])
    return np.concatenate(risk), np.concatenate(out).astype(bool), np.concatenate(sex)


def score_of(risk, sigma, rng):
    return np.log(np.asarray(risk, float)) + sigma * rng.standard_normal(len(risk))


def auc_within_sex(risk, y, sex, sigma, seed=0):
    """AUC pooled over the sex-specific AUCs (weighted by case count), so the
    score is credited only for discrimination it adds WITHIN sex - sex is
    already observed by the policy and must not be laundered through the
    score."""
    rng = np.random.default_rng(seed)
    S = score_of(risk, sigma, rng)
    num = den = 0.0
    for sx in (1, 2):
        m = sex == sx
        a = auc(S[m], y[m]); n1 = int(y[m].sum())
        num += a * n1; den += n1
    return num / den


def calibrate(targets=(0.55, 0.60, 0.65, 0.70), max_chunks=20, seed=0, tol=1e-4):
    """sigma achieving each target within-sex AUC (bisection)."""
    risk, y, sex = load_cohort(max_chunks)
    ceiling = auc_within_sex(risk, y, sex, 0.0, seed)
    out = {'ceiling_auc': ceiling, 'n': int(len(risk)), 'sigmas': {}}
    for t in targets:
        if t >= ceiling:
            continue
        lo, hi = 0.0, 40.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            a = auc_within_sex(risk, y, sex, mid, seed)
            if a > t:
                lo = mid
            else:
                hi = mid
            if abs(a - t) < tol:
                break
        sg = 0.5 * (lo + hi)
        out['sigmas'][f'{t:.2f}'] = dict(sigma=sg, auc=auc_within_sex(risk, y, sex, sg, seed))
    out['control'] = dict(sigma=CONTROL_SIGMA,
                          auc=auc_within_sex(risk, y, sex, CONTROL_SIGMA, seed))
    return out


def control_auc(default=0.5):
    """The uninformative control's MEASURED discrimination.

    sigma = CONTROL_SIGMA is a numerical stand-in for sigma -> infinity, not
    infinity, so a trace of signal survives it; reporting a nominal 0.500
    would overstate how uninformative the control is."""
    try:
        return float(json.load(open(CACHE))['control']['auc'])
    except (OSError, KeyError, ValueError):
        return default


# ---------------------------------------------------------------------------
def band_edges(n_bands):
    """Population quantile cut points for the score bands."""
    return [(i + 1) / n_bands for i in range(n_bands - 1)]


# ---------------------------------------------------------------------------
# Exact score-conditional belief.
#
# Binning the score before forming the belief throws away exactly the
# resolution the scenario is about: at sigma = 0 a top DECILE still mixes
# class 2 with classes 3-5, so a decile-conditioned "perfect score" would be
# less informative than the paper's perfect-class arm despite a higher AUC.
# Instead the belief is conditioned on the score VALUE, in closed form over
# CMOST's own 500-value individual_risk pool:
#
#   P(atom i | S = s) \propto n_i * phi((s - log r_i) / sigma)
#
# and the age-40 clinical distribution is taken from the never-screened
# cohort SEPARATELY FOR EACH ATOM, so the belief does not assume the score is
# independent of the clinical state given the class (it is not: within one
# class, higher-risk atoms carry twice the age-40 adenoma prevalence).
# ---------------------------------------------------------------------------
ATOMS = os.path.join(RES, 'riskscore_atoms.npz')


def nearest_atom(values, pool):
    """Index of the nearest pool atom for each value (pool must be sorted)."""
    v = np.asarray(values, float)
    j = np.searchsorted(pool, v)
    j = np.clip(j, 1, len(pool) - 1)
    left, right = pool[j - 1], pool[j]
    return np.where(np.abs(v - left) <= np.abs(right - v), j - 1, j).astype(np.int64)


def build_atom_table(max_chunks=None, cuts=(0.5, 0.8, 0.95, 0.965, 0.98)):
    """Age-40 joint counts of (risk atom, clinical state) per sex, from the
    never-screened cohort, restricted to persons alive and undiagnosed at the
    age-40 decision epoch."""
    from .common import MAP18_TO_CLIN, NC
    pool = np.unique(load_settings_pool())
    counts = np.zeros((2, len(pool), NC))
    paths = sorted(glob.glob(os.path.join(RUNS, 'nh_quarterly', '*.npz')))
    if max_chunks:
        paths = paths[:max_chunks]
    n_seen = 0
    for p in paths:
        d = np.load(p, allow_pickle=True)
        st = MAP18_TO_CLIN[d['qr'][157]]          # age-40 decision-time state
        ok = st >= 0
        # the stored risk is float32, so an exact match against the float64
        # settings pool drops ~5 % of people - and disproportionately the
        # large-magnitude (high-risk) atoms, where float32 spacing is coarsest.
        # Assign each person to the NEAREST atom instead.
        ai = nearest_atom(d['risk'].astype(float), pool)
        sx = d['sex'].astype(int) - 1
        np.add.at(counts, (sx[ok], ai[ok], st[ok]), 1)
        n_seen += int(ok.sum())
    assert counts.sum() == n_seen, 'atom assignment lost persons'
    # class cuts must be the kernels' own: quantiles of the FULL 500-entry
    # settings pool (with its duplicate multipliers), not of the 476 distinct
    # atom values, or the class boundaries shift and the belief no longer
    # mixes back to the population prior
    thr = risk_thresholds(load_settings_pool(), cuts) if len(cuts) else np.zeros(0)
    cls = risk_class_of(pool, thr)
    np.savez_compressed(ATOMS, counts=counts, pool=pool, cls=cls, thr=thr, cuts=np.array(cuts))
    return counts, pool, cls


def load_atoms():
    if not os.path.exists(ATOMS):
        build_atom_table()
    z = np.load(ATOMS, allow_pickle=True)
    return z['counts'], z['pool'], z['cls']


def _population_scores(sigma, n_mc, seed):
    """Monte-Carlo draw of the population score distribution. Atoms are drawn
    with their EMPIRICAL frequency in the age-40 population, not uniformly over
    distinct values: CMOST's pool repeats some multipliers (25 of 500 entries
    share one value) and survival to 40 differs by risk, so uniform sampling
    over the 476 distinct atoms under-weights the high-risk tail threefold."""
    counts, pool, _ = load_atoms()
    n_i = counts.sum(axis=(0, 2))
    p = n_i / n_i.sum()
    rng = np.random.default_rng(seed)
    r = pool[rng.choice(len(pool), size=n_mc, p=p)]
    S = np.log(r)
    if sigma > 0:
        S = S + sigma * rng.standard_normal(n_mc)
    return S


def score_grid(sigma, n_cells=2048, n_mc=4_000_000, seed=1):
    """Equal-mass cells of the population score distribution: returns the
    cell edges and the representative score of each cell."""
    S = _population_scores(sigma, n_mc, seed)
    qs = np.quantile(S, np.arange(1, n_cells) / n_cells)
    mids = np.quantile(S, (np.arange(n_cells) + 0.5) / n_cells)
    return qs, mids


def beliefs_for_scores(scores, sigma, n_class):
    """(m, n_class * NC) age-40 beliefs conditional on each score value."""
    from .common import NC
    counts, pool, cls = load_atoms()
    C = counts.sum(axis=0)                      # (atoms, NC), sex-pooled clinical mix
    n_i = C.sum(axis=1)
    lr = np.log(pool)
    s = np.asarray(scores, float)[:, None]
    if sigma > 0:
        logw = -0.5 * ((s - lr[None, :]) / sigma) ** 2
        logw -= logw.max(axis=1, keepdims=True)
        w = np.exp(logw) * n_i[None, :]
    else:                                        # sigma = 0: the atom is revealed
        j = nearest_atom(np.exp(s[:, 0]), pool)
        w = np.zeros((len(s), len(pool)))
        w[np.arange(len(s)), j] = n_i[j]
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    out = np.zeros((len(s), n_class * NC))
    frac = C / np.maximum(n_i[:, None], 1)       # P(clinical | atom)
    for c in range(n_class):
        sel = cls == c
        if not sel.any():
            continue
        out[:, c * NC:(c + 1) * NC] = w[:, sel] @ frac[sel]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-300)


def root_band_edges(n_roots=14):
    """Tail-dense quantile cuts for the PBVI roots, straddling every class
    cut (50/80/95/96.5/98th percentiles)."""
    return [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.965, 0.98, 0.99][:n_roots - 1]


def root_beliefs(sigma, n_class, n_roots=14, n_mc=2_000_000, seed=1):
    """Belief and population weight of each tail-dense score band."""
    S = _population_scores(sigma, n_mc, seed)
    qs = np.quantile(S, root_band_edges(n_roots))
    b = np.searchsorted(qs, S, side='right')
    mids, wts = [], []
    for k in range(n_roots):
        m = b == k
        if m.sum() == 0:
            continue
        mids.append(np.median(S[m])); wts.append(m.mean())
    B = beliefs_for_scores(np.array(mids), sigma, n_class)
    return B, np.array(wts), qs


def band_posteriors(sigma, cuts, n_bands=10, n_mc=4_000_000, seed=1):
    """P(latent class | score band) and P(band), by Monte Carlo over CMOST's
    own individual_risk pool crossed with the score noise.

    Returns (priors[n_bands, n_class], band_share[n_bands], edges[n_bands-1]).
    The bands are population quantiles of S, so each holds 1/n_bands of the
    population by construction. Atoms are drawn with their empirical age-40
    frequency, for the reason given in `_population_scores`: sampling the
    distinct pool values uniformly under-weights the high-risk tail and
    understates every band posterior in it."""
    pool = load_settings_pool()
    thr = risk_thresholds(pool, cuts) if len(cuts) else np.zeros(0)
    rng = np.random.default_rng(seed)
    counts, atoms, _ = load_atoms()
    n_i = counts.sum(axis=(0, 2))
    idx = rng.choice(len(atoms), size=n_mc, p=n_i / n_i.sum())
    r = atoms[idx]
    S = np.log(r) + sigma * rng.standard_normal(n_mc)
    cls = risk_class_of(r, thr)
    n_class = len(cuts) + 1
    qs = np.quantile(S, band_edges(n_bands))
    b = np.searchsorted(qs, S, side='right')
    priors = np.zeros((n_bands, n_class))
    share = np.zeros(n_bands)
    for k in range(n_bands):
        m = b == k
        share[k] = m.mean()
        c = np.bincount(cls[m], minlength=n_class).astype(float)
        priors[k] = c / max(c.sum(), 1)
    return priors, share, qs


def assign_bands(risk, sigma, edges, seed):
    """Per-person score band in the engine (own RNG: the engine's paired
    streams are untouched)."""
    rng = np.random.default_rng(seed + 9161)
    S = np.log(np.asarray(risk, float)) + sigma * rng.standard_normal(len(risk))
    return np.searchsorted(np.asarray(edges, float), S, side='right').astype(np.int16)


def assign_cells(risk, sigma, cell_edges, band_edges_, seed):
    """Per-person (belief-table cell, reporting band). The score noise uses the
    hook's OWN generator, seeded per chunk, so that the same person draws the
    SAME noise in every arm (the AUC arms are paired on the score as well as on
    the population) while the engine's own random stream is untouched."""
    rng = np.random.default_rng(seed + 9161)
    S = np.log(np.asarray(risk, float))
    if sigma > 0:
        S = S + sigma * rng.standard_normal(len(S))
    cell = np.searchsorted(np.asarray(cell_edges, float), S, side='right').astype(np.int32)
    band = np.searchsorted(np.asarray(band_edges_, float), S, side='right').astype(np.int16)
    return cell, band


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', type=float, nargs='*', default=[0.55, 0.60, 0.65, 0.70])
    ap.add_argument('--max-chunks', type=int, default=20)
    ap.add_argument('--bands', type=int, default=10)
    ap.add_argument('--cuts', type=float, nargs='*', default=[0.5, 0.8, 0.95, 0.965, 0.98])
    a = ap.parse_args()
    cal = calibrate(tuple(a.targets), max_chunks=a.max_chunks)
    print(f"AUC ceiling (sigma = 0, within sex): {cal['ceiling_auc']:.4f}  (n = {cal['n']:,})")
    for t, d in cal['sigmas'].items():
        print(f"  target AUC {t}: sigma = {d['sigma']:.4f}  (achieved {d['auc']:.4f})")
    cal['bands'] = a.bands
    cal['cuts'] = a.cuts
    cal['posteriors'] = {}
    for t, d in list(cal['sigmas'].items()) + [('ceiling', dict(sigma=0.0, auc=cal['ceiling_auc']))]:
        pri, share, edges = band_posteriors(d['sigma'], tuple(a.cuts), a.bands)
        cal['posteriors'][t] = dict(sigma=d['sigma'], priors=pri.tolist(), share=share.tolist(),
                                    edges=edges.tolist())
        top = pri[:, 3:].sum(axis=1)
        print(f"  AUC {t}: P(high-risk classes 3-5) by score decile = " +
              ' '.join(f'{v:.3f}' for v in top))
    with open(CACHE, 'w') as f:
        json.dump(cal, f, indent=1)
    print('saved', CACHE)


if __name__ == '__main__':
    main()
