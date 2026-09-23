"""Per-person attendance (adherence) processes shared by the engine hooks.

The original scenarios (dp/run_adherence.py, Table 10) draw every invitation
independently with one population-wide probability alpha -- missing
completely at random. US longitudinal data (Kaiser FIT programme, VA and
Group Health FOBT cohorts, the Inadomi/Liang RCT) show instead that
attendance is *habitual*: a minority never attends, the majority attends
almost always, and the strongest predictor of attending is having attended
last time. The processes below cover the standard ways of modelling that.

spec (JSON-able; a bare float is the legacy homogeneous alpha):
  {'kind': 'homog',   'alpha': 0.7}
  {'kind': 'beta',    'mean': 0.7, 'kappa': 1.0, 'rho': 0.0}
        p_i ~ Beta(mean*kappa, (1-mean)*kappa) drawn ONCE per person; every
        invitation is then Bernoulli(p_i). kappa -> inf is the homogeneous
        model, kappa < 1 gives the U-shaped never/always population.
  {'kind': 'mixture', 'probs': [0.15, 0.9], 'weights': [0.3, 0.7], 'rho': 0.0}
        latent adherence classes with fixed per-class propensity (the model
        the adherence-aware policy of dp/adherence_model.py reasons about).
  {'kind': 'markov',  'p_first': 0.5, 'p_aa': 0.85, 'p_ma': 0.25}
        state-dependent: P(attend | first invitation), P(attend | attended
        the previous invitation), P(attend | missed the previous one).
rho (beta / mixture): Gaussian-copula correlation between the person's
CMOST individual_risk rank and NON-adherence (rho > 0: higher-risk people
are less likely to attend, the missing-not-at-random scenario).

RNG discipline: propensities come from a per-chunk stream seeded by the
chunk seed only (identical for every arm run on the same chunk, so subgroup
contrasts are paired); per-invitation draws come from the hook-local stream
the legacy scenarios used (seed + 4242) so the cached homogeneous arms are
reproduced bit for bit.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import beta as beta_dist

PROPENSITY_SEED = 9797
INVITE_SEED = 4242


def normalize_spec(spec):
    if spec is None:
        return dict(kind='homog', alpha=1.0)
    if isinstance(spec, (int, float, np.floating)):
        return dict(kind='homog', alpha=float(spec))
    d = dict(spec)
    d.setdefault('kind', 'homog')
    return d


def spec_mean(spec):
    """Population-mean per-invitation attendance implied by the spec (for the
    markov kind: the stationary attendance of the two-state chain)."""
    s = normalize_spec(spec)
    k = s['kind']
    if k == 'homog':
        return float(s['alpha'])
    if k == 'beta':
        return float(s['mean'])
    if k == 'mixture':
        return float(np.dot(s['probs'], s['weights']) / np.sum(s['weights']))
    if k == 'markov':
        paa, pma = float(s['p_aa']), float(s['p_ma'])
        return pma / (1.0 - paa + pma)
    raise ValueError(k)


class AdherenceProcess:
    def __init__(self, spec, n, seed, risk=None):
        self.spec = normalize_spec(spec)
        self.kind = self.spec['kind']
        self.n = int(n)
        self.rng_inv = np.random.default_rng(int(seed) + INVITE_SEED)
        self.n_invited = np.zeros(n, dtype=np.int16)
        self.n_attended = np.zeros(n, dtype=np.int16)
        self.p = None            # per-person propensity (beta / mixture)
        self.cls = None          # latent class (mixture)
        self.last = np.full(n, -1, dtype=np.int8)     # markov: -1 none, 0 missed, 1 attended
        if self.kind in ('beta', 'mixture'):
            self.p, self.cls = propensities(self.spec, n, seed, risk)
        elif self.kind == 'homog':
            self.alpha = float(self.spec['alpha'])
        elif self.kind == 'markov':
            self.p_first, self.p_aa, self.p_ma = (float(self.spec['p_first']), float(self.spec['p_aa']),
                                                  float(self.spec['p_ma']))
        else:
            raise ValueError(f'unknown adherence kind {self.kind}')

    def prob(self, z):
        if self.kind == 'homog':
            return self.alpha
        if self.kind == 'markov':
            l = self.last[z]
            return self.p_first if l < 0 else (self.p_aa if l == 1 else self.p_ma)
        return float(self.p[z])

    def draw(self, z) -> bool:
        """Invite person z now; returns True if the invitation is attended."""
        self.n_invited[z] += 1
        pr = self.prob(z)
        # legacy stream: no draw at all under full adherence, else rng >= alpha is a no-show
        att = True if pr >= 1.0 else bool(self.rng_inv.random() < pr)
        if self.kind == 'markov':
            self.last[z] = 1 if att else 0
        if att:
            self.n_attended[z] += 1
        return att

    def person_arrays(self):
        out = dict(adh_invited=self.n_invited.copy(), adh_attended=self.n_attended.copy())
        if self.p is not None:
            out['adh_p'] = self.p.astype(np.float32)
        if self.cls is not None:
            out['adh_cls'] = self.cls.astype(np.int8)
        return out


def _copula_uniform(rng, n, risk, rho):
    """u ~ U(0,1) per person; with rho > 0 and a risk vector, u is
    Gaussian-copula correlated with the risk RANK so that higher risk maps to
    LOWER u (lower propensity / lower class)."""
    eps = rng.standard_normal(n)
    if not rho or risk is None:
        return ndtr(eps)
    r = np.asarray(risk, float)
    jitter = rng.random(n) * 1e-9 * (r.std() + 1e-12)          # random tie-break of the pooled risk values
    order = np.argsort(r + jitter, kind='stable')
    rank = np.empty(n)
    rank[order] = np.arange(n)
    z_r = ndtri((rank + 0.5) / n)
    z = -float(rho) * z_r + np.sqrt(1.0 - float(rho) ** 2) * eps
    return ndtr(z)


def propensities(spec, n, seed, risk=None):
    """(p, cls) for beta / mixture specs; deterministic in (spec, seed, n, risk)."""
    s = normalize_spec(spec)
    rng = np.random.default_rng(int(seed) + PROPENSITY_SEED)
    u = _copula_uniform(rng, n, risk, float(s.get('rho', 0.0)))
    if s['kind'] == 'beta':
        mean, kappa = float(s['mean']), float(s['kappa'])
        a, b = mean * kappa, (1.0 - mean) * kappa
        p = beta_dist.ppf(u, a, b)
        return np.clip(p, 0.0, 1.0), None
    if s['kind'] == 'mixture':
        probs = np.asarray(s['probs'], float)
        w = np.asarray(s['weights'], float)
        w = w / w.sum()
        cls = np.searchsorted(np.cumsum(w)[:-1], u, side='right').astype(np.int8)
        return probs[cls], cls
    raise ValueError(s['kind'])


def propensity_groups(p, edges=(0.3, 0.7)):
    """0 = low (< edges[0]), 1 = mid, 2 = high (>= edges[-1])."""
    return np.searchsorted(np.asarray(edges, float), np.asarray(p, float), side='right').astype(np.int8)
