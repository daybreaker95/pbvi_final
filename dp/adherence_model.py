"""Adherence-aware reduced POMDP: the latent adherence class joins the hidden
state, and a declined invitation is an observation.

Hidden state  (k, c, j): adherence class k in 0..K-1, risk class c, clinical
              state j;   S = K * S_base
Action        SCREEN now means "invite": attended with probability p_k, in
              which case the base SCREEN kernel applies (observation normal /
              adenoma / multi / advad / exit); otherwise the year is a WAIT
              year and the observation is NOSHOW.
Memory        NOSHOW leaves the observed memory on the WAIT track
              (tau + 1, same last finding), exactly as the plain policy's
              re-planning does.
Reward        lambda is charged per ATTENDED colonoscopy (lambda * p_k per
              invitation in state k) so that lambda keeps its meaning as the
              shadow price of a colonoscopy; the clinical terms are the
              p_k-weighted mix of the base SCREEN and WAIT flows.
Belief        b0 = prior(k) x b0_base; an attended / declined invitation
              multiplies the class-k block by p_k / (1 - p_k), which is how
              the policy learns a person's adherence from their history.

The block structure lets everything be built from the base model's flows by
Kronecker products, so the PBVI solver, the exact in-model evaluator and the
belief-tracking hook are reused unchanged (the in-model `colos` count is the
number of INVITATIONS; attended volume is read off the engine).
"""
from __future__ import annotations

import time

import numpy as np

from .common import SCREEN, WAIT, O_NOTEST
from .model import ReducedPOMDP, METRICS

O_NOSHOW = 6          # new observation code (after O_EXIT = 5); only this model emits it


class AdherencePOMDP(ReducedPOMDP):
    O_NOSHOW = O_NOSHOW
    # exact-tree evaluation merges beliefs rounded to 4 decimals (the solver's
    # own belief-set resolution) and drops branches below 1e-10 mass: the
    # exact 6-decimal tree grows with the whole invitation history
    tree_decimals = 4
    tree_prune = 1e-10

    def __init__(self, sex, kernels_npz, adh, **kw):
        probs = np.asarray(adh['probs'], float)
        prior = np.asarray(adh['prior'], float)
        assert len(probs) == len(prior) and abs(prior.sum() - 1) < 1e-9
        self.adh = dict(probs=[float(x) for x in probs], prior=[float(x) for x in prior])
        self.K = len(probs)
        super().__init__(sex, kernels_npz, **kw)

    # -- construction --------------------------------------------------
    def _build_flows(self):
        super()._build_flows()                   # base flows on S_base states
        Sb = self.S
        K, p = self.K, np.asarray(self.adh['probs'])
        self.S_base, self.S = Sb, K * Sb
        self.p_state = np.repeat(p, Sb)           # attendance probability of each joint state
        eye = np.eye(K)
        bd = lambda M: np.kron(eye, M)            # block diagonal
        bdw = lambda M, w: np.kron(np.diag(w), M)
        flow, Mx, succ = {m: {} for m in METRICS}, {}, {}
        for key in self.M:
            Mw = self.M[key][WAIT][O_NOTEST]
            Mx[key] = {WAIT: {O_NOTEST: bd(Mw)}}
            succ[key] = {WAIT: dict(self.succ[key][WAIT])}
            for m in METRICS:
                flow[m][key] = {WAIT: np.tile(self.flow[m][key][WAIT], K)}
            if SCREEN in self.M[key]:
                Ms = {o: bdw(M, p) for o, M in self.M[key][SCREEN].items()}
                Ms[O_NOSHOW] = bdw(Mw, 1.0 - p)
                sc = dict(self.succ[key][SCREEN])
                sc[O_NOSHOW] = self.succ[key][WAIT][O_NOTEST]
                Mx[key][SCREEN] = Ms
                succ[key][SCREEN] = sc
                for m in METRICS:
                    fs, fw = self.flow[m][key][SCREEN], self.flow[m][key][WAIT]
                    flow[m][key][SCREEN] = np.concatenate([pk * fs + (1.0 - pk) * fw for pk in p])
        self.flow, self.M, self.succ = flow, Mx, succ
        # the base caches are no longer needed (and are S_base-sized)
        self._Tw_cache.clear(); self._Ew_cache.clear(); self._K_cache.clear(); self._X_cache.clear()

    def set_objective(self, weights=None, lam=0.0):
        super().set_objective(weights, lam)       # charges lam per invitation
        for key in self.R:
            if SCREEN in self.R[key]:             # ... make it per attended colonoscopy
                self.R[key][SCREEN] = self.R[key][SCREEN] + self.lam * (1.0 - self.p_state)

    # -- beliefs -------------------------------------------------------
    def initial_belief(self, class_known=None, class_prior=None):
        b = super().initial_belief(class_known=class_known, class_prior=class_prior)   # S_base
        return np.kron(np.asarray(self.adh['prior']), b)

    def class_marginal(self, b):
        return np.asarray(b).reshape(self.K, self.n_class, self.NC).sum(axis=(0, 2))

    def clinical_marginal(self, b):
        return np.asarray(b).reshape(self.K, self.n_class, self.NC).sum(axis=(0, 1))

    def adh_marginal(self, b):
        return np.asarray(b).reshape(self.K, self.S_base).sum(axis=1)


def solve_adherence_policy(sex, kernels_npz, adh, weights=None, lam=0.0, cap=1500, rounds=4,
                           rollouts=300, eps=0.1, seed=0, p_ref=0.12, verbose=True, out=None):
    """Solve (and optionally save) an adherence-aware policy. Same PBVI
    settings as the headline high-cap policies (dp/run_pipeline.py)."""
    from .solver import PBVISolver
    t0 = time.time()
    m = AdherencePOMDP(sex, kernels_npz, adh, weights=weights, lam=lam)
    s = PBVISolver(m, cap=cap, seed=seed, verbose=verbose, ref_screen_prob=p_ref)
    pol = s.solve(rounds=rounds, rollouts=rollouts, eps=eps)
    pol.meta['adh'] = m.adh
    pol.meta['solve_sec'] = time.time() - t0
    if out:
        pol.save(out)
    return pol
