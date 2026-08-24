"""Reduced finite-horizon mixed-observability POMDP for colonoscopy
screening, grounded in the real-engine kernels of dp.estimate_kernels.

Observed state   age y, memory m = (tau = years since last colonoscopy, last finding)
Hidden state     s = (risk class c, clinical state j), j in {N, P1..P6, U1..U4};  S = n_class*11
Actions          WAIT / SCREEN at decision ages age_min..age_max
Observations     WAIT  : notest | exit
                 SCREEN: normal | adenoma | multi | advad | exit
Exits (diagnosis, death) are terminal for the decision process; their
remaining-lifetime contribution (future CRC death probability, life-years) is
folded into the one-step reward, so alpha-vectors live on the S alive-
undiagnosed states only.

Objective (maximised, undiscounted):
    sum_t [ -w_death*P(CRC death) - w_inc*P(diagnosis) - w_comp*P(complication death)
            + w_ly*(life-years) - lam*1[SCREEN] ]

Memory dynamics: WAIT at (tau, ol) -> (tau+1 (capped), ol) [never stays never];
SCREEN with non-exit outcome o -> the following natural-history year uses the
cell (tau=0, ol=o) and the next decision epoch is (tau=1, ol=o).
"""
from __future__ import annotations

import os

import numpy as np

from .common import (
    RES, NC, NX, O_NOTEST, SCREEN_OBS, WAIT, SCREEN, X_D1, X_DOTH, X_DCOMP,
)
from .kernels import (
    TAU_NEVER, TAU_CAP, N_OL, mem_index_scalar, next_tau_wait, OL_OF_OBS,
)
from .estimate_screen import BANDS, band_of

METRICS = ('death', 'inc', 'comp', 'ly')
_SIGN = dict(death=-1.0, inc=-1.0, comp=-1.0, ly=+1.0)


def mem_key(tau, ol):
    """Canonical observed-memory key used by solver/hook: (tau, ol) with tau
    in {-1 (never), 1..TAU_CAP} at decision epochs."""
    return (int(tau), int(ol) if tau != TAU_NEVER else 0)


class ReducedPOMDP:
    def __init__(self, sex: int, kernels_npz: str, age_min=40, age_max=80, life_max=99,
                 weights=None, lam=0.0):
        self.sex = sex
        sx = sex - 1
        self.kernels_npz = kernels_npz
        self.age_min, self.age_max, self.life_max = age_min, age_max, life_max
        kz = np.load(kernels_npz, allow_pickle=True)
        self.n_class = int(kz['n_class'])
        self.thr = np.asarray(kz['thr'], float)
        # clinical-axis width is carried by the kernels file so that ablation
        # variants with a coarser lesion axis (dp/ablate.py) load unchanged
        self.NC = int(kz['nc']) if 'nc' in kz.files else NC
        self.S = self.n_class * self.NC
        y0, y1 = int(kz['y0']), int(kz['y1'])
        assert y0 <= age_min and y1 >= life_max, 'kernel age range too small'
        TwA, EwA, KA, XA = kz['Tw'][sx], kz['Ew'][sx], kz['K'][sx], kz['X'][sx]   # [class, mem, age/band, ...]
        self._Tw_cache, self._Ew_cache, self._K_cache, self._X_cache = {}, {}, {}, {}
        self._TwA, self._EwA, self._KA, self._XA, self._y0 = TwA, EwA, KA, XA, y0
        Vd, Vl = kz['Vexit_death'][sx], kz['Vexit_ly'][sx]
        self.Vx = {m: {} for m in METRICS}
        for y in range(age_min, life_max + 1):
            yy = min(y, Vd.shape[0] - 1)
            self.Vx['death'][y] = np.r_[Vd[yy], 0.0, 0.0]
            self.Vx['inc'][y] = np.r_[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            self.Vx['comp'][y] = np.r_[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            self.Vx['ly'][y] = np.r_[Vl[yy], 0.0, 0.0]
        self._b0_joint = kz['b0'][sx].astype(float)
        self._mem_cells = self._enumerate_memory()
        self._Vnat_metric = {}
        self._build_flows()
        self.set_objective(weights, lam)

    # ------------------------------------------------------------------
    def _enumerate_memory(self):
        """All (tau, ol) decision-epoch memory keys + the post-screen cells."""
        cells = [mem_key(TAU_NEVER, 0)]
        for tau in range(1, TAU_CAP + 1):
            for ol in range(N_OL):
                cells.append(mem_key(tau, ol))
        return cells

    def Tw(self, y, tau, ol):
        key = (y, tau, ol)
        if key not in self._Tw_cache:
            m = mem_index_scalar(tau, ol)
            T = np.zeros((self.S, self.S)); E = np.zeros((self.S, NX))
            iy = min(y, self.life_max) - self._y0
            for c in range(self.n_class):
                blk = slice(c * self.NC, (c + 1) * self.NC)
                T[blk, blk] = self._TwA[c, m, iy]
                E[blk, :] = self._EwA[c, m, iy]
            self._Tw_cache[key] = T; self._Ew_cache[key] = E
        return self._Tw_cache[key]

    def Ew(self, y, tau, ol):
        self.Tw(y, tau, ol)
        return self._Ew_cache[(y, tau, ol)]

    def Ks(self, y, tau, ol):
        key = (y, tau, ol)
        if key not in self._K_cache:
            m = mem_index_scalar(tau, ol)
            b = int(band_of(np.array([y]))[0])
            if b < 0:
                b = 0 if y < BANDS[0][0] else len(BANDS) - 1
            Ko = {o: np.zeros((self.S, self.S)) for o in SCREEN_OBS}
            Xo = np.zeros((self.S, NX))
            for c in range(self.n_class):
                blk = slice(c * self.NC, (c + 1) * self.NC)
                for io, o in enumerate(SCREEN_OBS):
                    Ko[o][blk, blk] = self._KA[c, m, b, :, io, :]
                Xo[blk, :] = self._XA[c, m, b]
            self._K_cache[key] = Ko; self._X_cache[key] = Xo
        return self._K_cache[key]

    def Xs(self, y, tau, ol):
        self.Ks(y, tau, ol)
        return self._X_cache[(y, tau, ol)]

    # ------------------------------------------------------------------
    def _build_flows(self):
        """flow[metric][(y, tau, ol)][action] -> (S,) expected metric accrued
        in the year; M[(y,tau,ol)][action] -> {obs: (S,S)} non-exit kernels;
        succ[(y,tau,ol)][action][obs] -> next memory key."""
        self.flow = {m: {} for m in METRICS}
        self.M = {}
        self.succ = {}
        for y in range(self.age_min, self.life_max + 1):
            for (tau, ol) in self._mem_cells:
                key = (y, tau, ol)
                T = self.Tw(y, tau, ol); E = self.Ew(y, tau, ol)
                surv = T.sum(axis=1)
                fw = {m: E @ self.Vx[m][y] + (surv if m == 'ly' else 0.0) for m in METRICS}
                self.M[key] = {WAIT: {O_NOTEST: T}}
                self.succ[key] = {WAIT: {O_NOTEST: mem_key(next_tau_wait(tau), ol)}}
                for m in METRICS:
                    self.flow[m][key] = {WAIT: fw[m]}
                if y <= self.age_max:
                    Ko = self.Ks(y, tau, ol); Xo = self.Xs(y, tau, ol)
                    Ms, sc = {}, {}
                    fs = {m: Xo @ self.Vx[m][y] for m in METRICS}
                    for o in SCREEN_OBS:
                        olo = OL_OF_OBS[o]
                        T0 = self.Tw(y, 0, olo); E0 = self.Ew(y, 0, olo)
                        surv0 = T0.sum(axis=1)
                        Ms[o] = Ko[o] @ T0
                        sc[o] = mem_key(1, olo)
                        for m in METRICS:
                            fs[m] = fs[m] + Ko[o] @ (E0 @ self.Vx[m][y] + (surv0 if m == 'ly' else 0.0))
                    self.M[key][SCREEN] = Ms
                    self.succ[key][SCREEN] = sc
                    for m in METRICS:
                        self.flow[m][key][SCREEN] = fs[m]

    def set_objective(self, weights=None, lam=0.0):
        w = dict(death=1.0, inc=0.0, comp=0.0, ly=0.0)
        if weights:
            w.update(weights)
        self.weights, self.lam = w, float(lam)
        self.R = {}
        for key, fl in self.flow['death'].items():
            self.R[key] = {}
            for a in fl:
                r = sum(_SIGN[m] * w[m] * self.flow[m][key][a] for m in METRICS)
                if a == SCREEN:
                    r = r - self.lam
                self.R[key][a] = r
        self._Vnat = None

    # ------------------------------------------------------------------
    def natural_value(self, metric=None):
        """WAIT-only continuation value V[(y, tau, ol)] for y = age_min..life_max+1
        (cached per metric; the objective cache is reset by set_objective)."""
        if metric is not None and metric in self._Vnat_metric:
            return self._Vnat_metric[metric]
        V = {}
        for (tau, ol) in self._mem_cells:
            V[(self.life_max + 1, tau, ol)] = np.zeros(self.S)
        for y in range(self.life_max, self.age_min - 1, -1):
            for (tau, ol) in self._mem_cells:
                key = (y, tau, ol)
                r = self.R[key][WAIT] if metric is None else self.flow[metric][key][WAIT]
                nk = self.succ[key][WAIT][O_NOTEST]
                V[key] = r + self.M[key][WAIT][O_NOTEST] @ V[(y + 1,) + nk]
        if metric is not None:
            self._Vnat_metric[metric] = V
        return V

    def boundary(self, tau, ol):
        if self._Vnat is None:
            self._Vnat = self.natural_value()
        return self._Vnat[(self.age_max + 1,) + mem_key(tau, ol)]

    # ------------------------------------------------------------------
    def initial_belief(self, class_known=None):
        b = self._b0_joint.copy()
        if class_known is not None:
            m = np.zeros_like(b)
            blk = slice(class_known * self.NC, (class_known + 1) * self.NC)
            m[blk] = b[blk]
            b = m
        return b / b.sum()

    def initial_memory(self):
        return mem_key(TAU_NEVER, 0)

    def step(self, y, tau, ol, a, o):
        """(kernel (S,S), next memory key) for observation o after action a
        at decision epoch (y, tau, ol)."""
        y = min(max(int(y), self.age_min), self.life_max)
        key = (y, int(tau), int(ol) if tau != TAU_NEVER else 0)
        if a == SCREEN and y > self.age_max:
            a = WAIT
        return self.M[key][a][o], self.succ[key][a][o]

    def class_marginal(self, b):
        return np.array([b[c * self.NC:(c + 1) * self.NC].sum() for c in range(self.n_class)])

    def clinical_marginal(self, b):
        return np.array([b[j::self.NC].sum() for j in range(self.NC)])


# ----------------------------------------------------------------------
# Exact in-model evaluation of ANY policy by occupancy-tree propagation,
# vectorised per observed key: all tree nodes that share (age, tau, ol)
# are stacked into one (n, S) matrix; nodes with identical normalised
# beliefs (rounded) are merged -- exact, because the policy depends only on
# the normalised belief and the key, and the dynamics are linear in the
# unnormalised occupancy.
# ----------------------------------------------------------------------
def _merge_rows(U, decimals=6, max_rows=50_000):
    """Merge rows of U with identical normalised belief (rounded)."""
    if len(U) <= 1:
        return U
    mass = U.sum(axis=1)
    keep = mass > 0
    U = U[keep]; mass = mass[keep]
    if len(U) <= 1:
        return U
    keys = np.round(U / mass[:, None], decimals)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    out = np.zeros((int(inv.max()) + 1, U.shape[1]))
    np.add.at(out, inv, U)
    if len(out) > max_rows:
        m = out.sum(axis=1)
        order = np.argsort(-m)
        out = out[order[:max_rows]]
    return out


def policy_tree(model: ReducedPOMDP, action_batch, b0=None, prune=1e-12, decimals=6,
                max_rows=50_000, collect=False):
    """Propagate the occupancy tree of a policy (vectorised per key).

    action_batch(y, tau, ol, Bnorm) -> int array of actions for the rows of Bnorm.
    Returns (metrics dict, tree) where tree[(y, tau, ol)] = (U, actions) if collect.
    """
    b0 = model.initial_belief() if b0 is None else b0
    nodes = {model.initial_memory(): np.asarray(b0, float)[None, :]}
    acc = {m: 0.0 for m in METRICS}
    colos = 0.0; obj = 0.0; n_max = 1
    tree = {}
    for y in range(model.age_min, model.age_max + 1):
        nxt = {}
        for (tau, ol), U in nodes.items():
            key = (y, tau, ol)
            mass = U.sum(axis=1)
            ok = mass > prune
            if not ok.any():
                continue
            U = U[ok]; mass = mass[ok]
            acts = np.asarray(action_batch(y, tau, ol, U / mass[:, None]), dtype=np.int8)
            if collect:
                tree[key] = (U, acts)
            for a in (WAIT, SCREEN):
                sel = acts == a
                if not sel.any() or a not in model.M[key]:
                    continue
                Ua = U[sel]
                tot = Ua.sum(axis=0)
                for m in METRICS:
                    acc[m] += float(tot @ model.flow[m][key][a])
                obj += float(tot @ model.R[key][a])
                if a == SCREEN:
                    colos += float(tot.sum())
                for o, M in model.M[key][a].items():
                    V = Ua @ M
                    nk = model.succ[key][a][o]
                    nxt.setdefault(nk, []).append(V)
        nodes = {k: _merge_rows(np.vstack(v), decimals, max_rows) for k, v in nxt.items()}
        n_max = max(n_max, sum(len(v) for v in nodes.values()))
    Vn = {m: model.natural_value(m) for m in METRICS}
    for (tau, ol), U in nodes.items():
        tot = U.sum(axis=0)
        for m in METRICS:
            acc[m] += float(tot @ Vn[m][(model.age_max + 1, tau, ol)])
        obj += float(tot @ model.boundary(tau, ol))
    return dict(**acc, colos=colos, objective=obj, n_nodes_max=n_max), tree


def evaluate_policy(model: ReducedPOMDP, action_fn, b0=None, prune=1e-12, max_nodes=50_000,
                    action_batch=None):
    """Exact per-person expectations of a policy. action_fn(y, tau, ol, b) is
    applied row-wise unless a vectorised action_batch is given."""
    if action_batch is None:
        def action_batch(y, tau, ol, B):
            return np.array([action_fn(y, tau, ol, b) for b in B], dtype=np.int8)
    res, _ = policy_tree(model, action_batch, b0=b0, prune=prune, max_rows=max_nodes)
    return res


def fixed_schedule_fn(ages):
    s = set(int(a) for a in ages)
    return lambda y, tau, ol, b: SCREEN if y in s else WAIT


def evaluate_fixed(model: ReducedPOMDP, ages, b0=None):
    """Fixed schedules branch only on the last finding (memory), never on the
    belief: occupancy is tracked per memory cell -> exact and fast."""
    b0 = model.initial_belief() if b0 is None else b0
    s = set(int(a) for a in ages)
    nodes = {model.initial_memory(): np.asarray(b0, float).copy()}
    acc = {m: 0.0 for m in METRICS}
    colos = 0.0; obj = 0.0
    for y in range(model.age_min, model.age_max + 1):
        a = SCREEN if y in s else WAIT
        nxt = {}
        for (tau, ol), u in nodes.items():
            key = (y, tau, ol)
            for m in METRICS:
                acc[m] += float(u @ model.flow[m][key][a])
            obj += float(u @ model.R[key][a])
            if a == SCREEN:
                colos += u.sum()
            for o, M in model.M[key][a].items():
                nk = model.succ[key][a][o]
                nxt[nk] = nxt.get(nk, 0) + u @ M
        nodes = nxt
    Vn = {m: model.natural_value(m) for m in METRICS}
    for (tau, ol), U in nodes.items():
        for m in METRICS:
            acc[m] += float(U @ Vn[m][(model.age_max + 1, tau, ol)])
        obj += float(U @ model.boundary(tau, ol))
    return dict(**acc, colos=colos, objective=obj)


def default_kernels(tag='c3'):
    return os.path.join(RES, f'kernels_{tag}.npz')
