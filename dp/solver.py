"""Finite-horizon point-based value iteration (PBVI) for dp.model.ReducedPOMDP
(mixed-observability: belief sets and alpha-vector sets are indexed by the
observed key (age, tau, last finding)).

1. Belief sets B[key] are the closure of reachable beliefs from the initial
   belief(s) under a reference screening propensity; near-duplicates are
   merged (rounding) and every set is capped, keeping the largest reach
   weights.
2. One backward sweep over ages performs point-based Bellman backups
   (alpha-vector lower bound); the horizon is finite so the sweep is exact
   for the given belief sets.
3. The current policy's exact reachable beliefs (+ epsilon-greedy rollouts)
   are added and the sweep repeated until the policy's exact in-model
   objective stops improving.
4. A fast-informed bound (FIB) gives an upper bound / gap.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .common import SCREEN, WAIT, O_NOTEST
from .model import ReducedPOMDP, evaluate_policy, policy_tree, METRICS, mem_key


class Policy:
    def __init__(self, model: ReducedPOMDP, alphas: dict, acts: dict, meta: dict | None = None):
        self.model = model
        self.alphas = alphas      # (y, tau, ol) -> (K,S)
        self.acts = acts          # (y, tau, ol) -> (K,)
        self.meta = meta or {}

    def best_action(self, y, tau, ol, b):
        y = int(y)
        if y > self.model.age_max or y < self.model.age_min:
            return WAIT
        key = (y,) + mem_key(tau, ol)
        A = self.alphas.get(key)
        if A is None or len(A) == 0:
            return WAIT
        return int(self.acts[key][int(np.argmax(A @ b))])

    def best_action_batch(self, y, tau, ol, B):
        """Vectorised best_action for the rows of B (n,S)."""
        y = int(y)
        if y > self.model.age_max or y < self.model.age_min:
            return np.zeros(len(B), dtype=np.int8)
        key = (y,) + mem_key(tau, ol)
        A = self.alphas.get(key)
        if A is None or len(A) == 0:
            return np.zeros(len(B), dtype=np.int8)
        return self.acts[key][np.argmax(np.asarray(B) @ A.T, axis=1)].astype(np.int8)

    def value(self, y, tau, ol, b):
        key = (y,) + mem_key(tau, ol)
        return float(np.max(self.alphas[key] @ b))

    def action_fn(self):
        return lambda y, tau, ol, b: self.best_action(y, tau, ol, b)

    def evaluate(self):
        """Exact in-model metrics of this policy (vectorised)."""
        return evaluate_policy(self.model, None, action_batch=self.best_action_batch)

    def save(self, path):
        d = dict(meta=json.dumps(dict(self.meta, sex=self.model.sex, kernels_npz=self.model.kernels_npz,
                                      age_min=self.model.age_min, age_max=self.model.age_max,
                                      life_max=self.model.life_max, weights=self.model.weights,
                                      lam=self.model.lam)))
        keys = []
        for i, (key, A) in enumerate(self.alphas.items()):
            d[f'A_{i}'] = A; d[f'act_{i}'] = self.acts[key]; keys.append(key)
        d['keys'] = np.array(keys, dtype=np.int64)
        np.savez_compressed(path, **d)


def load_policy(path) -> Policy:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z['meta']))
    model = ReducedPOMDP(meta['sex'], meta['kernels_npz'], age_min=meta['age_min'], age_max=meta['age_max'],
                         life_max=meta['life_max'], weights=meta['weights'], lam=meta['lam'])
    alphas, acts = {}, {}
    for i, key in enumerate(z['keys']):
        k = tuple(int(v) for v in key)
        alphas[k] = z[f'A_{i}']; acts[k] = z[f'act_{i}']
    return Policy(model, alphas, acts, meta)


# ----------------------------------------------------------------------
class PBVISolver:
    def __init__(self, model: ReducedPOMDP, cap=2000, round_decimals=4, seed=0,
                 ref_screen_prob=0.12, init_beliefs=None, verbose=True):
        self.m = model
        self.cap = cap
        self.dec = round_decimals
        self.rng = np.random.default_rng(seed)
        self.p_ref = ref_screen_prob
        self.verbose = verbose
        self.roots = [model.initial_belief()] if init_beliefs is None else list(init_beliefs)
        self.B = {}       # key -> (n,S)
        self.W = {}       # key -> (n,)
        self.alphas, self.acts = {}, {}
        self.history = []

    # -- belief sets -----------------------------------------------------
    def _key(self, b):
        return np.round(b, self.dec).tobytes()

    def _merge_into(self, key, beliefs, weights):
        """Merge (beliefs, weights) into B[key], deduplicating by rounded
        belief (fully vectorised: one np.unique over the stacked matrix)."""
        B_new = np.asarray(beliefs, float)
        if B_new.ndim == 1:
            B_new = B_new[None, :]
        if len(B_new) == 0:
            return
        w_new = np.asarray(weights, float)
        if key in self.B:
            B_all = np.vstack([self.B[key], B_new])
            w_all = np.concatenate([self.W[key], w_new])
        else:
            B_all, w_all = B_new, w_new
        keys = np.round(B_all, self.dec)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        n_u = len(uniq)
        w = np.zeros(n_u)
        np.add.at(w, inv, w_all)
        # keep the first representative row of each group (exact, not rounded)
        first = np.full(n_u, len(inv), dtype=np.int64)
        np.minimum.at(first, inv, np.arange(len(inv)))
        B_u = B_all[first]
        if n_u > self.cap:
            top = np.argsort(-w)[:self.cap]
            B_u, w = B_u[top], w[top]
        self.B[key] = B_u
        self.W[key] = w

    def _keys_at(self, y):
        return [k for k in self.B if k[0] == y]

    def build_closure(self):
        m = self.m
        self.B, self.W = {}, {}
        root = (m.age_min,) + m.initial_memory()
        self._merge_into(root, self.roots, [1.0] * len(self.roots))
        for y in range(m.age_min, m.age_max):
            for key in self._keys_at(y):
                B, Wt = self.B[key], self.W[key]
                for a, pa in ((WAIT, 1 - self.p_ref), (SCREEN, self.p_ref)):
                    for o, M in m.M[key][a].items():
                        Bo = B @ M
                        po = Bo.sum(axis=1)
                        ok = po > 1e-10
                        if ok.any():
                            nk = (y + 1,) + m.succ[key][a][o]
                            self._merge_into(nk, Bo[ok] / po[ok, None], Wt[ok] * po[ok] * pa)

    def add_policy_reachable(self, policy: 'Policy', eps=0.0, n_rollouts=0, prune=1e-8):
        """Exact reachable set of the policy (vectorised tree), weighted by
        reach probability, plus optional epsilon-greedy sampled rollouts."""
        m = self.m
        _, tree = policy_tree(m, policy.best_action_batch, collect=True, prune=prune)
        for key, (U, acts) in tree.items():
            mass = U.sum(axis=1)
            self._merge_into(key, U / mass[:, None], mass)
        for _ in range(n_rollouts):
            b = self.roots[self.rng.integers(len(self.roots))]
            tau, ol = m.initial_memory()
            for y in range(m.age_min, m.age_max + 1):
                key = (y, tau, ol)
                self._merge_into(key, [b], [1e-3])
                if y == m.age_max:
                    break
                a = policy.best_action(y, tau, ol, b)
                if self.rng.random() < eps:
                    a = 1 - a
                kern = m.M[key][a]
                obs = list(kern.keys())
                probs = np.array([float((b @ kern[o]).sum()) for o in obs])
                tot = probs.sum()
                if tot <= 1e-12:
                    break
                o = obs[self.rng.choice(len(obs), p=probs / tot)]
                v = b @ kern[o]
                b = v / v.sum()
                tau, ol = m.succ[key][a][o]

    # -- backups ---------------------------------------------------------
    def _backup(self, key, B, A_next_of):
        """Point-based backup of beliefs B (n,S) at key; A_next_of(next_key)
        returns the alpha set at the successor key."""
        m = self.m
        nB, S = B.shape
        best_val = np.full(nB, -np.inf)
        best_alpha = np.zeros((nB, S)); best_act = np.zeros(nB, dtype=np.int8)
        for a, kern in m.M[key].items():
            alpha = np.tile(m.R[key][a], (nB, 1))
            for o, M in kern.items():
                A_next = A_next_of((key[0] + 1,) + m.succ[key][a][o])
                Bo = B @ M
                idx = np.argmax(Bo @ A_next.T, axis=1)
                alpha += A_next[idx] @ M.T
            vals = np.einsum('ms,ms->m', alpha, B)
            upd = vals > best_val + 1e-12
            best_val[upd] = vals[upd]; best_alpha[upd] = alpha[upd]; best_act[upd] = a
        return best_alpha, best_act

    def sweep(self):
        m = self.m
        self.alphas, self.acts = {}, {}

        if m._Vnat is None:
            m._Vnat = m.natural_value()
        Vnat = m._Vnat

        def A_next_of(nk):
            if nk[0] > m.age_max:
                return m.boundary(nk[1], nk[2])[None, :]
            A = self.alphas.get(nk)
            if A is None:
                # no belief points at this successor key: fall back to the
                # never-screen continuation (valid lower bound)
                return Vnat[nk][None, :]
            return A

        for y in range(m.age_max, m.age_min - 1, -1):
            for key in self._keys_at(y):
                alphas, acts = self._backup(key, self.B[key], A_next_of)
                _, keep = np.unique(np.round(alphas, 10), axis=0, return_index=True)
                keep = np.sort(keep)
                self.alphas[key] = alphas[keep]; self.acts[key] = acts[keep]
        return Policy(m, dict(self.alphas), dict(self.acts))

    # -- FIB upper bound -------------------------------------------------
    def fib_bound(self):
        """Fast informed bound: UB(b) = max_a b . u[key][a]; the continuation
        relaxes the belief-dependent max over next alphas to a per-STATE max."""
        m = self.m
        cells = m._mem_cells
        UB = {}
        for y in range(m.age_max, m.age_min - 1, -1):
            for c in cells:
                key = (y,) + c
                cur = {}
                for a, kern in m.M[key].items():
                    v = m.R[key][a].copy()
                    for o, M in kern.items():
                        nk = (y + 1,) + m.succ[key][a][o]
                        if nk[0] > m.age_max:
                            nxt = [m.boundary(nk[1], nk[2])]
                        else:
                            nxt = list(UB[nk].values())
                        cand = np.stack([M @ v2 for v2 in nxt], axis=0)
                        v += cand.max(axis=0)
                    cur[a] = v
                UB[key] = cur
        return UB

    # -- driver ----------------------------------------------------------
    def solve(self, rounds=8, tol=1e-7, rollouts=200, eps=0.1, time_limit=None):
        t0 = time.time()
        self.build_closure()
        pol = self.sweep()
        ev = pol.evaluate()
        best = (ev['objective'], pol, ev)
        self._log(0, ev, t0)
        stale = 0
        for r in range(1, rounds + 1):
            self.add_policy_reachable(pol, eps=eps, n_rollouts=rollouts)
            pol = self.sweep()
            ev = pol.evaluate()
            self._log(r, ev, t0)
            if ev['objective'] > best[0] + tol:
                best = (ev['objective'], pol, ev); stale = 0
            else:
                stale += 1
                if stale >= 2:
                    break
            if time_limit and time.time() - t0 > time_limit:
                break
        pol = best[1]
        UB = self.fib_bound()
        b0 = self.roots[0]
        root = (self.m.age_min,) + self.m.initial_memory()
        ub0 = max(float(b0 @ v) for v in UB[root].values())
        pol.meta.update(dict(objective=best[0], fib_upper=ub0, gap=ub0 - best[0], eval=best[2],
                             history=self.history, cap=self.cap))
        if self.verbose:
            print(f'  done: obj={best[0]:.6f}  FIB ub={ub0:.6f}  gap={ub0 - best[0]:.6f}', flush=True)
        return pol

    def _log(self, r, ev, t0):
        self.history.append(dict(round=r, objective=ev['objective'], colos=ev['colos'], death=ev['death'],
                                 inc=ev['inc'], n_points=int(sum(len(b) for b in self.B.values())),
                                 n_keys=len(self.B), t=time.time() - t0))
        if self.verbose:
            h = self.history[-1]
            print(f'  round {r}: obj={ev["objective"]:.6f} colos={ev["colos"]:.3f} death={ev["death"]:.5f} '
                  f'inc={ev["inc"]:.5f} pts={h["n_points"]} keys={h["n_keys"]} ({h["t"]:.0f}s)', flush=True)


def solve_policy(sex, kernels_npz, weights=None, lam=0.0, age_min=40, age_max=80, cap=2000, rounds=8,
                 rollouts=200, eps=0.1, seed=0, verbose=True, class_known_roots=False):
    m = ReducedPOMDP(sex, kernels_npz, age_min=age_min, age_max=age_max, weights=weights, lam=lam)
    roots = [m.initial_belief()]
    if class_known_roots:
        roots += [m.initial_belief(class_known=c) for c in range(m.n_class)]
    s = PBVISolver(m, cap=cap, seed=seed, init_beliefs=roots, verbose=verbose)
    return s.solve(rounds=rounds, rollouts=rollouts, eps=eps)
