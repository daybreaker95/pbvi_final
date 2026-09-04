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
    def __init__(self, model: ReducedPOMDP, alphas: dict, acts: dict, meta: dict | None = None,
                 roots=None, root_weights=None):
        self.model = model
        self.alphas = alphas      # (y, tau, ol) -> (K,S)
        self.acts = acts          # (y, tau, ol) -> (K,)
        self.meta = meta or {}
        # deployment population this policy is scored over: the population
        # prior when None, otherwise a weighted set (e.g. the score bands)
        self.roots = roots
        self.root_weights = root_weights
        # deployment diagnostics: how often best_action is asked at a key that
        # has no alpha-vectors (silent WAIT fallback)
        self.n_calls = 0
        self.n_fallback = 0

    def best_action(self, y, tau, ol, b):
        y = int(y)
        if y > self.model.age_max or y < self.model.age_min:
            return WAIT
        self.n_calls += 1
        key = (y,) + mem_key(tau, ol)
        A = self.alphas.get(key)
        if A is None or len(A) == 0:
            self.n_fallback += 1
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

    def evaluate(self, b0=None, weights=None):
        """Exact in-model metrics, averaged over the policy's deployment roots
        (the population prior when none were supplied)."""
        if b0 is None and self.roots is not None:
            b0, weights = self.roots, self.root_weights
        return evaluate_policy(self.model, None, action_batch=self.best_action_batch,
                               b0=b0, weights=weights)

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
                 ref_screen_prob=0.12, init_beliefs=None, verbose=True, root_weights=None):
        self.m = model
        self.cap = cap
        self.dec = round_decimals
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.p_ref = ref_screen_prob
        self.verbose = verbose
        self.roots = [model.initial_belief()] if init_beliefs is None else list(init_beliefs)
        self.root_weights = np.asarray(root_weights, float) if root_weights is not None else None
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
        R, W = self._roots_and_weights()
        self._merge_into(root, list(R), list(W))
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
        R, W = self._roots_and_weights()
        _, tree = policy_tree(m, policy.best_action_batch, collect=True, prune=prune,
                              b0=R, weights=W)
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

    def _roots_and_weights(self):
        w = (self.root_weights if self.root_weights is not None
             else np.full(len(self.roots), 1.0 / len(self.roots)))
        return np.asarray(self.roots, float), np.asarray(w, float)

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
        R, W = self._roots_and_weights()
        return Policy(m, dict(self.alphas), dict(self.acts), roots=R, root_weights=W)

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
        R, W = self._roots_and_weights()
        root = (self.m.age_min,) + self.m.initial_memory()
        ubs = [max(float(b @ v) for v in UB[root].values()) for b in R]
        ub0 = float(np.dot(W, ubs))
        pol.meta.update(dict(objective=best[0], fib_upper=ub0, gap=ub0 - best[0], eval=best[2],
                             fib_upper_by_root=ubs, history=self.history, cap=self.cap,
                             solver=dict(p_ref=self.p_ref, seed=self.seed, round_decimals=self.dec,
                                         rounds=rounds, rollouts=rollouts, eps=eps, tol=tol,
                                         best_round=int(next(h['round'] for h in self.history
                                                             if h['objective'] == best[0])))))
        if self.verbose:
            print(f'  done: obj={best[0]:.6f}  FIB ub={ub0:.6f}  gap={ub0 - best[0]:.6f}', flush=True)
        return pol

    # -- belief-set coverage diagnostic -----------------------------------
    def density_diagnostic(self, policy: 'Policy', prune=1e-6, chunk=400):
        """How well the final belief sets B cover (i) the beliefs the policy
        actually reaches and (ii) their one-step DEVIATIONS -- the successor
        beliefs of the action the policy does not take, which the point-based
        backup has to evaluate to rank the two actions. Distances are L1 to the
        nearest point of B at the same observed key, weighted by reach
        probability (times observation probability for deviations), by age.
        A worst-case density bound (Pineau et al.) is not computable from a
        weight-pruned reachable set; this weighted statistic is what governs
        the objective error of the solved policy."""
        m = self.m
        R, W = self._roots_and_weights()
        _, tree = policy_tree(m, policy.best_action_batch, collect=True, prune=prune, b0=R, weights=W)

        def nn_l1(Q, B):
            out = np.empty(len(Q))
            Bf = np.asarray(B, np.float32)
            for i in range(0, len(Q), chunk):
                q = np.asarray(Q[i:i + chunk], np.float32)
                out[i:i + chunk] = np.abs(q[:, None, :] - Bf[None, :, :]).sum(axis=2).min(axis=1)
            return out

        by_age = {}
        blank = lambda: dict(w_on=[], d_on=[], w_dev=[], d_dev=[])
        for key, (U, acts) in tree.items():
            y = key[0]
            mass = U.sum(axis=1)
            Bn = U / mass[:, None]
            rec = by_age.setdefault(y, blank())
            Bk = self.B.get(key)
            rec['w_on'].append(mass)
            rec['d_on'].append(nn_l1(Bn, Bk) if Bk is not None and len(Bk) else np.full(len(Bn), 2.0))
            if y >= m.age_max:
                continue
            for a in (WAIT, SCREEN):
                sel = acts != a                      # rows at which `a` is the deviation
                if not sel.any() or a not in m.M[key]:
                    continue
                Ua = U[sel]
                for o, M in m.M[key][a].items():
                    V = Ua @ M
                    pm = V.sum(axis=1)
                    ok = pm > prune
                    if not ok.any():
                        continue
                    nk = (y + 1,) + m.succ[key][a][o]
                    Bk2 = self.B.get(nk)
                    Vn = V[ok] / pm[ok, None]
                    rec2 = by_age.setdefault(y + 1, blank())
                    rec2['w_dev'].append(pm[ok])
                    rec2['d_dev'].append(nn_l1(Vn, Bk2) if Bk2 is not None and len(Bk2) else np.full(len(Vn), 2.0))

        def summarise(w, d):
            tot = float(w.sum())
            order = np.argsort(d)
            cw = np.cumsum(w[order]) / tot
            return dict(mass=tot, mean=float((w * d).sum() / tot),
                        p95=float(d[order][min(int(np.searchsorted(cw, 0.95)), len(d) - 1)]),
                        max=float(d.max()),
                        frac_le_001=float(w[d <= 0.01].sum() / tot), frac_le_005=float(w[d <= 0.05].sum() / tot),
                        frac_le_010=float(w[d <= 0.10].sum() / tot))

        rows = []
        all_on = ([], []); all_dev = ([], [])
        for y, rec in sorted(by_age.items()):
            row = dict(age=int(y), n_points=int(sum(len(self.B[k]) for k in self._keys_at(y))),
                       n_keys=len(self._keys_at(y)))
            if rec['w_on']:
                w = np.concatenate(rec['w_on']); d = np.concatenate(rec['d_on'])
                row['on_policy'] = summarise(w, d); all_on[0].append(w); all_on[1].append(d)
            if rec['w_dev']:
                w = np.concatenate(rec['w_dev']); d = np.concatenate(rec['d_dev'])
                row['deviation'] = summarise(w, d); all_dev[0].append(w); all_dev[1].append(d)
            rows.append(row)
        out = dict(prune=prune, by_age=rows,
                   n_points_total=int(sum(len(b) for b in self.B.values())), n_keys=len(self.B))
        if all_on[0]:
            out['on_policy'] = summarise(np.concatenate(all_on[0]), np.concatenate(all_on[1]))
        if all_dev[0]:
            out['deviation'] = summarise(np.concatenate(all_dev[0]), np.concatenate(all_dev[1]))
        return out

    def _log(self, r, ev, t0):
        self.history.append(dict(round=r, objective=ev['objective'], colos=ev['colos'], death=ev['death'],
                                 inc=ev['inc'], n_points=int(sum(len(b) for b in self.B.values())),
                                 n_keys=len(self.B), t=time.time() - t0))
        if self.verbose:
            h = self.history[-1]
            print(f'  round {r}: obj={ev["objective"]:.6f} colos={ev["colos"]:.3f} death={ev["death"]:.5f} '
                  f'inc={ev["inc"]:.5f} pts={h["n_points"]} keys={h["n_keys"]} ({h["t"]:.0f}s)', flush=True)


def solve_policy(sex, kernels_npz, weights=None, lam=0.0, age_min=40, age_max=80, cap=2000, rounds=8,
                 rollouts=200, eps=0.1, seed=0, verbose=True, class_known_roots=False,
                 root_priors=None, root_beliefs=None, root_weights=None, p_ref=0.12, density=False):
    """root_priors: list of class-prior vectors to seed the belief set with,
    e.g. the per-score-band posteriors of a noisy baseline risk score. One
    alpha-vector set serves every band, so a single solve per sex suffices.
    p_ref: reference screening propensity used to build the initial reachable
    closure; seed: rng seed of the epsilon-greedy rollouts; density: also
    compute the belief-set coverage diagnostic (stored in meta['density'])."""
    m = ReducedPOMDP(sex, kernels_npz, age_min=age_min, age_max=age_max, weights=weights, lam=lam)
    roots = [m.initial_belief()]
    if class_known_roots:
        roots += [m.initial_belief(class_known=c) for c in range(m.n_class)]
    if root_priors is not None:
        roots += [m.initial_belief(class_prior=pri) for pri in root_priors]
    if root_beliefs is not None:
        # an explicit deployment population (the score-band beliefs) REPLACES
        # the population prior, so the solver refines and scores exactly the
        # beliefs the policy is deployed from
        roots = [np.asarray(b, float) for b in root_beliefs]
    s = PBVISolver(m, cap=cap, seed=seed, init_beliefs=roots, verbose=verbose,
                   root_weights=root_weights, ref_screen_prob=p_ref)
    pol = s.solve(rounds=rounds, rollouts=rollouts, eps=eps)
    if density:
        t0 = time.time()
        pol.meta['density'] = s.density_diagnostic(pol)
        if verbose:
            d = pol.meta['density']
            print(f"  density: on-policy mean L1 {d.get('on_policy', {}).get('mean', float('nan')):.4f}, "
                  f"deviation mean {d.get('deviation', {}).get('mean', float('nan')):.4f} "
                  f"p95 {d.get('deviation', {}).get('p95', float('nan')):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return pol
