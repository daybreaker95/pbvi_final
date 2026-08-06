###############################################################################
#  pomdp/pbvi.py
#
#  Point-Based Value Iteration (Pineau, Gordon & Thrun 2003) specialised to the
#  finite-horizon, mixed-observability CRC screening POMDP.
#
#  Because age and the remaining colonoscopy budget k are fully observed, the
#  value function is represented by a separate set of 6-dimensional alpha-vectors
#  for each (age, k) context.  We solve by exact backward induction over age
#  (age_max ... age_min), doing a point-based Bellman backup at a fixed set of
#  belief points at each stage.  The boundary condition beyond age_max is the
#  no-screening continuation value (model.natural_value()).
#
#  The resulting policy (PBVIPolicy) tracks a belief over the 6 clinical states
#  by Bayesian filtering on the colonoscopy / symptom observations, so its
#  decisions depend on the individual's realised screening HISTORY -- exactly the
#  personalisation that a fixed population schedule (Zaika 2024) cannot express.
###############################################################################

from __future__ import annotations

import os
import numpy as np

from .model import (
    CRCScreeningPOMDP, NC, NO, NORMAL, EA, AA, ECA, ACA, DIAG, DOTH,
    WAIT, SCREEN,
)


class PBVI:
    def __init__(self, pomdp: CRCScreeningPOMDP, n_belief=400, seed=0):
        self.p = pomdp
        self.rng = np.random.default_rng(seed)
        self.B = self._sample_beliefs(n_belief)
        self.Gamma = {}         # (age,k) -> (alphas (m,6), actions (m,))
        self.Vnat = None
        self._mstack_cache = {}

    # ------------------------------------------------------------------
    def _sample_beliefs(self, n):
        p = self.p
        NSf = p.NS
        # transient full-state indices (Normal/EA/AA/ECA/ACA in each risk block)
        trans = [p.fidx(r, s) for r in range(p.n_risk) for s in (NORMAL, EA, AA, ECA, ACA)]
        nt = len(trans)
        pts = []
        # pure corners
        for idx in trans:
            b = np.zeros(NSf); b[idx] = 1.0; pts.append(b)
        # the model's own prior over (risk, Normal)
        pts.append(p.initial_belief())
        # clinically-plausible mixtures over the transient states, with the
        # correct risk prior applied to a normal-heavy clinical Dirichlet
        concs = [np.array([8., 1., .4, .1, .05]),
                 np.array([3., 2., 1., .2, .1]),
                 np.array([1., 1., 1., 1., 1.]),
                 np.array([.5, 1., 2., 1., .5])]
        per = max(1, (n - len(pts)) // (len(concs) * max(p.n_risk, 1)))
        for c in concs:
            for _ in range(per):
                b = np.zeros(NSf)
                if p.n_risk == 1:
                    b[[p.fidx(0, s) for s in (NORMAL, EA, AA, ECA, ACA)]] = self.rng.dirichlet(c)
                else:
                    # random split of risk mass, each block a clinical Dirichlet
                    rh = self.rng.uniform(0.0, 1.0)
                    for r, w in ((0, 1 - rh), (1, rh)):
                        cl = self.rng.dirichlet(c)
                        for j, s in enumerate((NORMAL, EA, AA, ECA, ACA)):
                            b[p.fidx(r, s)] = w * cl[j]
                pts.append(b)
        return np.array(pts)

    def add_reachable_beliefs(self, n_rollouts=120, horizon=10, max_belief=700):
        """Belief expansion: collect beliefs reachable under the current policy
        (Pineau's expansion heuristic), keep only novel ones, and cap the total
        belief-set size so the backup stays fast."""
        if not self.Gamma:
            return
        new = []
        for _ in range(n_rollouts):
            b = self.p.initial_belief()
            age = self.p.age_min; k = self.p.K
            for _ in range(horizon):
                if age > self.p.age_max:
                    break
                a = self.best_action(age, k, b)
                po = self.p.obs_probs(b, age, a)
                s = po.sum()
                if s <= 0:
                    break
                o = int(self.rng.choice(NO, p=po / s))
                bn = self.p.belief_update(b, age, a, o)
                if bn is None:
                    break
                b = bn; new.append(b.copy())
                if a == SCREEN:
                    k = max(0, k - 1)
                age += 1
        if not new:
            return
        new = np.array(new)
        # keep only beliefs that are novel (min L1 distance to B above a threshold)
        keep = []
        for bpt in new:
            if len(keep) == 0:
                d = np.abs(self.B - bpt).sum(axis=1).min()
            else:
                d = min(np.abs(self.B - bpt).sum(axis=1).min(),
                        np.abs(np.array(keep) - bpt).sum(axis=1).min())
            if d > 0.05:
                keep.append(bpt)
        if keep:
            room = max(0, max_belief - len(self.B))
            keep = np.array(keep)
            if len(keep) > room:
                idx = self.rng.choice(len(keep), size=room, replace=False)
                keep = keep[idx]
            if len(keep):
                self.B = np.vstack([self.B, keep])

    # ------------------------------------------------------------------
    def _next_gamma(self, age, k):
        if age + 1 > self.p.age_max:
            return np.array([self.Vnat[age + 1]]), np.array([WAIT])
        return self.Gamma[(age + 1, k)]

    def _backup(self, age, k):
        p = self.p
        M, R = p.M[age], p.R[age]
        B = self.B
        m = len(B)
        actions = [WAIT] + ([SCREEN] if k > 0 else [])
        best_val = np.full(m, -1e18)
        best_alpha = np.zeros((m, self.p.NS))
        best_act = np.zeros(m, dtype=int)
        for a in actions:
            nk = k - 1 if a == SCREEN else k
            A_next, _ = self._next_gamma(age, nk)          # (q,6)
            alpha = np.tile(R[a], (m, 1))                  # (m,6)
            Ma = M[a]
            for o in range(NO):
                Mo = Ma[o]
                Bo = B @ Mo                                # (m,6) unnormalised posterior
                if Bo.sum() <= 1e-12:
                    continue
                scores = Bo @ A_next.T                     # (m,q)
                idx = np.argmax(scores, axis=1)
                alpha_next = A_next[idx]                    # (m,6)
                g = alpha_next @ Mo.T                       # (m,6) back-projection
                alpha = alpha + p.gamma * g
            vals = np.einsum('ms,ms->m', alpha, B)
            upd = vals > best_val
            best_val[upd] = vals[upd]
            best_alpha[upd] = alpha[upd]
            best_act[upd] = a
        return best_alpha, best_act

    # ------------------------------------------------------------------
    def solve(self, expansions=1, verbose=True):
        p = self.p
        self.Vnat = p.natural_value()
        for it in range(expansions + 1):
            for age in range(p.age_max, p.age_min - 1, -1):
                for k in range(p.K + 1):
                    self.Gamma[(age, k)] = self._backup(age, k)
            if verbose:
                b0 = p.initial_belief()
                v = self.value(p.age_min, p.K, b0)
                vnat = self.Vnat[p.age_min][NORMAL]
                print(f"  PBVI pass {it}: |B|={len(self.B)}  "
                      f"V(normal@{p.age_min}, k={p.K})={v:.3f}  "
                      f"(no-screen={vnat:.3f}, gain={v-vnat:.3f})")
            if it < expansions:
                self.add_reachable_beliefs()
        return self

    # ------------------------------------------------------------------
    def q_values(self, age, k, b):
        p = self.p
        if age > p.age_max:
            return {WAIT: float(self.Vnat[min(age, p.life_max)] @ b)}
        M, R = p.M[age], p.R[age]
        actions = [WAIT] + ([SCREEN] if k > 0 else [])
        Q = {}
        for a in actions:
            nk = k - 1 if a == SCREEN else k
            A_next, _ = self._next_gamma(age, nk)              # (q, NS)
            Ms = self._Mstack(age, a)                          # (NO, NS, NS)
            Bo = np.einsum('s,ost->ot', b, Ms)                 # (NO, NS) unnormalised posteriors
            scores = Bo @ A_next.T                             # (NO, q)
            Q[a] = float(R[a] @ b + p.gamma * scores.max(axis=1).sum())
        return Q

    def _Mstack(self, age, action):
        """Cached (NO, NS, NS) stack of the belief-update matrices."""
        key = (age, action)
        st = self._mstack_cache.get(key)
        if st is None:
            Ma = self.p.M[age][action]
            st = np.stack([Ma[o] for o in range(NO)])
            self._mstack_cache[key] = st
        return st

    def best_action(self, age, k, b):
        Q = self.q_values(age, k, b)
        return max(Q, key=Q.get)

    def value(self, age, k, b):
        return max(self.q_values(age, k, b).values())

    # ------------------------------------------------------------------
    def save(self, path):
        arrs = {}
        for (age, k), (al, ac) in self.Gamma.items():
            arrs[f'a_{age}_{k}'] = al
            arrs[f'act_{age}_{k}'] = ac
        for age, v in self.Vnat.items():
            arrs[f'vnat_{age}'] = v
        np.savez_compressed(path, **arrs,
                            meta=np.array([self.p.age_min, self.p.age_max,
                                           self.p.life_max, self.p.K], dtype=object))

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        self.Gamma = {}
        self.Vnat = {}
        for key in z.files:
            if key.startswith('a_'):
                _, age, k = key.split('_')
                self.Gamma[(int(age), int(k))] = (z[key], z[f'act_{age}_{k}'])
            elif key.startswith('vnat_'):
                self.Vnat[int(key.split('_')[1])] = z[key]
        return self


# ---------------------------------------------------------------------------
# Policy usable inside the gym environment
# ---------------------------------------------------------------------------
class PBVIPolicy:
    """Belief-tracking POMDP policy.  Decisions depend on the realised
    colonoscopy/symptom history through the filtered belief b."""

    def __init__(self, pbvi: PBVI, budget=None):
        self.pbvi = pbvi
        self.p = pbvi.p
        self.K = budget if budget is not None else pbvi.p.K

    def reset(self):
        self.b = self.p.initial_belief()
        self.k = self.K
        self._last_a = None
        self._last_age = None

    def act(self, obs):
        age = int(obs['age'])
        # 1) Bayesian belief update using (previous action, new observation),
        #    only for steps taken inside the POMDP's modelled age range.
        if (self._last_a is not None
                and self.p.age_min <= self._last_age <= self.p.life_max):
            bn = self.p.belief_update(self.b, self._last_age, self._last_a, int(obs['obs']))
            if bn is not None:
                self.b = bn
            if self._last_a == SCREEN:
                self.k = max(0, self.k - 1)
        # 2) choose an action for the current context
        if age < self.p.age_min or age > self.p.age_max or self.k <= 0:
            a = WAIT
        else:
            a = self.pbvi.best_action(age, self.k, self.b)
        self._last_a = a
        self._last_age = age
        return a
