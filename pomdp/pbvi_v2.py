###############################################################################
#  pomdp/pbvi_v2.py
#
#  Point-Based Value Iteration for CRCScreeningPOMDP9 (model_v2.py) --
#  9-state, detected/undetected-as-CONTEXT, QALY reward, NO budget axis
#  (budget was dropped by design; screening frequency is governed purely by
#  the cost/QALY trade-off already baked into the reward, via WTP).
#
#  Backward induction over age only (no (age,k) double-indexing like
#  pomdp/pbvi.py's budget-augmented version) -- alpha-vectors are a plain
#  dict keyed by age. Kept as a SEPARATE file from pomdp/pbvi.py, which
#  stays untouched for the 7-state budget-aware pipeline.
###############################################################################

from __future__ import annotations

import numpy as np

from .model_v2 import (
    CRCScreeningPOMDP9, NORMAL, EARLY_POLYP, ADV_POLYP,
    CA_I, CA_II, CA_III, CA_IV, WAIT, SCREEN, NO,
)

TRANSIENT = (NORMAL, EARLY_POLYP, ADV_POLYP, CA_I, CA_II, CA_III, CA_IV)


class PBVI9:
    def __init__(self, pomdp: CRCScreeningPOMDP9, n_belief=400, seed=0):
        self.p = pomdp
        self.rng = np.random.default_rng(seed)
        self.B = self._sample_beliefs(n_belief)
        self.Gamma = {}      # age -> (alphas (m,NS), actions (m,))
        self.Vnat = None
        self._mstack_cache = {}

    # ------------------------------------------------------------------
    def _sample_beliefs(self, n):
        p = self.p
        NSf = p.NS
        trans = [p.fidx(r, s) for r in range(p.n_risk) for s in TRANSIENT]
        pts = []
        for idx in trans:
            b = np.zeros(NSf); b[idx] = 1.0; pts.append(b)
        pts.append(p.initial_belief())
        concs = [np.array([8., 1., .4, .1, .05, .02, .01]),
                 np.array([3., 2., 1., .3, .15, .08, .04]),
                 np.array([1., 1., 1., 1., 1., 1., 1.]),
                 np.array([.5, 1., 2., 1.5, 1., .5, .25])]
        per = max(1, (n - len(pts)) // (len(concs) * max(p.n_risk, 1)))
        for c in concs:
            for _ in range(per):
                b = np.zeros(NSf)
                if p.n_risk == 1:
                    b[[p.fidx(0, s) for s in TRANSIENT]] = self.rng.dirichlet(c)
                else:
                    rh = self.rng.uniform(0.0, 1.0)
                    for r, w in ((0, 1 - rh), (1, rh)):
                        cl = self.rng.dirichlet(c)
                        for j, s in enumerate(TRANSIENT):
                            b[p.fidx(r, s)] = w * cl[j]
                pts.append(b)
        return np.array(pts)

    def add_reachable_beliefs(self, n_rollouts=120, horizon=10, max_belief=700):
        if not self.Gamma:
            return
        new = []
        for _ in range(n_rollouts):
            b = self.p.initial_belief()
            age = self.p.age_min
            for _ in range(horizon):
                if age > self.p.age_max:
                    break
                a = self.best_action(age, b)
                po = self.p.obs_probs(b, age, a)
                s = po.sum()
                if s <= 0:
                    break
                o = int(self.rng.choice(NO, p=po / s))
                bn = self.p.belief_update(b, age, a, o)
                if bn is None:
                    break
                b = bn; new.append(b.copy())
                age += 1
        if not new:
            return
        new = np.array(new)
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
    def _next_gamma(self, age):
        if age + 1 > self.p.age_max:
            return np.array([self.Vnat[age + 1]]), np.array([WAIT])
        return self.Gamma[age + 1]

    def _backup(self, age):
        p = self.p
        M, R = p.M[age], p.R[age]
        B = self.B
        m = len(B)
        actions = [WAIT, SCREEN]
        best_val = np.full(m, -1e18)
        best_alpha = np.zeros((m, p.NS))
        best_act = np.zeros(m, dtype=int)
        A_next, _ = self._next_gamma(age)
        for a in actions:
            alpha = np.tile(R[a], (m, 1))
            Ma = M[a]
            for o in range(NO):
                Mo = Ma[o]
                Bo = B @ Mo
                if Bo.sum() <= 1e-12:
                    continue
                scores = Bo @ A_next.T
                idx = np.argmax(scores, axis=1)
                alpha_next = A_next[idx]
                g = alpha_next @ Mo.T
                alpha = alpha + p.gamma * g
            vals = np.einsum('ms,ms->m', alpha, B)
            upd = vals > best_val
            best_val[upd] = vals[upd]
            best_alpha[upd] = alpha[upd]
            best_act[upd] = a
        return best_alpha, best_act

    # ------------------------------------------------------------------
    def solve(self, expansions=1, verbose=True, max_belief=None,
              n_rollouts=120, horizon=10):
        """max_belief: cap for add_reachable_beliefs' belief set growth.
        Defaults to max(700, 4*len(self.B)) -- add_reachable_beliefs used to
        be called with its own hardcoded max_belief=700 default regardless
        of how large the initial n_belief sample was, so if the static
        Dirichlet sample alone already exceeded 700 (e.g. n_belief=1500 ->
        ~755 points), room=max(0,700-len(B)) was 0 from pass 0 onward and
        belief expansion silently never ran -- |B| looked "converged"
        because it was frozen, not because more points stopped helping."""
        p = self.p
        if max_belief is None:
            max_belief = max(700, 4 * len(self.B))
        self.Vnat = p.natural_value()
        for it in range(expansions + 1):
            for age in range(p.age_max, p.age_min - 1, -1):
                self.Gamma[age] = self._backup(age)
            if verbose:
                b0 = p.initial_belief()
                v = self.value(p.age_min, b0)
                vnat = self.Vnat[p.age_min][NORMAL]
                print(f"  PBVI9 pass {it}: |B|={len(self.B)}  "
                      f"V(normal@{p.age_min})={v:.3f}  "
                      f"(no-screen={vnat:.3f}, gain={v-vnat:.3f})")
            if it < expansions:
                self.add_reachable_beliefs(n_rollouts=n_rollouts, horizon=horizon,
                                            max_belief=max_belief)
        return self

    # ------------------------------------------------------------------
    def q_values(self, age, b):
        p = self.p
        if age > p.age_max:
            return {WAIT: float(self.Vnat[min(age, p.life_max)] @ b)}
        M, R = p.M[age], p.R[age]
        A_next, _ = self._next_gamma(age)
        Q = {}
        for a in (WAIT, SCREEN):
            Ms = self._Mstack(age, a)
            Bo = np.einsum('s,ost->ot', b, Ms)
            scores = Bo @ A_next.T
            Q[a] = float(R[a] @ b + p.gamma * scores.max(axis=1).sum())
        return Q

    def _Mstack(self, age, action):
        key = (age, action)
        st = self._mstack_cache.get(key)
        if st is None:
            Ma = self.p.M[age][action]
            st = np.stack([Ma[o] for o in range(NO)])
            self._mstack_cache[key] = st
        return st

    def best_action(self, age, b):
        Q = self.q_values(age, b)
        return max(Q, key=Q.get)

    def value(self, age, b):
        return max(self.q_values(age, b).values())

    # ------------------------------------------------------------------
    def save(self, path):
        arrs = {}
        for age, (al, ac) in self.Gamma.items():
            arrs[f'a_{age}'] = al
            arrs[f'act_{age}'] = ac
        for age, v in self.Vnat.items():
            arrs[f'vnat_{age}'] = v
        np.savez_compressed(path, **arrs,
                            meta=np.array([self.p.age_min, self.p.age_max,
                                           self.p.life_max], dtype=object))

    def load(self, path):
        z = np.load(path, allow_pickle=True)
        self.Gamma = {}
        self.Vnat = {}
        for key in z.files:
            if key.startswith('a_'):
                age = int(key.split('_')[1])
                self.Gamma[age] = (z[key], z[f'act_{age}'])
            elif key.startswith('vnat_'):
                self.Vnat[int(key.split('_')[1])] = z[key]
        return self


# ---------------------------------------------------------------------------
class PBVIPolicy9:
    """Belief-tracking policy for the 9-state, budget-free POMDP."""

    def __init__(self, pbvi: PBVI9):
        self.pbvi = pbvi
        self.p = pbvi.p

    def reset(self):
        self.b = self.p.initial_belief()
        self._last_a = None
        self._last_age = None

    def act(self, obs):
        age = int(obs['age'])
        if (self._last_a is not None
                and self.p.age_min <= self._last_age <= self.p.life_max):
            bn = self.p.belief_update(self.b, self._last_age, self._last_a, int(obs['obs']))
            if bn is not None:
                self.b = bn
        if age < self.p.age_min or age > self.p.age_max:
            a = WAIT
        else:
            a = self.pbvi.best_action(age, self.b)
        self._last_a = a
        self._last_age = age
        return a
