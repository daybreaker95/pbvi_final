###############################################################################
#  pomdp/fivi.py
#
#  FiVI -- Point-Based Value Iteration for Finite-Horizon POMDPs
#  (Walraven & Spaan, JAIR 65, 2019), implemented for CRCScreeningPOMDP9.
#  Separate file from pbvi_v2.py (which stays as the lower-bound-only,
#  randomly-rollout-expanded solver) -- this is a different algorithm, not a
#  patch: it maintains a per-timestep UPPER bound (sawtooth/corner-belief
#  interpolation, Algorithm 1) alongside the lower bound (alpha-vectors,
#  same backup as pbvi_v2.py), and expands belief sets via gap-guided
#  heuristic search (Algorithm 3) instead of random rollouts, so it can
#  report a REAL gap = upper - lower and terminate on it (Algorithm 2).
#
#  Timestep mapping: FiVI's t=1..h corresponds to age = age_min..age_max
#  (h = age_max - age_min + 1). Beyond age_max there is no more decision
#  (SCREEN never chosen), so t=h's continuation is not "no continuation"
#  (as in the paper's generic h-step setting) but the EXACT, already-known
#  boundary natural_value()[age_max+1] -- both bounds coincide there with
#  zero gap, so it's used as a fixed constant "t=h+1" rather than another
#  tracked B_t/Gamma_t.
###############################################################################
from __future__ import annotations

import numpy as np

from .model_v2 import CRCScreeningPOMDP9, WAIT, SCREEN, NO


class FiVI:
    def __init__(self, pomdp: CRCScreeningPOMDP9, seed=0, max_sawtooth_points=None):
        """max_sawtooth_points: Perseus-inspired cap on how many extra
        (non-corner) points UB()'s sawtooth search scans per call. The
        exact algorithm (Algorithm 1) scans ALL of them, which is the
        measured bottleneck (~89% of runtime, growing every iteration as
        extras accumulate) -- capping to a random subset keeps Algorithm
        1's guarantee (still a VALID upper bound, using any subset of the
        known points is legal) while bounding UB()'s cost to O(K) instead
        of O(n_extra). None = no cap (exact paper behaviour)."""
        self.p = pomdp
        self.h = pomdp.age_max - pomdp.age_min + 1
        self.rng = np.random.default_rng(seed)
        self.max_sawtooth_points = max_sawtooth_points
        self.Vnat = pomdp.natural_value()
        self._boundary = self.Vnat[pomdp.age_max + 1]     # fixed "t=h+1" alpha vector

        NS = pomdp.NS
        self.eye = np.eye(NS)
        # per-timestep corner-belief upper bounds (NS,), and extra (non-corner)
        # belief-bound pairs found by expand(). Both refined every sweep.
        self.corner_ub = {t: np.full(NS, np.inf) for t in range(1, self.h + 1)}
        self.extra_b = {t: np.zeros((0, NS)) for t in range(1, self.h + 1)}
        self.extra_ub = {t: np.zeros(0) for t in range(1, self.h + 1)}
        # lower bound: alpha-vectors per timestep (mirrors pbvi_v2.Gamma)
        self.Gamma = {t: (np.zeros((0, NS)), np.zeros(0, dtype=int)) for t in range(1, self.h + 1)}
        self._dep_cache = {}   # DBBU: t -> array of extra_b[t] indices frequently used by t-1's UB calls

        self.gap_history = []

    # ------------------------------------------------------------------
    def _age(self, t):
        return self.p.age_min + t - 1

    def _MR(self, t):
        age = self._age(t)
        return self.p.M[age], self.p.R[age]

    @staticmethod
    def _safe_dot(b, v):
        """b . v treating 0 * inf as 0 (v may hold inf for un-refined corner
        upper bounds -- a plain b@v turns those into NaN via 0*inf). Replace
        v BEFORE multiplying wherever b==0, so the 0*inf multiply itself
        never happens (not just masked away after the fact)."""
        v_safe = np.where(b > 0, v, 0.0)
        return float(np.sum(b * v_safe))

    @staticmethod
    def _safe_matvec(B, v):
        v_safe = np.where(B > 0, v[None, :], 0.0)
        return np.sum(B * v_safe, axis=1)

    # ------------------------------------------------------------------
    # Algorithm 1: sawtooth upper-bound interpolation UB(b, t)
    # ------------------------------------------------------------------
    def UB(self, b, t):
        if t > self.h:
            return float(b @ self._boundary)
        corner = self.corner_ub[t]
        baseline = self._safe_dot(b, corner)
        Bextra, Vextra = self.extra_b[t], self.extra_ub[t]
        if len(Vextra) == 0:
            return baseline
        K = self.max_sawtooth_points
        if K is not None and len(Vextra) > K:
            # integers() (with replacement) instead of choice(replace=False):
            # the latter's overhead dominated at 400k+ calls/run and erased
            # the speedup; duplicates don't threaten validity here, this is
            # already just an approximate (still-valid, just looser) bound.
            idx = self.rng.integers(0, len(Vextra), size=K)
            Bextra, Vextra = Bextra[idx], Vextra[idx]
        # f(b_i) = v_i - b_i . corner   (tightening each extra point provides
        # over the naive corner-only interpolation)
        fb = Vextra - self._safe_matvec(Bextra, corner)
        # c(b_i) = min_{s: b_i(s)>0} b(s)/b_i(s)  (largest fraction of b_i
        # "embeddable" inside the query belief b)
        mask = Bextra > 1e-12
        ratio = np.where(mask, b[None, :] / np.where(mask, Bextra, 1.0), np.inf)
        cb = ratio.min(axis=1)
        cand = cb * fb
        best = min(0.0, float(np.min(cand)) if len(cand) else 0.0)
        return baseline + best

    def UB_batch(self, B0, t, candidate_idx=None, freq_counter=None):
        """Batched UB(): computes the sawtooth upper bound for ALL rows of
        B0 (m, NS) in one shot. UB() calling this once per point (400k+
        Python-level calls in profiling) was the actual bottleneck -- not
        the per-call array size -- since numpy's fixed per-call overhead
        dominates at that call count. Loops only over NS (26, fixed) and
        over the (already-capped) extra points, never over the m query
        beliefs themselves.

        candidate_idx: restrict the sawtooth search to extra_b[t][candidate_idx]
        (DBBU's cached dependency subset) instead of the full/random-K pool.
        freq_counter: if given (len == extra pool searched), incremented at
        the argmin index actually selected for each of the m rows -- used
        to build the DBBU dependency cache during an exact pass."""
        if t > self.h:
            return B0 @ self._boundary
        corner = self.corner_ub[t]
        baseline = self._safe_matvec(B0, corner)          # (m,)
        Bextra_full, Vextra_full = self.extra_b[t], self.extra_ub[t]
        if len(Vextra_full) == 0:
            return baseline
        if candidate_idx is not None:
            pool_idx = candidate_idx
        else:
            K = self.max_sawtooth_points
            if K is not None and len(Vextra_full) > K:
                pool_idx = self.rng.integers(0, len(Vextra_full), size=K)
            else:
                pool_idx = np.arange(len(Vextra_full))
        Bextra, Vextra = Bextra_full[pool_idx], Vextra_full[pool_idx]
        if len(Vextra) == 0:
            return baseline
        fb = Vextra - self._safe_matvec(Bextra, corner)   # (n,)
        m, n = len(B0), len(Vextra)
        ratio = np.full((m, n), np.inf)
        for s in range(self.p.NS):
            col = Bextra[:, s]
            nz = col > 1e-12
            if not nz.any():
                continue
            r = B0[:, s:s + 1] / col[nz][None, :]          # (m, nnz)
            ratio[:, nz] = np.minimum(ratio[:, nz], r)
        cand = ratio * fb[None, :]                          # (m,n)
        argmins = cand.argmin(axis=1)
        best = np.minimum(0.0, cand[np.arange(m), argmins])
        if freq_counter is not None:
            sel = pool_idx[argmins]
            np.add.at(freq_counter, sel, 1)
        return baseline + best

    # ------------------------------------------------------------------
    # Lower-bound (alpha-vector) backup -- same Bellman backup as
    # pbvi_v2.PBVI9._backup, just batched per-timestep instead of over one
    # shared belief set.
    # ------------------------------------------------------------------
    def _backup_one(self, b, t, A_next, has_next):
        """Backup a SINGLE belief b (used by PBS's point-at-a-time loop)."""
        M, R = self._MR(t)
        best_val, best_alpha, best_act = -1e18, None, None
        for a in (WAIT, SCREEN):
            alpha = R[a].copy()
            for o in range(NO):
                Mo = M[a][o]
                if has_next:
                    bo = b @ Mo
                    if bo.sum() <= 1e-12:
                        continue
                    alpha_next = A_next[np.argmax(A_next @ bo)]
                    g = Mo @ alpha_next
                else:
                    g = Mo @ self._boundary
                alpha = alpha + self.p.gamma * g
            val = float(alpha @ b)
            if val > best_val:
                best_val, best_alpha, best_act = val, alpha, a
        return best_alpha, best_act

    def _backup_batch(self, B, t):
        M, R = self._MR(t)
        m = len(B)
        NS = self.p.NS
        best_val = np.full(m, -1e18)
        best_alpha = np.zeros((m, NS))
        best_act = np.zeros(m, dtype=int)
        has_next = t + 1 <= self.h
        if has_next:
            A_next, _ = self.Gamma[t + 1]
            has_next = len(A_next) > 0
        for a in (WAIT, SCREEN):
            alpha = np.tile(R[a], (m, 1))
            for o in range(NO):
                Mo = M[a][o]
                if has_next:
                    Bo = B @ Mo
                    if Bo.sum() <= 1e-12:
                        continue
                    idx = np.argmax(Bo @ A_next.T, axis=1)
                    alpha_next = A_next[idx]
                    g = alpha_next @ Mo.T
                else:
                    g = np.tile(Mo @ self._boundary, (m, 1))
                alpha = alpha + self.p.gamma * g
            vals = np.einsum('ms,ms->m', alpha, B)
            upd = vals > best_val
            best_val[upd] = vals[upd]
            best_alpha[upd] = alpha[upd]
            best_act[upd] = a
        return best_alpha, best_act

    # ------------------------------------------------------------------
    # Algorithm 4: Perseus Belief Selection (PBS). Instead of backing up
    # EVERY point in Bt exhaustively, repeatedly pick a RANDOM not-yet-
    # improved point, back it up, and drop from the queue every point
    # (not just that one) whose value under the growing Gamma_t already
    # meets or beats its value under the OLD Gamma_t -- a single backup
    # often improves many points at once, so most of the queue empties
    # without ever being individually backed up.
    # ------------------------------------------------------------------
    def _backup_pbs(self, B, t):
        m = len(B)
        NS = self.p.NS
        A_old, _ = self.Gamma[t]
        old_vals = (B @ A_old.T).max(axis=1) if len(A_old) else np.full(m, -1e18)
        has_next = t + 1 <= self.h
        A_next = None
        if has_next:
            A_next, _ = self.Gamma[t + 1]
            has_next = len(A_next) > 0

        out_alpha = np.zeros((0, NS))
        out_act = np.zeros(0, dtype=int)
        remaining = list(range(m))
        self.rng.shuffle(remaining)
        while remaining:
            i = remaining.pop()
            alpha, act = self._backup_one(B[i], t, A_next, has_next)
            out_alpha = np.vstack([out_alpha, alpha[None, :]])
            out_act = np.concatenate([out_act, [act]])
            new_vals_remaining = B[remaining] @ alpha if remaining else np.zeros(0)
            still_needed = []
            for j, idx in enumerate(remaining):
                if new_vals_remaining[j] >= old_vals[idx] - 1e-9:
                    continue   # covered by this backup -- drop from queue
                still_needed.append(idx)
            remaining = still_needed
        return out_alpha, out_act

    # ------------------------------------------------------------------
    # Upper-bound backup (Algorithm 2, lines 15-28): same max-over-actions
    # Bellman recursion, but the continuation term uses UB() interpolation
    # against B_{t+1} instead of an alpha-vector max.
    #
    # Algorithm 5: Dependency-Based Bound Updates (DBBU), simplified. The
    # paper caches, PER BELIEF, exactly which point(s) of B_{t+1} the
    # sawtooth argmin actually used, and reuses that tiny per-belief set on
    # non-exact iterations. Full per-belief dependency tracking needs
    # ragged (m, variable-size) indexing that fights the vectorised
    # UB_batch; this keeps the paper's core idea (spend real work every
    # delta iterations, coast on a cached cheap subset otherwise) but
    # shares ONE frequency-ranked subset across all beliefs being backed
    # up at this timestep, built from which extras were actually selected
    # during the exact pass -- a coarser, timestep-level dependency cache
    # rather than a per-belief one.
    # ------------------------------------------------------------------
    def _ub_backup_batch(self, B, t, exact=True, dep_cache_size=40):
        M, R = self._MR(t)
        m = len(B)
        best = np.full(m, -1e18)
        n_next = len(self.extra_ub.get(t + 1, [])) if t + 1 <= self.h else 0
        freq = np.zeros(n_next) if (exact and n_next > 0) else None
        cand_idx = None
        if not exact and n_next > 0:
            cand_idx = self._dep_cache.get(t + 1)
        for a in (WAIT, SCREEN):
            v = B @ R[a]
            for o in range(NO):
                Mo = M[a][o]
                Bo = B @ Mo
                p_o = Bo.sum(axis=1)
                nz = p_o > 1e-12
                if not nz.any():
                    continue
                bo_norm = np.zeros_like(Bo)
                bo_norm[nz] = Bo[nz] / p_o[nz, None]
                ub_vals = self.UB_batch(bo_norm, t + 1, candidate_idx=cand_idx, freq_counter=freq)
                v = v + np.where(nz, self.p.gamma * p_o * ub_vals, 0.0)
            best = np.maximum(best, v)
        if freq is not None and n_next > 0:
            top = np.argsort(freq)[::-1][:dep_cache_size]
            self._dep_cache[t + 1] = top[freq[top] > 0]
        return best

    # ------------------------------------------------------------------
    # Algorithm 3: gap-guided forward belief expansion. Finds h-1 new
    # belief points (one per timestep along a single trajectory) using the
    # CURRENT bounds -- optimistic action choice (trust the upper bound),
    # then the observation branch with the largest current gap.
    # ------------------------------------------------------------------
    def expand(self, stochastic=False, b_start=None):
        """b_start: belief to start the trajectory from; None = the model's
        single/first initial belief. solve() drives this over every entry of
        pomdp.initial_belief_set(), which is one start per OBSERVED risk
        class when the model hands the agent its risk label (observe_risk),
        and one mixed-prior start otherwise.

        stochastic=False: exact Algorithm 3 (pure gap-maximizing
        observation choice every step). With FIXED current bounds this
        picks the SAME branch every call -- confirmed empirically (gap
        frozen for 100+ iterations despite thousands of new points added).
        stochastic=True: action choice stays optimistic/gap-guided (still
        trusts the upper bound, still "FiVI"), but the OBSERVATION is
        sampled proportional to its TRUE probability P(o|b,a) instead of
        argmax-gap -- this is the same diversification pbvi_v2.py's
        add_reachable_beliefs() rollouts already relied on. Doesn't touch
        the paper's convergence argument (any valid belief can be added to
        Bt without breaking Algorithm 1/2's guarantees), just changes which
        ones get explored."""
        p = self.p
        b = p.initial_belief() if b_start is None else np.asarray(b_start, float)
        for t in range(1, self.h):
            M, R = self._MR(t)
            best_a, best_a_val = WAIT, -1e18
            for a in (WAIT, SCREEN):
                val = float(R[a] @ b)
                for o in range(NO):
                    Mo = M[a][o]
                    bo = b @ Mo
                    po = bo.sum()
                    if po > 1e-12:
                        val += self.p.gamma * po * self.UB(bo / po, t + 1)
                if val > best_a_val:
                    best_a_val, best_a = val, a
            a = best_a
            Mo_all = M[a]
            if stochastic:
                probs = np.array([(b @ Mo_all[o]).sum() for o in range(NO)])
                s = probs.sum()
                if s <= 1e-12:
                    break
                o_sel = int(self.rng.choice(NO, p=probs / s))
                bo = b @ Mo_all[o_sel]
                po = bo.sum()
                if po <= 1e-12:
                    break
                bo_n = bo / po
            else:
                best_o, best_o_val = None, -1e18
                A_next, _ = self.Gamma[t + 1]
                for o in range(NO):
                    Mo = Mo_all[o]
                    bo = b @ Mo
                    po = bo.sum()
                    if po <= 1e-12:
                        continue
                    bo_n = bo / po
                    lb = float(np.max(A_next @ bo_n)) if len(A_next) else float(bo_n @ self._boundary)
                    gap_o = self.UB(bo_n, t + 1) - lb
                    if gap_o > best_o_val:
                        best_o_val, best_o = gap_o, o
                        best_bo_n = bo_n
                if best_o is None:
                    break
                bo_n = best_bo_n
            self.extra_b[t + 1] = np.vstack([self.extra_b[t + 1], bo_n[None, :]])
            self.extra_ub[t + 1] = np.concatenate([self.extra_ub[t + 1], [np.inf]])
            b = bo_n

    # ------------------------------------------------------------------
    def solve(self, max_iters=50, time_limit=None, precision=1e-3, verbose=True,
              n_stochastic_trajectories=20, use_pbs=True, dbbu_interval=1):
        """use_pbs: Algorithm 4 (Perseus Belief Selection) for the lower
        bound instead of exhaustively backing up every point.
        dbbu_interval: Algorithm 5 (Dependency-Based Bound Updates,
        simplified -- see _ub_backup_batch docstring). 1 = exact every
        iteration (paper default when unspecified); >1 = coast on the
        cached dependency subset dbbu_interval-1 iterations out of every
        dbbu_interval."""
        import time
        t0 = time.time()
        p = self.p
        # One start belief per OBSERVED risk class when the agent is told its
        # class (observe_risk), else the single mixed-prior belief. Both the
        # expansion trajectories and the reported gap cover ALL of them --
        # converging only at the population prior would leave the two beliefs
        # real individuals actually start from unbounded.
        B1 = p.initial_belief_set()
        for kappa in range(1, max_iters + 1):
            for b_s in B1:
                self.expand(stochastic=False, b_start=b_s)   # exact gap-guided trajectory
                for _ in range(n_stochastic_trajectories):
                    self.expand(stochastic=True, b_start=b_s)  # diversify -- see expand()
            exact_ub = (dbbu_interval <= 1) or (kappa % dbbu_interval == 0)
            for t in range(self.h, 0, -1):
                B = np.vstack([self.eye, self.extra_b[t]])
                if use_pbs:
                    alphas, acts = self._backup_pbs(B, t)
                else:
                    alphas, acts = self._backup_batch(B, t)
                self.Gamma[t] = (alphas, acts)
                ub_vals = self._ub_backup_batch(B, t, exact=exact_ub)
                self.corner_ub[t] = ub_vals[:self.p.NS]
                if len(self.extra_ub[t]):
                    self.extra_ub[t] = ub_vals[self.p.NS:]

            A1, _ = self.Gamma[1]
            # worst (largest-gap) start belief drives both the report and the
            # stopping rule -- stopping on the mean would let one start stay
            # loose while the other carries the average.
            bounds = [(float(np.max(A1 @ b_s)), self.UB(b_s, 1)) for b_s in B1]
            per_start = [(u - l, l, u) for l, u in bounds]
            gap, vl, vu = max(per_start)
            self.gap_history.append({'iter': kappa, 'vl': vl, 'vu': vu, 'gap': gap,
                                      'gap_per_start': [g for g, _, _ in per_start],
                                      'n_extra': sum(len(v) for v in self.extra_ub.values())})
            if verbose:
                print(f"  FiVI iter {kappa}: vl={vl:.5f} vu={vu:.5f} gap={gap:.5f} "
                      f"n_extra={self.gap_history[-1]['n_extra']}")
            if gap <= precision:
                break
            if time_limit is not None and time.time() - t0 > time_limit:
                break
        return self

    # ------------------------------------------------------------------
    def best_action(self, age, b):
        t = age - self.p.age_min + 1
        alphas, acts = self.Gamma[t]
        return int(acts[np.argmax(alphas @ b)])
