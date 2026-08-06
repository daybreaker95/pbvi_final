###############################################################################
#  pomdp/model.py
#
#  7-clinical-state CRC-screening POMDP with mixed observability and an OPTIONAL
#  latent adenoma-risk class (low / high).
#
#  Full hidden (belief-tracked) state  =  (risk class r) x (clinical state s)
#      clinical s in {Normal, Early Adenoma, Advanced Adenoma,
#                     Early Cancer, Advanced Cancer, Diagnosed, Other Death}
#      risk r in {low, high}  (fixed for life; unobserved)
#  Fully observed CONTEXT (kept outside belief, like a MOMDP): age, remaining
#  colonoscopy budget k, and -- once reached -- "diagnosed" itself is a fully
#  observed fact (known with certainty the instant a colonoscopy or symptom
#  reveals a cancer), so DIAGNOSED is a single ABSORBING belief state, not a
#  family of post-diagnosis states: there are no more screening decisions once
#  diagnosed, so everything that happens afterwards (which stage, CRC-death
#  timing) is NOT modelled by further transitions -- it is pre-computed as a
#  terminal reward V_clin(age, severity) attached to the very transition that
#  enters DIAGNOSED. (The full post-diagnosis timing detail -- e.g. tau-phase
#  time-since-diagnosis -- lives only in the offline CMOST-vs-matrix
#  validation scripts under transitions/, not in this active decision model:
#  most of that detail concerns a period with zero remaining decisions.)
#  Actions:                WAIT (0), SCREEN (1)  [SCREEN only if k>0].
#
#  The latent risk class is what lets the policy personalise on screening
#  HISTORY: colonoscopy findings are informative about the clinical state, which
#  is in turn informative about the (correlated) risk class -- so a patient in
#  whom adenomas are repeatedly found is inferred to be high-risk and screened
#  sooner, exactly like post-polypectomy surveillance.  Detection sensitivity
#  does not depend on risk; only the natural-history transition matrix does
#  (T_low vs T_high, estimated empirically per class).
#
#  Reward (discounted life-years, matching Zaika 2024's LYG objective):
#     +1 life-year for every year begun alive in a transient (undiagnosed)
#        clinical state
#     + V_clin(age, 'early'|'advanced')  when a cancer transitions into
#        DIAGNOSED this period (screen- or symptom-triggered; CMOST's own
#        post-diagnosis mortality draw depends only on STAGE, not on how it
#        was found, so a single value per severity is used for both routes)
#     - screen_disutility for each colonoscopy performed
###############################################################################

from __future__ import annotations

import os
import numpy as np

# clinical states (within a risk block)
NORMAL, EA, AA, ECA, ACA, DIAG, DOTH = 0, 1, 2, 3, 4, 5, 6
NC = 7                                   # clinical states per risk block
# observations (aligned with env.crc_env obs codes)
O_NOTEST, O_NORMAL, O_ADENOMA, O_ADVAD, O_CANCER = 0, 1, 2, 3, 4
NO = 5
# actions
WAIT, SCREEN = 0, 1

RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))


class CRCScreeningPOMDP:
    def __init__(self,
                 transitions_npz=None,
                 stratified_npz=None,
                 effects_npz=None,
                 age_min=40, age_max=85, life_max=100,
                 budget=4, gamma=0.97,
                 screen_disutility=0.02,
                 use_risk_classes=True):
        transitions_npz = transitions_npz or os.path.join(RES, 'transitions_cmost13.npz')
        stratified_npz = stratified_npz or os.path.join(RES, 'transitions_stratified.npz')
        effects_npz = effects_npz or os.path.join(RES, 'pomdp_effects.npz')

        self.age_min, self.age_max, self.life_max = age_min, age_max, life_max
        self.K = budget
        self.gamma = gamma
        self.c = screen_disutility

        # ---- transition matrices per risk class ----
        if use_risk_classes and os.path.exists(stratified_npz):
            z = np.load(stratified_npz, allow_pickle=True)
            ages = z['ages']
            self._P = [{int(a): z['P_low'][i] for i, a in enumerate(ages)},
                       {int(a): z['P_high'][i] for i, a in enumerate(ages)}]
            self.frac_high = float(z['frac_high'])
            self.n_risk = 2
        else:
            z = np.load(transitions_npz, allow_pickle=True)
            ages = z['ages']
            self._P = [{int(a): z['P'][i] for i, a in enumerate(ages)}]
            self.frac_high = 0.0
            self.n_risk = 1
        self._agekeys = self._P[0]
        self.NS = self.n_risk * NC                 # full hidden-state count

        # ---- detection probabilities ----
        if os.path.exists(effects_npz):
            e = np.load(effects_npz, allow_pickle=True)
            self.d_EA = float(e['d_EA']); self.d_AA = float(e['d_AA'])
            if 'd_PC_stage' in e.files and len(e['d_PC_stage']) == 4:
                d_pc_stage = np.asarray(e['d_PC_stage'], float)   # [I, II, III, IV]
            else:
                d_pc_stage = np.full(4, float(e['d_PC']))
        else:
            self.d_EA, self.d_AA = 0.71, 0.91
            d_pc_stage = np.array([0.935, 0.935, 0.935, 0.985])
        # stage-specific screen-detection prob, averaged within severity bucket
        self.d_early = float(d_pc_stage[0:2].mean())
        self.d_adv = float(d_pc_stage[2:4].mean())

        # ---- age-dependent value of a diagnosed cancer (life-years), by
        # SEVERITY only -- CMOST's own post-diagnosis mortality draw
        # (mortality_matrix[stage-7, ...]) depends only on stage, not on
        # whether the diagnosis was screen- or symptom-triggered, so a single
        # value per severity bucket is used for both routes. ----
        from env.params import load_settings
        S = load_settings('CMOST13')
        p_death_stage = np.asarray(S['Mortality'], float)[6:10]     # stages I..IV
        lt = np.asarray(S['LifeTable'], float)
        ff = float(S['fraction_female'])
        q = np.clip(((1 - ff) * lt[:, 0] + ff * lt[:, 1])[:life_max + 1], 0, 1)
        surv = np.cumprod(1 - q)
        self._dRLE = np.zeros(life_max + 2)
        for age in range(1, life_max + 1):
            fut = np.arange(age, life_max + 1)
            self._dRLE[age] = float(np.sum(self.gamma ** (fut - age)
                                    * surv[fut - 1] / max(surv[age - 1], 1e-9)))
        self._p_death_early = float(p_death_stage[0:2].mean())
        self._p_death_adv = float(p_death_stage[2:4].mean())
        self._T_crc = 2.5 * (self.gamma ** 1.25)

        self._build()

    # ------------------------------------------------------------------
    def fidx(self, r, s):
        return r * NC + s

    def _V_clin(self, age, severity):
        rle = self._dRLE[min(max(age, 1), self.life_max)]
        pdeath = self._p_death_early if severity == 'early' else self._p_death_adv
        return (1 - pdeath) * rle + pdeath * self._T_crc

    def _Tnat_clin(self, age, r):
        """Class-r absorbing-Diagnosed/OtherDeath 7x7 clinical transition matrix."""
        keys = self._P[r]
        a = min(max(age, min(keys)), max(keys))
        T = keys[a].copy()
        T[DIAG, :] = 0.0; T[DIAG, DIAG] = 1.0
        T[DOTH, :] = 0.0; T[DOTH, DOTH] = 1.0
        for s in (NORMAL, EA, AA, ECA, ACA):
            tot = T[s].sum()
            if tot > 0:
                T[s] /= tot
        return T

    def _clinical_MR(self, age, r):
        """Build the 7x7 M[o] tensors and 7-vector rewards for one risk block."""
        Tn = self._Tnat_clin(age, r)
        V_early = self._V_clin(age, 'early')
        V_adv = self._V_clin(age, 'advanced')

        # WAIT: a spontaneous transition into DIAGNOSED (symptomatic
        # presentation, already present in the empirically-estimated matrix)
        # is observed as O_CANCER; everything else is silent.
        Mw = {o: np.zeros((NC, NC)) for o in range(NO)}
        for s in range(NC):
            for sp in range(NC):
                p = Tn[s, sp]
                if p <= 0:
                    continue
                if s in (ECA, ACA) and sp == DIAG:
                    Mw[O_CANCER][s, sp] += p
                else:
                    Mw[O_NOTEST][s, sp] += p
        rw = np.zeros(NC)
        for s in (NORMAL, EA, AA, ECA, ACA):
            rw[s] = 1.0
        rw[ECA] += Tn[ECA, DIAG] * V_early
        rw[ACA] += Tn[ACA, DIAG] * V_adv

        # SCREEN: colonoscopy (detect+treat) then natural history
        screen_map = {
            NORMAL: [(1.0, O_NORMAL, NORMAL)],
            EA:     [(self.d_EA, O_ADENOMA, NORMAL), (1 - self.d_EA, O_NORMAL, EA)],
            AA:     [(self.d_AA, O_ADVAD, NORMAL), (1 - self.d_AA, O_NORMAL, AA)],
            ECA:    [(self.d_early, O_CANCER, DIAG), (1 - self.d_early, O_NORMAL, ECA)],
            ACA:    [(self.d_adv, O_CANCER, DIAG), (1 - self.d_adv, O_NORMAL, ACA)],
            DIAG:   [(1.0, O_CANCER, DIAG)],
            DOTH:   [(1.0, O_NORMAL, DOTH)],
        }
        Ms = {o: np.zeros((NC, NC)) for o in range(NO)}
        for s, lst in screen_map.items():
            for prob, obs, m in lst:
                Ms[obs][s, :] += prob * Tn[m, :]
        rs = np.zeros(NC)
        for s in (NORMAL, EA, AA, ECA, ACA):
            rs[s] = 1.0 - self.c
        # detected now -> V(severity); missed -> still exposed to this
        # period's spontaneous-presentation chance at the same severity.
        rs[ECA] += self.d_early * V_early + (1 - self.d_early) * Tn[ECA, DIAG] * V_early
        rs[ACA] += self.d_adv * V_adv + (1 - self.d_adv) * Tn[ACA, DIAG] * V_adv
        return Mw, rw, Ms, rs

    def _build(self):
        self.M = {}     # M[age][action] -> dict{obs: (NS,NS)}
        self.R = {}     # R[age][action] -> (NS,)
        for age in range(self.age_min, self.life_max + 1):
            Mw_full = {o: np.zeros((self.NS, self.NS)) for o in range(NO)}
            Ms_full = {o: np.zeros((self.NS, self.NS)) for o in range(NO)}
            rw_full = np.zeros(self.NS)
            rs_full = np.zeros(self.NS)
            for r in range(self.n_risk):
                Mw, rw, Ms, rs = self._clinical_MR(age, r)
                blk = slice(r * NC, (r + 1) * NC)
                for o in range(NO):
                    Mw_full[o][blk, blk] = Mw[o]
                    Ms_full[o][blk, blk] = Ms[o]
                rw_full[blk] = rw
                rs_full[blk] = rs
            self.M[age] = {WAIT: Mw_full, SCREEN: Ms_full}
            self.R[age] = {WAIT: rw_full, SCREEN: rs_full}

    # ------------------------------------------------------------------
    def initial_belief(self):
        b = np.zeros(self.NS)
        if self.n_risk == 1:
            b[self.fidx(0, NORMAL)] = 1.0
        else:
            b[self.fidx(0, NORMAL)] = 1 - self.frac_high
            b[self.fidx(1, NORMAL)] = self.frac_high
        return b

    def natural_value(self):
        """No-screening continuation value Vnat[age] (NS,) by backward recursion."""
        Vnat = {self.life_max + 1: np.zeros(self.NS)}
        # clinical-index mask for DIAGNOSED/OtherDeath across blocks (zero continuation)
        zero_idx = [self.fidx(r, s) for r in range(self.n_risk) for s in (DIAG, DOTH)]
        for age in range(self.life_max, self.age_min - 1, -1):
            Mw = self.M[age][WAIT]
            rw = self.R[age][WAIT]
            cont = Vnat[age + 1].copy()
            cont[zero_idx] = 0.0
            # T_wait (full) = sum_o Mw[o]
            Tfull = sum(Mw[o] for o in range(NO))
            V = rw + self.gamma * (Tfull @ cont)
            V[zero_idx] = 0.0
            Vnat[age] = V
        return Vnat

    def obs_probs(self, b, age, action):
        M = self.M[age][action]
        return np.array([b @ M[o].sum(axis=1) for o in range(NO)])

    def belief_update(self, b, age, action, obs):
        M = self.M[age][action]
        bn = b @ M[obs]
        tot = bn.sum()
        if tot <= 1e-300:
            return None
        return bn / tot

    # convenience: marginal over clinical states / risk from a full belief
    def clinical_marginal(self, b):
        return np.array([sum(b[self.fidx(r, s)] for r in range(self.n_risk))
                         for s in range(NC)])

    def risk_marginal(self, b):
        return np.array([sum(b[self.fidx(r, s)] for s in range(NC))
                         for r in range(self.n_risk)])
