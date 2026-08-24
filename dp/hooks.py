"""Engine-facing policy hooks for cmost_engine/NumberCrunching_policy.py.

Hook protocol (engine side, once per person per year at q==1 for ages in
[policy_hook_age_min, policy_hook_age_max] while Alive & Included):

    s_true = _pomdp_state_idx(z)                 # 18-state BEFORE any policy colonoscopy
    a      = hook.decide_one(z, y)               # 0 = WAIT, 1 = SCREEN
    ... if a == 1: Colonoscopy(...)
    o      = hook.obs(a, s_true[, info])         # info only if hook.wants_info
    hook.step_update(z, a, o, y)

`info` (dp-patched engine) = dict(s_pre, s_post, early_removed, adv_removed,
cancer_detected, cancer_stage, comp_death, n_polyps_left) describing what the
engine REALLY did in that colonoscopy.
"""
from __future__ import annotations

import numpy as np

from .common import (
    E_D, E_DEAD_COMP, E_DEAD_CRC, E_DEAD_OTHER, MAP18_TO_CLIN, MAP18_TO_EXIT,
    O_ADENOMA, O_MULTI, O_ADVAD, O_EXIT, O_NORMAL, O_NOTEST, SCREEN, WAIT,
)
from .kernels import OL_OF_OBS


class FixedScheduleHook:
    """SCREEN iff the current age is in a fixed set (100 % adherence)."""
    wants_info = False

    def __init__(self, screen_ages):
        self.screen_ages = set(int(a) for a in screen_ages)

    def decide_one(self, z, y):
        return SCREEN if y in self.screen_ages else WAIT

    def obs(self, a, s_true):
        return 0

    def step_update(self, z, a, o, y):
        pass


class FixedScheduleNoDxHook(FixedScheduleHook):
    """Fixed schedule that (like any real programme) does not re-screen a
    person already diagnosed with CRC. Used for the fair fixed-schedule
    comparators: in the engine a diagnosed patient stays Alive & Included
    and would otherwise receive pointless 'screening' colonoscopies."""
    wants_info = True

    def __init__(self, screen_ages, n):
        super().__init__(screen_ages)
        self.diagnosed = np.zeros(n, dtype=bool)

    def decide_one(self, z, y, s_true=None):
        if s_true is not None and s_true in E_D:
            self.diagnosed[z] = True
        if self.diagnosed[z]:
            return WAIT
        return SCREEN if y in self.screen_ages else WAIT

    def obs(self, a, s_true, info=None):
        if s_true in E_D or (info is not None and info['cancer_detected']):
            return O_EXIT
        return O_NOTEST

    def step_update(self, z, a, o, y):
        if o == O_EXIT:
            self.diagnosed[z] = True


class RandomScheduleRecorderHook:
    """Randomised colonoscopy schedule + full decision log, used to estimate
    the empirical SCREEN kernel P(obs, post-state | pre-state, age, ...).

    Each person gets a first screening age ~ U{age_lo..age_hi_first} and
    subsequent intervals ~ U{1..max_interval} until age_max. Diagnosed
    persons are not screened again (mirrors the deployment hook)."""
    wants_info = True

    def __init__(self, n, seed, age_lo=40, age_hi_first=70, max_interval=12, age_max=80):
        rng = np.random.default_rng(seed + 11)
        self.rng = rng
        self.next_age = rng.integers(age_lo, age_hi_first + 1, size=n).astype(np.int16)
        self.max_interval = max_interval
        self.age_max = age_max
        self.diagnosed = np.zeros(n, dtype=bool)
        # decision log (only SCREEN decisions are informative for the kernel,
        # but WAIT rows are kept too: they give the pre-state mix for free)
        self.log_z, self.log_y, self.log_a = [], [], []
        self.log_spre, self.log_spost = [], []
        self.log_er, self.log_ar, self.log_cdet, self.log_cstage, self.log_comp, self.log_npl = [], [], [], [], [], []
        self._cur = None

    def decide_one(self, z, y, s_true=None):
        self._cur = (z, y)
        if s_true is not None and s_true in E_D:
            self.diagnosed[z] = True
        if self.diagnosed[z]:
            return WAIT
        if y >= self.next_age[z]:
            self.next_age[z] = y + int(self.rng.integers(1, self.max_interval + 1))
            return SCREEN
        return WAIT

    def obs(self, a, s_true, info=None):
        z, y = self._cur
        if a == SCREEN:
            self.log_z.append(z); self.log_y.append(y); self.log_a.append(a)
            self.log_spre.append(info['s_pre']); self.log_spost.append(info['s_post'])
            self.log_er.append(info['early_removed']); self.log_ar.append(info['adv_removed'])
            self.log_cdet.append(info['cancer_detected']); self.log_cstage.append(info['cancer_stage'])
            self.log_comp.append(info['comp_death']); self.log_npl.append(info['n_polyps_left'])
            return info_to_obs(a, s_true, info)
        if s_true in E_D:
            return O_EXIT
        return O_NOTEST

    def step_update(self, z, a, o, y):
        if o == O_EXIT:
            self.diagnosed[z] = True

    def log_arrays(self):
        return dict(
            z=np.asarray(self.log_z, np.int32), y=np.asarray(self.log_y, np.int16),
            s_pre=np.asarray(self.log_spre, np.int8), s_post=np.asarray(self.log_spost, np.int8),
            early_removed=np.asarray(self.log_er, np.int16), adv_removed=np.asarray(self.log_ar, np.int16),
            cancer_detected=np.asarray(self.log_cdet, bool), cancer_stage=np.asarray(self.log_cstage, np.int8),
            comp_death=np.asarray(self.log_comp, bool), n_polyps_left=np.asarray(self.log_npl, np.int16),
        )


def info_to_obs(a, s_true, info):
    """Map the engine's real colonoscopy result to the model observation.
    A person already diagnosed (symptomatically, since the previous
    decision) is an exit regardless of the action."""
    if s_true in E_D:
        return O_EXIT
    if a == SCREEN:
        if info['cancer_detected'] or info['comp_death']:
            return O_EXIT
        if info['adv_removed'] > 0:
            return O_ADVAD
        if info['early_removed'] >= 3:
            return O_MULTI
        if info['early_removed'] > 0:
            return O_ADENOMA
        return O_NORMAL
    # WAIT: the only thing the programme learns is whether a diagnosis happened
    if s_true in E_D:
        return O_EXIT
    return O_NOTEST


class BeliefPolicyHook:
    """Belief-tracking hook driving solved dp policies (one per sex).

    policies: dict sex(1|2) -> dp.solver.Policy (with .model ReducedPOMDP).
    sex_arr: engine gender_arr (1 male, 2 female).
    risk_class: per-person class index (used only when observed_class=True).
    observed_class: initial belief conditioned on the true class
              (risk-stratified deployment) instead of the population prior.
    The hook tracks, per person, the belief over (class, clinical state) and
    the observed memory (years since last colonoscopy, last finding).
    """
    wants_info = True

    def __init__(self, policies, sex_arr, risk_class=None, observed_class=False):
        self.pol = policies
        self.sex_arr = np.asarray(sex_arr).astype(int)
        n = len(self.sex_arr)
        self.diagnosed = np.zeros(n, dtype=bool)
        S = policies[1].model.S
        self.belief = np.zeros((n, S))
        tau0, ol0 = policies[1].model.initial_memory()
        self.tau = np.full(n, tau0, dtype=np.int16)
        self.ol = np.full(n, ol0, dtype=np.int8)
        for sex in (1, 2):
            m = policies[sex].model
            idx = np.where(self.sex_arr == sex)[0]
            if observed_class and risk_class is not None:
                for c in range(m.n_class):
                    ii = idx[np.asarray(risk_class)[idx] == c]
                    self.belief[ii] = m.initial_belief(class_known=c)[None, :]
            else:
                self.belief[idx] = m.initial_belief()[None, :]
        self.n_screen = np.zeros(n, dtype=np.int16)
        self.n_impossible = 0
        self._cur = None
        self.log_z, self.log_y, self.log_o = [], [], []

    def decide_one(self, z, y, s_true=None):
        self._cur = (z, y)
        if s_true is not None and s_true in E_D:
            self.diagnosed[z] = True
        if self.diagnosed[z]:
            return WAIT
        p = self.pol[self.sex_arr[z]]
        a = int(p.best_action(y, int(self.tau[z]), int(self.ol[z]), self.belief[z]))
        if a == SCREEN:
            self.n_screen[z] += 1
        return a

    def obs(self, a, s_true, info=None):
        return info_to_obs(a, s_true, info)

    def log_arrays(self):
        return dict(z=np.asarray(self.log_z, np.int32), y=np.asarray(self.log_y, np.int16),
                    obs=np.asarray(self.log_o, np.int8))

    def step_update(self, z, a, o, y):
        if a == SCREEN:
            self.log_z.append(z); self.log_y.append(y); self.log_o.append(o)
        if o == O_EXIT:
            self.diagnosed[z] = True
            return
        m = self.pol[self.sex_arr[z]].model
        M, (tau_n, ol_n) = m.step(y, int(self.tau[z]), int(self.ol[z]), a, o)
        bn = self.belief[z] @ M
        tot = bn.sum()
        if tot > 1e-300:
            self.belief[z] = bn / tot
        else:
            self.n_impossible += 1
        self.tau[z] = tau_n
        self.ol[z] = ol_n


class FindingRuleHook:
    """Belief-free surveillance rule: first colonoscopy at `start`, then the
    next interval depends only on the last finding (clinical-guideline
    style). intervals: dict ol-index -> years {0 normal, 1 adenoma, 2 multi,
    3 advad}. Used to validate the kernels (the same rule is evaluated
    exactly in the model) and as a transparent comparator."""
    wants_info = True

    def __init__(self, n, start=52, intervals=(10, 5, 3, 3), age_max=80):
        self.start, self.iv, self.age_max = int(start), tuple(intervals), int(age_max)
        self.next_age = np.full(n, self.start, dtype=np.int16)
        self.diagnosed = np.zeros(n, dtype=bool)
        self.log_z, self.log_y, self.log_o = [], [], []
        self._cur = None

    def decide_one(self, z, y, s_true=None):
        self._cur = (z, y)
        if s_true is not None and s_true in E_D:
            self.diagnosed[z] = True
        if self.diagnosed[z] or y > self.age_max:
            return WAIT
        return SCREEN if y >= self.next_age[z] else WAIT

    def obs(self, a, s_true, info=None):
        return info_to_obs(a, s_true, info)

    def step_update(self, z, a, o, y):
        if a == SCREEN:
            self.log_z.append(z); self.log_y.append(y); self.log_o.append(o)
        if o == O_EXIT:
            self.diagnosed[z] = True
            return
        if a == SCREEN:
            self.next_age[z] = y + self.iv[OL_OF_OBS[o]]

    def log_arrays(self):
        return dict(z=np.asarray(self.log_z, np.int32), y=np.asarray(self.log_y, np.int16),
                    obs=np.asarray(self.log_o, np.int8))


def rule_action_fn(start=52, intervals=(10, 5, 3, 3), age_max=80):
    """Model-side twin of FindingRuleHook: action depends on (y, tau, ol) only."""
    from .kernels import TAU_NEVER
    def fn(y, tau, ol, b):
        if y > age_max:
            return WAIT
        if tau == TAU_NEVER:
            return SCREEN if y >= start else WAIT
        return SCREEN if tau >= intervals[ol] else WAIT
    return fn
