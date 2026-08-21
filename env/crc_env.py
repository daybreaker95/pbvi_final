###############################################################################
#  crc_env.py
#
#  Gym-style single-patient environment wrapping the individual CMOST engine.
#
#  * The HIDDEN TRUE state is the full colon (polyps + cancers), discretised to
#    the 6 POMDP states and always available through info['true_state'].
#  * The AGENT only sees a partial OBSERVATION: the result of a colonoscopy
#    (or "no test"), the current age and the remaining screening budget.
#  * Actions are binary at each yearly decision epoch: WAIT (0) or SCREEN (1).
#  * Reward is (optionally discounted) life-years minus a per-colonoscopy
#    disutility, so cumulative reward approximates discounted LYG once compared
#    to the no-screening counterfactual.
#
#  The API mirrors Gymnasium (reset -> (obs, info); step -> (obs, reward,
#  terminated, truncated, info)) but does not require gymnasium to be installed.
###############################################################################

from __future__ import annotations

import os
import sys
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENV_DIR not in sys.path:
    sys.path.insert(0, _ENV_DIR)

from cmost_individual import (
    CRCEngine, Patient,
    NORMAL, EARLY_ADENOMA, ADVANCED_ADENOMA, EARLY_CANCER,
    ADVANCED_CANCER, DIAGNOSED, DEAD_OTHER, N_STATES, STATE_NAMES,
    OBS_NO_TEST, OBS_NORMAL, OBS_EARLY_ADENOMA, OBS_ADV_ADENOMA, OBS_CANCER,
    N_OBS, OBS_NAMES,
)

# actions
WAIT = 0
SCREEN = 1
N_ACTIONS = 2


@dataclass
class EnvConfig:
    start_age: int = 40        # first decision epoch (screening may begin here)
    stop_age: int = 85         # last age a screening decision is offered
    max_age: int = 100         # hard horizon
    budget: Optional[int] = None   # max lifetime colonoscopies (None = unlimited)
    discount: float = 0.0      # annual discount rate for life-years (0 = undiscounted)
    screen_disutility: float = 0.0  # life-year-equivalent cost per colonoscopy
    warmup_from: int = 20      # natural-history warm-up start age
    # individual_risk cut above which a patient is observed as high_risk
    # (1) rather than low_risk (0). CMOST's individual_risk is a fixed-for-
    # life trait, so the label is constant across the episode and is handed
    # to the agent in the observation dict rather than left to be inferred.
    # None = don't expose it at all (obs['risk_class'] is None), for policies
    # that are meant to treat the class as latent.
    risk_threshold: Optional[float] = None


class CRCScreeningEnv:
    """Single-patient CRC screening environment (POMDP)."""

    def __init__(self, engine: CRCEngine, config: EnvConfig = EnvConfig()):
        self.engine = engine
        self.cfg = config
        self.engine.deterministic_natural_death = True  # paired LYG accounting
        self.pt: Optional[Patient] = None
        self.age = config.start_age
        self.budget_left = config.budget if config.budget is not None else 10**9
        self._done = False
        self._diagnosed = False
        self.n_colo = 0

    # ------------------------------------------------------------------
    def reset(self, gender: Optional[int] = None,
              individual_risk: Optional[float] = None) -> Tuple[dict, dict]:
        self.pt = self.engine.new_patient(gender=gender,
                                          individual_risk=individual_risk,
                                          pre_sample_death=True)
        # natural-history warm-up (no screening) up to the first decision age
        for y in range(1, self.cfg.start_age):
            self.engine.step_year(self.pt, y, screen=False)
            if not self.pt.alive:
                break
        self.age = self.cfg.start_age
        self.budget_left = self.cfg.budget if self.cfg.budget is not None else 10**9
        self._done = not self.pt.alive
        self._diagnosed = self.pt.ever_clinical
        self.n_colo = 0
        obs = self._make_obs(OBS_NO_TEST)
        return obs, self._info()

    # ------------------------------------------------------------------
    def _make_obs(self, obs_code: int) -> dict:
        return {
            'age': self.age,
            'obs': obs_code,
            'budget_left': min(self.budget_left, 10**6),
            'risk_class': self.risk_class,
        }

    @property
    def risk_class(self) -> Optional[int]:
        """1 = high_risk, 0 = low_risk, None = not exposed (cfg.risk_threshold
        unset). The cut is a POPULATION quantile -- pass the same
        (1 - high_frac) quantile of individual_risk the policy's transition
        matrices were estimated at, e.g. pomdp.model_v2's
        individual_risk_threshold(pool, high_frac)."""
        if self.cfg.risk_threshold is None or self.pt is None:
            return None
        return int(self.pt.individual_risk >= self.cfg.risk_threshold)

    def _info(self) -> dict:
        return {
            'true_state': self.engine.clinical_state(self.pt),
            'alive': self.pt.alive,
            'diagnosed': self.pt.ever_clinical,
            'n_colonoscopies': self.pt.n_colonoscopies,
            'death_cause': self.pt.death_cause,
            'death_time': self.pt.death_time,
            'other_death_time': self.pt.other_death_time,
        }

    def _discount(self, age: float) -> float:
        if self.cfg.discount <= 0:
            return 1.0
        return (1.0 + self.cfg.discount) ** (-(age - self.cfg.start_age))

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[dict, float, bool, bool, dict]:
        if self._done:
            return self._make_obs(OBS_NO_TEST), 0.0, True, False, self._info()

        pt = self.pt
        y = self.age
        obs_code = OBS_NO_TEST
        reward = 0.0

        do_screen = (action == SCREEN and not self._diagnosed
                     and self.budget_left > 0 and pt.alive)
        # perform the colonoscopy at the start of the year, then 1 year of NH
        if do_screen:
            obs_code = self.engine.colonoscopy(pt, y, 1, modus='Scre')
            self.budget_left -= 1
            self.n_colo += 1
            reward -= self.cfg.screen_disutility * self._discount(y)

        alive_before = pt.alive
        symp_before = pt.ever_clinical
        # advance one year of natural history (screening already applied above)
        self.engine.step_year(pt, y, screen=False)

        # life-years accrued this year (paired natural-death clock)
        if pt.alive:
            ly = 1.0
        else:
            ly = max(0.0, pt.death_time - y)
        reward += ly * self._discount(y + 0.5)

        # a cancer surfacing clinically this year (symptoms) is observed
        if pt.ever_clinical and not symp_before:
            self._diagnosed = True
            if obs_code == OBS_NO_TEST:
                obs_code = OBS_CANCER

        self.age += 1
        # termination: dead or past horizon
        if not pt.alive or self.age > self.cfg.stop_age:
            # continue natural history silently to death for outcome accounting
            self._finish_life()
            self._done = True

        return (self._make_obs(obs_code), reward,
                self._done, False, self._info())

    def _finish_life(self):
        """Run remaining natural history (no more screening decisions) so that
        the final death cause / age / life-years are recorded."""
        pt = self.pt
        y = self.age
        while pt.alive and y <= self.cfg.max_age:
            self.engine.step_year(pt, y, screen=False)
            y += 1

    # ------------------------------------------------------------------
    def rollout(self, policy) -> dict:
        """Run one full episode under `policy` and return an outcome dict.

        `policy` must expose .reset() and .act(obs) -> action.  Life-years and
        outcomes are accumulated to death (beyond the decision horizon).
        """
        obs, info = self.reset()
        if hasattr(policy, 'reset'):
            # hand the observed risk class straight to policies that accept it
            # (e.g. pomdp.pbvi_v2.PBVIPolicy9), so their belief starts on the
            # right risk block instead of on the population prior
            if (obs.get('risk_class') is not None
                    and 'risk_class' in inspect.signature(policy.reset).parameters):
                policy.reset(risk_class=obs['risk_class'])
            else:
                policy.reset()
        while not self._done:
            a = policy.act(obs)
            obs, r, term, trunc, info = self.step(a)
        pt = self.pt
        crc_death = pt.death_cause in (2, 3)
        age_death = pt.death_time if pt.death_time > 0 else self.cfg.max_age
        ly_lost = max(0.0, pt.other_death_time - age_death) if crc_death else 0.0
        return {
            'life_years': age_death,
            'ly_lost_to_crc': ly_lost,
            'crc_death': int(pt.death_cause == 2),
            'colo_death': int(pt.death_cause == 3),
            'ever_clinical': int(pt.ever_clinical),
            'n_colonoscopies': self.n_colo,             # screening colonoscopies only
            'n_colo_total': pt.n_colonoscopies,         # incl. symptomatic/diagnostic
            'stage_at_dx': int(pt.det_stage[0]) if pt.det_stage else 0,
            'gender': pt.gender,
            'individual_risk': pt.individual_risk,
            'risk_class': self.risk_class,
        }


# ---------------------------------------------------------------------------
# Simple reference policies
# ---------------------------------------------------------------------------
class NoScreening:
    def reset(self): pass
    def act(self, obs): return WAIT


class FixedAgeSchedule:
    """Screen at a fixed set of ages (Zaika-style population schedule)."""
    def __init__(self, ages):
        self.ages = set(int(a) for a in ages)

    def reset(self): pass

    def act(self, obs):
        return SCREEN if obs['age'] in self.ages else WAIT


class UniformInterval:
    """Screen every `interval` years from the first decision age (guideline-style)."""
    def __init__(self, interval=10, start=45, stop=75):
        self.interval, self.start, self.stop = interval, start, stop

    def reset(self): pass

    def act(self, obs):
        a = obs['age']
        if self.start <= a <= self.stop and (a - self.start) % self.interval == 0:
            return SCREEN
        return WAIT
