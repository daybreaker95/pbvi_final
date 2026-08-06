###############################################################################
#  cmost_individual.py
#
#  Faithful INDIVIDUAL-LEVEL re-implementation of the CMOST colorectal-cancer
#  microsimulation (Prakash et al. 2017), extracted from the population engine
#  NumberCrunching_100000.py so that a single patient's natural history can be
#  stepped forward one (quarter-)year at a time and arbitrary screening actions
#  can be injected on demand.
#
#  Design goals
#  ------------
#  * Reuse the *exact* CMOST parameter bundle produced by
#    calculate_sub.prepare_parameters() so the per-patient dynamics are
#    parameter-identical to the validated population model.
#  * Reproduce, quarter by quarter, every stochastic event in the CMOST main
#    loop: natural death, cancer death, new adenoma, de-novo cancer, adenoma
#    growth (-> preclinical cancer), the "fast cancer" short-cut, adenoma
#    regression/healing, symptom-driven clinical presentation, cancer stage
#    progression and (optionally) a colonoscopy.
#  * Expose the hidden TRUE clinical state discretised to the 6 states used by
#    the POMDP:  Normal, Early Adenoma, Advanced Adenoma, Preclinical Cancer,
#    Clinical Cancer, Dead  (most-severe-lesion rule).
#
#  This module contains NO screening *schedule* logic: the caller (the gym
#  environment / transition estimator) decides when a colonoscopy happens.
#
#  Lesion coding (identical to CMOST):
#     polyp stage    1,2,3,4        -> early (non-advanced) adenoma
#     polyp stage    5,6            -> advanced adenoma          (CMOST: Tumor>4)
#     cancer stage   7,8,9,10       -> preclinical CRC stages I,II,III,IV
#     detected cancer 7..10         -> clinical (diagnosed) CRC
###############################################################################

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Make the CMOST python/ package importable no matter where we run from.
_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.abspath(os.path.join(_ENV_DIR, '..', '..'))  # .../python
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


# ---------------------------------------------------------------------------
# Discrete clinical states (the POMDP hidden state space)
#
# Redesigned axis: severity (early/advanced cancer stage) instead of
# detection-status (preclinical/clinical). Detection status ("diagnosed")
# is NOT a belief-tracked dimension any more -- it is a fully-observed
# context flag (like age), because once a colonoscopy or symptom reveals a
# cancer, whether-diagnosed is known with certainty from the observation
# history alone, and no further screening decisions are made afterwards.
#
# DIAGNOSED is kept as a single ABSORBING state in the matrix (mirroring
# DEAD_OTHER) purely so the transition matrix can represent "probability of
# being diagnosed this period" per (age, prior state); everything that
# happens AFTER diagnosis (CRC death timing, which stage it was caught at)
# is not modelled by further transitions here -- it is folded into a
# precomputed terminal reward at the moment of the Normal/EA/AA/EarlyCa/
# AdvCa -> DIAGNOSED transition (see pomdp/model.py).
# ---------------------------------------------------------------------------
NORMAL = 0
EARLY_ADENOMA = 1
ADVANCED_ADENOMA = 2
EARLY_CANCER = 3        # undiagnosed carcinoma, stage I/II (ca_stage 7,8)
ADVANCED_CANCER = 4     # undiagnosed carcinoma, stage III/IV (ca_stage 9,10)
DIAGNOSED = 5            # absorbing: cancer clinically known (screen- or symptom-detected)
DEAD_OTHER = 6           # absorbing: dead, not (yet) diagnosed with CRC (natural or colonoscopy-complication death)
N_STATES = 7
STATE_NAMES = [
    "Normal", "Early Adenoma", "Advanced Adenoma",
    "Early Cancer", "Advanced Cancer", "Diagnosed", "Other Death",
]

# Colonoscopy observation codes
OBS_NO_TEST = 0        # no colonoscopy performed this epoch
OBS_NORMAL = 1         # colonoscopy performed, nothing found
OBS_EARLY_ADENOMA = 2  # low-risk (non-advanced) adenoma removed
OBS_ADV_ADENOMA = 3    # advanced adenoma removed
OBS_CANCER = 4         # carcinoma detected
N_OBS = 5
OBS_NAMES = ["No test", "Normal", "Adenoma", "Advanced adenoma", "Cancer"]


# ---------------------------------------------------------------------------
# Per-patient mutable state
# ---------------------------------------------------------------------------
@dataclass
class Patient:
    gender: int = 1                 # 1 = male, 2 = female
    individual_risk: float = 1.0    # multiplicative personal polyp-risk factor

    # colon contents (parallel lists, one entry per lesion)
    polyp_stage: List[int] = field(default_factory=list)      # 1..6
    polyp_loc: List[int] = field(default_factory=list)        # 1..13
    polyp_year: List[float] = field(default_factory=list)     # onset time
    polyp_early_pct: List[int] = field(default_factory=list)  # 1..500
    polyp_adv_pct: List[int] = field(default_factory=list)    # 1..500

    ca_stage: List[int] = field(default_factory=list)         # 7..10
    ca_loc: List[int] = field(default_factory=list)
    ca_year: List[float] = field(default_factory=list)
    ca_dwell: List[float] = field(default_factory=list)
    ca_symptime: List[float] = field(default_factory=list)
    ca_sympstage: List[int] = field(default_factory=list)
    ca_tstage_I: List[float] = field(default_factory=list)
    ca_tstage_II: List[float] = field(default_factory=list)
    ca_tstage_III: List[float] = field(default_factory=list)

    # detected (clinical) cancers
    det_stage: List[int] = field(default_factory=list)        # stage at detection
    det_year: List[float] = field(default_factory=list)
    det_morttime: List[float] = field(default_factory=list)   # quarters to CRC death (25 = survived)
    det_onset_year: List[float] = field(default_factory=list)  # ca_year of the detected focus (sojourn-time bookkeeping)

    alive: bool = True
    death_cause: int = 0            # 0 none, 1 other, 2 CRC, 3 colonoscopy complication
    death_time: float = 0.0
    other_death_time: float = 1e18  # pre-sampled other-cause death (paired counterfactual)

    # history trackers (years); -100 == "never"
    last_colo: float = -100.0
    last_polyp: float = -100.0
    last_advpolyp: float = -100.0
    last_cancer: float = -100.0

    ever_clinical: bool = False     # has a CRC ever been clinically diagnosed?
    n_colonoscopies: int = 0
    n_polyps_removed: int = 0
    n_adv_removed: int = 0

    def max_polyp_stage(self) -> int:
        return max(self.polyp_stage) if self.polyp_stage else 0

    def max_ca_stage(self) -> int:
        return max(self.ca_stage) if self.ca_stage else 0


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class CRCEngine:
    """Individual-level CMOST natural-history engine.

    Parameters
    ----------
    params : dict
        The bundle returned by calculate_sub.prepare_parameters(handles).
    rng : np.random.Generator, optional
        Random generator (defaults to a fresh default_rng()).
    """

    def __init__(self, params: dict, rng: Optional[np.random.Generator] = None):
        self.P = params
        self.rng = rng if rng is not None else np.random.default_rng()

        sv = params['stage_variables']
        loc = params['location']
        fem = params['female']

        self.colo_detection = np.asarray(sv['Colo_Detection'], dtype=float)   # (10,)
        self.healing = np.asarray(sv['Healing'], dtype=float)                 # (10,)
        self.fast_cancer = np.asarray(sv['FastCancer'], dtype=float)          # (10,)

        self.age_progression = np.asarray(params['age_progression'], float)   # (6,150)
        self.new_polyp = np.asarray(params['new_polyp'], float)               # (150,)
        self.direct_cancer_rate = np.asarray(params['direct_cancer_rate'], float)  # (2,150)
        self.direct_cancer_speed = float(params['direct_cancer_speed'])
        self.life_table = np.asarray(params['life_table'], float)             # (>=100,2)
        self.mortality_matrix = np.asarray(params['mortality_matrix'])        # (4,100,1000)
        self.stage_duration = np.asarray(params['stage_duration'], float)     # (4,4)
        self.location_matrix = np.asarray(params['location_matrix'])          # (2,1000)
        self.risk_dist = params['risk_dist']
        self.early_risk = np.asarray(self.risk_dist['EarlyRisk'], float)
        self.advanced_risk = np.asarray(self.risk_dist['AdvancedRisk'], float)
        self.dwell_speed = params['dwell_speed']

        self.new_polyp_female = float(fem['new_polyp_female'])

        self.loc_coloDetection = np.asarray(loc['ColoDetection'], float)      # (13,)

        # ---- fast-index matrices (ported from NumberCrunching_100000) ----
        # GenderProgression (10 x 2)
        gp = np.ones((10, 2))
        gp[0:4, 1] = fem['early_progression_female']
        gp[4:6, 1] = fem['advanced_progression_female']
        self.gender_progression = gp

        # LocationProgression (10 x 13)
        lp = np.zeros((10, 13))
        early = np.asarray(loc['EarlyProgression'], float)
        adv = np.asarray(loc['AdvancedProgression'], float)
        canc = np.asarray(loc['CancerProgression'], float)
        for r in range(5):
            lp[r, :] = early
        lp[5, :] = adv
        for r in range(6, 10):
            lp[r, :] = canc
        self.location_progression = lp

        # StageMatrix (1000,) -> which cancer stage at onset
        sm = np.zeros(1000, dtype=int)
        sm[0:150] = 7
        sm[150:506] = 8
        sm[506:785] = 9
        sm[785:1000] = 10
        self.stage_matrix = sm

        # SojournMatrix (1000 x 4)
        tx1 = np.asarray(params['tx1'], float)   # (25,4)
        tx2 = np.arange(0.25, 6.50, 0.25)        # 25 elements
        sj = np.zeros((1000, 4))
        for f in range(4):
            tx3 = tx1[:, f].copy()
            tx3 = np.round(tx3 / np.sum(tx3) * 10 * 1000)
            counter = 0
            for f2 in range(25):
                if int(round(tx3[f2] / 10)) != 0:
                    end_idx = int(round(np.sum(tx3[0:f2 + 1]) / 10))
                    sj[counter:end_idx, f] = tx2[f2]
                    counter = end_idx
                    if f2 == 24 and counter != 1000:
                        sj[counter:1000, f] = tx2[f2]
        self.sojourn_matrix = sj

        # ColoReachMatrix (1000,)
        tmploc = np.zeros((13, 1000))
        for f in range(13):
            limit = int(round(1000 * loc['ColoReach'][f]))
            tmploc[f, 0:limit] = 1
        for f in range(12):
            tmploc[f + 1, :] = np.logical_or(tmploc[f + 1, :], tmploc[f, :])
        self.colo_reach_matrix = -np.sum(tmploc, axis=0) + 14

        # complication risks
        self.risc = params['risc']

        # population risk / gender source arrays for sampling new patients
        self._indiv_risk_pool = np.asarray(params['individual_risk'], float)
        self._fraction_female = float(fem['fraction_female'])

        # Optional deterministic (pre-sampled) other-cause death, used by the
        # gym environment so that life-years-gained can be computed with a
        # *paired* counterfactual (the same natural death age across policies).
        # Default OFF -> per-quarter life-table draws, identical to CMOST and to
        # the transition-estimation runs.
        self.deterministic_natural_death = False
        q_annual = np.clip(np.asarray(self.life_table[:100, :], float), 0.0, 1.0)
        year_survive = (1.0 - q_annual / 4.0) ** 4      # P(survive whole year), (100,2)
        self._year_death_prob = 1.0 - year_survive       # P(die during year y), (100,2)

    def _sample_natural_death_time(self, gender: int) -> float:
        """Draw an other-cause death time (year + random quarter) from the
        life table survival distribution for the given gender."""
        pdp = self._year_death_prob[:, gender - 1]
        surv = np.cumprod(1.0 - pdp)
        # P(die in year y) = surv[y-1] - surv[y]  (with surv[-1]=1)
        pmf = np.empty_like(pdp)
        pmf[0] = pdp[0]
        pmf[1:] = surv[:-1] * pdp[1:]
        tail = 1.0 - pmf.sum()
        pmf[-1] += max(tail, 0.0)
        year = int(self.rng.choice(len(pmf), p=pmf / pmf.sum())) + 1
        return year + float(self.rng.integers(0, 4)) / 4.0

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _r(self) -> float:
        return float(self.rng.random())

    def _idx1000(self) -> int:
        return int(round(self.rng.random() * 999))

    # ------------------------------------------------------------------
    # patient construction
    # ------------------------------------------------------------------
    def new_patient(self, gender: Optional[int] = None,
                    individual_risk: Optional[float] = None,
                    pre_sample_death: bool = False) -> Patient:
        if gender is None:
            gender = 2 if self._r() < self._fraction_female else 1
        if individual_risk is None:
            individual_risk = float(self.rng.choice(self._indiv_risk_pool))
        pt = Patient(gender=gender, individual_risk=individual_risk)
        if pre_sample_death or self.deterministic_natural_death:
            pt.other_death_time = self._sample_natural_death_time(gender)
        return pt

    # ------------------------------------------------------------------
    # clinical-state discretisation (most severe lesion)
    # ------------------------------------------------------------------
    def clinical_state(self, pt: Patient) -> int:
        if not pt.alive:
            # CRC death always follows a prior diagnosis in this engine (the
            # mortality clock only starts at det_year), so it is reported as
            # the same absorbing DIAGNOSED bucket, never reached as a fresh
            # transition from an active (undiagnosed) state.
            return DIAGNOSED if pt.death_cause == 2 else DEAD_OTHER
        if pt.ever_clinical:
            return DIAGNOSED
        if pt.ca_stage:                      # undetected cancer present
            return ADVANCED_CANCER if max(pt.ca_stage) >= 9 else EARLY_CANCER
        mps = pt.max_polyp_stage()
        if mps >= 5:
            return ADVANCED_ADENOMA
        if mps >= 1:
            return EARLY_ADENOMA
        return NORMAL

    # ------------------------------------------------------------------
    # cancer bookkeeping helper: create a cancer record from a polyp/de-novo
    # ------------------------------------------------------------------
    def _spawn_cancer(self, pt: Patient, time: float, loc: int, dwell: float):
        tmp1 = int(self.stage_matrix[self._idx1000()])        # onset stage 7..10
        tmp2 = float(self.sojourn_matrix[self._idx1000(), tmp1 - 7])  # sojourn (yrs)
        sd = self.stage_duration
        symptime = time + tmp2
        if tmp1 > 7:
            t_I = time + round(tmp2 * sd[tmp1 - 7, 0] * 4) / 4.0
        else:
            t_I = 1000.0
        if tmp1 > 8:
            t_II = time + round(tmp2 * np.sum(sd[tmp1 - 7, 0:2]) * 4) / 4.0
        else:
            t_II = 1000.0
        if tmp1 > 9:
            t_III = time + round(tmp2 * np.sum(sd[tmp1 - 7, 0:3]) * 4) / 4.0
        else:
            t_III = 1000.0
        pt.ca_stage.append(7)
        pt.ca_loc.append(loc)
        pt.ca_year.append(time)
        pt.ca_dwell.append(dwell)
        pt.ca_symptime.append(symptime)
        pt.ca_sympstage.append(tmp1)
        pt.ca_tstage_I.append(t_I)
        pt.ca_tstage_II.append(t_II)
        pt.ca_tstage_III.append(t_III)

    def _del_polyp(self, pt: Patient, f: int):
        for lst in (pt.polyp_stage, pt.polyp_loc, pt.polyp_year,
                    pt.polyp_early_pct, pt.polyp_adv_pct):
            del lst[f]

    def _del_cancer(self, pt: Patient, f: int):
        for lst in (pt.ca_stage, pt.ca_loc, pt.ca_year, pt.ca_dwell,
                    pt.ca_symptime, pt.ca_sympstage,
                    pt.ca_tstage_I, pt.ca_tstage_II, pt.ca_tstage_III):
            del lst[f]

    # ------------------------------------------------------------------
    # COLONOSCOPY  (port of Colonoscopy() natural-history relevant parts)
    # Returns the observation code for the most severe finding.
    # ------------------------------------------------------------------
    def colonoscopy(self, pt: Patient, y: int, q: int, modus: str = 'Scre') -> int:
        if not pt.alive:
            return OBS_NO_TEST
        time = y + (q - 1) / 4.0
        pt.n_colonoscopies += 1

        reach = int(self.colo_reach_matrix[self._idx1000()])
        reach_ok = np.zeros(13, dtype=bool)
        reach_ok[reach - 1:13] = True

        found_adv = False
        found_early = False
        counter = 0
        # iterate polyps backwards so deletion is safe
        for f in range(len(pt.polyp_stage) - 1, -1, -1):
            stage = pt.polyp_stage[f]
            loc = pt.polyp_loc[f]
            if (self._r() < self.colo_detection[stage - 1] * self.loc_coloDetection[loc - 1]
                    and reach_ok[loc - 1]):
                self._del_polyp(pt, f)
                counter += 1
                pt.n_polyps_removed += 1
                if stage > 4:
                    found_adv = True
                    pt.n_adv_removed += 1
                    pt.last_advpolyp = y
                else:
                    found_early = True
                    pt.last_polyp = y
        if counter > 2:
            pt.last_advpolyp = y
            found_adv = True

        found_cancer = False
        for f in range(len(pt.ca_stage) - 1, -1, -1):
            stage = pt.ca_stage[f]
            loc = pt.ca_loc[f]
            if (self._r() < self.colo_detection[stage - 1] and reach_ok[loc - 1]):
                # cancer becomes a detected (clinical) cancer
                morttime = float(self.mortality_matrix[stage - 7, y - 1, self._idx1000()])
                pt.det_stage.append(stage)
                pt.det_year.append(time)
                pt.det_morttime.append(morttime)
                pt.det_onset_year.append(pt.ca_year[f])
                pt.ever_clinical = True
                pt.last_cancer = y
                self._del_cancer(pt, f)
                found_cancer = True

        pt.last_colo = y

        # colonoscopy complications (may kill the patient)
        if counter == 0 and not found_cancer:
            factor = 0.75
        else:
            factor = 1.5
        if self._r() < self.risc['Colonoscopy_RiscPerforation'] * factor:
            if self._r() < self.risc['DeathPerforation']:
                pt.alive = False
                pt.death_cause = 3
                pt.death_time = time
        elif (self._r() < self.risc['Colonoscopy_RiscSerosaBurn'] * factor):
            pass
        elif (self._r() < self.risc['Colonoscopy_RiscBleeding'] * factor):
            pass
        elif (self._r() < self.risc['Colonoscopy_RiscBleedingTransfusion'] * factor):
            if self._r() < self.risc['DeathBleedingTransfusion']:
                pt.alive = False
                pt.death_cause = 3
                pt.death_time = time

        if found_cancer:
            return OBS_CANCER
        if found_adv:
            return OBS_ADV_ADENOMA
        if found_early:
            return OBS_EARLY_ADENOMA
        return OBS_NORMAL

    # ------------------------------------------------------------------
    # ONE QUARTER of natural history (no screening decision here)
    # ------------------------------------------------------------------
    def _step_quarter(self, pt: Patient, y: int, q: int):
        if not pt.alive:
            return
        yi = y - 1
        time = y + (q - 1) / 4.0
        g = pt.gender

        # 1) natural (other-cause) death
        if pt.other_death_time < 1e17:
            # deterministic pre-sampled natural death
            if time >= pt.other_death_time:
                pt.alive = False
                pt.death_cause = 1
                pt.death_time = pt.other_death_time
                return
        else:
            if self._r() < (self.life_table[yi, g - 1] / 4.0):
                pt.alive = False
                pt.death_cause = 1
                pt.death_time = time
                return

        # 2) cancer death (from a previously diagnosed cancer)
        for f in range(len(pt.det_stage)):
            mt = pt.det_morttime[f]
            if mt < 21:
                if (time - pt.det_year[f]) >= mt / 4.0:
                    pt.alive = False
                    pt.death_cause = 2
                    pt.death_time = time
                    return

        # 3) new adenoma
        polyp_rate = pt.individual_risk * self.new_polyp[yi]
        if g == 2:
            polyp_rate *= self.new_polyp_female
        if self._r() < polyp_rate and len(pt.polyp_stage) < 50:
            pt.polyp_stage.append(1)
            pt.polyp_year.append(time)
            pt.polyp_loc.append(int(self.location_matrix[0, self._idx1000()]))
            early_pct = int(round(self._r() * 499)) + 1
            pt.polyp_early_pct.append(early_pct)
            # CMOST: with risk correlation the two percentiles are identical.
            pt.polyp_adv_pct.append(early_pct)

        # 4) de-novo (direct) cancer
        if self._r() < self.direct_cancer_rate[g - 1, yi] * self.direct_cancer_speed \
                and len(pt.ca_stage) < 25:
            self._spawn_cancer(pt, time, int(self.location_matrix[1, self._idx1000()]), 0.0)

        # 5) adenoma progression (backwards; may convert to cancer)
        for f in range(len(pt.polyp_stage) - 1, -1, -1):
            stage = pt.polyp_stage[f]
            loc = pt.polyp_loc[f]
            if stage < 5:
                risk_mult = self.early_risk[pt.polyp_early_pct[f] - 1]
            else:
                risk_mult = self.advanced_risk[pt.polyp_adv_pct[f] - 1]
            prog = (self.age_progression[stage - 1, yi]
                    * self.location_progression[stage - 1, loc - 1]
                    * self.gender_progression[stage - 1, g - 1]
                    * risk_mult)
            if self._r() < prog:
                pt.polyp_stage[f] += 1
                if pt.polyp_stage[f] > 6:
                    # progressed to preclinical cancer
                    self._spawn_cancer(pt, time, loc, time - pt.polyp_year[f])
                    self._del_polyp(pt, f)
                continue
            # "fast cancer" short-cut (DwellSpeed == 'Slow' path is active)
            fast_p = (self.fast_cancer[stage - 1]
                      * self.age_progression[5, yi]
                      * self.location_progression[5, loc - 1]
                      * self.gender_progression[5, g - 1])
            if self._r() < fast_p:
                self._spawn_cancer(pt, time, loc, time - pt.polyp_year[f])
                self._del_polyp(pt, f)

        # 6) adenoma regression / healing
        for f in range(len(pt.polyp_stage) - 1, -1, -1):
            stage = pt.polyp_stage[f]
            if self._r() < self.healing[stage - 1]:
                pt.polyp_stage[f] -= 1
                if pt.polyp_stage[f] == 0:
                    self._del_polyp(pt, f)

        # 7) symptom-driven clinical presentation
        for f in range(len(pt.ca_stage) - 1, -1, -1):
            if time >= pt.ca_symptime[f]:
                self.colonoscopy(pt, y, q, modus='Symp')
                break

        # 8) cancer stage progression I->II->III->IV
        for f in range(len(pt.ca_stage)):
            st = pt.ca_stage[f]
            if st == 7 and time >= pt.ca_tstage_I[f]:
                pt.ca_stage[f] = 8
            elif st == 8 and time >= pt.ca_tstage_II[f]:
                pt.ca_stage[f] = 9
            elif st == 9 and time >= pt.ca_tstage_III[f]:
                pt.ca_stage[f] = 10

    # ------------------------------------------------------------------
    # ONE YEAR (4 quarters). Optionally perform a screening colonoscopy at q1.
    # Returns the colonoscopy observation (OBS_NO_TEST if screen=False).
    # ------------------------------------------------------------------
    def step_year(self, pt: Patient, y: int, screen: bool = False) -> int:
        obs = OBS_NO_TEST
        for q in (1, 2, 3, 4):
            if q == 1 and screen and pt.alive and not pt.ever_clinical:
                obs = self.colonoscopy(pt, y, q, modus='Scre')
            self._step_quarter(pt, y, q)
            if not pt.alive:
                break
        return obs
