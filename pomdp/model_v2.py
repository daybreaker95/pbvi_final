###############################################################################
#  pomdp/model_v2.py
#
#  NEW, separate POMDP model -- 13-state (full I-IV cancer stage, split into
#  UNDETECTED and DETECTED variants) with belief-tracked diagnosis, single
#  QALY reward. Does NOT modify pomdp/model.py (the currently-active 7-state
#  pipeline stays untouched).
#
#  Hidden (belief-tracked) state, LOCAL to this file (NOT env.state9's 9-state
#  layout, which stays exactly as-is for CMOST-matching estimation scripts):
#    0 Normal, 1 EarlyPolyp, 2 AdvPolyp, 3-6 Ca I..IV (UNDETECTED, action-
#    dependent WAIT/SCREEN dynamics), 7 CRC_DEATH, 8 OTHER_DEATH (absorbing),
#    9-12 Ca I..IV (DETECTED -- action-INDEPENDENT, governed entirely by
#    T_detected_tauphase.npz's per-stage (p_stay,p_crc,p_oth), since SCREEN
#    is never offered to an already-diagnosed patient)
#  x risk class (low/high, fixed per individual for life).
#
#  Earlier revision of this file treated "detected" as pure CONTEXT (not
#  belief) and tried to fold its whole remaining-lifetime value into a
#  diagnosis-year reward bonus -- but never actually built that bonus (only
#  a one-year utility swap), so post-diagnosis QALY/cost/CRC-mortality was
#  effectively dropped from the value function. Fixed by making DETECTED
#  Ca-stage states REAL belief-tracked states instead: diagnosis routes
#  belief there directly, and their own (action-independent) transition/
#  reward rows let the ordinary backward induction (natural_value/_backup)
#  accumulate the correct rest-of-life value automatically -- no separate
#  lump-sum injection or extra backward pass needed. (In expectation this
#  is IDENTICAL to injecting a lump sum at diagnosis -- the two are the same
#  Bellman sum computed two different ways -- but this way needs no special-
#  casing and can't silently be forgotten.)
#
#  Reward: single QALY objective (utility - procedure disutility -
#  cost/WTP), matching cmost_experiment_final's R_qaly / net-monetary-
#  benefit convention. LYG/cost/incidence/mortality are NOT separately
#  optimised here -- compute them only at Monte-Carlo evaluation time.
###############################################################################

from __future__ import annotations

import os
import numpy as np

from env.state9 import (
    NORMAL, EARLY_POLYP, ADV_POLYP, CA_I, CA_II, CA_III, CA_IV,
    CRC_DEATH, OTHER_DEATH, N_STATES9, STATE9_NAMES,
)

WAIT, SCREEN = 0, 1
N_ACTIONS = 2
CA_STAGES = (CA_I, CA_II, CA_III, CA_IV)
STAGE_KEYS = ['I', 'II', 'III', 'IV']
# observations: no-test, normal, early-polyp, adv-polyp, cancer
O_NOTEST, O_NORMAL, O_ADENOMA, O_ADVAD, O_CANCER = 0, 1, 2, 3, 4
NO = 5

# ---------------------------------------------------------------------------
# LOCAL (model_v2-only) extended state axis: the 9 env.state9 indices
# (0..8, unchanged) PLUS 4 new DETECTED Ca-stage states appended after them.
# NOT part of env.state9 / N_STATES9 -- those stay 9-wide for the CMOST-
# matching transition-estimation scripts, which have no notion of a
# "detected" axis at all.
# ---------------------------------------------------------------------------
DET_CA_I, DET_CA_II, DET_CA_III, DET_CA_IV = 9, 10, 11, 12
DET_CA_STAGES = (DET_CA_I, DET_CA_II, DET_CA_III, DET_CA_IV)
NC_LOCAL = 13

# ---------------------------------------------------------------------------
# Risk-class axis. The agent is TOLD which class an individual is in (see
# CRCScreeningPOMDP9.observe_risk / initial_belief(risk_class=...)): the class
# is defined by CMOST's own individual_risk trait -- the top
# DEFAULT_HIGH_FRAC of the population by individual_risk is "high_risk",
# everyone else is "low_risk". individual_risk is a fixed-for-life
# multiplicative polyp-rate factor, so the label is fixed for life too and
# the transition blocks stay block-diagonal on this axis.
# ---------------------------------------------------------------------------
RISK_LOW, RISK_HIGH = 0, 1
RISK_LABELS = ('low_risk', 'high_risk')
DEFAULT_HIGH_FRAC = 0.20


def individual_risk_threshold(individual_risk, high_frac=DEFAULT_HIGH_FRAC):
    """The individual_risk value at the (1 - high_frac) population quantile,
    i.e. the cut above which someone counts as high_risk."""
    return float(np.quantile(np.asarray(individual_risk, float), 1.0 - high_frac))


def risk_class_from_individual_risk(individual_risk, high_frac=DEFAULT_HIGH_FRAC,
                                    threshold=None):
    """Map CMOST individual_risk values to the POMDP's observed risk label.

    Returns (risk_class, threshold) where risk_class is an int array of
    RISK_HIGH (1) for the top high_frac by individual_risk and RISK_LOW (0)
    for the rest. Pass an explicit `threshold` to reuse a cut computed on a
    different (e.g. larger, or training-time) population instead of this
    cohort's own quantile -- the label the agent is given has to line up
    with the split the loaded policy's transition matrices were estimated
    under, or the high/low dynamics it reasons with are the wrong ones."""
    x = np.asarray(individual_risk, float)
    thr = float(threshold) if threshold is not None else individual_risk_threshold(x, high_frac)
    return (x >= thr).astype(int), thr


RES = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

# ---------------------------------------------------------------------------
# Phase-of-care utility DECREMENTS (USPSTF CRC screening decision analysis,
# Appendix Table 3.4 -- identical numbers to the flat-_BG version this
# replaces). These are subtracted from an AGE-SPECIFIC background weight
# instead of a fixed constant -- see AGE_WEIGHT below.
# ---------------------------------------------------------------------------
PHASE_UTIL_DECR = {
    'I':   [0.12, 0.05, 0.70, 0.05],
    'II':  [0.18, 0.05, 0.70, 0.05],
    'III': [0.24, 0.24, 0.70, 0.24],
    'IV':  [0.70, 0.70, 0.70, 0.70],
}

# Age-specific background utility weight, age 40..99 (USPSTF CRC screening
# decision analysis, Appendix Table 3.1). Source: Hanmer J, Lawrence WF,
# Anderson JP, Kaplan RM, Fryback DG. Report of nationally representative
# values for the noninstitutionalized US adult population for 7 health-
# related quality-of-life scores. Med Decis Making. 2006;26(4):391-400 --
# mean EQ-5D US values by sex x age-decile, extrapolated within each decade
# using the 2017 US life table's mean age by sex, then smoothed to remove
# discontinuities. Replaces the old flat _BG=0.92 constant: USPSTF's own
# QALY calc multiplies every phase disutility by this same age-varying
# weight rather than scaling colonoscopy procedure disutility by age.
_AGE_WEIGHT_40_99 = [
    0.888522, 0.885463, 0.882405, 0.879346, 0.876288, 0.873460, 0.870845, 0.868229, 0.865613, 0.862996,
    0.860377, 0.857758, 0.855138, 0.852516, 0.849893, 0.847402, 0.845015, 0.842624, 0.840231, 0.837835,
    0.835435, 0.833031, 0.830623, 0.828212, 0.825797, 0.822419, 0.818415, 0.814408, 0.810398, 0.806386,
    0.802370, 0.798351, 0.794328, 0.790300, 0.786267, 0.782535, 0.778817, 0.775075, 0.771305, 0.767506,
    0.763673, 0.759805, 0.755897, 0.751944, 0.747941, 0.743880, 0.739750, 0.735546, 0.731258, 0.726879,
    0.722403, 0.717824, 0.713137, 0.708341, 0.703433, 0.698415, 0.693291, 0.688068, 0.682755, 0.677363,
]
_AGE_WEIGHT_MIN, _AGE_WEIGHT_MAX = 40, 99


def age_weight(age):
    """Background (disease-free) utility at a given integer age, clamped to
    the tabulated 40-99 range (ages <40 reuse the age-40 value; the model's
    own age_min is 40 anyway; ages >99 reuse the age-99 value -- population
    mass this old is negligible and already heavily gamma-discounted)."""
    a = int(round(age))
    if a <= _AGE_WEIGHT_MIN:
        return _AGE_WEIGHT_40_99[0]
    if a >= _AGE_WEIGHT_MAX:
        return _AGE_WEIGHT_40_99[-1]
    return _AGE_WEIGHT_40_99[a - _AGE_WEIGHT_MIN]


def u_initial(age, i):    return age_weight(age) - PHASE_UTIL_DECR[STAGE_KEYS[i]][0]
def u_continuing(age, i): return age_weight(age) - PHASE_UTIL_DECR[STAGE_KEYS[i]][1]
def u_term_crc(age, i):   return age_weight(age) - PHASE_UTIL_DECR[STAGE_KEYS[i]][2]
def u_term_other(age, i): return age_weight(age) - PHASE_UTIL_DECR[STAGE_KEYS[i]][3]

WTP_DEFAULT = 100_000        # USD / QALY -- Neumann/Cohen/Weinstein (NEJM 2014) recommend
                              # $100k-150k for the US; $50k is a dated 1970s-dialysis-era
                              # benchmark. $100k is also the de facto standard WTP used
                              # across the CRC screening cost-effectiveness literature
                              # (including USPSTF-linked analyses).
PROC_DISUTIL = 0.000496     # colonoscopy procedure disutility: USPSTF CRC
# screening decision analysis, Appendix Table 3.10 (Swan et al., disutility
# 0.12 for 36.22 hours => 0.12*36.22/8766 = 0.000496 QALY/event; TMI-survey-
# based, not the "N days at 0.5 utility" convention). Was 0.0020 (Naber et
# al. 2019 JNCI Cancer Spectrum, "1.5 days at 0.5 utility") -- a different,
# also-credible source using a different measurement method (assumed partial
# utility vs a validated instrument), ~4x larger. Switched to the USPSTF/
# Swan value for consistency with age_weight/PHASE_UTIL_DECR (same USPSTF
# source family). comp_disutil (below, Naber-based) is UNCHANGED: it maps
# well onto USPSTF's own separate Appendix Table 3.3 adverse-event
# disutilities (e.g. "other GI event" 0.00274, same "N days at 0.5 utility"
# convention as Naber's complication figure), so no similar conflict there.

# Family-history-conditioned risk-class prior. CMOST itself has no family-
# history concept (individual_risk is an exogenous synthetic trait), so this
# is an EXTERNAL epidemiological anchor, reused as-is from experiments/
# risk_factors.py (7-state pipeline) for consistency: P(family history+ |
# high/low risk class). Deliberately family-history ONLY (no smoking/
# obesity/diabetes/etc.) per scope decision -- those factors are much
# weaker predictors (HR close to 1 in Jung et al. 2022) and would need
# their own class-conditional prevalence calibration.
FH_HIGH_PREV = 0.30   # P(family history + | high-risk class)
FH_LOW_PREV = 0.10    # P(family history + | low-risk class)


class CRCScreeningPOMDP9:
    def __init__(self,
                 transitions9_npz=None, stratified_npz=None,
                 age_min=40, age_max=80, life_max=100, gamma=0.97,
                 use_risk_classes=True, use_future_costs=False,
                 wtp=WTP_DEFAULT, sex: int | None = None,
                 sex_risk_npz=None, colo_penalty_qaly=0.0,
                 observe_risk=True):
        """sex: None = sex-pooled (legacy, ~50/50 mixed dynamics+mortality),
        1 = male-only, 2 = female-only. CMOST bakes sex into polyp onset
        (new_polyp_female), adenoma progression (gender_progression) and
        de-novo cancer rate (direct_cancer_rate), so male/female need their
        OWN T_undetected/d_symp (from estimate_transitions_9state_sex_risk.py)
        -- not just a reward/mortality tweak. CRC-specific post-diagnosis
        mortality (mortality_matrix) has no gender axis, so T_detected only
        needs its OTHER-DEATH component swapped to the sex-specific life
        table; the CRC-death component is reused unchanged.

        sex_risk_npz: override path for the sex x risk-class transitions file
        (defaults to results/transitions_9state_sex_risk.npz) -- lets a caller
        point at an alternate risk-threshold definition (e.g. a top-10%-high
        re-estimate) without touching the default file on disk.

        observe_risk: True (default) = the risk class is an OBSERVED input,
        handed to the agent at t=0 exactly like sex is -- callers pass the
        individual's own label to initial_belief(risk_class=RISK_LOW/
        RISK_HIGH) and the belief starts fully concentrated on that block.
        The transition blocks are block-diagonal on the risk axis (risk is
        fixed for life), so a degenerate risk prior stays degenerate: the
        agent reasons about WHICH clinical state the person is in, never
        about which risk class. False = the earlier behaviour, where the
        class stays latent and initial_belief() splits mass frac_high /
        (1 - frac_high) across the two blocks for the agent to infer from
        colonoscopy findings. Only affects anything when n_risk == 2; the
        flag itself only changes initial_belief_set() (which start beliefs
        the solver explores from) -- initial_belief()'s own default return
        value is unchanged either way.

        colo_penalty_qaly: fixed QALY "shadow price" subtracted from the
        SCREEN reward on top of PROC_DISUTIL/comp_disutil -- a budget lever
        (lambda) independent of wtp/$-cost, used to trade FEWER, better-
        targeted colonoscopies for a higher deaths-averted-per-colonoscopy
        ratio (see marginal-efficiency analysis: the policy's own uncapped
        volume already extends past the point where its OWN marginal
        colonoscopies are cheaper than q10y's average one)."""
        transitions9_npz = transitions9_npz or os.path.join(RES, 'transitions_9state.npz')
        stratified_npz = stratified_npz or os.path.join(RES, 'transitions_stratified.npz')
        self.age_min, self.age_max, self.life_max = age_min, age_max, life_max
        self.gamma = gamma
        self.wtp = wtp
        self.NC = NC_LOCAL
        self.sex = sex
        self.colo_penalty_qaly = colo_penalty_qaly
        self.observe_risk = observe_risk

        # ---- undetected-world severity dynamics, per risk class ----
        sex_risk_npz = sex_risk_npz or os.path.join(RES, 'transitions_9state_sex_risk.npz')
        stratified9_npz = os.path.join(RES, 'transitions_9state_stratified.npz')
        if sex is not None and use_risk_classes and os.path.exists(sex_risk_npz):
            zz = np.load(sex_risk_npz, allow_pickle=True)
            ages9 = zz['ages']
            tag = 'male' if sex == 1 else 'female'
            self._P_undet = [
                {int(a): zz[f'P_undetected_{tag}_low'][i] for i, a in enumerate(ages9)},
                {int(a): zz[f'P_undetected_{tag}_high'][i] for i, a in enumerate(ages9)},
            ]
            self._d_symp = [
                {int(a): zz[f'd_symp_{tag}_low'][i] for i, a in enumerate(ages9)},
                {int(a): zz[f'd_symp_{tag}_high'][i] for i, a in enumerate(ages9)},
            ]
            self.frac_high = float(zz[f'frac_high_{tag}'])
            self.n_risk = 2
        elif use_risk_classes and os.path.exists(stratified9_npz):
            # sex-pooled risk stratification (transitions9_stratified.npz)
            zz = np.load(stratified9_npz, allow_pickle=True)
            ages9 = zz['ages']
            self._P_undet = [
                {int(a): zz['P_undetected_low'][i] for i, a in enumerate(ages9)},
                {int(a): zz['P_undetected_high'][i] for i, a in enumerate(ages9)},
            ]
            self._d_symp = [
                {int(a): zz['d_symp_low'][i] for i, a in enumerate(ages9)},
                {int(a): zz['d_symp_high'][i] for i, a in enumerate(ages9)},
            ]
            self.frac_high = float(zz['frac_high'])
            self.n_risk = 2
        else:
            z = np.load(transitions9_npz, allow_pickle=True)
            ages9 = z['ages']
            self._P_undet = [{int(a): z['P_undetected'][i] for i, a in enumerate(ages9)}]
            self._d_symp = [{int(a): z['d_symp'][i] for i, a in enumerate(ages9)}]
            self.frac_high = 0.0
            self.n_risk = 1
        self.NS = self.n_risk * self.NC

        # ---- settings-derived pieces: detection probs, costs, mortality ----
        from env.params import load_settings
        S = load_settings('CMOST13')
        self._load_detection(S)
        self._build_costs(S, use_future_costs)
        self._build_T_detected(S)
        self._build_dRLE(S)
        self._build()
        self._burnin_dist = self._compute_burnin_dist()

    # ------------------------------------------------------------------
    def fidx(self, r, s):
        return r * self.NC + s

    # ------------------------------------------------------------------
    def _compute_burnin_dist(self):
        """Per-risk-class NC-vector state distribution AT age_min, obtained
        by propagating 100% Normal at age 1 through WAIT-only dynamics up
        to age_min-1 -- matches NumberCrunching_policy's own cohort start
        (y=0, Included/Alive=all-True, Last_Polyp=-100 i.e. "never") at
        age 1, fully healthy. Without this, a POMDP built with age_min=40
        silently assumes the age-40 population is 100% disease-free, when
        in reality ~10% already carry a precursor lesion or (rarely) an
        undetected cancer by then, and ~3% have already died of other
        causes (excluded/renormalized away below, since we're modeling
        decisions for the population that SURVIVES to age_min).
        No-op when age_min<=1 (empty range), so age_min=1 callers
        (e.g. compare_model_v2_vs_cmost.py) are unaffected."""
        out = []
        for r in range(self.n_risk):
            d = np.zeros(self.NC)
            d[NORMAL] = 1.0
            for age in range(1, self.age_min):
                Mw, rw, Ms, rs = self._undet_MR(age, r)
                Tfull = sum(Mw[o] for o in range(NO))
                d = d @ Tfull
            d[CRC_DEATH] = 0.0
            d[OTHER_DEATH] = 0.0
            tot = d.sum()
            if tot > 1e-12:
                d = d / tot
            out.append(d)
        return out

    # ------------------------------------------------------------------
    def _load_detection(self, S):
        effects_npz = os.path.join(RES, 'pomdp_effects.npz')
        if os.path.exists(effects_npz):
            e = np.load(effects_npz, allow_pickle=True)
            self.d_EA = float(e['d_EA']); self.d_AA = float(e['d_AA'])
            if 'd_PC_stage' in e.files and len(e['d_PC_stage']) == 4:
                self.d_PC_stage = np.asarray(e['d_PC_stage'], float)  # [I,II,III,IV]
            else:
                self.d_PC_stage = np.full(4, float(e['d_PC']))
        else:
            self.d_EA, self.d_AA = 0.71, 0.91
            self.d_PC_stage = np.array([0.935, 0.935, 0.935, 0.985])

    # ------------------------------------------------------------------
    def _build_costs(self, S, use_future):
        # Fut* columns are a separate, ~1.7-10x larger CMOST costing
        # convention (settings' own NearFuture toggle, default 'off') that
        # the real engine tracks in parallel but does NOT use for its own
        # headline Money['AllCost'] -- what every CMOST-in-loop comparison
        # here actually measures. use_future=True here would silently train
        # against a cost scale never being validated against.
        pfx = 'Fut' if use_future else ''
        C = S['Cost']
        self.c_colo = float(C['Colonoscopy'])
        self.c_polyp = float(C['Colonoscopy_Polyp'])
        self.c_cancer_colo = float(C['Colonoscopy_Cancer'])
        comp_cost = (
            C['Colonoscopy_RiscBleeding'] if 'Colonoscopy_RiscBleeding' in C else 0.0
        )
        # complication PROBABILITY comes from settings risk params, cost from Cost dict
        p_comp = (S['Colonoscopy_RiscBleeding'] + S['Colonoscopy_RiscBleedingTransfusion']
                  + S['Colonoscopy_RiscPerforation'] + S['Colonoscopy_RiscSerosaBurn'])
        comp_cost_amt = (
            S['Colonoscopy_RiscBleeding'] * C['Colonoscopy_bleed']
            + S['Colonoscopy_RiscBleedingTransfusion'] * C['Colonoscopy_bleed_transfusion']
            + S['Colonoscopy_RiscPerforation'] * C['Colonoscopy_Perforation']
            + S['Colonoscopy_RiscSerosaBurn'] * C['Colonoscopy_Serosal_burn']
        )
        self.comp_cost = float(comp_cost_amt)
        self.comp_disutil = float(p_comp) * 0.0041  # per-event disutility, Zaika-style

        # Complication DEATH probability (previously priced only as cost/
        # disutility above -- comp_cost/comp_disutil -- with zero chance of
        # actually dying in the transition model). Mirrors NumberCrunching's
        # own two-tier risk multiplier: 0.75x for a clean exam (nothing
        # found/removed this visit), 1.5x when a polyp/cancer was found.
        # Only perforation and bleeding-transfusion carry a DeathX
        # sub-probability in the engine; serosal burn / plain bleeding do not.
        d_perf = float(S['DeathPerforation'])
        d_bleedtr = float(S['DeathBleedingTransfusion'])
        p_perf = float(S['Colonoscopy_RiscPerforation'])
        p_bleedtr = float(S['Colonoscopy_RiscBleedingTransfusion'])
        self.p_comp_death_clean = p_perf * 0.75 * d_perf + p_bleedtr * 0.75 * d_bleedtr
        self.p_comp_death_found = p_perf * 1.5 * d_perf + p_bleedtr * 1.5 * d_bleedtr

        self.init_cost = np.array([C[f'{pfx}Initial_{k}'] for k in STAGE_KEYS], float)
        self.cont_cost = np.array([C[f'{pfx}Cont_{k}'] for k in STAGE_KEYS], float)
        self.final_cost = np.array([C[f'{pfx}Final_{k}'] for k in STAGE_KEYS], float)
        self.final_oc_cost = np.array([C[f'{pfx}Final_oc_{k}'] for k in STAGE_KEYS], float)

    # ------------------------------------------------------------------
    def _build_T_detected(self, S):
        """Post-diagnosis (per year) transitions, per stage-at-detection.

        Loads results/T_detected_tauphase.npz (built by transitions/
        build_T_detected_from_tauphase.py), which derives this by
        marginalizing the ALREADY-VALIDATED 97-state tau-phase matrix's
        occupancy over tau (quarters-since-diagnosis) at each age -- i.e.
        tau-phase-level accuracy for the CRC-death timing, without exposing
        tau as a belief-tracked dimension in this POMDP.

        There used to be a cruder same-year mortality_matrix Monte-Carlo
        fallback here (memoryless, no tau-occupancy weighting -- carries the
        known SCREENING_VALIDATION.md residual) for when the tauphase file
        hadn't been built yet. It's gone: the DETECTED Ca-stage states now
        carry their entire remaining-lifetime value through this table (see
        module docstring), so silently degrading to the naive estimate would
        silently degrade the POMDP's actual decisions, not just a diagnostic
        side table. Build the file first if this raises.
        """
        tauphase_npz = os.path.join(RES, 'T_detected_tauphase.npz')
        if not os.path.exists(tauphase_npz):
            raise FileNotFoundError(
                f"{tauphase_npz} not found. Run "
                "transitions/build_T_detected_from_tauphase.py first -- "
                "this POMDP no longer has a naive fallback for T_detected "
                "(DETECTED Ca-stage states depend on it directly for their "
                "entire remaining-lifetime value, not just a side table).")
        self.T_det = np.load(tauphase_npz)['T_detected']   # (age,4,3): [p_stay,p_crc,p_oth]
        self._T_det_source = 'tauphase_marginalized'
        if self.sex is not None:
            # CRC-death hazard (mortality_matrix) has no gender axis, so
            # it's reused unchanged; only the competing OTHER-DEATH
            # component is swapped from the sex-pooled life table to the
            # sex-specific column (the pooled tauphase estimate's
            # other-death hazard IS exactly this same life table, mixed
            # -- see _build_dRLE -- so this splice is exact, not a new
            # approximation).
            lt = np.asarray(S['LifeTable'], float)
            p_crc = self.T_det[:, :, 1].copy()
            p_oth_sex = np.clip(lt[:self.life_max, self.sex - 1], 0, 1)
            self.T_det = np.zeros_like(self.T_det)
            for stg in range(4):
                self.T_det[:, stg, 1] = p_crc[:, stg]
                self.T_det[:, stg, 2] = p_oth_sex
                self.T_det[:, stg, 0] = np.clip(1 - p_crc[:, stg] - p_oth_sex, 0, None)

    # ------------------------------------------------------------------
    def _build_dRLE(self, S):
        lt = np.asarray(S['LifeTable'], float)
        if self.sex is not None:
            q = np.clip(lt[:, self.sex - 1][:self.life_max + 1], 0, 1)
        else:
            ff = float(S['fraction_female'])
            q = np.clip(((1 - ff) * lt[:, 0] + ff * lt[:, 1])[:self.life_max + 1], 0, 1)
        surv = np.cumprod(1 - q)
        self._dRLE = np.zeros(self.life_max + 2)
        for age in range(1, self.life_max + 1):
            fut = np.arange(age, self.life_max + 1)
            self._dRLE[age] = float(np.sum(self.gamma ** (fut - age)
                                    * surv[fut - 1] / max(surv[age - 1], 1e-9)))

    # ------------------------------------------------------------------
    def _colo_action_cost(self, s):
        """Expected colonoscopy $ cost + complication cost for screening
        someone currently believed to be in undetected state s."""
        if s == NORMAL:
            return self.c_colo + self.comp_cost
        if s in (EARLY_POLYP, ADV_POLYP):
            return self.c_polyp + self.comp_cost
        if s in CA_STAGES:
            return self.c_cancer_colo + self.comp_cost
        return 0.0

    # ------------------------------------------------------------------
    def _qaly_undetected(self, age, s, action):
        """(utility, cost) for one year in undetected state s, action a.
        Returns (E_u, ann_cost)."""
        du = PROC_DISUTIL + self.comp_disutil + self.colo_penalty_qaly if action == SCREEN else 0.0
        cost = self._colo_action_cost(s) if action == SCREEN else 0.0
        if s in (NORMAL, EARLY_POLYP, ADV_POLYP):
            return age_weight(age) - du, cost
        # undetected cancer: treat like "continuing" utility at that stage
        # (already symptomatic-adjacent burden even before formal diagnosis)
        i = CA_STAGES.index(s)
        return u_continuing(age, i) - du, cost

    def _qaly_detected(self, age, stage_i, just_diagnosed):
        """(utility, cost) for one year in a DETECTED Ca-stage state."""
        u = u_initial(age, stage_i) if just_diagnosed else u_continuing(age, stage_i)
        cost = self.init_cost[stage_i] if just_diagnosed else self.cont_cost[stage_i]
        return u, cost

    # ------------------------------------------------------------------
    def _undet_MR(self, age, r):
        """Build the NC_LOCAL x NC_LOCAL WAIT/SCREEN observation tensors and
        NC_LOCAL-vector QALY rewards for risk-class block r, covering BOTH
        the undetected severity axis (indices 0..8: env.state9's 9 states,
        action-dependent) and the 4 DETECTED Ca-stage states (indices 9..12,
        action-INDEPENDENT -- screening is never offered to an already-
        diagnosed patient, so Mw/Ms and rw/rs are identical there).

        Diagnosis (either WAIT's symptomatic self-presentation via dsy, or
        SCREEN's active detection via d_PC_stage) now routes belief DIRECTLY
        into the matching DET_CA_STAGES[k] state instead of looping back
        into the undetected axis -- see module docstring for why."""
        Pd = self._P_undet[r]
        dd = self._d_symp[r]
        Tn = Pd[min(max(age, min(Pd)), max(Pd))]
        dsy = dd[min(max(age, min(dd)), max(dd))]
        d_pc = self.d_PC_stage

        Mw = {o: np.zeros((self.NC, self.NC)) for o in range(NO)}
        Ms = {o: np.zeros((self.NC, self.NC)) for o in range(NO)}
        rw = np.zeros(self.NC)
        rs = np.zeros(self.NC)

        def _route_diag(M, src, k, mass):
            """Route `mass` newly diagnosed at stage k (source state `src`,
            observation O_CANCER) THROUGH this same year's T_det hazard
            immediately, instead of landing the whole mass at DET_CA_STAGES[k]
            with a full "grace year" before T_det first applies. Matches the
            validated transitions/compare_9state_vs_cmost.py cohort
            convention (dist_d gets exit_mass added, THEN T_det[age] is
            applied to it, same age/year -- see run_cohort9_annual). Returns
            the extra diagnosis-year reward from same-year death (the
            terminal utility bump), to be added alongside the u_initial term."""
            p_stay_k, p_crc_k, p_oth_k = self.T_det[min(age, self.life_max) - 1, k]
            M[O_CANCER][src, DET_CA_STAGES[k]] += mass * p_stay_k
            M[O_CANCER][src, CRC_DEATH] += mass * p_crc_k
            M[O_CANCER][src, OTHER_DEATH] += mass * p_oth_k
            # CMOST's AddCosts() charges terminal-year Final_k cost only on
            # a CANCER death (mode='tu'); an other-cause death while
            # diagnosed keeps paying ordinary Cont_k, no separate bump --
            # matched below (p_oth_k gets no cost term of its own).
            u_bump = mass * (p_crc_k * u_term_crc(age, k) + p_oth_k * u_term_other(age, k))
            cost_bump = mass * p_crc_k * self.final_cost[k]
            return u_bump - cost_bump / self.wtp

        # ---- DETECTED block: action-independent, straight from T_det ----
        # (WAIT and SCREEN get the IDENTICAL row here -- this is what lets
        # ordinary backward induction accumulate the correct rest-of-life
        # value with no special-casing: R_det[age][k] this year, plus
        # gamma * p_stay * (next year's DET_CA_k value) via the recursion
        # natural_value()/_backup() already do for every state.)
        for k, ds in enumerate(DET_CA_STAGES):
            p_stay, p_crc, p_oth = self.T_det[min(age, self.life_max) - 1, k]
            for M in (Mw, Ms):
                M[O_NOTEST][ds, ds] = p_stay
                M[O_NOTEST][ds, CRC_DEATH] = p_crc
                M[O_NOTEST][ds, OTHER_DEATH] = p_oth
            u, cost = self._qaly_detected(age, k, just_diagnosed=False)
            # AddCosts() replaces this year's Cont_k with Final_k exactly
            # when the year ends in a CRC death -- swap that fraction here
            # (an other-cause death keeps paying Cont_k, see _route_diag).
            cost_blend = (p_stay + p_oth) * cost + p_crc * self.final_cost[k]
            r_det = u - cost_blend / self.wtp
            r_det += p_crc * u_term_crc(age, k)
            r_det += p_oth * u_term_other(age, k)
            rw[ds] = r_det
            rs[ds] = r_det

        # ---- WAIT (undetected axis, indices 0..8) ----
        for s in range(N_STATES9):
            if s in (CRC_DEATH, OTHER_DEATH):
                Mw[O_NOTEST][s, s] = 1.0
                continue
            for sp in range(N_STATES9):
                p = Tn[s, sp]
                if p > 0:
                    Mw[O_NOTEST][s, sp] += p
            u, cost = self._qaly_undetected(age, s, WAIT)
            rw[s] = u - cost / self.wtp
            if s in CA_STAGES:
                # dsy[s] is a 4-vector: P(exit to detected-at-stage-k), k
                # possibly LATER than s's own stage if the patient silently
                # progressed before self-presenting this year (4-exit-sink
                # fix in estimate_transitions_9state.py). Route each k's
                # mass to the matching DET_CA_STAGES[k] state directly.
                p_diag_vec = dsy[s]
                p_diag_total = float(p_diag_vec.sum())
                if p_diag_total > 0:
                    val_avg = 0.0
                    for k in range(4):
                        if p_diag_vec[k] > 0:
                            val_avg += _route_diag(Mw, s, k, p_diag_vec[k])
                            u_k, cost_k = self._qaly_detected(age, k, just_diagnosed=True)
                            val_avg += p_diag_vec[k] * (u_k - cost_k / self.wtp)
                    # diagnosis-year utility swap only -- the rest-of-life
                    # value now comes for free from DET_CA_k's own row above
                    rw[s] += val_avg - p_diag_total * (u - cost / self.wtp)

        # ---- SCREEN (undetected axis, indices 0..8): detect+treat then
        # natural history for the not-caught branch ----
        # A caught polyp is treated as CURED THIS SAME YEAR -- it blends
        # back into the undetected axis via Tn[NORMAL,:] (ordinary natural
        # history continuation from a clean baseline), matching this
        # POMDP's original convention.
        screen_map = {
            NORMAL: [(1.0, O_NORMAL, NORMAL)],
            EARLY_POLYP: [(1 - self.d_EA, O_NORMAL, EARLY_POLYP), (self.d_EA, O_ADENOMA, NORMAL)],
            ADV_POLYP: [(1 - self.d_AA, O_NORMAL, ADV_POLYP), (self.d_AA, O_ADVAD, NORMAL)],
        }
        for i, s in enumerate(CA_STAGES):
            # caught THIS check -> colonoscopy confirms the TRUE current
            # stage i immediately (no extra natural-history step blended
            # in, unlike the ORIGINAL version of this code -- there is no
            # "undetected" destination left to blend with once diagnosed).
            # missed -> ordinary natural-history continuation, same as
            # before.
            screen_map[s] = [(1 - d_pc[i], O_NORMAL, s)]
        for s, lst in screen_map.items():
            for prob, obs, m in lst:
                Ms[obs][s, :N_STATES9] += prob * Tn[m, :]
        Ms[O_NOTEST][CRC_DEATH, CRC_DEATH] = 1.0
        Ms[O_NOTEST][OTHER_DEATH, OTHER_DEATH] = 1.0
        for s in range(N_STATES9):
            if s in (CRC_DEATH, OTHER_DEATH):
                continue
            u, cost = self._qaly_undetected(age, s, SCREEN)
            rs[s] = u - cost / self.wtp
            if s in CA_STAGES:
                i = CA_STAGES.index(s)
                # detected THIS screen -> diagnosis-year value at the TRUE
                # current stage i (screening catches them NOW, no further
                # progression possible before the colonoscopy), same-year
                # T_det hazard applied immediately via _route_diag.
                extra = _route_diag(Ms, s, i, d_pc[i])
                u_i, cost_i = self._qaly_detected(age, i, just_diagnosed=True)
                val_i = u_i - cost_i / self.wtp
                rs[s] += d_pc[i] * (val_i - (u - cost / self.wtp)) + extra
                # missed by screen but self-presents later this same year,
                # possibly at a LATER stage k (same 4-vector fix as WAIT).
                # NOTE: this probability mass used to be dropped entirely
                # from Ms (only its EXPECTED REWARD was added below) --
                # rows for CA_STAGES states summed to < 1 under SCREEN,
                # leaking belief mass. Now routed to DET_CA_STAGES[k] like
                # everything else, so Ms rows sum to 1 again.
                p_diag_vec = (1 - d_pc[i]) * dsy[s]
                p_diag_total = float(p_diag_vec.sum())
                if p_diag_total > 0:
                    val_avg = 0.0
                    for k in range(4):
                        if p_diag_vec[k] > 0:
                            val_avg += _route_diag(Ms, s, k, p_diag_vec[k])
                            u_k, cost_k = self._qaly_detected(age, k, just_diagnosed=True)
                            val_avg += p_diag_vec[k] * (u_k - cost_k / self.wtp)
                    rs[s] += val_avg - p_diag_total * (u - cost / self.wtp)

        # ---- Complication death: every SCREEN colonoscopy on the
        # undetected axis (0..8) carries a small chance of a fatal
        # perforation/bleeding event -- previously priced only via
        # comp_cost/comp_disutil (reward), with the transition model
        # implicitly assuming a 0% chance of it actually killing anyone.
        # Redirect a small slice of probability mass into OTHER_DEATH, using
        # CMOST's own risk multiplier: 0.75x for a clean exam (nothing
        # found this visit), 1.5x when a polyp/cancer was found.
        for s in range(N_STATES9):
            if s in (CRC_DEATH, OTHER_DEATH):
                continue
            move_clean = Ms[O_NORMAL][s, :] * self.p_comp_death_clean
            Ms[O_NORMAL][s, :] -= move_clean
            Ms[O_NORMAL][s, OTHER_DEATH] += move_clean.sum()
            for o in (O_ADENOMA, O_ADVAD, O_CANCER):
                move_found = Ms[o][s, :] * self.p_comp_death_found
                Ms[o][s, :] -= move_found
                Ms[o][s, OTHER_DEATH] += move_found.sum()

        return Mw, rw, Ms, rs

    # ------------------------------------------------------------------
    def _build(self):
        """Build, per age, the WAIT and SCREEN observation tensors M[o] and
        QALY reward vectors R over the full 13-state axis (undetected +
        detected) x risk-class blocks stacked into NS = n_risk*NC. Detected
        Ca-stage dynamics (from T_det) don't depend on risk class, but are
        still built once per risk block by _undet_MR so they land in that
        block's own slice of the stacked block-diagonal matrix."""
        self.M_undet = {}   # age -> action -> {obs: (NS,NS)}
        self.R_undet = {}   # age -> action -> (NS,)

        for age in range(self.age_min, self.life_max + 1):
            Mw_full = {o: np.zeros((self.NS, self.NS)) for o in range(NO)}
            Ms_full = {o: np.zeros((self.NS, self.NS)) for o in range(NO)}
            rw_full = np.zeros(self.NS)
            rs_full = np.zeros(self.NS)
            for r in range(self.n_risk):
                Mw, rw, Ms, rs = self._undet_MR(age, r)
                blk = slice(r * self.NC, (r + 1) * self.NC)
                for o in range(NO):
                    Mw_full[o][blk, blk] = Mw[o]
                    Ms_full[o][blk, blk] = Ms[o]
                rw_full[blk] = rw
                rs_full[blk] = rs
            self.M_undet[age] = {WAIT: Mw_full, SCREEN: Ms_full}
            self.R_undet[age] = {WAIT: rw_full, SCREEN: rs_full}

    # ------------------------------------------------------------------
    def initial_belief(self, family_history: bool | None = None,
                       risk_class: int | None = None):
        """risk_class: RISK_HIGH (1) / RISK_LOW (0) = the individual's risk
        class is OBSERVED and handed to the agent, so all initial mass goes
        on that block (P(high) = 1 or 0). This is the observe_risk path:
        the label comes from CMOST's individual_risk (top DEFAULT_HIGH_FRAC
        = high_risk, see risk_class_from_individual_risk). None = class
        latent, use the population prior instead.

        family_history: None = population baseline prior (frac_high).
        True/False = Bayes-updated prior using FH_HIGH_PREV/FH_LOW_PREV
        (P(high risk class | family history status), via Bayes' rule on the
        population split). Ignored when risk_class is given -- family
        history is only ever a noisy PROXY for the class, so a directly
        observed label supersedes it rather than combining with it.

        Both only affect anything when n_risk == 2."""
        b = np.zeros(self.NS)
        if self.n_risk == 1:
            b[0:self.NC] = self._burnin_dist[0]
            return b
        if risk_class is not None:
            r = int(risk_class)
            if r not in (RISK_LOW, RISK_HIGH):
                raise ValueError(f'risk_class must be {RISK_LOW} (low_risk) or '
                                 f'{RISK_HIGH} (high_risk), got {risk_class!r}')
            b[r * self.NC:(r + 1) * self.NC] = self._burnin_dist[r]
            return b
        p_high = self.frac_high
        if family_history is not None:
            fh_p = FH_HIGH_PREV if family_history else (1 - FH_HIGH_PREV)
            fh_q = FH_LOW_PREV if family_history else (1 - FH_LOW_PREV)
            num = fh_p * p_high
            den = num + fh_q * (1 - p_high)
            p_high = num / den if den > 0 else p_high
        b[0 * self.NC:1 * self.NC] = (1 - p_high) * self._burnin_dist[0]
        b[1 * self.NC:2 * self.NC] = p_high * self._burnin_dist[1]
        return b

    def initial_belief_set(self):
        """Every belief the agent can actually START from, for solvers that
        seed their belief-set exploration at t=0. With observe_risk the
        agent is told its class up front, so there are two genuine starts
        (low_risk and high_risk), one per risk block -- expanding only from
        the mixed population prior would explore a belief manifold no real
        individual is ever on. Without it, there is exactly one start."""
        if self.n_risk == 1 or not self.observe_risk:
            return [self.initial_belief()]
        return [self.initial_belief(risk_class=r) for r in range(self.n_risk)]

    # ------------------------------------------------------------------
    # PBVI solver interface (mirrors pomdp/model.py's shape: M[age][action]
    # -> {obs: (NS,NS)}, R[age][action] -> (NS,)). DETECTED Ca-stage states
    # are just 4 more entries in this same NS-wide axis (WAIT and SCREEN
    # rows identical there, since screening isn't offered post-diagnosis),
    # so belief-tracking / backward induction handles them automatically --
    # no separate detected-world interface needed.
    # ------------------------------------------------------------------
    @property
    def M(self):
        return self.M_undet

    @property
    def R(self):
        return self.R_undet

    def obs_probs(self, b, age, action):
        M = self.M_undet[age][action]
        return np.array([b @ M[o].sum(axis=1) for o in range(NO)])

    def belief_update(self, b, age, action, obs):
        M = self.M_undet[age][action]
        bn = b @ M[obs]
        tot = bn.sum()
        if tot <= 1e-300:
            return None
        return bn / tot

    def natural_value(self):
        """No-screening continuation value Vnat[age] (NS,) by backward
        recursion. CRC_DEATH/OTHER_DEATH are absorbing with zero further
        continuation value (matches pomdp/model.py's DIAG/DOTH treatment)."""
        Vnat = {self.life_max + 1: np.zeros(self.NS)}
        zero_idx = [self.fidx(r, s) for r in range(self.n_risk) for s in (CRC_DEATH, OTHER_DEATH)]
        for age in range(self.life_max, self.age_min - 1, -1):
            Mw = self.M_undet[age][WAIT]
            rw = self.R_undet[age][WAIT]
            cont = Vnat[age + 1].copy()
            cont[zero_idx] = 0.0
            Tfull = sum(Mw[o] for o in range(NO))
            V = rw + self.gamma * (Tfull @ cont)
            V[zero_idx] = 0.0
            Vnat[age] = V
        return Vnat

    def clinical_marginal(self, b):
        return np.array([sum(b[self.fidx(r, s)] for r in range(self.n_risk))
                         for s in range(self.NC)])

    def risk_marginal(self, b):
        return np.array([sum(b[self.fidx(r, s)] for s in range(self.NC))
                         for r in range(self.n_risk)])
