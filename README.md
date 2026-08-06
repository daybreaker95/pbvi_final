# Personalized colonoscopy screening as a POMDP on the CMOST microsimulation

**Branch:** `pbvi_thlee`  ·  **Author:** T.H. Lee (with Claude)

This folder turns the CMOST colorectal-cancer (CRC) microsimulation
(Prakash et al. 2017) into an **individual-level, gym-style environment**,
uses it to **empirically estimate** a 6-state natural-history Markov model,
**validates** that model against the microsimulation and the literature, and
then solves a **partially observable Markov decision process (POMDP)** with
**point-based value iteration (PBVI)** to obtain a *personalized, history-
dependent* colonoscopy schedule. We compare it, in the true CMOST environment,
against the fixed population schedules of Zaika et al. (2024).

The scientific question: **can a policy that adapts each person's screening timing
to their own past colonoscopy findings improve on the best fixed population
schedule at a matched number of colonoscopies?** Under the corrected quarterly-
composed model the answer is nuanced: the adaptive policy is *statistically
indistinguishable* from a re-optimized fixed schedule on life-years (LYG is
Monte-Carlo-noise-dominated) but reaches *comparable CRC mortality with fewer
colonoscopies*, and it beats Zaika's published ages on CRC mortality at every
budget. See `paper/results_comparison.md` and `transitions/SCREENING_VALIDATION.md`.

---

## 1. Clinical state discretization

Following standard medical-decision-analytic practice, the patient's whole-colon
status is discretized by the **most severe lesion** into 6 states:

| index | state | CMOST definition |
|------|-------|------------------|
| 0 | Normal | no polyp / no cancer |
| 1 | Early Adenoma | max polyp stage 1–4 (non-advanced) |
| 2 | Advanced Adenoma | max polyp stage 5–6 (`Tumor>4` in CMOST) |
| 3 | Preclinical Cancer | undetected carcinoma (stage 7–10) |
| 4 | Clinical Cancer | symptomatically/screen-diagnosed carcinoma |
| 5 | Dead | any cause |

The belief state is `b_t = [p_Normal, p_Early, p_Advanced, p_Preclinical,
p_Clinical, p_Dead]` with `Σ p_i = 1`.

## 2. What is in this folder

```
pbvi_thlee/
├── env/                     individual-level CMOST engine + gym environment
│   ├── cmost_individual.py  faithful per-patient, quarterly-stepping CMOST engine
│   ├── crc_env.py           gym-style env: reset()/step(), true state + observation
│   └── params.py            builds the exact CMOST parameter bundle
├── transitions/             empirical transition estimation + validation
│   ├── estimate_transitions.py   age-stratified 6x6 matrices (+ bootstrap CI)
│   ├── estimate_stratified.py    low/high risk-class matrices
│   └── validate_transitions.py   V1 (Markov trace), V1b (Markov property),
│                                 V2 (Monte-Carlo convergence), V3 (vs literature)
├── pomdp/                    POMDP model + solver
│   ├── model.py             6-state (x risk) POMDP: T (empirical), O, reward
│   ├── pbvi.py              point-based value iteration + belief-tracking policy
│   └── estimate_effects.py  colonoscopy detection probs + cancer life-year values
├── experiments/
│   ├── evaluate_policies.py policy roll-outs with common random numbers
│   └── run_comparison.py    PBVI vs Zaika fixed schedules (efficiency frontier)
├── results/                 estimated matrices, validation figures, policies
├── paper/                   manuscript-ready figures + methods/results write-up
└── tests/                   smoke tests (engine, POMDP)
```

## 3. Pipeline (how to reproduce)

```bash
# 1. estimate natural-history transition matrices (100k patients, ~6 min)
python transitions/estimate_transitions.py -n 100000 --bootstrap 200
python transitions/estimate_stratified.py 100000        # low/high risk classes

# 2. validate them (Markov-trace, Monte-Carlo convergence, vs CRC-SPIN/SEER)
python transitions/validate_transitions.py

# 3. estimate colonoscopy detection + cancer life-year values
python pomdp/estimate_effects.py

# 4. solve the POMDP and compare against Zaika 2024 in the true CMOST env
python experiments/run_comparison.py 30000

# 5. follow-on analyses: when/where/for-whom does the adaptive policy help?
python experiments/nonadherence.py 30000   # robustness to imperfect adherence
python experiments/subgroups.py 40000       # patient subgroups + budget/surveillance regime
python experiments/risk_factors.py 25000    # baseline risk factors (FH + prior adenoma) + cost budget
python experiments/prs_targeting.py 25000   # strong PRS (AUC>=0.8) turns on mortality-targeting
```

### Follow-on findings (see `paper/results_*.md`)

The matched-adherence comparison (step 4) finds the adaptive policy *statistically
comparable* to a re-optimized fixed schedule. Steps 5 probe **when that changes**:

* **Non-adherence** (`results_nonadherence.md`). Under imperfect adherence a fixed
  schedule loses missed slots permanently, whereas the POMDP re-plans for free (a
  no-show yields no observation and consumes no budget, so re-recommending is
  optimal). As adherence falls the fixed-schedule benefit collapses toward
  no-screening while the adaptive policy holds; the **non-adherent subgroup** gains
  a ~0.45 pp (≈27–30 % relative) CRC-mortality reduction.
* **Subgroups / budget** (`results_subgroups.md`). At matched adherence PBVI gives
  **no reliable benefit in any risk subgroup** (it cannot pre-identify latent risk
  from clean screens); its real edge is **colonoscopy efficiency** and the
  **high-budget/surveillance regime**, where a uniform fixed schedule saturates.
* **Baseline risk factors** (`results_risk_factors.md`). Genuine risk-targeting
  needs two changes: a **personalized prior** from family history + prior-adenoma
  history, *and* a **cost-based budget** (not a hard per-person cap) so quantity can
  vary by risk. With both, PBVI+risk-factors < risk-stratified fixed < plain fixed
  on the efficiency frontier; at realistic discrimination (AUC≈0.67) the payoff is
  efficiency, not a high-risk mortality drop.
* **Polygenic risk score** (`results_prs.md`). Pushing the baseline discrimination
  to **AUC≥0.8** (a PRS) at a **low per-colonoscopy cost** turns ON
  mortality-targeting: the true high-risk class receives intensive surveillance and
  its CRC mortality drops **below** the fixed schedule at equal-or-lower total
  colonoscopy use.

## 4. Method summary

* **Environment.** `env/cmost_individual.py` re-implements every quarterly CMOST
  event (adenoma initiation, growth through 6 stages, the direct/fast cancer
  paths, regression, symptomatic presentation, stage progression, colonoscopy
  detection & complications, competing mortality) for a single patient, reusing
  the *exact* parameter bundle produced by `calculate_sub.prepare_parameters`.
  `env/crc_env.py` exposes it as a gym environment whose hidden true state is
  always available (`info['true_state']`) while the agent sees only colonoscopy
  observations.

* **Empirical transitions.** We simulate a large no-screening cohort, record the
  6-state trajectory of every patient-year, and estimate the age-specific
  transition matrix by maximum likelihood, with multinomial standard errors and
  a patient-level bootstrap. (`transitions/`)

* **Validation** (per Krijkamp et al. 2018, *Med Decis Making*, NIHMS931782):
  (V1) a deterministic cohort **Markov trace** reproduces the microsimulation
  occupancy (implementation check); (V1b) a genuine **Markov-property test**;
  (V2) **Monte-Carlo convergence** of the estimates; (V3) natural-history
  outputs vs **CRC-SPIN/CISNET** and SEER targets.

* **POMDP + PBVI.** Age and screening budget are observed; the 6 clinical states
  (× a latent low/high adenoma-risk class) form the belief. Colonoscopy
  observations are discriminative, so the belief tracks each individual's
  findings; PBVI (Pineau et al. 2003) computes the value function by backward
  induction over age. The resulting policy screens sooner for patients in whom
  adenomas are found (inferred high risk) and later for those repeatedly clean.

## 5. Key references

* Prakash et al. (2017) *CMOST*, PLoS ONE — the microsimulation.
* Zaika et al. (2024) *Optimal timing of colonoscopy screening*, PLoS ONE — the
  fixed-schedule comparator (same CMOST model).
* Krijkamp et al. (2018) *Microsimulation modeling … a tutorial*, Med Decis
  Making (NIHMS931782) — verification/validation methodology.
* Pineau, Gordon & Thrun (2003) *Point-based value iteration*, IJCAI.
* Rutter et al. — CRC-SPIN natural-history targets (CISNET).
```
