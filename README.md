# Risk-stratified colonoscopy screening as a POMDP on the CMOST microsimulation

**Branch:** `pbvi_thlee`  ·  **Author:** T.H. Lee (with Claude)

This project turns the CMOST colorectal-cancer (CRC) microsimulation (Prakash
et al. 2017) into an **individual-level, gym-style environment**, empirically
estimates a 9-state natural-history Markov model from it, layers a
**Korean-epidemiology-derived risk score** on top to stratify individuals into
high/low CRC-risk groups, and solves a **partially observable Markov decision
process (POMDP)** with a value-iteration solver (FiVI) to obtain a
**risk-aware, budget-tunable** colonoscopy schedule. The resulting policy is
compared, inside the same CMOST engine, against fixed-interval comparators
(uniform 10-year and 5-year screening).

The scientific question: **does routing screening intensity by a
population-derived risk score improve on uniform-interval screening at a
matched colonoscopy budget?** A single scalar shadow price (λ) on
colonoscopy cost lets the policy trade mortality reduction against
colonoscopy volume continuously; the Results in `manuscript/draft.md` report
where that trade currently lands.

---

## 0. NEW (2026-08): `dp/` — engine-grounded MOMDP pipeline (current)

The `dp/` package supersedes the `pomdp/` + `tests/` pipeline for the
per-colonoscopy-efficiency question. It estimates ALL model kernels directly
from the real engine (`cmost_engine/NumberCrunching_policy.py`, instrumented
with a quarterly recorder and a real-findings hook), conditions the natural
history on the observed memory (years since last colonoscopy x last finding)
and a 6-level latent risk class, solves the finite-horizon MOMDP by
vectorised point-based value iteration with exact in-model policy
evaluation, and evaluates everything back inside the real engine with
population-paired seeds. Result (n = 1M/arm): the DP policy family strictly
dominates fixed 10-y / 5-y schedules and the best fixed schedules from an
exhaustive 2112-schedule search, on both CRC deaths and diagnoses per
colonoscopy; it is robust to imperfect adherence, and a model-structure
ablation shows the dominance survives substantial misspecification.
The manuscript written on these results is `paper/manuscript.md` (the
earlier six-state version is kept as
`paper/manuscript_v1_6state_superseded.md`). See also `docs/DP_PLAN.md`,
`paper/dp_methods.md`, `paper/dp_results.md`, `results/dp/report_c6b.md`.

## 1. Clinical state discretization

The patient's whole-colon + cancer status is discretized into **9 states**
(`env/state9.py`):

| index | state | index | state |
|---|---|---|---|
| 0 | Normal | 5 | Cancer III |
| 1 | Early Polyp | 6 | Cancer IV |
| 2 | Advanced Polyp | 7 | CRC Death |
| 3 | Cancer I | 8 | Other-cause Death |
| 4 | Cancer II | | |

The belief state is `b_t ∈ Δ^9` over these indices; the agent observes the
patient's age, colonoscopy history, and prior findings, but not the true
state directly.

## 2. Risk stratification methodology

Colonoscopy budget is finite, so *who* gets screened more often matters as
much as *when*. Risk stratification here is built in three layers:

**(a) Composite risk score.** Each simulated patient is assigned a
log-additive relative-risk (RR) score from published hazard/odds ratios for
modifiable risk factors — BMI, diabetes, alcohol, family history, and 7
dietary factors (fiber, calcium, folate, processed meat, red meat, fruit,
vegetable). Age is deliberately **excluded** from the score: CMOST already
models age-dependent onset natively (`env/cmost_individual.py`'s
`new_polyp[yi]` curve), and `individual_risk` is a separate,
age-*independent* lifelong multiplier — including an age term in the score
would double-count age. See `tests/jeon_elbow_analysis.py`'s module
docstring for the exact per-factor sourcing and the reasoning above.

Two variants exist in `tests/`, reflecting an evolution during development:
  - `nhic_elbow_analysis*.py` — the original 5-factor score (BMI, glucose,
    cholesterol, family history, alcohol) from Shin et al. 2014 (Korean
    NHIC cohort), later extended with 7 dietary factors from Jeon et al.
    2018 (`nhic_elbow_analysis_diet.py`) to fix a severe combination-count
    shortage (only 69/16 distinct scores for men/women — far coarser than
    CMOST's own 500-slot risk pool, causing tie-breaking artifacts).
  - `jeon_elbow_analysis.py` (**current**) — all factors re-sourced from
    Jeon et al. 2018 alone, for source consistency (a single Western-cohort
    paper throughout, rather than mixing a Korean NHIC core with a
    Western-cohort dietary extension). This is the version the current
    `transitions/estimate_transitions_9state_jeon_risk.py` and
    `tests/jeon_4way_eval.py` / `tests/jeon_lambda_sweep_real_engine.py`
    pipeline uses.

**(b) Mapping onto CMOST's own risk pool.** CMOST's `individual_risk`
parameter (a multiplicative polyp-rate factor, `env/cmost_individual.py`
line ~477) is drawn from a 500-value, right-skewed native pool (built to
reproduce hereditary-syndrome-scale outliers CMOST's US calibration
targets — the composite score above, built from modifiable lifestyle
factors alone, cannot and should not reproduce that tail). Two mapping
strategies were evaluated:
  - **Continuous rank-preserving mapping** (`percentile_map_to_individual_risk`):
    each person's exact percentile rank in the composite score is matched to
    the same percentile in CMOST's pool. Used only for the diagnostic elbow
    sweep now (cheap: one simulation, many cutoffs evaluated post hoc).
  - **Binary bucket mapping** (`bucket_map_to_individual_risk`, **current
    default**): the top `high_frac` by composite score draws (with
    replacement) from CMOST's own top-`high_frac` sub-pool; the rest draws
    from the remaining sub-pool. This is deliberately coarser — the
    downstream POMDP agent only needs a high/low label, not a fine rank —
    and both the classification threshold and the CMOST donor-pool split
    use the *same* `high_frac`.

**(c) Cutoff selection.** `high_frac` (currently 0.20, i.e. top 20% =
"high risk") is chosen from an RR-vs-cutoff sweep (`--sweep` flag on the
elbow-analysis scripts): the point where the high/low CRC-death-rate ratio
stops improving with a stricter cutoff.

## 3. Absolute-risk validation (independent of CMOST)

Because the CMOST-internal RR at any cutoff is partly a mechanical
consequence of the mapping (higher `individual_risk` *always* raises
simulated incidence), it cannot validate whether the *composite score's own
magnitude* is epidemiologically realistic. A separate, CMOST-independent
check applies the composite score's relative-risk multiplier directly to
**KOSIS** (Statistics Korea) real 2023–2024 age/sex-specific CRC incidence
and mortality rates, producing an "excess cases per 10,000" figure at each
risk percentile — the same style of absolute-risk communication used by
Archambault et al. 2022 (JNCI) for early-onset CRC risk scores. This
sanity-checks the score's plausibility against real Korean population data,
completely independent of how it is later mapped into CMOST.

## 4. What is in this folder

```
pbvi_thlee/
├── env/                       individual-level CMOST engine + gym environment
│   ├── cmost_individual.py    faithful per-patient, quarterly-stepping CMOST engine
│   ├── crc_env.py             gym-style env: reset()/step(), true state + observation
│   ├── state9.py              9-state clinical discretization + classifier
│   └── params.py              builds the exact CMOST parameter bundle
├── transitions/                empirical transition estimation + risk-stratified variants
│   ├── estimate_transitions_9state.py            pooled age-stratified 9x9 matrices
│   ├── estimate_transitions_9state_sex_risk.py   sex x CMOST-native-risk matrices
│   ├── estimate_transitions_9state_nhic_risk.py  sex x NHIC-mapped-risk (5-factor)
│   ├── estimate_transitions_9state_nhic_diet_risk.py  + Jeon dietary factors (12-factor mix)
│   └── estimate_transitions_9state_jeon_risk.py  sex x Jeon-2018-only bucket risk (current)
├── pomdp/                      POMDP model + solver
│   ├── model_v2.py             CRCScreeningPOMDP9: 9-state (x sex x risk) POMDP, T/O/reward
│   ├── fivi.py                 value-iteration solver + belief-tracking policy
│   └── estimate_effects.py     colonoscopy detection probs + cancer life-year values
├── tests/                      real-engine evaluation, risk-score sweeps, lambda sweeps
│   ├── cmost_4way_eval.py      no_screen / q10y / q5y / policy comparison (shared engine)
│   ├── jeon_elbow_analysis.py  current risk-score construction + cutoff sweep
│   ├── jeon_4way_eval.py       4-way comparison using the Jeon-2018 bucket-mapped score
│   └── jeon_lambda_sweep_real_engine.py   colo_penalty_qaly (lambda) grid search
├── results/                    estimated matrices, sweep outputs, 4-way comparison JSON
└── paper/                      manuscript-adjacent figures + methods/results write-up
```

## 5. Pipeline (how to reproduce the current risk-stratified policy)

```bash
# 1. find the risk-score cutoff (RR-vs-cutoff elbow, KOSIS absolute-risk check
#    is a separate, standalone calculation -- see Section 3 above)
python tests/jeon_elbow_analysis.py --n 1000000 --sweep

# 2. estimate the 4 (sex x risk) transition matrices at the chosen cutoff
python transitions/estimate_transitions_9state_jeon_risk.py -n 200000 --high_frac 0.20

# 3. lambda=0 baseline for all 4 scenarios (no_screen / q10y / q5y / policy)
python tests/jeon_4way_eval.py --scenario no_screen -n 1000000
python tests/jeon_4way_eval.py --scenario q10y -n 1000000
python tests/jeon_4way_eval.py --scenario q5y -n 1000000
python tests/jeon_4way_eval.py --scenario policy -n 1000000

# 4. lambda (colonoscopy-cost shadow price) sweep -- run ALONE, not alongside
#    other heavy jobs (see that script's docstring: ~15x CPU-contention
#    slowdown was observed running 4 processes in parallel previously)
python tests/jeon_lambda_sweep_real_engine.py
```

## 6. Method summary

* **Environment.** `env/cmost_individual.py` re-implements every quarterly
  CMOST event (adenoma initiation, growth, direct/fast cancer paths,
  regression, symptomatic presentation, stage progression, colonoscopy
  detection & complications, competing mortality) for a single patient,
  reusing the exact parameter bundle from `calculate_sub.prepare_parameters`.

* **Empirical transitions.** A large no-screening cohort is simulated, and
  age- and (sex x risk)-specific 9x9 transition matrices are estimated by
  maximum likelihood (`transitions/`).

* **POMDP + solver.** Age, sex, and risk class are observed; the 9 clinical
  states form the belief. Colonoscopy observations are discriminative, so
  the belief tracks each individual's findings; the FiVI solver
  (`pomdp/fivi.py`) computes a policy by value iteration over the belief
  simplex. `colo_penalty_qaly` (λ) is a shadow price on colonoscopy cost in
  the reward function — sweeping it traces out the efficiency frontier
  between mortality reduction and colonoscopy volume.

## 7. Key references

* Prakash et al. (2017) *CMOST*, PLoS ONE — the microsimulation.
* Shin A, et al. (2014) *PLoS ONE* 9(2):e88079 — Korean NHIC cohort CRC risk
  model (metabolic factors, family history, alcohol).
* Jeon J, Du M, Schoen RE, et al. (2018) *Gastroenterology* 154(8):2152-2164.e19
  — lifestyle/environmental/genetic CRC risk score (E-score), current source
  for all composite-score hazard ratios.
* Archambault AN, et al. (2022) *JNCI* 114(4):528-539 — early-onset CRC risk
  stratification using combined genetic + environmental risk scores;
  methodological precedent for the absolute-risk (excess-cases-per-10,000)
  validation approach in Section 3.
* van den Puttelaar R, et al. (2023) *Clin Gastroenterol Hepatol*
  21(13):3415-3423.e29 — MISCAN-Colon risk-stratified screening
  cost-effectiveness; structural precedent for the percentile-mapping
  methodology and its group-vs-individual calibration caveat.
* Pashayan N, et al. (2018) *JAMA Oncol* 4(11):1504-1510 — risk-stratified
  breast-cancer screening; precedent for percentile-threshold NMB
  optimization (the λ-sweep here is the simulation-based analogue).
* KOSIS (Statistics Korea) — 2023 national cancer incidence
  (`DT_117N_A00023`) and 2024 cause-of-death mortality (`DT_1B34E01`)
  statistics, used for the absolute-risk validation in Section 3.
