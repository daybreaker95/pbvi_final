# Plan: DP-optimised colonoscopy policy that beats q10y/q5y per colonoscopy

Goal (owner): using dynamic programming on a Markov/POMDP approximation of
CMOST, find a colonoscopy policy whose CRC-death and CRC-incidence reduction
**per colonoscopy** exceeds the fixed 10-year (50/60/70) and 5-year (50..75)
schedules, verified inside the real CMOST engine.

## Diagnosis of the existing pipeline (2026-08-23)

| issue | evidence | consequence |
|---|---|---|
| Objective misaligned | `pomdp/model_v2.py` optimises QALY-NMB (γ=0.97, WTP) − λ | policy optimised for a different metric than the one reported |
| Synthetic observations | `tests/cmost_4way_eval.py` `obs()` samples o from the model's own kernel given the true pre-colonoscopy state, not from what the engine actually found/removed | belief and engine diverge; diagnosed patients keep receiving (wasted) policy colonoscopies |
| FiVI bounds uninformative | `results/fivi_gap_history.json`: vu frozen from iter 1 (block-diagonal risk structure defeats the sawtooth bound); policy flips across iterations/seeds | convergence unverifiable |
| Natural history from re-implementation | transition matrices estimated from `env/cmost_individual.CRCEngine`, which under-produces incidence by ~3.6 % vs the real engine | transfer gap |
| Post-polypectomy state | SCREEN operator sends a detected polyp to `Normal` w.p. 1; engine leaves residual polyps (per-lesion detection) | policy under-estimates post-polypectomy recurrence |
| Risk heterogeneity | `individual_risk`: median 0.78, top-10 % mean 17.1, top-1 % = 44 | a 2-class split wastes identifiable heterogeneity |
| Evaluation noise / contention | 1 M-person engine runs ≈ 10 GB each; 4 parallel 1 M runs swap (15× slowdown) | chunked parallel evaluation needed |

## New pipeline (`dp/` package, old code untouched)

1. **Engine instrumentation** (`cmost_engine/NumberCrunching_policy.py`):
   `quarterly_recorder` (as in `NumberCrunching_100000.py`) and a richer
   hook call `obs(a, s_true, info)` where `info` carries the engine's real
   colonoscopy result (`s_post`, polyps removed, cancer detected,
   complication death). Backward compatible (`wants_info` attribute).
2. **Chunked, paired, parallel engine runner** (`dp/engine_runner.py`):
   every arm simulates the same chunk seeds (identical population and RNG
   stream until first divergence) in 50 k-person chunks; worker pool size
   configurable; every (arm, chunk) result is cached as JSON → resumable.
3. **Model grounding in the real engine**
   * `dp/estimate_nh.py`: quarterly-recorded no-screening cohort →
     annual (composed) undetected-world matrices and 4-vector symptomatic
     exit probabilities per (sex × risk class); classes = quantile cuts of
     `individual_risk` (default 3 classes: <50 %, 50–90 %, ≥90 %).
   * `dp/estimate_screen.py`: randomised-interval screening cohort →
     empirical SCREEN kernel P(observation, post-state | pre-state, sex,
     class, age-band), including residual-polyp and missed-lesion mass.
   * Post-diagnosis survival reused from `results/T_detected_tauphase.npz`
     (already real-engine derived).
4. **Reduced POMDP** (`dp/model.py`): hidden state = class × {Normal,
   EarlyPolyp, AdvPolyp, U-I..IV}; diagnosis / death are observed terminal
   exits whose remaining value (future CRC death probability, life-years) is
   folded into the exit reward. Reward modes: `deaths`, `incidence`,
   `lifeyears`, or weighted; λ per colonoscopy; undiscounted.
5. **Solver** (`dp/solver.py`): finite-horizon point-based value iteration
   on the closure of reachable beliefs (dedup by rounding, probability-
   weighted cap), alternated with exact policy-reachable-set expansion until
   the policy is stable; FIB upper bound for a gap; **exact in-model policy
   evaluation** by forward propagation on the policy's belief tree (no MC
   noise) → deaths, diagnoses, colonoscopies, life-years per person.
6. **λ sweep** (`dp/sweep.py`): in-model efficiency frontier; λ values whose
   in-model colonoscopy volume matches q10y (≈2.6) and q5y (≈5.0).
7. **Fixed-schedule frontier** (`dp/fixed_search.py`): exhaustive in-model
   search over fixed schedules (equal-interval and free k ≤ 4) → best fixed
   schedule at each volume, verified in the engine. Makes the claim robust
   against "you only beat naive comparators".
8. **Engine evaluation** (`dp/evaluate.py`): arms = no_screen, q10y, q5y,
   best-fixed, policy(λ_i); paired chunk-level CIs; per-colonoscopy
   efficiency with the denominator = policy/screening colonoscopies (and
   total colonoscopies as sensitivity).
9. **Report** (`dp/report.py`): tables + figures for the manuscript.

## Compute plan

8 physical cores / 16 threads, 34 GB. Other user jobs currently occupy the
machine (8 × `run_calibration.py` + a 16-worker `run_stage12_recalibrate.py`
pool) — runner parallelism defaults to 6 and is configurable.
Engine cost ≈ 2.8 ms/person (contended) → 1 M persons ≈ 47 CPU-min.

## Status (2026-08-24): complete

Pipeline implemented in `dp/` and run end-to-end; final kernels `c6b`
(6 risk classes, 2M never-screened + 2M randomised-screening persons).
Headline engine evaluation (n = 1M/arm, paired): the DP policy family
strictly dominates the fixed 10-y and 5-y schedules AND the exhaustively
searched best fixed schedules on CRC deaths and diagnoses per colonoscopy
(matched-volume deaths −18/−19 %, per-colonoscopy efficiency +30 %/+18 %).
Numbers: `paper/dp_results.md`, `results/dp/report_c6b.md`; methods:
`paper/dp_methods.md`; figures: `paper/figures/dp_*.png`.
Reproduce: `python -m dp.run_cohorts` -> `python -m dp.run_pipeline --tag c6b
--cuts 0.5 0.8 0.95 0.965 0.98 --steps kernels,fixed,sweep,baseline,grid,headline,report
--objectives death,inc` (policies/runs are cached under `results/dp/`,
excluded from git for size).

## Follow-up analyses (2026-08-24/25)

* **Adherence** (`dp/run_adherence.py`, `results/dp/eval_adherence_c6b_n200000.json`):
  the same policy re-plans around no-shows without re-solving; at 50 %
  attendance it beats a recall-augmented fixed programme on mortality with
  22 % fewer colonoscopies, while the classic fixed programme collapses.
* **Life-year objective** (`dp/sweep.py --objective ly`): the in-model gain
  does not transfer to the engine; reported as a secondary null.
* **Model-structure ablation** (`dp/ablate.py`, `results/dp/ablation.json`,
  `dp/_abl_policy.py`): the (tau, last finding) memory is the load-bearing
  modelling choice for predictive accuracy, while the per-colonoscopy
  dominance over fixed schedules survives every coarsening tested.
* **Manuscript**: `paper/manuscript.md` (rewritten on these results;
  every number machine-verified against `results/dp/`).
* **Baseline risk score of finite discrimination** (`dp/riskscore.py`,
  `dp/run_riskscore.py`, `dp/score_frontier.py`, `dp/score_fixed.py`,
  `dp/report_riskscore.py`): score-conditional beliefs (continuous, not
  banded), sigma calibrated to a target within-sex AUC, one policy per sex
  solved over 14 tail-dense score-band roots, each level read at a matched
  colonoscopy volume off its own price-volume frontier. Adds the
  score-stratified FIXED comparator that separates information from
  adaptivity. Design was reviewed by a statistical/clinical/decision-analytic
  panel first; the panel's fatal findings (band-conditioned beliefs, a
  population-prior-only PBVI refinement, unmatched volumes, AUC-only
  anchoring) are all fixed in the shipped code.
* **Post-hoc verification of the risk-score section** (24-agent fact-check
  workflow, every claim recomputed from the artefacts) found and fixed:
  - the score-stratified FIXED arm deployed the MALE band schedules to
    everyone (`BandFixedScheduleHook` took no sex). Fixed; the corrected
    engine arm is 2.262 colos / 896.0 deaths per 100 000, versus
    2.220 / 938.0 before;
  - its in-model cell was reported at the price-bisection's landing volume
    (2.343) rather than at the target 2.369. `dp/score_fixed.py` now builds
    the full price-volume envelope and reads it at the target, exactly as
    `dp/score_frontier.py` does for the adaptive arms (946 -> 942), and also
    builds the like-for-like sigma -> infinity comparator (972);
  - the control's AUC was the hard-coded nominal 0.500. It is now measured
    (`riskscore.control_auc`, 0.507) and cached, because sigma = 60 is a
    stand-in for sigma -> infinity, not infinity;
  - `band_posteriors` sampled the risk pool uniformly rather than by age-40
    frequency (the same defect fixed earlier in the deployment path; the
    numbers barely move, 0.1409 vs 0.1419, but the paths now agree);
  - manuscript arithmetic and transcription: "a fifth" -> "a third" of what
    adaptivity buys; a bottom-decile colonoscopy count read off the AUC-0.65
    arm instead of the AUC-0.60 one (0.71 -> 1.11); "top 2 %" -> "top 5 %"
    for the five-colonoscopy band; "top band ... 0.14" -> top DECILE 0.14,
    top 1 % 0.23; the engine ladder sentence now states that the arms are not
    volume-matched and gives the volume-matched gaps (-79, -104).
  Belief conservation re-verified: max |sum_k w_k b_k - population prior| is
  0.0011 on the 14 deployment roots and 0.0001 on the 2048-cell table.
  A second verification round on the corrected text found two more:
  - the unstratified fixed comparator quoted next to Table 8 was scored on
    each sex's OWN age-40 prior while every Table 8 cell uses the score arms'
    sex-pooled belief. Re-scored on the pooled belief the best unisex
    schedule is 980.0, not 970.3, and the best score-blind SEX-SPECIFIC
    programme is 972.1 - which reproduces the no-score cell (971.6) to within
    0.5 and is therefore a second wiring check: an uninformative score must
    collapse the fixed arm to a sex-specific fixed programme, and it does;
  - "more than three times" overstated the adaptivity/score ratio, which is
    2.67 measured from the shared fixed/no-score corner and 3.04 measured
    with the other ingredient present. Now "about three times", in §3.6 and
    in the abstract.
