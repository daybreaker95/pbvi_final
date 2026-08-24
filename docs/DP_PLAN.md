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
