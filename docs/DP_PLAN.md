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

## Verification-driven experiments (2026-09-04)

An independent verification of `paper/manuscript.md` against the pipeline
(numbers, implementation, theory, gaps) motivated the following additions;
every item has a generating script and a results file under `results/dp/`.

* **Screening-only comparators disclosed; surveillance-augmented comparator
  added** (`dp/surveillance_arms.py`, hook `FixedSurveillanceHook` in
  `dp/hooks.py`; `eval_surveillance_n1000000.{json,md}`). Every engine arm
  runs with CMOST's built-in surveillance off, so the fixed schedules of the
  paper are screening-only. The new arms `q10y_surv` / `q5y_surv` reproduce
  CMOST13's `Polyp_Surveillance` block on the hook side (single annual
  decision, surveillance exams counted as programme colonoscopies, clocks
  reset by the engine's real findings) on top of CMOST's rolling screening
  semantics, at n = 1 000 000 paired with the headline arms; an adaptive
  policy solved at a price matched to the 10-y + surveillance volume
  (`c6bhi2`, lambda 0.001189, cap 1500) is deployed for a paired
  matched-volume contrast. Manuscript: section 2.6, section 3.3, Table 5b.
* **Deployment fallbacks counted** (`BeliefPolicyHook.counters()`,
  `Policy.n_fallback`; written to every chunk summary as `hook_counters`).
* **Solver robustness and belief-set coverage** (`dp/robustness.py`;
  `robustness_solver.{json,md}`, `eval_robustness_n200000.json`): the
  headline price re-solved with rollout seeds 1 and 2 and with reference
  propensities 0.06 and 0.25 (cap 1500), plus a same-seed re-solve
  (determinism check) that also computes `PBVISolver.density_diagnostic`
  -- reach-weighted L1 distance from the policy's reachable beliefs and
  their one-step deviations to the final belief sets, by age. All variants
  deployed in the engine on the headline arm's first four chunk seeds.
* **FIB optimality gaps** for every policy family and the cap 600 -> 1500
  sensitivity of both bounds (`dp/gap_table.py`, `fib_gaps.{json,md}`).
* **Kernel back-off usage** weighted by the headline policy's occupancy,
  own-cell sample sizes of the tau >= 13 cells, tau composition of their
  person-years (`dp/kernel_support.py`, `kernel_support.{json,md}`;
  `dp/estimate_kernels.py` now writes `tau_rowcounts`).
* **tau-support sensitivity** (`dp/estimate_kernels.py --tau-max 20`,
  `dp/tau_sensitivity.py`; `kernels_c6b_tau20.npz`, `tau_sensitivity.{json,md}`,
  `eval_tau_sensitivity_n200000.json`): WAIT person-years at tau > 20
  excluded, headline price re-solved, policy deployed. Re-estimating the
  paper's kernels with the current code reproduces `kernels_c6b.npz`
  bit-for-bit (`kernels_c6b_re.npz`).
* **Generating scripts for the engine tables** with t(19) intervals,
  person-level paired SEs and Holm-adjusted p-values
  (`dp/paired_tables.py`, `paired_tables.{json,md}`); one population
  definition (alive and undiagnosed at the age-40 decision snapshot,
  n = 965 258) now used for Tables 1, 2, 3 and 6 (`dp/ablate.py`
  `engine_reference` updated accordingly).
* **Engine instrumentation regression test**
  (`tests/test_engine_hook_regression.py`): `NumberCrunching_policy` with
  no hook, with and without recorders, is bit-identical to
  `NumberCrunching_100000` on the same seed.
* **Manifest** of the gitignored engine output and policy files
  (`dp/manifest.py`, `results/dp/manifest.json`).
* The engine cache lives in the worktree
  `.claude/worktrees/colonoscopy-policy-optimization-ecf7f0/results/dp/`;
  `results/dp/runs` and `results/dp/policies` in the main checkout are
  directory junctions to it.

Commands (from the repository root; engine workers were capped at 4 because
other jobs shared the machine):

    python -m dp.surveillance_arms --run --n 1000000 --workers 4 && python -m dp.surveillance_arms --analyse --n 1000000
    python -m dp.robustness --solve --workers 2 && python -m dp.robustness --deploy --n 200000 --workers 4 && python -m dp.robustness --report --n 200000
    python -m dp.gap_table
    python -m dp.kernel_support
    python -m dp.estimate_kernels --tag c6b_tau20 --tau-max 20 --cuts 0.5 0.8 0.95 0.965 0.98 --screen-runs results/dp/runs/screen_random_q,results/dp/runs/screen_random_q2
    python -m dp.tau_sensitivity --solve --workers 2 && python -m dp.tau_sensitivity --deploy --n 200000 --workers 4 && python -m dp.tau_sensitivity --report --n 200000
    python -m dp.sweep --kernels results/dp/kernels_c6b.npz --objective death --lams 0.001189 --tag c6bhi2 --cap 1500 --rounds 4 --rollouts 300 --workers 2
    python -m dp.paired_tables
    python -m dp.ablate
    python tests/test_engine_hook_regression.py
    python -m dp.manifest

Findings (details in the results files named above and in the manuscript):

* 10-y + CMOST surveillance: 2.946 colos/person (0.574 surveillance),
  879.8 CRC deaths /100k -- 20 +- 12 fewer than the adaptive lambda 0.001561
  policy but with 29 % more colonoscopies; the adaptive policy solved for its
  volume (lambda 0.001189, 2.874 colos) has 62 +- 12 fewer deaths at a similar
  number of diagnoses (+18 +- 17). 5-y + surveillance: 5.143 colos, 740.2
  deaths, 72 +- 11 more than the adaptive incidence-objective policy at the
  same volume (5.149). The surveillance programme is statistically
  indistinguishable from the paper's three-tier rule (+13 +- 13, +0.06 colos).
* FIB gap 57 % of the objective at the headline price (~700 deaths /100k in
  death-equivalents vs an in-model adaptive-minus-best-fixed advantage of
  ~90); cap 600 -> 1500 moves the lower bound by ~1e-6, the FIB bound by 0.
  Same-seed re-solve is bit-identical; belief-set coverage of one-step
  deviations: mean L1 0.007 / 0.004 (m/f), 93 / 96 % of mass within 0.01.
* Rollout seeds 1-2 and reference propensities 0.06 / 0.25: objectives within
  3e-7 of the headline; policy paths identical except one-year shifts of the
  exams after >= 3 adenomas (the "re-examine after one year" feature is a
  near-tie, not a robust prediction). Engine at 200k: 884.5-933.0 deaths
  /100k vs 889.5; the seed-1 variant extended to 1M: 903.8 +- 7.4, paired
  +4.0 +- 12.0 vs the headline arm, -128.3 +- 11.3 vs q10y.
* Deployment counters (6.84 M decisions): 0 fallbacks, 0 impossible
  observations, 0 low-probability updates; re-deployed headline arm identical
  to the cached one.
* Kernel support: 96.4 % of decision-age WAIT occupancy on own-cell rows at
  the exact age; repeat colonoscopies at tau 1-3 are the least-supported
  rows (4 % of colonoscopies); the tau >= 13 cell contains only tau 13-20
  person-years at decision ages. tau > 20 excluded: engine 887.5 (16.3) vs
  889.5 (paired -2 +- 25).
* Population definition unified (age-40 decision snapshot, n = 965 258):
  Table 1 errors 0.5 / 1.7 / 5.3 % (deaths), Table 2 engine row
  45.6 / 41.6 / 59.3 / 52.3 %.

