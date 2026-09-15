# Adaptive colonoscopy screening as a MOMDP on the CMOST microsimulation

**Author:** T.H. Lee (with Claude)

This project asks whether a colonoscopy policy that **adapts each person's
next interval to what their previous colonoscopies found** can prevent more
colorectal-cancer (CRC) deaths and diagnoses *per colonoscopy* than any fixed
schedule — including fixed schedules optimised in the same simulator.

The CMOST microsimulation (Prakash et al. 2017) is treated as ground truth.
Every kernel of a finite-horizon **mixed-observability Markov decision
process (MOMDP)** is estimated directly from the instrumented CMOST engine,
the MOMDP is solved by point-based value iteration with exact in-model policy
evaluation, and every policy is evaluated back inside the real engine against
fixed comparators with population-paired random-number streams.

* Manuscript: [`paper/manuscript.md`](paper/manuscript.md)
* Pipeline design, status and verification log: [`docs/DP_PLAN.md`](docs/DP_PLAN.md)
* Methods / results drafts: [`paper/dp_methods.md`](paper/dp_methods.md), [`paper/dp_results.md`](paper/dp_results.md)
* Headline tables: [`results/dp/report_c6b.md`](results/dp/report_c6b.md)
* Figures: `paper/figures/dp_*.png` (index: [`paper/figures_index.md`](paper/figures_index.md))

---

## 1. Headline results (engine, n = 1 000 000 per arm, paired)

| arm | colonoscopies / person | CRC deaths / 100 000 | deaths averted / 1000 colonoscopies |
|---|---|---|---|
| no screening | 0 | 1887.6 | — |
| fixed 10-yearly (50/60/70) | 2.58 | 1032.1 | 3.32 |
| best searched fixed schedule (54/64/74) | 2.44 | 980.9 | 3.71 |
| **adaptive policy, λ = 0.001561** | **2.29** | **899.8** | **4.32** |
| fixed 5-yearly (50–75) | 4.99 | 775.3 | 2.23 |
| **adaptive policy, λ = 0.00069** | **4.59** | **682.5** | 2.63 |
| adaptive + observed risk class, λ = 0.001561 | 1.67 | 832.9 | 6.34 |

* The adaptive policy **dominates every screening-only fixed comparator** on
  CRC deaths and diagnoses at lower colonoscopy volume: +30 % deaths averted
  and +25 % diagnoses averted per colonoscopy vs the 10-yearly schedule, and
  it also beats the best of 2 112 exhaustively searched fixed schedules.
* A 10-yearly programme with CMOST's own post-polypectomy surveillance
  reaches 879.8 deaths / 100 000 but needs 29 % more colonoscopies (2.95);
  an adaptive policy solved for that volume has 62 ± 12 fewer deaths.
* Robust to imperfect adherence (re-plans around no-shows without
  re-solving). On life-years the adaptive and fixed schedules are
  statistically indistinguishable (reported as a null).

Full numbers, CIs and the remaining analyses are in the manuscript §3.

## 2. Model

**CMOST engine** — `cmost_engine/NumberCrunching_policy.py` (Python port,
CMOST13 parameters) with two record-only instruments: a quarter-resolved
18-state recorder and an annual decision hook that receives the engine's
*actual* colonoscopy result. Without a hook the instrumented engine is
bit-identical to the un-instrumented port
(`tests/test_engine_hook_regression.py`).

**MOMDP** (`dp/model.py`, `dp/kernels.py`), one per sex:

| component | definition |
|---|---|
| observed | age; τ = years since last colonoscopy (never, capped at 13); last finding ∈ {normal, 1–2 early adenomas, ≥3 early adenomas, advanced adenoma} |
| hidden (belief) | 6 latent risk classes (quantile bins of `individual_risk`, cuts 50/80/95/96.5/98 %) × 11 clinical states (N, P1–P6, U1–U4) = 66 states |
| actions | WAIT / SCREEN, annually at ages 40–80 (outcomes accrue to 100) |
| exits | diagnosis at stage I–IV, other-cause death, complication death — terminal, with remaining-lifetime value folded into the reward |
| objective | minimise E[CRC deaths] (or diagnoses) + λ · E[colonoscopies], undiscounted; sweeping λ traces the efficiency frontier |

**Kernels** (`dp/estimate_kernels.py`) — maximum likelihood on annual
person-year windows from two engine cohorts: 2 M never-screened lives (WAIT
kernels) and 2 M lives on randomised colonoscopy schedules (SCREEN kernel and
post-colonoscopy natural history), with hierarchical back-off for sparse
cells. Conditioning on (τ, last finding) is the load-bearing modelling choice
(`dp/ablate.py`).

**Solver** (`dp/solver.py`) — finite-horizon point-based value iteration with
belief and α-vector sets indexed by the observed key (age, τ, finding):

1. belief sets = reachable closure from the initial belief under a reference
   screening propensity (0.12), rounded to 1e-4 and capped per key
   (600, or 1500 for the headline / observed-class policies);
2. one backward sweep of point-based backups (α-vectors are executable plans,
   so the value is a valid lower bound);
3. add the current policy's exact reachable set + ε-greedy rollouts
   (ε = 0.1) and re-sweep, stopping when the **exact in-model objective fails
   to improve by more than 1e-7 in two consecutive rounds** (best policy kept);
4. a fast-informed bound (FIB) gives an upper bound.

Convergence evidence is empirical, not certified: the objective is flat to
within 2e-6 after one or two rounds, cap 600 → 1500 changes mortality by
≤ 0.3 %, and rollout seed / reference propensity barely move the policy
(`dp/robustness.py`). The FIB gap stays large (44–57 % of the objective,
`results/dp/fib_gaps.md`), so optimality is not certified; see manuscript §2.5.

**Evaluation** (`dp/engine_runner.py`, `dp/evaluate.py`) — every arm runs on
the same 50 k-person chunk seeds (identical population and RNG stream until
the first diverging colonoscopy); paired chunk-level and person-level SEs.
Policy metrics in the model are computed exactly by forward propagation of
the belief tree (no Monte-Carlo noise).

## 3. Repository layout

```
pbvi_final/
├── dp/                        CURRENT pipeline (engine-grounded MOMDP)
│   ├── run_cohorts.py         simulate the two estimation cohorts in the engine
│   ├── estimate_kernels.py    all WAIT / SCREEN / exit kernels  -> results/dp/kernels_<tag>.npz
│   ├── kernels.py, model.py   observed memory, cell indexing, reduced MOMDP
│   ├── solver.py              PBVI + exact in-model evaluation + FIB bound
│   ├── sweep.py               lambda sweep -> in-model frontier, saved policies
│   ├── fixed_search.py        exhaustive search over 2 112 fixed schedules
│   ├── engine_runner.py       chunked, paired-seed, parallel engine runs (cached)
│   ├── hooks.py               engine hooks: fixed schedules, surveillance, belief policy
│   ├── evaluate.py, validate.py   engine evaluation; model-vs-engine validation
│   ├── run_pipeline.py        end-to-end driver (kernels -> ... -> report)
│   ├── report.py, figures.py, paired_tables.py   tables and figures for the paper
│   ├── ablate.py, _abl_policy.py                 model-structure ablation
│   ├── run_adherence.py                          imperfect-adherence scenarios
│   ├── riskscore.py, run_riskscore.py, score_frontier.py, score_fixed.py,
│   │   report_riskscore.py                       finite-discrimination risk score (§3.6)
│   ├── surveillance_arms.py                      fixed + CMOST surveillance comparators
│   ├── robustness.py, gap_table.py, kernel_support.py, tau_sensitivity.py
│   │                                             solver / bound / kernel-support diagnostics
│   └── manifest.py                               MD5 manifest of gitignored artefacts
├── cmost_engine/              Python CMOST engine (NumberCrunching_policy.py = instrumented)
├── results/dp/                kernels, sweeps, engine evaluations, reports (runs/ and policies/ gitignored)
├── paper/                     manuscript.md, dp_methods.md, dp_results.md, figures/,
│                              ksaim_poster/ (KoSAIM abstract), archive/ (superseded 6-state work)
├── docs/DP_PLAN.md            design, status, follow-ups and verification log
├── tests/test_engine_hook_regression.py      engine instrumentation regression test
└── env/ transitions/ pomdp/ experiments/ tests/   LEGACY pipeline (see §5)
```

## 4. Reproducing

Requirements: Python 3 with `numpy`, `scipy`, `matplotlib`. Engine runs are
heavy (≈ 2.8 ms/person, so 1 M persons ≈ 47 CPU-min), and every step is
cached and resumable under `results/dp/`.

```bash
# 1. estimation cohorts in the real engine
python -m dp.run_cohorts --nh-n 2000000 --screen-n 1000000 --workers 6

# 2. kernels, fixed-schedule search, lambda sweeps, engine evaluation, report
python -m dp.run_pipeline --tag c6b --cuts 0.5 0.8 0.95 0.965 0.98 --steps kernels,fixed,sweep,baseline,grid,headline,report --objectives death,inc

# 3. figures
python -m dp.figures --tag c6b
```

The paper's SCREEN kernel pools two randomised-schedule cohorts
(`screen_random_q`, `screen_random_q2`). Follow-up and verification analyses
(adherence, risk score, surveillance comparators, robustness, FIB gaps,
kernel support, τ sensitivity, paired tables, ablation) each have their own
entry point; the exact commands are listed in `docs/DP_PLAN.md`
("Verification-driven experiments") and in each module's docstring.

`results/dp/runs/` (per-chunk engine output) and `results/dp/policies/`
(α-vector sets) are excluded from git for size. Regenerated files can be
checked against the paper's runs with:

```bash
python -m dp.manifest --check
```

## 5. Legacy pipeline (superseded)

Before the `dp/` rewrite the project used a 9-state (and earlier 6-state)
natural-history Markov model estimated from a re-implemented individual
engine (`env/cmost_individual.py`, `transitions/`), a QALY-NMB POMDP
(`pomdp/model_v2.py`) solved with FiVI (Walraven & Spaan 2019;
`pomdp/fivi.py`), a Korean-epidemiology / Jeon et al. (2018) composite risk
score mapped onto CMOST's risk pool, and evaluation scripts in `tests/` and
`experiments/` (including the AUROC sweep, `experiments/auroc_sweep.py`).
It was replaced because the objective did not match the reported metrics,
observations were synthetic rather than the engine's real findings, the
re-implemented engine under-produced incidence, and FiVI's sawtooth upper
bound stayed frozen so convergence could not be verified
(`docs/DP_PLAN.md`, "Diagnosis of the existing pipeline"). The code is kept
for reference; the superseded manuscript and result write-ups are in
[`paper/archive/`](paper/archive/README.md).

## 6. Key references

* Prakash MK, et al. (2017) *PLoS ONE* — CMOST microsimulation.
* Zaika V, et al. (2024) — CMOST-based optimisation of fixed colonoscopy schedules.
* Ong SCW, et al. (2010) *Int J Robot Res* — mixed observability (MOMDP).
* Pineau J, Gordon G, Thrun S (2003) *IJCAI* — point-based value iteration.
* Walraven E, Spaan MTJ (2019) *JAIR* 65 — point-based value iteration for finite-horizon POMDPs.
* Hauskrecht M (2000) *JAIR* 13 — value-function approximations for POMDPs (fast-informed bound).
* Smallwood RD, Sondik EJ (1973) *Oper Res* — optimal control of POMDPs.
* Gupta S, et al. (2020) *Gastroenterology* — US MSTF post-polypectomy surveillance recommendations.

Full reference list: `paper/manuscript.md`.
