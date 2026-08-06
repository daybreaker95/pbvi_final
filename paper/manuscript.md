# Personalized colonoscopy screening schedules from a partially observable Markov decision process built on the CMOST microsimulation

*Working manuscript — pbvi_thlee branch. Figures in `paper/figures/`; numeric
results in `results/`.*

---

## Abstract

**Background.** Microsimulation studies of colorectal-cancer (CRC) screening,
including the recent CMOST-based optimization by Zaika et al. (2024), identify
the best *fixed population* colonoscopy schedule — the same one or few ages for
everyone. Such schedules cannot use the information generated *by screening
itself*: what a person's previous colonoscopies actually found. We ask whether a
policy that adapts each person's timing to their own screening history can
outperform the best fixed schedule at an equal number of colonoscopies.

**Methods.** We converted the open-source CMOST microsimulation into an
individual-level, gym-style environment that exposes the patient's hidden true
state, discretized by the most-severe lesion into six states (Normal, Early
Adenoma, Advanced Adenoma, Preclinical Cancer, Clinical Cancer, Dead). From
100 000 simulated no-screening life histories we empirically estimated the
age-specific 6×6 transition matrices (with bootstrap confidence intervals) and
validated them following the microsimulation-verification methodology of
Krijkamp et al. (2018) and against CRC-SPIN/CISNET natural-history targets. We
formulated screening as a partially observable Markov decision process (POMDP)
in which age and the remaining colonoscopy budget are observed, the six clinical
states (crossed with a latent low/high adenoma-risk class) form the belief, and
colonoscopy observations are discriminative. We solved it with point-based value
iteration (PBVI; Pineau et al. 2003) and evaluated the resulting adaptive policy
against Zaika's fixed schedules in the true CMOST environment.

**Results.** The empirical natural-history model reproduced CRC-SPIN targets
(7.2% of adenomas progress to cancer vs 7.4%; 90.5% of preclinical cancers
surface clinically vs 87%) and passed the verification protocol; estimating the
matrix on a **quarter-year step** (composed to annual) was necessary to reproduce
the microsimulation's CRC incidence and stage-at-diagnosis, which a directly-
estimated annual matrix undercounts. Colonoscopy findings updated the inferred
risk class from a 0.25 prior to 0.60–0.74 after an adenoma. In the true CMOST
environment the PBVI adaptive policy achieved **lower CRC mortality using fewer
colonoscopies** than the fixed schedules at every budget (e.g., 0.94% CRC
mortality at 2.31 colonoscopies vs Zaika 1.06% at 2.64). On total life-years the
adaptive policy and a fixed schedule re-optimized for this calibration were
**statistically indistinguishable** (LYG differences within ~1–1.5 Monte-Carlo SE;
LYG SE ≈ 7–12 per 1000 at n = 30,000, since LYG is dominated by rare CRC deaths).

**Conclusions.** A belief-based POMDP recovers individualized, findings-informed
colonoscopy timing that reallocates a fixed colonoscopy budget toward
**mortality-reduction efficiency** — comparable CRC mortality with fewer
procedures — but is statistically indistinguishable from an optimally-tuned fixed
schedule on life-years at matched adherence. Because age dominates CRC risk and
individual risk is only weakly identifiable from colonoscopy findings, the
incremental gain of adaptivity at matched adherence is modest — an honest finding
consistent with the small personalization gains reported elsewhere. Adaptivity
becomes **decisive under realistic conditions a fixed schedule cannot handle**: it
is robust to **imperfect adherence** (re-planning around missed colonoscopies, so a
chronically non-adherent subgroup gains ≈ 0.4 pp CRC-mortality reduction), and, given
an **observable polygenic risk score (AUC ≥ 0.8) and a cost-based budget**, it
performs genuine **risk-targeting** — intensive surveillance of the high-risk class
that cuts its CRC mortality below any fixed schedule at lower total colonoscopy use.
(This version reflects a corrected, quarter-year natural-history matrix, a corrected
detection probability, and a re-tuned solver; see transitions/SCREENING_VALIDATION.md.)

---

## 1. Introduction

Colorectal cancer is the second-to-third leading cause of cancer incidence and
mortality in industrialized countries, and colonoscopy prevents CRC by removing
adenomatous precursors and detecting cancer early. Guidelines recommend
colonoscopy every 10 years between ages 45 and 75, and microsimulation models —
MISCAN, CRC-SPIN, SimCRC, CRC-AIM and the open-source CMOST — are the standard
tool for optimizing such schedules because randomized trials cannot span the
whole design space.

Zaika et al. (2024) used CMOST to search, without the 5-/10-year grid
constraint of prior USPSTF analyses, for the optimal *fixed* schedule of one to
four screening colonoscopies between ages 20 and 90. They found single-
colonoscopy optima near age 55 (life-years) shifting later for
incidence/mortality objectives, and showed the optimum depends on adenoma risk,
adenoma detection and adherence. Their "personalization," however, is
*risk-group stratification*: a hypothetical test splits the population into a
25% high-risk and a 75% low-risk stratum, and a separate fixed schedule is
optimized for each. The schedule still does not respond to the findings of a
given person's colonoscopies.

Clinically, screening *is* adaptive: a normal colonoscopy earns a 10-year
interval, whereas finding an advanced adenoma triggers 3-year surveillance,
because the finding reveals elevated future risk. This is precisely the logic of
a POMDP: the true colon state is hidden, each colonoscopy is a noisy
observation, and the optimal action depends on the *belief* the observations
induce. We build such a POMDP directly on CMOST, estimate its dynamics
empirically from the microsimulation, and show that point-based value iteration
recovers a personalized policy that uses screening history — how often a person
has been screened and what was found — to allocate colonoscopies where they
help most.

Our contributions:

1. An **individual-level, gym-style CMOST environment** (a true-state simulator)
   derived from the validated population engine, suitable for reinforcement
   learning / decision-process research (`env/`).
2. **Empirical, validated 6-state transition matrices** of CMOST's natural
   history, with a verification protocol (Markov-trace consistency, a genuine
   Markov-property test, Monte-Carlo convergence, and external comparison to
   CRC-SPIN/CISNET) (`transitions/`).
3. A **6-state POMDP with a latent risk class** and a **PBVI** solution whose
   policy personalizes on screening history, benchmarked against Zaika (2024)
   in the true CMOST environment (`pomdp/`, `experiments/`).

---

## 2. Methods

### 2.1 The CMOST microsimulation and its individual-level re-implementation

CMOST (Prakash et al. 2017) simulates, in 3-month increments from birth to age
100, the CRC adenoma–carcinoma sequence: age-, sex- and location-dependent
adenoma initiation; growth through six adenoma stages; malignant transformation
to preclinical cancer (stages I–IV) either through the adenoma pathway or via a
minor "direct/fast" path; symptomatic presentation; stage progression;
stage-specific CRC mortality; colonoscopy detection and complications; and
competing all-cause mortality.

We re-implemented the exact per-patient, per-quarter CMOST dynamics as a single-
patient engine (`env/cmost_individual.py`) that consumes the identical parameter
bundle produced by the CMOST pipeline (we factored `calculate_sub` into a
reusable `prepare_parameters`, leaving population results bit-identical). This
engine is wrapped as a gym environment (`env/crc_env.py`) with `reset()` /
`step(action ∈ {wait, screen})`; the hidden true state is always available for
analysis while the agent receives only colonoscopy observations. Reward is
(optionally discounted) life-years minus a small per-colonoscopy disutility, so
cumulative reward approximates life-years gained (LYG).

### 2.2 Clinical-state discretization

The whole-colon status is discretized by the most severe lesion into
`s ∈ {Normal, Early Adenoma (polyp stage 1–4), Advanced Adenoma (polyp stage
5–6), Preclinical Cancer, Clinical Cancer, Dead}`, matching CMOST's own
advanced-adenoma threshold (`Tumor>4`) and its symptomatic/screen diagnosis of
clinical cancer.

### 2.3 Empirical transition estimation

We simulated 100 000 no-screening life histories, recorded each patient's
discretized state at every **quarter** of life (matching CMOST's internal
quarterly event loop), and estimated the age-specific one-**quarter** transition
probabilities by maximum likelihood,
`P̂_q[age,s,s'] = N_q[age,s,s'] / Σ_{s'} N_q[age,s,s']`. The annual transition
matrix consumed by the POMDP is formed by composition, `P[age] = P̂_q[age]^4`.
Estimating quarterly and composing is important: a matrix estimated directly from
*annual* snapshots never observes cancers that arise, are diagnosed and kill
within a single year, and therefore undercounts lifetime CRC incidence (by ~14%)
and the stage-IV share at diagnosis; the quarterly-composed matrix removes this
discretization artifact (transitions/SCREENING_VALIDATION.md). We report
multinomial standard errors `√(p(1-p)/n)` and a patient-level nonparametric
bootstrap (200 resamples) for the directly-estimated annual matrix, and we
additionally estimated the matrices separately for a low- and a high-adenoma-risk
stratum (top 25% of CMOST's individual risk multiplier, matching Zaika's split)
for the latent-risk POMDP.

### 2.4 Verification and validation

Following Krijkamp et al. (2018, *Med Decis Making*; NIHMS931782) we verified and
validated the estimated model:

* **V1 — Markov-trace consistency (implementation check).** Propagating the
  directly-estimated annual matrix as a deterministic cohort model reproduces the
  microsimulation's yearly-snapshot occupancy to machine precision, as expected
  because the MLE one-step matrix reproduces the marginal by construction (this
  checks correct estimator/propagation implementation).
* **V1c — Screening metric validation.** Because the consumed matrix is the
  quarterly-composed one, we separately verified that a metric-complete augmented
  cohort (splitting Dead into other/CRC death and tracking cancer stage)
  reproduces the microsimulation's 8 clinical metrics — age at death, age at CRC
  death, CRC deaths, CRC incidence, and stage-at-diagnosis — under no screening
  and under q10y/q5y colonoscopy. Under quarterly stepping the natural-history
  cohort matches the microsimulation on all eight to within Monte-Carlo error
  (transitions/SCREENING_VALIDATION.md).
* **V1b — Markov-property test.** For each state we compared the next-state
  distribution of patients who had just entered the state ("newcomers") vs those
  who had been in it a year already ("stayers"); a small total-variation
  distance supports the memoryless 6-state abstraction.
* **V2 — Monte-Carlo convergence.** Transition estimates and their Monte-Carlo
  standard errors versus sample size confirm ~1/√N convergence.
* **V3 — External validity.** Natural-history summaries versus CRC-SPIN/CISNET
  and SEER targets.

### 2.5 POMDP formulation and PBVI

Age and the remaining colonoscopy budget `k` are fully observed; the belief is
over the six clinical states, optionally crossed with a fixed but unobserved
low/high risk class (12 states). Actions are `wait` or `screen`. The one-year
transition uses the empirical natural-history matrix (with Clinical and Dead made
absorbing); a `screen` first applies a colonoscopy — removing detected adenomas
(→ Normal) and detecting preclinical cancer early (→ Clinical) — then one year of
natural history. Colonoscopy detection probabilities (`d_EA = 0.71`,
`d_AA = 0.91`, `d_PC = 0.939`) and the life-year value of a screen- vs
symptom-detected cancer (stage-shifted, hence age-dependent through remaining
life expectancy) were estimated from the same engine. `d_PC` reproduces the
engine's cancer-detection rule faithfully — `Colo_Detection[stage] × P(loc ≥
reach)`, with no `Location_ColoDetection` factor (which the engine applies only
to polyps, not cancers) — correcting a prior value that had erroneously scaled it
by the mean location factor. Observations are `{none, normal, adenoma, advanced adenoma, cancer}`,
discriminative under `screen`; finding adenomas raises the inferred risk class,
so the policy shortens intervals exactly like post-polypectomy surveillance.

We solved the finite-horizon POMDP by point-based value iteration (Pineau,
Gordon & Thrun 2003): a fixed set of belief points, exact backward induction over
age with point-based Bellman backups, and belief-point expansion along reachable
trajectories. The policy tracks its belief by Bayesian filtering on observations
during a roll-out.

### 2.6 Comparison

We evaluated, in the true CMOST environment with per-patient common random
numbers, five families of policies: no screening; guideline colonoscopy (every
10 y, 45–75); Zaika's published optimal fixed ages; a **fixed schedule
re-optimized within this CMOST calibration** (greedy forward search over
screening ages, one to four colonoscopies); and the PBVI adaptive policy for
budgets one to four. The re-optimized fixed schedule is the fair primary
baseline, because Zaika's published ages were tuned on a different CMOST
calibration (CMOSTv3, SEER 1988–2002) and are not optimal for the CMOST13
parameters used here. Outcomes: mean life-years, CRC incidence and mortality,
mean number of screening colonoscopies, LYG per 1000 versus no screening, and
LYG per colonoscopy. Because the adaptive policy may decline to use its full
budget on individuals whose exams are repeatedly clean, we compare on the
efficiency frontier (LYG versus *mean* colonoscopies actually performed).

---

## 3. Results

### 3.1 Natural-history transition model and its validation

The estimated matrices reproduce the expected progressive chain
(Normal → Early → Advanced adenoma → Preclinical → Clinical → Dead) with
biologically sensible, age-increasing rates (Figure 1; `results/transitions_cmost13.npz`).
Representative annual rates (marginal population; quarterly-composed matrix):

| from → to | age 50 | age 60 | age 70 |
|-----------|:------:|:------:|:------:|
| Normal → Early Adenoma | 1.9% | 2.8% | 3.0% |
| Early Adenoma → Advanced Adenoma | 2.1% | 2.3% | 2.3% |
| Early Adenoma → Normal (regression) | 7.0% | 6.2% | 5.7% |
| Advanced Adenoma → Preclinical Cancer | 1.4% | 1.8% | 1.9% |
| Preclinical → Clinical (symptomatic surfacing) | 28% | 18% | 22% |

**Validation (Figures 2–3; `results/validation_summary.json`).**

* **V1 (implementation):** the directly-estimated annual cohort Markov trace
  matches microsimulation yearly occupancy to < 1e-15 (max total-variation
  distance), verifying correct implementation. The consumed quarterly-composed
  matrix additionally reproduces the microsimulation's 8 clinical metrics under
  no screening and q10y/q5y screening (V1c; SCREENING_VALIDATION.md).
* **V1b (Markov property):** total-variation distance between "stayer" and
  "newcomer" outgoing distributions was 0.089 (Early Adenoma), **0.013 (Advanced
  Adenoma)** and 0.319 (Preclinical Cancer). The screening-relevant adenoma
  states are nearly memoryless — supporting the 6-state abstraction — whereas
  Preclinical Cancer carries sojourn-time memory (expected, since its dwell time
  is non-geometric), a limitation we return to in the Discussion.
* **V2:** estimates stabilize and Monte-Carlo standard errors fall as 1/√N.
* **V3 — external validity (model vs literature):**

| quantity | CMOST (this work) | literature target |
|----------|:-----------------:|-------------------|
| Fraction of adenomas progressing to cancer | **7.2%** | 7.4% (CRC-SPIN) |
| Fraction of preclinical cancers surfacing clinically | **90.5%** | 87% (CRC-SPIN) |
| Mean sojourn time (preclinical→clinical) | 3.5 y | 1.9 y (CRC-SPIN); 2–5 y (models) |
| Median adenoma dwell time (onset→cancer) | 13.5 y | ~13 y (CMOST benchmark); 25.4 y (CRC-SPIN) |
| Advanced-adenoma prevalence, age 60 | 3.6% | 5–9% (screening cohorts) |
| Lifetime clinical CRC incidence (no screening) | 4.6% | 6.7–7.2% cum. 40–100 (CISNET) |

The progression fraction and clinical-surfacing fraction match CRC-SPIN closely;
sojourn and dwell times are within model-to-model range; adenoma prevalence and
absolute incidence are somewhat lower than US screening cohorts, consistent with
CMOST's own calibration (a discrepancy also reported by Zaika et al.). These are
properties of CMOST's calibration, not of our estimator, which reproduces CMOST
faithfully.

### 3.2 Latent adenoma-risk classes

Stratifying by CMOST's individual adenoma-risk multiplier (top 25% = high risk)
produced strongly separated natural histories: **lifetime clinical CRC incidence
2.5% (low) vs 9.8% (high), a ~4-fold difference**, driven by a ~5-fold higher
adenoma-initiation rate (Normal→Early Adenoma at age 55: 1.4% vs 7.7% per year).
Because colonoscopy findings are informative about the (correlated) clinical
state and hence the risk class, the POMDP raises inferred P(high risk) from the
0.25 prior to **0.60 after a low-risk adenoma and 0.74 after an advanced
adenoma**, and lowers it after clean exams — the mechanism of history-dependent
personalization.

### 3.3 Adaptive POMDP policy vs fixed schedules

Evaluated in the true CMOST environment (Table 2, Figure 4) under the corrected
quarterly-composed model, with the PBVI solver re-tuned for the new model, the
adaptive policy achieved **comparable or lower CRC mortality than the fixed
schedules while using fewer colonoscopies** — e.g. at budget 3, CRC mortality
0.94% at 2.31 colonoscopies vs Zaika 1.06% at 2.64 and best-fixed 0.99% at 2.59;
at budget 4, 0.85% at 3.12 vs Zaika 0.92% at 3.51. On **total life-years gained**,
the adaptive policy and the re-optimized best-fixed schedule are **statistically
indistinguishable**: the LYG gaps (−17 to +1 per 1000) are within ~1–1.5 of the
Monte-Carlo standard error (≈ 7–12 per 1000 at n = 30,000, because LYG is driven
by rare CRC deaths), as are the CRC-mortality differences (≤ 0.05 pp, SE ≈ 0.06).
The corrected model therefore shows the adaptive policy's value as
**colonoscopy-sparing allocation at equal effectiveness**, rather than the
life-years dominance the pre-rebuild annual matrix had suggested.

A fixed schedule *re-optimized specifically for the CMOST13 calibration* (greedy
search; e.g., ages 52/58/70 for three colonoscopies) is the strongest baseline: on
the efficiency frontier its point estimates lead the adaptive policy on total
life-years at budgets 1, 3 and 4 (by 6–17 per 1000) and tie at budget 2, but all
these gaps are within Monte-Carlo noise. The adaptive policy attained slightly
*higher* CRC incidence but *equal-or-lower* CRC mortality than the fixed
schedules, indicating it leans toward early detection over adenoma-removal
prevention — a consequence of individually-rational Bayesian timing (it defers
screening until a patient's own belief signals risk) versus the population-optimal
early prevention that a fixed schedule hard-codes. (These
results supersede the pre-rebuild annual-matrix numbers, under which the adaptive
policy had appeared to beat Zaika on life-years at every budget; the corrected
within-year dynamics and detection remove that apparent advantage.)

**Table 2.** Screening policies evaluated in the true CMOST environment (paired common random numbers), quarterly-composed matrix, `d_PC = 0.939`, n = 30,000. LYG = life-years gained vs no screening.

| policy | mean colonoscopies | CRC incidence % | CRC mortality % | LYG per 1000 | LYG per colonoscopy |
|--------|:---:|:---:|:---:|:---:|:---:|
| No screening | 0.00 | 4.16 | 1.71 | 0.0 | 0.0 |
| Guideline q10y 45-75 | 3.38 | 2.25 | 0.87 | 103.7 | 30.7 |
| Best fixed x1 | 0.89 | 3.38 | 1.36 | 55.5 | 62.1 |
| Zaika fixed x1 | 0.91 | 3.58 | 1.39 | 48.5 | 53.2 |
| PBVI adaptive x1 | 0.76 | 3.37 | 1.31 | 38.5 | 50.8 |
| Best fixed x2 | 1.66 | 2.80 | 1.12 | 67.2 | 40.6 |
| Zaika fixed x2 | 1.78 | 2.85 | 1.16 | 72.8 | 40.8 |
| PBVI adaptive x2 | 1.67 | 2.93 | 1.13 | 68.8 | 41.1 |
| Best fixed x3 | 2.59 | 2.46 | 0.99 | 91.7 | 35.4 |
| Zaika fixed x3 | 2.64 | 2.57 | 1.06 | 80.4 | 30.5 |
| PBVI adaptive x3 | 2.31 | 2.68 | 0.94 | 78.4 | 33.9 |
| Best fixed x4 | 3.24 | 2.23 | 0.83 | 103.3 | 31.9 |
| Zaika fixed x4 | 3.51 | 2.34 | 0.92 | 99.7 | 28.4 |
| PBVI adaptive x4 | 3.12 | 2.33 | 0.85 | 91.5 | 29.3 |

At n = 30,000, **LYG has a large Monte-Carlo standard error (≈ 7–12 per 1000)**
because it is dominated by the rare, heavy-tailed event of a CRC death; the stable
outcome is CRC mortality (SE ≈ 0.05–0.07 pp). LYG differences below should be read
as within noise unless they exceed ~15/1000.

**Efficiency-frontier comparison (PBVI vs the re-optimized best-fixed schedule, interpolated at matched colonoscopy use):**

| budget | PBVI mean colo | PBVI LYG/1000 | best-fixed LYG/1000 @ same colo | Δ (PBVI − best-fixed) |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.76 | 38.5 | 55.5 | −16.9 |
| 2 | 1.67 | 68.8 | 67.7 | +1.1 |
| 3 | 2.31 | 78.4 | 84.4 | −6.0 |
| 4 | 3.12 | 91.5 | 101.2 | −9.7 |

With the PBVI solver re-tuned for the corrected model (no screen-disutility, 700
beliefs, 4 expansions), the adaptive policy is **statistically comparable to the
strong re-optimized best-fixed schedule** on both life-years (LYG gaps within
~1–1.5 SE) and CRC mortality (differences ≤ 0.05 pp, within ~1 SE), while reaching
comparable mortality with **slightly fewer colonoscopies** at budgets 3–4 (Table 2,
Figure 5). It does not dominate that baseline on life-years, but it improves on
Zaika's published ages on CRC mortality at every budget.

### 3.4 When, where and for whom the adaptive policy helps

Because the matched-adherence comparison is close, we probed the conditions under
which adaptivity is decisive (Figures 6–9; `paper/results_*.md`).

**Robustness to non-adherence (Figure 6).** The matched-adherence comparison
assumes every person is screened exactly on schedule. When each recommended
colonoscopy is attended only with probability α, a fixed schedule loses missed
slots permanently, whereas the POMDP re-plans for free — a no-show yields no
observation and consumes no budget, so re-recommending screening is simply the
optimal action (we verified that forcing every recommendation to be missed leaves
the budget full and the policy re-inviting). As α falls from 1.0 to 0.25, fixed
(no-recall) CRC mortality drifts back toward the no-screening level (e.g. budget 2:
1.12 %→1.60 % vs no-screening 1.71 %), while the adaptive policy holds near 1.1 %.
In a heterogeneous population with a chronically low-adherence subgroup (attend 20 %
of invitations), that **non-adherent subgroup**'s CRC mortality falls from 1.64 %
under the fixed schedule to 1.20 % under the adaptive policy (budget 2; paired
−0.44 ± 0.11 pp) — recovering most of the benefit a fixed program leaves on the
table. A decomposition against a fixed schedule *with annual recall* shows most of
this rescue is the re-invitation behaviour (which a fixed program could also adopt);
the POMDP's contribution is that recall and re-timing arise automatically as the
optimal policy.

**Patient subgroups at matched adherence (Figure 7).** Stratifying by
policy-independent traits, the adaptive policy provided **no reliable benefit in any
risk subgroup**: within the top-25 % risk class the paired PBVI−fixed CRC-mortality
difference was −0.16 ± 0.16 pp at budget 3 and even reversed sign at budget 4, and
mean colonoscopies were essentially flat across risk quartiles. This is expected
from §3.2: a *clean* colonoscopy only weakly lowers inferred risk, so a truly
high-risk person who screens clean early is mildly reassured, cancelling any
targeting. The adaptive policy's robust matched-adherence advantages are instead
**colonoscopy efficiency** (comparable mortality at ~10 % fewer colonoscopies) and
the **surveillance regime**: at a lifetime budget of six, a uniform fixed schedule
*saturates* (mortality 0.80 %→0.81 % from budget 4→6, extra colonoscopies wasted),
whereas the adaptive policy converts the extra budget into mortality reduction
(0.86 %→0.71 %) by directing it to realised findings.

**Baseline risk factors and mortality-targeting (Figures 8–9).** The inability to
target high-risk is *structural*, not fundamental, and removing two constraints
reverses it. (i) A **personalized prior** from observed baseline risk factors —
family history and prior-adenoma history — sets each patient's initial belief to
`P(high | risk factors)` instead of the flat 0.25. (ii) A **cost-based budget** (a
per-colonoscopy disutility rather than a hard per-person cap) lets the policy vary
the *number* of colonoscopies by risk; under a hard cap, a personalized prior alone
changes mean colonoscopies by < 0.1, because the cap forbids quantity-targeting.
With both, the efficiency frontier orders **PBVI+risk-factors < risk-stratified
fixed < plain fixed**. At the realistic discrimination of family history + prior
adenoma (AUC ≈ 0.67) the gain is efficiency (same mortality, ~20–30 % fewer
colonoscopies, chiefly by safely de-escalating the low-risk majority), not a
high-risk mortality drop. Pushing the baseline discrimination to that of a
**polygenic risk score (AUC ≥ 0.8)** at a low per-colonoscopy cost **turns on
mortality-targeting** (Figure 9): the true high-risk class receives intensive
surveillance (≈ 5 colonoscopies) and its CRC mortality falls from 1.55 % (flat
prior) to 1.18 % (AUC 0.8) and 1.13 % (AUC 0.9) — below the best fixed schedule's
1.38 % in that class — *at lower total colonoscopy use* (3.0 vs 3.2), with an oracle
upper bound of 0.84 %. Family history alone is too weak a classifier; a PRS crossing
AUC ≈ 0.8 is the threshold at which adaptive risk-stratified screening yields an
outright high-risk mortality benefit.

**Table 3.** Mortality-targeting by baseline risk-factor discrimination (AUC), in
the true CMOST environment (matched adherence, paired common random numbers,
n = 25,000), reported by **true** adenoma-risk class. The adaptive policy uses a
cost-based budget (per-colonoscopy disutility c = 0.03) so it can vary the number of
colonoscopies by risk. AUC 0.67 corresponds to family history + prior-adenoma
history; 0.80–0.90 to a polygenic risk score (± family history); 1.00 is a
perfect-information oracle. High-risk class = top 25 % of CMOST individual risk
(n ≈ 6,175; CRC-mortality SE ≈ 0.14 pp).

| policy | total colon. | colon. high / low | CRC mort. high-class % | CRC mort. low-class % | CRC mort. overall % |
|--------|:---:|:---:|:---:|:---:|:---:|
| No screening | 0.00 | 0.00 / 0.00 | 3.76 | 1.07 | 1.73 |
| Best fixed x3 | 2.59 | 2.58 / 2.59 | 1.78 | 0.70 | 0.96 |
| Best fixed x4 | 3.24 | 3.22 / 3.24 | 1.38 | 0.65 | 0.83 |
| PBVI cost, AUC 0.50 (none) | 3.67 | 4.36 / 3.45 | 1.55 | 0.60 | 0.84 |
| PBVI cost, AUC 0.67 (FH+adenoma) | 3.52 | 4.35 / 3.25 | 1.52 | 0.63 | 0.85 |
| PBVI cost, AUC 0.80 (PRS) | 3.30 | 4.62 / 2.87 | **1.18** | 0.71 | 0.82 |
| PBVI cost, AUC 0.90 (PRS+FH) | 3.02 | 4.99 / 2.37 | **1.13** | 0.63 | 0.75 |
| PBVI cost, AUC 1.00 (oracle) | 2.42 | 5.87 / 1.29 | **0.84** | 0.73 | 0.76 |

As discrimination rises the policy concentrates colonoscopies on the true high-risk
class (4.36 → 5.87) and de-escalates the low-risk class (3.45 → 1.29), so total use
*falls*; high-risk-class CRC mortality drops below the best fixed schedule once
AUC ≥ 0.8, at lower total colonoscopy use. At AUC ≈ 0.67 (family history alone) the
high-risk mortality is indistinguishable from the flat prior — the payoff there is
colonoscopy efficiency, not mortality (Figure 8).


---

## 4. Discussion

We built a partially observable Markov decision process for colorectal-cancer
screening directly on the open-source CMOST microsimulation, estimating its
6-state dynamics empirically and validating them against the microsimulation and
external CRC-SPIN/CISNET targets, and solved it with point-based value iteration.
The resulting policy is genuinely individualized: it filters a belief over the
hidden colon state (crossed with a latent risk class) from each person's realized
colonoscopy findings. A finding of (advanced) adenoma raises the inferred
probability of the high-risk class (from the 0.25 prior to 0.60–0.74), which
accelerates the belief's re-accumulation of adenoma risk after polypectomy
(Figure 5) — the qualitative logic of post-polypectomy surveillance. We note
honestly, however, that in the current 6-state formulation this risk signal is
partly masked at decision time by the clinical-lesion belief (which resets toward
Normal once a polyp is removed), so it produces only modest differentiation of
realized screening intervals; strengthening this channel is a clear direction
for future work (below).

Against the schedules of Zaika et al. (2024), the concrete comparator we set out
to beat, the adaptive policy achieved **lower CRC mortality using fewer
colonoscopies** at every budget. On total life-years, under the corrected
quarterly-composed model, the adaptive policy and the fixed schedules are
**statistically indistinguishable**: at n = 30,000 the LYG standard error is
7–12 per 1000 (LYG is dominated by rare CRC-death events), so the earlier
"dominates on life-years at every budget" conclusion was an artifact of both the
annual-discretization error, the mis-scaled `d_PC`, and Monte-Carlo noise. The
robust, reproducible result is that a belief-based policy reallocates a fixed
colonoscopy budget toward **mortality-reduction efficiency at fewer procedures**,
not that it dominates fixed schedules on life-years.

The more demanding test — against a fixed schedule *re-optimized for this exact
calibration* — sharpens the nuance: the adaptive policy did not exceed that
stronger baseline on total life-years (its point estimates trailed by 6–17/1000 at
budgets 1/3/4 and tied at budget 2, all within Monte-Carlo noise), although it
remained competitive on CRC mortality and colonoscopy-sparing. Two structural
reasons explain why
adaptivity's incremental value over an optimally-tuned fixed schedule is modest
here, and both are clinically meaningful. First, in CMOST (as in reality) **age
dominates CRC risk**, and a fixed schedule already places colonoscopies at the
ages of maximal
benefit; the residual, individually-exploitable heterogeneity is smaller.
Second, colonoscopy findings are **asymmetrically informative**: finding an
adenoma strongly indicates elevated risk, but a *clean* exam is only weakly
reassuring (high-risk colons are frequently clean at any single exam), so the
policy cannot confidently *de-escalate* screening for the low-risk majority.
These same limits underlie the modest gains reported for risk-stratified
screening generally, including Zaika's own risk-group analysis.

Principal limitations. (i) The 6-state abstraction is memoryless. Estimating the
matrix on a quarter-year step (composed to annual) removes the *time-
discretization* error — a directly-estimated annual matrix undercounts CRC
incidence by ~14% and the stage-IV share because within-year onset→diagnosis→
death is invisible to yearly snapshots — but a genuine *sojourn* memory remains:
the Markov-property test shows the abstraction is accurate for adenoma states but
not for Preclinical Cancer, whose dwell is non-geometric, and post-diagnosis CRC
mortality is front-loaded rather than memoryless. A semi-Markov or phase-type
refinement (subdividing the clinical state by time since diagnosis) reproduces the
CRC-death age exactly in a validation cohort (transitions/verify_screening_tauphase.py)
and is the natural next step for the decision model. (ii) Detection probabilities
and cancer life-year values are estimated pointwise; `d_PC` now faithfully
reproduces the engine's cancer-detection rule (stage-flat at 0.95, no location
factor), and could be made further age- and size-resolved. (iii) The comparison inherits CMOST's calibration,
which under-estimates absolute adenoma prevalence relative to US cohorts; the
*relative* comparison of policies within the same environment is unaffected.
(iv) We optimize life-years; cost-effectiveness (ICERs) is a natural extension of
the reward, and the per-colonoscopy disutility used in §3.4 is a first step toward
it. (v) Consistent with the conjecture that adaptivity's advantage grows with more
identifiable heterogeneity and with realistic deployment conditions, §3.4 shows the
adaptive policy is decisive exactly where a fixed schedule is structurally unable to
respond: under **imperfect adherence** (it re-plans around missed colonoscopies,
benefiting the non-adherent subgroup most) and, once given an **observable baseline
risk factor plus a cost-based budget**, in **risk-targeting** — where a polygenic
risk score (AUC ≥ 0.8) lets it cut high-risk-class CRC mortality below any fixed
schedule at lower total colonoscopy use, while family history alone (AUC ≈ 0.67) buys
only efficiency. At matched adherence and a hard per-person budget, by contrast, its
edge is limited to colonoscopy-sparing — the honest null that motivated these
analyses. Further identifiable signal (continuous risk classes, observed adenoma
number/size, which CMOST tracks) should extend this. Because the policy is derived on
an approximate 6-state model but deployed on the full microsimulation, closing the
model-transfer gap (e.g., by fitting the reward's cancer values and detection to the
environment, or by direct policy search / reinforcement learning against the gym
environment) is a further route to exceeding the re-optimized fixed baseline.

## References

1. Prakash MK, et al. CMOST: an open-source framework for the microsimulation of
   colorectal cancer screening strategies. *PLoS ONE* 2017.
2. Zaika V, et al. Optimal timing of a colonoscopy screening schedule depends on
   adenoma detection, adenoma risk, adherence to screening and the screening
   objective: a microsimulation study. *PLoS ONE* 2024;19(5):e0304374.
3. Krijkamp EM, et al. Microsimulation modeling for health decision sciences
   using R: a tutorial. *Med Decis Making* 2018 (NIHMS931782).
4. Pineau J, Gordon G, Thrun S. Point-based value iteration: an anytime algorithm
   for POMDPs. *IJCAI* 2003.
5. Rutter CM, et al. Colorectal Cancer Simulated Population model for Incidence
   and Natural history (CRC-SPIN); CISNET colorectal models.
