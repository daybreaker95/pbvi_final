# Methods (new DP pipeline, `dp/`) — draft text for the manuscript

## Decision model: a mixed-observability Markov decision process built on CMOST

We approximate the CMOST microsimulation by a finite-horizon, mixed-
observability Markov decision process (MOMDP; Ong et al. 2010). Decisions are
made once a year at ages 40–80 (WAIT or SCREEN = colonoscopy). The state has
an *observed* component — age, sex, the number of years since the last
colonoscopy (τ; "never" if none) and the finding at the last colonoscopy
(normal / 1–2 early adenomas / ≥3 early adenomas / advanced adenoma) — and a
*hidden* component over which a belief is maintained: an individual
adenoma-risk class (6 classes = cuts of CMOST's `individual_risk`
distribution at its 50th, 80th, 95th, 96.5th and 98th percentiles; the three
top classes isolate CMOST's synthetic hereditary-scale tail, whose relative
polyp rate ranges from ~6x to ~54x) crossed with 11 clinical states (no
lesion; most advanced polyp at CMOST stage 1–6; undetected cancer stage
I–IV). Diagnosis of CRC (symptomatic or at colonoscopy), death from other
causes and death from a colonoscopy complication are *exits* from the
decision process; their remaining-lifetime consequences (probability of CRC
death, remaining life-years) are folded into the exit reward, so the belief
only ranges over the 66 alive-and-undiagnosed states. The number of classes
was chosen by a group-wise transfer diagnostic: with three classes the model
matched the class-posterior of every finding exactly but under-predicted the
subsequent cancer incidence of persons with ≥3 adenomas ~2.5-fold, because
those persons concentrate in the extreme within-class tail; six classes
remove most of that error and eight add nothing.

The observed memory (τ, last finding) is not a modelling convenience but a
necessity: in CMOST every polyp carries its own growth speed and every
person a lifelong risk multiplier, so the post-polypectomy natural history
of a screened person differs from that of the never-screened cross-section
(new polyps arise faster and progress faster, because the unscreened
cross-section is enriched in slow-growing lesions). Conditioning the
transition kernels on (τ, last finding) — both known to any screening
programme — captures this memory; without it a Markov model either under-
or over-predicted the engine's benefit of a fixed 10-year schedule by
10–25 %.

## Kernel estimation from the real engine

All kernels were estimated by maximum likelihood directly from the real
CMOST engine (the MATLAB-ported `NumberCrunching` implementation used for
all evaluations), instrumented with a quarter-resolved state recorder:

* **Natural-history (WAIT) kernel.** Annual transition probabilities from
  the decision-time state at age y to the decision-time state at y+1, with
  exits (diagnosis at stage I–IV, other death) read from the intermediate
  quarterly snapshots so that within-year onset → diagnosis → death is never
  missed, stratified by sex × risk class × memory cell (τ group × last
  finding) × age. Two cohorts were pooled: 2 000 000 never-screened persons
  and 1 000 000 persons screened on randomised schedules (first colonoscopy
  at a uniformly random age 40–75, subsequent intervals uniform on 1–20
  years; 2 x 1 000 000 chunks), the latter supplying the post-colonoscopy
  cells. Sparse cells back
  off hierarchically (wider age window → pooled last finding → pooled τ →
  pooled class → pooled sex; minimum 150 person-years).
* **Colonoscopy (SCREEN) kernel.** From the randomised-schedule cohort's
  decision log (≈3.5 million colonoscopies with the engine's true
  pre-/post-procedure states), the joint probability of the observation
  (normal / 1–2 adenomas / ≥3 adenomas / advanced adenoma / cancer found /
  complication death) and of the post-polypectomy clinical state, given the
  pre-procedure state, sex, class, memory cell and 5-year age band. This
  kernel carries the engine's per-lesion detection, residual (missed) polyps
  and reach.
* **Exit values.** For each sex, diagnosis age and stage: the probability
  of eventual CRC death and the expected remaining life-years, from the
  same cohorts.
* **Initial belief.** The empirical joint distribution of (class, clinical
  state) at the age-40 decision among persons alive and undiagnosed.

Validation: propagating fixed schedules through the model reproduces the
engine's decision-time prevalence of early adenoma, advanced adenoma and
undetected cancer at every age under no screening and under 10-year
screening (Figure dp_validation), and the engine's CRC mortality and
incidence under no screening / 10-year / 5-year schedules to within a few
percent.

## Objective and solution

The objective is the expected number of CRC deaths (alternatively: CRC
diagnoses, or life-years) minus λ times the number of colonoscopies,
undiscounted, over ages 40–100. λ is the shadow price of a colonoscopy; the
family of optimal policies over λ traces the efficiency frontier between
outcome and colonoscopy volume, and comparisons with fixed schedules are
made at matched volume. Because life-years, incidence and complication
deaths are all linear in the state occupancy, every metric of every policy
is computed *exactly* in the model by forward propagation of the policy's
belief tree (no Monte-Carlo noise), which also makes exhaustive in-model
search over fixed schedules (2 112 candidates) cheap.

The MOMDP was solved by finite-horizon point-based value iteration: belief
sets indexed by the observed key (age, τ, last finding) were generated as
the closure of beliefs reachable from the initial belief (rounded-merge,
capped per key by reach probability), a single backward sweep of point-based
Bellman backups produced one α-vector per belief point and key, the
policy's own exact reachable beliefs plus ε-greedy rollouts were added and
the sweep repeated until the policy's exact in-model objective stopped
improving. A fast-informed bound gives an upper bound. The resulting policy
maps (age, τ, last finding, belief) to WAIT/SCREEN.

## Evaluation in the engine

Every policy was deployed inside the real engine through a hook that
receives the engine's true colonoscopy result (polyps removed by stage
group, cancer detected, complication death) and a symptomatic-diagnosis
flag, updates the belief with the model's kernels, and never re-screens a
diagnosed patient. Fixed comparators (10-year 50/60/70; 5-year 50–75; and
the best in-model fixed schedules at matched volume) use the same hook
mechanism and the same no-re-screening rule. All arms are simulated on the
same chunk seeds (identical populations and random-number streams until
the first colonoscopy), 1 000 000 persons per headline arm, 200 000 per
λ-grid point; standard errors are computed across 50 000-person chunks.
Outcomes: CRC deaths and CRC diagnoses per 100 000 (ages ≥ 40), policy
colonoscopies per person, life-years from 40, complication deaths; the
primary efficiency measure is the reduction versus no screening per 1000
colonoscopies.
