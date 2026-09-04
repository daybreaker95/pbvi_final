# Adaptive colonoscopy screening from a mixed-observability Markov decision process built on the CMOST microsimulation: more cancers and deaths prevented per colonoscopy than any fixed schedule

*Working manuscript. Figures in `paper/figures/dp_*.png`; numeric results in
`results/dp/` (`report_c6b.md`, `eval_headline_c6b_n1000000.json`,
`eval_adherence_c6b_n200000.json`, and the verification-driven artefacts
`paired_tables.md`, `eval_surveillance_n1000000.md`, `fib_gaps.md`,
`kernel_support.md`, `robustness_solver.md`, `tau_sensitivity.md`); pipeline
in `dp/` (see `docs/DP_PLAN.md`).
The earlier six-state / QALY version of this analysis is preserved as
`paper/archive/manuscript_v1_6state_superseded.md` and is superseded by this one.*

---

## Abstract

**Background.** Colonoscopy is effective but scarce and invasive, so the
relevant question for a screening programme is not only how much colorectal
cancer (CRC) a schedule prevents but how much it prevents *per colonoscopy*.
Microsimulation studies, including the CMOST-based optimisation of Zaika et
al. (2024), search for the best *fixed* schedule — the same ages for
everyone — and therefore cannot use the information that screening itself
generates: what a person's previous colonoscopies found. We ask whether a
policy that adapts each person's next interval to their own findings can
beat the best fixed schedule at equal or lower colonoscopy volume.

**Methods.** We formulated screening as a finite-horizon *mixed-observability*
Markov decision process (MOMDP) whose observed component is the patient's
age, sex, years since the last colonoscopy and the finding at that
colonoscopy, and whose hidden component is a six-level latent adenoma-risk
class crossed with eleven clinical states (no lesion, CMOST polyp stages
1–6, undetected cancer stages I–IV). Every transition, observation and exit
kernel was estimated by maximum likelihood **directly from the CMOST engine
itself** — instrumented with a quarter-resolved state recorder — using
2 000 000 never-screened and 2 000 000 randomly-screened simulated lives
(4 520 648 colonoscopies). The objective, minimised, is the expected number of CRC deaths (or
diagnoses) plus a shadow price λ per colonoscopy; sweeping λ traces an
efficiency frontier. We solved the MOMDP by point-based value
iteration with exact (Monte-Carlo-free) in-model policy evaluation, and
evaluated every policy back inside the real CMOST engine against uniform
10-year and 5-year schedules, against the best of 2 112 exhaustively
searched fixed schedules, and against a 10-year programme augmented with
CMOST's own post-polypectomy surveillance rule, with population-paired
random-number streams (1 000 000 persons per arm).

**Results.** The reduced model reproduced the engine's CRC mortality and
incidence to within 0.5 % (no screening), 1.7 % (10-yearly) and 5.3 %
(5-yearly), and its age-specific lesion prevalence trajectories. In the
engine, the adaptive policy **dominated every screening-only fixed
comparator on CRC deaths and diagnoses at lower colonoscopy volume**.
At 2.29 colonoscopies per person (11 % fewer than the 10-yearly schedule's
2.58) CRC mortality was 899.8 vs 1032.1 per 100 000 (paired difference
−132.3 ± 12.3) and incidence 2486.4 vs 2686.9 (−200.5 ± 22.9); deaths
averted per 1000 colonoscopies rose from 3.32 to 4.32 (**+30 %**) and
diagnoses averted from 7.22 to 9.00 (**+25 %**). At 4.59 colonoscopies
(8 % fewer than the 5-yearly schedule's 4.99) mortality was 682.5 vs 775.3
(−92.8 ± 10.4), an 18 % gain in deaths averted per colonoscopy. The policy also beat the *re-optimised* fixed schedules while using fewer
colonoscopies than they did (2.29 vs 2.44 and 4.59 vs 4.88 per person; −81.1
± 9.2 and −42.0 ± 9.7 deaths per 100 000). The 10-yearly programme with
CMOST's own post-polypectomy surveillance added — the comparator closest to
guideline practice — reached 879.8 deaths per 100 000, 20 ± 12 fewer than the
adaptive policy, but only by performing 29 % more colonoscopies (2.95 per
person, a fifth of them surveillance exams); per colonoscopy it averted 3.42
deaths per 1000 against the adaptive policy's 4.32, and an adaptive policy
solved for its volume (2.87 colonoscopies) had 62 ± 12 fewer deaths at a
similar number of diagnoses; the 5-yearly analogue (5.14 colonoscopies)
reached 740 deaths per 100 000, 72 ± 11 more than the adaptive policy at
the same volume. Making the risk class observable at baseline turned the policy into an
explicit risk-targeting programme: it withheld screening from the
lowest-risk half of the population and gave the highest-risk 5 % 7.5
colonoscopies each, cutting that group's mortality from 5535 to 1956 per 100
000 while using **1.66 colonoscopies per person overall** - 6.34 deaths
averted per 1000 colonoscopies, nearly twice the 10-yearly schedule's
efficiency. With a baseline score of realistic discrimination instead of
perfect class knowledge - a polygenic-score-like AUC of 0.60, whose top
decile carries 1.8 times average risk - the gain shrank to 24 fewer CRC
deaths per 100 000 at equal colonoscopy volume, and information and
adaptivity proved nearly additive, with adaptivity worth about three times a
realistic score. Under imperfect adherence the adaptive
policy re-planned around no-shows without re-solving and held a 44–53 %
mortality reduction at every attendance rate, whereas the fixed programme
decayed from 45 % to 18 %. On life-years the adaptive and fixed schedules
were statistically indistinguishable, and a life-year-objective sweep did
not transfer its in-model advantage to the engine — an honest null we
report alongside the positive findings.

**Conclusions.** Individualising colonoscopy timing on realised findings —
not merely on baseline risk — yields a screening programme that prevents
substantially more colorectal cancers and cancer deaths per colonoscopy than
any fixed schedule, including schedules optimised for the same simulator.
The gains are large enough to matter for capacity-limited programmes, are
robust to imperfect adherence, and are captured to a useful degree by a
simple three-tier surveillance rule that the optimal policy itself suggests.

---

## 1. Introduction

Colorectal cancer is among the leading causes of cancer incidence and
mortality in industrialised countries, and colonoscopy prevents it both by
removing adenomatous precursors and by detecting cancer early. Guidelines
therefore recommend colonoscopy at fixed intervals — typically every ten
years from age 45 or 50 — and microsimulation models (MISCAN, CRC-SPIN,
SimCRC, CRC-AIM and the open-source CMOST) are the standard instrument for
choosing those intervals, because randomised trials cannot span the design
space.

Colonoscopy capacity, however, is finite, and each procedure carries cost,
bowel preparation burden and a small risk of perforation or bleeding. The
decision-relevant quantity for a programme is thus the *efficiency* of the
schedule: cancers and cancer deaths prevented per colonoscopy performed.
Zaika et al. (2024) used CMOST to search, free of the 5-/10-year grid, for
the optimal fixed schedule of one to four colonoscopies between ages 20 and
90, and showed that the optimum shifts with adenoma risk, adenoma detection
and adherence. Their personalisation is nevertheless *stratification*: a
hypothetical test splits the population into risk strata and a separate
fixed schedule is optimised within each. The schedule still does not
respond to what a given person's colonoscopies actually find.

Clinical practice is, by contrast, already adaptive: a normal colonoscopy
earns a ten-year interval, whereas three or more adenomas, or an advanced
adenoma, trigger surveillance at three to five years, because the finding
reveals elevated future risk (Gupta et al. 2020). This is exactly the
structure of a partially observable Markov decision process: the colon's
true state is hidden, each colonoscopy is a noisy observation of it, and the
optimal action depends on the belief the observations induce. POMDP
formulations of cancer screening have been developed for breast cancer
(Ayer et al. 2012) and for colonoscopy surveillance (Erenay et al. 2014),
but they have generally been solved on abstract models calibrated to
summary statistics rather than derived from — and validated back against —
a full microsimulation.

We take the microsimulation itself as ground truth. Our contributions are:

1. **A decision model estimated from the simulator, not from summaries.**
   Every kernel of the MOMDP is a maximum-likelihood estimate from
   quarter-resolved trajectories of the real CMOST engine, including the
   post-polypectomy dynamics that a never-screened cohort cannot reveal
   (§2.3). The model reproduces the engine's mortality and incidence under
   no screening and under both guideline schedules to within a few percent
   (§3.1).
2. **A mixed-observability formulation with the right memory.** Conditioning
   on the observed pair (years since last colonoscopy, last finding), and on
   a six-level latent risk class spanning CMOST's 23-fold risk gradient,
   is what makes a Markov abstraction of CMOST accurate enough to optimise
   against (§2.2, §3.1–3.2).
3. **An engine-verified dominance result.** With population-paired random
   numbers and one million persons per arm, the adaptive policy prevents
   30 % more CRC deaths and 25 % more CRC diagnoses per colonoscopy than the
   uniform 10-year schedule, and also beats fixed schedules re-optimised
   inside the same simulator (§3.3).
4. **Deployment-relevant extensions**: an explicit risk-targeting variant
   when the risk class is observable (§3.5), robustness to imperfect
   adherence without re-solving (§3.7), a transparent three-tier rule
   distilled from the policy (§3.4), and an honest life-year null (§3.8).

---

## 2. Methods

### 2.1 The CMOST microsimulation and its instrumentation

CMOST (Prakash et al. 2017) simulates, in three-month increments from birth
to age 100, the adenoma–carcinoma sequence: age-, sex- and
location-dependent adenoma initiation; growth through six polyp stages;
malignant transformation through the adenoma pathway or a minor direct
path; symptomatic presentation; stage progression; stage-specific CRC
mortality; colonoscopy detection, polypectomy and complications; and
competing all-cause mortality. Each simulated person carries a lifelong
`individual_risk` multiplier on adenoma initiation, drawn from a
right-skewed pool, and each polyp its own growth speed.

We used the MATLAB-ported Python engine (`cmost_engine/NumberCrunching_policy.py`,
CMOST13 parameter set) and added two record-only instruments: a
**quarter-resolved state recorder** that stores each person's discretised
state at the start of every quarter of life, and a **decision hook** that,
once a year at each person's decision epoch, receives the pre-decision true
state, returns WAIT or SCREEN, and is told the engine's *actual*
colonoscopy result — how many early polyps and advanced polyps were
removed, whether cancer was detected and at which stage, and whether the
procedure was fatal. Neither instrument alters the dynamics: with no hook
attached, and with or without the recorders attached, the instrumented
engine reproduces the un-instrumented port bit-for-bit on the same seed —
every death cause and time, diagnosis, polyp removed and cost — which a
regression test in the repository checks
(`tests/test_engine_hook_regression.py`).

### 2.2 Decision model: a finite-horizon MOMDP

Decisions are annual, at ages 40–80; outcomes accrue to age 100. The state
factorises into an observed and a hidden component (Ong et al. 2010).

*Observed*: age; sex; τ = years since the last colonoscopy ("never" if
none, capped at 13); and the finding at that colonoscopy, in four
categories — **normal**, **1–2 early adenomas** (CMOST polyp stages 1–4),
**≥3 early adenomas**, **advanced adenoma** (stages 5–6).

*Hidden* (belief-tracked): a **risk class** — one of six quantile bins of
CMOST's `individual_risk` distribution, cut at the 50th, 80th, 95th, 96.5th
and 98th percentiles — crossed with eleven **clinical states**: no lesion;
most advanced polyp at stage 1, 2, 3, 4, 5 or 6; undetected cancer at stage
I, II, III or IV. Sixty-six hidden states per sex.

Actions are WAIT and SCREEN. Observations under WAIT are *no event* or
*exit*; under SCREEN they are the four finding categories or *exit*.
**Exits** — CRC diagnosis at stage k (symptomatic or screen-detected),
death from other causes, and death from a colonoscopy complication — are
terminal for the decision process. Their remaining-lifetime consequences
(probability of eventual CRC death, remaining life-years) enter the one-step
reward through estimated exit values, so the belief simplex has only the 66
alive-and-undiagnosed dimensions.

Two modelling choices deserve emphasis, because both were forced on us by
diagnostic failures of simpler alternatives.

**Keeping CMOST's six polyp stages.** The finer lesion axis is retained for
a related but weaker reason, and its contribution is conditional (§3.3 shows
that the policy's advantage over fixed scheduling survives every coarsening
tested; what the coarsenings cost is the model's predictive accuracy): §3.1
reports an ablation in which each abstraction is re-estimated from the
*same* engine cohorts and asked to predict what the engine does under the
guideline schedules. Pooling the six polyp stages into "early" and
"advanced" — the discretisation of our earlier six-state analysis and of
most published abstractions — turns out to be nearly harmless *provided the
memory is kept*. The two coarsenings interact, however: without the memory,
the stage-resolved model over-predicts the mortality reduction of 10-yearly
screening by 8.9 percentage points while the pooled model under-predicts it
by 8.0, so a memoryless abstraction is wrong not only by a large margin but
with a sign that depends on the lesion axis. Six risk classes matter mainly
at higher screening intensity, where one or three classes over-predict the
5-yearly mortality reduction by 4.4 and 3.4 percentage points.

**Conditioning the kernels on (τ, last finding).** The natural history of a
person *after* a colonoscopy is not the natural history of the
never-screened cross-section at the same age and clinical state. Two
mechanisms drive the difference in CMOST, and both are real in the disease:
polyps that were removed leave behind a person whose within-class risk is
higher than average (they had polyps), and the unscreened cross-section is
enriched for slow-growing lesions that have simply not yet progressed
(length-biased sampling). Empirically, for men of the second risk class at
age 60, the annual probability that a stage-4 polyp becomes advanced is
0.18 in never-screened person-years but 0.21–0.31 in post-colonoscopy
person-years when the post-screen cells are pooled over the last finding by
years since the colonoscopy (0.21, 0.28, 0.28, 0.31 and 0.30 at τ = 0, 1, 2,
3 and 4–5), and 0.19–0.32 across the individual finding-specific cells,
most of which are backed-off estimates (§2.3). Because both τ and the last
finding are known to any screening programme, conditioning on them buys
this accuracy for free, without enlarging the belief space.

Three assumptions should be stated explicitly. The hidden process is
*assumed* Markov given (risk class, clinical state, τ, last finding) and
time-inhomogeneous in age; the kernels estimated from a randomised-schedule
cohort are assumed policy-invariant given that conditioning set; and
within-class heterogeneity remains (class 0 alone spans `individual_risk`
0.03–0.77, a 22-fold range) and is what the memory conditioning has to
absorb. §3.1 tests these assumptions on fixed schedules only. No
stationarity, ergodicity or compactness argument is needed: the horizon is
finite, the belief lives on a compact 66-simplex per observed key, and the
solver is backward induction on that structure (§2.5).

### 2.3 Kernel estimation from the engine

All kernels are maximum-likelihood estimates on person-year windows from two
instrumented cohorts:

* **Never-screened cohort**: 2 000 000 lives with no screening or
  surveillance colonoscopy (only the 0.049 diagnostic colonoscopies per
  person that follow a symptomatic presentation).
* **Randomly-screened cohort**: 2 000 000 lives whose first colonoscopy age
  is uniform on 40–75 and whose subsequent intervals are uniform on 1–20
  years, yielding 4 520 648 colonoscopies with the engine's true pre- and
  post-procedure states. This cohort supplies the post-colonoscopy memory
  cells, and its randomised design ensures the estimated kernels are not
  confounded with any particular screening policy.

**WAIT kernel** `T[sex, class, (τ, finding), age]`: the annual transition
from the decision-time state at age *y* to that at *y*+1, with exits read
from the intermediate quarterly snapshots so that a cancer that arises, is
diagnosed and kills within a single year is never missed. The window's end
state is the *pre-colonoscopy* state when a colonoscopy occurs at *y*+1, so
that screening effects never leak into the natural-history kernel.

**SCREEN kernel** `K[sex, class, (τ, finding), age band]`: the joint
probability of the observed finding and of the post-polypectomy clinical
state given the pre-procedure state, plus the probabilities of the exits at
the procedure itself (cancer detected at stage k, fatal complication). This
kernel carries the engine's per-lesion detection probabilities, colonoscope
reach, residual missed polyps and complication risks; nothing about
colonoscopy is specified by hand.

**Exit values**: for each sex, age and stage at diagnosis, the probability
of eventual CRC death and the expected remaining life-years. These depend
on (sex, age, stage) only — pooled over symptomatic and screen-detected
cases, over classes and over screening history — so post-diagnosis survival
is memoryless in the model; §3.8 returns to this.

**Initial belief**: the empirical joint distribution of (class, clinical
state) at the age-40 decision among persons alive and undiagnosed at that
epoch, i.e. at the start of the second quarter of age 40. This snapshot
defines the population on which every model-versus-engine comparison of
§3.1–3.2 is made (965 258 of the 1 000 000 persons of the paired arms).

The memory axis is estimated on τ groups {never, 0, 1, 2, 3, 4–5, 6–8,
9–12, 13+} crossed with the four findings (33 cells), so, for instance,
τ = 7 and τ = 8 share a kernel although the observed state tracks τ
exactly up to 13. Sparse cells back off hierarchically — widening the age
window (±0, 1, 2, 4, 8 years for the WAIT kernel; ±0, 1, 2, 4, 8 five-year
age bands for the SCREEN kernel, whose age axis is banded), then pooling τ
groups while keeping the last finding, then pooling findings within τ
groups, then pooling all post-screen cells, then all memory cells, then
class, then sex — with a minimum of 150 observations (person-years for the
WAIT kernel, colonoscopies for the SCREEN kernel) before a cell stands on
its own. A row observed at no level would default to "stay" (WAIT) or
"detected at the pre-procedure stage" (SCREEN); no row required this
default, and none required the sex-pooled level. The level at which each
row was resolved is stored with the kernels, and §3.1 reports which levels
the solved policy actually relies on.

### 2.4 Objective, shadow price and comparators

The objective, maximised and undiscounted over ages 40–100, is

    − w_death · E[CRC deaths] − w_inc · E[CRC diagnoses]
    − w_comp · E[complication deaths] + w_ly · E[life-years] − λ · E[colonoscopies],

with (w_death = 1) for the mortality objective, (w_inc = 1) for the
incidence objective and (w_ly = 1) for the life-year objective; the other
weights are zero in each case, so in the mortality objective a fatal
colonoscopy complication enters the optimisation only through λ (as a
colonoscopy performed), not as a death, although complication deaths are
reported as an engine outcome (Table 4). λ is the shadow price of a
colonoscopy: sweeping it traces the whole efficiency frontier, and
comparisons with fixed schedules are made at matched volume rather than at
a matched arbitrary budget.

Because every metric is linear in the state occupancy, each policy's exact
expected deaths, diagnoses, complication deaths, life-years and colonoscopy
count are computed by forward propagation of its belief tree — no
Monte-Carlo noise, milliseconds per policy. This in turn makes an
**exhaustive search over fixed schedules** cheap: we evaluated all 2 112
candidates (every equal-interval schedule with start 45–65 and interval
3–15 years; all free schedules of one to three colonoscopies on a two-year
grid from 44 to 80; all four-colonoscopy schedules on a three-year grid;
plus the empty schedule and the guideline 5-yearly schedule, which those
families do not contain) and kept the best at each volume as a comparator. The fixed comparators are
therefore not straw men: they are optimised in the same model the adaptive
policy is optimised in.

### 2.5 Solver

We solved the MOMDP by finite-horizon point-based value iteration (Pineau et
al. 2003; Walraven & Spaan 2019), exploiting the mixed observability by
indexing belief sets and α-vector sets by the observed key (age, τ,
finding) (Ong et al. 2010). The value function is piecewise-linear and
convex in the 66-dimensional hidden belief at each key (Smallwood & Sondik
1973), and the policy is the action attached to the maximising α-vector.
Belief sets are initialised as the closure of beliefs reachable from the
initial belief under a reference screening propensity of 0.12 at every
key, merged after rounding to 10⁻⁴ and capped per key — at 600 points for
the λ-grid, life-year and ablation policies and 1500 for the headline and
observed-class policies — keeping the largest reach probabilities. One
backward sweep of point-based Bellman backups yields one α-vector per
belief point; the resulting policy's own exact reachable set (weighted by
reach probability) plus 150 (cap 600) or 300 (cap 1500) ε-greedy rollouts
with ε = 0.1 are then added and the sweep repeated until the policy's
*exact in-model objective* fails to improve by more than 10⁻⁷ in two
consecutive rounds, within a budget of six, four (headline) or three
(ablation) rounds; the best policy seen is kept. This stopping rule is a
policy-improvement fixed point, not a Bellman-residual or belief-density
criterion, and the expansion is reachability-based rather than
distance-based (Pineau et al. 2006; Shani et al. 2013): coverage is
guaranteed where the reference and current policies go, not where the
optimal policy goes. Because every α-vector is the value of an executable
plan (backups compose executable plans; the horizon boundary and the
fallback used at a successor key with no belief points are WAIT-only
plans), the representation is a valid lower bound wherever it is
evaluated, so the scheme can only be pessimistic. A fast-informed bound
(Hauskrecht 2000) provides an upper bound.

Convergence evidence is empirical rather than certified. (i) The exact
in-model objective is flat to within 2 × 10⁻⁶ after one or two expansion
rounds for all twelve cap-1500 headline policies. (ii) Raising the
belief-point cap from 600 to 1500 per key changes a headline policy's
in-model mortality by at most 0.3 % and its colonoscopy volume by at most
0.6 % (0.17 % and 0.36 % once the sexes are pooled), moves the lower bound
by about 10⁻⁶ in objective units, and does not move the fast-informed
bound at all. (iii) The fast-informed gap is nevertheless large: 57 % of
the objective at λ = 0.001561, 44–48 % at the other mortality prices,
33–46 % for the incidence objective and 51–57 % for the ablation policies
(`results/dp/fib_gaps.md`) — about 700 deaths per 100 000 in death
equivalents, against an in-model advantage of the adaptive policy over the
best fixed schedule of about 90. The bound therefore cannot certify
optimality; looseness of the bound and genuine suboptimality cannot be
separated, and the density bound of Pineau et al. (2006) is not computable
for a weight-pruned reachable set (at this horizon it would need a
worst-case density below about 10⁻⁵ to be informative). Two consequences
follow. The policies reported below are *solved* policies — lower bounds
on the optimum — and every in-model dominance statement over fixed
schedules is conservative, because a suboptimal adaptive policy already
wins. And instead of a worst-case density we report the coverage the
belief sets actually achieve, together with the sensitivity of the solved
policy to the rollout seed and to the reference propensity (§3.3). The
six headline policies and the two observed-class variants use the larger
budget; the λ-grid policies that trace the frontier (Figure 1) and the
life-year sweep of §3.8 were solved at the 600-point cap.

### 2.6 Evaluation in the engine

Each policy is deployed inside the real engine through the hook of §2.1: it
receives the engine's actual findings, updates its belief with the model's
own kernels, and never re-screens a person already diagnosed with CRC. The
fixed comparators use the same hook mechanism and the same
no-re-screening rule, so the arms differ only in decision logic.

Every arm, fixed or adaptive, runs with CMOST's built-in polyp and cancer
surveillance switched off (CMOST13's default), so the fixed comparators of
Tables 4–5 are *screening-only* programmes, and part of what the adaptive
policy adds is precisely the post-polypectomy surveillance those
programmes lack. To make that explicit we also evaluate a **10-yearly +
surveillance** comparator: CMOST's own screening semantics (a colonoscopy
whenever 50 ≤ age ≤ 70 and ten years have passed since the last
colonoscopy of any kind, which is 50/60/70 for a person with no findings)
combined with CMOST13's own post-polypectomy surveillance rule —
re-examination five years after an early adenoma, repeated in each of the
following years up to nine if no colonoscopy has intervened, and three
years after an advanced adenoma or after three or more adenomas, then
five-yearly thereafter. The rule is implemented in the hook rather than
through the engine's flag so that a surveillance exam is a single annual
decision (the engine's block would otherwise add its own colonoscopy in
the same year as the hook's), is counted as a programme colonoscopy, and
resets the surveillance clocks with the engine's real findings exactly as
the engine does (`dp/hooks.py`, `FixedSurveillanceHook`). A 5-yearly
analogue (50–75, five-year interval) is evaluated the same way (§3.3).

The belief-tracking hook has two silent fallbacks — WAIT when the policy
has no α-vector at the observed key, and an unchanged belief when the
engine's observation has zero probability under the model — and both are
counted per arm and reported (§3.3).

Arms are simulated on identical chunk seeds, so populations and
random-number streams coincide until the first colonoscopy diverges them;
differences are therefore paired at the chunk level. Headline arms use
1 000 000 persons (20 chunks of 50 000), λ-grid arms 200 000. Standard
errors are computed over the chunk pairs (ddof = 1) and intervals use t
with 19 degrees of freedom; because paired arms contain the same persons
in the same order, person-level paired standard errors are also computed
(`results/dp/paired_tables.md`); they run from 11.6 to 13.4 per 100 000
across the Table 5 contrasts against 9.2 to 13.1 for the chunk-level ones,
so no conclusion depends on the choice. p-values for the family of Table 5
death contrasts are Holm-adjusted. Endpoints:
CRC deaths per 100 000 (at ages ≥ 40, the decision window), CRC diagnoses
per 100 000, colonoscopies initiated by the policy per person,
colonoscopy-complication deaths, life-years from age 40 and total cost.
The primary efficiency measures are deaths and diagnoses averted versus no
screening per 1000 colonoscopies.

---

## 3. Results

### 3.1 The reduced model reproduces the engine

Propagated as a cohort model, the MOMDP reproduces the engine's outcomes for
policies it was not fitted to (Table 1). Errors are within Monte-Carlo range
for no screening, small for the 10-yearly schedule, and within 5.3 % for
the 5-yearly schedule, whose intensity lies at the edge of the
randomised-screening cohort's support.

**Table 1.** Model prediction vs engine, on the model's own population
(persons alive and undiagnosed at the age-40 decision epoch, the start of
the second quarter of age 40; engine n = 965 258 of 1 000 000; the same
definition is used for Tables 2 and 3, whereas the engine arms of Tables
4–6 count every simulated person).

| schedule | CRC deaths /100k (model → engine) | rel. error | CRC diagnoses /100k (model → engine) | rel. error |
|---|---|---|---|---|
| no screening | 1934 → 1943 | −0.5 % | 4641 → 4628 | +0.3 % |
| 10-yearly (50/60/70) | 1040 → 1058 | −1.7 % | 2685 → 2703 | −0.7 % |
| 5-yearly (50–75) | 750 → 792 | −5.3 % | 2130 → 2207 | −3.5 % |

Age-specific prevalence trajectories also match: under both no screening and
10-yearly screening the model tracks the engine's decision-time prevalence
of early adenoma, advanced adenoma and undetected cancer across ages 40–99,
including the saw-tooth depletion and regrowth around each screening age
(Figure 2).

The ablation of Table 2 shows which parts of the abstraction earn their
keep. Dropping the (τ, last finding) memory is the single most damaging
simplification, and its error changes sign with the lesion axis: the
stage-resolved memoryless model over-states the mortality reduction of
10-yearly screening by 8.9 percentage points, while the pooled memoryless model — which reproduces two of the coarsenings
of our earlier six-state analysis, its pooled lesion axis and its lack of
memory, though it retains six latent risk classes and the four undetected
cancer stages — under-states it by 8.0. With the memory kept, pooling the polyp stages is nearly harmless for
these fixed schedules. Coarsening the latent risk to one or three classes is
harmless at 10-yearly intensity but over-states the 5-yearly reduction by
4.4 and 3.4 points, because a coarse class cannot represent the very
high-risk tail that repeated screening increasingly selects for.

**Table 2.** Model-structure ablation. Each variant is re-estimated from the
same two engine cohorts and asked, as a cohort model, to predict what the
engine does; entries are the predicted reduction versus no screening. The
model used in this paper is the first row after the engine.

| model structure | 10-y: death red. | 10-y: inc. red. | 5-y: death red. | 5-y: inc. red. |
|---|---|---|---|---|
| **engine (truth)** | **45.6 %** | **41.6 %** | **59.3 %** | **52.3 %** |
| 11 states, memory, 6 classes (this paper) | 46.2 % | 42.2 % | 61.2 % | 54.1 % |
| polyp stages pooled to early/advanced | 46.0 % | 42.0 % | 59.4 % | 52.1 % |
| no (τ, last finding) memory | 54.5 % | 50.6 % | 65.1 % | 58.1 % |
| pooled stages *and* no memory | 37.6 % | 31.6 % | 51.7 % | 41.7 % |
| 3 risk classes | 45.6 % | 41.5 % | 62.7 % | 55.7 % |
| 1 risk class (no latent risk) | 45.2 % | 41.1 % | 63.7 % | 56.7 % |

**Which parts of the kernels the policy relies on.** Because the level of
the back-off ladder at which every kernel row was resolved is stored, we
can ask where the rows the solved policy actually uses come from
(`results/dp/kernel_support.md`; rows weighted by the exact occupancy of
the headline policy). At the decision ages, 96.4 % of the WAIT-kernel
occupancy falls on rows estimated in their own memory cell at the exact
age and 99.6 % within ±8 years; for the SCREEN kernel the figures are
88 % in the own cell at the exact age band and 96 % within the widened
bands. The exceptions are informative. Repeat colonoscopies one to three
years after the previous one — the exams the policy orders after a finding
of three or more adenomas — are rare in the randomised cohort (92–99 % of
their occupancy sits on cells with fewer than 150 own colonoscopies), so
their SCREEN rows are estimated from the same finding pooled over τ; they
carry 4 % of the policy's colonoscopies. The open-ended τ ≥ 13 cell, on
which the long interval after a clean first exam rests, is used almost
only with a normal last finding (9 % of WAIT occupancy) and is then
estimated in its own cell at the exact age from 6.2 million person-years.
Because the randomised design continues screening to age 80 with
intervals of 1–20 years, that cell contains only τ = 13–20 person-years at
every decision age; persons whose screening has simply stopped enter it
only after age 80, where it feeds the WAIT-only continuation (17 % of the
cell's person-years overall). §3.3 reports a re-estimation that excludes
every person-year beyond τ = 20.

The underlying natural history is that of CMOST's own calibration: lifetime
clinical CRC incidence 4.56 %, lifetime CRC mortality 1.92 %, mean age at
diagnosis 72.3 years, stage distribution at diagnosis 15.4 / 36.1 / 27.4 /
21.1 % (I/II/III/IV), advanced-adenoma prevalence 4.1 % at age 60 and 6.6 %
at age 70. As previously reported for CMOST, absolute adenoma prevalence and
incidence sit somewhat below US screening-cohort estimates; because every
comparison here is made between policies inside the same simulator, this
affects the external interpretation of absolute rates but not the relative
ranking of policies.

### 3.2 Six latent risk classes span a 23-fold gradient

CMOST's `individual_risk` distribution is heavily right-skewed: half the
population lies below 0.77 and 2 % above 36. Six quantile classes capture
this without wasting resolution (Table 3). Three classes proved
insufficient: with three, the model matched the class posterior of every
finding closely, yet under-predicted the subsequent cancer incidence of
persons with ≥3 adenomas 2.5-fold in men and 3.6-fold in women (2.9-fold
pooled, in a 100 000-person diagnostic run), because such persons
concentrate in the extreme within-class tail. Going from six to eight
classes produced no further improvement in the diagnostic run that guided
this choice; that run is not retained as an artefact, so we report it only
as the reason six classes were kept.

**Table 3.** Latent risk classes (never-screened cohort, persons alive and
undiagnosed at age 40).

| class | share | `individual_risk` range (mean) | lifetime CRC incidence | lifetime CRC mortality |
|---|---|---|---|---|
| 0 | 50.0 % | 0.03–0.77 (0.24) | 1.78 % | 0.74 % |
| 1 | 30.0 % | 0.78–3.62 (2.16) | 4.23 % | 1.75 % |
| 2 | 15.0 % | 3.62–4.24 (3.89) | 6.46 % | 2.70 % |
| 3 | 1.4 % | 6.23–18.19 (12.21) | 16.05 % | 6.75 % |
| 4 | 1.6 % | 20.18–34.13 (27.12) | 29.31 % | 12.63 % |
| 5 | 2.0 % | 36.12–54.05 (45.05) | 41.61 % | 18.23 % |

The gradient between the top 2 % and the bottom 50 % is 23-fold in incidence
and 25-fold in mortality. The class is latent: the policy starts from the
population prior and infers it from findings. The inference is accurate: after a
first colonoscopy at age 52 the model's posterior over the six classes
agrees with the engine's empirical class composition of the same people to
within 0.03 in probability for every finding, and it is informative — a
finding of ≥3 adenomas moves P(classes 3–5) from the 0.05 prior to 0.93 in
men and 0.98 in women (P(top class) 0.02 → 0.53 and 0.71), whereas a normal
exam lowers it to 0.007 and 0.017.

### 3.3 The adaptive policy dominates every fixed comparator

Sweeping λ produces a frontier of policies that lies strictly below the
fixed-schedule frontier over the whole clinically relevant volume range
(Figure 1). Table 4 gives the headline arms at one million persons each;
"dominates" is used throughout in the sense of *fewer CRC deaths and
diagnoses at a lower colonoscopy volume*, with complication deaths
reported separately.

**Table 4.** Engine evaluation, n = 1 000 000 per arm, population-paired
seeds. "per 1000 colos" = events averted versus no screening per 1000
colonoscopies performed.

| arm | colos / person | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000 colos | dx averted /1000 colos | complication deaths /100k |
|---|---|---|---|---|---|---|
| no screening | 0.000 | 1887.6 (15.1) | 4545.1 (17.9) | — | — | 0.1 |
| fixed 10-yearly (50/60/70) | 2.575 | 1032.1 (9.1) | 2686.9 (13.2) | 3.32 | 7.22 | 9.9 |
| best fixed at that volume (54/64/74) | 2.441 | 980.9 (9.0) | 2641.2 (17.5) | 3.71 | 7.80 | 7.3 |
| **adaptive, λ = 0.001561** | **2.289** | **899.8 (7.6)** | **2486.4 (17.5)** | **4.32** | **9.00** | 9.3 |
| adaptive (incidence objective), λ = 0.005125 | 2.164 | 932.4 (12.9) | 2516.7 (13.9) | 4.42 | 9.38 | 8.4 |
| three-tier rule (52; 10/5/3/3 y) | 2.890 | 866.5 (10.2) | 2344.9 (15.2) | 3.53 | 7.61 | 11.3 |
| fixed 10-yearly + CMOST surveillance (rolling 50–70) | 2.946 | 879.8 (9.7) | 2350.1 (11.3) | 3.42 | 7.45 | 11.4 |
| adaptive, λ = 0.001189 (matched to the surveillance programme) | 2.874 | 818.1 (6.8) | 2331.7 (15.3) | 3.72 | 7.70 | 11.1 |
| fixed 5-yearly (50–75) | 4.986 | 775.3 (8.8) | 2208.0 (13.4) | 2.23 | 4.69 | 19.4 |
| best fixed at that volume (48–78, 6-yearly) | 4.879 | 724.5 (9.7) | 2140.5 (11.7) | 2.38 | 4.93 | 16.0 |
| **adaptive, λ = 0.00069** | **4.586** | **682.5 (6.6)** | **2056.9 (13.7)** | **2.63** | **5.43** | 15.7 |
| adaptive (incidence objective), λ = 0.001724 | 5.149 | 668.4 (8.6) | 1977.6 (15.3) | 2.37 | 4.99 | 17.5 |
| fixed 5-yearly + CMOST surveillance (rolling 50–75) | 5.143 | 740.2 (6.5) | 2126.3 (13.4) | 2.23 | 4.70 | 15.8 |
| **adaptive + observed risk class, λ = 0.001561** | **1.665** | **832.9 (6.6)** | **2303.3 (14.6)** | **6.34** | **13.47** | 7.2 |

**Table 5.** Population-paired differences (identical seeds; mean ± SE per
100 000; |Δ|/SE in parentheses).

| comparison | Δ colonoscopies | Δ CRC deaths | Δ CRC diagnoses |
|---|---|---|---|
| adaptive λ = 0.001561 − fixed 10-yearly | −0.287 | **−132.3 ± 12.3** (10.7) | **−200.5 ± 22.9** (8.8) |
| adaptive λ = 0.001561 − best fixed (54/64/74) | −0.153 | **−81.1 ± 9.2** (8.8) | −154.8 ± 21.0 (7.4) |
| adaptive λ = 0.00069 − fixed 5-yearly | −0.400 | **−92.8 ± 10.4** (9.0) | **−151.1 ± 13.8** (10.9) |
| adaptive λ = 0.00069 − best fixed (48–78) | −0.294 | **−42.0 ± 9.7** (4.3) | −83.6 ± 16.0 (5.2) |
| adaptive + observed class − fixed 10-yearly | −0.911 | **−199.2 ± 12.1** (16.5) | −383.6 ± 17.9 (21.4) |
| three-tier rule − fixed 10-yearly | +0.315 | −165.6 ± 13.1 (12.7) | −342.0 ± 20.0 (17.1) |
| best fixed (54/64/74) − fixed 10-yearly | −0.134 | −51.2 ± 12.0 (4.3) | −45.7 ± 21.6 (2.1) |

At the 10-yearly schedule's volume the adaptive policy uses 11 % fewer
colonoscopies and still leaves CRC mortality 12.8 % and incidence 7.5 %
lower; per colonoscopy it averts 30 % more deaths and 25 % more diagnoses.
At the 5-yearly schedule's volume it uses 8 % fewer colonoscopies for 12.0 %
lower mortality (+18 % deaths averted per colonoscopy, +16 % diagnoses).
Every paired mortality difference involving an adaptive policy exceeds
four chunk-level standard errors; the t(19) 95 % intervals exclude zero by
a wide margin (the narrowest, adaptive versus the best fixed schedule at
the 5-yearly volume, is −62 to −22 per 100 000), every contrast remains
beyond 3.6 person-level standard errors, and the largest Holm-adjusted
p-value in the family is 7 × 10⁻⁴ (`results/dp/paired_tables.md`).
Re-optimising the
fixed schedule inside the same model captures only about 39 % of the
adaptive policy's mortality advantage over the guideline schedule
(−51.2 of −132.3 per 100 000), so most of the gain comes from adaptivity
itself rather than from better fixed ages. Complication deaths are *lower*
under the adaptive policies than under the guideline schedules they replace
(9.3 vs 9.9 and 15.7 vs 19.4 per 100 000), so the mortality gain is not
bought with procedural harm; against the best fixed schedules they sit
higher at the 10-yearly volume (9.3 vs 7.3; paired +2.0 ± 0.9 per 100 000)
and level at the 5-yearly volume (15.7 vs 16.0; −0.3 ± 1.6).

Optimising incidence instead of mortality moves the policy only slightly
along the same frontier (Table 4), as expected when most CRC deaths are
prevented by preventing the cancer.

**The comparator closest to guideline practice.** The fixed schedules above
are screening-only (§2.6). Adding CMOST's own post-polypectomy
surveillance rule to the 10-yearly programme changes the comparison in an
instructive way (Table 5b; n = 1 000 000, paired with every other arm).
The augmented programme performs 2.946 colonoscopies per person — 2.372
screening and 0.574 surveillance exams, a fifth of its volume — and
reaches 879.8 CRC deaths and 2350 diagnoses per 100 000: 152 ± 10 fewer
deaths than the screening-only 10-yearly schedule for 0.37 more
colonoscopies, and 20 ± 12 fewer than the adaptive policy at λ = 0.001561,
which however uses 0.66 fewer colonoscopies per person (2.289; 29 % fewer).
Per colonoscopy the ordering is unchanged: 3.42 deaths and 7.45 diagnoses
averted per 1000 colonoscopies for the surveillance-augmented programme
against 4.32 and 9.00 for the adaptive policy. For a like-for-like
contrast we solved the adaptive policy at the price whose engine volume
sits just below the augmented programme's (λ = 0.001189, cap 1500) and
deployed it on the same million persons: at 2.874 colonoscopies per person
(0.07 fewer) it has 818.1 CRC deaths per 100 000, 62 ± 12 fewer than the
augmented programme, and 2332 diagnoses, a similar number (−18 ± 17). The
augmented programme therefore closes most of the *incidence* gap — its
surveillance exams remove the polyps a finding-responsive rule is meant to
catch — but only about a third of the *mortality* gap, because the
adaptive policy also allocates its screening colonoscopies by belief. It
is, in fact, almost exactly the three-tier rule of §3.4 (+13 ± 13 deaths,
+0.06 colonoscopies): CMOST's surveillance block *is* a finding-responsive
rule, and what the solved policy adds beyond it is the sex offset, the
belief-driven lengthening of intervals after clean exams and the earlier
stop. Complication deaths under the augmented programme (11.4 per
100 000) are higher than under the screening-only schedule (9.9) and
marginally higher than under the volume-matched adaptive policy (11.1),
in proportion to volume.

At the 5-yearly intensity the same exercise gives an exactly
volume-matched contrast, because the augmented 5-yearly programme (4.415
screening plus 0.728 surveillance colonoscopies, 5.143 per person) lands
on the volume of the incidence-objective adaptive arm of Table 4 (5.149).
The augmented programme reaches 740.2 deaths and 2126 diagnoses per
100 000 — 35 ± 10 fewer deaths than the screening-only 5-yearly schedule
for 0.16 more colonoscopies — but 72 ± 11 more deaths and 149 ± 15 more
diagnoses than the adaptive policy at the same volume, and 71 ± 15 more
than the mortality-objective policy at λ = 0.000525 (5.270 colonoscopies).
Surveillance therefore closes part of the gap between fixed and adaptive
programmes at low intensity, where it adds the most volume, and little of
it at high intensity, where 5-yearly screening already re-examines
everyone.

**Table 5b.** Fixed programmes with CMOST's post-polypectomy surveillance
rule (engine, n = 1 000 000, population-paired differences, mean ± SE per
100 000).

| comparison | Δ colonoscopies | Δ CRC deaths | Δ CRC diagnoses | Δ complication deaths |
|---|---|---|---|---|
| 10-yearly + surveillance − fixed 10-yearly | +0.371 | −152.3 ± 10.4 | −336.8 ± 13.6 | +1.5 ± 1.3 |
| 10-yearly + surveillance − best fixed (54/64/74) | +0.505 | −101.1 ± 9.0 | −291.1 ± 21.4 | +4.1 ± 1.3 |
| 10-yearly + surveillance − adaptive λ = 0.001561 | +0.657 | −20.0 ± 12.1 | −136.3 ± 21.2 | +2.1 ± 1.6 |
| 10-yearly + surveillance − adaptive λ = 0.001189 (matched volume) | +0.071 | **+61.7 ± 12.4** | +18.4 ± 16.8 | +0.3 ± 1.4 |
| 10-yearly + surveillance − three-tier rule (52; 10/5/3/3) | +0.056 | +13.3 ± 13.3 | +5.2 ± 18.6 | +0.1 ± 1.6 |
| 10-yearly + surveillance − adaptive + observed class | +1.281 | +46.9 ± 11.8 | +46.8 ± 16.5 | +4.2 ± 1.1 |
| 5-yearly + surveillance − fixed 5-yearly | +0.157 | −35.1 ± 10.3 | −81.7 ± 19.6 | −3.6 ± 2.1 |
| 5-yearly + surveillance − best fixed (48–78, 6-yearly) | +0.264 | +15.7 ± 11.8 | −14.2 ± 15.0 | −0.2 ± 1.9 |
| 5-yearly + surveillance − adaptive λ = 0.00069 | +0.557 | +57.7 ± 10.9 | +69.4 ± 20.1 | +0.1 ± 1.8 |
| 5-yearly + surveillance − adaptive λ = 0.000525 | −0.127 | +71.3 ± 14.6 | +113.9 ± 18.4 | −2.1 ± 2.4 |
| 5-yearly + surveillance − adaptive (incidence objective), λ = 0.001724 | −0.005 | +71.8 ± 11.0 | +148.7 ± 14.9 | −1.7 ± 1.7 |

The dominance is also robust to the model the policy was derived from. We re-solved the same shadow price on three of the five ablated abstractions
of Table 2 (at the 600-point cap and a three-round budget, i.e. less
converged than the headline policies) and deployed the resulting policies in the engine (n = 200 000
per arm): the pooled-stage, the memoryless and even the class-free (single
latent class) model all produce policies that beat the 10-yearly schedule by
121–172 CRC deaths per 100 000 (paired over four matched 50 000-person
chunks, SE 38–39) at 2.07–2.46 colonoscopies per person,
i.e. 4.21–4.76 deaths averted per 1000 colonoscopies against the schedule's
3.35. The elaborate abstraction is therefore what makes the model's
*predictions* accurate (Table 2); the *policy* advantage over fixed
scheduling survives considerable model misspecification, because it rests on
responding to findings at all rather than on the precise magnitude of the
response.

**Solver diagnostics.** Since the fast-informed bound cannot certify the
solved policies (§2.5), we report what can be measured
(`results/dp/robustness_solver.md`). Re-solving the headline price with the
same seed reproduces the headline policy's objective to all printed digits
in both sexes, so the solve is deterministic. Its final belief sets hold
2.30 million points over 1809 observed keys per sex; the beliefs the
policy actually visits are covered essentially exactly (reach-weighted
mean L1 distance to the nearest point 10⁻⁵), and their one-step
*deviations* — the successor beliefs of the action the policy does not
take, which the backup must evaluate to rank the two actions — are covered
to a mean L1 distance of 0.007 in men and 0.004 in women, with 93 % and
96 % of the deviation mass within 0.01 and 96 % and 98 % within 0.05. The
coverage is weakest at the end of the horizon (ages 77–80, mean 0.02–0.04,
95th percentile 0.10–0.17), where the stopping decision is taken and the
value at stake is smallest.

Re-solving with rollout seeds 1 and 2 and with reference propensities 0.06
and 0.25 gives in-model objectives within 3 × 10⁻⁷ of the headline's
(pooled mortality 880.5–880.8 per 100 000 at 2.367–2.369 colonoscopies),
α-vector sets that are not identical, and policy paths that coincide with
the headline's except for one-year shifts in the exams that follow a
finding of three or more adenomas (§3.4). Deployed in the engine on the
headline arm's first four chunk seeds (200 000 persons), the four variants
give 884.5–933.0 CRC deaths per 100 000 at 2.284–2.292 colonoscopies per
person against the headline's 889.5 — paired differences of −5 ± 27 to
+44 ± 19 — and every one of them beats the 10-yearly schedule by 101–150
deaths per 100 000 while using 11 % fewer colonoscopies. The one variant
that stood out at that sample size (seed 1, +44 ± 19) was extended to the
full million: at 2.288 colonoscopies per person it gives 903.8 ± 7.4 deaths
per 100 000, +4.0 ± 12.0 against the headline arm and −128.3 ± 11.3
against the 10-yearly schedule, so the 200 000-person deviation was
noise. The τ-restricted
kernels of §3.1 (person-years beyond τ = 20 excluded — 0.8 % of all
person-years, changing no kernel row below age 74) move the in-model
predictions of Table 1 by at most 1.3 % (10-yearly mortality 1040 → 1054
per 100 000), reproduce the headline policy's paths apart from the same
≥3-adenoma shift, and give 887.5 ± 16.3 deaths per 100 000 in the engine
against the headline's 889.5 (paired −2 ± 25) at the same volume
(`results/dp/tau_sensitivity.md`). The solved policy is thus one member
of a family of near-ties whose in-model values differ by less than one
death per 100 000; what the engine confirms is that family's dominance
over fixed scheduling, not the identity of its best member.

**Deployment diagnostics.** Re-deploying the headline policy on the first
four chunk seeds reproduces the cached arm outcome-for-outcome (every
death, diagnosis and colonoscopy count identical), and over the 6.84
million annual decisions of those 200 000 persons the hook never fell
back to WAIT for want of an α-vector, never met an engine observation of
zero probability under the model, and never met one whose probability
under the current belief fell below 10⁻⁶: the belief sets cover every
observed key the engine visits, and the estimated kernels assign
non-negligible probability to every finding the engine produces.

### 3.4 What the policy does

The solved policy is legible, and it reproduces the logic of
post-polypectomy surveillance guidelines while sharpening the quantities
(Figure 3; λ = 0.001561, the 10-yearly-volume-matched policy).

| observation path | men | women |
|---|---|---|
| all findings normal | 49, 69, 78 | 58, 73 |
| 1–2 adenomas at first, then normal | 49, 58, 67, 75 | 58, 63, 73, 80 |
| ≥3 adenomas at first, then normal | 49, 50, 57, 64, 75 | 58, 59, 63, 69, 74, 79 |
| advanced adenoma at first, then normal | 49, 52, 64, 75 | 58, 63, 73, 80 |
| advanced adenoma twice | 49, 52, 53, 61, 67, 75 | 58, 63, 67, 71, 76, 80 |

Three regularities stand out. First, **a clean first colonoscopy earns a
very long interval** — twenty years for men, fifteen for women — because in
this model a negative exam at that age is strong evidence of a low risk
class, and the belief takes years to re-accumulate. Second, **the response
to a finding is graded and immediate**: 1–2 adenomas shorten the next
interval to nine years in men and five in women, an advanced adenoma to
three years in men and five in women, and ≥3 adenomas trigger re-examination after a single year in both sexes,
followed by intervals of four to seven years (lengthening to eleven at the
end of the horizon in men) — the policy discovers, rather than is told, the
three-tier structure of the US Multi-Society Task Force recommendations.
The timing of that first re-examination is the one feature of these paths
that the objective does not pin down: solves with a different rollout seed,
a different reference propensity or the τ-restricted kernels (§3.3) place
it one to four years after the finding in men and shift the later exams of
the same path by a year, at in-model objective differences below 3 × 10⁻⁷
(0.03 deaths per 100 000), and the repeat-colonoscopy kernel rows at τ =
1–3 are the least-supported cells of the model (§3.1). What is robust is the
shape of the response — three or more adenomas bring the next exam
forward to within four years and keep the subsequent intervals at four to
seven — not the year.
Third, **the sexes are offset by roughly nine years** at the first
colonoscopy (49 vs 58), consistent with CMOST's sex-specific adenoma onset,
a difference no unisex fixed schedule can express.

Distilling this into a fixed *rule* — first colonoscopy at 52; next interval
10 years after a normal exam, 5 years after 1–2 adenomas, 3 years after ≥3
or advanced adenomas — already recovers a large share of the benefit
(866.5 deaths per 100 000 at 2.89 colonoscopies, Table 4): better mortality
than the 10-yearly schedule at 12 % more colonoscopies, and better
per-colonoscopy efficiency (3.53 vs 3.32). The rule is not optimal — it
neither uses sex nor lets intervals lengthen with age — but it shows that
most of the gain is implementable without belief tracking.

### 3.5 When the risk class is observable, the policy becomes a targeting programme

If a baseline test (a polygenic risk score, or any classifier of comparable
discrimination) revealed the risk class at age 40, the same objective and
shadow price produce a radically different, explicitly targeted programme
(Table 6). Screening ages along the all-normal path for men become: class 0,
**none at all**; class 1, ages 57/67/76; class 2, 41/56/66/75/80; class 5,
eleven colonoscopies from age 40 to 80. Overall volume falls to 1.665
colonoscopies per person.

**Table 6.** Engine outcomes by *true* risk class (n = 1 000 000). "low" =
class 0 (50 % of the population), "mid" = classes 1–2 (45 %), "high" =
classes 3–5 (5 %).

| arm | low: colos / deaths / dx | mid: colos / deaths / dx | high: colos / deaths / dx |
|---|---|---|---|
| no screening | 0.00 / 712 / 1741 | 0.00 / 1994 / 4857 | 0.00 / 12665 / 29726 |
| fixed 10-yearly | 2.58 / 580 / 1578 | 2.57 / 1033 / 2716 | 2.49 / 5535 / 13484 |
| fixed 5-yearly | 5.00 / 520 / 1535 | 4.99 / 750 / 2188 | 4.82 / 3547 / 9098 |
| adaptive (latent class) | 1.97 / 583 / 1559 | 2.35 / 966 / 2738 | **4.92 / 3469 / 9487** |
| adaptive (observed class) | 0.00 / 713 / 1766 | 2.87 / 841 / 2475 | **7.49 / 1956 / 6126** |

(per 100 000 within class)

Two findings deserve comment. First, **the latent-class policy already
targets**: without any baseline test, purely by inference from findings, it
gives the high-risk 5 % nearly twice the colonoscopies the fixed schedule
does (4.92 vs 2.49) and cuts their mortality by 37 % relative to the
10-yearly schedule (3469 vs 5535), while *reducing* the low-risk half's
colonoscopies from 2.58 to 1.97. Our earlier six-state analysis concluded
that such targeting required an observable score with AUC ≥ 0.8; that
conclusion was an artefact of a risk representation too coarse (two classes)
and of findings too coarsely categorised to identify it.

Second, with the class observed, targeting becomes extreme: the high-risk
5 % receive 7.49 colonoscopies and their mortality falls to 1956 per
100 000 — 65 % below the 10-yearly schedule, and 45 % below even the
5-yearly schedule which uses three times the total volume — while the
lowest-risk half is not screened at all. That last decision is a property of
the shadow price, not a recommendation: at λ = 0.001561 a colonoscopy in
class 0 does not pay for itself, and the arm's low-risk mortality is
identical to no screening (713 vs 712 per 100 000). A programme unwilling to
withhold screening entirely can simply solve at a smaller λ for that class;
the frontier is continuous.

### 3.6 A baseline risk score of realistic discrimination

Section 3.5 gives the risk class away exactly. No instrument does that, so
we repeated the exercise with a baseline score of finite discrimination: S =
log(individual_risk) + sigma x N(0, 1), observed once at age 40, with sigma
calibrated to a target AUC for lifetime CRC (measured within sex, since sex
is already observed by the policy). The belief is conditioned on the score
VALUE rather than on a score band - in closed form over CMOST's own
476-value risk pool, with each atom's age-40 clinical distribution taken
from the never-screened cohort, so the score is not assumed independent of
the clinical state given the class. Banding first would have been
self-defeating: at sigma = 0 even a top decile still mixes half a class, so
a "perfect" score would have looked less informative than the perfect-class
arm.

Two properties of this construction are worth stating because they bound
what the scenario can show. First, the score is a classically unbiased,
perfectly calibrated measurement of the model's own polyp-rate multiplier -
the most favourable structure any real instrument could have. Second, its
discrimination has a ceiling: because CRC remains stochastic given the
multiplier, even sigma = 0 attains only AUC 0.751 on lifetime CRC. No
instrument can do better under this model, so we make no claim about AUC-0.8
scores. We report each level by the risk gradient it induces as well as by
AUC, because AUC is rank-invariant and hides the tail magnitude that
targeting actually exploits (Table 7); at AUC 0.60 the top score decile
carries 1.76 times the population's lifetime CRC risk and the top-to-bottom
decile ratio is 3.2, which is what published CRC polygenic risk scores
achieve.

**Table 7.** Score levels: discrimination and the risk gradient each induces
(lifetime CRC relative risk by score decile), with the value of the score at a
matched colonoscopy volume. Every level is solved on its own price-volume
frontier and read at 2.369 colonoscopies per person, the latent-class
policy's volume, so the levels differ in information and not in budget. The
control's sigma = 60 is a numerical stand-in for sigma -> infinity, so a
trace of signal survives it: AUC 0.507 rather than 0.500, and a
top-to-bottom decile risk ratio of 1.1 rather than 1.0.

| level | sigma | AUC | RR top decile | top/bottom decile | CRC deaths /100k at matched volume | vs uninformative |
|---|---|---|---|---|---|---|
| uninformative (control) | 60 | 0.507 | 1.04 | 1.1 | 892 | — |
| family-history-like | 9.39 | 0.550 | 1.33 | 1.8 | 883 | −9 |
| **CRC polygenic score** | 4.49 | 0.599 | 1.76 | 3.2 | **867** | **−24** |
| PRS + lifestyle | 2.63 | 0.650 | 2.32 | 5.4 | 837 | −55 |
| optimistic upper reference | 1.52 | 0.700 | 3.08 | 8.6 | 787 | −105 |
| ceiling (risk known exactly) | 0 | 0.751 | 3.94 | 12.0 | 714 | −177 |

The uninformative control is the scenario's wiring check: with sigma large
the score carries nothing, and the machinery must reproduce the latent-class
policy. It does, to the resolution of the comparison. Scored on the same
sex-pooled age-40 belief at the same volume, the control gives 892 per 100
000 against 901–902 for the latent-class policy (the headline policy and
its 600-point-cap twin, which the score arms share; 880 when that policy is
scored on the model's own sex-specific age-40 prior rather than on the
sex-pooled one the score arms are built from); in the engine the control
gives 945.5 +- 13.8 at 2.02 colonoscopies per person against about 921 for
the latent-class frontier interpolated at that volume — a gap of 24 per
100 000, within one standard error of the frontier points (29–32) and 1.8
of the control's own. So the ladder above measures information and not
implementation.

The value of a realistic score is therefore real but modest: a CRC polygenic
score buys 24 fewer CRC deaths per 100 000 at equal colonoscopy volume,
about one seventh of what perfect knowledge of the latent risk would buy,
and about a third of what adaptivity itself buys. The engine cannot resolve
the individual steps of this ladder at 200 000 per arm: the minimum
detectable difference at two-sided 5 % is ~78 per 100 000 (1.96 times the
largest paired standard error against the adaptive arm), and the deployment prices were interpolated
rather than re-matched, so the arms do not sit at a common volume. Only the
two strongest levels separate from the no-score policy, and they do so
partly by spending more - AUC 0.70 uses 2.46 colonoscopies per person for
792.0 +- 28.6 and the ceiling 2.67 for 732.0 +- 15.2, against 2.29 and 889.5
+- 19.7 without a score. Read against the latent-class engine frontier at
those same volumes (871 and 836), the volume-matched engine gaps are about
-79 and -104 per 100 000: the same ordering as the in-model -105 and -177,
attenuated as transfers to the engine consistently are. The ladder's
inference is in-model, and the engine is consistent with its endpoints.

**Information and adaptivity are complements, not substitutes.** The obvious
alternative to a belief-tracking policy is a risk-tiered FIXED programme, so
we gave the same score to the fixed-schedule search: for each score band the
best of the same 2 112 candidate schedules was selected at a common price,
tuned to the same volume. The resulting 2 x 2 separates the two ingredients
(Table 8). A realistic score is worth 24-30 CRC deaths per 100 000 whether
the programme is fixed or adaptive, and adaptivity is worth 74-80 whether or
not a score is available: the two are close to additive, overlapping by only
about 5 per 100 000, and responding to findings is worth about three times a
realistic baseline score. That is the scenario's main message - one
colonoscopy showing three or more adenomas moves P(high-risk classes) from
the 0.05 prior to 0.93, whereas the top decile of an AUC-0.60 score reaches
only 0.14 and even its top 1 % only 0.23.

**Table 8.** Information versus adaptivity, in-model, all four cells read at
2.369 colonoscopies per person (CRC deaths per 100 000). Neither fixed cell
is a single attainable schedule at exactly that volume: one price selects
schedules in discrete steps, so each is interpolated on its own price-volume
envelope (no score, between 2.367 → 972 and 2.373 → 971; with score, between
2.343 → 946 and 2.379 → 940). Both fixed cells are built by the same
band-stratified machinery and differ only in what the bands know. The
exhaustive fixed search of §3.3 is not on the same footing, because it scores
schedules on each sex's own age-40 prior rather than on the sex-pooled belief
the score arms are built from; re-scored on that same belief, its best
*unisex* schedule gives 980 at this volume (interpolated between the best
schedules at the adjacent volumes; 976 when read on the same
price-selection envelope as the fixed cells of Table 8) and its best
score-blind but *sex-specific* programme 972 — the no-score cell to within
0.5, which is a second wiring check: with an uninformative score the bands carry nothing and
the fixed arm must collapse to a sex-specific fixed programme, and it does.

| | no score | score (AUC 0.60) | the score buys |
|---|---|---|---|
| best fixed schedule | 972 | 942 | −30 |
| adaptive policy | 892 | 867 | −24 |
| **adaptivity buys** | **−80** | **−74** | |

The score-stratified fixed programme is itself a reasonable clinical object,
and worth reporting. At AUC 0.60 it gives men in the bottom three score
deciles two colonoscopies (66/78, 65/75, 64/75), deciles 4-8 three
(58/68/78), the 80th-95th percentile four (56/64/72/80) and the top 5 % five
(48/56/64/72/80); women's bands carry their own lists, from a single
colonoscopy at 74 in the bottom decile to six (45/52/59/66/73/80) in the top
percentile. The adaptive policy at the same volume instead gives the bottom
decile 1.11 colonoscopies and the top 2 % 4.0-4.5, and it places them in
response to findings rather than by band alone.

In the engine this score-stratified programme is a far stronger comparator
than any unstratified schedule, and it is the honest limit of what we can
claim there. At n = 200 000 it reaches 896.0 CRC deaths per 100 000 with
2.262 colonoscopies per person, against 1034.0 at 2.577 for the 10-yearly
schedule (paired −138.0 ± 27.1) and 889.5 at 2.289 for the adaptive policy
(paired +6.5 ± 25.1). The engine therefore cannot separate "a score plus a
fixed schedule" from "adaptivity without a score" at this sample size,
although its interval comfortably contains the in-model gap of 50 per 100
000. What the engine does support is the weaker and more useful statement:
both routes to targeting beat an unstratified 10-yearly programme by roughly
140 CRC deaths per 100 000 while using fewer colonoscopies, and the two are
complementary rather than alternative.

### 3.7 Robustness to imperfect adherence

Fixed schedules assume attendance. We repeated the comparison with each
invitation attended with probability α, drawn independently per person and
invitation, for the fixed programme in two variants — a missed slot is lost,
or the invitation is repeated annually until attended — and for the adaptive
policy **unchanged** (a no-show yields no observation, so the policy simply
re-plans at the next annual decision; no re-solving per adherence level).

**Table 9.** Adherence scenarios (engine, n = 200 000 per arm; colonoscopies
per person → CRC deaths per 100 000).

| α | fixed 10-yearly (slot lost) | fixed 10-yearly + annual recall | adaptive policy |
|---|---|---|---|
| 1.0 | 2.58 → 1034 | — | 2.29 → **890** |
| 0.7 | 1.80 → 1246 | 2.56 → 1072 | 2.13 → **935** |
| 0.5 | 1.29 → 1352 | 2.54 → 1023 | 1.98 → **975** |
| 0.3 | 0.77 → 1558 | 2.44 → 1030 | 1.73 → 1067 |

The classic fixed programme decays from a 46 % to an 18 % mortality
reduction as α falls to 0.3. Annual recall restores the mortality but not
the efficiency: it spends the full colonoscopy volume regardless. The
adaptive policy holds a 44–53 % mortality reduction at every attendance
level while its volume falls with α; at α = 0.5 it reaches *lower* mortality
than the recall-augmented fixed programme (975 vs 1023) with 22 % fewer
colonoscopies (4.67 vs 3.44 deaths averted per 1000 colonoscopies, +36 %),
and at α = 0.3 comparable mortality with 29 % fewer colonoscopies (+35 %).
Re-planning is automatic because the belief, not a calendar, drives the
recommendation (Figure 4).

### 3.8 Life-years: an honest null

We also swept λ under an undiscounted life-year objective. In the model
these policies gain about 170–200 life-years per 1000 persons over the
guideline schedules (202 at 2.65 colonoscopies per person against the
10-yearly schedule, 173 at 4.84 against the 5-yearly), and about 110–130
over the fixed schedule with the most life-years within ±0.15
colonoscopies of the policy's volume, by shifting colonoscopies to younger
ages. The
advantage does not survive transfer to the engine: population-paired
differences are −11 ± 31 life-years per 1000 versus the 10-yearly schedule
(2.52 vs 2.58 colonoscopies per person) and −43 ± 11 versus the 5-yearly
schedule (4.67 vs 4.99). The
mortality-objective policies are, on life-years, statistically
indistinguishable from the fixed schedules (−1.7 ± 14.6 and +12.6 ± 14.1
per 1000 at n = 1 000 000) while dominating them on deaths and diagnoses
per colonoscopy.

We attribute this to the endpoint rather than to the policies. Life-years
depend on the *timing* of deaths, which the reduced model represents less
faithfully than event counts: post-diagnosis survival enters through
memoryless exit values, and competing mortality dominates the variance. The
endpoint is also numerically ill-conditioned for point-based methods —
policy differences of order 0.05 life-years ride on an absolute value of
39.7 — and solutions at more than ten colonoscopies per person were
demonstrably under-converged. This null reproduces the life-year
equivalence reported by our earlier six-state analysis, and we therefore
present life-years as a secondary, noise-limited endpoint and base the
primary claims on CRC deaths and diagnoses.

---

## 4. Discussion

We built a decision model of colorectal-cancer screening directly on the
CMOST microsimulation, estimated every one of its kernels from the
simulator's own quarter-resolved trajectories, solved it as a
mixed-observability Markov decision process, and evaluated the resulting
policies back inside the simulator with population-paired random numbers.
The adaptive policy prevents 30 % more CRC deaths and 25 % more CRC
diagnoses per colonoscopy than the uniform 10-year schedule, and 18 % and
16 % more than the uniform 5-year schedule — while using *fewer*
colonoscopies in both comparisons, so the result is a dominance rather than
a trade-off. It also beats fixed schedules re-optimised inside the same
model, which is the comparison that matters: the gain is attributable to
adaptivity, not to better calendar ages, since re-optimising the fixed ages
recovers only about two-fifths of it.

Against Zaika et al. (2024), the concrete comparator we set out to beat, the
difference is one of kind rather than degree. Their optimisation searches
the space of fixed schedules, optionally stratified by an assumed baseline
risk test; ours searches the space of *policies*, which contains the fixed
schedules as a degenerate subset and, in our exhaustive in-model search over
2 112 of them, strictly contains better elements at every volume. The
mechanism is visible in the solved policy: a clean exam is evidence, and the
policy converts it into a long interval; a multi-adenoma exam is stronger
evidence in the opposite direction, and it converts that into immediate
re-examination. Where fixed schedules must set one interval for a
population whose CRC risk spans a 23-fold range, the adaptive policy learns
each person's position in that range from data the programme is already
collecting and paying for.

Why this analysis reaches a positive conclusion where our own earlier,
six-state version reached a null is worth stating plainly, because it is a
methodological lesson rather than a change of data. Three differences each contributed. (i) The natural-history kernels were
estimated on a never-screened cohort and applied to screened people, which
underestimates post-polypectomy risk for two distinct reasons (residual
within-class risk and length-biased sampling of the unscreened
cross-section); combined with the pooled two-stage lesion axis, that is the
`pooled stages and no memory` row of Table 2, which under-states the benefit
of 10-yearly screening by 8.0 percentage points. (ii) In the engine-in-the-loop version of that earlier line of work the
policy was evaluated with *synthetic* observations drawn from the model's
own kernel rather than the engine's actual findings
(`tests/cmost_4way_eval.py`), so belief and reality drifted apart during
deployment; the six-state comparison itself filtered on real findings, but
through a two-category lesion alphabet. (iii) The optimised objective was a QALY surrogate
rather than the endpoint being reported. Of these, our ablation can only rule (i) out as the decisive difference for
the *policy*: as §3.3 shows, dropping the (τ, last finding) memory, pooling
the polyp stages, or removing the latent risk class altogether each still
yields a policy that beats fixed scheduling once it is deployed with real
findings and optimised for the reported endpoint — which leaves (ii) and
(iii) as the likely causes, though these runs cannot separate them. The abstraction's fidelity is what licenses the
*quantitative* claims — and abstractions of microsimulations should be
validated on the policy-relevant conditional distributions, not only on
marginal prevalence.

**Clinical reading.** The optimal policy is not exotic. It is a three-tier
surveillance rule with a sex offset and an age-dependent stopping rule, and a fixed rule of that form — colonoscopy at 52, then 10/5/3 years by
finding — captures part of the gain without any belief tracking: it closes
about a fifth of the adaptive policy's per-colonoscopy advantage over the
10-yearly schedule (3.53 vs 3.32 and 4.32 deaths averted per 1000
colonoscopies), while reaching a lower absolute mortality than either (866.5
per 100 000) at a 12 % higher colonoscopy volume. Where the
adaptive policy goes beyond guidelines is in the *length* of the interval it
grants a clean first exam (15–20 years rather than 10) and in its
willingness to re-examine multi-adenoma patients within one to two years.
Both are consequences of taking the information content of a colonoscopy
seriously in a population with strong latent risk heterogeneity. The same
logic explains why a baseline risk score adds relatively little on top
(§3.6): a single colonoscopy is a far stronger signal about a person's risk
class than any presently achievable baseline score, so the two are
complements in which the cheaper signal is also the weaker one.

**Limitations.** First, the model structure is justified by its predictive
accuracy rather than by necessity: the ablation of §3.3 shows that simpler
abstractions yield policies of similar engine performance at this shadow
price, so the six risk classes and eleven lesion states should be read as
what makes the reported numbers trustworthy, not as what makes adaptivity
work. Second, all results are internal to CMOST: the model
inherits the simulator's calibration, including an absolute adenoma
prevalence somewhat below US screening cohorts, and its synthetic
risk-multiplier distribution, whose 23-fold top-to-bottom gradient sets the
scale of the achievable personalisation gain. A simulator with weaker latent
heterogeneity would show a smaller gain; the qualitative mechanism would
remain. Third, the decision model is memoryless given (τ, last finding);
the sojourn structure of preclinical cancer is only approximated, which is
the likeliest source of the residual 5 % error at 5-yearly intensity and of
the life-year null. A semi-Markov or phase-type refinement is the natural
next step, and the same memorylessness applies to the post-diagnosis exit
values (§2.3). Fourth, our adherence model is missing-completely-at-random;
informative non-attendance (people who skip screening being systematically
higher- or lower-risk) would change the size, though probably not the sign,
of the adaptive advantage. Fifth, we optimise events per colonoscopy, not
cost-effectiveness; costs are recorded but a full ICER analysis, and the
disutility of the surveillance burden the policy imposes on high-risk
patients, remain future work. Finally, the score arm of §3.6 assumes a classically unbiased, perfectly
calibrated measurement of the model's own risk multiplier - the most
favourable structure a real instrument could have - so a score correlated
with risk through a different mechanism, or one that transports poorly
across ancestries, would buy less than Table 7 implies at the same nominal
AUC.

**Reproducibility.** The complete pipeline — cohort simulation, kernel estimation, solver, exhaustive fixed-schedule search, λ sweeps, engine evaluation, ablation and reporting — is in `dp/`, with the engine-side instrumentation (quarterly state recorder and findings-carrying policy hook) in `cmost_engine/NumberCrunching_policy.py`; every step is cached and resumable, and the whole analysis is driven by the commands documented in `docs/DP_PLAN.md`. Every table in this paper has a generating script: `dp/paired_tables.py` (Tables 1, 4, 5, 6 and the life-year and complication contrasts), `dp/surveillance_arms.py` (Table 5b), `dp/gap_table.py` (optimality gaps), `dp/kernel_support.py` (kernel back-off usage), `dp/robustness.py` and `dp/tau_sensitivity.py` (solver and τ-support sensitivity), and `dp/ablate.py` (Table 2); `tests/test_engine_hook_regression.py` checks the engine instrumentation. The per-chunk engine output (`results/dp/runs/`, 0.55 GB) and the solved α-vector sets (`results/dp/policies/`, 10 GB) are excluded from the repository for size; a manifest of their sizes and checksums (`results/dp/manifest.json`) is committed, and both are regenerated by the pipeline commands.

---

## References

1. Prakash MK, Lang B, Heinrich H, et al. CMOST: an open-source framework
   for the microsimulation of colorectal cancer screening strategies.
   *BMC Medical Informatics and Decision Making* 2017;17(1):80.
2. Zaika V, et al. Optimal timing of a colonoscopy screening schedule
   depends on adenoma detection, adenoma risk, adherence to screening and
   the screening objective: a microsimulation study. *PLoS ONE*
   2024;19(5):e0304374.
3. Gupta S, Lieberman D, Anderson JC, et al. Recommendations for follow-up
   after colonoscopy and polypectomy: a consensus update by the US
   Multi-Society Task Force on Colorectal Cancer. *Gastroenterology* 2020.
4. US Preventive Services Task Force. Screening for colorectal cancer: US
   Preventive Services Task Force recommendation statement. *JAMA* 2021.
5. Knudsen AB, Rutter CM, Peterse EFP, et al. Colorectal cancer screening:
   an updated modeling study for the US Preventive Services Task Force.
   *JAMA* 2021.
6. Rutter CM, Savarino JE. An evidence-based microsimulation model for
   colorectal cancer: validation and application (CRC-SPIN).
   *Cancer Epidemiology, Biomarkers & Prevention* 2010.
7. Krijkamp EM, Alarid-Escudero F, Enns EA, Jalal HJ, Hunink MGM,
   Pechlivanoglou P. Microsimulation modeling for health decision sciences
   using R: a tutorial. *Medical Decision Making* 2018.
8. Ong SCW, Png SW, Hsu D, Lee WS. Planning under uncertainty for robotic
   tasks with mixed observability. *International Journal of Robotics
   Research* 2010;29(8):1053–1068.
9. Pineau J, Gordon G, Thrun S. Point-based value iteration: an anytime
   algorithm for POMDPs. *IJCAI* 2003.
10. Walraven E, Spaan MTJ. Point-based value iteration for finite-horizon
    POMDPs. *Journal of Artificial Intelligence Research* 2019;65:307–341.
11. Hauskrecht M. Value-function approximations for partially observable
    Markov decision processes. *Journal of Artificial Intelligence Research*
    2000;13:33–94.
12. Erenay FS, Alagoz O, Said A. Optimizing colonoscopy screening for
    colorectal cancer prevention and surveillance. *Manufacturing & Service
    Operations Management* 2014;16(3):381–400.
13. Ayer T, Alagoz O, Stout NK. A POMDP approach to personalize mammography
    screening decisions. *Operations Research* 2012;60(5):1019–1034.
14. Smallwood RD, Sondik EJ. The optimal control of partially observable
    Markov processes over a finite horizon. *Operations Research*
    1973;21(5):1071–1088.
15. Pineau J, Gordon G, Thrun S. Anytime point-based approximations for
    large POMDPs. *Journal of Artificial Intelligence Research*
    2006;27:335–380.
16. Shani G, Pineau J, Kaplow R. A survey of point-based POMDP solvers.
    *Autonomous Agents and Multi-Agent Systems* 2013;27(1):1–51.
17. Smith T, Simmons R. Heuristic search value iteration for POMDPs.
    *Proceedings of the 20th Conference on Uncertainty in Artificial
    Intelligence (UAI)* 2004:520–527.
18. Kurniawati H, Hsu D, Lee WS. SARSOP: efficient point-based POMDP
    planning by approximating optimally reachable belief spaces.
    *Robotics: Science and Systems* 2008.
19. Steimle LN, Denton BT. Markov decision processes for screening and
    treatment of chronic diseases. In: Boucherie RJ, van Dijk NM, eds.
    *Markov Decision Processes in Practice*. Springer 2017:189–222.

---

## Figures

| figure | file | content |
|---|---|---|
| 1 | `figures/dp_frontier_c6b.png` | Efficiency frontiers in the engine: CRC deaths and CRC diagnoses versus colonoscopies per person, for the adaptive policy family (both objectives), the guideline schedules, the best fixed schedules, the 10-yearly and 5-yearly programmes augmented with CMOST's post-polypectomy surveillance rule, and the in-model fixed-schedule frontier. |
| 2 | `figures/dp_validation_c6b.png` | Model vs engine decision-time prevalence of early adenoma, advanced adenoma and undetected cancer by age in men, under 10-yearly screening (left) and no screening (right). |
| 3 | `figures/dp_policy_c6b_lam0.001561.png` | Screening ages of the headline policy along canonical observation paths and by known risk class, by sex. |
| 4 | `figures/dp_riskscore_c6b.png` | Value of a baseline risk score against its discrimination, at matched colonoscopy volume (left), and the engine arms on the efficiency plane (right), including the score-stratified fixed programme, which the engine cannot separate from the adaptive no-score policy at this sample size. |
| 5 | `figures/dp_adherence_c6b.png` | CRC mortality and colonoscopy use versus attendance probability for the fixed programme (with and without recall) and the adaptive policy. |
| 6 | `figures/dp_per_colonoscopy_c6b.png` | Deaths and diagnoses averted per 1000 colonoscopies, all headline arms and the surveillance-augmented programmes. |
