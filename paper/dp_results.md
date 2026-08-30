# Results (new DP pipeline)

All evaluations are inside the real CMOST engine (`NumberCrunching_policy`),
population-paired across arms (identical chunk seeds), colonoscopy counts =
policy-initiated colonoscopies at ages 40–80, outcomes measured to age 100.

## Baseline (n = 1,000,000 per arm)

| arm | colos/person | CRC deaths /100k | CRC diagnoses /100k | deaths averted /1000 colos | dx averted /1000 colos |
|---|---|---|---|---|---|
| no screening | 0 | 1888 ± 15 | 4545 ± 18 | — | — |
| 10-yearly (50/60/70) | 2.575 | 1032 ± 9 | 2687 ± 13 | 332 | 722 |
| 5-yearly (50–75) | 4.986 | 775 ± 9 | 2208 ± 13 | 223 | 469 |

(CRC deaths counted at ages >= 40, the decision window; diagnoses at any age.)

## Adaptive DP frontier (death objective; engine, n = 200,000 per λ)

| λ (deaths/colonoscopy) | colos | deaths /100k | dx /100k | deaths averted /1000c | dx averted /1000c |
|---|---|---|---|---|---|
| 0.0004 | 6.11 | 621 | 1984 | 209 | 421 |
| 0.000525 | 5.27 | 630 | 1923 | 241 | 500 |
| 0.00069 | 4.58 | 669 | 1998 | 268 | 558 |
| 0.000905 | 3.72 | 751 | 2174 | 308 | 640 |
| 0.001189 | 2.88 | 801 | 2286 | 382 | 789 |
| 0.001561 | 2.28 | 901 | 2475 | 437 | 912 |
| 0.00205 | 1.96 | 926 | 2576 | 496 | 1009 |
| 0.002691 | 1.81 | 956 | 2583 | 521 | 1091 |
| 0.003534 | 1.08 | 1150 | 2977 | 692 | 1459 |

## Matched-volume comparison (interpolated on the engine frontier)

| comparator | volume | comparator deaths | policy deaths | Δ | comparator dx | policy dx | Δ |
|---|---|---|---|---|---|---|---|
| q10y | 2.577 | 1034 ± 30* | **851 ± 33** | **−18 %** | 2700 ± 27 | **2381 ± 70** | **−12 %** |
| q5y | 4.989 | 801 ± 11* | **645 ± 12** | **−19 %** | 2227 ± 35 | **1953 ± 26** | **−12 %** |

*200k-arm values (paired with the λ grid); 1M values are 1032 / 775.

Per-colonoscopy efficiency at matched volume (deaths averted per 1000
colonoscopies): 406 vs 335 (q10y volume, **+21 %**) and 251 vs 220 (q5y
volume, **+14 %**); diagnoses averted per 1000 colonoscopies: 843 vs 719
(**+17 %**) and 521 vs 466 (**+12 %**).

Dominance points (no interpolation needed):
* λ = 0.001189: **2.88 colonoscopies → 801 deaths** — the same mortality as
  the 5-yearly schedule with **42 % fewer colonoscopies** (2.88 vs 4.99), and
  vs the 10-yearly schedule +0.30 colonoscopies for −233 deaths/100k.
* λ = 0.001561: 2.28 colonoscopies (0.30 fewer than q10y) → 901 deaths
  (−133/100k vs q10y) and 2475 dx (−225/100k): strictly dominates q10y.
* λ = 0.000525: 5.27 colonoscopies (+0.28 vs q5y) → 630 deaths (−171) and
  1923 dx (−304).

The incidence-objective frontier is nearly identical (both objectives are
served by the same prevention-oriented policies); the exhaustive in-model
search over 2112 fixed schedules shows the best FIXED schedule at every
volume also lies above the adaptive frontier (Figure dp_frontier).

## Policy structure (λ = 0.001561, the q10y-volume-matched policy)

Screening ages along canonical observation paths (male / female):

| path | male | female |
|---|---|---|
| first colonoscopy | 49 | 58 |
| all findings normal | 49, 69, 78 | 58, 73 |
| adenoma (1–2) at first, then normal | 49, 58, 67, 75 | 58, 63, 73, 80 |
| ≥3 adenomas at first, then normal | 49, 53, 61, 67, 75 | 58, 60, 65, 71, 76, 80 |
| advanced adenoma at first, then normal | 49, 57, 67, 75 | 58, 63, 73, 80 |
| risk class known: lowest (50 %) | 49, 78 | 58, 72 |
| risk class known: highest (2 %) | 49, 53, 61, 67, 73, 78, 80 | 58, 66, 73, 77, 79 |

The policy reproduces — and quantitatively sharpens — the logic of
post-polypectomy surveillance guidelines: a normal colonoscopy earns a long
interval (~20 years, or exit from screening for the low-risk majority),
1–2 small adenomas a ~9-year interval, ≥3 adenomas an early re-examination
(3–4 years first, then ~6–8), advanced adenomas ~8 years, with all intervals
shortening at higher ages and for men.

## Headline confirmation (n = 1,000,000 per arm, paired chunk seeds)

| arm | colos/person | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000c | dx averted /1000c |
|---|---|---|---|---|---|
| no screening | 0.000 | 1887.6 (15.1) | 4545.1 (17.9) | — | — |
| fixed 10-y (50/60/70) | 2.575 | 1032.1 (9.1) | 2686.9 (13.2) | 3.32 | 7.22 |
| fixed 5-y (50–75) | 4.986 | 775.3 (8.8) | 2208.0 (13.4) | 2.23 | 4.69 |
| best fixed at q10y volume [54/64/74] | 2.441 | 980.9 (9.0) | 2641.2 (17.5) | 3.71 | 7.80 |
| best fixed at q5y volume [48–78 q6y] | 4.879 | 724.5 (9.7) | 2140.5 (11.7) | 2.38 | 4.93 |
| finding rule 52 start /10/5/3/3 | 2.890 | 866.5 (10.2) | 2344.9 (15.2) | 3.53 | 7.61 |
| **DP policy λ=0.001561** | **2.289** | **899.8 (7.6)** | **2486.4 (17.5)** | **4.32** | **9.00** |
| **DP policy λ=0.00069** | **4.586** | **682.5 (6.6)** | **2056.9 (13.7)** | **2.63** | **5.43** |
| DP policy λ=0.000525 | 5.270 | 668.9 (10.4) | 2012.4 (14.1) | 2.31 | 4.81 |
| DP (inc objective) λ=0.005125 | 2.164 | 932.4 (12.9) | 2516.7 (13.9) | 4.42 | 9.38 |
| DP (inc objective) λ=0.001724 | 5.149 | 668.4 (8.6) | 1977.6 (15.3) | 2.37 | 4.99 |
| **DP + observed risk class, λ=0.001561** | **1.665** | **832.9 (6.6)** | **2303.3 (14.6)** | **6.34** | **13.47** |

Population-paired differences (identical chunk seeds; mean ± SE per 100k):

| comparison | Δ colonoscopies | Δ CRC deaths | Δ CRC diagnoses |
|---|---|---|---|
| DP λ=0.001561 − fixed 10-y | **−0.29** | **−132.3 ± 12.3** | **−200.5 ± 22.9** |
| DP λ=0.001561 − best fixed [54/64/74] | −0.15 | **−81.1 ± 9.2** | −154.8 ± 21.0 |
| DP λ=0.00069 − fixed 5-y | **−0.40** | **−92.8 ± 10.4** | **−151.1 ± 13.8** |
| DP λ=0.00069 − best fixed [48–78 q6y] | −0.29 | **−42.0 ± 9.7** | −83.6 ± 16.0 |
| DP obsclass − fixed 10-y | −0.91 | **−199.2 ± 12.1** | −383.6 ± 17.9 |
| finding rule − fixed 10-y | +0.32 | −165.6 ± 13.1 | −342.0 ± 20.0 |

**Primary claim.** At 1,000,000 persons per arm the adaptive DP policy
*strictly dominates* both guideline comparators and the exhaustively
searched best fixed schedules: with 11 % fewer colonoscopies than the
10-yearly schedule it leaves CRC mortality 12.8 % lower and CRC incidence
7.5 % lower (deaths averted per 1000 colonoscopies 4.32 vs 3.32, **+30 %**;
diagnoses averted per 1000 colonoscopies 9.00 vs 7.22, **+25 %**); with 8 %
fewer colonoscopies than the 5-yearly schedule mortality is 12.0 % lower
(2.63 vs 2.23 deaths averted per 1000 colonoscopies, **+18 %**; diagnoses
+16 %). Every difference exceeds 4 standard errors of the paired contrast.

**Secondary findings.** (i) A transparent finding-based rule distilled from
the policy's structure (first colonoscopy at 52; next interval 10 y after a
normal exam, 5 y after 1–2 adenomas, 3 y after ≥3 or advanced adenomas)
already captures much of the gain — a practice-ready summary of what the DP
policy does. (ii) If the risk class is observable at baseline (a
PRS-like scenario), the same λ yields 833 deaths/100k using only 1.67
colonoscopies per person — twice the per-colonoscopy efficiency of the
10-yearly schedule (6.34 vs 3.32). (iii) Colonoscopy-complication deaths
are lower under the DP policies than under the volume-matched fixed
schedules (9.3 vs 9.9 per 100k at the q10y volume; 15.7 vs 19.4 at the q5y
volume), so the mortality gain is not bought with procedural harm.

## Additional analyses (2026-08-24 evening)

### Robustness to imperfect adherence (engine, n = 200,000 per arm)

Each due invitation is attended with probability α (independent draws). The
SAME DP policy as in the headline (λ = 0.001561) is deployed unchanged — a
no-show simply yields no observation and the policy re-plans at the next
annual decision. Comparators: the fixed 10-y programme where a missed slot
is lost, and the fixed 10-y programme with annual re-invitation until
attended (recall).

| α | fixed 10-y (slot lost) | fixed 10-y + recall | adaptive DP policy |
|---|---|---|---|
| 1.0 | 2.58 colos → 1034 deaths | — | 2.29 → **890** |
| 0.7 | 1.80 → 1246 | 2.56 → 1072 | 2.13 → **935** |
| 0.5 | 1.29 → 1352 | 2.54 → 1023 | 1.98 → **975** |
| 0.3 | 0.77 → 1558 | 2.44 → 1030 | 1.73 → 1067 |

The classic fixed programme degrades from a 45 % to an 18 % mortality
reduction as α falls to 0.3. Annual recall restores the mortality but at an
undiminished colonoscopy volume. The adaptive policy holds a 44–53 %
mortality reduction at EVERY adherence level while its volume falls with α:
at α = 0.5 it achieves lower mortality than the recall-augmented fixed
programme (975 vs 1023) with 22 % fewer colonoscopies (deaths averted per
1000 colonoscopies 4.67 vs 3.44, +36 %), and at α = 0.3 comparable
mortality with 29 % fewer colonoscopies (4.82 vs 3.56 per 1000, +35 %).
Re-planning around no-shows is automatic — no re-solving per adherence
level. (Figure dp_adherence_c6b.)

### Life-year objective (engine, n = 200,000 per λ)

A 12-point λ sweep with the undiscounted life-year objective produces
policies that, in the model, gain ~150–200 life-years per 1000 persons over
the fixed schedules at matched volume (by shifting colonoscopies to younger
ages). This advantage does NOT transfer to the engine: population-paired
engine differences are −11 ± 31 LYG/1000 vs the 10-yearly schedule at
matched volume and −43 ± 11 vs the 5-yearly schedule; the death-objective
policies are statistically indistinguishable from the fixed schedules on
life-years (−2 ± 15 and +13 ± 14 LYG/1000 at 1M) while dominating them on
deaths and diagnoses per colonoscopy. Life-years depend on the *timing* of
deaths, which the reduced model (memoryless post-diagnosis survival, exit
values) represents less faithfully than event counts; we therefore report
life-years as a secondary, noise-limited endpoint — consistent with the
LYG-equivalence finding of the earlier 6-state analysis — and base the
primary claims on CRC deaths and diagnoses. (At the low-λ extreme the
life-year objective is also numerically ill-conditioned for point-based
value iteration: policy differences of ~0.05 LY ride on an absolute value
of ~39.7 LY, and solutions at 10+ colonoscopies/person were demonstrably
under-converged; the 2.5–7 colonoscopy range shown is converged.)


### Model-structure ablation (`results/dp/ablation.json`)

Each abstraction re-estimated from the same two engine cohorts, then asked
to predict the engine's outcomes under the guideline schedules (reduction
versus no screening; engine truth in the first row):

| model structure | 10-y death | 10-y inc | 5-y death | 5-y inc |
|---|---|---|---|---|
| **engine** | **45.6 %** | **41.7 %** | **59.3 %** | **52.4 %** |
| 11 states, memory, 6 classes (used) | 46.2 % | 42.2 % | 61.2 % | 54.1 % |
| polyp stages pooled | 46.0 % | 42.0 % | 59.4 % | 52.1 % |
| no (tau, last finding) memory | 54.5 % | 50.6 % | 65.1 % | 58.1 % |
| pooled *and* no memory | 37.6 % | 31.6 % | 51.7 % | 41.7 % |
| 3 risk classes | 45.6 % | 41.5 % | 62.7 % | 55.7 % |
| 1 risk class | 45.2 % | 41.1 % | 63.7 % | 56.7 % |

The memory is the load-bearing part, and its omission errs with a sign that
depends on the lesion axis (+8.9 pts stage-resolved, -8.0 pts pooled). Note
that the pooled+memoryless row shares only two coarsenings with the earlier
six-state analysis (pooled lesion axis, no memory); it keeps six risk
classes and four undetected cancer stages.

**Policies solved on the ablated models still beat the fixed schedule**
(engine, n = 200 000 per arm, `results/dp/eval_ablation_policy_n200000.json`):

| policy solved on | colos | deaths /100k | deaths averted /1000 colos | paired vs q10y |
|---|---|---|---|---|
| q10y (comparator) | 2.577 | 1034.0 | 3.35 | — |
| full model | 2.289 | 889.5 | 4.41 | −144.5 ± 47.9 |
| pooled stages | 2.098 | 909.0 | 4.71 | −125.0 ± 38.8 |
| no memory | 2.067 | 913.5 | 4.76 | −120.5 ± 38.5 |
| 1 risk class | 2.461 | 862.0 | 4.21 | −172.0 ± 37.8 |

So the model structure buys *predictive accuracy*, while the per-colonoscopy
dominance over fixed scheduling is robust to substantial misspecification.
(Three of the five ablated abstractions were re-solved; paired SEs are
38-39 over four matched 50k chunks. The full model's row is shown for
reference and has SE 47.9 at this sample size.)


## Baseline risk score of finite discrimination (`results/dp/score_frontier.json`,
`eval_riskscore_n200000.json`, `eval_scorefixed_0.60_n200000.json`)

Score: S = log(individual_risk) + sigma*N(0,1) observed at 40; the belief is
conditioned on the score VALUE (closed form over CMOST's 476-value pool, with
each atom's own age-40 clinical distribution), not on a band. sigma is
calibrated to a within-sex AUC for lifetime CRC. Ceiling = 0.751 (knowing the
multiplier exactly still leaves the disease stochastic).

In-model, every level read at 2.369 colonoscopies/person:

| level | sigma | AUC | RR top decile | top/bottom | deaths /100k | vs uninformative |
|---|---|---|---|---|---|---|
| uninformative (control) | 60 | 0.507 | 1.04 | 1.1 | 892 | — |
| family-history-like | 9.39 | 0.550 | 1.33 | 1.8 | 883 | −9 |
| **CRC PRS** | 4.49 | 0.599 | 1.76 | 3.2 | **867** | **−24** |
| PRS + lifestyle | 2.63 | 0.650 | 2.32 | 5.4 | 837 | −55 |
| optimistic | 1.52 | 0.700 | 3.08 | 8.6 | 787 | −105 |
| ceiling | 0 | 0.751 | 3.94 | 12.0 | 714 | −177 |

Engine (n = 200 000 per arm): q10y 2.577/1034.0; adaptive no score 2.289/889.5;
score AUC 0.55 2.275/919.5; 0.60 2.298/904.0; 0.65 2.256/869.0; 0.70
2.457/792.0; ceiling 2.665/732.0; uninformative control 2.023/945.5; perfect
class 1.665/826.0. These engine arms are NOT volume-matched: the deployment
price came from an interpolated probe, not from a re-matched frontier, so the
arms range over 2.02-2.67 colonoscopies/person. Only the two strongest levels
separate from the no-score policy at this sample size (MDD ~78 per 100 000),
and part of that separation is extra volume. Read against the latent-class
engine frontier interpolated at each arm's own volume (871 at 2.457, 836 at
2.665), the volume-matched engine gaps are -79 (AUC 0.70) and -104 (ceiling),
against the in-model -105 and -177.

**Information vs adaptivity (in-model, all four cells read at 2.369
colos/person, CRC deaths/100k):**

| | no score | score AUC 0.60 | score buys |
|---|---|---|---|
| best fixed schedule | 972 | 942 | −30 |
| adaptive policy | 892 | 867 | −24 |
| **adaptivity buys** | **−80** | **−74** | |

Both fixed cells come from the same band-stratified machinery (the no-score
one with sigma = CONTROL_SIGMA), so they differ only in what the bands know.
Neither is a single attainable schedule at exactly 2.369: one price selects
schedules in discrete steps, so each is interpolated on its own price-volume
envelope (no score, 2.367 -> 972 and 2.373 -> 971; with score, 2.343 -> 946
and 2.379 -> 940). The exhaustive fixed search is NOT on the same footing: it scores schedules
on each sex's own age-40 prior, not the sex-pooled belief the score arms use.
Re-scored on the pooled belief, the best unisex schedule gives 980.0 at this
volume ([58,68,78] 2.352/980.4, [58,68,76] 2.402/979.2) and the best
score-blind SEX-SPECIFIC programme 972.1 - the no-score cell (971.6) to
within 0.5, i.e. a second wiring check: an uninformative score collapses the
fixed arm to a sex-specific fixed programme. On its own (sex-specific) prior
the same unisex search gives 970.3.

Engine, score-stratified fixed (sex-specific band schedules): 2.262 colos /
896.0 deaths/100k, paired −138.0 ± 27.1 vs q10y and +6.5 ± 25.1 vs the
adaptive no-score policy (2.289/889.5). At n = 200 000 the engine cannot
separate "score + fixed" from "adaptive, no score", though the interval
contains the in-model gap of 50.

Nearly additive: a realistic score is worth 24-30 deaths/100k whether the
programme is fixed or adaptive, and adaptivity 74-80 either way, the two
overlapping by about 5. One colonoscopy showing >=3 adenomas moves
P(high-risk classes) from 0.05 to 0.93; the top DECILE of an AUC-0.60 score
reaches 0.14 and its top 1 % only 0.23.

Wiring control: with sigma large the score carries nothing and the machinery
reproduces the latent-class policy to the resolution of the comparison - 892
for the control against 902 for the latent-class policy when both are scored
on the same sex-pooled age-40 belief at the same volume (880 for that policy
on the model's own sex-specific prior), and 945.5 +- 13.8 at 2.023 in the
engine against ~921 for the latent frontier interpolated at that volume.
Belief conservation (sum_k w_k b_k = population prior) holds to <0.0011 on the
14 deployment roots and to <0.0001 on the 2048-cell table, at every level.
