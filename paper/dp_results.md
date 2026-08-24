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
10-yearly schedule it prevents 13 % more CRC deaths and 7 % more diagnoses
(deaths averted per 1000 colonoscopies 4.32 vs 3.32, **+30 %**; diagnoses
averted per 1000 colonoscopies 9.00 vs 7.22, **+25 %**); with 8 % fewer
colonoscopies than the 5-yearly schedule it prevents 12 % more deaths
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
