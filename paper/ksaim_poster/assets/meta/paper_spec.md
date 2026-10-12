# Adaptive Colonoscopy Screening from a Mixed-Observability Markov Decision Process Built on the CMOST Microsimulation
**Authors:** Taehyung Lee¹, Inseo Choi², Jaehoon Shim³
**Institutes:** 1 University of Ulsan College of Medicine; 2 Chosun University College of Medicine; 3 Seoul National University
**Corresponding author:** none
**Venue:** KoSAIM 2026 Annual Meeting (October 23-24, 2026, Seoul)
**Code:** https://github.com/daybreaker95/pbvi_final
**title_audio_script:** Adaptive Colonoscopy Screening from a Mixed-Observability Markov Decision Process Built on the CMOST Microsimulation, by Taehyung Lee, Inseo Choi and Jaehoon Shim. Colonoscopy capacity is limited, and every fixed screening schedule throws away the information a person's own colonoscopies generate. We built a decision model directly on the CMOST microsimulation, solved it for finding-adaptive policies, and verified inside the same simulator that they prevent more cancer deaths per colonoscopy than any fixed schedule.

## Problem
**Necessary:** Colonoscopy prevents colorectal cancer (CRC), but it is scarce and invasive, so what matters is benefit per colonoscopy. Microsimulation studies optimize a fixed schedule — the same ages for everyone, or for everyone in a baseline risk stratum — and so cannot use what a person's previous colonoscopies found.
**Additional:** Clinical practice is already adaptive — a normal exam earns ten years; ≥3 adenomas or an advanced adenoma trigger surveillance at 3–5 years — yet simulator-optimized schedules, including the CMOST-based search of Zaika et al. (2024), ignore it. Can finding-adaptive timing beat the best fixed schedule at equal or lower volume?
**Widget hint:** vs — left: "Fixed schedule: same ages for everyone, findings ignored"; right: "Adaptive policy: next interval set by what the last exam found"
**Audio script:** Colonoscopy is an effective colorectal cancer screening tool, but it is invasive and capacity is finite, so the question is not only how much cancer a programme prevents but how much it prevents per colonoscopy. Microsimulation studies, including the CMOST optimisation of Zaika and colleagues, search over fixed schedules, the same ages for everyone or for everyone in a baseline risk stratum, and so cannot act on what a person's own previous colonoscopies found. We asked whether adapting to findings can beat the best fixed schedule.

## Motivation
**Necessary:** After a first colonoscopy at 52, ≥3 adenomas raise the probability of a high-risk class from 0.05 to 0.93 (men) / 0.98 (women); a normal exam lowers it to 0.007 / 0.017. Hidden risk spans a 23-fold gradient in lifetime CRC incidence (top 2% vs bottom 50%); no single fixed schedule can express that.
**Additional:** Six latent risk classes (cut at the 50th, 80th, 95th, 96.5th and 98th percentiles of CMOST's individual_risk) span lifetime CRC incidence from 1.78% (bottom half) to 41.61% (top 2%). The class is never observed; the policy infers it from findings.
**Widget hint:** stat-strip — "0.93 | P(high-risk class) after ≥3 adenomas at a first exam at 52, men (prior 0.05)" ;; "0.007 | after a normal first exam at 52, men (prior 0.05)" ;; "23× | lifetime CRC incidence, top 2% vs bottom 50%"
**Audio script:** Why should adapting to findings matter? Because a single colonoscopy is a very strong signal about a person's hidden risk. In our model, finding three or more adenomas at a first colonoscopy at fifty-two raises the probability of a high-risk class from five percent to ninety-three percent in men and ninety-eight in women, while a normal exam drops it below two percent. And hidden risk spans a twenty-three-fold gradient in lifetime cancer incidence, which no single fixed schedule can express.

## Contribution
**Necessary:**
- A decision model estimated from the simulator itself: every kernel is a maximum-likelihood fit to CMOST trajectories.
- A mixed-observability formulation with the right memory: (years since last colonoscopy, last finding) plus six latent risk classes.
- An engine-verified dominance result — fewer CRC deaths and diagnoses at lower colonoscopy volume: 30% more CRC deaths and 25% more diagnoses averted per colonoscopy than 10-yearly screening.
**Additional:** none
**Audio script:** We make three contributions. First, a decision model whose every transition, observation and exit kernel is estimated by maximum likelihood from the CMOST engine itself, not from summary statistics. Second, a mixed-observability formulation that conditions on years since the last colonoscopy and its finding, together with six latent risk classes. Third, an engine-verified dominance result: thirty percent more deaths and twenty-five percent more diagnoses averted per colonoscopy than ten-yearly screening, at lower colonoscopy volume.

## Method
**Necessary:**
- Mixed-observability Markov decision process (MOMDP). Observed: age, sex, τ = years since the last colonoscopy, and its finding (normal / 1–2 early adenomas / ≥3 early adenomas / advanced adenoma). Hidden, tracked as a belief: 6 latent risk classes × 11 clinical states. Each year from age 40 to 80 the policy chooses WAIT or SCREEN; outcomes are counted to age 100.
- Every kernel estimated by maximum likelihood from the instrumented CMOST engine: 2,000,000 never-screened + 2,000,000 randomly-screened lives, 4,520,648 colonoscopies, conditioned on (τ, last finding), hierarchical back-off: a cell stands on its own only with ≥150 observations, sparser cells borrow from wider age windows, pooled τ groups or pooled findings.
- Solved by finite-horizon point-based value iteration (backward induction over belief points). Sweeping the shadow price λ — the cost charged per colonoscopy — traces the efficiency frontier of CRC deaths vs colonoscopies. Comparators: 10-yearly, 5-yearly, the best of 2,112 exhaustively searched fixed schedules, and 10-yearly + CMOST surveillance — all deployed inside CMOST with the engine's real findings, 1,000,000 population-paired persons per arm.
**Additional:**
- Exits — CRC diagnosis at stage I–IV, death from other causes, fatal complication — are terminal; their remaining-lifetime consequences enter through estimated exit values, so the belief lives on a 66-dimensional simplex per observed key.
- Why condition on (τ, finding): for men of the second risk class at age 60, the annual chance that a stage-4 polyp becomes advanced is 0.18 in never-screened person-years but 0.21–0.31 after a colonoscopy.
- Every metric is linear in state occupancy, so each policy's exact in-model expected deaths, diagnoses and colonoscopies are computed by forward propagation of its belief tree — no Monte-Carlo noise, milliseconds per policy; engine results are separate Monte-Carlo runs with paired standard errors.
- Belief points per observed key capped at 600 (λ-grid, ablation) or 1,500 (headline); sweeps repeat until the exact in-model objective improves by less than 10⁻⁷ in two consecutive rounds; every α-vector is an executable plan, so the solved value is a lower bound.
**Key equation:** $$\max_{\pi}\; -\,\mathbb{E}_\pi[\text{CRC deaths}] \;-\; \lambda\,\mathbb{E}_\pi[\text{colonoscopies}]$$ where λ is the shadow price of a colonoscopy (undiscounted, ages 40–100); sweeping λ traces the whole efficiency frontier.
**Figure:** assets/figures/momdp_pipeline.png — From engine to policy and back: CMOST trajectories → maximum-likelihood kernels → MOMDP solved by point-based value iteration → policies deployed back inside CMOST.
**Audio script:** The observed state is age, sex, years since the last colonoscopy and its finding; the hidden state crosses six latent risk classes with eleven clinical states. Each year from forty to eighty the policy chooses to wait or to screen. We estimated every kernel from four million instrumented CMOST lives, solved the model by point-based value iteration, swept the shadow price lambda to trace an efficiency frontier, and deployed every policy back inside the real engine against fixed-schedule comparators.

## Dataset / Benchmark
**Necessary:**
- CMOST microsimulation (CMOST13 parameters): the adenoma–carcinoma sequence in 3-month steps from birth to age 100, with a lifelong per-person risk multiplier.
- Estimation: 2,000,000 never-screened + 2,000,000 randomly-screened lives (4,520,648 colonoscopies). Evaluation: 1,000,000 population-paired persons per arm, identical seeds.
**Additional:** CMOST is the open-source microsimulation of Prakash et al. (2017); the risk multiplier acts on adenoma initiation. In the randomly-screened cohort the first colonoscopy is uniform on ages 40–75, intervals uniform on 1–20 years, screening stops at 80; evaluation arms run as 20 chunks of 50,000 persons. Validation on the cohort alive and undiagnosed at age 40 (engine n = 965,258; the Key Result arms count all 1,000,000 persons; model → engine, CRC deaths per 100,000): no screening 1934 → 1943 (−0.5%); 10-yearly 1040 → 1058 (−1.7%); 5-yearly 750 → 792 (−5.3%). CRC diagnoses: +0.3%, −0.7%, −3.5%. Age-specific prevalence of early adenoma, advanced adenoma and undetected cancer also matches, including the saw-tooth depletion and regrowth around each screening age.
**Figure:** assets/figures/dp_validation_c6b.png — Model (dashed) vs engine (solid): prevalence of early adenoma, advanced adenoma and undetected cancer by age, men; 10-yearly screening (left), no screening (right).
**Audio script:** Our ground truth is CMOST, an open-source microsimulation of the adenoma-carcinoma sequence in three-month steps from birth to age one hundred. Two instrumented cohorts of two million lives each, one never screened and one screened at random ages and intervals, supplied four and a half million colonoscopies for estimation. The reduced model reproduces the engine's cancer deaths and diagnoses within half a percent without screening, within two percent under ten-yearly screening and within five point three percent under five-yearly screening.

## Key Result
**Necessary:** At 2.29 colonoscopies per person (11% fewer than 10-yearly screening's 2.58), CRC deaths fell from 1032.1 to 899.8 per 100,000 (−132.3 ± 12.3) and diagnoses from 2686.9 to 2486.4 (−200.5 ± 22.9). Deaths averted per 1,000 colonoscopies rose from 3.32 to 4.32 (+30%), diagnoses averted from 7.22 to 9.00 (+25%). The policy also beat the best re-optimized fixed schedule at lower volume (−81.1 ± 9.2 deaths per 100,000). At the 5-yearly volume: 682.5 vs 775.3 deaths (−92.8 ± 10.4), +18% deaths averted per colonoscopy. The comparator closest to guideline practice — 10-yearly + CMOST post-polypectomy surveillance — reaches 879.8 deaths per 100,000, 20 ± 12 fewer than the headline policy but with 29% more colonoscopies (2.95 per person); the adaptive policy solved for that volume (2.87 colonoscopies) has 62 ± 12 fewer deaths, and per colonoscopy the ordering is unchanged (3.42 vs 4.32).
**Table:** Engine evaluation, n = 1,000,000 per arm, population-paired seeds.
| arm | colonoscopies / person | CRC deaths /100,000 (SE) | deaths averted /1,000 colonoscopies |
|---|---|---|---|
| fixed 10-yearly (50/60/70) | 2.575 | 1032.1 (9.1) | 3.32 |
| best fixed at that volume (54/64/74) | 2.441 | 980.9 (9.0) | 3.71 |
| **adaptive, λ = 0.001561 (ours)** | **2.289** | **899.8 (7.6)** | **4.32** |
| fixed 10-yearly + CMOST surveillance (rolling 50–70) | 2.946 | 879.8 (9.7) | 3.42 |
| adaptive, λ = 0.001189 (volume-matched to the surveillance programme) | 2.874 | 818.1 (6.8) | 3.72 |
| fixed 5-yearly (50–75) | 4.986 | 775.3 (8.8) | 2.23 |
| best fixed at that volume (48–78, 6-yearly) | 4.879 | 724.5 (9.7) | 2.38 |
| **adaptive, λ = 0.00069 (ours)** | **4.586** | **682.5 (6.6)** | **2.63** |
**Additional:** A fifth of the surveillance-augmented programme's 2.95 colonoscopies are surveillance exams. Re-optimizing fixed ages captures only ~39% of the adaptive gain (−51.2 of −132.3). Complication deaths are lower than under the guideline schedule (9.3 vs 9.9 per 100,000) but higher than under the best fixed schedule at that volume (9.3 vs 7.3; paired +2.0 ± 0.9).
**Headline delta:** +30% CRC deaths averted per 1,000 colonoscopies with 11% fewer colonoscopies
**Figure:** assets/figures/dp_frontier_c6b.png — Engine efficiency frontiers, CRC deaths (left) and diagnoses (right) vs colonoscopies per person: λ-sweep curves are engine arms at n = 200,000, the marked comparator arms at n = 1,000,000, and the dotted line is the in-model fixed-schedule frontier from the exhaustive search; on deaths the adaptive policies lie below every fixed comparator at matched volume, on diagnoses the adaptive policy volume-matched to the 10-yearly + surveillance programme is level with it (−18 ± 17 per 100,000).
**Audio script:** Inside the engine, with one million population-paired persons per arm, the adaptive policy used eleven percent fewer colonoscopies than ten-yearly screening and still lowered colorectal cancer mortality from one thousand thirty-two to about nine hundred per hundred thousand. Deaths averted per thousand colonoscopies rose from three point three to four point three, a thirty percent gain, and diagnoses averted rose twenty-five percent. It also beat the best re-optimised fixed schedule while using fewer colonoscopies than it did. The comparator closest to guideline practice, ten-yearly screening with CMOST's own post-polypectomy surveillance, reaches twenty fewer deaths per hundred thousand than this policy, but only by performing twenty-nine percent more colonoscopies; the adaptive policy solved for that volume has sixty-two fewer deaths, and per colonoscopy the ordering is unchanged.

## Ablation Study
**Necessary:**
- Policies re-solved on simplified models (pooled polyp stages, no finding memory, one risk class) still beat 10-yearly by 121–172 deaths/100,000 (engine, n = 200,000): adaptivity itself drives the gain.
- As a predictor, dropping the (years-since-exam, last-finding) memory hurts most: 10-yearly mortality reduction 54.5% predicted vs 45.6% true.
**Table:**
| model structure (predicting the engine) | 10-yearly: CRC death reduction vs no screening | 5-yearly: CRC death reduction vs no screening |
|---|---|---|
| **engine (truth)** | **45.6 %** | **59.3 %** |
| 11 states, memory, 6 classes (this paper) | 46.2 % | 61.2 % |
| no (τ, last finding) memory | 54.5 % | 65.1 % |
**Additional:** Ablated-model policies run at 2.07–2.46 colonoscopies per person, 4.21–4.76 deaths averted per 1,000 (engine, n = 200,000 per arm, paired SE 38–39; solved at the 600-point cap, less converged than the headline). Solver diagnostics: the solve is deterministic; re-solves with other rollout seeds or reference propensities differ by less than 10⁻⁶ in in-model objective, and every variant beats 10-yearly screening by 101–150 deaths per 100,000 (engine, n = 200,000). Optimality is not certified (the upper bound is loose); what the engine confirms is the dominance of a family of near-tied policies over fixed scheduling, not the identity of its best member.
**Audio script:** Two ablations. First, we re-solved the policy on deliberately simplified models: pooled polyp stages, no memory of the last finding, or a single risk class. Every one still beat ten-yearly screening by more than one hundred and twenty deaths per hundred thousand, so adaptivity itself, not the exact abstraction, drives the gain. Second, as a predictive model, dropping the memory of years since the last colonoscopy and its finding is the most damaging simplification.

## Headline Numbers
**Hero:** +30% | CRC deaths averted per 1,000 colonoscopies | 4.32 vs 3.32, adaptive vs fixed 10-yearly
**Supporting:** −132 | CRC deaths/100,000 vs 10-yearly ;; −11% | colonoscopies per person ;; 44–53% | mortality cut at attendance 0.3–1.0 (engine, n = 200,000)
**Audio script:** The headline: thirty percent more colorectal cancer deaths averted per thousand colonoscopies, four point three two against three point three two, while performing eleven percent fewer colonoscopies than ten-yearly screening. That is one hundred and thirty-two fewer deaths per hundred thousand people. And when attendance is imperfect, the adaptive policy keeps a forty-four to fifty-three percent mortality reduction at every attendance level we tested, down to thirty percent, while the fixed programme decays from forty-six to eighteen percent.

## Takeaway
**Necessary:** Individualizing colonoscopy timing on each person's own findings prevents more cancers and deaths per colonoscopy than any fixed schedule, including simulator-optimized ones — and the solved policy has a familiar shape: a three-tier surveillance rule with different starting ages for men and women, longer intervals after clean exams and an age-dependent stop.
**Additional:** Complication deaths are lower than under the guideline schedules it replaces (9.3 vs 9.9 and 15.7 vs 19.4 per 100,000), though not below the best fixed schedule at the 10-yearly volume (9.3 vs 7.3), and the gain survives imperfect adherence.
**Audio script:** The takeaway is simple. Deciding each person's next colonoscopy from what their last one found prevents more cancers and deaths per colonoscopy than any fixed schedule, including schedules optimised inside the same simulator. And the solved policy is not exotic: it has the shape of a three-tier surveillance rule with different starting ages for men and women, longer intervals after clean exams and an age-dependent stop, which clinicians will recognise. A clean first exam earns fifteen to twenty years; three or more adenomas bring the next exam forward to within a few years.

## Custom: What the policy does
**Necessary:** Three regularities. A clean first colonoscopy earns a very long interval: 20 years in men, 15 in women. The response to a finding is graded and immediate: 1–2 adenomas shorten the next interval to 9 years (men) / 5 years (women), an advanced adenoma to 3 / 5 years, and ≥3 adenomas bring the next exam forward to within 1–4 years (1 year in the headline solve; the exact year is the feature the objective pins down least — re-solves with other seeds, propensities or kernels place it 1–4 years after the finding in men, at in-model differences below 10⁻⁶ in objective, i.e. under 0.08 deaths per 100,000). The sexes are offset by about nine years at the first exam (49 vs 58).
**Table:**
| observation path | men | women |
|---|---|---|
| all findings normal | 49, 69, 78 | 58, 73 |
| 1–2 adenomas at first, then normal | 49, 58, 67, 75 | 58, 63, 73, 80 |
| ≥3 adenomas at first, then normal | 49, 50, 57, 64, 75 | 58, 59, 63, 69, 74, 79 |
| advanced adenoma at first, then normal | 49, 52, 64, 75 | 58, 63, 73, 80 |
| advanced adenoma twice | 49, 52, 53, 61, 67, 75 | 58, 63, 67, 71, 76, 80 |
**Additional:** Table note: row 3 (≥3 adenomas) shifts by 1–4 years and row 5 (advanced adenoma twice) by 1–2 years in women across equally good re-solves, one of which gives that path five exams rather than six; rows 1, 2 and 4 are identical in every solve. Distilled rule — first colonoscopy at 52; then 10 years after a normal exam, 5 after 1–2 adenomas, 3 after ≥3 or advanced adenomas — gives 866.5 deaths per 100,000 at 2.89 colonoscopies and 3.53 deaths averted per 1,000 colonoscopies (vs 3.32 for 10-yearly), about a fifth of the per-colonoscopy gain and 48 deaths per 100,000 behind the adaptive policy solved for that volume.
**Figure:** assets/figures/dp_policy_c6b_lam0.001561.png — Screening ages of the headline policy (λ = 0.001561) along observation paths (orange) and when its belief is fixed on one risk class (blue; not the separately solved observed-class programme of the Risk-information section, which screens class 0 not at all); men (left), women (right).
**Audio script:** What does the policy actually do? Men get a first colonoscopy at forty-nine, women at fifty-eight. If that exam is clean, the next one is twenty years later for men and fifteen for women. One or two adenomas shorten the interval to nine years in men and five in women; an advanced adenoma to three or five; three or more adenomas bring the next exam forward to within one to four years, the one feature of the policy the objective pins down least. A simple fixed rule, fifty-two then ten, five or three years by finding, captures much of the absolute mortality gain.

## Custom: Robustness to imperfect adherence
**Necessary:** Each invitation is attended with probability α (engine, n = 200,000 per arm, so the α = 1.0 row differs slightly from the 1,000,000-person headline table: 890 vs 899.8 for the same policy). Fixed 10-yearly screening decays from a 46% to an 18% mortality reduction as α falls from 1.0 to 0.3, while the adaptive policy — unchanged, re-planning around no-shows without re-solving — holds a 44–53% reduction at every attendance level. At α = 0.5 it beats even the annual-recall fixed programme (975 vs 1023 deaths per 100,000) with 22% fewer colonoscopies.
**Table:**
| α (attendance); cells = colonoscopies/person → CRC deaths/100,000 (engine, n = 200,000) | fixed 10-yearly (slot lost) | fixed 10-yearly + annual recall | adaptive policy (unchanged) |
|---|---|---|---|
| 1.0 | 2.58 → 1034 | — | 2.29 → **890** |
| 0.7 | 1.80 → 1246 | 2.56 → 1072 | 2.13 → **935** |
| 0.5 | 1.29 → 1352 | 2.54 → 1023 | 1.98 → **975** |
| 0.3 | 0.77 → 1558 | 2.44 → 1030 | 1.73 → 1067 |
**Additional:** none
**Figure:** assets/figures/dp_adherence_c6b.png — CRC mortality (left) and colonoscopy use (right) vs attendance probability: fixed 10-yearly with/without annual recall, and the unchanged adaptive policy (engine, n = 200,000).
**Audio script:** Fixed schedules assume everyone attends. We let each invitation be attended with probability alpha. The classic fixed programme decays from a forty-six percent to an eighteen percent mortality reduction as alpha falls to zero point three. The adaptive policy, unchanged and simply re-planning around no-shows, holds a forty-four to fifty-three percent reduction at every attendance level, and at fifty percent attendance it beats even an annual-recall fixed programme with twenty-two percent fewer colonoscopies.

## Custom: Risk information vs adaptivity
**Necessary:** If the latent risk class could be observed at baseline — a hypothetical oracle that no current test provides — the same objective and shadow price would give an explicit targeting programme: no colonoscopy for the lowest-risk half (a property of the shadow price at λ = 0.001561, not a recommendation; a smaller λ restores screening in that class), 7.49 colonoscopies for the highest-risk 5% (their mortality falls from 5535 under 10-yearly screening to 1956 per 100,000), 1.665 colonoscopies per person overall and 6.34 deaths averted per 1,000 colonoscopies (engine, n = 1,000,000). In the model (not the engine), a realistic baseline score (AUC 0.60, polygenic-score-like) buys 24 fewer CRC deaths per 100,000 at equal volume (2.369 colonoscopies per person); information and adaptivity are nearly additive, and adaptivity is worth about three times a realistic score. The engine cannot resolve a step this small at n = 200,000. The modelled score is an unbiased, perfectly calibrated measurement of the model's own risk multiplier — the most favourable structure a real instrument could have — so a real score of the same nominal AUC would buy less.
**Table:** In-model (not engine) CRC deaths per 100,000, all four cells read at 2.369 colonoscopies per person; fixed cells interpolated on their price–volume envelope.
| CRC deaths/100,000 at 2.369 colonoscopies/person (in-model) | no score | score (AUC 0.60) | the score buys |
|---|---|---|---|
| best fixed schedule | 972 | 942 | −30 |
| adaptive policy | 892 | 867 | −24 |
| **adaptivity buys** | **−80** | **−74** | |
**Additional:** Engine limit: at n = 200,000 the engine cannot separate "score + fixed schedule" (896.0 deaths per 100,000 at 2.262 colonoscopies) from "adaptivity without a score" (889.5 at 2.289; paired +6.5 ± 25.1); both beat unstratified 10-yearly screening by roughly 140 deaths per 100,000 while using fewer colonoscopies.
**Figure:** assets/figures/dp_riskscore_c6b.png — Left: in-model value of a baseline score vs its AUC at matched volume; right: engine arms (n = 200,000), where score-stratified fixed and no-score adaptive overlap.
**Audio script:** If the hidden risk class could be known at baseline, a hypothetical oracle that no current test provides, the policy would become an explicit targeting programme: no screening for the lowest-risk half, which is a property of the shadow price rather than a recommendation and is reversed by a smaller lambda, seven and a half colonoscopies for the highest-risk five percent, and six point three deaths averted per thousand colonoscopies overall. A realistic baseline score with AUC zero point six, like a polygenic score, buys, in the model, twenty-four fewer deaths per hundred thousand, a difference the engine cannot resolve at two hundred thousand persons per arm. Information and adaptivity are nearly additive, and adaptivity is worth about three times a realistic score.

## Custom: Limitations
**Necessary:**
- All results are internal to CMOST and inherit its calibration and synthetic risk spread.
- Disease dynamics depend only on years since the last colonoscopy and its finding, so the time course of preclinical cancer is only approximated.
- Non-attendance is modelled as random; informative non-attendance would change the size of the gain.
- Life-years: the mortality-optimized policies did not differ significantly from fixed schedules, and a life-year-optimized sweep did not transfer its in-model gain to the engine (honest null). We optimise events per colonoscopy, not cost-effectiveness: no ICER, and the surveillance burden on high-risk patients is not valued.
**Additional:** The baseline risk score of the Risk-information section is modelled as an unbiased, perfectly calibrated measurement of the model's own risk multiplier — the most favourable structure a real instrument could have — so a real score of the same nominal AUC would buy less than the 24 deaths per 100,000 quoted.
**Audio script:** Four limitations. All results are internal to CMOST, so they inherit its calibration and its synthetic spread of latent risk; a simulator with weaker heterogeneity would show a smaller gain. The decision model is memoryless given years since the last colonoscopy and its finding, so the sojourn structure of preclinical cancer is only approximated. Non-attendance is modelled as missing completely at random. And on life-years the adaptive and fixed schedules did not differ significantly, an honest null, and we optimise events per colonoscopy, not cost-effectiveness.
