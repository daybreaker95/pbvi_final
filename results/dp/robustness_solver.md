# Solver robustness at lambda = 0.001561 (mortality objective, cap 1500)

| variant | p_ref | seed | in-model deaths /100k (pooled) | colos | objective | FIB gap | alphas = headline (m/f) | paths = headline (m/f) |
|---|---|---|---|---|---|---|---|---|
| headline | 0.12 | 0 | 880.5 | 2.369 | -0.012503 | 0.00711 | - | - |
| base | 0.12 | 0 | 880.5 | 2.369 | -0.012503 | 0.00711 | yes/yes | yes/yes |
| seed1 | 0.12 | 1 | 880.5 | 2.369 | -0.012503 | 0.00711 | no/no | yes/no |
| seed2 | 0.12 | 2 | 880.5 | 2.369 | -0.012503 | 0.00711 | no/no | yes/no |
| pref006 | 0.06 | 0 | 880.5 | 2.369 | -0.012503 | 0.00711 | no/no | yes/no |
| pref025 | 0.25 | 0 | 880.8 | 2.367 | -0.012504 | 0.00711 | no/no | no/no |

## Screening ages along canonical paths (male / female)

### headline
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 69, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

### base
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 69, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

### seed1
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 70, 75, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

### seed2
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 70, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

### pref006
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 70, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 72, 79]

### pref025
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 53, 61, 67, 75] / [58, 60, 65, 70, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 74, 79]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 66, 72, 76, 80]

## Engine (n = 200,000, paired with the headline arm)

| arm | colos | CRC deaths /100k (SE) | paired vs headline arm | identical outcomes | counters |
|---|---|---|---|---|---|
| none | 0.000 | 1898.0 (39.6) |  |  |  |
| q10y | 2.577 | 1034.0 (29.8) |  |  |  |
| dp_death_lam0.001561_q10y | 2.289 | 889.5 (19.7) | +0.0 +- 0.0 | True |  |
| dp_death_lam0.001561_q10y_rerun | 2.289 | 889.5 (19.7) | +0.0 +- 0.0 | True | {'n_decisions': 6844761, 'n_fallback_wait': 0, 'n_policy_calls': 6844761, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |
| rob_base_lam0.001561 | 2.289 | 889.5 (19.7) | +0.0 +- 0.0 | True | {'n_decisions': 6844761, 'n_fallback_wait': 0, 'n_policy_calls': 6844761, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |
| rob_seed1_lam0.001561 | 2.285 | 933.0 (5.2) | +43.5 +- 19.2 | False | {'n_decisions': 6839495, 'n_fallback_wait': 0, 'n_policy_calls': 6839495, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |
| rob_seed2_lam0.001561 | 2.292 | 885.5 (9.3) | -4.0 +- 17.8 | False | {'n_decisions': 6847806, 'n_fallback_wait': 0, 'n_policy_calls': 6847806, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |
| rob_pref006_lam0.001561 | 2.284 | 884.5 (9.5) | -5.0 +- 26.9 | False | {'n_decisions': 6838963, 'n_fallback_wait': 0, 'n_policy_calls': 6838963, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |
| rob_pref025_lam0.001561 | 2.286 | 914.0 (31.2) | +24.5 +- 26.7 | False | {'n_decisions': 6844902, 'n_fallback_wait': 0, 'n_policy_calls': 6844902, 'n_impossible': 0, 'n_lowmass6': 0, 'n_lowmass12': 0, 'n_missed': 0} |

## Belief-set coverage (base re-solve, male / female)

- sex 1: 2,302,846 belief points over 1809 keys; on-policy weighted mean L1 0.0000 (p95 0.0000, max 0.736); one-step deviations: mean 0.0069, p95 0.0359, max 1.941, mass within 0.01 / 0.05 / 0.10: 93.1 / 96.0 / 97.8 %
- sex 2: 2,302,581 belief points over 1809 keys; on-policy weighted mean L1 0.0000 (p95 0.0000, max 0.609); one-step deviations: mean 0.0044, p95 0.0000, max 1.818, mass within 0.01 / 0.05 / 0.10: 95.6 / 97.9 / 98.7 %

## Seed-1 variant extended to n = 1 000 000 (paired with the headline arm)

- colos 2.288, deaths 903.8 (7.4), dx 2507.7; paired vs headline +4.0 +- 12.0; vs q10y -128.3 +- 11.3
