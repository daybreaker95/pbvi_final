# tau-support sensitivity (WAIT person-years at tau > 20 excluded)

- person-years excluded: 1,246,834 of 154,854,388 (0.81 %)
- kernel rows changed: 65,387 (of which at decision ages 40-80: 4,847); max |dT| ages 40-80 = 0.0701, ages 81-99 = 0.2351

## Table 1 predictions (deaths / dx per 100k)

- none: base 1934.5 / 4641.5; tau20 1934.5 / 4641.5
- q10y: base 1040.5 / 2685.0; tau20 1054.2 / 2707.3
- q5y: base 750.0 / 2129.7; tau20 757.3 / 2142.2

## Headline price re-solved on the tau20 kernels (in-model, pooled)

- headline: deaths 880.5, dx 2397, colos 2.369, objective -0.012503, gap 0.00711
- tau20: deaths 887.0, dx 2410, colos 2.366, objective -0.012564, gap 0.00714

## Cross-evaluation (exact, pooled)

- policy headline on kernels base: deaths 880.5, colos 2.369, objective -0.012503
- policy headline on kernels tau20: deaths 886.6, colos 2.369, objective -0.012564
- policy tau20 on kernels base: deaths 881.0, colos 2.366, objective -0.012504
- policy tau20 on kernels tau20: deaths 887.0, colos 2.366, objective -0.012564

## Screening ages along canonical paths (male / female)

### headline
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 50, 57, 64, 75] / [58, 59, 63, 69, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

### tau20
- all_normal: [49, 69, 78] / [58, 73]
- adenoma_then_normal: [49, 58, 67, 75] / [58, 63, 73, 80]
- multi_then_normal: [49, 53, 61, 67, 75] / [58, 59, 63, 70, 74, 79]
- advad_then_normal: [49, 52, 64, 75] / [58, 63, 73, 80]
- adenoma_adenoma: [49, 58, 67, 75] / [58, 63, 69, 76, 80]
- advad_advad: [49, 52, 53, 61, 67, 75] / [58, 63, 67, 71, 76, 80]

## Engine (n = 200,000, paired chunks)

- none: colos 0.000, deaths 1898.0 (39.6)
- q10y: colos 2.577, deaths 1034.0 (29.8)
- dp_death_lam0.001561_q10y: colos 2.289, deaths 889.5 (19.7)
- tau20_lam0.001561: colos 2.286, deaths 887.5 (16.3); paired vs headline: deaths -2.0 +- 24.9, colos -0.003
