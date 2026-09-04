# Engine tables recomputed from results/dp/runs (20 chunks of 50 000 per arm)

## Table 4

| arm | colos | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000 colos | dx averted /1000 colos | comp deaths /100k | LY from 40 |
|---|---|---|---|---|---|---|---|
| none | 0.000 | 1887.6 (15.1) | 4545.1 (17.9) | nan | nan | 0.1 | 37.410 |
| q10y | 2.575 | 1032.1 (9.1) | 2686.9 (13.2) | 3.32 | 7.22 | 9.9 | 37.521 |
| bestfixed_q10y_54_64_74 | 2.441 | 980.9 (9.0) | 2641.2 (17.5) | 3.71 | 7.80 | 7.3 | 37.511 |
| dp_death_lam0.001561_q10y | 2.289 | 899.8 (7.6) | 2486.4 (17.5) | 4.32 | 9.00 | 9.3 | 37.519 |
| dp_inc_lam0.005125_q10y | 2.164 | 932.4 (12.9) | 2516.7 (13.9) | 4.41 | 9.38 | 8.4 | 37.530 |
| rule_52_10_5_3_3 | 2.890 | 866.5 (10.2) | 2344.9 (15.2) | 3.53 | 7.61 | 11.3 | 37.523 |
| q5y | 4.986 | 775.3 (8.8) | 2208.0 (13.4) | 2.23 | 4.69 | 19.4 | 37.557 |
| bestfixed_q5y_48_54_60_66_72_78 | 4.879 | 724.5 (9.7) | 2140.5 (11.7) | 2.38 | 4.93 | 16.0 | 37.549 |
| dp_death_lam0.00069_q5y | 4.586 | 682.5 (6.6) | 2056.9 (13.7) | 2.63 | 5.43 | 15.7 | 37.570 |
| dp_inc_lam0.001724_q5y | 5.149 | 668.4 (8.6) | 1977.6 (15.3) | 2.37 | 4.99 | 17.5 | 37.548 |
| dp_death_lam0.001561_obsclass | 1.665 | 832.9 (6.6) | 2303.3 (14.6) | 6.34 | 13.47 | 7.2 | 37.560 |

## Table 5 (paired; chunk SE, t(19) 95 % CI, person-level SE, Holm-adjusted p for the death contrasts)

| contrast | d colos | d deaths: mean +- chunk SE [95 % CI] (person SE) | |d|/SE | p (Holm) | d dx: mean +- SE (person SE) | d comp deaths | d LY /1000 |
|---|---|---|---|---|---|---|---|
| dp_death_lam0.001561_q10y - q10y | -0.287 | -132.3 +- 12.3 [-158.1, -106.5] (13.1) | 10.7 | 1.7e-09 (8.5e-09) | -200.5 +- 22.9 (20.8) | -0.6 +- 1.5 | -1.7 +- 14.6 |
| dp_death_lam0.001561_q10y - bestfixed_q10y_54_64_74 | -0.153 | -81.1 +- 9.2 [-100.4, -61.8] (12.9) | 8.8 | 4.1e-08 (1.2e-07) | -154.8 +- 21.0 (20.6) | +2.0 +- 0.9 | +8.3 +- 14.6 |
| dp_death_lam0.00069_q5y - q5y | -0.400 | -92.8 +- 10.4 [-114.5, -71.1] (11.9) | 9.0 | 3.0e-08 (1.2e-07) | -151.1 +- 13.8 (19.7) | -3.7 +- 2.0 | +12.6 +- 14.1 |
| dp_death_lam0.00069_q5y - bestfixed_q5y_48_54_60_66_72_78 | -0.294 | -42.0 +- 9.7 [-62.2, -21.8] (11.6) | 4.3 | 3.5e-04 (6.9e-04) | -83.6 +- 16.0 (19.6) | -0.3 +- 1.6 | +21.3 +- 15.4 |
| dp_death_lam0.001561_obsclass - q10y | -0.911 | -199.2 +- 12.1 [-224.5, -173.9] (13.4) | 16.5 | 1.1e-12 (7.4e-12) | -383.6 +- 17.9 (21.4) | -2.7 +- 1.0 | +39.0 +- 18.0 |
| rule_52_10_5_3_3 - q10y | +0.315 | -165.6 +- 13.1 [-192.9, -138.3] (13.0) | 12.7 | 1.0e-10 (6.1e-10) | -342.0 +- 20.0 (20.3) | +1.4 +- 1.4 | +2.4 +- 20.5 |
| bestfixed_q10y_54_64_74 - q10y | -0.134 | -51.2 +- 12.0 [-76.4, -26.0] (13.3) | 4.3 | 4.3e-04 (6.9e-04) | -45.7 +- 21.6 (20.9) | -2.6 +- 1.5 | -10.1 +- 16.4 |
| dp_death_lam0.001561_q10y - q10y (comp, LY) | | | | | | -0.6 +- 1.5 | -1.7 +- 14.6 |
| dp_death_lam0.00069_q5y - q5y (comp, LY) | | | | | | -3.7 +- 2.0 | +12.6 +- 14.1 |

## Table 6 (by true risk class; colos / deaths per 100k / dx per 100k)

| arm | low | mid | high |
|---|---|---|---|
| none | 0.00 / 712 / 1741 | 0.00 / 1994 / 4857 | 0.00 / 12665 / 29726 |
| q10y | 2.58 / 580 / 1578 | 2.57 / 1033 / 2716 | 2.49 / 5535 / 13484 |
| q5y | 5.00 / 520 / 1535 | 4.99 / 750 / 2188 | 4.82 / 3547 / 9098 |
| dp_death_lam0.001561_q10y | 1.97 / 583 / 1559 | 2.35 / 966 / 2738 | 4.92 / 3469 / 9487 |
| dp_death_lam0.001561_obsclass | 0.00 / 713 / 1766 | 2.87 / 841 / 2475 | 7.49 / 1956 / 6126 |

## Table 1 population definitions (engine)

| arm | mask | n | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|---|
| none | snapshot_q2_age40 | 965,258 | 1943.4 | 4628.3 |
| none | death_year_ge41_and_undiagnosed_at_40 | 963,860 | 1942.6 | 4626.4 |
| q10y | snapshot_q2_age40 | 965,258 | 1058.2 | 2703.2 |
| q10y | death_year_ge41_and_undiagnosed_at_40 | 963,860 | 1056.4 | 2698.5 |
| q5y | snapshot_q2_age40 | 965,258 | 791.7 | 2207.1 |
| q5y | death_year_ge41_and_undiagnosed_at_40 | 963,860 | 789.8 | 2201.7 |

Model (exact): none: 1934.5 / 4641.5, q10y: 1040.5 / 2685.0, q5y: 750.0 / 2129.7

## Prose quantities

- colos_fewer_vs_q10y_pct: 11.13
- colos_fewer_vs_q5y_pct: 8.03
- colos_fewer_vs_bf10_pct: 6.26
- colos_fewer_vs_bf5_pct: 6.02
- mort_lower_vs_q10y_pct: 12.82
- inc_lower_vs_q10y_pct: 7.46
- mort_lower_vs_q5y_pct: 11.97
- inc_lower_vs_q5y_pct: 6.84
- eff_gain_vs_q10y_pct: 29.93
- effx_gain_vs_q10y_pct: 24.67
- eff_gain_vs_q5y_pct: 17.80
- effx_gain_vs_q5y_pct: 15.76
- bestfixed_share_of_adaptive_gain_pct: 38.70
- model_error_q5y_death_pct: -5.27
- model_error_q10y_death_pct: -1.67
- model_error_none_death_pct: -0.46
