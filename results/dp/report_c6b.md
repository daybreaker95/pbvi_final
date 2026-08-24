# dp pipeline report (c6b)

## Headline engine evaluation (n=1,000,000 per arm, paired chunk seeds)

| arm | colos/person | CRC death/100k | SE | incidence/100k | SE | death red. % | inc red. % | deaths averted /1000 colos | cases averted /1000 colos | LYG/1000 | comp death/100k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 1887.6 | 15.1 | 4545.1 | 17.9 | None | None | None | None | None | 0.1 |
| q10y | 2.575 | 1032.1 | 9.1 | 2686.9 | 13.2 | 45.3 | 40.9 | 3.322 | 7.216 | 110.8 | 9.9 |
| q5y | 4.986 | 775.3 | 8.8 | 2208.0 | 13.4 | 58.9 | 51.4 | 2.231 | 4.687 | 147.2 | 19.4 |
| rule_52_10_5_3_3 | 2.890 | 866.5 | 10.2 | 2344.9 | 15.2 | 54.1 | 48.4 | 3.533 | 7.613 | 113.2 | 11.3 |
| dp_death_lam0.000525_q5y | 5.270 | 668.9 | 10.4 | 2012.4 | 14.1 | 64.6 | 55.7 | 2.313 | 4.806 | 141.9 | 17.9 |
| dp_death_lam0.00069_q5y | 4.586 | 682.5 | 6.6 | 2056.9 | 13.7 | 63.8 | 54.7 | 2.628 | 5.426 | 159.8 | 15.7 |
| dp_death_lam0.001561_q10y | 2.289 | 899.8 | 7.6 | 2486.4 | 17.5 | 52.3 | 45.3 | 4.316 | 8.996 | 109.0 | 9.3 |
| dp_death_lam0.001561_obsclass | 1.665 | 832.9 | 6.6 | 2303.3 | 14.6 | 55.9 | 49.3 | 6.336 | 13.466 | 149.7 | 7.2 |
| dp_inc_lam0.001724_q5y | 5.149 | 668.4 | 8.6 | 1977.6 | 15.3 | 64.6 | 56.5 | 2.368 | 4.987 | 137.5 | 17.5 |
| dp_inc_lam0.002264_q5y | 4.080 | 726.8 | 7.6 | 2099.8 | 11.1 | 61.5 | 53.8 | 2.845 | 5.994 | 129.0 | 13.5 |
| dp_inc_lam0.005125_q10y | 2.164 | 932.4 | 12.9 | 2516.7 | 13.9 | 50.6 | 44.6 | 4.415 | 9.375 | 119.6 | 8.4 |
| dp_inc_lam0.005125_obsclass | 1.729 | 837.7 | 8.8 | 2298.5 | 15.1 | 55.6 | 49.4 | 6.071 | 12.992 | 141.5 | 6.0 |
| bestfixed_q10y_54_64_74 | 2.441 | 980.9 | 9.0 | 2641.2 | 17.5 | 48.0 | 41.9 | 3.714 | 7.799 | 100.7 | 7.3 |
| bestfixed_q5y_48_54_60_66_72_78 | 4.879 | 724.5 | 9.7 | 2140.5 | 11.7 | 61.6 | 52.9 | 2.384 | 4.928 | 138.5 | 16.0 |

## Lambda grid, objective = death (engine, n=200,000 per arm)

| arm | colos/person | CRC death/100k | SE | incidence/100k | SE | death red. % | inc red. % | deaths averted /1000 colos | cases averted /1000 colos | LYG/1000 | comp death/100k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 1898.0 | 39.6 | 4554.5 | 27.1 | None | None | None | None | None | 0.0 |
| q10y | 2.577 | 1034.0 | 29.8 | 2700.0 | 27.3 | 45.5 | 40.7 | 3.352 | 7.195 | 130.9 | 7.5 |
| q5y | 4.989 | 801.0 | 11.1 | 2227.0 | 35.0 | 57.8 | 51.1 | 2.199 | 4.666 | 144.3 | 17.0 |
| c6b_death_lam0.0004 | 6.107 | 621.0 | 24.0 | 1984.0 | 26.6 | 67.3 | 56.4 | 2.091 | 4.209 | 127.5 | 18.0 |
| c6b_death_lam0.000525 | 5.268 | 629.5 | 12.1 | 1923.0 | 25.8 | 66.8 | 57.8 | 2.408 | 4.995 | 151.5 | 17.0 |
| c6b_death_lam0.00069 | 4.584 | 668.5 | 9.7 | 1997.5 | 19.1 | 64.8 | 56.1 | 2.682 | 5.579 | 127.5 | 12.5 |
| c6b_death_lam0.000905 | 3.723 | 750.5 | 13.8 | 2173.5 | 19.8 | 60.5 | 52.3 | 3.082 | 6.395 | 135.6 | 8.0 |
| c6b_death_lam0.001189 | 2.875 | 801.0 | 33.2 | 2286.0 | 70.0 | 57.8 | 49.8 | 3.816 | 7.892 | 151.7 | 13.5 |
| c6b_death_lam0.001561 | 2.281 | 901.0 | 32.4 | 2475.0 | 43.0 | 52.5 | 45.7 | 4.371 | 9.116 | 117.3 | 10.0 |
| c6b_death_lam0.00205 | 1.961 | 926.0 | 28.7 | 2575.5 | 49.1 | 51.2 | 43.5 | 4.956 | 10.091 | 98.7 | 8.5 |
| c6b_death_lam0.002691 | 1.808 | 955.5 | 20.4 | 2582.5 | 54.3 | 49.7 | 43.3 | 5.212 | 10.906 | 94.4 | 5.0 |
| c6b_death_lam0.003534 | 1.081 | 1149.5 | 4.3 | 2977.0 | 15.0 | 39.4 | 34.6 | 6.923 | 14.591 | 37.8 | 4.5 |
| c6b_death_lam0.00464 | 1.012 | 1201.5 | 21.9 | 3132.0 | 20.1 | 36.7 | 31.2 | 6.879 | 14.049 | 84.6 | 5.0 |
| c6b_death_lam0.006093 | 0.807 | 1335.5 | 9.0 | 3439.0 | 48.7 | 29.6 | 24.5 | 6.972 | 13.826 | 47.6 | 5.0 |
| c6b_death_lam0.008 | 0.000 | 1898.0 | 39.6 | 4554.5 | 27.1 | 0.0 | 0.0 | nan | nan | 0.0 | 0.0 |

### Frontier interpolated at the comparator volumes (death)

| comparator | colos | comparator deaths | policy deaths (interp.) | comparator dx | policy dx (interp.) |
|---|---|---|---|---|---|
| q10y | 2.577 | 1034 +- 30 | 851 +- 33 | 2700 +- 27 | 2381 +- 70 |
| q5y | 4.989 | 801 +- 11 | 645 +- 12 | 2227 +- 35 | 1953 +- 26 |

## In-model frontier, objective = death (exact, sex-pooled)

| lambda | colos | death/100k | inc/100k | LY | objective |
|---|---|---|---|---|---|
| 0.0004 | 6.324 | 530 | 1687 | 39.444 | -0.007829 |
| 0.000525 | 5.459 | 574 | 1774 | 39.504 | -0.008610 |
| 0.00069 | 4.756 | 620 | 1849 | 39.554 | -0.009478 |
| 0.000905 | 3.868 | 693 | 1999 | 39.616 | -0.010431 |
| 0.001189 | 2.978 | 792 | 2201 | 39.560 | -0.011466 |
| 0.001561 | 2.360 | 882 | 2401 | 39.562 | -0.012504 |
| 0.00205 | 2.032 | 942 | 2555 | 39.592 | -0.013582 |
| 0.002691 | 1.872 | 980 | 2637 | 39.583 | -0.014835 |
| 0.003534 | 1.121 | 1214 | 3105 | 39.503 | -0.016100 |
| 0.00464 | 1.047 | 1244 | 3213 | 39.480 | -0.017298 |
| 0.006093 | 0.834 | 1360 | 3532 | 39.504 | -0.018678 |
| 0.008 | 0.000 | 1934 | 4641 | 39.441 | -0.019345 |

## Lambda grid, objective = inc (engine, n=200,000 per arm)

| arm | colos/person | CRC death/100k | SE | incidence/100k | SE | death red. % | inc red. % | deaths averted /1000 colos | cases averted /1000 colos | LYG/1000 | comp death/100k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.000 | 1898.0 | 39.6 | 4554.5 | 27.1 | None | None | None | None | None | 0.0 |
| q10y | 2.577 | 1034.0 | 29.8 | 2700.0 | 27.3 | 45.5 | 40.7 | 3.352 | 7.195 | 130.9 | 7.5 |
| q5y | 4.989 | 801.0 | 11.1 | 2227.0 | 35.0 | 57.8 | 51.1 | 2.199 | 4.666 | 144.3 | 17.0 |
| c6b_inc_lam0.001 | 6.221 | 620.5 | 17.3 | 1895.5 | 23.1 | 67.3 | 58.4 | 2.053 | 4.274 | 189.8 | 16.0 |
| c6b_inc_lam0.001313 | 5.707 | 658.5 | 17.0 | 1927.0 | 33.6 | 65.3 | 57.7 | 2.172 | 4.604 | 161.9 | 14.5 |
| c6b_inc_lam0.001724 | 5.147 | 680.5 | 13.5 | 1998.0 | 21.7 | 64.1 | 56.1 | 2.365 | 4.967 | 119.2 | 13.5 |
| c6b_inc_lam0.002264 | 4.076 | 770.0 | 26.2 | 2109.0 | 38.7 | 59.4 | 53.7 | 2.767 | 6.000 | 141.3 | 15.0 |
| c6b_inc_lam0.002972 | 3.901 | 734.5 | 30.9 | 2088.5 | 31.6 | 61.3 | 54.1 | 2.982 | 6.321 | 145.4 | 12.5 |
| c6b_inc_lam0.003903 | 3.025 | 800.5 | 14.4 | 2297.0 | 24.3 | 57.8 | 49.6 | 3.628 | 7.463 | 157.0 | 15.0 |
| c6b_inc_lam0.005125 | 2.159 | 900.0 | 20.8 | 2488.5 | 19.6 | 52.6 | 45.4 | 4.624 | 9.571 | 135.7 | 10.5 |
| c6b_inc_lam0.006729 | 2.000 | 905.5 | 22.6 | 2475.5 | 36.8 | 52.3 | 45.6 | 4.962 | 10.393 | 133.2 | 8.0 |
| c6b_inc_lam0.008835 | 1.972 | 967.5 | 25.6 | 2560.5 | 21.9 | 49.0 | 43.8 | 4.719 | 10.112 | 128.4 | 6.0 |
| c6b_inc_lam0.011601 | 1.113 | 1147.5 | 11.6 | 2986.5 | 12.4 | 39.5 | 34.4 | 6.742 | 14.086 | 69.7 | 7.0 |
| c6b_inc_lam0.015232 | 1.056 | 1158.5 | 25.0 | 3047.5 | 20.8 | 39.0 | 33.1 | 7.004 | 14.274 | 64.3 | 4.0 |
| c6b_inc_lam0.02 | 0.524 | 1481.0 | 12.0 | 3684.5 | 22.1 | 22.0 | 19.1 | 7.953 | 16.592 | 48.4 | 2.0 |

### Frontier interpolated at the comparator volumes (inc)

| comparator | colos | comparator deaths | policy deaths (interp.) | comparator dx | policy dx (interp.) |
|---|---|---|---|---|---|
| q10y | 2.577 | 1034 +- 30 | 852 +- 21 | 2700 +- 27 | 2396 +- 24 |
| q5y | 4.989 | 801 +- 11 | 694 +- 26 | 2227 +- 35 | 2014 +- 39 |

## In-model frontier, objective = inc (exact, sex-pooled)

| lambda | colos | death/100k | inc/100k | LY | objective |
|---|---|---|---|---|---|
| 0.001 | 6.399 | 528 | 1600 | 39.363 | -0.027682 |
| 0.001313 | 5.895 | 551 | 1641 | 39.396 | -0.029668 |
| 0.001724 | 5.320 | 584 | 1704 | 39.440 | -0.032053 |
| 0.002264 | 4.231 | 666 | 1860 | 39.496 | -0.034838 |
| 0.002972 | 4.050 | 685 | 1906 | 39.538 | -0.037946 |
| 0.003903 | 3.130 | 778 | 2116 | 39.567 | -0.041153 |
| 0.005125 | 2.234 | 912 | 2391 | 39.541 | -0.044474 |
| 0.006729 | 2.070 | 939 | 2467 | 39.559 | -0.047990 |
| 0.008835 | 2.035 | 946 | 2488 | 39.553 | -0.052321 |
| 0.011601 | 1.155 | 1208 | 3058 | 39.506 | -0.056063 |
| 0.015232 | 1.094 | 1225 | 3123 | 39.498 | -0.060149 |
| 0.02 | 0.551 | 1554 | 3806 | 39.460 | -0.064625 |

## In-model best fixed schedules by volume (deaths objective)

| colos | death/100k | inc/100k | ages |
|---|---|---|---|
| 0.81 | 1460 | 3653 | [68] |
| 1.47 | 1194 | 3217 | [64, 78] |
| 2.00 | 1135 | 3171 | [66, 78, 80] |
| 2.47 | 964 | 2619 | [54, 65, 76] |
| 2.99 | 892 | 2584 | [56, 65, 77, 80] |
| 3.40 | 841 | 2310 | [49, 58, 67, 76] |
| 3.91 | 793 | 2310 | [56, 62, 68, 74, 80] |
| 4.10 | 756 | 2189 | [48, 56, 64, 72, 80] |
| 4.89 | 706 | 2092 | [50, 56, 62, 68, 74, 80] |
| 5.05 | 697 | 2028 | [48, 54, 60, 66, 72, 78] |
| 5.82 | 657 | 1967 | [49, 54, 59, 64, 69, 74, 79] |
| 6.48 | 637 | 1997 | [52, 56, 60, 64, 68, 72, 76, 80] |
| 6.70 | 617 | 1918 | [45, 50, 55, 60, 65, 70, 75, 80] |
| 7.46 | 585 | 1876 | [48, 52, 56, 60, 64, 68, 72, 76, 80] |
| 7.69 | 601 | 1863 | [46, 50, 54, 58, 62, 66, 70, 74, 78] |
| 8.35 | 604 | 1941 | [51, 54, 57, 60, 63, 66, 69, 72, 75, 78] |

## Policy structure (screening ages along canonical observation paths)

### dp_death_lam0.000525_q5y

*male* - in-model: deaths 549/100k, dx 1774/100k, colos 5.462, FIB gap 0.00364

| observation path | screening ages |
|---|---|
| all_normal | [40, 52, 61, 69, 75, 78] |
| adenoma_then_normal | [40, 43, 52, 61, 67, 75, 78] |
| multi_then_normal | [40, 42, 48, 52, 57, 61, 67, 74, 78, 80] |
| advad_then_normal | [40, 41, 52, 58, 64, 67, 75, 78] |
| adenoma_adenoma | [40, 43, 50, 56, 61, 67, 74, 78, 80] |
| advad_advad | [40, 41, 42, 48, 52, 57, 61, 67, 74, 78, 80] |
| class 0 known, all normal | [40, 52, 60, 69, 75, 78] |
| class 1 known, all normal | [40, 52, 61, 67, 74, 76] |
| class 2 known, all normal | [40, 52, 58, 63, 67, 74, 76, 78] |
| class 3 known, all normal | [40, 52, 57, 61, 67, 71, 75, 79, 80] |
| class 4 known, all normal | [40, 52, 57, 61, 66, 71, 76, 78, 79] |
| class 5 known, all normal | [40, 52, 55, 61, 66, 67, 71, 74, 77, 78, 79, 80] |

*female* - in-model: deaths 600/100k, dx 1770/100k, colos 5.460, FIB gap 0.00391

| observation path | screening ages |
|---|---|
| all_normal | [43, 54, 64, 72, 76, 80] |
| adenoma_then_normal | [43, 47, 52, 58, 66, 71, 76, 80] |
| multi_then_normal | [43, 44, 47, 52, 57, 61, 65, 70, 73, 78, 79] |
| advad_then_normal | [43, 44, 47, 52, 58, 65, 71, 76, 80] |
| adenoma_adenoma | [43, 47, 50, 54, 58, 63, 69, 73, 78, 80] |
| advad_advad | [43, 44, 45, 48, 52, 58, 62, 66, 71, 74, 78, 79] |
| class 0 known, all normal | [43, 58, 66, 74, 80] |
| class 1 known, all normal | [43, 58, 67, 74, 78, 80] |
| class 2 known, all normal | [43, 52, 58, 67, 71, 76, 79] |
| class 3 known, all normal | [43, 47, 52, 58, 62, 66, 71, 74, 78, 79] |
| class 4 known, all normal | [43, 45, 50, 54, 59, 63, 66, 70, 73, 76, 79, 80] |
| class 5 known, all normal | [43, 46, 50, 54, 59, 61, 63, 67, 71, 74, 78, 79] |

engine: colonoscopies per 1000 persons by age: 40:477, 41:5, 43:514, 47:23, 50:9, 52:465, 53:11, 54:456, 55:13, 56:50, 57:8, 58:49, 59:7, 61:423, 63:16, 64:403, 65:16, 66:32, 67:118, 69:301, 70:44, 71:43, 72:327, 73:43, 74:54, 75:210, 76:365, 77:10, 78:324, 79:48, 80:385

engine: number of colonoscopies per person: 0:3.7%, 1:3.6%, 2:5.9%, 3:9.0%, 4:9.1%, 5:8.5%, 6:38.8%, 7:12.4%, 8:3.9%, 9:2.1%, 10:0.8%, 11:0.5%, 12:0.4%, 13:0.5%, 14:0.4%, 15:0.3%

engine: findings per colonoscopy: normal:82.9%, adenoma:13.3%, multi:2.5%, advad:1.2%, exit:0.2%

### dp_death_lam0.00069_q5y

*male* - in-model: deaths 578/100k, dx 1799/100k, colos 4.998, FIB gap 0.00438

| observation path | screening ages |
|---|---|
| all_normal | [40, 52, 61, 69, 76] |
| adenoma_then_normal | [40, 43, 52, 61, 67, 75, 78] |
| multi_then_normal | [40, 42, 48, 52, 57, 61, 67, 74, 78, 80] |
| advad_then_normal | [40, 41, 52, 61, 67, 75, 78] |
| adenoma_adenoma | [40, 43, 50, 57, 61, 67, 74, 78, 80] |
| advad_advad | [40, 41, 42, 48, 52, 57, 61, 67, 74, 76] |
| class 0 known, all normal | [40, 52, 69, 78] |
| class 1 known, all normal | [40, 52, 61, 67, 74, 76] |
| class 2 known, all normal | [40, 52, 58, 63, 67, 75, 78] |
| class 3 known, all normal | [40, 52, 57, 61, 67, 71, 76, 79, 80] |
| class 4 known, all normal | [40, 52, 57, 61, 66, 73, 78, 79] |
| class 5 known, all normal | [40, 52, 57, 61, 67, 71, 73, 77, 78, 79, 80] |

*female* - in-model: deaths 662/100k, dx 1899/100k, colos 4.512, FIB gap 0.00466

| observation path | screening ages |
|---|---|
| all_normal | [47, 58, 69, 74, 80] |
| adenoma_then_normal | [47, 50, 58, 65, 71, 76, 80] |
| multi_then_normal | [47, 48, 52, 58, 61, 65, 71, 76, 80] |
| advad_then_normal | [47, 51, 58, 65, 71, 76, 80] |
| adenoma_adenoma | [47, 50, 55, 58, 63, 71, 76, 80] |
| advad_advad | [47, 51, 52, 58, 61, 65, 71, 76, 79] |
| class 0 known, all normal | [47, 62, 74] |
| class 1 known, all normal | [47, 58, 69, 74, 80] |
| class 2 known, all normal | [47, 54, 63, 69, 73, 78, 79] |
| class 3 known, all normal | [47, 52, 58, 63, 70, 73, 78, 79] |
| class 4 known, all normal | [47, 51, 57, 61, 65, 69, 73, 76, 79, 80] |
| class 5 known, all normal | [47, 50, 54, 60, 63, 65, 69, 72, 76, 78, 79] |

engine: colonoscopies per 1000 persons by age: 40:477, 41:5, 43:30, 47:482, 50:30, 51:8, 52:447, 53:9, 55:10, 56:50, 57:9, 58:457, 59:7, 61:393, 63:74, 64:11, 65:22, 66:8, 67:117, 69:624, 70:38, 71:54, 72:13, 73:40, 74:309, 75:54, 76:258, 77:11, 78:123, 79:40, 80:361

engine: number of colonoscopies per person: 0:4.1%, 1:4.1%, 2:7.5%, 3:8.4%, 4:12.6%, 5:43.2%, 6:8.4%, 7:5.5%, 8:2.9%, 9:1.1%, 10:0.6%, 11:0.5%, 12:0.3%, 13:0.2%, 14:0.2%, 15:0.2%

engine: findings per colonoscopy: normal:81.6%, adenoma:13.9%, multi:2.9%, advad:1.4%, exit:0.2%

### dp_death_lam0.001561_q10y

*male* - in-model: deaths 816/100k, dx 2307/100k, colos 2.648, FIB gap 0.00698

| observation path | screening ages |
|---|---|
| all_normal | [49, 69, 78] |
| adenoma_then_normal | [49, 58, 67, 75] |
| multi_then_normal | [49, 50, 57, 64, 75] |
| advad_then_normal | [49, 52, 64, 75] |
| adenoma_adenoma | [49, 58, 67, 75] |
| advad_advad | [49, 52, 53, 61, 67, 75] |
| class 0 known, all normal | [49, 78] |
| class 1 known, all normal | [49, 67, 76] |
| class 2 known, all normal | [49, 58, 67, 75, 80] |
| class 3 known, all normal | [49, 57, 67, 71, 78, 80] |
| class 4 known, all normal | [49, 54, 61, 67, 75, 79] |
| class 5 known, all normal | [49, 54, 61, 67, 71, 76, 78, 79, 80] |

*female* - in-model: deaths 945/100k, dx 2488/100k, colos 2.090, FIB gap 0.00723

| observation path | screening ages |
|---|---|
| all_normal | [58, 73] |
| adenoma_then_normal | [58, 63, 73, 80] |
| multi_then_normal | [58, 59, 63, 69, 74, 79] |
| advad_then_normal | [58, 63, 73, 80] |
| adenoma_adenoma | [58, 63, 69, 76, 80] |
| advad_advad | [58, 63, 67, 71, 76, 80] |
| class 0 known, all normal | [58, 72] |
| class 1 known, all normal | [58, 71, 80] |
| class 2 known, all normal | [58, 67, 73, 79] |
| class 3 known, all normal | [58, 63, 70, 73, 79] |
| class 4 known, all normal | [58, 63, 69, 73, 76, 79, 80] |
| class 5 known, all normal | [58, 63, 69, 74, 78, 79] |

engine: colonoscopies per 1000 persons by age: 49:462, 50:6, 52:10, 57:6, 58:503, 61:10, 63:54, 64:15, 67:41, 69:332, 71:17, 72:7, 73:365, 74:6, 75:95, 76:17, 78:184, 79:32, 80:105

engine: number of colonoscopies per person: 0:7.8%, 1:15.9%, 2:38.4%, 3:26.5%, 4:6.7%, 5:2.2%, 6:0.8%, 7:0.7%, 8:0.4%, 9:0.2%, 10:0.3%, 11:0.1%

engine: findings per colonoscopy: normal:74.8%, adenoma:17.1%, multi:4.8%, advad:3.1%, exit:0.3%

### dp_death_lam0.001561_obsclass

*male* - in-model: deaths 817/100k, dx 2311/100k, colos 2.643, FIB gap 0.00698

| observation path | screening ages |
|---|---|
| all_normal | [49, 69, 78] |
| adenoma_then_normal | [49, 58, 67, 75] |
| multi_then_normal | [49, 53, 61, 67, 75] |
| advad_then_normal | [49, 52, 64, 75] |
| adenoma_adenoma | [49, 58, 67, 75] |
| advad_advad | [49, 52, 53, 61, 67, 75] |
| class 0 known, all normal | [] |
| class 1 known, all normal | [57, 67, 76] |
| class 2 known, all normal | [41, 56, 66, 75, 80] |
| class 3 known, all normal | [40, 48, 57, 67, 71, 79, 80] |
| class 4 known, all normal | [40, 46, 53, 59, 61, 64, 75, 79] |
| class 5 known, all normal | [40, 44, 52, 57, 64, 67, 71, 73, 77, 79, 80] |

*female* - in-model: deaths 945/100k, dx 2488/100k, colos 2.090, FIB gap 0.00723

| observation path | screening ages |
|---|---|
| all_normal | [58, 73] |
| adenoma_then_normal | [58, 63, 73, 80] |
| multi_then_normal | [58, 59, 63, 69, 74, 79] |
| advad_then_normal | [58, 63, 73, 80] |
| adenoma_adenoma | [58, 63, 69, 76, 80] |
| advad_advad | [58, 63, 67, 71, 76, 80] |
| class 0 known, all normal | [] |
| class 1 known, all normal | [58, 73, 80] |
| class 2 known, all normal | [43, 58, 67, 73, 79] |
| class 3 known, all normal | [41, 50, 59, 60, 71, 74, 79] |
| class 4 known, all normal | [40, 45, 53, 59, 64, 69, 73, 78, 80] |
| class 5 known, all normal | [40, 45, 49, 54, 60, 63, 67, 72, 76, 79] |

engine: colonoscopies per 1000 persons by age: 40:41, 41:79, 43:80, 44:6, 45:11, 48:9, 50:12, 51:12, 53:14, 55:6, 56:64, 57:144, 58:211, 59:14, 60:5, 61:31, 63:7, 64:14, 65:8, 66:58, 67:176, 68:6, 69:17, 70:6, 71:29, 72:15, 73:143, 74:7, 75:53, 76:95, 77:14, 78:18, 79:96, 80:142

engine: number of colonoscopies per person: 0:53.8%, 1:5.0%, 2:6.6%, 3:17.2%, 4:5.1%, 5:8.3%, 6:0.6%, 7:0.6%, 8:0.7%, 9:0.7%, 10:0.3%, 11:0.5%, 12:0.3%, 13:0.1%

engine: findings per colonoscopy: normal:60.7%, adenoma:28.1%, multi:7.0%, advad:4.0%, exit:0.3%

### dp_inc_lam0.001724_q5y

*male* - in-model: deaths 582/100k, dx 1735/100k, colos 4.987, FIB gap 0.00970

| observation path | screening ages |
|---|---|
| all_normal | [40, 49, 61, 69, 75] |
| adenoma_then_normal | [40, 44, 52, 58, 64, 67, 75] |
| multi_then_normal | [40, 41, 46, 52, 57, 61, 67, 75, 76] |
| advad_then_normal | [40, 41, 49, 56, 61, 67, 75] |
| adenoma_adenoma | [40, 44, 49, 53, 56, 61, 67, 75, 76] |
| advad_advad | [40, 41, 42, 46, 52, 57, 61, 67, 75, 76] |
| class 0 known, all normal | [40, 49, 61, 69, 76] |
| class 1 known, all normal | [40, 49, 56, 61, 67, 76] |
| class 2 known, all normal | [40, 49, 53, 56, 61, 67, 75, 78] |
| class 3 known, all normal | [40, 49, 52, 57, 61, 67, 71, 75, 79, 80] |
| class 4 known, all normal | [40, 49, 53, 56, 61, 65, 67, 72, 76, 79] |
| class 5 known, all normal | [40, 49, 53, 56, 61, 65, 67, 71, 72, 76, 78, 79, 80] |

*female* - in-model: deaths 587/100k, dx 1673/100k, colos 5.653, FIB gap 0.01131

| observation path | screening ages |
|---|---|
| all_normal | [43, 50, 58, 67, 74, 80] |
| adenoma_then_normal | [43, 47, 52, 58, 63, 69, 73, 80] |
| multi_then_normal | [43, 44, 47, 50, 54, 58, 63, 67, 73, 78, 79] |
| advad_then_normal | [43, 44, 47, 52, 58, 63, 69, 73, 80] |
| adenoma_adenoma | [43, 47, 50, 54, 58, 63, 67, 73, 78] |
| advad_advad | [43, 44, 45, 47, 52, 58, 61, 64, 71, 74, 78, 79] |
| class 0 known, all normal | [43, 54, 64, 73] |
| class 1 known, all normal | [43, 50, 58, 67, 74, 80] |
| class 2 known, all normal | [43, 51, 54, 58, 63, 67, 73, 78, 79] |
| class 3 known, all normal | [43, 47, 52, 58, 61, 64, 71, 74, 78, 79] |
| class 4 known, all normal | [43, 45, 50, 54, 57, 60, 63, 67, 71, 72, 75, 78, 80] |
| class 5 known, all normal | [43, 45, 48, 51, 55, 57, 59, 61, 62, 65, 68, 72, 75, 78, 79] |

engine: colonoscopies per 1000 persons by age: 40:477, 41:7, 43:486, 44:33, 47:23, 49:438, 50:463, 52:45, 53:10, 54:9, 55:26, 56:46, 57:13, 58:472, 59:5, 61:409, 62:11, 63:48, 64:24, 65:19, 66:6, 67:495, 69:294, 70:42, 71:27, 72:10, 73:85, 74:306, 75:282, 76:38, 77:13, 78:143, 79:52, 80:276

engine: number of colonoscopies per person: 0:3.7%, 1:2.2%, 2:5.3%, 3:7.9%, 4:10.2%, 5:26.4%, 6:30.1%, 7:6.2%, 8:2.9%, 9:1.6%, 10:0.7%, 11:0.5%, 12:0.4%, 13:0.4%, 14:0.4%, 15:0.4%, 16:0.3%, 17:0.2%

engine: findings per colonoscopy: normal:82.9%, adenoma:13.1%, multi:2.6%, advad:1.2%, exit:0.1%

### dp_inc_lam0.002264_q5y

*male* - in-model: deaths 639/100k, dx 1836/100k, colos 4.321, FIB gap 0.01180

| observation path | screening ages |
|---|---|
| all_normal | [40, 49, 65, 75] |
| adenoma_then_normal | [40, 44, 52, 61, 67, 75] |
| multi_then_normal | [40, 42, 48, 52, 57, 61, 67, 75] |
| advad_then_normal | [40, 43, 52, 61, 67, 75] |
| adenoma_adenoma | [40, 44, 50, 56, 61, 67, 75] |
| advad_advad | [40, 43, 44, 51, 57, 61, 67, 75] |
| class 0 known, all normal | [40, 49, 65, 76] |
| class 1 known, all normal | [40, 49, 61, 67, 76] |
| class 2 known, all normal | [40, 49, 56, 61, 67, 75, 78] |
| class 3 known, all normal | [40, 49, 52, 57, 61, 67, 71, 76, 79, 80] |
| class 4 known, all normal | [40, 49, 52, 57, 61, 65, 67, 75, 79] |
| class 5 known, all normal | [40, 49, 53, 56, 61, 65, 67, 71, 72, 77, 78, 79, 80] |

*female* - in-model: deaths 694/100k, dx 1883/100k, colos 4.141, FIB gap 0.01365

| observation path | screening ages |
|---|---|
| all_normal | [47, 58, 67, 76] |
| adenoma_then_normal | [47, 50, 54, 58, 67, 73, 80] |
| multi_then_normal | [47, 48, 52, 57, 60, 63, 67, 73, 78] |
| advad_then_normal | [47, 50, 54, 60, 67, 73, 80] |
| adenoma_adenoma | [47, 50, 55, 58, 63, 67, 73, 78] |
| advad_advad | [47, 50, 51, 57, 60, 65, 70, 73, 78] |
| class 0 known, all normal | [47, 62, 73] |
| class 1 known, all normal | [47, 58, 67, 73, 80] |
| class 2 known, all normal | [47, 54, 58, 63, 69, 73, 79] |
| class 3 known, all normal | [47, 51, 57, 60, 65, 70, 73, 75, 79] |
| class 4 known, all normal | [47, 50, 54, 59, 61, 63, 67, 71, 74, 77, 80] |
| class 5 known, all normal | [47, 50, 54, 59, 61, 63, 65, 68, 72, 75, 78, 79] |

engine: colonoscopies per 1000 persons by age: 40:477, 43:5, 44:29, 47:481, 49:428, 50:39, 52:29, 54:25, 55:7, 56:45, 57:14, 58:453, 59:6, 60:8, 61:69, 63:39, 64:14, 65:345, 67:450, 68:64, 69:32, 70:30, 71:23, 72:10, 73:78, 74:18, 75:291, 76:298, 77:11, 78:113, 79:44, 80:80

engine: number of colonoscopies per person: 0:4.2%, 1:3.4%, 2:9.2%, 3:13.0%, 4:44.1%, 5:11.5%, 6:6.4%, 7:3.4%, 8:1.5%, 9:0.8%, 10:0.5%, 11:0.4%, 12:0.5%, 13:0.4%, 14:0.3%, 15:0.3%, 16:0.2%

engine: findings per colonoscopy: normal:80.5%, adenoma:14.5%, multi:3.3%, advad:1.5%, exit:0.2%

### dp_inc_lam0.005125_q10y

*male* - in-model: deaths 882/100k, dx 2353/100k, colos 2.289, FIB gap 0.01925

| observation path | screening ages |
|---|---|
| all_normal | [49, 69] |
| adenoma_then_normal | [49, 56, 67, 75] |
| multi_then_normal | [49, 50, 57, 61, 67, 75] |
| advad_then_normal | [49, 52, 61, 67, 75] |
| adenoma_adenoma | [49, 56, 61, 67, 75] |
| advad_advad | [49, 52, 53, 57, 64, 75] |
| class 0 known, all normal | [49, 66] |
| class 1 known, all normal | [49, 61, 72] |
| class 2 known, all normal | [49, 58, 64, 75] |
| class 3 known, all normal | [49, 57, 65, 72, 76, 80] |
| class 4 known, all normal | [49, 55, 61, 67, 75, 79] |
| class 5 known, all normal | [49, 55, 61, 67, 71, 77, 78, 79, 80] |

*female* - in-model: deaths 940/100k, dx 2425/100k, colos 2.183, FIB gap 0.02164

| observation path | screening ages |
|---|---|
| all_normal | [54, 72] |
| adenoma_then_normal | [54, 58, 67, 76] |
| multi_then_normal | [54, 55, 60, 64, 71, 74, 79] |
| advad_then_normal | [54, 58, 65, 73, 80] |
| adenoma_adenoma | [54, 58, 63, 71, 79] |
| advad_advad | [54, 58, 59, 63, 69, 73, 79] |
| class 0 known, all normal | [54, 72] |
| class 1 known, all normal | [54, 65, 74] |
| class 2 known, all normal | [54, 63, 72, 79] |
| class 3 known, all normal | [54, 60, 64, 71, 74, 79] |
| class 4 known, all normal | [54, 60, 64, 68, 72, 76, 79, 80] |
| class 5 known, all normal | [54, 60, 63, 68, 72, 75, 79] |

engine: colonoscopies per 1000 persons by age: 49:462, 50:6, 52:10, 54:471, 56:47, 57:6, 58:47, 61:25, 63:12, 64:14, 65:8, 67:72, 69:326, 70:6, 71:17, 72:364, 73:17, 75:98, 76:28, 77:6, 78:53, 79:23, 80:28

engine: number of colonoscopies per person: 0:6.9%, 1:15.9%, 2:55.8%, 3:11.2%, 4:5.1%, 5:2.2%, 6:0.8%, 7:0.5%, 8:0.5%, 9:0.5%, 10:0.3%, 11:0.2%, 12:0.1%

engine: findings per colonoscopy: normal:74.6%, adenoma:16.9%, multi:5.2%, advad:3.0%, exit:0.3%

### dp_inc_lam0.005125_obsclass

*male* - in-model: deaths 882/100k, dx 2353/100k, colos 2.289, FIB gap 0.01926

| observation path | screening ages |
|---|---|
| all_normal | [49, 69] |
| adenoma_then_normal | [49, 56, 67, 75] |
| multi_then_normal | [49, 50, 57, 61, 67, 75] |
| advad_then_normal | [49, 52, 61, 67, 75] |
| adenoma_adenoma | [49, 56, 61, 67, 75] |
| advad_advad | [49, 52, 53, 57, 64, 75] |
| class 0 known, all normal | [] |
| class 1 known, all normal | [49, 61, 72] |
| class 2 known, all normal | [41, 56, 66, 75] |
| class 3 known, all normal | [42, 48, 59, 65, 72, 76, 80] |
| class 4 known, all normal | [40, 46, 53, 57, 61, 65, 72, 78, 79] |
| class 5 known, all normal | [40, 44, 51, 55, 61, 66, 70, 72, 77, 78, 79, 80] |

*female* - in-model: deaths 940/100k, dx 2425/100k, colos 2.184, FIB gap 0.02165

| observation path | screening ages |
|---|---|
| all_normal | [54, 72] |
| adenoma_then_normal | [54, 58, 67, 76] |
| multi_then_normal | [54, 55, 60, 63, 69, 74, 79] |
| advad_then_normal | [54, 58, 65, 73, 80] |
| adenoma_adenoma | [54, 58, 63, 69, 78] |
| advad_advad | [54, 58, 60, 63, 69, 74, 79] |
| class 0 known, all normal | [] |
| class 1 known, all normal | [58, 73, 80] |
| class 2 known, all normal | [43, 58, 67, 73, 79] |
| class 3 known, all normal | [41, 50, 58, 63, 72, 75, 79] |
| class 4 known, all normal | [40, 45, 50, 54, 59, 63, 67, 71, 75, 78, 80] |
| class 5 known, all normal | [40, 41, 50, 54, 60, 63, 68, 72, 75, 79] |

engine: colonoscopies per 1000 persons by age: 40:35, 41:84, 42:8, 43:76, 44:12, 45:6, 47:5, 48:8, 49:141, 50:20, 51:12, 53:10, 54:10, 55:7, 56:63, 57:14, 58:232, 59:23, 60:21, 61:126, 63:11, 64:13, 65:16, 66:58, 67:74, 68:8, 69:11, 70:16, 71:10, 72:88, 73:174, 74:5, 75:55, 76:31, 77:9, 78:24, 79:74, 80:129

engine: number of colonoscopies per person: 0:53.1%, 1:4.3%, 2:6.6%, 3:17.7%, 4:6.9%, 5:7.2%, 6:0.6%, 7:0.5%, 8:0.7%, 9:0.5%, 10:0.5%, 11:0.4%, 12:0.5%, 13:0.4%, 14:0.1%

engine: findings per colonoscopy: normal:62.2%, adenoma:27.1%, multi:6.8%, advad:3.7%, exit:0.3%
