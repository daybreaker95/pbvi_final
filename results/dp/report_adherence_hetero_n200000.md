# Heterogeneous adherence (engine, n = 200,000 per arm)

No screening: 1898 CRC deaths / 100 000. Cells: colonoscopies per person → CRC deaths per 100 000 (mortality reduction). Fixed = 10-yearly at 50/60/70, slot lost if missed; recall = annual re-invitation until attended; adaptive = headline policy (λ = 0.001561) unchanged, re-planning after a no-show.


## E1 — same mean attendance, growing between-person heterogeneity (Beta propensities)

| scenario | mean attend. | never-attended share | fixed | fixed + recall | adaptive (unchanged) |
|---|---|---|---|---|---|
| homogeneous, α = 0.7 (Table 10) | 0.70 |  | 1.80 → 1246 (34 %) | 2.56 → 1072 (44 %) | 2.13 → 934 (51 %) |
| Beta κ = 5 | 0.70 | 1 % | 1.80 → 1212 (36 %) | 2.55 → 1016 (46 %) | 2.11 → 938 (51 %) |
| Beta κ = 2 | 0.70 | 2 % | 1.80 → 1248 (34 %) | 2.50 → 1024 (46 %) | 2.06 → 944 (50 %) |
| Beta κ = 1 | 0.70 | 5 % | 1.80 → 1273 (33 %) | 2.39 → 1110 (42 %) | 1.98 → 1012 (47 %) |
| Beta κ = 0.5 | 0.70 | 10 % | 1.80 → 1284 (32 %) | 2.23 → 1114 (41 %) | 1.88 → 1065 (44 %) |

### E1, Beta κ = 1: outcomes within adherence subgroups

CRC deaths / 100 000 within the subgroup (mortality reduction vs the same persons unscreened); colonoscopies per person in the subgroup.

| arm | low<0.3 (17 % of pop.) | mid (21 % of pop.) | high>=0.7 (61 % of pop.) |
|---|---|---|---|
| q10y_e1_beta_k1 | 1873 (5 %), 0.34 c | 1296 (31 %), 1.32 c | 1093 (42 %), 2.39 c |
| q10y_recall_e1_beta_k1 | 1367 (31 %), 1.56 c | 970 (49 %), 2.54 c | 1086 (42 %), 2.58 c |
| dp_e1_beta_k1 | 1370 (30 %), 1.06 c | 1005 (47 %), 1.98 c | 911 (52 %), 2.25 c |
| dpaware_lam0.001561_e1_beta_k1 | 1215 (38 %), 1.42 c | 967 (49 %), 2.46 c | 929 (51 %), 2.44 c |
| dpaware_lam0.00205_e1_beta_k1 | 1347 (32 %), 1.22 c | 960 (49 %), 2.02 c | 938 (50 %), 2.08 c |
| dpaware_lam0.002691_e1_beta_k1 | 1407 (29 %), 0.99 c | 1021 (46 %), 1.74 c | 990 (47 %), 1.81 c |

## E2 — state-dependent (habitual) attendance: P(attend | first / after attending / after missing)

| scenario | mean attend. | never-attended share | fixed | fixed + recall | adaptive (unchanged) |
|---|---|---|---|---|---|
| Markov .5/.85/.25 | 0.62 | 2 % | 1.40 → 1364 (28 %) | 2.50 → 1068 (44 %) | 2.03 → 992 (48 %) |
| Markov .5/.90/.15 | 0.60 | 5 % | 1.35 → 1386 (27 %) | 2.39 → 1096 (42 %) | 1.95 → 1019 (46 %) |

## E3 — adherence-aware policy (latent 2-class adherence in the hidden state)

| scenario | mean attend. | never-attended share | fixed | fixed + recall | adaptive (unchanged) | adaptive (adherence-aware, matched volume) |
|---|---|---|---|---|---|---|
| mixture 30 % @ .15, 70 % @ .90 | 0.67 | 2 % | 1.74 → 1264 (33 %) | 2.40 → 1068 (44 %) | 1.96 → 995 (48 %) | 1.91 → 955 (50 %) |
| Beta κ = 1 | 0.70 | 5 % | 1.80 → 1273 (33 %) | 2.39 → 1110 (42 %) | 1.98 → 1012 (47 %) | 1.92 → 1014 (47 %) |
| Markov .5/.85/.25 | 0.62 | 2 % | 1.40 → 1364 (28 %) | 2.50 → 1068 (44 %) | 2.03 → 992 (48 %) | 2.03 → 970 (49 %) |

Aware-policy λ grid under the mixture truth (λ per attended colonoscopy):

| λ | colonoscopies | invitations / person | CRC deaths / 100k | deaths averted / 1000 colos |
|---|---|---|---|---|
| 0.001561 | 2.24 | 5.45 | 964 | 4.17 |
| 0.00205 | 1.91 | 4.66 | 955 | 4.95 |
| 0.002691 | 1.63 | 3.87 | 1091 | 4.94 |

### E3, mixture truth: outcomes within adherence subgroups

CRC deaths / 100 000 within the subgroup (mortality reduction vs the same persons unscreened); colonoscopies per person in the subgroup.

| arm | class_low (30 % of pop.) | class_high (70 % of pop.) |
|---|---|---|
| q10y_e3_mix | 1680 (13 %), 0.38 c | 1084 (43 %), 2.32 c |
| q10y_recall_e3_mix | 1117 (42 %), 2.00 c | 1046 (45 %), 2.57 c |
| dp_e3_mix | 1065 (45 %), 1.32 c | 964 (49 %), 2.24 c |
| dpaware_lam0.001561_e3_mix | 1083 (44 %), 1.78 c | 913 (52 %), 2.44 c |
| dpaware_lam0.00205_e3_mix | 1118 (42 %), 1.52 c | 885 (53 %), 2.07 c |
| dpaware_lam0.002691_e3_mix | 1278 (33 %), 1.23 c | 1011 (46 %), 1.81 c |

## E4 — non-adherence correlated with CRC risk (missing not at random)

| scenario | mean attend. | never-attended share | fixed | fixed + recall | adaptive (unchanged) | adaptive (adherence-aware, matched volume) |
|---|---|---|---|---|---|---|
| Beta κ = 1 | 0.70 | 5 % | 1.80 → 1273 (33 %) | 2.39 → 1110 (42 %) | 1.98 → 1012 (47 %) | 1.92 → 1014 (47 %) |
| Beta κ = 1, ρ = 0.2 | 0.70 | 5 % | 1.80 → 1364 (28 %) | 2.39 → 1122 (41 %) | 1.96 → 1053 (45 %) | — |
| Beta κ = 1, ρ = 0.4 | 0.70 | 5 % | 1.81 → 1388 (27 %) | 2.39 → 1116 (41 %) | 1.93 → 1118 (41 %) | 1.86 → 1088 (43 %) |

### E4, ρ = 0.4: outcomes within adherence subgroups

CRC deaths / 100 000 within the subgroup (mortality reduction vs the same persons unscreened); colonoscopies per person in the subgroup.

| arm | low<0.3 (17 % of pop.) | mid (21 % of pop.) | high>=0.7 (61 % of pop.) |
|---|---|---|---|
| q10y_e4_beta_k1_rho04 | 3044 (7 %), 0.33 c | 1531 (27 %), 1.32 c | 866 (40 %), 2.40 c |
| q10y_recall_e4_beta_k1_rho04 | 2079 (36 %), 1.55 c | 1038 (51 %), 2.53 c | 868 (40 %), 2.58 c |
| dp_e4_beta_k1_rho04 | 2251 (31 %), 1.09 c | 1118 (47 %), 2.02 c | 794 (45 %), 2.14 c |
| dpaware_lam0.001561_e4_beta_k1_rho04 | 2110 (35 %), 1.49 c | 860 (59 %), 2.51 c | 743 (48 %), 2.31 c |
| dpaware_lam0.00205_e4_beta_k1_rho04 | 2125 (35 %), 1.28 c | 970 (54 %), 2.08 c | 832 (42 %), 1.95 c |
| dpaware_lam0.002691_e4_beta_k1_rho04 | 2182 (33 %), 1.05 c | 1095 (48 %), 1.78 c | 868 (40 %), 1.73 c |
