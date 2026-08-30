# Finite-discrimination risk score (engine, n = 200,000 per arm)

Every score arm is solved at the shadow price that matches the latent-class policy's in-model colonoscopy volume (2.369 per person), so the levels differ in information and not in budget.

## Score calibration and the risk gradient it induces

| level | sigma | AUC (within sex, lifetime CRC) | RR top decile | RR bottom decile | top/bottom |
|---|---|---|---|---|---|
| uninformative | 60.00 | 0.507 | 1.04 | 0.95 | 1.1 |
| 0.55 | 9.39 | 0.550 | 1.33 | 0.74 | 1.8 |
| 0.60 | 4.49 | 0.599 | 1.76 | 0.55 | 3.2 |
| 0.65 | 2.63 | 0.650 | 2.32 | 0.43 | 5.4 |
| 0.70 | 1.52 | 0.700 | 3.08 | 0.36 | 8.6 |
| ceiling | 0.00 | 0.751 | 3.94 | 0.33 | 12.0 |

## Overall engine outcomes

| arm | AUC | colos/person | CRC deaths /100k (SE) | CRC dx /100k (SE) | deaths averted /1000 colos | paired vs q10y |
|---|---|---|---|---|---|---|
| none | - | 0.000 | 1898.0 (39.6) | 4554.5 (27.1) | nan | - |
| q10y | - | 2.577 | 1034.0 (29.8) | 2700.0 (27.3) | 3.35 | +0.0 +- 0.0 |
| dp_death_lam0.001561_q10y | - | 2.289 | 889.5 (19.7) | 2460.0 (45.4) | 4.41 | -144.5 +- 47.9 |
| score_0.55 | 0.550 | 2.275 | 919.5 (14.6) | 2502.5 (35.2) | 4.30 | -114.5 +- 36.7 |
| score_0.60 | 0.599 | 2.298 | 904.0 (13.0) | 2493.0 (31.5) | 4.33 | -130.0 +- 30.5 |
| score_0.65 | 0.650 | 2.256 | 869.0 (26.1) | 2417.0 (25.7) | 4.56 | -165.0 +- 40.2 |
| score_0.70 | 0.700 | 2.457 | 792.0 (28.6) | 2289.0 (52.0) | 4.50 | -242.0 +- 46.6 |
| score_ceiling | 0.751 | 2.665 | 732.0 (15.2) | 2088.5 (24.0) | 4.37 | -302.0 +- 29.6 |
| score_uninformative | 0.500 | 2.023 | 945.5 (13.8) | 2611.0 (27.5) | 4.71 | -88.5 +- 39.2 |
| dp_death_lam0.001561_obsclass | class known | 1.665 | 826.0 (25.2) | 2297.5 (19.7) | 6.44 | -208.0 +- 47.8 |

### q10y: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 2.58 | 580 | 1578 |
| mid (classes 1-2, 45 %) | 2.57 | 1033 | 2716 |
| high (classes 3-5, 5 %) | 2.49 | 5535 | 13484 |

### dp_death_lam0.001561_q10y: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.97 | 583 | 1559 |
| mid (classes 1-2, 45 %) | 2.35 | 966 | 2738 |
| high (classes 3-5, 5 %) | 4.92 | 3469 | 9487 |

### score_0.55: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.89 | 623 | 1652 |
| mid (classes 1-2, 45 %) | 2.42 | 998 | 2748 |
| high (classes 3-5, 5 %) | 4.86 | 3179 | 8794 |

### score_0.55: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.100 | 1.83 | 748 | 2084 | 0.023 |
| 2 | 0.101 | 1.93 | 833 | 2246 | 0.029 |
| 3 | 0.101 | 1.97 | 936 | 2442 | 0.036 |
| 4 | 0.101 | 2.00 | 953 | 2608 | 0.044 |
| 5 | 0.100 | 1.99 | 963 | 2495 | 0.043 |
| 6 | 0.100 | 2.06 | 1063 | 2672 | 0.049 |
| 7 | 0.100 | 2.53 | 850 | 2466 | 0.053 |
| 8 | 0.100 | 2.60 | 917 | 2545 | 0.064 |
| 9 | 0.098 | 2.83 | 900 | 2618 | 0.069 |
| 10 | 0.050 | 2.98 | 1015 | 2895 | 0.079 |
| 11 | 0.015 | 3.00 | 1150 | 2739 | 0.090 |
| 12 | 0.015 | 3.02 | 1044 | 2905 | 0.093 |
| 13 | 0.010 | 3.12 | 925 | 2672 | 0.110 |
| 14 | 0.010 | 3.21 | 1027 | 2928 | 0.118 |

*Misclassified cells*: truly high-risk people in the lower half of the score (3.41 % of the population) receive 4.60 colonoscopies and die of CRC at 3457 per 100 000; truly low-risk people in the top two bands (0.61 %) receive 2.54 colonoscopies for a CRC mortality of 489.

### score_0.60: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.81 | 621 | 1651 |
| mid (classes 1-2, 45 %) | 2.52 | 958 | 2697 |
| high (classes 3-5, 5 %) | 5.16 | 3249 | 9075 |

### score_0.60: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.100 | 1.11 | 707 | 2051 | 0.008 |
| 2 | 0.100 | 1.79 | 926 | 2251 | 0.015 |
| 3 | 0.100 | 1.83 | 911 | 2320 | 0.020 |
| 4 | 0.101 | 1.92 | 854 | 2288 | 0.029 |
| 5 | 0.102 | 1.98 | 825 | 2426 | 0.035 |
| 6 | 0.100 | 2.21 | 971 | 2709 | 0.044 |
| 7 | 0.100 | 2.57 | 964 | 2610 | 0.052 |
| 8 | 0.099 | 2.91 | 890 | 2434 | 0.067 |
| 9 | 0.099 | 3.05 | 953 | 2818 | 0.090 |
| 10 | 0.050 | 3.39 | 956 | 2909 | 0.111 |
| 11 | 0.015 | 3.64 | 1092 | 3003 | 0.142 |
| 12 | 0.015 | 3.76 | 1374 | 3403 | 0.152 |
| 13 | 0.010 | 4.03 | 1368 | 3141 | 0.183 |
| 14 | 0.010 | 4.48 | 568 | 3049 | 0.240 |

*Misclassified cells*: truly high-risk people in the lower half of the score (2.69 % of the population) receive 4.60 colonoscopies and die of CRC at 3678 per 100 000; truly low-risk people in the top two bands (0.31 %) receive 3.44 colonoscopies for a CRC mortality of 965.

### score_0.65: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.56 | 614 | 1611 |
| mid (classes 1-2, 45 %) | 2.65 | 933 | 2688 |
| high (classes 3-5, 5 %) | 5.75 | 2848 | 8032 |

### score_0.65: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.100 | 0.71 | 680 | 1686 | 0.001 |
| 2 | 0.100 | 0.89 | 774 | 1959 | 0.004 |
| 3 | 0.101 | 1.62 | 824 | 2210 | 0.007 |
| 4 | 0.101 | 1.81 | 750 | 2128 | 0.013 |
| 5 | 0.101 | 1.92 | 849 | 2429 | 0.019 |
| 6 | 0.100 | 2.07 | 954 | 2337 | 0.031 |
| 7 | 0.100 | 2.59 | 906 | 2718 | 0.043 |
| 8 | 0.098 | 3.13 | 993 | 2602 | 0.065 |
| 9 | 0.099 | 3.42 | 920 | 2905 | 0.100 |
| 10 | 0.051 | 4.02 | 979 | 2899 | 0.157 |
| 11 | 0.015 | 4.47 | 1171 | 3244 | 0.198 |
| 12 | 0.015 | 4.63 | 1015 | 3486 | 0.236 |
| 13 | 0.010 | 5.00 | 1084 | 3793 | 0.308 |
| 14 | 0.010 | 5.76 | 1176 | 3732 | 0.434 |

*Misclassified cells*: truly high-risk people in the lower half of the score (1.83 % of the population) receive 4.71 colonoscopies and die of CRC at 3608 per 100 000; truly low-risk people in the top two bands (0.11 %) receive 3.95 colonoscopies for a CRC mortality of 0.

### score_0.70: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.37 | 593 | 1622 |
| mid (classes 1-2, 45 %) | 3.16 | 867 | 2567 |
| high (classes 3-5, 5 %) | 6.96 | 2106 | 6447 |

### score_0.70: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.100 | 0.37 | 692 | 1684 | 0.000 |
| 2 | 0.100 | 0.69 | 747 | 1787 | 0.000 |
| 3 | 0.101 | 1.16 | 648 | 1824 | 0.000 |
| 4 | 0.100 | 1.63 | 662 | 1952 | 0.002 |
| 5 | 0.100 | 1.90 | 705 | 2231 | 0.004 |
| 6 | 0.100 | 2.62 | 719 | 2333 | 0.010 |
| 7 | 0.100 | 2.96 | 844 | 2317 | 0.020 |
| 8 | 0.100 | 3.42 | 777 | 2452 | 0.042 |
| 9 | 0.098 | 4.13 | 989 | 2754 | 0.092 |
| 10 | 0.051 | 5.02 | 959 | 2986 | 0.193 |
| 11 | 0.015 | 5.69 | 1060 | 3557 | 0.288 |
| 12 | 0.015 | 6.11 | 1151 | 3945 | 0.399 |
| 13 | 0.010 | 6.81 | 1374 | 4071 | 0.539 |
| 14 | 0.010 | 7.68 | 1933 | 5451 | 0.750 |

*Misclassified cells*: truly high-risk people in the lower half of the score (0.79 % of the population) receive 4.73 colonoscopies and die of CRC at 2331 per 100 000; truly low-risk people in the top two bands (0.00 %) receive 4.89 colonoscopies for a CRC mortality of 0.

### score_ceiling: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 0.79 | 648 | 1711 |
| mid (classes 1-2, 45 %) | 4.07 | 713 | 2113 |
| high (classes 3-5, 5 %) | 8.74 | 1745 | 5645 |

### score_ceiling: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.098 | 0.64 | 604 | 1652 | 0.000 |
| 2 | 0.102 | 0.64 | 643 | 1527 | 0.000 |
| 3 | 0.099 | 0.65 | 684 | 1725 | 0.000 |
| 4 | 0.099 | 0.72 | 664 | 1701 | 0.000 |
| 5 | 0.099 | 1.27 | 652 | 1986 | 0.000 |
| 6 | 0.101 | 2.98 | 617 | 1791 | 0.000 |
| 7 | 0.103 | 3.64 | 721 | 2008 | 0.000 |
| 8 | 0.099 | 3.96 | 759 | 2248 | 0.000 |
| 9 | 0.098 | 5.12 | 737 | 2308 | 0.000 |
| 10 | 0.050 | 5.15 | 741 | 2274 | 0.000 |
| 11 | 0.016 | 6.99 | 849 | 3301 | 0.877 |
| 12 | 0.014 | 8.15 | 1274 | 5096 | 1.000 |
| 13 | 0.010 | 10.07 | 2457 | 6118 | 1.000 |
| 14 | 0.012 | 10.26 | 2661 | 8269 | 1.000 |


### score_uninformative: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 1.69 | 589 | 1620 |
| mid (classes 1-2, 45 %) | 2.13 | 1047 | 2890 |
| high (classes 3-5, 5 %) | 4.34 | 3600 | 10007 |

### score_uninformative: by observed score band

| band | share | colos | CRC deaths /100k | CRC dx /100k | P(true class 3-5) |
|---|---|---|---|---|---|
| 1 | 0.099 | 1.99 | 896 | 2568 | 0.044 |
| 2 | 0.101 | 2.01 | 979 | 2521 | 0.047 |
| 3 | 0.102 | 2.01 | 840 | 2470 | 0.050 |
| 4 | 0.100 | 2.01 | 991 | 2757 | 0.047 |
| 5 | 0.100 | 2.02 | 898 | 2530 | 0.049 |
| 6 | 0.099 | 2.01 | 957 | 2668 | 0.048 |
| 7 | 0.100 | 2.02 | 976 | 2759 | 0.054 |
| 8 | 0.100 | 2.03 | 971 | 2598 | 0.052 |
| 9 | 0.099 | 2.04 | 994 | 2567 | 0.052 |
| 10 | 0.050 | 2.02 | 874 | 2713 | 0.054 |
| 11 | 0.015 | 2.04 | 1149 | 2705 | 0.054 |
| 12 | 0.015 | 2.07 | 1092 | 2934 | 0.063 |
| 13 | 0.010 | 2.19 | 696 | 1840 | 0.059 |
| 14 | 0.010 | 2.55 | 1139 | 2899 | 0.052 |

*Misclassified cells*: truly high-risk people in the lower half of the score (3.92 % of the population) receive 4.35 colonoscopies and die of CRC at 3664 per 100 000; truly low-risk people in the top two bands (0.94 %) receive 2.00 colonoscopies for a CRC mortality of 530.

### dp_death_lam0.001561_obsclass: by true risk class

| group | colos | CRC deaths /100k | CRC dx /100k |
|---|---|---|---|
| low (class 0, 50 %) | 0.00 | 713 | 1766 |
| mid (classes 1-2, 45 %) | 2.87 | 841 | 2475 |
| high (classes 3-5, 5 %) | 7.49 | 1956 | 6126 |
