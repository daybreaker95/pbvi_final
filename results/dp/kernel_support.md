# Kernel support under the headline policy (lambda = 0.001561)

## Back-off level of the kernel rows, weighted by policy occupancy

| level | WAIT rows, ages 40-80 | WAIT rows, ages 81-99 | SCREEN rows |
|---|---|---|---|
| own cell, +-0 y | 96.4 % | 93.5 % | 88.0 % (own cell, +-0 bands) |
| own cell, +-1 y | 2.2 % | 2.8 % | 3.5 % (own cell, +-1 bands) |
| own cell, +-2 y | 0.5 % | 1.0 % | 1.9 % (own cell, +-2 bands) |
| own cell, +-4 y | 0.4 % | 0.8 % | 1.9 % (own cell, +-4 bands) |
| own cell, +-8 y | 0.2 % | 0.7 % | 0.5 % (own cell, +-8 bands) |
| tau pooled (finding kept), +-2 y | 0.1 % | 0.3 % | 3.6 % (tau pooled (finding kept), +-2 bands) |
| tau pooled (finding kept), +-4 y | 0.0 % | 0.2 % | 0.3 % (tau pooled (finding kept), +-4 bands) |
| tau pooled (finding kept), +-8 y | 0.0 % | 0.2 % | 0.0 % (tau pooled (finding kept), +-8 bands) |
| finding pooled (tau kept), +-2 y | 0.0 % | 0.2 % | 0.1 % (finding pooled (tau kept), +-2 bands) |
| finding pooled (tau kept), +-4 y | 0.0 % | 0.0 % | 0.0 % (finding pooled (tau kept), +-4 bands) |
| finding pooled (tau kept), +-8 y | 0.0 % | 0.0 % | 0.0 % (finding pooled (tau kept), +-8 bands) |
| all post-screen cells, +-4 y | 0.0 % | 0.0 % | 0.1 % (all post-screen cells, +-4 bands) |
| all post-screen cells, +-8 y | 0.0 % | 0.0 % | 0.0 % (all post-screen cells, +-8 bands) |
| all memory cells, +-8 y | 0.0 % | 0.0 % | 0.0 % (all memory cells, +-8 bands) |
| class pooled | 0.0 % | 0.2 % | 0.0 % (class pooled) |
| sex pooled | 0.0 % | 0.0 % | 0.0 % (sex pooled) |

## WAIT rows by tau group (ages 40-80)

| tau group | share of occupancy | own-cell (+-0..8 y) share | occupancy-weighted own-cell person-years | mass on cells with < 150 own person-years |
|---|---|---|---|---|
| never | 37.4 % | 100.0 % | 559,962 | 0.1 % |
| tau 0 | 6.7 % | 99.1 % | 14,994 | 11.0 % |
| tau 1 | 6.2 % | 99.3 % | 14,252 | 10.1 % |
| tau 2 | 5.9 % | 99.4 % | 13,143 | 9.9 % |
| tau 3 | 5.3 % | 99.4 % | 12,463 | 9.3 % |
| tau 4-5 | 9.7 % | 99.8 % | 22,559 | 4.8 % |
| tau 6-8 | 11.1 % | 99.9 % | 28,966 | 2.7 % |
| tau 9-12 | 8.7 % | 99.9 % | 26,545 | 1.7 % |
| tau 13+ | 9.0 % | 99.9 % | 15,131 | 2.1 % |

## SCREEN rows by tau group

| tau group | share of colonoscopies | own-cell (+-0..8 bands) share | occupancy-weighted own-cell colonoscopies | mass on cells with < 150 own colonoscopies |
|---|---|---|---|---|
| never | 40.3 % | 99.9 % | 39,997 | 0.9 % |
| tau 1 | 1.5 % | 54.3 % | 33 | 98.5 % |
| tau 2 | 0.9 % | 45.0 % | 70 | 92.2 % |
| tau 3 | 2.0 % | 48.6 % | 38 | 98.4 % |
| tau 4-5 | 6.1 % | 81.4 % | 417 | 60.3 % |
| tau 6-8 | 8.2 % | 92.8 % | 1,935 | 23.2 % |
| tau 9-12 | 12.6 % | 98.2 % | 11,414 | 9.4 % |
| tau 13+ | 28.3 % | 99.6 % | 23,167 | 1.9 % |

## The open-ended tau >= 13 cells

| last finding | share of all WAIT occupancy | own-cell person-years (all ages) | own-cell person-years (ages 53-80) | occupancy-weighted own-cell person-years | own-cell share |
|---|---|---|---|---|---|
| normal | 9.05 % | 6,218,286 | 2,533,781 | 15,131 | 99.9 % |
| adenoma | 0.00 % | 954,932 | 265,695 | 0 | 0.0 % |
| multi | 0.00 % | 123,761 | 30,017 | 0 | 0.0 % |
| advad | 0.00 % | 193,360 | 51,601 | 0 | 0.0 % |

Rows estimated at the sex-pooled level: WAIT 0 of 261360, SCREEN 0 of 34848; rows never observed at any level (defaulted): WAIT 0, SCREEN 0.

## tau composition of the person-years in the tau >= 13 group (kernels_c6b_re.npz)

- 7,490,339 person-years with tau >= 13; 83.4 % at tau 13-20 (inside the randomised design's interval support), 16.6 % at tau > 20, 0.9 % at tau > 30.
- share at tau > 20 by decision age: 40: 0 %, 45: 0 %, 50: 0 %, 55: 0 %, 60: 0 %, 65: 0 %, 70: 0 %, 75: 0 %, 80: 0 %, 85: 11 %, 90: 27 %, 95: 51 %
