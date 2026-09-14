# Adaptive Colonoscopy Screening from a Mixed-Observability Markov Decision Process Built on the CMOST Microsimulation

## Objective

Colonoscopy capacity is limited, yet microsimulation studies optimize fixed schedules that ignore prior colonoscopy findings. We asked whether adapting each person's next screening interval to their own findings could outperform the best fixed schedule at equal or lower colonoscopy volume.

## Methods

We formulated screening as a finite-horizon mixed-observability Markov decision process (MOMDP). The observed state comprised age, sex, years since the last colonoscopy and its finding; the hidden state crossed six latent adenoma-risk classes with eleven clinical states. All kernels were estimated by maximum likelihood from the instrumented CMOST microsimulation engine using 4,000,000 simulated lives (4,520,648 colonoscopies). The objective minimized expected colorectal cancer (CRC) deaths plus a shadow price per colonoscopy. Policies were solved by point-based value iteration and evaluated in CMOST against 10-yearly and 5-yearly schedules, the best of 2,112 searched fixed schedules, and 10-yearly screening with post-polypectomy surveillance, with 1,000,000 population-paired persons per arm.

## Results

The reduced model reproduced engine CRC mortality and incidence within 5.3%. At 2.29 colonoscopies per person (11% fewer than 10-yearly screening), the adaptive policy lowered CRC mortality from 1032.1 to 899.8 per 100,000 (paired difference −132.3 ± 12.3 [SE]), raising deaths averted per 1,000 colonoscopies from 3.32 to 4.32 (+30%). It also outperformed re-optimized fixed schedules at lower volume (−81.1 ± 9.2 deaths per 100,000). Compared with 10-yearly screening plus surveillance, it averted more deaths per 1,000 colonoscopies (4.32 vs 3.42). Under imperfect adherence, the adaptive policy maintained a 44–53% mortality reduction, whereas the fixed program decayed from 46% to 18%. Life-years did not differ significantly.

## Conclusion

Individualizing colonoscopy timing on realized findings prevented more CRC deaths per colonoscopy than fixed schedules, including simulator-optimized ones, and was robust to imperfect adherence. We propose this simulator-grounded framework as one possible approach to developing personalized health-screening policies.
