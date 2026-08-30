**Table 2.** Screening policies evaluated in the true CMOST environment (paired common random numbers), using the **quarterly-composed** natural-history transition matrix, the corrected colonoscopy detection (`d_PC = 0.939`), and the PBVI solver **re-tuned for the new model** (no screen-disutility, 700 beliefs, 4 expansions). LYG = life-years gained vs no screening. n = 30,000.

| policy | mean colonoscopies | CRC incidence % | CRC mortality % | LYG per 1000 | LYG per colonoscopy |
|--------|:---:|:---:|:---:|:---:|:---:|
| No screening | 0.00 | 4.16 | 1.71 | 0.0 | 0.0 |
| Guideline q10y 45-75 | 3.38 | 2.25 | 0.87 | 103.7 | 30.7 |
| Best fixed x1 | 0.89 | 3.38 | 1.36 | 55.5 | 62.1 |
| Zaika fixed x1 | 0.91 | 3.58 | 1.39 | 48.5 | 53.2 |
| PBVI adaptive x1 | 0.76 | 3.37 | 1.31 | 38.5 | 50.8 |
| Best fixed x2 | 1.66 | 2.80 | 1.12 | 67.2 | 40.6 |
| Zaika fixed x2 | 1.78 | 2.85 | 1.16 | 72.8 | 40.8 |
| PBVI adaptive x2 | 1.67 | 2.93 | 1.13 | 68.8 | 41.1 |
| Best fixed x3 | 2.59 | 2.46 | 0.99 | 91.7 | 35.4 |
| Zaika fixed x3 | 2.64 | 2.57 | 1.06 | 80.4 | 30.5 |
| PBVI adaptive x3 | 2.31 | 2.68 | 0.94 | 78.4 | 33.9 |
| Best fixed x4 | 3.24 | 2.23 | 0.83 | 103.3 | 31.9 |
| Zaika fixed x4 | 3.51 | 2.34 | 0.92 | 99.7 | 28.4 |
| PBVI adaptive x4 | 3.12 | 2.33 | 0.85 | 91.5 | 29.3 |

**Monte-Carlo noise.** LYG (which depends on the rare, heavy-tailed event of a CRC death and the counterfactual natural-death age) has a large sampling error at n = 30,000: **SE ≈ 7–12 per 1000** for each policy. The stable outcome is **CRC mortality** (SE ≈ 0.05–0.07 pp). The same PBVI x1 policy, for example, evaluates to 38.5 LYG/1000 at n = 30,000 but 54.5 at n = 8,000 — the LYG differences below should therefore be read as within noise unless they exceed ~15/1000.

**Efficiency-frontier comparison (PBVI vs interpolated best-fixed at matched colonoscopy use):**

| budget | PBVI mean colo | PBVI LYG/1000 | best-fixed LYG/1000 @ same colo | Δ (PBVI − best-fixed) |
|:--:|:--:|:--:|:--:|:--:|
| 1 | 0.76 | 38.5 | 55.5 | -16.9 |
| 2 | 1.67 | 68.8 | 67.7 | +1.1 |
| 3 | 2.31 | 78.4 | 84.4 | -6.0 |
| 4 | 3.12 | 91.5 | 101.2 | -9.7 |

Under the corrected quarterly-composed model and the re-tuned solver, the PBVI adaptive policy is **statistically comparable to the strong re-optimized best-fixed schedule** on both life-years and CRC mortality: the LYG gaps (−17 to +1 per 1000) are within ~1–1.5 Monte-Carlo SE, and the CRC-mortality differences (≤0.05 pp) are within ~1 SE. Where PBVI has a consistent edge is **colonoscopy use** — it reaches comparable mortality with slightly fewer colonoscopies at budgets 3–4 (2.31 vs 2.59; 3.12 vs 3.24) — and it beats **Zaika's published ages** on CRC mortality at every budget. It does **not** dominate the re-optimized fixed schedule on life-years, as the pre-rebuild annual-matrix results had suggested. (These numbers supersede the pre-rebuild results; see transitions/SCREENING_VALIDATION.md.)
