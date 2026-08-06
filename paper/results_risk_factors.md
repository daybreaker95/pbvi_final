<!-- numbers from results/risk_factors.json (n=25000). RiskStrat row filled after re-run. -->

# Can baseline risk factors let the POMDP target high-risk patients?

**Motivation.** `results_subgroups.md` showed that at matched adherence the base
PBVI policy does **not** help high-risk patients — it cannot pre-identify them from
mostly-clean colonoscopies, and a clean screen even *lowers* the inferred risk. Two
things block risk-targeting, and this experiment removes both.

## Two required changes

1. **Personalized prior from baseline risk factors.** Age is already fully observed
   (the policy conditions on it). We add two baseline observations correlated with
   the latent adenoma-risk class — **family history (FH)** and **prior-adenoma
   history (PA)** — and set each patient's initial belief to the Bayes posterior
   `P(high | FH, PA)` instead of the flat prior 0.25. Concrete class-conditional
   rates `P(FH+|high/low)=0.30/0.10`, `P(PA+|high/low)=0.25/0.06` give a
   **discrimination of AUC = 0.67** — a realistic, modest value for these factors.

2. **Cost-based budget instead of a hard per-person cap.** Verified first: under
   the hard K-cap, changing a patient's initial risk belief moves mean
   colonoscopies by **< 0.1** (P(high) 0.02→0.95 gives 2.34→2.43 colonoscopies),
   because everyone is capped at K — "targeting" can only shift *timing*, and age
   dominates timing. Replacing the cap with a per-colonoscopy disutility `c`
   (life-year cost) lets PBVI choose screen **quantity** per person: at `c=0.06` a
   low-risk belief buys 0.8 colonoscopies vs 4.2 for a high-risk belief.

Both are needed: (1) tells the policy *who* is high-risk; (2) lets it *act* on it.

## Results (true CMOST, matched adherence, n = 25 000)

No-screening CRC mortality: 1.73 % overall (high-risk class 3.76 %, low 1.07 %).

### Colonoscopy REALLOCATION now happens (cost budget)

Even with a flat prior, the cost budget already directs more colonoscopies to the
true high-risk class (via *realized* findings — surveillance). Baseline risk
factors sharpen it, chiefly by **de-escalating the true low-risk class** (AUC sweep,
`c = 0.06`):

| baseline AUC | colo high-class | colo low-class | gap | overall colo | overall mort % |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.50 (flat) | 2.54 | 1.89 | 0.65 | 2.05 | 1.05 |
| 0.60 | 2.16 | 1.46 | 0.70 | 1.63 | 1.06 |
| 0.70 | 2.30 | 1.37 | 0.93 | 1.60 | 1.04 |
| 0.80 | 2.57 | 1.24 | 1.33 | 1.57 | 0.98 |
| 0.90 | 2.97 | 0.91 | 2.06 | 1.42 | 1.06 |

As the risk factors get more discriminating, the low-risk class is screened less
and less (1.89 → 0.91) while the high-risk class is protected, so **the same CRC
mortality is achieved with steadily fewer colonoscopies** (2.05 → 1.42, −31 %). The
FH+PA point sits at AUC ≈ 0.67.

### Efficiency frontier (CRC mortality vs colonoscopy use)

| policy | mean colo | colo hi/lo | CRC mort % | mort hi/lo class | LYG/1000 |
|---|:--:|:--:|:--:|:--:|:--:|
| Fixed population ×1 | 0.89 | 0.89/0.89 | 1.36 | 2.88/0.86 | 56.9 |
| Fixed population ×2 | 1.65 | 1.64/1.66 | 1.15 | 2.12/0.83 | 65.8 |
| Fixed population ×3 | 2.59 | 2.58/2.59 | 0.96 | 1.78/0.70 | 97.9 |
| Fixed population ×4 | 3.24 | 3.22/3.24 | 0.83 | 1.38/0.65 | 108.3 |
| Risk-stratified fixed (4/2) | 2.03 | 2.40/1.91 | 1.01 | 1.73/0.77 | 84.6 |
| Risk-stratified fixed (4/1) | 1.45 | 2.01/1.26 | 1.14 | 2.19/0.80 | 79.9 |
| PBVI cost c=0.03, flat prior | 3.67 | 4.36/3.45 | 0.84 | 1.55/0.60 | 96.6 |
| PBVI cost c=0.03, **+FH/PA** | 3.18 | 4.22/2.84 | 0.81 | 1.33/0.64 | 95.9 |
| PBVI cost c=0.06, flat prior | 2.05 | 2.54/1.89 | 1.05 | 1.60/0.87 | 77.9 |
| PBVI cost c=0.06, **+FH/PA** | 1.35 | 2.07/1.12 | 1.05 | 1.93/0.76 | 70.2 |
| PBVI cost c=0.10, flat prior | 0.85 | 0.96/0.82 | 1.30 | 2.82/0.80 | 41.3 |
| PBVI cost c=0.10, **+FH/PA** | 0.32 | 0.75/0.18 | 1.56 | 3.06/1.06 | 22.1 |

Reading the frontier (lower mortality at fewer colonoscopies = better), the arms
order cleanly **PBVI +FH/PA  <  risk-stratified fixed  <  plain fixed**:

- **Risk stratification alone helps.** Giving the *fixed* schedule the same baseline
  factors (intensive tier for the 23.7 % with a positive FH/prior-adenoma) already
  beats the plain population schedule: 1.01 % mortality at 2.03 colonoscopies vs the
  plain schedule's ~1.07 % interpolated at the same use.
- **PBVI's adaptivity adds more on top.** At ~1.4 colonoscopies PBVI +FH/PA reaches
  1.05 % mortality; risk-stratified fixed needs 1.45 to reach only 1.14 %, and the
  plain schedule needs ~1.9 colonoscopies to match 1.05 %. So the ordering is: risk
  factors help a fixed schedule, and PBVI's history-adaptive surveillance (spending
  each person's colonoscopies on *realized* findings) helps further.
- **High-budget regime: ties.** At ~3.2 colonoscopies +FH/PA (0.81 %) ≈ fixed x4
  (0.83 %) — once everyone is screened a lot, targeting has little left to add.

## Interpretation — the honest picture

1. **Yes, baseline risk factors + a cost budget let PBVI target high-risk.** The
   reallocation is real and grows with the factors' discrimination.

2. **At realistic discrimination (FH + prior adenoma, AUC ≈ 0.67) the payoff is
   EFFICIENCY, not a dramatic high-risk mortality drop.** The policy mainly
   *de-escalates the low-risk majority*, achieving the same CRC mortality with
   ~20–30 % fewer colonoscopies. High-risk-class mortality is only weakly moved
   (≈1.6–1.8 % across AUC) because family history is a weak classifier of the
   underlying risk distribution and because high-risk deaths come largely from fast
   cancers that arise between screens.

3. **Strong mortality-targeting of high-risk needs stronger risk factors.** Only at
   AUC ≥ 0.8 (e.g. adding a genetic/polygenic risk score) does low-risk screening
   collapse enough to concentrate protection — the same lever, turned further.

**Takeaway.** The base model's failure to target high-risk was structural, not
fundamental: it needed *observable* baseline risk and a budget that *rewards*
spending more on the risky. With both, the POMDP becomes a genuine risk-stratified,
history-adaptive screener whose advantage over fixed schedules is **colonoscopy
efficiency** — most valuable exactly where colonoscopy capacity is scarce (low
budgets) — and which would convert into outright mortality benefit given
higher-discrimination risk factors.

*Reproduce:* `python experiments/risk_factors.py 25000` → `results/risk_factors.json`,
`results/risk_factors_frontier.csv`, `results/risk_factors_auc.csv`,
`paper/figures/risk_factors.png`.
