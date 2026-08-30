<!-- numbers from results/prs_targeting.json (n=25000); filled after the full run. -->

# A polygenic risk score turns on mortality-targeting

**Question.** `results_risk_factors.md` showed that baseline family-history +
prior-adenoma factors (discrimination AUC ≈ 0.67), together with a cost-based
budget, let PBVI *reallocate* colonoscopies toward high-risk — but at that
discrimination and a mid cost the payoff was **efficiency** (same mortality, fewer
colonoscopies), not a **mortality** reduction in the high-risk class. Does a
stronger risk factor — a **polygenic risk score (PRS)** reaching AUC ≥ 0.8 —
actually cut CRC mortality in the truly high-risk class?

**Design.** Two limits are removed together: (i) the baseline signal's
discrimination is pushed to AUC = 0.80 / 0.90 (PRS ± family history), with an
**oracle (AUC → 1.0)** upper bound; and (ii) the per-colonoscopy cost is set
**low** (c = 0.02–0.03) so an identified high-risk patient can receive 5–6 lifetime
colonoscopies — genuine surveillance — rather than the ~2.5 a higher cost allows.
Outcomes are reported **by true risk class** in the true CMOST environment (matched
adherence, paired CRN, n = 25 000). The fixed population schedules give the
reference high-risk-class mortality. (`experiments/prs_targeting.py`.)

## Result — targeting turns on at AUC ≥ 0.8

No-screening high-risk-class CRC mortality **3.76 %**. Fixed reference: **x3 = 1.78 %**
@ 2.59 total colonoscopies; **x4 = 1.38 %** @ 3.24. High-class n ≈ 6 175 (SE ≈ 0.14 pp).

**Cost c = 0.03 (surveillance-enabling), by true risk class:**

| baseline AUC | total colo | colo high / low | **mort high-class %** | mort low-class % | overall mort % |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.50 (none) | 3.67 | 4.36 / 3.45 | 1.55 | 0.60 | 0.84 |
| 0.67 (FH+PA) | 3.52 | 4.35 / 3.25 | 1.52 | 0.63 | 0.85 |
| 0.80 (PRS) | 3.30 | 4.62 / 2.87 | **1.18** | 0.71 | 0.82 |
| 0.90 (PRS+FH) | 3.02 | 4.99 / 2.37 | **1.13** | 0.63 | 0.75 |
| 1.00 (oracle) | 2.42 | 5.87 / 1.29 | **0.84** | 0.73 | 0.76 |

(c = 0.02, more colonoscopies overall, gives the same pattern with high-class
mortality 1.36 → 1.10 → 1.07 → **0.83** → 0.87 across AUC 0.50 → 1.00.)

Reading:

- As discrimination rises, the policy **concentrates colonoscopies on the true
  high-risk class** (colo_high 4.36 → 5.87) and **de-escalates the low-risk class**
  (colo_low 3.45 → 1.29), so **total** colonoscopy use actually *falls* (3.67 → 2.42)
  even as high-risk surveillance intensifies.
- **High-risk-class CRC mortality falls once AUC ≥ 0.8** — from 1.55 % at the flat
  prior to 1.18 % (AUC 0.8) and 1.13 % (AUC 0.9), a ~0.4 pp (≈ 3 SE) drop — passing
  **below** the fixed x4 benchmark (1.38 %) **at lower total colonoscopy use**
  (3.02 vs 3.24). The oracle reaches 0.84 % (≈ 5 SE below the flat prior).
- At realistic FH+PA discrimination (AUC ≈ 0.67) high-class mortality (1.52 %) is
  **indistinguishable from the flat prior** — confirming that the earlier
  "efficiency, not mortality" result (`results_risk_factors.md`) was a discrimination
  limit, not a structural one. Family history + prior adenoma are simply too weak a
  classifier; a PRS crossing AUC ≈ 0.8 is what flips the switch.

## Interpretation

1. **Mortality-targeting is real but gated on two things**: strong enough risk
   discrimination (a PRS, AUC ≥ 0.8 — family history alone is too weak) **and** a
   budget structure that lets the policy spend more on the identified high-risk
   (low per-colonoscopy cost, no hard per-person cap).
2. Given both, the POMDP implements exactly the clinically desired behaviour —
   **intensive surveillance for the genetically high-risk, de-escalation for the
   low-risk** — and reduces high-risk-class CRC mortality below any fixed population
   schedule at equal or lower total colonoscopy use.
3. This closes the arc: the base model's inability to help high-risk patients
   (`results_subgroups.md`) was not fundamental. It reflected (a) weak observability
   of latent risk and (b) a hard budget cap. Supplying an observable PRS and a
   cost-based budget removes both, and the adaptive advantage that was invisible at
   matched adherence and matched budget becomes an outright mortality benefit in the
   population that screening is meant to protect.

**Caveat.** The PRS here is a *virtual* risk factor parameterized only by its
discrimination (AUC); we do not model a specific assay. The point is the
*sensitivity of the policy's value to discrimination*, which identifies AUC ≈ 0.8 as
the threshold where mortality-targeting becomes worthwhile in this model.

*Reproduce:* `python experiments/prs_targeting.py 25000` → `results/prs_targeting.json`,
`results/prs_targeting.csv`, `paper/figures/prs_targeting.png`.
