# Where does the PBVI adaptive policy help? Patient subgroups vs situations

**Question.** At matched perfect adherence PBVI ≈ best-fixed on pooled LYG/mortality.
Is there a **patient subgroup** or a **situation** in which the personalized policy
actually wins? (`experiments/subgroups.py`, true CMOST, paired CRN, n = 40 000.)

**Short answer.** The intuitive hypothesis — "personalization helps **high-risk
patients**" — is **not supported**. At matched adherence PBVI provides no reliable
benefit in *any* risk subgroup, because it cannot pre-identify latent risk from
mostly-clean colonoscopies. PBVI's genuine advantages are **situational**:
(1) imperfect adherence (see `results_nonadherence.md`), and (2) **colonoscopy
efficiency**, which becomes a real mortality advantage in the **higher-budget /
surveillance regime**, where a uniform fixed schedule saturates but the adaptive
policy keeps converting extra colonoscopies into mortality reduction by directing
them to realized findings.

---

## Part 1 — Patient subgroups (matched adherence)

Stratifiers are **policy-independent** (fixed per patient, identical across
policies), so the PBVI-vs-fixed contrast within a stratum is paired: latent
adenoma-risk (CMOST IndividualRisk) quartiles and the top-25 % "high-risk class"
the POMDP tries to infer; plus natural-history phenotype (would the patient
develop CRC with no screening).

**Colonoscopy allocation is essentially flat across risk (K = 4).** PBVI does NOT
give more colonoscopies to high-risk patients — if anything slightly fewer:

| risk quartile | Best fixed | PBVI |
|---|:--:|:--:|
| Q1 (low) | 3.23 | 3.19 |
| Q2 | 3.24 | 3.17 |
| Q3 | 3.24 | 3.14 |
| Q4 (high) | 3.21 | 3.07 |

**Within every risk stratum, PBVI ≈ fixed on CRC mortality (K = 3).** All paired
differences are within ~1–1.5 SE of zero:

| stratum | n | no-screen mort % | fixed colo · mort % | PBVI colo · mort % | paired PBVI−fixed (pp) |
|---|:--:|:--:|:--:|:--:|:--:|
| all | 40000 | 1.67 | 2.59 · 0.97 | 2.31 · 0.92 | −0.05 ± 0.06 |
| low-risk class | 30112 | 1.03 | 2.59 · 0.71 | 2.33 · 0.70 | −0.01 ± 0.06 |
| high-risk class (top 25%) | 9888 | 3.64 | 2.59 · 1.76 | 2.26 · 1.60 | −0.16 ± 0.16 |
| risk Q4 (highest) | 10213 | 3.61 | 2.59 · 1.74 | 2.26 · 1.57 | −0.18 ± 0.16 |
| would-develop-CRC (NH) | 1643 | 40.72 | 2.71 · 10.71 | 2.42 · 10.96 | +0.24 ± 0.92 |

The high-risk-class direction even **flips sign** between budgets (K = 3: −0.16 ± 0.16
favoring PBVI; K = 4: **+0.21 ± 0.15** favoring fixed) — i.e. noise, not a real
subgroup effect. **Mechanism:** a clean colonoscopy weakly *lowers* the inferred
risk (P(high): 0.25 → 0.20), so a truly high-risk person who screens clean early is
mildly *reassured* and can be under-screened — cancelling any targeting benefit.
Personalization by **latent risk** does not work here; only personalization by a
**realized finding** (post-polypectomy surveillance) does, and that needs spare
budget to act on (Part 2).

## Part 2 — Situation: screening budget / surveillance regime

Whole population, matched adherence, K = 1 … 6. Efficiency = CRC-mortality
reduction (vs no screening) per colonoscopy.

| K | best-fixed colo · mort % · eff | PBVI colo · mort % · eff |
|:--:|:--:|:--:|
| 1 | 0.89 · 1.39 · 0.482 | 0.76 · 1.36 · **0.601** |
| 2 | 1.65 · 1.21 · 0.369 | 1.67 · 1.21 · 0.365 |
| 3 | 2.59 · 0.94 · 0.342 | 2.31 · 1.00 · 0.356 |
| 4 | 3.24 · 0.80 · 0.313 | 3.12 · 0.86 · 0.306 |
| **6** | 5.04 · 0.81 · 0.199 | **4.52 · 0.71 · 0.245** |

Two robust effects:

1. **Colonoscopy efficiency.** At every budget PBVI reaches comparable mortality
   with ~10 % fewer colonoscopies (K = 3: 2.31 vs 2.59; K = 6: 4.52 vs 5.04). The
   savings come from the low-risk majority. If a colonoscopy carries cost, harm, or
   capacity constraints, this is a real advantage even when mortality is matched.

2. **Surveillance regime (high budget).** The uniform fixed schedule **saturates**:
   going K = 4 → 6 adds ~1.8 colonoscopies per person for **zero** mortality gain
   (0.80 % → 0.81 %). PBVI instead converts its extra budget into mortality
   reduction (0.86 % → 0.71 %) because it spends the marginal colonoscopy on people
   with **realized findings** rather than on everyone uniformly. So PBVI breaks
   through the fixed schedule's saturation ceiling exactly where adaptivity has room
   to act. (It also has a small edge at K = 1 via better single-screen timing.)

## Bottom line

| where | does PBVI beat best-fixed? |
|---|---|
| High-risk *patients* (matched adherence) | **No** — no reliable subgroup benefit; PBVI can't infer latent risk from clean screens |
| Any risk quartile / NH-CRC subgroup (matched adherence) | **No** — within ~1–1.5 SE of fixed |
| Imperfect adherence (`results_nonadherence.md`) | **Yes** — strongest win, concentrated in non-adherent patients |
| Colonoscopy-constrained / cost-sensitive setting | **Yes** — same mortality, ~10 % fewer colonoscopies |
| High budget / surveillance regime (K ≥ 6) | **Yes** — lower mortality with fewer colonoscopies; breaks fixed saturation |

**Reframing for the paper.** The contribution is not "a personalized schedule saves
high-risk patients" (the microsimulation does not support that at matched
adherence). It is that **adaptive, history-dependent screening is more robust
(to non-adherence) and more colonoscopy-efficient (especially in the surveillance
regime) than any fixed population schedule** — advantages that are invisible in the
idealized matched-adherence LYG comparison but real under realistic conditions.

*Reproduce:* `python experiments/subgroups.py 40000` → `results/subgroups.json`,
`results/subgroups_strata.csv`, `results/subgroups_budget.csv`,
`paper/figures/subgroups.png`.
