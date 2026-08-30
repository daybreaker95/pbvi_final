<!-- numbers from results/auroc_sweep.json (n=50,000).
     Reproduce: python experiments/auroc_sweep.py 50000 --workers 5 -->

# How much risk discrimination does risk-stratified screening need, and which tests supply it?

**Question.** `results_prs.md` showed, on a four-point grid plus an oracle, that
high-risk-class mortality-targeting "turns on" somewhere near AUC 0.8. Two things
were left open. First, the grid was too coarse to locate the turn-on, and the
oracle arm (AUC → 1.0) is not a reachable operating point — no assay perfectly
observes a *lifelong latent adenoma-risk class*. Second, an AUROC is not a
screening programme: nothing said which **combination** of risk factors and tests
puts a programme at a given AUROC, or what the marginal value of the next test is.

**Design.** Discrimination is swept on an 11-point grid from 0.50 to a deliberate
ceiling of **0.85** — an ambitious but defensible target for a combined
questionnaire + PRS + biomarker panel — at **four colonoscopy budgets**
(per-colonoscopy cost c = 0.02 / 0.03 / 0.06 / 0.10), in two tracks:

* **Track A — continuous.** An abstract calibrated risk score at each AUROC. This
  is the response curve: what does the POMDP do with exactly this much information?
* **Track B — panels.** Nine named risk-factor / risk-test **combinations**
  (`experiments/risk_panels.py`), each rung adding exactly one modality, combined
  by naive Bayes into a calibrated posterior `P(high | markers)`. Each panel's
  AUROC is **measured** from the simulated patients, not assumed.

All 85 arms run in the true CMOST environment with matched adherence and paired
common random numbers (same patients, same marker draws), reported **by true risk
class**. n = 50 000 (high-risk class n ≈ 12 400, SE on its mortality ≈ 0.10 pp).

References: no screening — overall **1.67 %**, high-class **3.64 %**, low-class
1.03 % (high-class fraction 0.248). Fixed population schedules:

| schedule | total colo | high-class mort % | overall mort % |
|---|:--:|:--:|:--:|
| Fixed ×1 | 0.89 | 2.84 | 1.35 |
| Fixed ×2 | 1.66 | 2.15 | 1.10 |
| Fixed ×3 | 2.59 | 1.78 | 1.00 |
| Fixed ×4 | 3.24 | 1.30 | 0.83 |

---

## Track A — the response curve is budget-dependent, and non-monotone when the budget is tight

High-risk-class CRC mortality (%) and total colonoscopies per person:

| AUROC | c=0.02 mort / colo | c=0.03 mort / colo | c=0.06 mort / colo | c=0.10 mort / colo |
|:--:|:--:|:--:|:--:|:--:|
| 0.50 (none) | 1.16 / 4.08 | 1.30 / 3.64 | 1.60 / 2.02 | **2.56** / 0.86 |
| 0.55 | 1.18 / 4.10 | 1.30 / 3.64 | 1.86 / 1.57 | 2.77 / 0.72 |
| 0.60 | 1.11 / 4.19 | 1.24 / 3.55 | 1.86 / 1.53 | 2.82 / 0.58 |
| 0.65 | 1.02 / 4.26 | 1.20 / 3.47 | 1.80 / 1.53 | **2.83** / 0.52 |
| 0.70 | **0.97** / 4.29 | 1.18 / 3.41 | 1.74 / 1.53 | 2.76 / 0.52 |
| 0.725 | 0.98 / 4.28 | 1.12 / 3.38 | 1.72 / 1.53 | 2.69 / 0.53 |
| 0.75 | 1.00 / 4.26 | 1.22 / 3.34 | 1.69 / 1.53 | 2.59 / 0.54 |
| 0.775 | 0.98 / 4.23 | 1.16 / 3.29 | 1.72 / 1.51 | 2.47 / 0.55 |
| 0.80 | 1.02 / 4.19 | 1.10 / 3.23 | 1.74 / 1.49 | 2.31 / 0.56 |
| 0.825 | 0.98 / 4.13 | 1.14 / 3.17 | 1.68 / 1.47 | 2.22 / 0.58 |
| **0.85** | 0.99 / 4.07 | **1.10** / 3.10 | 1.70 / 1.43 | **2.20** / 0.59 |

Three distinct regimes, and only one of them matches the story in `results_prs.md`:

**1. Loose budget (c = 0.02): discrimination saturates at AUROC ≈ 0.70.**
High-class mortality falls 1.16 → 0.97 % by AUROC 0.70 (≈ 2 SE) and is then flat to
0.85. Past 0.70 the extra information is spent on **volume**, not mortality: total
colonoscopies peak at 4.29 (AUROC 0.70) and fall back to 4.07 at 0.85, because
high-class use keeps rising (4.72 → 5.34) while low-class use falls away
(4.03 at the peak → 3.65). When colonoscopy is nearly free, a moderate
classifier already buys everything the mortality axis has to give.

**2. Mid budget (c = 0.03): discrimination pays on both axes, and beats every fixed
schedule.** High-class mortality falls 1.30 → 1.10 % while total colonoscopies fall
3.64 → 3.10. At AUROC 0.85 this is **below Fixed ×4 (1.30 % @ 3.24)** on both axes
simultaneously — lower high-risk mortality at 4 % fewer colonoscopies. Note that at
AUROC 0.50 the same policy needs 3.64 colonoscopies to match Fixed ×4's mortality,
so the whole gain is attributable to the risk information.

**3. Tight budget (c = 0.06, 0.10): a valley of harm at intermediate discrimination.**
This is the finding the earlier coarse grid could not see. At c = 0.10, high-class
mortality **rises** from 2.56 % (flat prior) to 2.83 % at AUROC 0.65 — a ~2 SE
*deterioration* — and only crosses back below the flat prior at AUROC ≈ 0.775,
reaching 2.20 % at 0.85. At c = 0.06 the same dip appears (1.60 → 1.86 % at AUROC
0.55–0.60) and never fully recovers by 0.85 (1.70 %).

The mechanism is visible in the allocation. At c = 0.10, going from AUROC 0.50 to
0.65 cuts low-class use 0.82 → 0.44 but barely moves high-class use (0.96 → 0.77),
so total volume collapses 0.86 → 0.52. A weak classifier under a tight budget
**de-escalates faster than it can escalate**, and because it is weak, a large share
of the truly high-risk sit in the de-escalated group. Only past AUROC ≈ 0.75 does
the policy start buying volume *back* for the people who need it: total use rises
0.52 → 0.59 while high-class use nearly triples (0.77 → 1.49) and low-class use
keeps falling (0.44 → 0.30).

**Practical reading:** the answer to "how far do we need to push discrimination"
depends entirely on colonoscopy capacity. Where capacity is ample, ~0.70 suffices.
Where capacity is scarce — exactly where risk stratification is most attractive —
a half-built classifier (AUROC 0.60–0.75) can leave the high-risk class **worse off
than screening everyone uniformly**, and only a panel reaching ≳ 0.80 turns the
budget saving into a mortality gain.

## Track B — which combinations get there, and what each added test is worth

Measured AUROC of each panel and its outcome at c = 0.03:

| panel | added modality | measured AUROC | total colo | colo hi / lo | high-class mort % | overall mort % |
|---|---|:--:|:--:|:--:|:--:|:--:|
| P1 | family history alone | 0.604 | 3.58 | 4.28 / 3.35 | 1.30 | 0.78 |
| P2 | + prior-adenoma history *(today's baseline)* | 0.670 | 3.18 | 4.22 / 2.84 | 1.25 | 0.79 |
| P3 | + lifestyle/environmental E-score | 0.702 | 3.32 | 4.33 / 2.98 | 1.21 | 0.79 |
| P4 | + PRS, current generation (~140 loci) | 0.739 | 3.36 | 4.44 / 3.01 | 1.17 | 0.74 |
| P5 | PRS → genome-wide / multi-ancestry | 0.765 | 3.32 | 4.50 / 2.93 | 1.25 | 0.78 |
| P6 | + faecal microbiome signature | 0.775 | 3.29 | 4.53 / 2.89 | 1.18 | 0.76 |
| P7 | + quantitative faecal haemoglobin | 0.803 | 3.23 | 4.61 / 2.78 | 1.18 | 0.76 |
| P8 | + multi-target stool DNA/RNA | 0.826 | 3.17 | 4.67 / 2.67 | 1.19 | 0.78 |
| P9 | + blood methylated cfDNA score | **0.846** | 3.11 | 4.73 / 2.57 | **1.02** | 0.72 |

- **Today's clinical baseline (FH + prior adenoma) reaches only 0.670.** Reaching
  0.85 needs the whole stack: a lifestyle score, a next-generation PRS, and at
  least two of the emerging assays used as *risk stratifiers* rather than as
  yes/no tests. No single addition moves the AUROC more than ~0.07.
- **The marginal AUROC gain per added modality shrinks** (+0.066, +0.032, +0.037,
  +0.026, +0.010, +0.028, +0.023, +0.020) because the markers are conditionally
  independent and combine as `d² = Σ dᵢ²` — the *fifth* test adds much less than
  the second. This is the arithmetic reason 0.85 is a hard ceiling rather than a
  waypoint.
- **The panels land on the Track-A curve** (figure panel f): P9 at AUROC 0.846
  gives 1.02 % vs Track A's 1.10 % at 0.85, within ~1 SE. The abstract knob is a
  fair summary of a real combination — at this budget.

**But at a tight budget, the *shape* of the risk distribution matters as much as
its AUROC.** At c = 0.10 the discrete panels are far more aggressive de-escalators
than a Gaussian score of the same AUROC:

| | Track A @ AUROC 0.60 | P1 (FH only, AUROC 0.604) |
|---|:--:|:--:|
| total colo | 0.58 | **0.16** |
| high-class mortality | 2.82 % | **3.28 %** |

Family history alone yields a two-point posterior: 85 % of people (the family-history-negative) sit at a
posterior *below* the population prior, and under a tight budget that pushes nearly
all of them to zero screening. A continuous score with the same AUROC spreads
people out, so far fewer fall under the screen-worthy threshold. Two classifiers
with identical discrimination are therefore **not** interchangeable in a
budget-constrained programme — a caveat that a pure AUROC-based target misses
entirely.

The tight-budget panel ladder is also where the added tests pay most, precisely
because they convert the coarse two-point posterior into a continuous one
(c = 0.10): high-class mortality 3.28 % (P1) → 3.10 (P2) → 2.86 (P3) → 2.65 (P4)
→ 2.50 (P5) → 2.39 (P7) → **2.40 (P9)**, against the flat-prior 2.56 % and
Fixed ×1's 2.84 % @ 0.89 colonoscopies. The full panel reaches 2.40 % at **0.59**
colonoscopies per person — a third fewer than Fixed ×1, for markedly lower
high-risk mortality.

## Best operating points found

| arm | total colo | high-class mort % | overall mort % | vs fixed |
|---|:--:|:--:|:--:|---|
| PBVI c=0.03, AUROC 0.85 | 3.10 | 1.10 | 0.74 | beats Fixed ×4 (1.30 % @ 3.24) on both axes |
| PBVI c=0.03, panel P9 | 3.11 | 1.02 | 0.72 | same, with a nameable panel |
| PBVI c=0.06, AUROC 0.85 | 1.43 | 1.70 | 1.03 | Fixed ×3's mortality (1.78 %) at **45 % fewer** colonoscopies |
| PBVI c=0.10, AUROC 0.85 | 0.59 | 2.20 | 1.28 | beats Fixed ×1 (2.84 % @ 0.89) at 34 % fewer |

Note the overall (population) mortality is nearly flat across the sweep at the
looser budgets (c = 0.03: 0.78 → 0.74 %). Discrimination **reallocates**; it does
not conjure much extra population-level benefit when everyone is already screened
a lot. Its value is concentrated in the high-risk class and in colonoscopy volume.

## Caveats

1. **What these AUROCs measure.** The target label is CMOST's *latent lifelong
   adenoma-risk class* (`individual_risk` above/below the top-25 % cut), not
   prevalent cancer. Published discrimination figures for PRS, stool DNA and cfDNA
   assays are mostly for *detecting existing neoplasia*, a different and generally
   easier label. The per-marker values in `risk_panels.py` are therefore
   deliberately conservative **scenario anchors** — "a test of this kind, used as a
   risk stratifier, reaching this much risk-class discrimination" — not
   reproductions of any single reported statistic. The scientific content is the
   sweep; the panels locate plausible, nameable points on it.
2. **Conditional independence.** Markers are combined by naive Bayes. Real panels
   share information (a lifestyle score and a microbiome signature are correlated),
   so the measured panel AUROCs here are, if anything, optimistic for a given set
   of components — which strengthens rather than weakens the "0.85 is a ceiling"
   conclusion.
3. **Numbers differ from `results_prs.md`.** That write-up's arms were produced
   before the POMDP's hidden state grew from 6 to 7 clinical states per risk block,
   and the alpha-vector caches under `results/policies/` had gone stale against the
   current model (loading one now raises a matmul-width error).
   `solve_pbvi_cost` gained a width check and re-solves instead of loading a stale
   cache, so the arms here are all from the current model. The qualitative story
   (targeting strengthens with discrimination; the payoff is efficiency at
   realistic FH+PA discrimination) survives; the exact percentages do not match.
4. **Age is observed, sex is not modelled here.** As in the sibling experiments,
   the risk signal is an age-independent lifelong multiplier; the policy already
   conditions on age directly.

*Figure:* `paper/figures/auroc_sweep.png` — (a) high-class mortality vs AUROC per
budget, (b) reallocation by true class, (c) total colonoscopy use, (d) efficiency
frontier, (e) measured panel AUROCs, (f) panels vs the Track-A curve.

*Reproduce:* `python experiments/auroc_sweep.py 50000 --workers 5` →
`results/auroc_sweep.json`, `results/auroc_sweep_gauss.csv`,
`results/auroc_sweep_panels.csv`, `paper/figures/auroc_sweep.png`.
Panel definitions and their measured AUROCs alone:
`python -m experiments.risk_panels`.
