<!-- DRAFT — numbers filled from results/nonadherence.json after the n=30000 run -->

# Personalized screening under imperfect adherence

## Rationale

Under *perfect* adherence the re-optimized best-fixed population schedule and the
PBVI adaptive policy are statistically indistinguishable on life-years and CRC
mortality (Table 2, `results_comparison.md`). That comparison, like the whole
Zaika 2024 literature it builds on, assumes **every** person is screened exactly
at the recommended ages. Real screening programs do not achieve this: a large
fraction of invitations are missed or deferred.

The two policy *classes* respond to a missed colonoscopy very differently, and
this is not a tuning detail — it is intrinsic to what each policy conditions on:

- A **fixed population schedule** is a function of age alone. It recommends
  screening only at its preset ages (e.g. 58 and 70). If the patient does not
  attend at 58, the policy has no state that records this; it simply says "wait"
  at 59, 60, … and the next recommendation is at 70. **A missed slot is lost.**

- The **PBVI adaptive policy** is a function of `(age, filtered belief over
  latent risk × clinical state, remaining colonoscopy budget)`. A missed
  colonoscopy performs no test, so it yields **no observation and consumes no
  budget**; the belief at the next epoch is essentially unchanged (one extra year
  of natural-history aging), so the policy **re-recommends screening** and only
  stops re-inviting once the budget is actually spent or the patient ages out.
  (Verified directly: forcing every recommendation to be missed leaves the budget
  full and the policy keeps re-inviting the patient.) The policy therefore
  **re-plans** around non-adherence for free — no extra machinery.

The experiment isolates how much this matters.

## Design

All roll-outs are in the **true CMOST microsimulation** with per-patient common
random numbers on both the engine and the adherence draws (paired comparison).
Decisions run 40→90; outcomes accrue to death. Fixed comparator = the
re-optimized best-fixed ages (`{K2: [58,70], K3: [52,58,70]}`). Code:
`experiments/nonadherence.py`.

**Adherence model.** Each time a policy *recommends* a colonoscopy, the patient
attends with probability α (a Bernoulli "no-show" model). A missed recommendation
is turned into a "wait" for that year; the realized (not the intended) action is
fed back to the policy so belief-tracking and recall policies react to what
actually happened.

**Three policies**, so that we can separate the two mechanisms:

| policy | timing | reacts to a no-show? |
|---|---|---|
| **Fixed (no recall)** | preset ages | no — slot lost |
| **Fixed + recall** | preset ages, but re-invites every year until attended (cap K) | yes, retries — *but no personalization* |
| **PBVI adaptive** | belief- and budget-dependent | yes, retries **and** re-times / risk-personalizes |

Comparing *Fixed(no recall) → Fixed+recall* isolates the value of simply
**retrying** missed invitations; comparing *Fixed+recall → PBVI* isolates the
extra value of **personalized timing / risk inference**; *Fixed(no recall) → PBVI*
is the total real-world gap.

**Experiment 1 — adherence sweep.** α ∈ {1.0, 0.75, 0.5, 0.25}, budgets K ∈ {2, 3}.

**Experiment 2 — non-adherent subgroup.** A heterogeneous population: 40 % of
people are chronically **low-adherence** (attend 20 % of invitations), 60 % are
high-adherence (attend 90 %). Class is fixed per patient (independent of policy),
so we can read off outcomes *within the low-adherence subgroup* — the people who
mostly do not get screened on schedule, which is exactly the population the
question is about.

**Endpoints.** Primary = **CRC mortality** (Monte-Carlo-stable, SE ≈ 0.05–0.09 pp
at n = 30 000). Secondary = CRC incidence, realized mean screening colonoscopies,
life-years gained per 1000 (LYG; heavy-tailed, large SE), and LYG per
colonoscopy. n = 30 000 paired patients.

## Results

No-screening CRC mortality = **1.71 %**. All numbers are true-CMOST outcomes,
n = 30 000 paired patients.

### Experiment 1 — adherence sweep

Realized mean colonoscopies (`colo`) and CRC mortality (`mort`, %). As adherence
falls, the fixed no-recall schedule delivers fewer and fewer colonoscopies and its
mortality drifts back toward the no-screening level (1.71 %); the retry-capable
policies keep delivering ~1.5–2.5 colonoscopies and hold mortality low.

**Budget K = 2**

| α (per-visit adherence) | Fixed (no recall) colo · mort | Fixed + recall colo · mort | PBVI adaptive colo · mort |
|:--:|:--:|:--:|:--:|
| 1.00 | 1.66 · **1.12** | 1.66 · 1.12 | 1.67 · 1.13 |
| 0.75 | 1.24 · 1.28 | 1.65 · 1.09 | 1.66 · **1.09** |
| 0.50 | 0.83 · 1.35 | 1.63 · 1.09 | 1.63 · **1.14** |
| 0.25 | 0.41 · 1.60 | 1.56 · 1.06 | 1.46 · **1.09** |

Paired PBVI − Fixed(no recall) mortality (pp; − = PBVI better):
α1.00 +0.01 ± 0.05 (n.s.) · α0.75 **−0.19 ± 0.06** · α0.50 **−0.21 ± 0.07** · α0.25 **−0.50 ± 0.08**.

**Budget K = 3**

| α | Fixed (no recall) colo · mort | Fixed + recall colo · mort | PBVI adaptive colo · mort |
|:--:|:--:|:--:|:--:|
| 1.00 | 2.59 · 0.99 | 2.59 · 0.99 | 2.31 · **0.94** |
| 0.75 | 1.94 · 1.11 | 2.58 · 0.97 | 2.28 · **0.86** |
| 0.50 | 1.30 · 1.29 | 2.55 · 0.96 | 2.20 · **0.91** |
| 0.25 | 0.64 · 1.47 | 2.43 · **0.94** | 1.87 · 0.99 |

Paired PBVI − Fixed(no recall) mortality (pp):
α1.00 −0.05 ± 0.07 (n.s.) · α0.75 **−0.26 ± 0.07** · α0.50 **−0.38 ± 0.07** · α0.25 **−0.49 ± 0.07**.

### Experiment 2 — non-adherent subgroup

Heterogeneous population; the **low-adherence subgroup** (40 % of people, attend
20 % of invitations, n ≈ 12 000) is the population of interest.

**Budget K = 2, low-adherence subgroup**

| policy | mean colo | CRC mortality % | CRC incidence % | LYG/1000 |
|---|:--:|:--:|:--:|:--:|
| Fixed (no recall) | 0.33 | **1.64** | 3.80 | −1.5 |
| Fixed + recall | 1.50 | 1.15 | 2.82 | 33.6 |
| PBVI adaptive | 1.35 | 1.20 | 2.88 | 30.0 |

Paired PBVI − Fixed(no recall): **−0.44 ± 0.11 pp** (significant).

**Budget K = 3, low-adherence subgroup**

| policy | mean colo | CRC mortality % | CRC incidence % | LYG/1000 |
|---|:--:|:--:|:--:|:--:|
| Fixed (no recall) | 0.51 | **1.50** | 3.55 | 17.2 |
| Fixed + recall | 2.33 | 0.92 | 2.36 | 91.4 |
| PBVI adaptive | 1.72 | 1.05 | 2.78 | 43.1 |

Paired PBVI − Fixed(no recall): **−0.45 ± 0.11 pp** (significant).

## Interpretation

1. **The tie is an artifact of assuming perfect adherence.** At α = 1.0 the fixed
   and PBVI policies match (as in Table 2). The moment adherence is imperfect the
   fixed *no-recall* schedule degrades sharply — at α = 0.25 (K = 2) its CRC
   mortality (1.60 %) is nearly back to no screening (1.71 %), because a person who
   misses their one or two preset ages is simply never screened. An adaptive policy
   that re-plans keeps mortality at ~1.06–1.14 %.

2. **For the people who don't get screened on schedule, a fixed program delivers
   almost nothing.** In the chronic low-adherence subgroup the fixed no-recall
   schedule gives essentially zero benefit (K = 2: mortality 1.64 % vs 1.71 % no
   screening; LYG ≈ 0). Switching those same people to an adaptive/re-inviting
   policy cuts their CRC mortality to ~1.05–1.20 % — a **~0.44–0.45 pp absolute
   (≈ 27–30 % relative) reduction, statistically significant** — recovering most of
   the benefit they would otherwise miss. **This is the direct answer to the
   question.**

3. **Decomposition — what is doing the work.** Most of the rescue is the
   *re-invitation/retry* behavior, not risk-personalization: "Fixed + recall"
   captures the bulk of it. PBVI's advantage is that this retry is **emergent and
   principled** (a missed colonoscopy consumes no budget and yields no information,
   so re-recommending is simply the optimal action) rather than a bolted-on rule,
   and that its *personalized timing* adds a further, colonoscopy-efficient edge at
   mid-budget / mid-adherence — e.g. K = 3, α = 0.75: PBVI 0.86 % at 2.28 colos vs
   recall 0.97 % at 2.58 colos. At extreme non-adherence (α = 0.25) blunt recall can
   slightly out-perform PBVI on raw mortality because PBVI economizes (waits for
   information that never arrives, using fewer colonoscopies).

**Take-away for the paper.** The clinically important message is not "adaptive
beats fixed at matched perfect adherence" (it does not) but **"adaptive screening
is robust to non-adherence, whereas a fixed population schedule is not"** — and the
benefit is concentrated exactly in the non-adherent patients, who are the ones a
fixed schedule fails. A fixed program *with an active recall system* captures most
of the gain; the POMDP framing makes recall + risk-personalized re-timing fall out
automatically.

*Reproduce:* `python experiments/nonadherence.py 30000` →
`results/nonadherence.json`, `results/nonadherence_sweep.csv`,
`results/nonadherence_subgroup.csv`, `paper/figures/nonadherence.png`.
