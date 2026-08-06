# Transition-matrix vs CMOST microsimulation: screening validation

Does the empirically-estimated transition matrix reproduce the CMOST natural
history under **no screening**, **colonoscopy q10y (ages 50,60,70)** and
**q5y (50,55,60,65,70,75)**, for 8 clinical metrics?

- Avg age at death (all), Avg age at CRC death, CRC deaths /100k,
  CRC incidence /100k, Stage I/II/III/IV % at diagnosis.

Ground truth = the individual CMOST engine (`env/cmost_individual.py`).
The stock 6-state matrix (`transitions_cmost13.npz`) cannot produce 6 of these
(single `Dead` absorbing state, no cancer stage), so we build a **metric-complete
augmented cohort**: `Dead` is split into Other/CRC death and cancer stage is
tracked (preclinical P1–P4, clinical dx by stage C1–C4). Screening is applied as
a detect+treat operator (`d_EA, d_AA, d_PC` from `pomdp_effects.npz`).

## Scripts (each runs the 3 scenarios at `-n 100000`)

| script | matrix | fixes |
|---|---|---|
| `verify_screening_yearly.py` | 14-state, **annual** step | baseline; includes snapshot diagnostic |
| `verify_screening_quarterly.py` | 14-state, **quarterly** step | (1) incidence, (3) stage IV |
| `verify_screening_tauphase.py` | + clinical **τ-phases**, + stage-specific `d_PC` | (4) CRC-death timing, (b) detection |

## Findings (diff = transition-matrix − microsim, n=100k; none / q10y / q5y)

| metric | annual | quarterly | quarterly + τ-phase |
|---|---|---|---|
| Avg age at CRC death | +0.1 / −2.3 / −3.6 | +0.0 / −2.8 / −4.2 | −0.0 / **−1.2 / −2.4** |
| CRC incidence /100k | −758 / −557 / −481 | −2 / −55 / −119 | −2 / −55 / −118 |
| Stage IV % | −2.3 / −3.5 / −5.1 | 0.0 / −1.3 / −2.9 | 0.0 / −1.4 / −3.1 |

Under **no screening the τ-phase cohort matches the microsim on all 8 metrics**
(incidence −2/100k, every stage within 0.1 pt, CRC-death age within 0.02 yr).

## Root causes (each proven, not assumed)

1. **(1) CRC incidence undercount & (3) Stage IV undercount = annual time
   discretization.** Cancers that arise, are diagnosed and kill within one year
   are never seen in the clinical state at a year boundary, so an annual matrix
   routes them straight to `Dead`. The `yearly` diagnostic shows the cohort
   equals the microsim's *annual-snapshot* incidence exactly; the whole gap vs
   the (event-based) truth is discretization. **Quarterly stepping removes it.**
   (Deaths are still captured, so the annual matrix *overstates* case fatality.)

2. **(4) CRC-death timing/age = memoryless post-diagnosis survival.** CMOST draws
   a fixed time-to-CRC-death at diagnosis (front-loaded, then cured). A plain
   clinical state with a constant hazard misplaces the age of CRC death (−2 to
   −4 yr under screening). **Subdividing the clinical state by τ = quarters since
   diagnosis halves this error** (residual is the transfer limitation below).

3. **(b) Stage-specific `d_PC` is faithful but small.** `d_PC[I..IV] ≈
   [0.935, 0.935, 0.935, 0.985]` reproduces the engine's cancer-detection line
   (`Colo_Detection[stage-1] × P(loc≥reach)`; cancers use no location-detection
   factor, unlike polyps — the old single 0.91 wrongly multiplied by `mean_loc`).
   Effect is minor because CMOST cancer sensitivity is nearly stage-flat.

4. **Residual screening errors are NOT natural-history fidelity.** The cohort's
   *prevalent* preclinical stage distribution matches the microsim to <1.5 pt at
   every screening age, so the memoryless preclinical sojourn is ruled out. The
   remaining Stage I +3 pt / avg-CRC-death −1–2 yr come from (i) estimating
   screening-relevant clinical dynamics from a **no-screening** cohort (young
   early-stage cancers are sparse there) and (ii) the annual instantaneous,
   stage-flat screening **operator** vs the engine's per-lesion quarterly
   colonoscopy.

## Verdict

The transition matrix reproduces CMOST's **all-cause survival and total CRC
mortality** well in all scenarios, and — once estimated **quarterly** with
**τ-phased** clinical states — reproduces **CRC incidence, stage-at-diagnosis and
CRC-death age** essentially exactly under natural history, with small,
well-attributed residuals under screening.
