# PanhandleStrike — running notes

Chronological working log. Newest entries at the bottom. Study area: Escambia,
Santa Rosa, Okaloosa, Walton, Bay counties, FL.

---

## Key numbers snapshot (current as of 2026-09-15)

Headline anomaly-classifier results, kept up to date as the model evolves.
Full derivation and caveats for the 4-year figures are in
[4-year training window](#4-year-training-window-2022-2025--does-oni-become-interpretable-srcpull_glm_2022_2023py-srcbuild_4yr_featurespy-srcretrain_4yr_modelpy) below.

**ONI (ENSO) coefficient, LogReg, standardized:**

| | 2-year model (2024-25) | 4-year model (2022-25) |
|---|---|---|
| ONI coefficient | +0.93 | **+0.26** |
| p-value | <0.0001 | **9.7e-26** |
| Odds ratio / SD | 2.53 | **1.29** |
| Year variable in model | `is_2025` dummy (confounded w/ ONI) | **none** (dropped by design) |

No run has tested ONI against a competing `C(year)` term in the 4-year model
— see "Deferred" below.

**Anomaly positive rate by year (4-year window, recomputed baselines):**

| Year | Positive rate |
|---|---|
| 2022 | 0.284 |
| 2023 | 0.381 (highest) |
| **2024** | **0.262** (recalculated under 4-yr baseline; not identical to 2-yr-only figure) |
| **2025** | **0.264** (recalculated under 4-yr baseline; not identical to 2-yr-only figure) |
| All 4 years | 0.298 (n=11,780) |

**2026 out-of-time AUC:**

| Model | 2-yr-trained (saved baseline) | 4-yr-trained | Δ |
|---|---|---|---|
| RandomForest | 0.772 | **0.788** | +0.016 |
| LogReg | 0.812 | **0.781** | −0.031 |

**Report framing — use this language, not the raw numbers alone:** the 2-year
LogReg's **0.812 is a broken, saturated result, not a real one** — it flagged
100% of 2026 rows as anomalous (recall 1.0), an artifact of extrapolating an
oversized ONI coefficient past the training range. The 4-year LogReg's **0.781
is the honest, usable number** — 40% flagged vs. a 35% true rate, prec 0.592 /
rec 0.671. Report the 4-year LogReg result as the model; cite the 2-year 0.812
only to explain why it was discarded, never as a competing/better score.

**Deferred (not run — timeline):** ONI-vs-competing-year-term test (adding a
`C(year)` factor to the 4-year LogReg to check whether ONI's significance
survives a direct competing year variable, rather than just the absence of
one). Noted as a possible future refinement, not required for the current
report.

**OPTION B ADOPTED (2026-09-15): the 4-year window is now the single primary
result for the ENTIRE report**, not just the model retrain. Every figure and
statistic uses 4-year (2022-2025) data except one dedicated, clearly-labeled
comparison section (figs 16/18/19) that deliberately keeps the 2-yr-vs-4-yr
contrast as evidence of a real problem found and fixed - not leftover
numbers. Status of everything rebuilt under Option B:

- Dashboard + shortlist rebuilt: **shortlist 42 → 37**; Walton improves
  127→102 (still listed); N. Okaloosa cluster stays 3-and-3 but `...12` drops
  off and `...21` joins.
- June 2025 validation re-confirmed under the shifted baseline (all 5 days
  still trigger).
- Figs 1-2 (data overview), 7-8 (suitability), 15 (6-model comparison, now
  LogReg edges RF but all top 4 within noise), 17 (impurity → **permutation**
  importance) all rebuilt on 4-year data.
- Coastal buffer analysis rebuilt on 4-year flashes (new
  `coastal_buffer_10km_4yr/`): shortlist 37→47 under the buffer; **Walton's
  "drops off the shortlist" claim no longer holds** under the 4-yr baseline
  (now stays on, rank 102→134) - report text must not repeat the old 2-yr
  claim unqualified.
- Figs 11-12 (routing/coverage-gap) confirmed **unchanged by design** -
  population/shelters/roads are static inputs, independent of the GLM year
  window.

Full detail in [2026-09-15](#2026-09-15) below, especially the "Option B
adopted" subsection at the end of that day's entries.

**Full audit completed (2026-09-15):** every number above and in the
2026-09-15 section was re-derived directly from saved files / fresh script
reruns, not trusted from prior summaries - all confirmed exact matches. Found
and fixed: (1) several 2026-09-10 sections (suitability re-rank,
combined-priority tallies, the 42-shortlist, and especially the
coastal-buffer "Walton drops off" claim, which was **contradicted** by the
4-yr buffer result) still stated 2-year numbers as if current; (2) a harmless
2-flash (0.0002%) discrepancy between two independently-computed 4-year
spatial joins, documented where it occurs, not propagated to any reported
number.

**Cleanup completed (2026-09-16):** rather than leaving the stale claims in
place with a superseded-pointer warning next to them, every live section now
states the current 4-year number directly; the original 2-year numbers were
moved, not deleted, into a single **"Superseded history"** section at the
very bottom of this file, so a normal top-to-bottom reading only ever
encounters current, active numbers. The Walton coastal-buffer claim
specifically now reads correctly in place ("stays on the shortlist") with no
false statement left sitting in the main narrative. Separately, figs 16 and
18's four comparison-section AUCs (0.812/0.772/0.794/0.788) no longer live as
hardcoded constants in `generate_report_figures.py` — `model_ceiling_push.py`
and `retrain_4yr_model.py` were refactored into importable functions
(`nested_cv()`, `evaluate_2026()`) alongside `validate_2026.py`'s matching
`evaluate()`, so all three comparison figures (16/18/19) now recompute live
from the saved 2-year and 4-year artifact files on every run and cannot
silently drift.

---

## Prior work (summary through 2026-09-09)

**Data collection — complete.** Four datasets + one validation set, all filtered
to the 5-county study area:

| Dataset | Source | Result |
|---|---|---|
| Lightning flashes | GOES-16/19 GLM (NOAA, AWS Open Data) | 1,958,298 flashes, May–Sep 2024 + May–Sep 2025, panhandle bbox |
| Population | US Census 2020 Decennial, block group | 595 block groups, 972,094 people |
| Shelters | FL Div. of Emergency Management, Risk Shelter Inventory (ArcGIS REST) | 194 facilities (137 general, 46 pet-friendly, 11 special-needs) |
| Road network | OpenStreetMap via osmnx | 61,253 nodes / 152,235 edges, drivable |
| Storm Events (validation) | NOAA Storm Events Database | 1,034 severe-weather rows; only 59 lightning-specific |

Source pivots from the original plan:
- **Lightning:** XWeather historical archive requires a paid tier (free key returns
  `404` / `insufficient_scope`). Pivoted to GOES GLM pulled directly from NOAA's
  public AWS bucket. GOES-16 covers 2024; GOES-19 covers 2025 (GOES-16 retired as
  operational GOES-East, spring 2025).
- **Shelters:** FGDL has no emergency-shelter layer (checked full 452-dataset
  catalog). FDEM's statewide Risk Shelter Inventory covers all 5 counties.

**GLM pull run:** 13.9 h, 16-process pool, 1.32M files fetched, 0 errors, 37.8 MB
of Parquet (10 monthly files). Threading was benchmarked and rejected for the
HDF5 parse step (GIL / HDF5-lock serialized; see `src/calibrate_*.py`).

**June 2025 flash spike verified real.** 412,344 flashes that month vs ~309k next
highest. Not a data artifact and not a tropical cyclone (TS Barry formed June
28–29, tracked to Mexico). Multi-day convective pattern; NOAA Storm Events reports
(flash flooding June 9–10, EF0 tornado June 17, severe wind/hail June 25) land
day-for-day on the GLM flash peaks. Validation days adopted: **June 9, 10, 17,
23, 25, 2025.**

**Feature engineering — started 2026-09-09:**
- `src/features_spatial_join.py` — point-in-polygon flash → block group.
  501,502 of 1.96M flashes fall on the 5-county land area (rest over the Gulf).
- `src/anomaly_detection.py` — dual method: (1) historical daily z-score vs each
  block group's own baseline (z ≥ 3); (2) absolute rolling-30-min burst threshold
  (≥ 5 flashes/window, baseline-independent).
- `src/suitability_model.py` — weighted overlay (population + flash density +
  distance to nearest shelter), percentile-ranked.
- `src/routing_analysis.py` — drive time from each block group to nearest shelter
  on the OSM network (edge speeds from `maxspeed` tags; multi-source Dijkstra from
  all shelter nodes). Coverage: 86.8% of population within 15-min drive; 13.2%
  gap (128,659 people, 83 block groups), concentrated in rural N. Okaloosa and
  Walton, up to 28 min out.

---

## 2026-09-10

### Water block group filtering — now an explicit methodology step

The Census block-group layer has **6 zero-population "water" polygons** (tract
code `99xxxx`) covering the Gulf and coastal bays:
`120059900000, 120339900000, 120919901000, 120919902000, 121139900000, 121319900000`.

These were previously noted only as a caveat. They are now filtered up front in
both `anomaly_detection.py` and `suitability_model.py` (and `routing_analysis.py`
already excluded them), documented in each script's docstring and in the README.
Rationale: a shelter cannot be sited on open water and these polygons protect no
population; leaving them in inflated a handful of peak flash counts with offshore
lightning.

Effect (small, as expected — water BGs were ~5% of flags):
- Anomaly: flashes on land drop 501,502 → **407,486** (−94,016). Flagged
  block-group-days 9,936 → **9,479**. Validation-day Jaccard overlap between the
  two methods **unchanged at 40%** (vs 27% baseline across all days).
- Suitability: **589** land block groups scored (was 595). Top-15 ranking
  unchanged — water BGs never ranked (zero population).
- All 5 June 2025 validation days still trip **both** methods across 38–202 block
  groups each.

### Suitability model — gap feature switched to network drive time

The "coverage gap" factor in `suitability_model.py` was straight-line distance to
the nearest shelter. Replaced with **network drive time** (`drive_min_to_shelter`
from `routing_analysis.py` output), so the factor reflects real travel cost over
the road network. Weights unchanged (0.34 population / 0.33 flash density / 0.33
gap), still percentile-ranked 0–1 across the 589 land block groups.

This methodology carried forward unchanged into the 4-year (Option B primary)
rebuild in `suitability_model_4yr.py` — only the flash-density input changed,
from 2024-25 counts to 2022-25 counts.

**Current (4-year, primary) top-3:** #1 `120910211024` (Okaloosa, unchanged),
#2 `121319505022` (Walton, 4,841 people), #3 `120050027071` (Bay, 2,602
people) — Bay and Walton swap #2/#3 from the original 2-year ranking. Top-50
by-county has not been recomputed for the 4-year baseline. The original
2-year re-rank detail (including the top-50 breakdown) is preserved in
*Superseded history* at the bottom of this file.

### Combined priority table

`src/combined_priority.py` (2-year) / `src/combined_priority_4yr.py` (4-year,
Option B primary). One row per land block group (589), joining:
- suitability rank + score
- anomaly flag counts (`n_days_zscore_flag`, `n_days_burst_flag`,
  `n_days_both_flag`, `n_days_any_flag`) and maxima (`max_zscore`,
  `max_peak_30min`, `total_burst_episodes`)
- drive time + coverage tier (`0-5` / `5-10` / `10-15` / `>15 min (GAP)`)

**Current (4-year, primary) criterion tallies across the 589:**
- top-quartile suitability (rank ≤ 148): **148** (unchanged from the 2-year
  table — same cutoff arithmetic on the same 589 rows)
- in the >15-min coverage gap: **83** (unchanged — geography-only, invariant)
- ≥ 1 day flagged by **both** anomaly methods: **563** (was 494 under the
  2-year window — more history means more block groups trip the filter at
  least once; the intersection with the other two criteria is still what
  actually narrows the field)

### Final shortlist — 37 block groups (4-year, Option B primary)

All three criteria simultaneously: **top-quartile suitability AND >15-min coverage
gap AND ≥ 1 both-method anomaly day.** Rule unchanged since 2026-09-10; only the
underlying suitability/anomaly inputs moved to the 4-year (2022-2025) window.

- **37 block groups, 74,091 people.**
- By county: **Okaloosa 16, Walton 8, Santa Rosa 8, Bay 4, Escambia 1.**
- Highest-ranked entries: Okaloosa `120910211024` (rank 1), Walton
  `121319505022` (rank 2), Bay `120050027071` (rank 3).

(The original 2-year shortlist — 42 block groups, 81,702 people, a different
county breakdown — is preserved in *Superseded history* at the bottom of this
file.)

### ML layer — anomaly classifier (`src/model_anomaly.py`)

**Question:** can a block group's static profile (population, baseline flash
density `flash_per_km2`, drive time to nearest shelter) predict whether it is
ever flagged as anomalous by **both** detection methods
(`has_confirmed_anomaly`)?

**Target reframe:** the three features are static per block group, so a
block-group-*day* target is not learnable from them (every day identical). Target
is block-group level: ≥1 both-method day. 494 True / 95 False — **83.9% positive**.

**Validation:** leave-one-county-out CV (5 folds). Supersedes the earlier
`src/model_random_forest.py`, whose comment claimed a spatial split but actually
did `train_test_split(..., stratify=county)` — a *random* split that leaks
neighbouring block groups across train/test.

**Logistic regression** (standardized, full data):

| term | coef | p-value | odds ratio / SD |
|---|---|---|---|
| flash_per_km2 | 5.79 | <0.0001 | 328 |
| P1_001N (population) | 1.27 | <0.0001 | 3.58 |
| drive_min_to_shelter | 0.22 | **0.143 (n.s.)** | 1.24 |

McFadden pseudo-R² 0.39.

**Pooled out-of-fold metrics:**

| model | accuracy | precision(True) | recall(True) | AUC | False-class recall |
|---|---|---|---|---|---|
| naive baseline (always True) | 0.839 | 0.839 | 1.00 | 0.500 | 0.00 |
| logistic regression | 0.873 | 0.880 | 0.982 | 0.866 | 0.31 |
| random forest | 0.849 | 0.886 | 0.941 | 0.821 | 0.37 |

**RF feature importance:** impurity — flash_per_km2 0.49, population 0.27,
drive-time 0.24. Permutation (AUC drop, held-out) — flash_per_km2 0.214,
population 0.091, **drive-time 0.043**.

**Honest read:**
- On **accuracy**, RF barely clears the naive baseline (+0.010, within noise);
  logistic regression clears it modestly (+0.034). The target is 84% positive so
  accuracy is a weak metric here.
- On **AUC / ranking**, both beat chance clearly (LR 0.87, RF 0.82) — the models
  *rank* block groups by anomaly propensity well, they just don't convert that to
  good hard classification at a 0.5 threshold.
- Both are **poor at the minority (not-anomalous) class**: recall 0.31 / 0.37.
  Unusable for "find the safe block groups"; usable for ranking risk.
- **drive-time-to-shelter has no independent predictive power** once population
  and flash density are in (LR p=0.14; permutation importance ~0). The RF impurity
  score of 0.24 is an artifact — impurity importance is biased toward
  high-variance continuous features. This corrects an earlier read that took the
  impurity score at face value.
- **Population is a real (if modest) predictor** (p<0.0001, OR 3.6/SD) despite the
  target not being built from it — more-populous block groups do coincide with
  confirmed lightning bursts here.
- **Circularity caveat:** the target is derived from flash counts and
  `flash_per_km2` is derived from flash counts, so that feature's dominance is
  near-tautological. The non-circular question is whether population / drive-time
  add anything: population a little, drive-time no.

### ML layer, expanded — tuning, threshold, new features (`src/model_anomaly_expanded.py`)

New features engineered (`src/features_extra.py`):
- `spatial_lag_flash_density` — mean `flash_per_km2` of queen-contiguity
  neighbours. Correlation with own `flash_per_km2` = 0.36 (semi-independent).
- `coast_dist_km` — distance from block-group internal point to nearest TIGER
  coastline segment. Correlation with `flash_per_km2` = −0.09 (fully independent).

Validation upgraded to **nested spatial CV**: outer leave-one-county-out (5
folds), inner `RandomizedSearchCV` (25 iters, `GroupKFold(4)` on training
counties, `roc_auc`) tuning `n_estimators` / `max_depth` / `min_samples_split` /
`min_samples_leaf` / `max_features` for the RF. Held-out county never seen during
tuning, so pooled OOF AUC is honest.

**Framing A — block-group level** (target `has_confirmed_anomaly`, 84% positive,
directly comparable to the earlier 0.87 / 0.84):

| feature set | logreg AUC | RF AUC |
|---|---|---|
| base3 (pop, flash density, drive time) | 0.873 | 0.843 |
| + spatial lag + coast distance | 0.876 (+0.004) | 0.839 (−0.004) |

- **The new features do not move AUC on this framing.** The target is 84%
  positive — almost no discriminative room left, and the tuned RF collapses to
  `max_depth` 3–4 (little structure to fit).
- But `coast_dist_km` **is** a significant LR predictor (p = 0.001, coef +0.78,
  OR 2.17/SD; farther inland → more anomaly-prone, consistent with sea-breeze
  convergence peaking inland). Pseudo-R² 0.388 → 0.414. It improves *fit* without
  moving *ranking* on a saturated target. `spatial_lag` is not significant
  (p = 0.29) — adds nothing beyond own density at block-group level.
- **Threshold tuning** (F1-optimal on pooled OOF): logreg best at 0.11 —
  confusion `[[27,68],[7,487]]`, i.e. catches 487/494 anomalies but only 27/95
  true-negatives. Moves the operating point; does not create signal. RF best
  stayed at 0.50.

**Framing B — block-group × month** (target: ≥1 both-method day that month;
5,890 rows; **29% positive** — the reframe also fixes the degenerate imbalance):

| feature set | logreg AUC | RF AUC |
|---|---|---|
| static only (5 per-BG features) | 0.685 | 0.689 |
| + month one-hot + year | **0.764 (+0.080)** | **0.775 (+0.086)** |

- **This is the real, meaningful gain.** Static features alone can't say *when* a
  block group lights up (same values every month) — ceiling ~0.69. Month
  indicators lift AUC ~0.08.
- Month coefficients (ref = May): June +0.19 (p<0.0001), July/Aug ≈ 0,
  **September −1.28 (p<0.0001, OR 0.28)** — matches the validated seasonal
  pattern (Sept flash totals far below other months). `year_2025` coef exactly
  0.0 — no year effect.
- `coast_dist_km` significant here too (p<0.0001, +0.49). `drive_min_to_shelter`
  becomes significant with the larger sample (p<0.0001) but modest (OR 1.40),
  plausibly a rural/inland proxy.
- Threshold tuning does real work here (balanced target): RF at 0.34 →
  recall 0.75 / precision 0.47, vs 0.38 / 0.58 at 0.50.

**Honest verdict.** As the advisor predicted, tuning and threshold adjustment
give limited, operating-point-only gains. Among the new features: `spatial_lag`
is a dud; `coast_dist_km` is genuine signal but can't move a saturated
block-group-level metric. The one meaningful improvement (+0.08 AUC) comes from
the **block-group-month reframe plus month-of-year indicators** — and that signal
is just the seasonality already validated in the June-2025 spike analysis, now
made explicit to the model. The block-group-level "ever flagged" target is too
saturated to be a useful ML problem; block-group-month is the right unit.

### Model comparison (`src/model_comparison.py`)

Six models on the block-group×month target (5,890 rows, 29% positive), same
nested LOCO spatial CV, features = population, flash density, drive time, coast
distance, month indicators (May reference; spatial lag dropped as a prior dud).

| model | pooled AUC | fold AUC mean ± std | prec@0.5 | rec@0.5 |
|---|---|---|---|---|
| Random forest (tuned) | 0.767 | 0.762 ± 0.020 | 0.567 | 0.347 |
| Logistic regression (plain) | 0.762 | 0.755 ± 0.021 | 0.556 | 0.349 |
| Logistic regression + interactions | 0.759 | 0.754 ± 0.023 | 0.561 | 0.361 |
| HistGradientBoosting (tuned) | 0.759 | 0.761 ± 0.017 | 0.582 | 0.300 |
| Decision tree (tuned) | 0.750 | 0.747 ± 0.016 | 0.489 | 0.509 |
| Naive Bayes | 0.731 | 0.732 ± 0.026 | 0.442 | 0.725 |

**Verdict — no meaningful winner.** Fold-to-fold AUC std is ~0.021 for every
model; the gap from the top (RF 0.767) to plain logreg (0.762) is 0.005, and RF /
logreg / logreg+interactions / HistGB are all within 0.008 — **statistically
indistinguishable**. Per-fold table confirms it (plain logreg beats RF on Walton;
RF beats it on Bay/Escambia — no model dominates all 5 counties).

- **Gradient boosting did not beat Random Forest** (0.759 vs 0.767 pooled; 0.761
  vs 0.762 fold-mean — a tie). The expected tabular-data upgrade did not
  materialise at this size / signal level.
- **Interaction terms did not help** (0.759, a hair below plain logreg's 0.762).
  Some interactions are significant on full data (pop×month_7/8, coast×flash, all
  p<0.01 — the population effect weakens mid-summer) but they add no
  out-of-sample lift.
- **Single decision tree = 0.750**, within noise of the full RF (0.767). The
  ensemble buys ~0.017 AUC over one tree — inside the noise band. The predictable
  structure here is simple and mostly additive, which is why plain logistic
  regression keeps pace with every ensemble.
- **Naive Bayes (0.731) is the only distinguishable model** (gap 0.036 > noise) —
  meaningfully worse, so feature independence is violated enough to cost ~0.03
  AUC, but even the sanity-floor model is close.
- At F1-optimal thresholds (~0.26–0.44) all six land near precision 0.44 /
  recall 0.78 — near-identical operating characteristics regardless of model.
- **Deliberately excluded:** neural nets / deep learning — 5,890 rows is far too
  small for that family to have any advantage.

**Practical pick:** plain logistic regression — matches the ensembles' AUC, runs
instantly, interpretable coefficients. The real ceiling (~0.77 AUC) reflects that
block-group-month anomaly status is only partly predictable from static geography
+ calendar month; that is an honest, expected limit, not a modelling failure.

### Ceiling-push: new information, not more tuning (`src/model_ceiling_push.py`)

New features added to the block-group×month problem:
`prev_month_flagged`, `same_month_last_yr_flagged`, `is_may`, `is_2025`, `oni`
(NOAA Oceanic Niño Index, centred season), NLCD 2021 land-cover shares
(`pct_developed` / `pct_forest` / `pct_wetland`, via `src/features_nlcd.py`,
72 MB WCS clip from MRLC), and cyclical day-of-year (`doy_sin` / `doy_cos`).

Nested LOCO spatial CV, LogReg + tuned RF:

| feature set | model | pooled AUC | fold mean ± std | Δ vs baseline |
|---|---|---|---|---|
| baseline (static4 + month dummies) | LogReg | 0.762 | 0.755 ± 0.021 | — |
| baseline | RF | 0.767 | 0.762 ± 0.020 | — |
| expanded (NLCD + lags + ONI + cyclical DOY) | LogReg | 0.790 | 0.777 ± 0.032 | +0.028 (within noise) |
| expanded | RF | **0.812** | 0.800 ± 0.025 | **+0.045 (clears noise)** |
| expanded + month dummies | LogReg | 0.800 | 0.786 ± 0.027 | +0.038 |
| expanded + month dummies | RF | **0.812** | 0.797 ± 0.028 | **+0.045** |

**This is the first expansion in the ML arc that clears the fold-to-fold noise
band.** RF ~0.77 → ~0.81 AUC. Modest but real.

**Attribution (expanded LogReg coefficients, standardized):**
- `oni` +0.93 (p<0.0001, OR 2.53) — largest coefficient, BUT with only 2 seasons
  ONI is nearly collinear with year×month; its apparent effect **cannot be
  separated from generic year-to-year difference** and is not defensible as an
  ENSO *mechanism* from this sample. `is_2025` +0.50 (p<0.0001) sits right
  alongside it.
- `doy_cos` −0.87, `doy_sin` −0.57 (both p<0.0001) — cyclical calendar, strong.
  With only 5 months, cyclical DOY and month dummies are interchangeable (RF
  identical at 0.812 either way).
- `pct_developed` −0.68 (p<0.0001, OR 0.51) — **the one genuinely-new spatial
  signal that contributed.** More developed land → *less* anomaly-prone (opposite
  the urban-convection hypothesis; likely a coastal-proximity confound).
  `pct_forest` +0.17 (p=0.004), consistent inland/rural direction.
- `prev_month_flagged` +0.045, **p=0.22 — not significant.** The advisor's top
  pick (month-to-month persistence) did not help once calendar position and
  geography are known. `same_month_last_yr_flagged` +0.10 (p=0.007) — weak.
- `coast_dist_km` and `drive_min_to_shelter` fall to non-significant here — the
  NLCD + calendar features absorbed their signal.

pseudo-R² 0.183 → 0.218.

**Honest verdict.** New information pushed past the ~0.77 ceiling to ~0.81 AUC,
but the gain is mostly a fuller description of *when* (calendar + a year effect
dressed up as ENSO), plus a real but small land-cover contribution. The lag
features — the most-anticipated lever — did essentially nothing. ~0.81 is still
far from a strong classifier, which is consistent with the stated possibility
that *exactly when and where* a localized convective burst strikes a specific
small block group in a specific month is partly irreducibly unpredictable.

### 2026 out-of-time validation (`src/pull_glm_2026.py`, `src/validate_2026.py`)

Pulled GLM May-Aug 2026 from GOES-19 (976,566 flashes, 522,605 files, 0 errors,
4.43h; 2026-07-15 was ~85% complete on the archive, absorbed by month-level
aggregation). Trained LogReg + RF on **all** of 2024-25 expanded features,
tested on 2026 - genuinely held out, no CV, no resampling.

2026 positive rate 0.354 (vs 0.289 in training - 2026 was a more anomaly-heavy
period). 2026 sits in a **strong El Nino** (ONI +0.95 to +1.80), entirely outside
the 2024-25 training range (-0.43 to +0.43) - the extrapolation risk flagged
during the ceiling-push analysis.

| model | AUC | prec@0.5 | rec@0.5 |
|---|---|---|---|
| LogReg | 0.812 | 0.354 | 1.000 |
| RandomForest | **0.772** | 0.607 | 0.548 |
| *(CV reference, RF)* | *0.812* | — | — |

**Two different failure modes, both honest findings:**

- **LogReg's AUC looks unchanged (0.812 = 0.812), but that number is misleading.**
  precision = the 2026 base rate exactly and recall = 1.000 means it predicted
  "anomalous" for **100% of the 2356 test rows**. The huge `oni` coefficient
  (+0.93/SD) extrapolated against 2026's far-outside-range ONI pushed every
  probability above 0.5, saturating the classifier. AUC survived only because
  the *relative ordering* of those saturated probabilities still tracked the
  outcome somewhat - the model is ranking correctly but is **operationally
  useless** at any standard threshold in 2026. This is exactly the fragility the
  ceiling-push write-up warned was untested.
- **Random Forest - the model actually asked about - AUC dropped 0.812 → 0.772
  (-0.040).** That is larger than the ~0.02-0.03 fold-to-fold noise seen
  throughout the spatial CV, so this reads as a real generalization gap, not
  noise. It did **not** saturate like LogReg (precision 0.607 / recall 0.548,
  reasonably balanced; mean predicted probability 0.373 vs actual 0.354 - decent
  calibration) - tree splits handled the out-of-range ONI more gracefully than
  the linear model's extrapolated coefficient, but discrimination still degraded.

**Bottom line - the honest answer to "does it generalize forward, not just
backward":** no, not as well. Spatial (county) held-out performance was stable
at ~0.81 ± 0.02-0.03. Temporal (year) held-out performance is measurably lower
for RF (0.772) and only nominally equal for LogReg once you look past the
saturated predictions. The model was validated for interpolating across space
within the climate conditions it was trained on; it was not shown to extrapolate
across a genuinely different climate year. That is a real limitation to state
plainly in the report, not a bug to fix quietly.

### 4-year training window (2022-2025) — does ONI become interpretable? (`src/pull_glm_2022_2023.py`, `src/build_4yr_features.py`, `src/retrain_4yr_model.py`)

Extended training from 2 storm seasons to 4, specifically to give ONI genuine
variation to be estimated against, instead of two ENSO-neutral seasons where it
was nearly a proxy for "which year."

**Satellite confirmed empirically** (same check that caught the 2024→2025
GOES-16→19 handoff): GOES-16 was operational GOES-East for all of 2022-2023
(GOES-19 didn't exist until 2024, operational only from April 2025). Both years
pulled from `noaa-goes16`. Pull: 1,315,548 files, 2,345,970 flashes, 11 errors
(0.0008%), 13.56h.

**Real ONI (NOAA CPC) confirmed the premise:** 2022 solid La Niña (−0.70 to
−0.87 across May-Sep), 2023 swinging hard toward El Niño by late season (+0.46
→ +1.50), 2024-25 near-neutral (as established). 2023's upper values (+1.25 to
+1.50) meaningfully overlap the low end of 2026's range (+0.95 to +1.80),
shrinking the extrapolation gap that hurt the 2-year model.

**Caught and fixed a CRS mismatch bug** during the spatial join (block groups
in NAD83/EPSG:4269 vs flashes in WGS84/EPSG:4326, un-reprojected) - reran after
fixing and got byte-identical results (936,720 land flashes both times, same
per-year positive rates), confirming the ~1-2 m NAD83/WGS84 offset in this
region is negligible at block-group scale. Verified, not assumed.

**Audit note (2026-09-15):** `flashes_with_blockgroup_4yr.parquet` (this
script's unprojected WGS84 `within` join) has **936,720** rows, confirmed by
direct count. `coastal_buffer_analysis_4yr.py`'s own internal join (same
flashes, reprojected to EPSG:5070 before `within`) independently counts
**936,718** on land - a 2-flash (0.0002%) discrepancy, almost certainly a
flash sitting essentially on a block-group boundary that a projection
rounds differently. Harmless in practice: the buffer script's saved outputs
(`flash_count_comparison_4yr.csv` etc.) load counts from the canonical
936,720-row file, not its own 936,718 figure, so no reported number is
affected - flagged here only for completeness.

**Design choice:** dropped the `is_2025` year dummy for this run. With only 2
years it was nearly redundant with ONI (which is exactly why the earlier ONI
coefficient couldn't be trusted); with 4 years and no competing year indicator,
ONI has to carry any real signal on its own.

**Training table:** 11,780 rows (589 BGs × 4 years × 5 months), positive rate
0.298 (2022 0.284, 2023 0.381 - the highest, 2024 0.262, 2025 0.264).

**Q1 - is ONI now interpretable? Yes, substantially:**

| | 2-year model | 4-year model |
|---|---|---|
| ONI coefficient (std) | +0.93 | **+0.26** |
| p-value | <0.0001 | **9.7e-26** |
| Odds ratio / SD | 2.53 | **1.29** |
| Competing year dummy | `is_2025` (confounded) | none |

The coefficient shrank to a much more modest, defensible size, stayed highly
significant with more data and no year dummy to hide behind, and its sign
(positive - more El Niño-like → more anomaly-prone) is consistent with the
2-year run. Honest caveat: raw year-level positive rates aren't perfectly
monotonic in ONI (2023 > 2022 > 2025 > 2024 in rate, vs 2023 > 2024 > 2025 > 2022
in ONI) - the regression coefficient is a multivariate partial effect, not
proof of a clean standalone ENSO mechanism; other unmodeled year-specific
factors likely remain intertwined with it. But it is no longer the near-tautological
year-proxy it was.

**Q2 - does 2026 out-of-time performance improve from 0.772? Yes, on both
counts, with LogReg's improvement being the more important one:**

| Model | 2yr-trained on 2026 | 4yr-trained on 2026 |
|---|---|---|
| RandomForest AUC | 0.772 | **0.788 (+0.016)** |
| LogReg AUC | 0.812* | 0.781 |
| LogReg % predictions ≥0.5 | **100%** (saturated) | **40%** (2026 actual rate: 35%) |

\*The 2-year LogReg's 0.812 was a mirage - it predicted "anomalous" for every
single 2026 row (recall 1.0), saturated by extrapolating the oversized ONI
coefficient. The 4-year LogReg's raw AUC is nominally lower (0.781) but it is
now an **actually usable, reasonably calibrated classifier** (prec 0.592, rec
0.671, 40% flagged vs a 35% true rate) instead of a broken one with a
misleadingly high number attached. This is the more important result of the
whole experiment.

**Report-writing note:** describe these two numbers asymmetrically, not as
two results to compare at face value. The 2-year **0.812 is a broken,
saturated model** — cite it only as the reason the 2-year approach was
abandoned, never as a working AUC. The 4-year **0.781 is the honest, usable
result** and is what the report should present as the model's out-of-time
performance.

**Future refinement, not done now (timeline):** an optional test of whether
ONI stays significant against a competing `C(year)` term (rather than just in
the absence of one, as run here) would sharpen the causal claim further. Out
of scope for the current report; flagged for later if there's time.

**Trade-off worth stating plainly:** the 4-year nested-CV AUC is *lower* than
the 2-year CV AUC (RF 0.794 vs 0.812; LogReg 0.766 vs 0.790). Training on a
harder, more climate-diverse distribution makes the cross-validation problem
itself harder - the model can no longer lean on the narrow, easier 2024-25
regime. That is the expected and correct trade: worse on the (now harder)
spatial CV, better calibrated and modestly better on genuine forward-in-time
generalization. 2026's ONI still exceeds the training max in July-August
(1.80 vs training max 1.50), so some extrapolation risk remains, just far less
severe than before (training max was 0.43).

### Coastal 10 km buffer variant (`src/coastal_buffer_analysis_4yr.py`, 4-year Option B primary; originally `src/coastal_buffer_analysis.py`)

Separate, clearly-labelled dataset variant — **originals untouched**, outputs
under `data/processed/coastal_buffer_10km_4yr/` (4-year, current) and
`data/processed/coastal_buffer_10km/` (original 2-year, preserved for
reference). Land block-group shapes buffered outward 10 km; offshore flashes
within that distance credited to the nearest coastal block group (via
`sjoin_nearest`, `max_distance=10 km`); flashes farther out stay unmatched, as
in the land-only join.

**Methodology note:** 10 km is a reasonable estimate based on the typical ~10–20
km horizontal extent of lightning-producing storm cells (established convective
meteorology), not a tuned value. A data-driven buffer (e.g. informed by actual
storm-cell tracking) is identified as future work. Both land-only and buffered
versions are kept so the comparison itself is a reportable finding.

**Flash re-assignment (2022–25, current):**
- On land: 936,720 (21.8% of 4,304,268 pulled)
- Newly credited within 10 km of coast: **461,532 (+49% over the land total)**
- Still unmatched (>10 km offshore): 2,906,018

**Coastal block groups (≤5 km from coast, n=128):** flashes 61,645 → 195,991
(**+218%**). The buffer is a large change for coastal areas, negligible for
interior ones. (Direction and rough magnitude replicate the original 2-year
finding of +246% — see *Superseded history* for that exact figure.)

**Shortlist: 37 → 47** (34 stayed, **13 added, 3 dropped**). The buffer
**materially reorders priorities** — it is not cosmetic.

**Featured findings under the buffer (current, 4-year):**
- **Walton `121319503053` — rank 102 → 134, STAYS ON the shortlist.** Its
  flash count is unchanged (53,947; it is inland, gets no buffer benefit),
  and coastal block groups still leapfrog it on flash density — a real -32
  place demotion — but the wider 4-year top-quartile cutoff (≤148) still
  contains it. The demotion direction is the same finding as the original
  2-year analysis; the pass/fail outcome is not (2-year: dropped off, rank
  127→158; see *Superseded history*). Report this contrast explicitly rather
  than either claim alone — it is itself evidence that the buffer's effect on
  the shortlist boundary is sensitive to the choice of hazard baseline.
- **N. Okaloosa cluster** — under the buffer, **4 of 6** cluster block groups
  are shortlisted: `...11` (rank 4→3), `...21` (120→29), `...23` (24→27) were
  already on the land-only 4-year list; `...13` (189→38) newly flips **on**.
  `...12` (172→174) and `...22` (277→281) stay off — both still register
  weak both-method activity (max 30-min peaks of 7 and 5, right at the
  5-flash burst threshold) but not enough to clear the suitability cutoff.

**Take:** the choice of buffer is consequential. Land-only is conservative
(counts only strikes that hit the neighbourhood); 10 km-buffered credits
plausibly-same-cell offshore strikes but on an estimated radius. Neither is
definitively correct — the report should present both and treat the
buffer-distance choice as an open methodological question.

### Buffered flash density in the model (`src/model_buffered_flash_density.py`)

Ablation: re-run the best model (expanded RF, nested LOCO CV) with only
`flash_per_km2` swapped from land-only to 10 km-buffered; target and all 13
other features unchanged.

| flash_per_km2 source | pooled AUC | fold mean ± std | prec@.5 | rec@.5 |
|---|---|---|---|---|
| existing CV reference | 0.812 | — | — | — |
| unbuffered (land-only) | 0.809 | 0.798 ± 0.024 | 0.595 | 0.456 |
| buffered (10 km coastal) | 0.803 | 0.791 ± 0.027 | 0.587 | 0.405 |

Buffered − unbuffered = **−0.006**, well inside the fold noise (~0.027) →
**no meaningful change** (LogReg identical too: 0.790 → 0.789). The unbuffered
re-run reproduces the 0.812 within RandomizedSearchCV stochasticity; land
`flash_per_km2` recompute matched `features_extra` exactly.

**Finding:** the coastal buffer more than doubles coastal block-group flash
counts and materially reshuffles the descriptive rankings / shortlist, but it is
**not a lever for model performance** — the RF extracts the same anomaly signal
either way, and the ~0.80–0.81 ceiling holds regardless of flash-density
definition. **Not rebuilt on the 4-year baseline** (out of scope per the
2026-09-15 Option B decision) — treat this ablation's specific numbers as
2-year-window only; the qualitative "buffer doesn't move model AUC" conclusion
has not been re-verified under 4-year data. (Caveat: the target here is the land-only anomaly labels; a fully
buffered variant that also recomputes the target was not run — "keep everything
else the same".)

### Interactive dashboard (`src/build_dashboard_4yr.py`, 4-year Option B primary; originally `src/build_dashboard.py`)

`dashboard/panhandle_priority_dashboard.html` — single-file Folium map, built
on the 4-year (2022-2025) suitability/anomaly baseline, 3 toggleable layers:
1. Suitability choropleth, all 589 block groups, blue→yellow→red by
   `suitability_score`, click popup with population / flash density / drive time /
   coverage tier / suitability rank / anomaly day counts (z-score, burst, both) /
   max z / max 30-min peak / shortlist flag.
2. Final shortlist — the **37** block groups, drawn as a bold dark-blue outline on top.
3. Featured findings — Walton `121319503053` (red fill + bolt marker) and the six
   N. Okaloosa cluster block groups (orange fill + marker if shortlisted, grey if
   not), each with a summary popup.

Tiles: OpenStreetMap (CartoDB now needs an API key). Needs internet for tiles +
Leaflet CDN; the HTML itself is one self-contained file (~4.6 MB). The
original 2-year dashboard (42-block-group shortlist) is preserved at
`dashboard/archive/panhandle_priority_dashboard_2yr_baseline.html` and described in
*Superseded history* at the bottom of this file.

### Featured findings (4-year, Option B primary; see `notes/featured-findings-writeup.md`)

- **Walton `121319503053`** — on the shortlist; the strongest single case in the
  study area. Rank 102/589 (capped only by population 1,135), but 28.2-min drive
  (most isolated block group in the study area), 53,947 flashes (4 seasons,
  2022-25), 28 both-method anomaly days, peak 30-min burst of 1207.
- **N. Okaloosa cluster `120910210011`–`023`** (6 contiguous block groups, all in
  the >15-min gap) splits 3-and-3. On shortlist: `...11` (rank 4), `...23`
  (rank 24), `...21` (rank 120). Off: `...12`, `...13`, `...22` — the burst
  method fires only weakly there (max 30-min peaks of 5–12, near the 5-flash
  minimum threshold, vs. 9–50 for the on-list three); they carry z-score
  anomalies (relative) but limited confirmed absolute-intensity bursts, and
  their suitability ranks fall outside the top quartile. Clean illustration
  that relative anomaly ≠ absolute hazard.

(The original 2-year findings — Walton rank 127; Okaloosa on-list
`...11`/`...23`/`...12` — are preserved in *Superseded history* at the bottom
of this file. Note the on-list Okaloosa **membership** changed, not just the
ranks: `...12` was on the 2-year list and is off the 4-year list; `...21` is
the reverse.)

---

## 2026-09-15

### Report figures 16, 18, 19 regenerated on the 4-year retrain (`src/generate_report_figures.py`)

Fig 16 (AUC progression) gets a 4th point: the 2-yr feature-expansion endpoint
(0.812) now continues, in a distinct color/dashed line, to the 4-yr retrain's
nested-CV pooled AUC (**0.794**), annotated as a deliberate step down (harder
CV, better temporal generalization - see Fig 18).

Fig 18 (CV vs 2026 out-of-time) is now grouped bars, 2-yr model vs 4-yr
retrain, both AUCs sourced from `src/retrain_4yr_model.py`'s printed output
(not re-derived a new way):

| | Cross-validation (county held-out) | 2026 out-of-time (year held-out) | Gap |
|---|---|---|---|
| 2-yr model (2024-25) | 0.812 | 0.772 | −0.040 |
| 4-yr retrain (2022-25) | **0.794** | **0.788** | **−0.006 (essentially closed)** |

Fig 19 (LogReg 2026 probability histogram) is now a two-panel before/after:
left panel is the 2-yr model, explicitly labeled **"BROKEN - not a real
result"** (100% of predictions saturate above 0.5); right panel is the 4-yr
retrain, labeled **"FIXED - the honest, usable result"** (40% above 0.5 vs a
35% true rate, prec 0.592 / rec 0.671). Matches the framing established in the
Key numbers snapshot at the top of this file - report the 4-yr number as the
model, cite the 2-yr number only to explain why it was discarded.

### Dashboard, suitability, and shortlist rebuilt on the 4-year baseline
(`src/suitability_model_4yr.py`, `src/combined_priority_4yr.py`, `src/build_dashboard_4yr.py`, `src/anomaly_detection_4yr.py`)

Same methods throughout (percentile-overlay weights 0.34/0.33/0.33; shortlist
rule = top-quartile suitability AND >15-min drive gap AND ≥1 confirmed
both-method anomaly day) - only the hazard input (flash density) and the
anomaly-day counts now come from the full 2022-2025 record instead of
2024-25 only. Drive time and population are geography-based and unchanged.
Outputs under `data/processed/suitability_4yr/`, `data/processed/anomaly_4yr/`,
`data/processed/final_4yr/`. Dashboard rebuilt at
`dashboard/panhandle_priority_dashboard.html` (previous 2-yr version archived
as `dashboard/archive/panhandle_priority_dashboard_2yr_baseline.html`).

**Shortlist: 42 → 37.** 34 block groups unchanged, **8 dropped**
(`120050002011`, `120050027061`, `120050027062`, `120050027092`,
`120910209001`, `120910210012`, `120910211011`, `121319505013`), **3 added**
(`120910210021`, `121130108094`, `121130108143`). The 4-year hazard baseline
is not cosmetic here either - it reorders enough of the top quartile to move
11 block groups across the line.

**Walton `121319503053` — improves, does not drop.** Rank 127 → **102**
(moved up, not down) under the 4-year hazard baseline; still on the
shortlist. Flash count over the full 4 years is 53,947 (vs 22,066 over the
original 2 seasons - expected, longer window); both-method anomaly days 14 →
28; drive time and population unchanged (geography). The "strongest single
case" finding holds and, if anything, strengthens.

**N. Okaloosa cluster `120910210011`–`023` — still 3-and-3, but a different
three.** Under the 2-yr baseline: on-list was `...11` (16), `...23` (44),
`...12` (142); off was `...13`, `...21`, `...22`. Under the 4-yr baseline:
on-list is now `...11` (rank **4**), `...23` (rank **24**), and **`...21`
(rank 120, newly added - was off-list at rank 201)**; `...12` **drops off**
(rank 172, was on-list at rank 142); `...13` and `...22` stay off in both
versions. Net: the *size* of the split is stable (3-and-3) and the headline
claim ("splits down the middle, relative anomaly ≠ absolute hazard") still
holds, but which specific block groups sit on which side is baseline-
sensitive and should be reported as such, not treated as fixed.

### June 2025 validation re-checked against the 4-year rolling baseline (`src/anomaly_detection_4yr.py`)

The original June 2025 validation (5 days matched to NOAA Storm Events: flash
flooding Jun 9-10, EF0 tornado Jun 17, severe wind/hail Jun 23 & 25) was built
against a z-score baseline computed only over 2024-25. Re-ran both detection
methods over the full 2022-2025 daily record (more history → different
per-block-group mean/std) and re-checked all 5 days:

| Day | Both-method flags, 2-yr baseline | Both-method flags, 4-yr baseline |
|---|---|---|
| Jun 9 | 43 | 41 |
| Jun 10 | 24 | 18 |
| Jun 17 | 13 | 9 |
| Jun 23 | 62 | 55 |
| Jun 25 | 143 | 130 |

**All 5 days still trip ≥1 both-method block group under the 4-year
baseline** - the validation finding survives the wider window. Counts shift
modestly downward (more history → higher estimated variance → the z≥3
threshold is mildly harder to clear), which is the expected, correct
consequence of a more stable baseline, not a failure of the finding.

The specific Fig 4 illustration (3 Santa Rosa block groups matching the Jun
9-10 flash-flood report) is **unchanged at the individual-flag level**:
`121130101001` and `121130102001` both-flag on Jun 9 (not Jun 10),
`121130104001` both-flags on Jun 10 (not Jun 9) - identical pattern, only the
z-scores themselves shift by a few tenths. No figure update needed here.

### Option B adopted: 4-year data is now the single primary result, report-wide (`src/generate_report_figures.py` + new `*_4yr` scripts)

Decision, agreed with advisor input: every figure and statistic in the report
uses the **4-year (2022-2025) window as the one, primary truth**, except a
single dedicated comparison section (still figs 16/18/19 - unchanged from the
earlier update) that deliberately keeps the 2-yr-vs-4-yr contrast, framed
explicitly as a test of training-data diversity vs. real-world
generalization, not as leftover superseded numbers. All 19 figures
regenerated together; only figs 3/4/11/12 are functionally unchanged (see
below for why).

**Newly rebuilt this pass (not touched in the previous update):**

- **Figs 1-2 (data overview):** Fig 1 is now a 4-way grouped bar (2022/23/24/25
  × May-Sep), not just 2024-vs-2025. Fig 2's flash KPI is now **4,304,268**
  (May-Sep 2022-2025 raw pulled total - cross-checked: matches the coastal
  buffer script's independently-computed "Total 2022-2025 flashes" exactly).
  Population/shelters/road-network cards are explicitly labeled "(unchanged -
  static, year-independent input)".
- **Figs 7-8 (suitability map, top-15):** now read `final_4yr/` (already
  rebuilt in the prior pass, just newly wired into these two figures - they
  previously still pointed at the 2-yr `final/` output).
- **Fig 15 (6-model comparison):** new script `src/model_comparison_4yr.py`,
  identical method/feature-scope/CV protocol as the original
  `model_comparison.py`, rerun on the 4-year table (11,780 rows vs 5,890).
  **Notable result: the ranking changes.** 2-yr window had RandomForest on
  top (0.767); 4-yr window has **LogReg on top (0.752)**, with RF essentially
  tied for 3rd (0.749) - all top 4 models (LogReg, LogReg+interactions, RF,
  HistGB) are within fold-to-fold noise of each other. Worth a line in the
  report: the "best" model choice at this feature scope is not robust to the
  training-window choice either.
- **Fig 17 (feature importance):** switched from impurity-based
  `feature_importances_` to **permutation importance**
  (`sklearn.inspection.permutation_importance`, scoring=roc_auc, 15 repeats),
  computed on the full 4-year expanded-feature refit (same model as
  `retrain_4yr_model.py`'s final RF). Top features by permutation importance:
  `pct_developed`, `doy_cos`, `doy_sin`, `oni` (all clearly above the rest);
  the lag features (`prev_month_flagged`, `same_month_last_yr_flagged`,
  `is_may`) are least important - consistent with, but not identical to, the
  old impurity-based ranking.

**Coastal buffer analysis - rebuilt (new script `src/coastal_buffer_analysis_4yr.py`)**

Confirmed this **did** need a rebuild: the buffer's entire mechanism (crediting
offshore flashes to the nearest coastal block group) operates on the flash
join, so it inherits whichever flash-density baseline is primary. Rerun on
the full 2022-2025 flash record (4,304,268 flashes - matches Fig 2 exactly),
compared against the 4-year land-only outputs. Separate output directory
`data/processed/coastal_buffer_10km_4yr/`; the original 2-yr buffered outputs
are untouched.

- **Shortlist: 37 (land-only, 4yr) → 47 (buffered, 4yr).** 34 stayed, 13
  added, 3 dropped. Coastal block groups' flashes: 61,645 → 195,991 (+218%,
  consistent in direction/magnitude with the original 2-yr buffer finding of
  +246%).
- **Walton `121319503053` - narrative changes materially.** Under the
  original 2-yr baseline, the buffer *dropped Walton off* the shortlist
  (rank 127→158, crossing the top-quartile cutoff). Under the **4-year
  baseline, Walton stays on the shortlist** even after buffering (rank
  102→134, well inside the wider baseline's top-quartile cutoff of ≤148).
  The buffer still demotes it substantially (-32 places) - the *direction* of
  the effect (coastal areas leapfrog it) replicates - but the qualitative
  "drops off the list" claim from the 2-yr report **does not hold** under the
  4-year baseline and must not be repeated as-is; Fig 10 has been rewritten
  to state both facts side by side.
- **N. Okaloosa cluster under the buffer:** land-only 4yr split was 3-on
  (`...11`, `...23`, `...21`) / 3-off (`...12`, `...13`, `...22`). Buffered
  4yr split becomes **4-on** (`...11`, `...13`, `...21`, `...23`) / 2-off
  (`...12`, `...22`) - `...13` flips on under the buffer (rank 189→38). Not
  currently reflected in Fig 14 (which is land-only by design, matching its
  own title); flag for the write-up if the buffered cluster picture is
  discussed in text.

**Routing / coverage-gap figures (11-12): confirmed no rebuild needed.**
Population (2020 Census, static), shelter locations (FDEM inventory, static),
and the OSM road network (static) are the only inputs to drive-time routing -
none depend on the GLM flash record or its year window. Figs 11 and 12 are
therefore already correct under Option B and were left as-is (code comment
added at that section header making this explicit, so it doesn't read as an
oversight).

**Not rebuilt / explicitly out of scope:** the buffered-flash-density model
ablation (`src/model_buffered_flash_density.py`, showed the coastal buffer
doesn't move model AUC) was a 2-yr-window analysis; re-running it on 4-yr
data was not requested and is noted here only so it isn't mistaken for
already-current.

---

## Superseded history — 2026-09-10 snapshot, 2-year (2024-25) window

Preserved verbatim as a historical record of what the original 2-year
analysis found, before the 2026-09-15 Option B move to the 4-year
(2022-2025) window as the report's single primary result. **None of the
numbers in this section should be cited as current** - each has a live,
current replacement in the main narrative above. Kept here rather than
deleted because the *contrast itself* (what changed, and by how much, when
the training window widened) is part of the report's methodology story.

### Suitability re-rank (2-year, `src/suitability_model.py`)

- New #1: `120910211024` (Okaloosa, 3,291 people, 21.3-min drive) — was #2.
- Former #1 `120050027071` (Bay) → #2.
- New #3: `121319505022` (Walton, 4,841 people, 1,537 flashes).
- Top-50 by county: Okaloosa 14, Santa Rosa 14, Bay 8, Walton 8, Escambia 6
  (was Santa Rosa 15, Okaloosa 12, Bay 9, Walton 9, Escambia 5 — Okaloosa
  gains from its longer rural drive times).

*Current (4-year): see "Suitability model — gap feature switched to network
drive time" above.*

### Combined priority table tallies (2-year, `src/combined_priority.py`)

Criterion tallies across the 589:
- top-quartile suitability (rank ≤ 148): **148**
- in the >15-min coverage gap: **83**
- ≥ 1 day flagged by **both** anomaly methods: **494**

*Current (4-year): see "Combined priority table" above (148 and 83 are
unchanged; the both-method count is now 563).*

### Final shortlist (2-year): 42 block groups

All three criteria simultaneously: top-quartile suitability AND >15-min
coverage gap AND ≥ 1 both-method anomaly day.

- **42 block groups, 81,702 people.**
- By county: Okaloosa 18, Walton 9, Bay 8, Santa Rosa 6, Escambia 1.
- Highest-ranked entries: Okaloosa `120910211024` (rank 1), Bay
  `120050027071` (rank 2), Walton `121319505022` (rank 3).

*Current (4-year): see "Final shortlist — 37 block groups" above.*

### Coastal 10 km buffer variant (2-year, `src/coastal_buffer_analysis.py`)

**Flash re-assignment (2024–25):**
- On land (original): 407,486 (20.8% of 1.96M)
- Newly credited within 10 km of coast: 204,061 (+50% over the land total)
- Still unmatched (>10 km offshore): 1,346,751

**Coastal block groups (≤5 km from coast, n=128):** flashes 26,957 → 93,228
(+246%).

**Shortlist: 42 → 51** (39 stayed, 12 added, 3 dropped). Added members were
almost all coastal/near-bay Bay and Okaloosa block groups that leapfrog on
flash density (e.g. Bay `120050027032` rank 296 → 77).

**Featured findings under the 2-year buffer:**
- **Walton `121319503053` — rank 127 → 158, DROPS OFF the shortlist.** Flash
  count unchanged (22,066; inland, no buffer benefit), but enough coastal
  block groups leapfrog it on flash density that it crosses the top-quartile
  cutoff and falls off. **This specific outcome (dropping off) does not
  replicate under the 4-year baseline** — see the live section above, where
  Walton stays on the list under the buffer (102→134). The underlying
  mechanism (coastal block groups leapfrogging it) does replicate; only the
  pass/fail result at the shortlist boundary differs.
- **N. Okaloosa cluster** — reshuffled toward inclusion (near Choctawhatchee
  Bay). `...210011` rank 16 → 3 (3rd overall); `...210013` (206 → 58) and
  `...210021` (201 → 79) newly added. Under the 2-year buffer, 5 of 6 cluster
  block groups were shortlisted (was 3 of 6 land-only); only `...210022`
  stayed off.

*Current (4-year): see "Coastal 10 km buffer variant" above.*

### Interactive dashboard (2-year, `src/build_dashboard.py`)

`dashboard/panhandle_priority_dashboard.html` (as it existed before the
2026-09-15 rebuild; now archived as
`dashboard/archive/panhandle_priority_dashboard_2yr_baseline.html`) — single-file
Folium map, 3 toggleable layers: suitability choropleth (589 block groups);
final shortlist (the 42 block groups of that time); featured findings
(Walton + the six N. Okaloosa cluster block groups). Same field list and
layer structure as the current 4-year dashboard, described above.

### Featured findings (2-year)

- **Walton `121319503053`** — on the shortlist; the strongest single case in
  the study area. Rank 127/589 (capped only by population 1,135), 28.2-min
  drive (most isolated block group in the study area), 22,066 flashes, 14
  both-method anomaly days, peak 30-min burst of 994.
- **N. Okaloosa cluster `120910210011`–`023`** splits 3-and-3. On shortlist:
  `...11` (rank 16), `...23` (rank 44), `...12` (rank 142). Off: `...13`,
  `...21`, `...22` — the burst method never fired there (peak 30-min windows
  of 3–4 flashes).

*Current (4-year): see "Featured findings" above. Note the on-list Okaloosa
membership itself changed (not just ranks): `...12` was on this 2-year list
and is off the 4-year list; `...21` is the reverse.*
