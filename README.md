# PanhandleStrike

**Where should the Florida Panhandle put its next lightning shelters?**

PanhandleStrike is a GIS and machine-learning site-suitability study for
lightning-shelter siting across five Florida Panhandle counties: Escambia,
Santa Rosa, Okaloosa, Walton, and Bay. It combines four storm seasons of
satellite lightning data, 2020 Census population, the state's emergency-shelter
inventory, and road-network drive times. The output is a shortlist of Census
block groups that need a shelter most.

**[Open the interactive dashboard](https://shaheen2300.github.io/panhandlestrike/dashboard/panhandle_priority_dashboard.html)**
(Folium map; about 4.6 MB, so it takes a moment to load)

Team: Shaheen Memon, Koushal Doddipatla (MS Data Science capstone, University of West Florida)

---

## Headline results (4-year primary window, May–Sep 2022–2025)

| | |
|---|---|
| Lightning flashes analyzed | **4,304,268** GOES-16/19 GLM flashes (936,720 over land in the study area) |
| Population covered | **972,094** residents, 589 inhabited block groups |
| Existing shelters | **194** (137 general, 46 pet-friendly, 11 special-needs) |
| Shelter coverage | **86.8%** of residents are within a 15-minute drive of a shelter. The remaining **128,659 people** (83 block groups) are mostly in rural northern Okaloosa and Walton, up to 28 min away |
| Final shortlist | **37 block groups, 74,091 people**: Okaloosa 16, Walton 8, Santa Rosa 8, Bay 4, Escambia 1 |
| Anomaly classifier (2026 out-of-time test) | Logistic regression AUC **0.781**, random forest AUC **0.788**, trained on 2022–25 and tested on unseen 2026 data |

A block group makes the **shortlist** only if it meets all three criteria:

1. **Top-quartile suitability**: a weighted overlay of population, flash
   density, and network drive time to the nearest shelter (0.34 / 0.33 / 0.33),
   percentile-ranked.
2. **Coverage gap**: more than 15 minutes' drive from any existing shelter.
3. **Confirmed lightning anomaly**: at least one day flagged by **both**
   anomaly methods (see below).

### Featured findings

- **Walton `121319503053`, the strongest single case.** It has the longest
  drive to shelter in the study area (28.2 min) and 53,947 flashes over four
  seasons. It had 28 days flagged by both methods and a peak burst of 1,207
  flashes in 30 minutes. Its suitability rank (102 of 589) is held down only by
  its small population (1,135).
- **Northern Okaloosa cluster: relative anomaly is not absolute hazard.** Six
  contiguous block groups (~11,900 residents) are all 21–25 min from shelter.
  All six show z-score anomalies, because their baseline is near zero. Only
  three also show real 30-minute bursts, so the cluster splits 3-and-3 on the
  shortlist. Requiring both methods to agree keeps statistical noise off the
  list.
- **Validation against real storms.** The June 2025 flash spike (412k flashes)
  lines up day for day with NOAA Storm Events reports: flash flooding on June
  9–10, an EF0 tornado on June 17, and severe wind and hail on June 23 and 25.
  All five days trip both detectors under the 4-year baseline.

## Method

```
GOES GLM (AWS S3) ─┐
Census 2020 ───────┼─► spatial join (flash → block group) ─► anomaly detection ─┐
FDEM shelters ─────┤                                                            ├─► combined priority ─► shortlist + dashboard
OSM road network ──┴─► multi-source Dijkstra drive time ─► suitability overlay ─┘
                                                     └──► ML anomaly classifier (block group × month)
```

- **Anomaly detection** (`src/anomaly_detection_4yr.py`) uses two methods.
  (1) A daily z-score against each block group's own history (z ≥ 3).
  (2) An absolute rolling 30-minute burst threshold (≥ 5 flashes). A day counts
  as confirmed only when both agree.
- **Routing** (`src/routing_analysis.py`) computes drive time from every block
  group to its nearest shelter over the OSM drivable network. Edge speeds come
  from `maxspeed` tags, and a multi-source Dijkstra runs from all shelter nodes
  at once.
- **Water polygons excluded.** Six zero-population Census "water" block groups
  (tract `99xxxx`) are removed up front, because no shelter can be built on
  open water.
- **Coastal 10 km buffer variant** (`src/coastal_buffer_analysis_4yr.py`)
  credits offshore flashes within 10 km to the nearest coastal block group.
  This raises flashes in coastal block groups by 218% and changes the
  shortlist from 37 to 47. Both versions are kept, because the buffer distance
  is an open methodological choice.

### Machine-learning layer

This layer predicts whether a block group will have a confirmed anomaly in a
given month. The data is 11,780 block-group-months, 29.8% positive, validated
with nested leave-one-county-out spatial cross-validation.

- **Six models compared** (LogReg, LogReg with interactions, random forest,
  HistGradientBoosting, decision tree, Naive Bayes). The top four fall within
  fold-to-fold noise of each other, so plain logistic regression is the
  practical pick.
- **Features that helped:** calendar/seasonality, NLCD land cover (more
  developed land means *fewer* anomalies), and the ENSO index (ONI).
  Month-to-month persistence features did not help.
- **Out-of-time test on 2026 found and fixed a real problem.** The 2-year model
  (2024–25) saturated on 2026's strong El Niño, predicting "anomaly" for 100% of
  rows. Its 0.812 AUC was therefore misleading. Retraining on four seasons that
  span La Niña to El Niño fixed it. The retrained model flags 40% of 2026 rows
  against a true rate of 35% (precision 0.592, recall 0.671). The gap between
  cross-validation and 2026 performance shrank from −0.040 to −0.006.

Figures 16, 18 and 19 in [`report_figures/`](report_figures/index.md) document
that before/after comparison. All other figures use the 4-year window.

## Data sources

| Dataset | Source | Used for |
|---|---|---|
| Lightning flashes | GOES-16 (2022–24) / GOES-19 (2025–26) Geostationary Lightning Mapper, NOAA on AWS Open Data | Hazard, anomalies |
| Population | US Census 2020 Decennial, block group | Exposure |
| Shelters | FL Division of Emergency Management, Risk Shelter Inventory (ArcGIS REST) | Coverage gap |
| Road network | OpenStreetMap via `osmnx` (61,253 nodes / 152,235 edges) | Drive-time routing |
| Land cover | USGS/MRLC NLCD 2021 | ML features |
| ENSO index | NOAA CPC Oceanic Niño Index | ML features |
| Severe-weather reports | NOAA Storm Events Database, 2016–2025 | Validation |

**Changes from the original plan:**
- **Lightning.** The XWeather historical archive needed a paid tier, so the
  project pulls GLM data directly from NOAA's public S3 buckets instead:
  about 2.6M NetCDF files for 2022–25 (11 failed downloads in total), plus
  523k files for 2026.
- **Shelters.** FGDL has no shelter layer. FDEM's statewide inventory covers all
  five counties from one authoritative source.

## Repository layout

```
panhandlestrike/
├── dashboard/        interactive Folium map (4-year) + archived 2-year version
├── data/processed/   analysis outputs (GLM Parquet, features, anomalies, suitability, final tables)
├── notes/            running-notes.md (full methodology log + audit trail), featured-findings write-up
├── notebooks/        exploration script
├── report_figures/   19 report figures + index.md
├── src/              every pull / feature / model / figure script
└── requirements.txt
```

Key outputs:
- `data/processed/final_4yr/blockgroup_combined_priority_4yr.{csv,geojson}`
  has one row per block group, with suitability, anomaly counts, drive time,
  and shortlist flag.
- `data/processed/features/model_comparison_4yr_results.csv` holds the
  six-model comparison.

`data/raw/`, `data/processed/osm/` and `data/processed/census/` are git-ignored,
because they can be re-downloaded from public sources.

## Reproducing

```bash
pip install -r requirements.txt
```

The Census pull needs a free `CENSUS_API_KEY` in a git-ignored `.env` file.
Every other source is public and needs no credentials.

Rough pipeline order (each script's docstring gives its inputs and outputs):

1. **Pulls:** `pull_glm_data.py`, `pull_glm_2022_2023.py`, `pull_glm_2026.py`,
   `pull_census_population.py`, `pull_census_geometry.py`, `pull_shelters.py`,
   `pull_osm_roads.py`, `pull_storm_events.py`
2. **Features:** `features_spatial_join.py`, `build_4yr_features.py`,
   `features_extra.py`, `features_nlcd.py`
3. **Analysis:** `routing_analysis.py`, `anomaly_detection_4yr.py`,
   `suitability_model_4yr.py`, `combined_priority_4yr.py`,
   `coastal_buffer_analysis_4yr.py`
4. **Models:** `model_comparison_4yr.py`, `retrain_4yr_model.py`,
   `validate_2026.py`
5. **Outputs:** `build_dashboard_4yr.py`, `generate_report_figures.py`

Scripts without the `_4yr` suffix are the original 2-year (2024–25) versions.
They are kept for the comparison section and the audit trail.

The full GLM pull is long: the 2024–25 pull alone took 13.9 h on a 16-process
pool. See `src/calibrate_*.py` for the concurrency benchmarking behind that
design.

## Limitations

- Suitability weights are equal by design, not calibrated against outcomes.
  NOAA records only 59 lightning-specific events in these counties over ten
  years, which is too few to fit weights to.
- The shortlist edge depends on the baseline. Moving from 2 to 4 seasons moved
  11 block groups across the line (42 → 37). Treat individual borderline block
  groups as sensitive to that choice.
- The ML ceiling is about 0.8 AUC. Exactly when a small block group gets a
  convective burst is partly unpredictable from geography and calendar alone.
- 2026 ONI (up to +1.80) still exceeds the training maximum (+1.50), so some
  extrapolation risk remains.

Full methodology, every number's derivation, and the audit trail are in
[`notes/running-notes.md`](notes/running-notes.md).
