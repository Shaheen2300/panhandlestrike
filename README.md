# PanhandleStrike

GIS-based site suitability model for identifying priority locations for lightning shelter infrastructure across the Florida Panhandle (Escambia, Santa Rosa, Okaloosa, Walton, and Bay counties).

## Project overview

The model combines multi-year historical lightning strike density, population/visitor exposure, and existing shelter/facility gaps using a weighted overlay methodology, adapted from established GIS suitability modeling approaches previously applied to hurricane and earthquake shelter siting. Model outputs are validated against NOAA's historical lightning fatality and injury records.

A secondary component includes network analysis (closest facility / service area analysis) to estimate travel time from any point in the study area to the nearest existing shelter, plus a live lightning monitoring + real-time routing feature.

## Team

- Shaheen
- Koushal Doddipatla

## Folder structure

```
panhandlestrike/
├── data/         # raw and processed data (see .gitignore for exclusions)
├── notebooks/    # Jupyter notebooks for exploration and analysis
├── src/          # reusable Python scripts (data pulling, feature engineering, modeling)
├── README.md
└── requirements.txt
```

## Data sources

| Dataset | Source | Purpose |
|---|---|---|
| Lightning flashes | GOES-16/19 GLM (NOAA satellite, AWS Open Data) | Hazard feature: flash density |
| Lightning fatalities/injuries | NOAA Storm Events Database | Validation / target variable |
| Population | US Census 2020 Decennial (block group) | Exposure feature |
| Recreational points | OpenStreetMap | Exposure feature |
| Existing shelters | FL Division of Emergency Management (Risk Shelter Inventory) | Gap analysis |
| Road/path network | OpenStreetMap | Routing analysis |
| Land cover | USGS/MRLC NLCD | Contextual feature |

Notes on source changes from the original plan:
- **Lightning:** the XWeather historical archive (`/lightning/archive`) requires a paid subscription tier not available on the free key (confirmed: `404`/`insufficient_scope`). Pivoted to GOES GLM satellite data pulled directly from NOAA's public AWS bucket. GOES-16 covers 2024; GOES-19 covers 2025 (GOES-16 was retired as operational GOES-East in spring 2025).
- **Shelters:** FGDL has no emergency-shelter layer (checked its full 452-dataset catalog). FDEM's statewide Risk Shelter Inventory (ArcGIS REST) covers all 5 counties from one authoritative source.

## Setup

```bash
pip install -r requirements.txt
```

Some pulls need credentials in a `.env` file (git-ignored): `CENSUS_API_KEY` for the
Census population pull. GLM, shelter, OSM, and Storm Events pulls need no credentials.

## Status

**Data collection: complete.** All layers pulled and filtered to the 5-county study
area (Escambia, Santa Rosa, Okaloosa, Walton, Bay). Outputs in `data/processed/`.

| Dataset | Records | Coverage | Format |
|---|---|---|---|
| GLM lightning flashes | 1,958,298 flashes | May–Sep 2024 + May–Sep 2025, panhandle bbox (29.5–31.5°N, -88.0 to -84.5°E) | 10 monthly Parquet files (38 MB) |
| Census population | 595 block groups, 972,094 people (2020) | 5 counties | GeoJSON w/ geometry + `pop_density_per_sqkm` |
| Shelters | 194 facilities (137 general, 46 pet-friendly, 11 special-needs) | 5 counties | GeoJSON (per-type + combined) |
| Road network | 61,253 nodes / 152,235 edges | 5 counties, drivable network | GraphML (routable) + GeoJSON |
| Storm Events (validation) | 1,034 severe-weather rows; 59 lightning-specific | 5 counties, 2016–2025 | CSV |

GLM pull ran in 13.9 h (16-process pool, 1.32M files, 0 errors). See
`data/processed/glm/glm_pull_log.txt` for the full run log and `src/calibrate_*.py`
for the concurrency benchmarking that set the approach.

**In progress: feature engineering.** Two parallel deliverables, both built on
one flash→block-group spatial join (`src/features_spatial_join.py`; 501,502 of
1.96M flashes fall on the 5-county land area, rest over the Gulf).

Both deliverables first exclude the 6 zero-population Census "water" block groups
(tract codes `99xxxx`) covering the Gulf and coastal bays — a shelter can't be
sited on open water and they protect no population. After exclusion: 589 land
block groups, 407,486 flashes.

- **Anomaly / high-intensity alerting** (`src/anomaly_detection.py`) — dual method:
  (1) historical daily z-score vs each block group's own baseline (z≥3), and
  (2) absolute rolling-30-min burst threshold (≥5 flashes/window). The June 2025
  validation days (9, 10, 17, 23, 25) trip both methods across dozens of block
  groups each; flag-set overlap is 40% on those days vs 27% baseline.
- **Road-network routing** (`src/routing_analysis.py`) — drive time from each
  block group to the nearest existing shelter along the OSM network (edge speeds
  imputed from `maxspeed` tags; multi-source Dijkstra from all shelter nodes).
  86.8% of the 972k population is within a 15-minute drive of a shelter;
  the 13.2% gap (128,659 people, 83 block groups) is concentrated in rural
  northern Okaloosa and Walton counties, up to 28 minutes out.
- **Shelter siting suitability** (`src/suitability_model.py`) — weighted overlay
  of population + flash density + **network drive time to nearest shelter**
  (from the routing step, not straight-line distance), percentile-ranked,
  one score per block group.
- **Combined priority + shortlist** (`src/combined_priority.py`) — one table per
  block group joining all three, plus a final shortlist of block groups that are
  simultaneously top-quartile suitability, in the >15-min coverage gap, and have
  ≥1 day flagged by **both** anomaly methods. 42 block groups (81,702 people):
  Okaloosa 18, Walton 9, Bay 8, Santa Rosa 6, Escambia 1.

**Outputs:** `data/processed/final/blockgroup_combined_priority.{csv,geojson}`.

**Next:** live lightning monitoring + real-time routing feature; write-up /
validation against NOAA lightning fatality records.
