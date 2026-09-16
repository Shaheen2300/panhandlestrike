"""Deliverable 2 (parallel): shelter siting suitability score per block group.

Weighted overlay of three normalized factors, one row per Census block group:

  - exposure   : 2020 population (more people -> higher need)
  - hazard     : GLM flash density = flashes per km2 of land over both
                 storm seasons (more lightning -> higher need)
  - gap        : road-network drive time from the block group's internal
                 point to the nearest existing shelter (longer -> bigger
                 coverage gap). Uses the routing_analysis.py output, not
                 straight-line distance, so the feature reflects real
                 travel cost over the actual road network.

Each factor is percentile-ranked to 0-1 across the 589 land block groups,
then combined with weights. Higher score = higher priority for new
lightning shelter infrastructure.

Uses the same underlying GLM flash data as the anomaly deliverable, but is
otherwise independent of it. Depends on routing_analysis.py having been run.

Methodology - water block group exclusion: the Census block-group layer
includes 6 zero-population "water" polygons (tract codes 99xxxx) over the
Gulf and coastal bays. They are dropped before scoring, since a shelter
cannot be sited on open water and they carry no population to protect.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
FLASH_PATH = "data/processed/features/flashes_with_blockgroup.parquet"
DRIVETIME_PATH = "data/processed/routing/blockgroup_shelter_drivetime.csv"
OUT_DIR = "data/processed/suitability"
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = {"exposure": 0.34, "hazard": 0.33, "gap": 0.33}

bg = gpd.read_file(BG_PATH)

# Drop zero-population Gulf/bay water block groups (Census tract code 99xxxx)
n_before = len(bg)
water_mask = bg["TRACTCE"].str.startswith("99") | (bg["P1_001N"] == 0)
print(f"Excluding {int(water_mask.sum())} zero-population water block groups: "
      f"{sorted(bg.loc[water_mask, 'GEOID'])}")
bg = bg[~water_mask].reset_index(drop=True)
print(f"Block groups scored: {len(bg)} (dropped {n_before - len(bg)})")

# --- hazard: flash count + density per block group ---
flashes = pd.read_parquet(FLASH_PATH)
flash_counts = flashes.groupby("GEOID").size().rename("flash_count")
bg = bg.merge(flash_counts, on="GEOID", how="left")
bg["flash_count"] = bg["flash_count"].fillna(0)
land_km2 = (bg["ALAND"] / 1_000_000).clip(lower=0.01)  # guard tiny/zero land area
bg["flash_per_km2"] = bg["flash_count"] / land_km2

# --- gap: road-network drive time to nearest existing shelter ---
drivetime = pd.read_csv(DRIVETIME_PATH, dtype={"GEOID": str})
bg = bg.merge(
    drivetime[["GEOID", "drive_min_to_shelter", "within_5min", "within_10min", "within_15min"]],
    on="GEOID", how="left",
)
if bg["drive_min_to_shelter"].isna().any():
    raise SystemExit(f"{bg['drive_min_to_shelter'].isna().sum()} block groups missing drive time")

# --- normalize each factor to 0-1 by percentile rank ---
bg["exposure_norm"] = bg["P1_001N"].rank(pct=True)
bg["hazard_norm"] = bg["flash_per_km2"].rank(pct=True)
bg["gap_norm"] = bg["drive_min_to_shelter"].rank(pct=True)

bg["suitability_score"] = (
    WEIGHTS["exposure"] * bg["exposure_norm"]
    + WEIGHTS["hazard"] * bg["hazard_norm"]
    + WEIGHTS["gap"] * bg["gap_norm"]
)
bg["suitability_rank"] = bg["suitability_score"].rank(ascending=False, method="min").astype(int)

# --- save ---
keep_cols = [
    "GEOID", "COUNTY_NAME", "NAMELSAD", "P1_001N", "pop_density_per_sqkm",
    "flash_count", "flash_per_km2", "drive_min_to_shelter",
    "within_5min", "within_10min", "within_15min",
    "exposure_norm", "hazard_norm", "gap_norm",
    "suitability_score", "suitability_rank", "geometry",
]
out = bg[keep_cols].sort_values("suitability_rank")
out.to_file(os.path.join(OUT_DIR, "blockgroup_suitability.geojson"), driver="GeoJSON")
out.drop(columns="geometry").to_csv(os.path.join(OUT_DIR, "blockgroup_suitability.csv"), index=False)

print(f"Scored {len(out)} block groups. Weights: {WEIGHTS}")
print(f"Saved to {OUT_DIR}/blockgroup_suitability.(geojson|csv)")
print("\nTop 15 priority block groups:")
disp = out.drop(columns="geometry").head(15)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(disp[["suitability_rank", "GEOID", "COUNTY_NAME", "P1_001N",
                "flash_count", "flash_per_km2", "drive_min_to_shelter", "suitability_score"]].to_string(index=False))

print("\nSuitability score distribution:")
print(out["suitability_score"].describe())
print("\nTop-50 priority block groups by county:")
print(out.head(50)["COUNTY_NAME"].value_counts())
