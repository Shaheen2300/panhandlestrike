"""4-year rerun of suitability_model.py: identical weighted-overlay method
(exposure 0.34 / hazard 0.33 / gap 0.33, percentile-ranked), with the hazard
factor swapped from the 2-year flash_per_km2 to the 4-year (2022-2025)
flash_per_km2_4yr computed in build_4yr_features.py. Population and
drive-time are geography-based and unchanged by the training-window choice.

Output: data/processed/suitability_4yr/blockgroup_suitability_4yr.(csv|geojson)
"""

import os

import geopandas as gpd
import pandas as pd

BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
FLASH_DENSITY_4YR_PATH = "data/processed/features/blockgroup_flash_density_4yr.csv"
FLASHES_4YR_PATH = "data/processed/features/flashes_with_blockgroup_4yr.parquet"
DRIVETIME_PATH = "data/processed/routing/blockgroup_shelter_drivetime.csv"
OUT_DIR = "data/processed/suitability_4yr"
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = {"exposure": 0.34, "hazard": 0.33, "gap": 0.33}

bg = gpd.read_file(BG_PATH)

n_before = len(bg)
water_mask = bg["TRACTCE"].str.startswith("99") | (bg["P1_001N"] == 0)
print(f"Excluding {int(water_mask.sum())} zero-population water block groups")
bg = bg[~water_mask].reset_index(drop=True)
print(f"Block groups scored: {len(bg)} (dropped {n_before - len(bg)})")

# --- hazard: 4-year (2022-2025) flash count + density per block group ---
flashes4 = pd.read_parquet(FLASHES_4YR_PATH, columns=["GEOID"])
count4 = flashes4.groupby("GEOID").size().rename("flash_count_4yr").reset_index()
density4 = pd.read_csv(FLASH_DENSITY_4YR_PATH, dtype={"GEOID": str})
bg = bg.merge(count4, on="GEOID", how="left").merge(density4, on="GEOID", how="left")
bg["flash_count_4yr"] = bg["flash_count_4yr"].fillna(0)
bg["flash_per_km2_4yr"] = bg["flash_per_km2_4yr"].fillna(0)
# keep the same column names as the 2-year table for downstream compatibility
bg["flash_count"] = bg["flash_count_4yr"]
bg["flash_per_km2"] = bg["flash_per_km2_4yr"]

# --- gap: road-network drive time (unchanged - geography, not year-dependent) ---
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

keep_cols = [
    "GEOID", "COUNTY_NAME", "NAMELSAD", "P1_001N", "pop_density_per_sqkm",
    "flash_count", "flash_per_km2", "drive_min_to_shelter",
    "within_5min", "within_10min", "within_15min",
    "exposure_norm", "hazard_norm", "gap_norm",
    "suitability_score", "suitability_rank", "geometry",
]
out = bg[keep_cols].sort_values("suitability_rank")
out.to_file(os.path.join(OUT_DIR, "blockgroup_suitability_4yr.geojson"), driver="GeoJSON")
out.drop(columns="geometry").to_csv(os.path.join(OUT_DIR, "blockgroup_suitability_4yr.csv"), index=False)

print(f"Scored {len(out)} block groups (4-year hazard). Weights: {WEIGHTS}")
print(f"Saved to {OUT_DIR}/blockgroup_suitability_4yr.(geojson|csv)")
print("\nTop 15 priority block groups (4-year hazard):")
disp = out.drop(columns="geometry").head(15)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(disp[["suitability_rank", "GEOID", "COUNTY_NAME", "P1_001N",
                "flash_count", "flash_per_km2", "drive_min_to_shelter", "suitability_score"]].to_string(index=False))
