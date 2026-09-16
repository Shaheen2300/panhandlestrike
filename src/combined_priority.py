"""Combines the three feature-engineering deliverables into one table per
block group, and derives a final prioritized siting shortlist.

Inputs (each produced by its own script):
  - suitability : data/processed/suitability/blockgroup_suitability.csv
  - anomaly     : data/processed/anomaly/blockgroup_anomaly_summary.csv
  - routing     : data/processed/routing/blockgroup_shelter_drivetime.csv

Shortlist = block groups that meet all three of:
  1. high suitability   : suitability_rank in the top quartile (top 25%)
  2. coverage gap        : > 15 minute drive to the nearest existing shelter
  3. confirmed anomaly   : >= 1 day flagged by BOTH the z-score and the
                           rolling-30-min burst method
"""

import os

import geopandas as gpd
import pandas as pd

SUIT_PATH = "data/processed/suitability/blockgroup_suitability.csv"
ANOM_PATH = "data/processed/anomaly/blockgroup_anomaly_summary.csv"
ROUTE_PATH = "data/processed/routing/blockgroup_shelter_drivetime.csv"
SUIT_GEOJSON = "data/processed/suitability/blockgroup_suitability.geojson"
OUT_DIR = "data/processed/final"
os.makedirs(OUT_DIR, exist_ok=True)

HIGH_SUITABILITY_QUANTILE = 0.25  # top 25% by suitability rank

HIGHLIGHT = {
    "121319503053": "Walton - most isolated block group (routing analysis)",
    "120910210011": "N. Okaloosa cluster",
    "120910210012": "N. Okaloosa cluster",
    "120910210013": "N. Okaloosa cluster",
    "120910210021": "N. Okaloosa cluster",
    "120910210022": "N. Okaloosa cluster",
    "120910210023": "N. Okaloosa cluster",
}

suit = pd.read_csv(SUIT_PATH, dtype={"GEOID": str})
anom = pd.read_csv(ANOM_PATH, dtype={"GEOID": str})
route = pd.read_csv(ROUTE_PATH, dtype={"GEOID": str})

df = suit.merge(anom, on="GEOID", how="left").merge(
    route[["GEOID", "drive_min_to_shelter", "within_5min", "within_10min", "within_15min"]],
    on="GEOID", how="left", suffixes=("", "_route"),
)

for c in ["n_days_zscore_flag", "n_days_burst_flag", "n_days_both_flag", "n_days_any_flag"]:
    df[c] = df[c].fillna(0).astype(int)
df["max_zscore"] = df["max_zscore"].round(1)

# coverage tier
def tier(m):
    if m <= 5:
        return "0-5 min"
    if m <= 10:
        return "5-10 min"
    if m <= 15:
        return "10-15 min"
    return ">15 min (GAP)"

df["coverage_tier"] = df["drive_min_to_shelter"].apply(tier)

n_bg = len(df)
rank_cutoff = df["suitability_rank"].quantile(HIGH_SUITABILITY_QUANTILE)
df["high_suitability"] = df["suitability_rank"] <= rank_cutoff
df["in_coverage_gap"] = df["drive_min_to_shelter"] > 15
df["has_confirmed_anomaly"] = df["n_days_both_flag"] >= 1
df["on_shortlist"] = df["high_suitability"] & df["in_coverage_gap"] & df["has_confirmed_anomaly"]

col_order = [
    "GEOID", "COUNTY_NAME", "NAMELSAD", "P1_001N",
    "suitability_rank", "suitability_score",
    "flash_count", "flash_per_km2",
    "drive_min_to_shelter", "coverage_tier",
    "n_days_zscore_flag", "n_days_burst_flag", "n_days_both_flag", "n_days_any_flag",
    "max_zscore", "max_peak_30min", "total_burst_episodes",
    "high_suitability", "in_coverage_gap", "has_confirmed_anomaly", "on_shortlist",
]
combined = df[col_order].sort_values("suitability_rank")
combined.to_csv(os.path.join(OUT_DIR, "blockgroup_combined_priority.csv"), index=False)

# attach geometry for a mappable output
geom = gpd.read_file(SUIT_GEOJSON)[["GEOID", "geometry"]]
combined_gdf = geom.merge(combined, on="GEOID", how="right")
gpd.GeoDataFrame(combined_gdf, geometry="geometry").to_file(
    os.path.join(OUT_DIR, "blockgroup_combined_priority.geojson"), driver="GeoJSON"
)

print(f"Combined table: {n_bg} block groups")
print(f"High suitability (top {int(HIGH_SUITABILITY_QUANTILE*100)}%, rank <= {rank_cutoff:.0f}): "
      f"{df['high_suitability'].sum()}")
print(f"In >15-min coverage gap: {df['in_coverage_gap'].sum()}")
print(f"Has >=1 confirmed (both-method) anomaly day: {df['has_confirmed_anomaly'].sum()}")
print(f"\n=== FINAL SHORTLIST (all three criteria): {df['on_shortlist'].sum()} block groups ===")

shortlist = combined[combined["on_shortlist"]].sort_values("suitability_rank")
short_pop = shortlist["P1_001N"].sum()
print(f"Total population in shortlisted block groups: {short_pop:,}")
with pd.option_context("display.width", 240, "display.max_columns", 30):
    print(shortlist[[
        "suitability_rank", "GEOID", "COUNTY_NAME", "P1_001N", "suitability_score",
        "drive_min_to_shelter", "flash_count", "n_days_both_flag", "max_zscore", "max_peak_30min",
    ]].to_string(index=False))

print("\nShortlist by county:")
print(shortlist["COUNTY_NAME"].value_counts())

print("\n" + "=" * 70)
print("HIGHLIGHTED BLOCK GROUPS")
print("=" * 70)
for geoid, label in HIGHLIGHT.items():
    row = combined[combined["GEOID"] == geoid]
    if row.empty:
        print(f"{geoid} ({label}): not in table")
        continue
    r = row.iloc[0]
    print(f"\n{geoid}  [{label}]  {r['COUNTY_NAME']}, pop {r['P1_001N']}")
    print(f"  suitability rank:   {r['suitability_rank']} / {n_bg}  (score {r['suitability_score']:.3f})")
    print(f"  drive to shelter:   {r['drive_min_to_shelter']:.1f} min  ->  {r['coverage_tier']}")
    print(f"  anomaly days:       z-score {r['n_days_zscore_flag']}, burst {r['n_days_burst_flag']}, "
          f"BOTH {r['n_days_both_flag']}  (max z {r['max_zscore']}, max 30-min peak {r['max_peak_30min']})")
    print(f"  criteria:  high_suitability={r['high_suitability']}  "
          f"in_gap={r['in_coverage_gap']}  confirmed_anomaly={r['has_confirmed_anomaly']}")
    print(f"  ON SHORTLIST: {r['on_shortlist']}")

print(f"\nSaved to {OUT_DIR}/blockgroup_combined_priority.(csv|geojson)")
