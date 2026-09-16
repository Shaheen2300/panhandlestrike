"""4-year rerun of combined_priority.py: identical shortlist rule (top-quartile
suitability AND >15-min drive gap AND >=1 confirmed both-method anomaly day),
applied to the 4-year suitability (suitability_model_4yr.py, hazard =
2022-2025 flash density) and 4-year anomaly summary (anomaly_detection_4yr.py,
baseline = full 2022-2025 daily record) instead of the original 2024-25-only
inputs.

Output: data/processed/final_4yr/blockgroup_combined_priority_4yr.(csv|geojson)
"""

import os

import geopandas as gpd
import pandas as pd

SUIT_PATH = "data/processed/suitability_4yr/blockgroup_suitability_4yr.csv"
SUIT_GEOJSON = "data/processed/suitability_4yr/blockgroup_suitability_4yr.geojson"
ANOM_PATH = "data/processed/anomaly_4yr/blockgroup_anomaly_summary_4yr.csv"
ROUTE_PATH = "data/processed/routing/blockgroup_shelter_drivetime.csv"
OUT_DIR = "data/processed/final_4yr"
os.makedirs(OUT_DIR, exist_ok=True)

HIGH_SUITABILITY_QUANTILE = 0.25

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
combined.to_csv(os.path.join(OUT_DIR, "blockgroup_combined_priority_4yr.csv"), index=False)

geom = gpd.read_file(SUIT_GEOJSON)[["GEOID", "geometry"]]
combined_gdf = geom.merge(combined, on="GEOID", how="right")
gpd.GeoDataFrame(combined_gdf, geometry="geometry").to_file(
    os.path.join(OUT_DIR, "blockgroup_combined_priority_4yr.geojson"), driver="GeoJSON"
)

print(f"Combined table (4-year baseline): {n_bg} block groups")
print(f"High suitability (top {int(HIGH_SUITABILITY_QUANTILE*100)}%, rank <= {rank_cutoff:.0f}): "
      f"{df['high_suitability'].sum()}")
print(f"In >15-min coverage gap: {df['in_coverage_gap'].sum()}")
print(f"Has >=1 confirmed (both-method) anomaly day: {df['has_confirmed_anomaly'].sum()}")
print(f"\n=== FINAL SHORTLIST, 4-YEAR BASELINE (all three criteria): {df['on_shortlist'].sum()} block groups ===")

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

# --- direct comparison vs the original (2-year) shortlist ---
orig_path = "data/processed/final/blockgroup_combined_priority.csv"
if os.path.exists(orig_path):
    orig = pd.read_csv(orig_path, dtype={"GEOID": str})
    orig_set = set(orig.loc[orig["on_shortlist"], "GEOID"])
    new_set = set(combined.loc[combined["on_shortlist"], "GEOID"])
    print("\n" + "=" * 70)
    print("SHORTLIST CHANGE: 2-year baseline vs 4-year baseline")
    print("=" * 70)
    print(f"2-year shortlist: {len(orig_set)}   4-year shortlist: {len(new_set)}")
    print(f"Unchanged (in both): {len(orig_set & new_set)}")
    print(f"Dropped (was on 2yr list, not on 4yr list): {sorted(orig_set - new_set)}")
    print(f"Added (new to 4yr list, wasn't on 2yr list): {sorted(new_set - orig_set)}")

print("\n" + "=" * 70)
print("HIGHLIGHTED BLOCK GROUPS - 4-YEAR BASELINE")
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
    print(f"  ON SHORTLIST (4yr): {r['on_shortlist']}")
    if os.path.exists(orig_path):
        orow = orig[orig["GEOID"] == geoid]
        if not orow.empty:
            o = orow.iloc[0]
            print(f"  [was: rank {int(o['suitability_rank'])}, on_shortlist(2yr)={bool(o['on_shortlist'])}]")

print(f"\nSaved to {OUT_DIR}/blockgroup_combined_priority_4yr.(csv|geojson)")
