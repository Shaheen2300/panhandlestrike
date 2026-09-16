"""4-year rerun of coastal_buffer_analysis.py: identical method (10 km
sjoin_nearest credit of offshore flashes to the nearest coastal block group;
10 km is a meteorological estimate, not tuned), identical shortlist rule,
but built on the full 2022-2025 flash record and compared against the 4-year
land-only outputs (final_4yr/) instead of the 2024-25-only ones.

This script does NOT touch the original 2-year buffered outputs
(data/processed/coastal_buffer_10km/) - it writes a separate
data/processed/coastal_buffer_10km_4yr/ directory.
"""

import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd

BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
GLM_DIR = "data/processed/glm"
ORIG_JOIN_4YR = "data/processed/features/flashes_with_blockgroup_4yr.parquet"
ORIG_COMBINED_4YR = "data/processed/final_4yr/blockgroup_combined_priority_4yr.csv"
DRIVETIME = "data/processed/routing/blockgroup_shelter_drivetime.csv"
EXTRA = "data/processed/features/blockgroup_features_extra.csv"

OUT_DIR = "data/processed/coastal_buffer_10km_4yr"
os.makedirs(OUT_DIR, exist_ok=True)

BUFFER_M = 10_000
PROJ = "EPSG:5070"
WEIGHTS = {"exposure": 0.34, "hazard": 0.33, "gap": 0.33}
Z_THRESHOLD = 3.0
BURST_WINDOW = pd.Timedelta(minutes=30)
BURST_MIN = 5
HIGH_SUIT_Q = 0.25
WALTON = "121319503053"
OKALOOSA_CLUSTER = ["120910210011", "120910210012", "120910210013",
                    "120910210021", "120910210022", "120910210023"]

bg = gpd.read_file(BG_PATH)
water_mask = bg["TRACTCE"].str.startswith("99") | (bg["P1_001N"] == 0)
land = bg[~water_mask].copy()

# ---------------------------------------------------------------------------
# 1. buffered spatial join - 2022-2025 (all 4 training years, matches final_4yr)
# ---------------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(GLM_DIR, "glm_panhandle_20*_0*.parquet")))
files = [f for f in files if "2026" not in f]
fl = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
# restrict to the 4-year training window months/years, matching build_4yr_features.py
fl["timestamp"] = pd.to_datetime(fl["timestamp"])
fl = fl[fl["timestamp"].dt.year.isin([2022, 2023, 2024, 2025])]
pts = gpd.GeoDataFrame(fl, geometry=gpd.points_from_xy(fl["flash_lon"], fl["flash_lat"]),
                       crs="EPSG:4326").to_crs(PROJ)
land_proj = land[["GEOID", "COUNTY_NAME", "geometry"]].to_crs(PROJ)

within = gpd.sjoin(pts, land_proj, how="left", predicate="within")
within = within[~within.index.duplicated(keep="first")]
on_land = within[within["GEOID"].notna()].copy()

offshore = within[within["GEOID"].isna()].drop(columns=["GEOID", "COUNTY_NAME", "index_right"])
near = gpd.sjoin_nearest(offshore, land_proj, how="left", max_distance=BUFFER_M,
                         distance_col="coast_gap_m")
near = near[~near.index.duplicated(keep="first")]
near_matched = near[near["GEOID"].notna()].copy()

buffered = pd.concat([
    on_land[["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy", "timestamp", "GEOID", "COUNTY_NAME"]],
    near_matched[["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy", "timestamp", "GEOID", "COUNTY_NAME"]],
], ignore_index=True)
buffered.to_parquet(os.path.join(OUT_DIR, "flashes_with_blockgroup_buffered10km_4yr.parquet"), index=False)

n_total = len(fl)
n_land = len(on_land)
n_buffer_gained = len(near_matched)
print(f"Total 2022-2025 flashes:                {n_total:,}")
print(f"On land (original join):                {n_land:,}  ({n_land/n_total*100:.1f}%)")
print(f"Newly credited within 10 km of coast:   {n_buffer_gained:,}  (+{n_buffer_gained/n_land*100:.1f}% over land)")
print(f"Still unmatched (>10 km offshore):      {n_total - n_land - n_buffer_gained:,}")

# ---------------------------------------------------------------------------
# 2. flash-count comparison per block group (coastal focus)
# ---------------------------------------------------------------------------
orig_counts = pd.read_parquet(ORIG_JOIN_4YR).groupby("GEOID").size().rename("flash_orig")
buf_counts = buffered.groupby("GEOID").size().rename("flash_buffered")
extra = pd.read_csv(EXTRA, dtype={"GEOID": str})[["GEOID", "coast_dist_km"]]

cmp = (land[["GEOID", "COUNTY_NAME", "P1_001N", "ALAND"]]
       .merge(orig_counts, on="GEOID", how="left")
       .merge(buf_counts, on="GEOID", how="left")
       .merge(extra, on="GEOID", how="left"))
cmp[["flash_orig", "flash_buffered"]] = cmp[["flash_orig", "flash_buffered"]].fillna(0)
cmp["flash_delta"] = cmp["flash_buffered"] - cmp["flash_orig"]
cmp["pct_change"] = np.where(cmp["flash_orig"] > 0, cmp["flash_delta"] / cmp["flash_orig"] * 100, np.nan)
cmp["is_coastal"] = cmp["coast_dist_km"] <= 5

coastal = cmp[cmp["is_coastal"]]
print(f"\nCoastal block groups (<=5 km from coast): {len(coastal)} of {len(cmp)}")
print(f"  their flashes: {int(coastal['flash_orig'].sum()):,} -> {int(coastal['flash_buffered'].sum()):,} "
      f"(+{int(coastal['flash_delta'].sum()):,}, "
      f"{coastal['flash_delta'].sum()/max(coastal['flash_orig'].sum(),1)*100:+.0f}%)")
cmp.to_csv(os.path.join(OUT_DIR, "flash_count_comparison_4yr.csv"), index=False)

# ---------------------------------------------------------------------------
# 3. anomaly flags on the buffered join (both methods)
# ---------------------------------------------------------------------------
def both_method_summary(join_df):
    j = join_df.copy()
    j["date"] = pd.to_datetime(j["timestamp"]).dt.date
    daily = j.groupby(["GEOID", "date"]).size().rename("c").reset_index()
    all_days = pd.date_range(pd.to_datetime(j["timestamp"]).min().normalize(),
                             pd.to_datetime(j["timestamp"]).max().normalize(), freq="D").date
    full = (daily.set_index(["GEOID", "date"])
            .reindex(pd.MultiIndex.from_product([land["GEOID"], all_days], names=["GEOID", "date"]), fill_value=0)
            .reset_index())
    st = full.groupby("GEOID")["c"].agg(mu="mean", sd="std").reset_index()
    full = full.merge(st, on="GEOID")
    full["z"] = np.where(full["sd"] > 0, (full["c"] - full["mu"]) / full["sd"], np.nan)
    full["z_flag"] = full["z"] >= Z_THRESHOLD

    peaks = []
    for g, grp in j.groupby("GEOID"):
        t = np.sort(grp["timestamp"].values)
        pk = {}
        k = 0
        for i in range(len(t)):
            while t[i] - t[k] >= BURST_WINDOW:
                k += 1
            d = pd.Timestamp(t[i]).date()
            pk[d] = max(pk.get(d, 0), i - k + 1)
        for d, v in pk.items():
            peaks.append((g, d, v))
    pk = pd.DataFrame(peaks, columns=["GEOID", "date", "peak_30min"])
    pk["burst_flag"] = pk["peak_30min"] >= BURST_MIN
    m = full.merge(pk, on=["GEOID", "date"], how="left")
    m["burst_flag"] = m["burst_flag"].fillna(False)
    m["both"] = m["z_flag"] & m["burst_flag"]
    return m.groupby("GEOID")["both"].sum().rename("n_days_both_flag").reset_index()


anom_buf = both_method_summary(buffered)

# ---------------------------------------------------------------------------
# 4. suitability re-run with buffered 4-year flash density
# ---------------------------------------------------------------------------
dt = pd.read_csv(DRIVETIME, dtype={"GEOID": str})[["GEOID", "drive_min_to_shelter", "within_15min"]]
s = (land[["GEOID", "COUNTY_NAME", "P1_001N", "ALAND"]]
     .merge(buf_counts.rename("flash_count"), on="GEOID", how="left")
     .merge(dt, on="GEOID", how="left"))
s["flash_count"] = s["flash_count"].fillna(0)
s["flash_per_km2"] = s["flash_count"] / (s["ALAND"] / 1e6).clip(lower=0.01)
s["exposure_norm"] = s["P1_001N"].rank(pct=True)
s["hazard_norm"] = s["flash_per_km2"].rank(pct=True)
s["gap_norm"] = s["drive_min_to_shelter"].rank(pct=True)
s["suitability_score"] = (WEIGHTS["exposure"] * s["exposure_norm"]
                          + WEIGHTS["hazard"] * s["hazard_norm"]
                          + WEIGHTS["gap"] * s["gap_norm"])
s["suitability_rank"] = s["suitability_score"].rank(ascending=False, method="min").astype(int)
s.sort_values("suitability_rank").to_csv(os.path.join(OUT_DIR, "blockgroup_suitability_buffered10km_4yr.csv"), index=False)

# ---------------------------------------------------------------------------
# 5. buffered shortlist + comparison to the 4-year land-only shortlist
# ---------------------------------------------------------------------------
buf = (s[["GEOID", "COUNTY_NAME", "P1_001N", "suitability_rank", "suitability_score",
          "flash_count", "drive_min_to_shelter", "within_15min"]]
       .merge(anom_buf, on="GEOID", how="left"))
buf["n_days_both_flag"] = buf["n_days_both_flag"].fillna(0).astype(int)
cut = buf["suitability_rank"].quantile(HIGH_SUIT_Q)
buf["on_shortlist_buffered"] = ((buf["suitability_rank"] <= cut)
                                & (buf["drive_min_to_shelter"] > 15)
                                & (buf["n_days_both_flag"] >= 1))
buf.sort_values("suitability_rank").to_csv(
    os.path.join(OUT_DIR, "blockgroup_combined_priority_buffered10km_4yr.csv"), index=False)

orig = pd.read_csv(ORIG_COMBINED_4YR, dtype={"GEOID": str})
merged = orig[["GEOID", "COUNTY_NAME", "P1_001N", "suitability_rank", "on_shortlist"]].merge(
    buf[["GEOID", "suitability_rank", "on_shortlist_buffered", "n_days_both_flag"]],
    on="GEOID", suffixes=("_orig", "_buf"))
merged["rank_shift"] = merged["suitability_rank_orig"] - merged["suitability_rank_buf"]

print("\n" + "=" * 74)
print("SHORTLIST (4-YEAR BASELINE): land-only  vs  10 km buffered")
print("=" * 74)
n_orig = int(merged["on_shortlist"].sum())
n_buf = int(merged["on_shortlist_buffered"].sum())
added = merged[(~merged["on_shortlist"]) & (merged["on_shortlist_buffered"])]
dropped = merged[(merged["on_shortlist"]) & (~merged["on_shortlist_buffered"])]
stayed = merged[(merged["on_shortlist"]) & (merged["on_shortlist_buffered"])]
print(f"land-only (4yr) shortlist: {n_orig}   buffered (4yr) shortlist: {n_buf}")
print(f"  stayed on: {len(stayed)}   added: {len(added)}   dropped: {len(dropped)}")
if len(added):
    print("\n  ADDED by buffer:")
    print(added[["GEOID", "COUNTY_NAME", "P1_001N", "suitability_rank_orig", "suitability_rank_buf", "n_days_both_flag"]].to_string(index=False))
if len(dropped):
    print("\n  DROPPED by buffer:")
    print(dropped[["GEOID", "COUNTY_NAME", "P1_001N", "suitability_rank_orig", "suitability_rank_buf"]].to_string(index=False))

print("\n" + "=" * 74)
print("FEATURED FINDINGS under the 4-yr buffer")
print("=" * 74)
for gid, lbl in [(WALTON, "Walton 121319503053")] + [(g, "N. Okaloosa cluster") for g in OKALOOSA_CLUSTER]:
    r = merged[merged["GEOID"] == gid]
    if r.empty:
        continue
    r = r.iloc[0]
    cd = cmp[cmp["GEOID"] == gid].iloc[0]
    print(f"{gid} [{lbl}]  rank {int(r['suitability_rank_orig'])} -> {int(r['suitability_rank_buf'])}  "
          f"(shift {int(r['rank_shift']):+d})  flashes {int(cd['flash_orig'])} -> {int(cd['flash_buffered'])}  "
          f"shortlist {bool(r['on_shortlist'])} -> {bool(r['on_shortlist_buffered'])}")

print(f"\nAll 4-yr buffered outputs written under {OUT_DIR}/")
