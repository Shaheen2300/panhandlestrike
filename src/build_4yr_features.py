"""Extends the anomaly-detection feature pipeline from the 2-year (2024-25)
window to a 4-year window (2022-2025), so ONI spans genuinely different ENSO
states in training: 2022 La Nina, 2023 transitioning La Nina -> El Nino,
2024-25 near-neutral.

Deliberately does NOT keep a per-year dummy (the 2-year model's `is_2025`
term): with only 2 years, that dummy and ONI were nearly redundant proxies for
"which year is this", which is exactly why the earlier ONI coefficient could
not be trusted as a climate signal. Dropping the year dummy here gives ONI an
uncorrupted test - if it now carries real information, it has to do so on its
own, not by piggy-backing on a year indicator.

Baseline note: the z-score anomaly baseline (per-block-group daily mean/std)
is recomputed over the FULL 4-year daily record, so 2024-25 anomaly labels in
this run are not numerically identical to the original 2-year-only labels
(more history -> a more stable baseline estimate). This is the correct and
expected consequence of "rerun the feature pipeline" on the wider window.

Outputs:
  data/processed/features/flashes_with_blockgroup_4yr.parquet
  data/processed/features/blockgroup_month_4yr.csv           (2022-25 training table)
  data/processed/features/blockgroup_month_2026_test_4yr.csv (2026 test table, same features)
"""

import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd

BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
EXTRA = pd.read_csv("data/processed/features/blockgroup_features_extra.csv", dtype={"GEOID": str})
NLCD = pd.read_csv("data/processed/features/blockgroup_nlcd.csv", dtype={"GEOID": str})
OUT_DIR = "data/processed/features"

Z_THRESHOLD = 3.0
BURST_WINDOW = pd.Timedelta(minutes=30)
BURST_MIN = 5
TRAIN_YEARS = [2022, 2023, 2024, 2025]
MONTHS = range(5, 10)
MID_DOY = {5: 135, 6: 166, 7: 196, 8: 227, 9: 258}

# real NOAA CPC ONI, centred 3-month season mapped to the study month
ONI = {
    (2022, 5): -0.83, (2022, 6): -0.73, (2022, 7): -0.70, (2022, 8): -0.78, (2022, 9): -0.87,
    (2023, 5): 0.46, (2023, 6): 0.73, (2023, 7): 1.00, (2023, 8): 1.25, (2023, 9): 1.50,
    (2024, 5): 0.43, (2024, 6): 0.18, (2024, 7): 0.06, (2024, 8): -0.04, (2024, 9): -0.12,
    (2025, 5): -0.04, (2025, 6): -0.02, (2025, 7): -0.11, (2025, 8): -0.26, (2025, 9): -0.43,
    (2026, 5): 0.95, (2026, 6): 1.39, (2026, 7): 1.80, (2026, 8): 1.80,
}

bg = gpd.read_file(BG_PATH).to_crs("EPSG:4326")
water_mask = bg["TRACTCE"].str.startswith("99") | (bg["P1_001N"] == 0)
land = bg[~water_mask].copy()
LAND_IDS = set(land["GEOID"])


def spatial_join_years(years):
    files = []
    for y in years:
        for m in MONTHS:
            files.extend(glob.glob(f"data/processed/glm/glm_panhandle_{y}_{m:02d}.parquet"))
    fl = pd.concat([pd.read_parquet(f) for f in sorted(files)], ignore_index=True)
    fl["timestamp"] = pd.to_datetime(fl["timestamp"])
    pts = gpd.GeoDataFrame(fl, geometry=gpd.points_from_xy(fl["flash_lon"], fl["flash_lat"]), crs="EPSG:4326")
    j = gpd.sjoin(pts, land[["GEOID", "COUNTY_NAME", "geometry"]], how="inner", predicate="within")
    return j[["flash_id", "flash_lat", "flash_lon", "flash_area", "flash_energy",
             "timestamp", "GEOID", "COUNTY_NAME"]].reset_index(drop=True)


def anomaly_both_flags(flash_df):
    """Per-BG-day z-score (baseline = whole supplied period) + burst; returns
    a GEOID x year x month -> target (>=1 both-flag day) table."""
    j = flash_df.copy()
    j["date"] = j["timestamp"].dt.date
    daily = j.groupby(["GEOID", "date"]).size().rename("c").reset_index()
    all_days = pd.date_range(j["timestamp"].min().normalize(), j["timestamp"].max().normalize(), freq="D").date
    full = (daily.set_index(["GEOID", "date"])
            .reindex(pd.MultiIndex.from_product([land["GEOID"], all_days], names=["GEOID", "date"]), fill_value=0)
            .reset_index())
    stats = full.groupby("GEOID")["c"].agg(mu="mean", sd="std").reset_index()
    full = full.merge(stats, on="GEOID")
    full["z"] = np.where(full["sd"] > 0, (full["c"] - full["mu"]) / full["sd"], np.nan)
    full["z_flag"] = full["z"] >= Z_THRESHOLD

    peaks = []
    for gid, grp in j.groupby("GEOID"):
        t = np.sort(grp["timestamp"].values)
        k = 0
        peak = {}
        for i in range(len(t)):
            while t[i] - t[k] >= BURST_WINDOW:
                k += 1
            d = pd.Timestamp(t[i]).date()
            peak[d] = max(peak.get(d, 0), i - k + 1)
        for d, v in peak.items():
            peaks.append((gid, d, v))
    pk = pd.DataFrame(peaks, columns=["GEOID", "date", "peak_30min"])
    pk["burst_flag"] = pk["peak_30min"] >= BURST_MIN

    m = full.merge(pk, on=["GEOID", "date"], how="left")
    m["burst_flag"] = m["burst_flag"].fillna(False)
    m["both"] = m["z_flag"] & m["burst_flag"]
    m["date"] = pd.to_datetime(m["date"])
    m["year"], m["month"] = m["date"].dt.year, m["date"].dt.month
    return m.groupby(["GEOID", "year", "month"])["both"].any().astype(int).rename("target").reset_index()


def attach_features(df, oni_map):
    df = df.merge(EXTRA[["GEOID", "COUNTY_NAME", "P1_001N", "drive_min_to_shelter", "coast_dist_km"]],
                  on="GEOID", how="left")
    df = df.merge(NLCD[["GEOID", "pct_developed", "pct_forest", "pct_wetland"]], on="GEOID", how="left")
    df["is_may"] = (df["month"] == 5).astype(int)
    df["oni"] = [oni_map[(y, m)] for y, m in zip(df["year"], df["month"])]
    doy = df["month"].map(MID_DOY).astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


if __name__ == "__main__":
    print("Spatial-joining 2022-2025 flashes to land block groups...")
    j4 = spatial_join_years(TRAIN_YEARS)
    j4.to_parquet(os.path.join(OUT_DIR, "flashes_with_blockgroup_4yr.parquet"), index=False)
    print(f"  {len(j4):,} flashes on land, 2022-2025")

    # 4-year baseline flash density (replaces the 2-year flash_per_km2)
    counts4 = j4.groupby("GEOID").size().rename("flash_count_4yr")
    land_km2 = (land.set_index("GEOID")["ALAND"] / 1_000_000).clip(lower=0.01)
    flash_density_4yr = (counts4 / land_km2).rename("flash_per_km2_4yr").reset_index()
    flash_density_4yr.to_csv(os.path.join(OUT_DIR, "blockgroup_flash_density_4yr.csv"), index=False)

    print("Computing 4-year both-method anomaly labels...")
    bm4 = anomaly_both_flags(j4)

    grid = pd.MultiIndex.from_product(
        [land["GEOID"], [(y, m) for y in TRAIN_YEARS for m in MONTHS]], names=["GEOID", "ym"]
    ).to_frame(index=False)
    grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
    grid = grid.drop(columns="ym").merge(bm4, on=["GEOID", "year", "month"], how="left")
    grid["target"] = grid["target"].fillna(0).astype(int)

    lut = grid.set_index(["GEOID", "year", "month"])["target"].to_dict()
    grid["prev_month_flagged"] = [lut.get((g, y, m - 1), 0) if m > 5 else 0
                                  for g, y, m in zip(grid.GEOID, grid.year, grid.month)]
    grid["same_month_last_yr_flagged"] = [lut.get((g, y - 1, m), 0) if y > 2022 else 0
                                          for g, y, m in zip(grid.GEOID, grid.year, grid.month)]
    grid = grid.merge(flash_density_4yr, on="GEOID", how="left")
    grid = attach_features(grid, ONI)

    train_path = os.path.join(OUT_DIR, "blockgroup_month_4yr.csv")
    grid.to_csv(train_path, index=False)
    print(f"Saved 4-year training table: {len(grid)} rows, positive rate {grid['target'].mean():.3f} -> {train_path}")
    print(grid.groupby("year")["target"].mean().round(3))

    # ---- rebuild the 2026 test table with the SAME feature list (no year dummy) ----
    print("\nRebuilding 2026 test table (matching 4-year feature set)...")
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import validate_2026 as v26

    j26 = v26.spatial_join_2026()
    bm26 = v26.anomaly_flags_2026(j26)  # 2024-25 baseline, as validated before - kept as-is for the 2026 target
    lut26 = {(g, m): int(bm26[(bm26.GEOID == g) & (bm26.month == m)]["target"].sum() > 0)
            for g in EXTRA["GEOID"] for m in range(5, 9)}
    _, train_lut_2yr = v26.build_training()
    last_yr_2026 = {(g, m): train_lut_2yr.get((g, 2025, m), 0) for g in EXTRA["GEOID"] for m in range(5, 9)}
    prev_2026 = {(g, m): lut26.get((g, m - 1), 0) if m > 5 else 0 for g in EXTRA["GEOID"] for m in range(5, 9)}

    full26 = (pd.MultiIndex.from_product([EXTRA["GEOID"], range(5, 9)], names=["GEOID", "month"])
             .to_frame(index=False).merge(bm26, on=["GEOID", "month"], how="left"))
    full26["target"] = full26["target"].fillna(0).astype(int)
    full26["year"] = 2026
    full26["prev_month_flagged"] = [prev_2026[(g, m)] for g, m in zip(full26.GEOID, full26.month)]
    full26["same_month_last_yr_flagged"] = [last_yr_2026[(g, m)] for g, m in zip(full26.GEOID, full26.month)]
    full26 = full26.merge(flash_density_4yr, on="GEOID", how="left")
    full26 = attach_features(full26, ONI)

    test_path = os.path.join(OUT_DIR, "blockgroup_month_2026_test_4yr.csv")
    full26.to_csv(test_path, index=False)
    print(f"Saved 2026 test table: {len(full26)} rows, positive rate {full26['target'].mean():.3f} -> {test_path}")
