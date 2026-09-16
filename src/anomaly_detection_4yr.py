"""4-year rerun of anomaly_detection.py's two methods, over the full 2022-2025
flash record instead of 2024-25 only. Same thresholds, same exclusions -
nothing recomputed a new way, only the baseline window is wider.

Purpose: (1) recalculated per-block-group anomaly summary to feed the 4-year
suitability/shortlist rebuild (combined_priority_4yr.py), and (2) re-check the
June 2025 validation days (originally built against the 2024-25-only z-score
baseline) against the now-shifted 4-year baseline - more history changes each
block group's own mean/std, so a day that was anomalous under the narrow
baseline is not guaranteed to still be anomalous under the wider one.

Output: data/processed/anomaly_4yr/blockgroup_anomaly_summary_4yr.csv
        data/processed/anomaly_4yr/flagged_blockgroup_days_4yr.csv
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

FLASH_PATH = "data/processed/features/flashes_with_blockgroup_4yr.parquet"
BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
OUT_DIR = "data/processed/anomaly_4yr"
os.makedirs(OUT_DIR, exist_ok=True)

Z_THRESHOLD = 3.0
BURST_WINDOW = pd.Timedelta(minutes=30)
BURST_MIN_FLASHES = 5
VALIDATION_DAYS = [pd.Timestamp(f"2025-06-{d:02d}").date() for d in (9, 10, 17, 23, 25)]

_bg = gpd.read_file(BG_PATH)
WATER_GEOIDS = set(_bg.loc[_bg["TRACTCE"].str.startswith("99") | (_bg["P1_001N"] == 0), "GEOID"])
LAND_GEOIDS = set(_bg["GEOID"]) - WATER_GEOIDS
print(f"Excluding {len(WATER_GEOIDS)} zero-population water block groups")

df = pd.read_parquet(FLASH_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date
n_before = len(df)
df = df[df["GEOID"].isin(LAND_GEOIDS)].reset_index(drop=True)
print(f"Flashes after water-block-group exclusion: {len(df):,} (dropped {n_before - len(df):,})")
print(f"Date span (4-year baseline window): {df['date'].min()} to {df['date'].max()}")

# ----------------------------------------------------------------------------
# Method 1: historical daily z-score per block group, baseline = full 4-year record
# ----------------------------------------------------------------------------
daily = df.groupby(["GEOID", "date"]).size().rename("flash_count").reset_index()

all_geoids = df["GEOID"].unique()
all_dates = pd.date_range(df["timestamp"].min().normalize(), df["timestamp"].max().normalize(), freq="D").date
full_index = pd.MultiIndex.from_product([all_geoids, all_dates], names=["GEOID", "date"])
daily_full = daily.set_index(["GEOID", "date"]).reindex(full_index, fill_value=0).reset_index()

stats = daily_full.groupby("GEOID")["flash_count"].agg(baseline_mean="mean", baseline_std="std").reset_index()
daily_full = daily_full.merge(stats, on="GEOID")
daily_full["zscore"] = np.where(
    daily_full["baseline_std"] > 0,
    (daily_full["flash_count"] - daily_full["baseline_mean"]) / daily_full["baseline_std"],
    np.nan,
)
daily_full["zscore_anomaly"] = daily_full["zscore"] >= Z_THRESHOLD

n_z_flags = int(daily_full["zscore_anomaly"].sum())
print(f"\n[Method 1] daily z-score >= {Z_THRESHOLD} (4-yr baseline): {n_z_flags:,} block-group-days flagged "
      f"across {daily_full.loc[daily_full['zscore_anomaly'], 'GEOID'].nunique()} block groups")

# ----------------------------------------------------------------------------
# Method 2: rolling 30-minute burst count per block group (baseline-independent,
# unchanged in method - but now computed over all 4 years of timestamps)
# ----------------------------------------------------------------------------
def peak_and_episodes(times):
    times = np.sort(times)
    n = len(times)
    out = {}
    j = 0
    day_peak = {}
    for i in range(n):
        while times[i] - times[j] >= BURST_WINDOW:
            j += 1
        w = i - j + 1
        d = pd.Timestamp(times[i]).date()
        if w > day_peak.get(d, 0):
            day_peak[d] = w
    episodes = {}
    i = 0
    while i < n:
        k = i
        while k < n and times[k] - times[i] < BURST_WINDOW:
            k += 1
        if k - i >= BURST_MIN_FLASHES:
            d = pd.Timestamp(times[i]).date()
            episodes[d] = episodes.get(d, 0) + 1
            i = k
        else:
            i += 1
    for d, pk in day_peak.items():
        out[d] = (pk, episodes.get(d, 0))
    return out


burst_rows = []
for geoid, grp in df.groupby("GEOID"):
    res = peak_and_episodes(grp["timestamp"].values)
    for d, (peak, eps) in res.items():
        burst_rows.append((geoid, d, peak, eps))

burst = pd.DataFrame(burst_rows, columns=["GEOID", "date", "peak_30min", "burst_episodes"])
burst["burst_alert"] = burst["peak_30min"] >= BURST_MIN_FLASHES

n_burst_flags = int(burst["burst_alert"].sum())
print(f"[Method 2] rolling 30-min burst >= {BURST_MIN_FLASHES}: {n_burst_flags:,} block-group-days flagged "
      f"across {burst.loc[burst['burst_alert'], 'GEOID'].nunique()} block groups")

# ----------------------------------------------------------------------------
# Combine + save
# ----------------------------------------------------------------------------
combined = daily_full.merge(burst, on=["GEOID", "date"], how="left")
combined["peak_30min"] = combined["peak_30min"].fillna(0).astype(int)
combined["burst_episodes"] = combined["burst_episodes"].fillna(0).astype(int)
combined["burst_alert"] = combined["burst_alert"].fillna(False)
combined["either_flag"] = combined["zscore_anomaly"] | combined["burst_alert"]
combined["both_flag"] = combined["zscore_anomaly"] & combined["burst_alert"]

flagged = combined[combined["either_flag"]].sort_values(["date", "flash_count"], ascending=[True, False])
flagged_path = os.path.join(OUT_DIR, "flagged_blockgroup_days_4yr.csv")
flagged.to_csv(flagged_path, index=False)
print(f"\nSaved {len(flagged):,} flagged block-group-days to {flagged_path}")

summary = combined.groupby("GEOID").agg(
    n_days_zscore_flag=("zscore_anomaly", "sum"),
    n_days_burst_flag=("burst_alert", "sum"),
    n_days_both_flag=("both_flag", "sum"),
    n_days_any_flag=("either_flag", "sum"),
    max_zscore=("zscore", "max"),
    max_peak_30min=("peak_30min", "max"),
    total_burst_episodes=("burst_episodes", "sum"),
    n_active_days=("flash_count", lambda s: int((s > 0).sum())),
).reset_index()
summary_path = os.path.join(OUT_DIR, "blockgroup_anomaly_summary_4yr.csv")
summary.to_csv(summary_path, index=False)
print(f"Saved per-block-group anomaly summary ({len(summary)} rows) to {summary_path}")

n_either = int(combined["either_flag"].sum())
n_both = int(combined["both_flag"].sum())
print(f"Flagged by z-score OR burst: {n_either:,} | by BOTH: {n_both:,} "
      f"({n_both/n_either*100:.0f}% of flagged days trip both methods)")

# ----------------------------------------------------------------------------
# June 2025 validation-day report, RE-RUN AGAINST THE 4-YEAR BASELINE
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("JUNE 2025 VALIDATION DAYS - re-checked against 4-YEAR (2022-2025) baseline")
print("(original finding was built against the 2024-25-only baseline)")
print("=" * 78)

orig_summary_path = "data/processed/anomaly/blockgroup_anomaly_summary.csv"
recheck_rows = []
for d in VALIDATION_DAYS:
    day = combined[combined["date"] == d]
    day_active = day[day["flash_count"] > 0]
    total_flashes = int(day["flash_count"].sum())
    n_z = int(day["zscore_anomaly"].sum())
    n_burst = int(day["burst_alert"].sum())
    n_both_day = int(day["both_flag"].sum())
    max_z = day["zscore"].max()
    max_peak = int(day["peak_30min"].max())
    print(f"\n{d}: {total_flashes:,} flashes across {len(day_active)} block groups")
    print(f"  z-score anomalies (z>={Z_THRESHOLD}): {n_z} block groups   (max z = {max_z:.1f})")
    print(f"  burst alerts (>=5/30min):        {n_burst} block groups   (max 30-min peak = {max_peak})")
    print(f"  flagged by BOTH methods:         {n_both_day} block groups")
    recheck_rows.append((d, total_flashes, n_z, n_burst, n_both_day, max_z, max_peak))

recheck = pd.DataFrame(recheck_rows, columns=["date", "total_flashes", "n_z", "n_burst",
                                              "n_both", "max_z", "max_peak30"])
recheck.to_csv(os.path.join(OUT_DIR, "june2025_validation_recheck_4yr.csv"), index=False)

val = combined[combined["date"].isin(VALIDATION_DAYS)]
val_active = val[val["flash_count"] > 0]
z_set = set(val.loc[val["zscore_anomaly"], ["GEOID", "date"]].itertuples(index=False, name=None))
b_set = set(val.loc[val["burst_alert"], ["GEOID", "date"]].itertuples(index=False, name=None))
print("\n" + "=" * 78)
print("AGREEMENT ON VALIDATION DAYS - 4-YEAR BASELINE")
print("=" * 78)
print(f"active block-group-days on the 5 validation days: {len(val_active)}")
print(f"flagged by z-score:        {len(z_set)}")
print(f"flagged by burst:          {len(b_set)}")
print(f"flagged by both:           {len(z_set & b_set)}")
overlap = len(z_set & b_set) / len(z_set | b_set) * 100 if (z_set | b_set) else 0
print(f"Jaccard overlap of the two flag sets: {overlap:.0f}%")
print(f"ALL 5 DAYS STILL TRIP >=1 BOTH-METHOD BLOCK GROUP: "
      f"{all(n > 0 for n in recheck['n_both'])}")

# Direct old-vs-new comparison, if the original 2-year summary exists
if os.path.exists(orig_summary_path):
    print("\n" + "=" * 78)
    print("2-YEAR BASELINE (original) vs 4-YEAR BASELINE (this run) - z-score sensitivity")
    print("=" * 78)
    old = pd.read_csv(orig_summary_path, dtype={"GEOID": str})
    cmp = summary.merge(old, on="GEOID", suffixes=("_4yr", "_2yr"))
    print(f"Correlation of n_days_both_flag, 2yr vs 4yr baseline (all-history counts, not directly "
          f"comparable in scale but check rank stability): "
          f"{cmp['n_days_both_flag_4yr'].corr(cmp['n_days_both_flag_2yr']):.3f}")
