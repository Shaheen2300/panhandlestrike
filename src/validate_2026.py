"""Out-of-time validation: train the expanded model on ALL of 2024-25, test on
2026 (May-Aug), which the model saw in no form during training or CV.

Pipeline for 2026 (mirrors the training pipeline, with two deliberate choices
that keep the test genuinely forward-looking):
  - flash_per_km2 / spatial-lag style baseline features stay at their
    2024-25 values - the model's "prior knowledge" of each block group.
  - the z-score anomaly baseline (per-BG daily-count mean/std) is the
    2024-25 baseline applied to 2026 daily counts, i.e. what a
    history-calibrated detector would flag in 2026. The burst rule is
    absolute and unchanged.

Known limitations, reported honestly:
  - the model carries an `is_2025` dummy; 2026 rows get is_2025 = 0, so the
    model treats 2026 like 2024 for that term.
  - 2026 ONI (AMJ..JAS) is +0.95..+1.80 - a strong El Nino, entirely outside
    the 2024-25 training range (-0.43..+0.43). The model's large ONI
    coefficient is being extrapolated well past its support.
  - JAS-2026 ONI is not yet published; August uses the JJA value (1.80).
"""

import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

BG_PATH = "data/processed/census/panhandle_block_groups_population.geojson"
EXTRA = pd.read_csv("data/processed/features/blockgroup_features_extra.csv", dtype={"GEOID": str})
NLCD = pd.read_csv("data/processed/features/blockgroup_nlcd.csv", dtype={"GEOID": str})
TRAIN_FLASH = "data/processed/features/flashes_with_blockgroup.parquet"
FLAGS_TRAIN = pd.read_csv("data/processed/anomaly/flagged_blockgroup_days.csv", dtype={"GEOID": str})

BBOX = (29.5, 31.5, -88.0, -84.5)
Z_THRESHOLD = 3.0
BURST_WINDOW = pd.Timedelta(minutes=30)
BURST_MIN = 5
ONI_2026 = {5: 0.95, 6: 1.39, 7: 1.80, 8: 1.80}  # AMJ, MJJ, JJA, JAS(=JJA, not yet published)
MID_DOY = {5: 135, 6: 166, 7: 196, 8: 227, 9: 258}

WATER = set(gpd.read_file(BG_PATH).pipe(
    lambda g: g.loc[g["TRACTCE"].str.startswith("99") | (g["P1_001N"] == 0), "GEOID"]))


# ---------------------------------------------------------------------------
def spatial_join_2026():
    files = sorted(glob.glob("data/processed/glm/glm_panhandle_2026_*.parquet"))
    if not files:
        raise SystemExit("no 2026 GLM parquet files yet - run src/pull_glm_2026.py first")
    fl = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fl["timestamp"] = pd.to_datetime(fl["timestamp"])
    gdf = gpd.GeoDataFrame(fl, geometry=gpd.points_from_xy(fl["flash_lon"], fl["flash_lat"]),
                           crs="EPSG:4326")
    bg = gpd.read_file(BG_PATH).to_crs("EPSG:4326")[["GEOID", "COUNTY_NAME", "geometry"]]
    j = gpd.sjoin(gdf, bg, how="inner", predicate="within")
    j = j[~j["GEOID"].isin(WATER)]
    return j[["flash_id", "timestamp", "GEOID", "COUNTY_NAME"]].reset_index(drop=True)


def burst_peaks(times):
    t = np.sort(times); n = len(t); j = 0; peak = {}
    for i in range(n):
        while t[i] - t[j] >= BURST_WINDOW:
            j += 1
        d = pd.Timestamp(t[i]).date()
        peak[d] = max(peak.get(d, 0), i - j + 1)
    return peak


def anomaly_flags_2026(j2026):
    # 2024-25 z-score baseline per BG
    tr = pd.read_parquet(TRAIN_FLASH)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"])
    tr = tr[~tr["GEOID"].isin(WATER)]
    tr["date"] = tr["timestamp"].dt.date
    tr_daily = tr.groupby(["GEOID", "date"]).size().rename("c").reset_index()
    all_days = pd.date_range(tr["timestamp"].min().normalize(), tr["timestamp"].max().normalize(), freq="D").date
    full = (tr_daily.set_index(["GEOID", "date"])
            .reindex(pd.MultiIndex.from_product([EXTRA["GEOID"], all_days], names=["GEOID", "date"]), fill_value=0)
            .reset_index())
    base = full.groupby("GEOID")["c"].agg(mu="mean", sd="std").reset_index()

    j2026 = j2026.copy()
    j2026["date"] = j2026["timestamp"].dt.date
    daily = j2026.groupby(["GEOID", "date"]).size().rename("c").reset_index().merge(base, on="GEOID", how="left")
    daily["z"] = np.where(daily["sd"] > 0, (daily["c"] - daily["mu"]) / daily["sd"], np.nan)
    daily["z_flag"] = daily["z"] >= Z_THRESHOLD

    peak_rows = []
    for g, grp in j2026.groupby("GEOID"):
        for d, pk in burst_peaks(grp["timestamp"].values).items():
            peak_rows.append((g, d, pk))
    peaks = pd.DataFrame(peak_rows, columns=["GEOID", "date", "peak_30min"])
    peaks["burst_flag"] = peaks["peak_30min"] >= BURST_MIN

    m = daily.merge(peaks, on=["GEOID", "date"], how="outer")
    m["z_flag"] = m["z_flag"].fillna(False)
    m["burst_flag"] = m["burst_flag"].fillna(False)
    m["both"] = m["z_flag"] & m["burst_flag"]
    m["date"] = pd.to_datetime(m["date"])
    m["month"] = m["date"].dt.month
    bm = m.groupby(["GEOID", "month"])["both"].any().astype(int).rename("target").reset_index()
    return bm


def build_features(df, year, prev_flag_lookup, last_yr_lookup):
    df = df.merge(EXTRA[["GEOID", "COUNTY_NAME", "P1_001N", "flash_per_km2",
                         "drive_min_to_shelter", "coast_dist_km"]], on="GEOID", how="left")
    df = df.merge(NLCD[["GEOID", "pct_developed", "pct_forest", "pct_wetland"]], on="GEOID", how="left")
    df["prev_month_flagged"] = [prev_flag_lookup.get((g, m - 1), 0) if m > 5 else 0
                                for g, m in zip(df["GEOID"], df["month"])]
    df["same_month_last_yr_flagged"] = [last_yr_lookup.get((g, m), 0)
                                        for g, m in zip(df["GEOID"], df["month"])]
    df["is_may"] = (df["month"] == 5).astype(int)
    df["is_2025"] = 1 if year == 2025 else 0
    df["oni"] = df["month"].map(ONI_2026 if year == 2026 else ONI_TRAIN[year])
    doy = df["month"].map(MID_DOY).astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


FEATS = ["P1_001N", "flash_per_km2", "drive_min_to_shelter", "coast_dist_km",
         "pct_developed", "pct_forest", "pct_wetland", "prev_month_flagged",
         "same_month_last_yr_flagged", "is_may", "is_2025", "oni", "doy_sin", "doy_cos"]

ONI_TRAIN = {
    2024: {5: 0.43, 6: 0.18, 7: 0.06, 8: -0.04, 9: -0.12},
    2025: {5: -0.04, 6: -0.02, 7: -0.11, 8: -0.26, 9: -0.43},
}


def evaluate(train_df, test_df, feats=None):
    """Fit LogReg + RF on train_df, evaluate on test_df. Returns
    {"LogReg": {...}, "RandomForest": {...}} with auc/proba/prec/rec - live,
    re-derived from whatever is currently on disk, not a cached number.
    Also used by src/generate_report_figures.py for the 2-year side of the
    2yr-vs-4yr comparison figures (16/18/19)."""
    feats = feats or FEATS
    sc = StandardScaler().fit(train_df[feats])
    Xtr, Xte = sc.transform(train_df[feats]), sc.transform(test_df[feats])
    ytr, yte = train_df["target"].to_numpy(), test_df["target"].to_numpy()
    lr = LogisticRegression(max_iter=4000).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=4,
                                max_features="sqrt", random_state=42, n_jobs=-1).fit(Xtr, ytr)
    out = {}
    for name, mdl in (("LogReg", lr), ("RandomForest", rf)):
        p = mdl.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        out[name] = {
            "auc": roc_auc_score(yte, p), "proba": p,
            "prec@0.5": precision_score(yte, pred, zero_division=0),
            "rec@0.5": recall_score(yte, pred, zero_division=0),
        }
    return out


def build_training():
    FLAGS_TRAIN["date"] = pd.to_datetime(FLAGS_TRAIN["date"])
    FLAGS_TRAIN["year"] = FLAGS_TRAIN["date"].dt.year
    FLAGS_TRAIN["month"] = FLAGS_TRAIN["date"].dt.month
    pos = (FLAGS_TRAIN[FLAGS_TRAIN["both_flag"]].groupby(["GEOID", "year", "month"]).size()
           .rename("n").reset_index())
    grid = pd.MultiIndex.from_product(
        [EXTRA["GEOID"], [(2024, m) for m in range(5, 10)] + [(2025, m) for m in range(5, 10)]],
        names=["GEOID", "ym"]).to_frame(index=False)
    grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
    grid = grid.drop(columns="ym").merge(pos, on=["GEOID", "year", "month"], how="left")
    grid["target"] = (grid["n"].fillna(0) > 0).astype(int)
    grid = grid[~grid["GEOID"].isin(WATER)]

    lut = grid.set_index(["GEOID", "year", "month"])["target"].to_dict()
    frames = []
    for yr in (2024, 2025):
        sub = grid[grid["year"] == yr].copy()
        prev = {(g, m): lut.get((g, yr, m), 0) for g in EXTRA["GEOID"] for m in range(5, 10)}
        lastyr = {(g, m): lut.get((g, yr - 1, m), 0) for g in EXTRA["GEOID"] for m in range(5, 10)}
        frames.append(build_features(sub, yr, prev, lastyr))
    return pd.concat(frames, ignore_index=True), lut


# ===========================================================================
if __name__ == "__main__":
    print("building 2024-25 training frame...")
    train, train_lut = build_training()
    print(f"  train rows {len(train)}  positive rate {train['target'].mean():.3f}")

    print("spatial-joining 2026 flashes + deriving anomaly labels...")
    j26 = spatial_join_2026()
    print(f"  2026 flashes on land: {len(j26):,}")
    bm26 = anomaly_flags_2026(j26)
    prev26 = {(g, m): int(bm26[(bm26.GEOID == g) & (bm26.month == m)]["target"].sum() > 0)
              for g in EXTRA["GEOID"] for m in range(5, 9)}
    lastyr26 = {(g, m): train_lut.get((g, 2025, m), 0) for g in EXTRA["GEOID"] for m in range(5, 9)}
    full26 = (pd.MultiIndex.from_product([EXTRA["GEOID"], range(5, 9)], names=["GEOID", "month"])
              .to_frame(index=False).merge(bm26, on=["GEOID", "month"], how="left"))
    full26["target"] = full26["target"].fillna(0).astype(int)
    test = build_features(full26, 2026, prev26, lastyr26)
    print(f"  test rows {len(test)}  positive rate {test['target'].mean():.3f}")

    res = evaluate(train, test)

    print("\n" + "=" * 74)
    print("OUT-OF-TIME TEST  -  train 2024-25, test May-Aug 2026")
    print("=" * 74)
    print(f"{'model':<16}{'AUC':>8}{'prec@0.5':>10}{'rec@0.5':>9}")
    for name, m in res.items():
        print(f"{name:<16}{m['auc']:>8.3f}{m['prec@0.5']:>10.3f}{m['rec@0.5']:>9.3f}")
    print(f"\nCV reference (block-group x month, expanded, RF): 0.812")
    print(f"2026 positive rate {test['target'].mean():.3f}  vs  2024-25 positive rate {train['target'].mean():.3f}")
    print(f"mean predicted prob (RF) on 2026: {res['RandomForest']['proba'].mean():.3f}  "
          f"(train base rate {train['target'].mean():.3f})")
