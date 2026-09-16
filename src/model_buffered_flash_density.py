"""Ablation: does the best model (Random Forest, expanded feature set) improve
when its flash-density feature is measured from the 10 km coastal-buffer join
instead of the land-only join?

ONLY `flash_per_km2` changes. Everything else is held identical to
model_ceiling_push.py's "expanded" run: same 13 other features, same
block-group x month target (land-only anomaly labels), same nested
leave-one-county-out spatial CV, same RF search space.

  unbuffered flash_per_km2 = land-only flash count / land area   (reproduces 0.812)
  buffered   flash_per_km2 = (land + <=10 km offshore) count / land area
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42

EXTRA = pd.read_csv("data/processed/features/blockgroup_features_extra.csv", dtype={"GEOID": str})
NLCD = pd.read_csv("data/processed/features/blockgroup_nlcd.csv", dtype={"GEOID": str})
FLAGS = pd.read_csv("data/processed/anomaly/flagged_blockgroup_days.csv", dtype={"GEOID": str})
BUFCMP = pd.read_csv("data/processed/coastal_buffer_10km/flash_count_comparison.csv", dtype={"GEOID": str})

ONI = {(2024, 5): 0.43, (2024, 6): 0.18, (2024, 7): 0.06, (2024, 8): -0.04, (2024, 9): -0.12,
       (2025, 5): -0.04, (2025, 6): -0.02, (2025, 7): -0.11, (2025, 8): -0.26, (2025, 9): -0.43}
MID_DOY = {5: 135, 6: 166, 7: 196, 8: 227, 9: 258}

# ---- buffered flash density per block group ----
BUFCMP["flash_per_km2_buffered"] = BUFCMP["flash_buffered"] / (BUFCMP["ALAND"] / 1e6).clip(lower=0.01)
BUFCMP["flash_per_km2_land"] = BUFCMP["flash_orig"] / (BUFCMP["ALAND"] / 1e6).clip(lower=0.01)
# sanity: land recompute should match EXTRA's flash_per_km2
chk = EXTRA.merge(BUFCMP[["GEOID", "flash_per_km2_land"]], on="GEOID")
print(f"land flash_per_km2 recompute max abs diff vs features_extra: "
      f"{(chk['flash_per_km2'] - chk['flash_per_km2_land']).abs().max():.4f}")

# ---- block-group x month table ----
FLAGS["date"] = pd.to_datetime(FLAGS["date"])
FLAGS["year"], FLAGS["month"] = FLAGS["date"].dt.year, FLAGS["date"].dt.month
pos = (FLAGS[FLAGS["both_flag"]].groupby(["GEOID", "year", "month"]).size().rename("n").reset_index())
grid = pd.MultiIndex.from_product(
    [EXTRA["GEOID"], [(2024, m) for m in range(5, 10)] + [(2025, m) for m in range(5, 10)]],
    names=["GEOID", "ym"]).to_frame(index=False)
grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
grid = grid.drop(columns="ym").merge(pos, on=["GEOID", "year", "month"], how="left")
grid["target"] = (grid["n"].fillna(0) > 0).astype(int)

df = (grid
      .merge(EXTRA[["GEOID", "COUNTY_NAME", "P1_001N", "drive_min_to_shelter", "coast_dist_km"]], on="GEOID", how="left")
      .merge(NLCD[["GEOID", "pct_developed", "pct_forest", "pct_wetland"]], on="GEOID", how="left")
      .merge(BUFCMP[["GEOID", "flash_per_km2_land", "flash_per_km2_buffered"]], on="GEOID", how="left"))

lut = grid.set_index(["GEOID", "year", "month"])["target"].to_dict()
df["prev_month_flagged"] = [lut.get((g, y, m - 1), 0) if m > 5 else 0
                            for g, y, m in zip(df["GEOID"], df["year"], df["month"])]
df["same_month_last_yr_flagged"] = [lut.get((g, y - 1, m), 0) if y == 2025 else 0
                                    for g, y, m in zip(df["GEOID"], df["year"], df["month"])]
df["is_may"] = (df["month"] == 5).astype(int)
df["is_2025"] = (df["year"] == 2025).astype(int)
df["oni"] = [ONI[(y, m)] for y, m in zip(df["year"], df["month"])]
doy = df["month"].map(MID_DOY).astype(float)
df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

OTHER = ["P1_001N", "drive_min_to_shelter", "coast_dist_km", "pct_developed", "pct_forest",
         "pct_wetland", "prev_month_flagged", "same_month_last_yr_flagged", "is_may",
         "is_2025", "oni", "doy_sin", "doy_cos"]

RF_SPACE = {"n_estimators": [150, 250, 350, 500], "max_depth": [3, 4, 6, 8, 12, None],
            "min_samples_split": [2, 5, 10, 20, 40], "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.5, 1.0]}


def tuned_rf(X, y, g):
    s = RandomizedSearchCV(RandomForestClassifier(random_state=RNG, n_jobs=-1), RF_SPACE,
                           n_iter=25, scoring="roc_auc", cv=GroupKFold(4),
                           random_state=RNG, n_jobs=-1)
    s.fit(X, y, groups=g)
    return s.best_estimator_


def nested_loco(feat_cols, label):
    counties = sorted(df["COUNTY_NAME"].unique())
    oof_y, oof_lr, oof_rf, pf_lr, pf_rf = [], [], [], [], []
    for c in counties:
        tr, te = df["COUNTY_NAME"] != c, df["COUNTY_NAME"] == c
        ytr, yte = df.loc[tr, "target"].to_numpy(), df.loc[te, "target"].to_numpy()
        g = df.loc[tr, "COUNTY_NAME"].to_numpy()
        sc = StandardScaler().fit(df.loc[tr, feat_cols])
        Xtr, Xte = sc.transform(df.loc[tr, feat_cols]), sc.transform(df.loc[te, feat_cols])
        lr = LogisticRegression(max_iter=4000).fit(Xtr, ytr)
        rf = tuned_rf(Xtr, ytr, g)
        plr, prf = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
        oof_y += list(yte); oof_lr += list(plr); oof_rf += list(prf)
        pf_lr.append(roc_auc_score(yte, plr)); pf_rf.append(roc_auc_score(yte, prf))
    y = np.array(oof_y)
    out = {}
    for name, p, pf in (("LogReg", oof_lr, pf_lr), ("RandomForest", oof_rf, pf_rf)):
        p = np.array(p)
        out[name] = dict(
            pooled_AUC=round(roc_auc_score(y, p), 3),
            fold_mean=round(float(np.mean(pf)), 3), fold_std=round(float(np.std(pf)), 3),
            prec=round(precision_score(y, p >= 0.5, zero_division=0), 3),
            rec=round(recall_score(y, p >= 0.5, zero_division=0), 3),
        )
    print(f"\n[{label}]  {out}")
    return out


unbuf = nested_loco(["flash_per_km2_land"] + OTHER, "UNBUFFERED (land-only flash density)")
buf = nested_loco(["flash_per_km2_buffered"] + OTHER, "BUFFERED (10 km coastal buffer flash density)")

print("\n" + "=" * 78)
print("DIRECT COMPARISON  -  expanded RF, only flash_per_km2 differs")
print("=" * 78)
row = "{:<34}{:>10}{:>16}{:>10}{:>9}"
print(row.format("feature source", "AUC", "fold mean±std", "prec@.5", "rec@.5"))
for lbl, d in [("existing CV reference", None),
               ("unbuffered (land-only)", unbuf["RandomForest"]),
               ("buffered (10 km coastal)", buf["RandomForest"])]:
    if d is None:
        print(row.format(lbl, "0.812", "—", "—", "—"))
    else:
        print(row.format(lbl, f"{d['pooled_AUC']:.3f}",
                         f"{d['fold_mean']:.3f}±{d['fold_std']:.3f}",
                         f"{d['prec']:.3f}", f"{d['rec']:.3f}"))

d_auc = buf["RandomForest"]["pooled_AUC"] - unbuf["RandomForest"]["pooled_AUC"]
noise = max(unbuf["RandomForest"]["fold_std"], buf["RandomForest"]["fold_std"])
verdict = ("IMPROVES" if d_auc > noise else "WORSE" if d_auc < -noise else "NO CHANGE (within fold noise)")
print(f"\nbuffered - unbuffered AUC: {d_auc:+.3f}   (fold noise ~{noise:.3f})  ->  {verdict}")
print("LogReg for reference: "
      f"unbuffered {unbuf['LogReg']['pooled_AUC']:.3f}  buffered {buf['LogReg']['pooled_AUC']:.3f}")
