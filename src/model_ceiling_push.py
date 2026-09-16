"""Attempt to push past the ~0.77 AUC ceiling on the block-group x month target
with NEW INFORMATION rather than more tuning:

  - prev_month_flagged        : was this same block group flagged last month
                                (0 for the May rows - no in-season predecessor;
                                is_may flag added so the model can special-case)
  - same_month_last_yr_flagged: was this block group flagged in the same month
                                of the other season (0 for 2024 rows)
  - oni                        : NOAA Oceanic Nino Index for the 3-month season
                                centred on that month (ENSO climate state)
  - pct_developed / pct_forest / pct_wetland : NLCD 2021 land-cover shares
  - doy_sin / doy_cos          : cyclical day-of-year (mid-month), replacing the
                                blocky month dummies

Comparison (nested leave-one-county-out spatial CV, LogReg + tuned RF):
  baseline  = static4 + month dummies         (reproduces the current ~0.77)
  expanded  = static4 + NLCD + lags + ONI + cyclical-DOY   (no month dummies)
  exp+mon   = expanded + month dummies

Honest question: does any of this move AUC beyond fold-to-fold noise (~0.02)?
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

# ONI (centred-season) for each study month, from NOAA CPC oni.ascii.txt
ONI = {
    (2024, 5): 0.43, (2024, 6): 0.18, (2024, 7): 0.06, (2024, 8): -0.04, (2024, 9): -0.12,
    (2025, 5): -0.04, (2025, 6): -0.02, (2025, 7): -0.11, (2025, 8): -0.26, (2025, 9): -0.43,
}
MID_DOY = {5: 135, 6: 166, 7: 196, 8: 227, 9: 258}  # ~15th of each month

STATIC4 = ["P1_001N", "flash_per_km2", "drive_min_to_shelter", "coast_dist_km"]
MONTHD = [f"month_{m}" for m in (6, 7, 8, 9)]
NEW = ["pct_developed", "pct_forest", "pct_wetland", "prev_month_flagged",
       "same_month_last_yr_flagged", "is_may", "is_2025", "oni", "doy_sin", "doy_cos"]

RF_SPACE = {
    "n_estimators": [150, 250, 350, 500], "max_depth": [3, 4, 6, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20, 40], "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}


def build_df():
    """Block-group x month (2024-25) table with the full expanded feature set.
    Reads directly from the saved 2-year artifact files every call - no
    caching - so a rerun always reflects whatever is currently on disk."""
    flags = FLAGS.copy()
    flags["date"] = pd.to_datetime(flags["date"])
    flags["year"], flags["month"] = flags["date"].dt.year, flags["date"].dt.month
    pos = (flags[flags["both_flag"]].groupby(["GEOID", "year", "month"]).size()
           .rename("n").reset_index())
    grid = pd.MultiIndex.from_product(
        [EXTRA["GEOID"], [(2024, m) for m in range(5, 10)] + [(2025, m) for m in range(5, 10)]],
        names=["GEOID", "ym"],
    ).to_frame(index=False)
    grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
    grid = grid.drop(columns="ym").merge(pos, on=["GEOID", "year", "month"], how="left")
    grid["target"] = (grid["n"].fillna(0) > 0).astype(int)

    df = (grid
          .merge(EXTRA[["GEOID", "COUNTY_NAME", "P1_001N", "flash_per_km2",
                        "drive_min_to_shelter", "coast_dist_km"]], on="GEOID", how="left")
          .merge(NLCD[["GEOID", "pct_developed", "pct_forest", "pct_wetland"]], on="GEOID", how="left"))

    tgt_lookup = df.set_index(["GEOID", "year", "month"])["target"].to_dict()
    df["prev_month_flagged"] = [
        tgt_lookup.get((g, y, m - 1), 0) if m > 5 else 0
        for g, y, m in zip(df["GEOID"], df["year"], df["month"])
    ]
    df["same_month_last_yr_flagged"] = [
        tgt_lookup.get((g, y - 1, m), 0) if y == 2025 else 0
        for g, y, m in zip(df["GEOID"], df["year"], df["month"])
    ]
    df["is_may"] = (df["month"] == 5).astype(int)
    df["is_2025"] = (df["year"] == 2025).astype(int)
    df["oni"] = [ONI[(y, m)] for y, m in zip(df["year"], df["month"])]
    doy = df["month"].map(MID_DOY).astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for m in (6, 7, 8, 9):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    return df


def feature_sets():
    return {
        "baseline (static4 + month dummies)": STATIC4 + MONTHD,
        "expanded (static4 + NLCD + lags + ONI + cyclical DOY)": STATIC4 + NEW,
        "exp + month dummies": STATIC4 + NEW + MONTHD,
    }


def tuned_rf(X, y, g):
    s = RandomizedSearchCV(RandomForestClassifier(random_state=RNG, n_jobs=-1), RF_SPACE,
                           n_iter=25, scoring="roc_auc", cv=GroupKFold(4),
                           random_state=RNG, n_jobs=-1)
    s.fit(X, y, groups=g)
    return s.best_estimator_


def nested_cv(df, feats):
    """Nested leave-one-county-out CV for one feature set. Returns
    {"LogReg": {...}, "RandomForest": {...}} with pooled_AUC/fold_mean/
    fold_std/prec/rec - the live, re-derived numbers, not cached results."""
    counties = sorted(df["COUNTY_NAME"].unique())
    oof = {"y": [], "lr": [], "rf": []}
    pf = {"lr": [], "rf": []}
    for c in counties:
        tr, te = df["COUNTY_NAME"] != c, df["COUNTY_NAME"] == c
        ytr, yte = df.loc[tr, "target"].to_numpy(), df.loc[te, "target"].to_numpy()
        g = df.loc[tr, "COUNTY_NAME"].to_numpy()
        sc = StandardScaler().fit(df.loc[tr, feats])
        Xtr, Xte = sc.transform(df.loc[tr, feats]), sc.transform(df.loc[te, feats])
        lr = LogisticRegression(max_iter=4000).fit(Xtr, ytr)
        rf = tuned_rf(Xtr, ytr, g)
        for k, mdl in (("lr", lr), ("rf", rf)):
            p = mdl.predict_proba(Xte)[:, 1]
            oof[k] += list(p)
        oof["y"] += list(yte)
        pf["lr"].append(roc_auc_score(yte, oof["lr"][-len(yte):]))
        pf["rf"].append(roc_auc_score(yte, oof["rf"][-len(yte):]))
    y = np.array(oof["y"])
    out = {}
    for k, name in (("lr", "LogReg"), ("rf", "RandomForest")):
        p = np.array(oof[k])
        out[name] = {
            "pooled_AUC": roc_auc_score(y, p),
            "fold_mean": np.mean(pf[k]), "fold_std": np.std(pf[k]),
            "prec@0.5": precision_score(y, p >= 0.5, zero_division=0),
            "rec@0.5": recall_score(y, p >= 0.5, zero_division=0),
        }
    return out


# ===========================================================================
if __name__ == "__main__":
    df = build_df()
    FEATURE_SETS = feature_sets()
    print(f"rows {len(df)}  positive rate {df['target'].mean():.3f}\n")
    results = []
    for label, feats in FEATURE_SETS.items():
        r = nested_cv(df, feats)
        for name, m in r.items():
            results.append({"feature_set": label, "model": name,
                            "pooled_AUC": round(m["pooled_AUC"], 3),
                            "fold_mean": round(m["fold_mean"], 3), "fold_std": round(m["fold_std"], 3),
                            "prec@0.5": round(m["prec@0.5"], 3), "rec@0.5": round(m["rec@0.5"], 3)})

    res = pd.DataFrame(results)
    print("=" * 96)
    print("CEILING-PUSH COMPARISON  -  block-group x month, nested LOCO spatial CV")
    print("=" * 96)
    print(res.to_string(index=False))

    base = res[res["feature_set"].str.startswith("baseline")].set_index("model")["pooled_AUC"]
    print("\nAUC change vs baseline feature set:")
    for _, r in res.iterrows():
        if r["feature_set"].startswith("baseline"):
            continue
        d = r["pooled_AUC"] - base[r["model"]]
        verdict = "within noise" if abs(d) <= r["fold_std"] else ("IMPROVES" if d > 0 else "worse")
        print(f"  {r['model']:<13} {r['feature_set'][:40]:<42} {r['pooled_AUC']:.3f}  ({d:+.3f})  -> {verdict}")

    # coefficient view for the expanded LogReg (interpretability)
    import statsmodels.api as sm
    exp_feats = FEATURE_SETS["expanded (static4 + NLCD + lags + ONI + cyclical DOY)"]
    Xz = StandardScaler().fit_transform(df[exp_feats].to_numpy(float))
    li = sm.Logit(df["target"].to_numpy(), sm.add_constant(Xz)).fit(disp=0)
    ct = pd.DataFrame({"term": ["const"] + exp_feats, "coef_std": li.params, "p": li.pvalues,
                       "odds_ratio_per_SD": np.exp(li.params)})
    print("\nExpanded LogReg coefficients (full data, standardized):")
    print(ct.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"McFadden pseudo-R^2 {li.prsquared:.3f}  (baseline expanded run was 0.183)")
