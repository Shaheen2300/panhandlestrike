"""ML layer, expanded: hyperparameter tuning, decision-threshold tuning, and
genuinely-new features, evaluated honestly against the earlier ~0.85 AUC.

Two target framings are run:

  A. Block-group level (589 rows, target has_confirmed_anomaly, ~84% positive) -
     same target as model_anomaly.py, so AUC is directly comparable to the
     earlier 0.87 (logreg) / 0.82 (RF). Feature sets compared:
        base3     = population, flash_per_km2, drive_min_to_shelter
        expanded5 = base3 + spatial_lag_flash_density + coast_dist_km
     Month/season indicators do NOT apply here - the target is a single
     binary over both whole seasons, so there is no month dimension.

  B. Block-group x month (589 x 10 = 5,890 rows, target = >=1 both-method
     anomaly day in that calendar month, ~29% positive). This reframe is
     what makes month/season indicators meaningful. Feature sets compared:
        static    = population, flash_per_km2, drive_min_to_shelter,
                    spatial_lag_flash_density, coast_dist_km
        +time     = static + month one-hot + year

Validation: nested spatial CV. Outer = leave-one-county-out (5 folds).
Inner = RandomizedSearchCV over RF hyperparameters with GroupKFold(4) on the
training counties, scoring roc_auc. The held-out county is never seen during
tuning, so pooled out-of-fold metrics are honest.

Decision threshold: swept on the pooled out-of-fold probabilities, best by F1;
confusion matrices reported at 0.5 and at the tuned threshold.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42

EXTRA = pd.read_csv("data/processed/features/blockgroup_features_extra.csv", dtype={"GEOID": str})
FLAGS = pd.read_csv("data/processed/anomaly/flagged_blockgroup_days.csv", dtype={"GEOID": str})

RF_SEARCH_SPACE = {
    "n_estimators": [150, 250, 350, 500],
    "max_depth": [3, 4, 6, 8, 12, None],
    "min_samples_split": [2, 5, 10, 20, 40],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5, 1.0],
}


def tune_rf(X, y, groups):
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RNG, n_jobs=-1),
        RF_SEARCH_SPACE, n_iter=25, scoring="roc_auc",
        cv=GroupKFold(n_splits=4), random_state=RNG, n_jobs=-1,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_


def nested_loco(df, feat_cols, target_col, label):
    counties = sorted(df["COUNTY_NAME"].unique())
    oof = {k: [] for k in ["y", "lr_p", "rf_p"]}
    best_params_seen = []
    for c in counties:
        tr, te = df["COUNTY_NAME"] != c, df["COUNTY_NAME"] == c
        sc = StandardScaler().fit(df.loc[tr, feat_cols])
        Xtr, Xte = sc.transform(df.loc[tr, feat_cols]), sc.transform(df.loc[te, feat_cols])
        ytr, yte = df.loc[tr, target_col].to_numpy(int), df.loc[te, target_col].to_numpy(int)
        g = df.loc[tr, "COUNTY_NAME"].to_numpy()

        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        rf, bp = tune_rf(Xtr, ytr, g)
        best_params_seen.append(bp)

        oof["y"] += list(yte)
        oof["lr_p"] += list(lr.predict_proba(Xte)[:, 1])
        oof["rf_p"] += list(rf.predict_proba(Xte)[:, 1])

    y = np.array(oof["y"])
    lr_auc = roc_auc_score(y, oof["lr_p"])
    rf_auc = roc_auc_score(y, oof["rf_p"])
    print(f"\n[{label}]  n={len(df)}  positive rate={y.mean():.3f}")
    print(f"  logreg  AUC {lr_auc:.3f}")
    print(f"  RF      AUC {rf_auc:.3f}   (example tuned params: {best_params_seen[-1]})")
    return y, np.array(oof["lr_p"]), np.array(oof["rf_p"]), lr_auc, rf_auc


def threshold_report(y, proba, model_label):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"\n  {model_label}: best F1 threshold = {best_t:.2f} (F1 {best_f1:.3f})")
    for t, tag in [(0.5, "default 0.50"), (best_t, f"tuned {best_t:.2f}")]:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y, pred)
        print(f"    @ {tag}: acc {accuracy_score(y, pred):.3f}  "
              f"prec {precision_score(y, pred, zero_division=0):.3f}  "
              f"rec {recall_score(y, pred, zero_division=0):.3f}")
        print(f"      confusion [[TN FP][FN TP]] = {cm.tolist()}")


def logit_coefs(df, feat_cols, target_col, label):
    Xz = StandardScaler().fit_transform(df[feat_cols].to_numpy(float))
    res = sm.Logit(df[target_col].to_numpy(int), sm.add_constant(Xz)).fit(disp=0)
    tbl = pd.DataFrame({
        "term": ["intercept"] + feat_cols,
        "coef_std": res.params, "p_value": res.pvalues,
        "odds_ratio_per_SD": np.exp(res.params),
    })
    print(f"\n[{label}] logistic regression coefficients:")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  McFadden pseudo-R^2 {res.prsquared:.3f}")


# ===========================================================================
# FRAMING A - block-group level
# ===========================================================================
print("=" * 80)
print("FRAMING A - block-group level (target: has_confirmed_anomaly)")
print("=" * 80)

A = EXTRA.copy()
A["has_confirmed_anomaly"] = A["has_confirmed_anomaly"].astype(int)
base3 = ["P1_001N", "flash_per_km2", "drive_min_to_shelter"]
expanded5 = base3 + ["spatial_lag_flash_density", "coast_dist_km"]

logit_coefs(A, base3, "has_confirmed_anomaly", "A / base3")
logit_coefs(A, expanded5, "has_confirmed_anomaly", "A / expanded5")

yA3, lrA3, rfA3, lrA3_auc, rfA3_auc = nested_loco(A, base3, "has_confirmed_anomaly", "A / base3")
yA5, lrA5, rfA5, lrA5_auc, rfA5_auc = nested_loco(A, expanded5, "has_confirmed_anomaly", "A / expanded5")

print("\n--- FRAMING A: threshold tuning on the expanded5 models ---")
threshold_report(yA5, lrA5, "logreg expanded5")
threshold_report(yA5, rfA5, "RF expanded5")

print("\n--- FRAMING A: AUC change from adding spatial-lag + coast ---")
print(f"  logreg: {lrA3_auc:.3f} -> {lrA5_auc:.3f}   ({lrA5_auc - lrA3_auc:+.3f})")
print(f"  RF    : {rfA3_auc:.3f} -> {rfA5_auc:.3f}   ({rfA5_auc - rfA3_auc:+.3f})")

# ===========================================================================
# FRAMING B - block-group x month
# ===========================================================================
print("\n" + "=" * 80)
print("FRAMING B - block-group x month (target: >=1 both-method day that month)")
print("=" * 80)

FLAGS["date"] = pd.to_datetime(FLAGS["date"])
FLAGS["year"] = FLAGS["date"].dt.year
FLAGS["month"] = FLAGS["date"].dt.month
bm_pos = (FLAGS[FLAGS["both_flag"]].groupby(["GEOID", "year", "month"]).size()
          .rename("n_both_days").reset_index())

months = [(2024, m) for m in range(5, 10)] + [(2025, m) for m in range(5, 10)]
grid = pd.MultiIndex.from_product(
    [EXTRA["GEOID"], months], names=["GEOID", "ym"]
).to_frame(index=False)
grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
grid = grid.drop(columns="ym").merge(bm_pos, on=["GEOID", "year", "month"], how="left")
grid["target"] = (grid["n_both_days"].fillna(0) > 0).astype(int)

B = grid.merge(
    EXTRA[["GEOID", "COUNTY_NAME", "P1_001N", "flash_per_km2", "drive_min_to_shelter",
           "spatial_lag_flash_density", "coast_dist_km"]],
    on="GEOID", how="left",
)
static = ["P1_001N", "flash_per_km2", "drive_min_to_shelter",
          "spatial_lag_flash_density", "coast_dist_km"]
for mth in range(6, 10):
    B[f"month_{mth}"] = (B["month"] == mth).astype(int)  # May is the reference
B["year_2025"] = (B["year"] == 2025).astype(int)
time_cols = [f"month_{m}" for m in range(6, 10)] + ["year_2025"]

logit_coefs(B, static + time_cols, "target", "B / static+time")

yBs, lrBs, rfBs, lrBs_auc, rfBs_auc = nested_loco(B, static, "target", "B / static only")
yBt, lrBt, rfBt, lrBt_auc, rfBt_auc = nested_loco(B, static + time_cols, "target", "B / static + month/year")

print("\n--- FRAMING B: threshold tuning on the static+time models ---")
threshold_report(yBt, lrBt, "logreg static+time")
threshold_report(yBt, rfBt, "RF static+time")

print("\n--- FRAMING B: AUC change from adding month/year indicators ---")
print(f"  logreg: {lrBs_auc:.3f} -> {lrBt_auc:.3f}   ({lrBt_auc - lrBs_auc:+.3f})")
print(f"  RF    : {rfBs_auc:.3f} -> {rfBt_auc:.3f}   ({rfBt_auc - rfBs_auc:+.3f})")

# ===========================================================================
print("\n" + "=" * 80)
print("SUMMARY vs the earlier ~0.85 AUC (model_anomaly.py, base3, block-group level)")
print("=" * 80)
print(f"  earlier logreg base3 (non-nested LOCO):           AUC 0.866")
print(f"  A / base3     nested LOCO   logreg {lrA3_auc:.3f}  RF {rfA3_auc:.3f}")
print(f"  A / expanded5 nested LOCO   logreg {lrA5_auc:.3f}  RF {rfA5_auc:.3f}")
print(f"  B / static    nested LOCO   logreg {lrBs_auc:.3f}  RF {rfBs_auc:.3f}")
print(f"  B / static+time nested LOCO logreg {lrBt_auc:.3f}  RF {rfBt_auc:.3f}")
print("\n(interpretation left for the written report - see running-notes.md)")
