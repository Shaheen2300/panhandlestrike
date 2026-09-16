"""ML layer: can a block group's static profile (population, baseline flash
density, drive time to nearest shelter) predict whether it ever gets flagged
as anomalous by BOTH detection methods (z-score AND rolling-30-min burst) on
at least one day?

Target reframe note: the three features are static per block group - they do
not vary by day - so a block-group-DAY target is not learnable from them
(every day of a given block group has identical features). The target is
therefore at the block-group level: has_confirmed_anomaly = at least one day
flagged by both methods.

Validation: leave-one-county-out cross-validation (5 folds). Nearby block
groups are spatially autocorrelated, so a random row split would leak
neighbours across train/test and inflate the scores. Holding out whole
counties is a genuine spatial split. Metrics are computed once on the pooled
out-of-fold predictions.

Naive baseline: always predict the majority class (True). Its accuracy equals
the positive rate; its AUC is 0.5. Both models are checked against it.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

DATA = "data/processed/final/blockgroup_combined_priority.geojson"
FEATURES = ["P1_001N", "flash_per_km2", "drive_min_to_shelter"]
TARGET = "has_confirmed_anomaly"

df = gpd.read_file(DATA)
df[TARGET] = df[TARGET].astype(bool)
y_all = df[TARGET].values
pos_rate = y_all.mean()

print(f"Rows: {len(df)}  |  target positive rate: {pos_rate:.3f} "
      f"({y_all.sum()} True / {(~y_all).sum()} False)")
print(f"Naive baseline (always predict True): accuracy = {pos_rate:.3f}, AUC = 0.500\n")
print("Per-county base rate:")
print(df.groupby("COUNTY_NAME")[TARGET].agg(n="count", n_true="sum", rate="mean").round(3), "\n")

# ---------------------------------------------------------------------------
# Logistic regression coefficients + p-values (statsmodels, full data,
# standardized features so coefficients are directly comparable)
# ---------------------------------------------------------------------------
Xz = StandardScaler().fit_transform(df[FEATURES].values)
logit = sm.Logit(y_all.astype(int), sm.add_constant(Xz)).fit(disp=0)
coef_tbl = pd.DataFrame({
    "term": ["intercept"] + FEATURES,
    "coef_std": logit.params,
    "std_err": logit.bse,
    "z": logit.tvalues,
    "p_value": logit.pvalues,
    "odds_ratio_per_SD": np.exp(logit.params),
})
print("=" * 78)
print("LOGISTIC REGRESSION  -  standardized coefficients (full data)")
print("=" * 78)
print(coef_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"pseudo R^2 (McFadden): {logit.prsquared:.4f}   LLR p-value: {logit.llr_pvalue:.4g}")

# ---------------------------------------------------------------------------
# Leave-one-county-out CV for both models
# ---------------------------------------------------------------------------
counties = sorted(df["COUNTY_NAME"].unique())
oof = {"y": [], "lr_pred": [], "lr_proba": [], "rf_pred": [], "rf_proba": [], "base_pred": []}
per_fold = []
rf_importances = []

for c in counties:
    tr = df["COUNTY_NAME"] != c
    te = df["COUNTY_NAME"] == c
    scaler = StandardScaler().fit(df.loc[tr, FEATURES])
    Xtr, Xte = scaler.transform(df.loc[tr, FEATURES]), scaler.transform(df.loc[te, FEATURES])
    ytr, yte = y_all[tr.values], y_all[te.values]

    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=400, random_state=42).fit(Xtr, ytr)
    rf_importances.append(rf.feature_importances_)

    lr_p, rf_p = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
    base_pred = np.full(yte.shape, ytr.mean() >= 0.5)  # majority class of TRAIN

    oof["y"] += list(yte)
    oof["lr_pred"] += list(lr.predict(Xte)); oof["lr_proba"] += list(lr_p)
    oof["rf_pred"] += list(rf.predict(Xte)); oof["rf_proba"] += list(rf_p)
    oof["base_pred"] += list(base_pred)

    per_fold.append({
        "held_out_county": c, "n_test": int(te.sum()), "test_pos_rate": round(yte.mean(), 3),
        "baseline_acc": round(accuracy_score(yte, base_pred), 3),
        "logreg_acc": round(accuracy_score(yte, lr.predict(Xte)), 3),
        "rf_acc": round(accuracy_score(yte, rf.predict(Xte)), 3),
        "logreg_auc": round(roc_auc_score(yte, lr_p), 3) if len(set(yte)) > 1 else np.nan,
        "rf_auc": round(roc_auc_score(yte, rf_p), 3) if len(set(yte)) > 1 else np.nan,
    })

y = np.array(oof["y"])
print("\n" + "=" * 78)
print("LEAVE-ONE-COUNTY-OUT CV  -  per fold")
print("=" * 78)
print(pd.DataFrame(per_fold).to_string(index=False))


def summarize(name, pred, proba=None):
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, pos_label=True, zero_division=0)
    rec = recall_score(y, pred, pos_label=True, zero_division=0)
    auc = roc_auc_score(y, proba) if proba is not None else float("nan")
    print(f"\n{name}")
    print(f"  accuracy  {acc:.3f}   precision(True) {prec:.3f}   recall(True) {rec:.3f}   "
          f"AUC {auc:.3f}" if proba is not None else
          f"  accuracy  {acc:.3f}   precision(True) {prec:.3f}   recall(True) {rec:.3f}")
    return acc, auc


print("\n" + "=" * 78)
print("POOLED OUT-OF-FOLD METRICS")
print("=" * 78)
base_acc, _ = summarize("Naive baseline (majority class)", np.array(oof["base_pred"]))
lr_acc, lr_auc = summarize("Logistic regression", np.array(oof["lr_pred"]), np.array(oof["lr_proba"]))
rf_acc, rf_auc = summarize("Random forest", np.array(oof["rf_pred"]), np.array(oof["rf_proba"]))

print("\nFull classification report - Logistic regression:")
print(classification_report(y, np.array(oof["lr_pred"]), zero_division=0))
print("Full classification report - Random forest:")
print(classification_report(y, np.array(oof["rf_pred"]), zero_division=0))

# ---------------------------------------------------------------------------
# Feature importance (RF impurity, averaged over folds) + permutation importance
# ---------------------------------------------------------------------------
imp = pd.Series(np.mean(rf_importances, axis=0), index=FEATURES).sort_values(ascending=False)
print("=" * 78)
print("RANDOM FOREST  -  impurity feature importance (mean over folds)")
print("=" * 78)
print(imp.to_string(float_format=lambda v: f"{v:.3f}"))

rf_full = RandomForestClassifier(n_estimators=400, random_state=42).fit(
    StandardScaler().fit_transform(df[FEATURES]), y_all)
perm = permutation_importance(rf_full, StandardScaler().fit_transform(df[FEATURES]), y_all,
                              n_repeats=30, random_state=42, scoring="roc_auc")
perm_s = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
print("\nPermutation importance (drop in AUC when feature shuffled):")
print(perm_s.to_string(float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICT vs NAIVE BASELINE")
print("=" * 78)
print(f"baseline pooled accuracy: {base_acc:.3f}")
for name, acc, auc in [("Logistic regression", lr_acc, lr_auc), ("Random forest", rf_acc, rf_auc)]:
    beats_acc = acc > base_acc + 0.02
    beats_auc = auc > 0.60
    verdict = "adds signal" if (beats_acc or beats_auc) else "DOES NOT meaningfully beat baseline"
    print(f"  {name:<22} acc {acc:.3f} ({acc - base_acc:+.3f} vs base)   AUC {auc:.3f}   ->  {verdict}")
print("\nNote: minority class here is False (not-anomalous). Check its precision/recall")
print("in the classification reports above - that is where a majority-lean model fails.")
