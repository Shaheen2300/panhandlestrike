"""4-year rerun of model_comparison.py: identical 6-model bake-off, identical
feature scope (population, flash density, drive time, coastal distance, month
indicators - May as reference; no ONI/NLCD/lags, that's the separate expanded
model), identical nested leave-one-county-out spatial CV protocol. Only the
training window widens from 2024-25 (5,890 rows) to 2022-2025 (11,780 rows),
and the hazard input becomes the 4-year flash density.

Output: printed comparison table + data/processed/features/model_comparison_4yr_results.csv
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 42

df = pd.read_csv("data/processed/features/blockgroup_month_4yr.csv", dtype={"GEOID": str})
df = df.rename(columns={"flash_per_km2_4yr": "flash_per_km2"})

STATIC = ["P1_001N", "flash_per_km2", "drive_min_to_shelter", "coast_dist_km"]
MONTHS = [6, 7, 8, 9]
MONTH_COLS = [f"month_{m}" for m in MONTHS]
FEATURES = STATIC + MONTH_COLS
for m in MONTHS:
    df[f"month_{m}"] = (df["month"] == m).astype(int)

print(f"rows: {len(df)}   positive rate: {df['target'].mean():.3f}   years: {sorted(df['year'].unique())}")
print(f"counties: {sorted(df['COUNTY_NAME'].unique())}\n")


def add_interactions(frame):
    X = frame[FEATURES].copy()
    for m in MONTH_COLS:
        X[f"pop_x_{m}"] = frame["P1_001N"] * frame[m]
        X[f"flash_x_{m}"] = frame["flash_per_km2"] * frame[m]
    X["coast_x_flash"] = frame["coast_dist_km"] * frame["flash_per_km2"]
    return X


INT_COLS = list(add_interactions(df.head(1)).columns)

SEARCH = {
    "tree": (DecisionTreeClassifier(random_state=RNG), {
        "max_depth": [2, 3, 4, 6, 8, 12, None],
        "min_samples_split": [2, 10, 20, 50, 100],
        "min_samples_leaf": [1, 5, 10, 20, 50],
        "criterion": ["gini", "entropy"],
    }),
    "rf": (RandomForestClassifier(random_state=RNG, n_jobs=-1), {
        "n_estimators": [150, 250, 350, 500],
        "max_depth": [3, 4, 6, 8, 12, None],
        "min_samples_split": [2, 5, 10, 20, 40],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 1.0],
    }),
    "hgb": (HistGradientBoostingClassifier(random_state=RNG), {
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 400],
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0.0, 0.1, 1.0],
    }),
}


def tuned(kind, X, y, groups):
    est, space = SEARCH[kind]
    s = RandomizedSearchCV(est, space, n_iter=25, scoring="roc_auc",
                           cv=GroupKFold(4), random_state=RNG, n_jobs=-1)
    s.fit(X, y, groups=groups)
    return s.best_estimator_


MODELS = ["NaiveBayes", "DecisionTree", "LogReg", "LogReg+interactions", "RandomForest", "HistGB"]
counties = sorted(df["COUNTY_NAME"].unique())
oof = {m: {"y": [], "p": []} for m in MODELS}
per_fold_auc = {m: [] for m in MODELS}

for c in counties:
    tr, te = df["COUNTY_NAME"] != c, df["COUNTY_NAME"] == c
    ytr, yte = df.loc[tr, "target"].to_numpy(), df.loc[te, "target"].to_numpy()
    g = df.loc[tr, "COUNTY_NAME"].to_numpy()

    sc = StandardScaler().fit(df.loc[tr, FEATURES])
    Xtr, Xte = sc.transform(df.loc[tr, FEATURES]), sc.transform(df.loc[te, FEATURES])
    sc_i = StandardScaler().fit(add_interactions(df.loc[tr]))
    Xtr_i, Xte_i = sc_i.transform(add_interactions(df.loc[tr])), sc_i.transform(add_interactions(df.loc[te]))

    fitted = {
        "NaiveBayes": (GaussianNB().fit(Xtr, ytr), Xte),
        "DecisionTree": (tuned("tree", Xtr, ytr, g), Xte),
        "LogReg": (LogisticRegression(max_iter=2000).fit(Xtr, ytr), Xte),
        "LogReg+interactions": (LogisticRegression(max_iter=5000).fit(Xtr_i, ytr), Xte_i),
        "RandomForest": (tuned("rf", Xtr, ytr, g), Xte),
        "HistGB": (tuned("hgb", Xtr, ytr, g), Xte),
    }
    for name, (mdl, Xe) in fitted.items():
        p = mdl.predict_proba(Xe)[:, 1]
        oof[name]["y"] += list(yte)
        oof[name]["p"] += list(p)
        per_fold_auc[name].append(roc_auc_score(yte, p))

rows = []
for name in MODELS:
    y = np.array(oof[name]["y"]); p = np.array(oof[name]["p"])
    auc = roc_auc_score(y, p)
    rows.append({
        "model": name,
        "pooled_AUC": round(auc, 3),
        "fold_AUC_mean": round(np.mean(per_fold_auc[name]), 3),
        "fold_AUC_std": round(np.std(per_fold_auc[name]), 3),
        "prec@0.5": round(precision_score(y, p >= 0.5, zero_division=0), 3),
        "rec@0.5": round(recall_score(y, p >= 0.5, zero_division=0), 3),
    })
res = pd.DataFrame(rows).sort_values("pooled_AUC", ascending=False)
print("=" * 100)
print("MODEL COMPARISON (4-YEAR WINDOW, 2022-2025)  -  block-group x month target, nested LOCO spatial CV")
print("=" * 100)
print(res.to_string(index=False))
res.to_csv("data/processed/features/model_comparison_4yr_results.csv", index=False)

best = res.iloc[0]
typical_std = np.mean([r["fold_AUC_std"] for r in rows])
print(f"\nmean per-fold AUC std across models: {typical_std:.3f}")
for _, r in res.iterrows():
    gap = best["pooled_AUC"] - r["pooled_AUC"]
    flag = "WITHIN NOISE" if gap <= typical_std else "distinguishable"
    print(f"  {r['model']:<22} AUC {r['pooled_AUC']:.3f}  gap to top {gap:+.3f}  -> {flag}")

print(f"\nSaved to data/processed/features/model_comparison_4yr_results.csv")
