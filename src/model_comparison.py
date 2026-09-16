"""Model comparison on the block-group x month target (>=1 both-method anomaly
day that month; 5,890 rows; ~29% positive), using the same nested
leave-one-county-out spatial CV as model_anomaly_expanded.py.

Features (fixed across all models): population, flash density, drive time,
coastal distance, month indicators (May = reference). Spatial lag is dropped -
it was not significant in the earlier run.

Models:
  1. Naive Bayes (GaussianNB)          - too-simple sanity floor
  2. Decision tree (tuned)             - single-tree floor: how much does
                                         ensembling actually buy?
  3. Logistic regression (plain)       - additive baseline
  4. Logistic regression + interactions- population x month, flash x month,
                                         coast x flash (stays interpretable)
  5. Random forest (tuned)             - bagged-tree ensemble
  6. HistGradientBoosting (tuned)      - boosted-tree ensemble, the model most
                                         likely to beat RF on tabular data

Tuned models get an inner RandomizedSearchCV (GroupKFold(4) on training
counties, roc_auc). Metrics are pooled out-of-fold AUC / precision / recall,
plus per-county-fold AUC mean +/- std so between-model gaps can be judged
against fold-to-fold noise.

Deliberately excluded: neural nets / deep learning - 5,890 rows is far too
small for that family to have any advantage here.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 42

EXTRA = pd.read_csv("data/processed/features/blockgroup_features_extra.csv", dtype={"GEOID": str})
FLAGS = pd.read_csv("data/processed/anomaly/flagged_blockgroup_days.csv", dtype={"GEOID": str})

STATIC = ["P1_001N", "flash_per_km2", "drive_min_to_shelter", "coast_dist_km"]
MONTHS = [6, 7, 8, 9]
MONTH_COLS = [f"month_{m}" for m in MONTHS]
FEATURES = STATIC + MONTH_COLS

# ---- build block-group x month table ----
FLAGS["date"] = pd.to_datetime(FLAGS["date"])
FLAGS["year"] = FLAGS["date"].dt.year
FLAGS["month"] = FLAGS["date"].dt.month
pos = (FLAGS[FLAGS["both_flag"]].groupby(["GEOID", "year", "month"]).size()
       .rename("n").reset_index())
grid = pd.MultiIndex.from_product(
    [EXTRA["GEOID"], [(2024, m) for m in range(5, 10)] + [(2025, m) for m in range(5, 10)]],
    names=["GEOID", "ym"],
).to_frame(index=False)
grid[["year", "month"]] = pd.DataFrame(grid["ym"].tolist(), index=grid.index)
grid = grid.drop(columns="ym").merge(pos, on=["GEOID", "year", "month"], how="left")
grid["target"] = (grid["n"].fillna(0) > 0).astype(int)
df = grid.merge(EXTRA[["GEOID", "COUNTY_NAME"] + STATIC], on="GEOID", how="left")
for m in MONTHS:
    df[f"month_{m}"] = (df["month"] == m).astype(int)

print(f"rows: {len(df)}   positive rate: {df['target'].mean():.3f}")
print(f"counties: {sorted(df['COUNTY_NAME'].unique())}\n")

# ---- interaction feature builder (for the interaction logreg) ----
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

# ---- results table ----
rows = []
for name in MODELS:
    y = np.array(oof[name]["y"]); p = np.array(oof[name]["p"])
    auc = roc_auc_score(y, p)
    # F1-optimal threshold on pooled OOF
    ts = np.arange(0.05, 0.96, 0.01)
    f1s = [(2 * precision_score(y, p >= t, zero_division=0) * recall_score(y, p >= t, zero_division=0) /
            (precision_score(y, p >= t, zero_division=0) + recall_score(y, p >= t, zero_division=0) + 1e-9)) for t in ts]
    bt = ts[int(np.argmax(f1s))]
    rows.append({
        "model": name,
        "pooled_AUC": round(auc, 3),
        "fold_AUC_mean": round(np.mean(per_fold_auc[name]), 3),
        "fold_AUC_std": round(np.std(per_fold_auc[name]), 3),
        "prec@0.5": round(precision_score(y, p >= 0.5, zero_division=0), 3),
        "rec@0.5": round(recall_score(y, p >= 0.5, zero_division=0), 3),
        "prec@F1opt": round(precision_score(y, p >= bt, zero_division=0), 3),
        "rec@F1opt": round(recall_score(y, p >= bt, zero_division=0), 3),
        "F1opt_thr": round(bt, 2),
    })
res = pd.DataFrame(rows).sort_values("pooled_AUC", ascending=False)
print("=" * 100)
print("MODEL COMPARISON  -  block-group x month target, nested LOCO spatial CV")
print("=" * 100)
print(res.to_string(index=False))

# ---- noise assessment ----
best = res.iloc[0]
print("\n" + "=" * 100)
print("IS THE WINNER REAL?")
print("=" * 100)
print(f"top model: {best['model']}  pooled AUC {best['pooled_AUC']}  "
      f"(per-fold {best['fold_AUC_mean']} +/- {best['fold_AUC_std']})")
typical_std = np.mean([r["fold_AUC_std"] for r in rows])
print(f"mean per-fold AUC std across models: {typical_std:.3f}")
for _, r in res.iterrows():
    gap = best["pooled_AUC"] - r["pooled_AUC"]
    flag = "WITHIN NOISE" if gap <= typical_std else "distinguishable"
    print(f"  {r['model']:<22} AUC {r['pooled_AUC']:.3f}  gap to top {gap:+.3f}  -> {flag}")
print("\nPer-fold AUC by county (rows = models):")
pf = pd.DataFrame(per_fold_auc, index=counties).T.round(3)
print(pf.to_string())

# ---- interaction logreg coefficients (interpretability, full data) ----
Xi = StandardScaler().fit_transform(add_interactions(df))
li = sm.Logit(df["target"].to_numpy(), sm.add_constant(Xi)).fit(disp=0)
ct = pd.DataFrame({"term": ["const"] + INT_COLS, "coef": li.params, "p": li.pvalues})
print("\nInteraction logreg - significant terms (p < 0.05), full data:")
print(ct[ct["p"] < 0.05].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"McFadden pseudo-R^2 {li.prsquared:.3f}")
