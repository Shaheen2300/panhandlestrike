"""Retrain the expanded model on the 4-year (2022-2025) window and answer the
two questions this experiment is for:

  1. Does the ONI coefficient become properly interpretable now that training
     spans genuinely different ENSO states (2022 La Nina, 2023 transitioning
     to El Nino, 2024-25 neutral) instead of two neutral seasons?
  2. Does performance on the held-out 2026 El Nino data improve from 0.772?

No per-year dummy is included (see build_4yr_features.py) so ONI is not
competing with a redundant year-indicator for the same variance.

nested_cv() and evaluate_2026() are also imported by
src/generate_report_figures.py so the 4-year side of the 2yr-vs-4yr
comparison figures (16/18/19) is a live recompute, not a cached number.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
RNG = 42

TRAIN_PATH = "data/processed/features/blockgroup_month_4yr.csv"
TEST26_PATH = "data/processed/features/blockgroup_month_2026_test_4yr.csv"

FEATS = ["P1_001N", "flash_per_km2_4yr", "drive_min_to_shelter", "coast_dist_km",
        "pct_developed", "pct_forest", "pct_wetland", "prev_month_flagged",
        "same_month_last_yr_flagged", "is_may", "oni", "doy_sin", "doy_cos"]

RF_SPACE = {"n_estimators": [150, 250, 350, 500], "max_depth": [3, 4, 6, 8, 12, None],
           "min_samples_split": [2, 5, 10, 20, 40], "min_samples_leaf": [1, 2, 4, 8],
           "max_features": ["sqrt", "log2", 0.5, 1.0]}


def load_data():
    """Reads the saved 4-year artifact files fresh every call."""
    train = pd.read_csv(TRAIN_PATH, dtype={"GEOID": str})
    test26 = pd.read_csv(TEST26_PATH, dtype={"GEOID": str})
    return train, test26


def tuned_rf(X, y, g):
    s = RandomizedSearchCV(RandomForestClassifier(random_state=RNG, n_jobs=-1), RF_SPACE,
                           n_iter=25, scoring="roc_auc", cv=GroupKFold(4), random_state=RNG, n_jobs=-1)
    s.fit(X, y, groups=g)
    return s.best_estimator_


def nested_cv(train_df, feats=None):
    """Nested leave-one-county-out spatial CV. Returns
    {"LogReg": {...}, "RandomForest": {...}} with pooled_AUC/fold_mean/
    fold_std/prec/rec - live, re-derived, not cached."""
    feats = feats or FEATS
    counties = sorted(train_df["COUNTY_NAME"].unique())
    oof_y, oof_lr, oof_rf, pf_lr, pf_rf = [], [], [], [], []
    for c in counties:
        tr, te = train_df["COUNTY_NAME"] != c, train_df["COUNTY_NAME"] == c
        ytr, yte = train_df.loc[tr, "target"].to_numpy(), train_df.loc[te, "target"].to_numpy()
        g = train_df.loc[tr, "COUNTY_NAME"].to_numpy()
        sc = StandardScaler().fit(train_df.loc[tr, feats])
        Xtr, Xte = sc.transform(train_df.loc[tr, feats]), sc.transform(train_df.loc[te, feats])
        lr = LogisticRegression(max_iter=4000).fit(Xtr, ytr)
        rf = tuned_rf(Xtr, ytr, g)
        plr, prf = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
        oof_y += list(yte); oof_lr += list(plr); oof_rf += list(prf)
        pf_lr.append(roc_auc_score(yte, plr)); pf_rf.append(roc_auc_score(yte, prf))
    y = np.array(oof_y)
    out = {}
    for name, p, pf in (("LogReg", oof_lr, pf_lr), ("RandomForest", oof_rf, pf_rf)):
        p = np.array(p)
        out[name] = {
            "pooled_AUC": roc_auc_score(y, p), "fold_mean": np.mean(pf), "fold_std": np.std(pf),
            "prec@0.5": precision_score(y, p >= .5, zero_division=0),
            "rec@0.5": recall_score(y, p >= .5, zero_division=0),
        }
    return out


def evaluate_2026(train_df, test_df, feats=None):
    """Final fit on all training rows, evaluated on the untouched 2026 test
    set. Returns {"LogReg": {...}, "RandomForest": {...}} with auc/proba/
    prec/rec/mean_prob/pct_above_0.5 - live, re-derived, not cached."""
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
            "mean_prob": p.mean(), "pct_above_0.5": (p >= 0.5).mean() * 100,
        }
    return out


# ===========================================================================
if __name__ == "__main__":
    train, test26 = load_data()

    print(f"Training rows: {len(train)}  positive rate {train['target'].mean():.3f}")
    print(f"ONI range in training: {train['oni'].min():.2f} to {train['oni'].max():.2f}")
    print(f"ONI range in 2026 test: {test26['oni'].min():.2f} to {test26['oni'].max():.2f}")
    print(train.groupby("year")[["target", "oni"]].agg(pos_rate=("target", "mean"), oni_mean=("oni", "mean")).round(3))

    # -----------------------------------------------------------------------
    # 1. LogReg coefficients on full 4-year training data - is ONI interpretable?
    # -----------------------------------------------------------------------
    Xz = StandardScaler().fit_transform(train[FEATS].to_numpy(float))
    logit = sm.Logit(train["target"].to_numpy(int), sm.add_constant(Xz)).fit(disp=0)
    coef_tbl = pd.DataFrame({
        "term": ["intercept"] + FEATS, "coef_std": logit.params, "p_value": logit.pvalues,
        "odds_ratio_per_SD": np.exp(logit.params),
    })
    print("\n" + "=" * 74)
    print("LOGISTIC REGRESSION - 4-year training data, no year dummy")
    print("=" * 74)
    print(coef_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"McFadden pseudo-R^2 {logit.prsquared:.3f}")
    oni_row = coef_tbl[coef_tbl["term"] == "oni"].iloc[0]
    print(f"\nONI coefficient: {oni_row['coef_std']:.4f}  p={oni_row['p_value']:.4g}  "
          f"OR/SD={oni_row['odds_ratio_per_SD']:.3f}")

    # -----------------------------------------------------------------------
    # 2. Nested leave-one-county-out CV (same protocol as the 2-year runs)
    # -----------------------------------------------------------------------
    cv = nested_cv(train)
    print("\n" + "=" * 74)
    print("NESTED LOCO SPATIAL CV - 4-year training window")
    print("=" * 74)
    for name, m in cv.items():
        print(f"{name:<14} pooled AUC {m['pooled_AUC']:.3f}   "
             f"fold mean {m['fold_mean']:.3f} +/- {m['fold_std']:.3f}   "
             f"prec@.5 {m['prec@0.5']:.3f}   rec@.5 {m['rec@0.5']:.3f}")
    print("(2-year, 2024-25-only CV reference: RF 0.812, LogReg 0.790)")

    # -----------------------------------------------------------------------
    # 3. Final fit on ALL 4 years -> evaluate on the untouched 2026 test set
    # -----------------------------------------------------------------------
    oot = evaluate_2026(train, test26)
    print("\n" + "=" * 74)
    print("2026 OUT-OF-TIME TEST - trained on 4 years (2022-2025), same protocol as before")
    print("=" * 74)
    print(f"2026 positive rate: {test26['target'].mean():.3f}  (4yr training positive rate: {train['target'].mean():.3f})")
    for name, m in oot.items():
        print(f"{name:<14} AUC {m['auc']:.3f}   prec@.5 {m['prec@0.5']:.3f}   "
             f"rec@.5 {m['rec@0.5']:.3f}   mean predicted prob {m['mean_prob']:.3f}   "
             f"pct >=0.5 {m['pct_above_0.5']:.0f}%")

    print("\n(2-year-trained RF baseline on this exact 2026 test: AUC 0.772)")
    print("(2-year-trained LogReg baseline on this exact 2026 test: AUC 0.812, but 100% of predictions saturated above 0.5)")
