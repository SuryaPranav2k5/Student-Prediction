"""Step 5 — modeling.

Trains and compares, on the SAME feature set:
  * Baseline (mandatory): LogisticRegression(class_weight='balanced')
  * Main: LightGBM (objective=binary, is_unbalance=True), no SMOTE.
  * Second algorithm family: RandomForest(class_weight='balanced').

Two LightGBM strategies:
  (a) pooled model across all checkpoints with week_elapsed as a feature
  (b) three separate models, one per checkpoint

Uses 2014B for validation comparisons and evaluates the final held-out test set
(2014J) separately.
Saves versioned model artifacts to artifacts/models/.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ARTIFACTS, MODELS, PROCESSED, FAIRNESS_COLUMNS

RANDOM_SEED = 42

ID_COLS = [
    "id_student", "module_presentation", "code_module", "code_presentation",
    "checkpoint_label", "cutoff_day", "week_elapsed", "at_risk", "final_result", "split",
] + FAIRNESS_COLUMNS  # protected attributes: never model inputs

MODEL_DIR = MODELS
MANIFEST = ARTIFACTS / "model_manifest.json"


def feature_columns(features: pd.DataFrame, include_week_elapsed: bool) -> list:
    cols = [c for c in features.columns if c not in ID_COLS]
    if not include_week_elapsed:
        cols = [c for c in cols if c != "week_elapsed"]
    # Hard guard: a protected attribute must never reach the model.
    leaked = sorted(set(cols) & set(FAIRNESS_COLUMNS))
    assert not leaked, (
        f"PROTECTED ATTRIBUTES in model feature set: {leaked}. "
        "gender/disability/age_band/imd_band are Step-6 fairness inputs only."
    )
    return cols


def prep(df: pd.DataFrame, cols: list):
    X = df[cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    y = df["at_risk"].values.astype(int)
    return X, y


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def train_and_evaluate(features_path: Path = PROCESSED / "features.parquet") -> dict:
    features = pd.read_parquet(features_path)
    train = features[features["split"] == "train"].copy()
    validation = features[features["split"] == "validation"].copy()
    test = features[features["split"] == "test"].copy()
    print(f"train rows={len(train)} validation rows={len(validation)} test rows={len(test)}")

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {"timestamp": timestamp, "models": []}

    # ---- Baseline: LogisticRegression (pooled, week_elapsed included) ----
    # StandardScaler is required: features span wildly different scales
    # (click counts in the thousands vs scores 0-100), and unscaled lbfgs hits
    # the iteration limit without converging, which understates the baseline.
    # A knowingly under-converged baseline would inflate the main model's
    # apparent advantage, so scaling is not optional here.
    cols_lr = feature_columns(features, include_week_elapsed=True)
    Xtr, ytr = prep(train, cols_lr)
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED),
    )
    lr.fit(Xtr, ytr)
    joblib.dump(lr, MODEL_DIR / f"lr_pooled_{timestamp}.joblib")
    manifest["models"].append({"name": "lr_pooled", "path": f"lr_pooled_{timestamp}.joblib",
                               "checkpoint": "pooled", "strategy": "baseline"})
    for split_name, split_df in [("validation", validation), ("test", test)]:
        for cp in ["week_4", "week_8", "week_12"]:
            te = split_df[split_df["checkpoint_label"] == cp]
            if te.empty:
                continue
            Xte, yte = prep(te, cols_lr)
            score = lr.predict_proba(Xte)[:, 1]
            results.append({"model": "lr_pooled (baseline)", "split": split_name,
                            "checkpoint": cp, "pr_auc": pr_auc(yte, score),
                            "n": len(te)})

    # ---- LightGBM pooled (week_elapsed feature) ----
    cols_lgb = feature_columns(features, include_week_elapsed=True)
    Xtr, ytr = prep(train, cols_lgb)
    lgb_pooled = LGBMClassifier(objective="binary", is_unbalance=True, random_state=RANDOM_SEED,
                                n_estimators=300, learning_rate=0.05, num_leaves=31,
                                min_child_samples=20, verbosity=-1, n_jobs=-1)
    lgb_pooled.fit(Xtr, ytr)
    joblib.dump(lgb_pooled, MODEL_DIR / f"lgb_pooled_{timestamp}.joblib")
    manifest["models"].append({"name": "lgb_pooled", "path": f"lgb_pooled_{timestamp}.joblib",
                               "checkpoint": "pooled", "strategy": "pooled"})
    feature_list = list(cols_lgb)
    with open(MODEL_DIR / f"lgb_pooled_{timestamp}_features.json", "w") as fh:
        json.dump(feature_list, fh)
    for split_name, split_df in [("validation", validation), ("test", test)]:
        for cp in ["week_4", "week_8", "week_12"]:
            te = split_df[split_df["checkpoint_label"] == cp]
            if te.empty:
                continue
            Xte, yte = prep(te, cols_lgb)
            score = lgb_pooled.predict_proba(Xte)[:, 1]
            results.append({"model": "lgb_pooled", "split": split_name,
                            "checkpoint": cp, "pr_auc": pr_auc(yte, score),
                            "n": len(te)})

    # ---- RandomForest (second algorithm family) ----
    # A bagged tree ensemble is a genuinely different bias/variance profile from
    # both the linear baseline and the boosted trees, so this is the "second
    # reasonable model" the brief asks for -- not another LightGBM variant.
    cols_rf = feature_columns(features, include_week_elapsed=True)
    Xtr, ytr = prep(train, cols_rf)
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED,
        n_jobs=-1, min_samples_leaf=2,
    )
    rf.fit(Xtr, ytr)
    joblib.dump(rf, MODEL_DIR / f"rf_pooled_{timestamp}.joblib")
    manifest["models"].append({"name": "rf_pooled", "path": f"rf_pooled_{timestamp}.joblib",
                               "checkpoint": "pooled", "strategy": "random_forest"})
    for split_name, split_df in [("validation", validation), ("test", test)]:
        for cp in ["week_4", "week_8", "week_12"]:
            te = split_df[split_df["checkpoint_label"] == cp]
            if te.empty:
                continue
            Xte, yte = prep(te, cols_rf)
            score = rf.predict_proba(Xte)[:, 1]
            results.append({"model": "rf_pooled (random forest)", "split": split_name,
                            "checkpoint": cp, "pr_auc": pr_auc(yte, score),
                            "n": len(te)})

    # ---- LightGBM separate per-checkpoint ----
    for cp in ["week_4", "week_8", "week_12"]:
        tr = train[train["checkpoint_label"] == cp]
        te = test[test["checkpoint_label"] == cp]
        if tr.empty or te.empty:
            continue
        cols_cp = feature_columns(features, include_week_elapsed=False)
        Xtr, ytr = prep(tr, cols_cp)
        Xte, yte = prep(te, cols_cp)
        lgb = LGBMClassifier(objective="binary", is_unbalance=True, random_state=RANDOM_SEED,
                             n_estimators=300, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, verbosity=-1, n_jobs=-1)
        lgb.fit(Xtr, ytr)
        fname = f"lgb_{cp}_{timestamp}.joblib"
        joblib.dump(lgb, MODEL_DIR / fname)
        with open(MODEL_DIR / f"lgb_{cp}_{timestamp}_features.json", "w") as fh:
            json.dump(list(cols_cp), fh)
        manifest["models"].append({"name": f"lgb_{cp}", "path": fname,
                                   "checkpoint": cp, "strategy": "separate"})
        score = lgb.predict_proba(Xte)[:, 1]
        results.append({"model": f"lgb_{cp} (separate)", "split": "test",
                        "checkpoint": cp, "pr_auc": pr_auc(yte, score), "n": len(te)})

    res = pd.DataFrame(results)
    print("\n=== Step 5: PR-AUC comparison ===")
    print(res[res["split"] == "validation"].pivot(index="model", columns="checkpoint",
          values="pr_auc").round(4).to_string())
    print("\nHeld-out test results:")
    print(res[res["split"] == "test"].pivot(index="model", columns="checkpoint",
          values="pr_auc").round(4).to_string())
    print()
    print("per-checkpoint counts:")
    print(res.groupby("checkpoint")["n"].first().to_string())

    # Pin the production model under a FIXED filename so serving / eval / the
    # dashboard load a stable reference instead of "whatever timestamp sorts
    # last". Timestamped artifacts remain as the version history.
    production_names = {
        f"lgb_pooled_{timestamp}.joblib": "lgb_pooled_production.joblib",
        f"lgb_week_4_{timestamp}.joblib": "lgb_week_4_production.joblib",
        f"lgb_week_8_{timestamp}.joblib": "lgb_week_8_production.joblib",
        f"lgb_week_12_{timestamp}.joblib": "lgb_week_12_production.joblib",
        f"lr_pooled_{timestamp}.joblib": "lr_pooled_production.joblib",
    }
    for src_name, dst_name in production_names.items():
        src = MODEL_DIR / src_name
        if not src.exists():
            continue
        shutil.copyfile(src, MODEL_DIR / dst_name)
        src_feat = MODEL_DIR / src_name.replace(".joblib", "_features.json")
        if src_feat.exists():  # LR baseline has no separate feature-list file
            shutil.copyfile(src_feat, MODEL_DIR / f"{dst_name.replace('.joblib', '_features.json')}")
    print("Pinned production models:", ", ".join(production_names.values()))

    # Persist the pooled-vs-separate comparison for the dashboard / README.
    comparison = {
        "timestamp": timestamp,
        "rows": res.round(6).to_dict("records"),
        "pivot": {
            split_name: split_df.pivot(index="model", columns="checkpoint", values="pr_auc")
            .round(6).to_dict()
            for split_name, split_df in res.groupby("split")
        },
    }
    with open(ARTIFACTS / "comparison.json", "w") as fh:
        json.dump(comparison, fh, indent=2)
    print("Wrote", ARTIFACTS / "comparison.json")

    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("Wrote", MANIFEST)
    return res


if __name__ == "__main__":
    train_and_evaluate()
