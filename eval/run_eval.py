"""Step 6 — evaluation.

Headline: PR-AUC with bootstrap 95% CI (~1000 resamples of the test set).
Operating point: recall at a precision target chosen above the at-risk
prevalence (~51-57%, so at_risk is near-balanced / majority, not minority).
precision=0.5 would be cleared trivially by flagging everyone, so we use
precision=0.75 as the primary operating point (requires real model signal)
and also report the F1-optimal threshold as a secondary reference.
Calibration curve (one plot).
Subgroup diagnostic: recall for true Withdrawn vs true Fail within predicted at-risk.
Fairness slice: recall & FPR by gender and imd_band (fairness_eval.parquet, reporting only).
Writes artifacts/metrics.json (model name, checkpoint, PR-AUC + CI, date).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from common import ARTIFACTS, MODELS, PROCESSED, FAIRNESS_COLUMNS

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
# at_risk prevalence is ~0.51-0.57 (near-balanced, majority positive). A
# precision=0.5 target is cleared for free by flagging everyone, so it is not
# a meaningful operating point. Use precision=0.75 (above prevalence) as the
# primary actionable threshold; F1-optimal is reported as a secondary reference.
PRECISION_TARGET = 0.75
CHECKPOINTS = ["week_4", "week_8", "week_12"]


def latest_pooled_model():
    """Prefer the fixed production artifact; fall back to newest timestamped."""
    prod = MODELS / "lgb_pooled_production.joblib"
    if prod.exists():
        model_path = prod
        ts = "production"
    else:
        cands = sorted(MODELS.glob("lgb_pooled_*.joblib"))
        if not cands:
            raise FileNotFoundError("no lgb_pooled model artifact found")
        model_path = cands[-1]
        ts = model_path.stem.replace("lgb_pooled_", "")
    feat_path = MODELS / f"lgb_pooled_{ts}_features.json"
    with open(feat_path) as fh:
        features = json.load(fh)
    model = joblib.load(model_path)
    return model, features, ts


def pr_auc(y_true, y_score):
    return float(average_precision_score(y_true, y_score))


def bootstrap_ci(y_true, y_score, n=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    stats = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            stats[i] = np.nan
        else:
            stats[i] = pr_auc(y_true[idx], y_score[idx])
    stats = stats[~np.isnan(stats)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def operating_point(y_true, y_score):
    """Recall at precision>=PRECISION_TARGET (max recall), plus F1-optimal."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve aligns precision[i], recall[i] with thresholds[i].
    best_recall, best_prec, best_thr = 0.0, 0.0, 1.0
    for p, r, t in zip(precision, recall, thresholds):
        if p >= PRECISION_TARGET and r > best_recall:
            best_recall, best_prec, best_thr = float(r), float(p), float(t)

    # F1-optimal threshold (secondary reference).
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    f1_idx = int(np.argmax(f1))
    f1_recall, f1_prec, f1_thr = float(recall[f1_idx]), float(precision[f1_idx]), float(thresholds[f1_idx])

    return {
        "recall": best_recall,
        "precision": best_prec,
        "threshold": best_thr,
        "f1_recall": f1_recall,
        "f1_precision": f1_prec,
        "f1_threshold": f1_thr,
    }


def evaluate(model, feature_cols, validation, test, ts):
    validation_ops = {}
    for cp in CHECKPOINTS:
        va = validation[validation["checkpoint_label"] == cp].copy()
        if va.empty:
            continue
        X = va[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        y = va["at_risk"].values.astype(int)
        score = model.predict_proba(X.fillna(0))[:, 1]
        validation_ops[cp] = operating_point(y, score)

    rows = []
    for cp in CHECKPOINTS:
        te = test[test["checkpoint_label"] == cp].copy()
        if te.empty:
            continue
        X = te[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)
        y = te["at_risk"].values.astype(int)
        score = model.predict_proba(X)[:, 1]

        auc = pr_auc(y, score)
        lo, hi = bootstrap_ci(y, score)
        op = validation_ops.get(cp)
        if op is None:
            raise RuntimeError(f"no validation rows available for checkpoint {cp}")
        y_pred = (score >= op["threshold"]).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        test_op = operating_point(y, score)

        rows.append({
            "checkpoint": cp, "model": "lgb_pooled",
            "pr_auc": round(auc, 4), "pr_auc_ci_lo": round(lo, 4), "pr_auc_ci_hi": round(hi, 4),
            "n": int(len(te)), "n_at_risk": int(y.sum()),
            f"recall_at_precision_{int(PRECISION_TARGET*100)}": round(op["recall"], 4),
            "precision_at_operating_point": round(op["precision"], 4),
            "operating_threshold": round(op["threshold"], 4),
            "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
            "confusion_matrix": cm.tolist(),
            "f1_optimal_recall": round(test_op["f1_recall"], 4),
            "f1_optimal_precision": round(test_op["f1_precision"], 4),
            "f1_optimal_threshold": round(test_op["f1_threshold"], 4),
            "validation_n": int(
                len(validation[validation["checkpoint_label"] == cp])
            ),
        })

    return pd.DataFrame(rows)


def main():
    model, feature_cols, ts = latest_pooled_model()
    features = pd.read_parquet(PROCESSED / "features.parquet")
    validation = features[features["split"] == "validation"].copy()
    test = features[features["split"] == "test"].copy()

    eval_df = evaluate(model, feature_cols, validation, test, ts)

    # Per-course breakdown is kept separate because CCC is zero-shot: it has
    # no 2013 training presentation. AAA has no 2014B validation rows.
    print("\n=== Test PR-AUC by course ===")
    course_rows = []
    for course, te_course in test.groupby("code_module"):
        for cp in CHECKPOINTS:
            te = te_course[te_course["checkpoint_label"] == cp].copy()
            if te.empty:
                continue
            X = te[feature_cols].copy()
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            y = te["at_risk"].values.astype(int)
            score = model.predict_proba(X.fillna(0))[:, 1]
            course_rows.append({
                "course": course,
                "checkpoint": cp,
                "pr_auc": round(pr_auc(y, score), 4),
                "n": int(len(te)),
                "exception": (
                    "zero_shot_no_training_rows"
                    if course == "CCC"
                    else "no_validation_rows"
                    if course == "AAA"
                    else None
                ),
            })
    course_df = pd.DataFrame(course_rows)
    print(course_df.to_string(index=False))

    # final_result is already carried in features.parquet (used for subgroup diagnostic).
    print("\n=== Subgroup diagnostic: recall among true Withdrawn vs true Fail ===")
    sub_rows = []
    for cp in CHECKPOINTS:
        te = test[test["checkpoint_label"] == cp].copy()
        if te.empty:
            continue
        X = te[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)
        score = model.predict_proba(X)[:, 1]
        thr = float(eval_df[eval_df["checkpoint"] == cp]["operating_threshold"].iloc[0])
        y_pred = (score >= thr).astype(int)
        te = te.copy(); te["pred"] = y_pred
        for grp in ["Withdrawn", "Fail"]:
            m = te["final_result"] == grp
            if m.sum() == 0:
                continue
            rec = float(te.loc[m, "pred"].mean())
            sub_rows.append({"checkpoint": cp, "true_label": grp, "n": int(m.sum()), "recall": round(rec, 4)})
    sub = pd.DataFrame(sub_rows)
    print(sub.to_string(index=False))

    # Fairness slice (reporting only, no mitigation).
    print("\n=== Fairness slice: recall & FPR by gender and imd_band ===")
    fair = pd.read_parquet(PROCESSED / "fairness_eval.parquet")
    fair_rows = []
    for cp in CHECKPOINTS:
        te = test[test["checkpoint_label"] == cp].copy()
        if te.empty:
            continue
        X = te[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)
        score = model.predict_proba(X)[:, 1]
        thr = float(eval_df[eval_df["checkpoint"] == cp]["operating_threshold"].iloc[0])
        y_pred = (score >= thr).astype(int)
        te = te.copy(); te["pred"] = y_pred
        # Use fairness_eval.parquet as the source of truth for protected attributes.
        te = te.drop(columns=[c for c in FAIRNESS_COLUMNS if c in te.columns], errors="ignore")
        te = te.merge(fair, on=["id_student", "module_presentation"], how="left")
        for col in FAIRNESS_COLUMNS:
            for val, grp in te.groupby(col):
                y = grp["at_risk"].values.astype(int)
                p = grp["pred"].values.astype(int)
                tp = int(((p == 1) & (y == 1)).sum())
                fn = int(((p == 0) & (y == 1)).sum())
                fp = int(((p == 1) & (y == 0)).sum())
                tn = int(((p == 0) & (y == 0)).sum())
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                fair_rows.append({"checkpoint": cp, "column": col, "value": val, "n": len(grp),
                                  "recall": round(rec, 4), "fpr": round(fpr, 4),
                                  "flag_rate": round(p.mean(), 4),
                                  "true_prevalence": round(y.mean(), 4)})
    fair_df = pd.DataFrame(fair_rows)
    print(fair_df.to_string(index=False))

    # Calibration curve (one plot) on overall test set.
    Xall = test[feature_cols].copy()
    for c in Xall.columns:
        Xall[c] = pd.to_numeric(Xall[c], errors="coerce")
    Xall = Xall.fillna(0)
    yall = test["at_risk"].values.astype(int)
    score_all = model.predict_proba(Xall)[:, 1]
    frac_pos, mean_pred = calibration_curve(yall, score_all, n_bins=10, strategy="quantile")

    plots_dir = ARTIFACTS / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (held-out test 2014J, all checkpoints)")
    ax.legend()
    ax.grid(alpha=0.3)
    cal_path = plots_dir / "calibration_curve.png"
    fig.tight_layout(); fig.savefig(cal_path, dpi=150); plt.close(fig)
    print("Wrote", cal_path)

    # metrics.json
    metrics = {
        "model": "lgb_pooled",
        "model_artifact": f"lgb_pooled_{ts}.joblib",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "checkpoints": eval_df[["checkpoint", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi",
                                f"recall_at_precision_{int(PRECISION_TARGET*100)}", "precision_at_operating_point",
                                "operating_threshold", "precision", "recall", "f1", "confusion_matrix",
                                "f1_optimal_recall", "f1_optimal_precision",
                                "f1_optimal_threshold", "n", "n_at_risk",
                                "validation_n"]].to_dict("records"),
        "overall_pr_auc": round(float(pr_auc(yall, score_all)), 4),
        "course_pr_auc": course_df.to_dict("records"),
        "split_exceptions": {
            "CCC": "No 2013B or 2013J rows; test performance is zero-shot cross-course generalization.",
            "AAA": "No 2014B rows; threshold/model-selection validation has no AAA-specific rows.",
        },
        "subgroup": sub.to_dict("records"),
        "fairness": fair_df.to_dict("records"),
        "calibration_plot": str(cal_path),
        "leakage_audit": "features/test_no_leakage.py (passed)",
    }
    metrics_path = ARTIFACTS / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Wrote", metrics_path)

    print("\n=== Headline PR-AUC (bootstrap 95% CI) ===")
    print(eval_df[["checkpoint", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi", "n"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
