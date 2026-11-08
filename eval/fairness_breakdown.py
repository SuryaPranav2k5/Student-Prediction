"""Fairness slice deep-dive and SHAP standing of no_assessment_closed.

Prints the full per-group breakdown (recall / FPR / flag-rate / true base
rate) at each checkpoint's operating point, base-rate prevalence by group,
and the overall SHAP rank & magnitude of `no_assessment_closed`.

These are the tables quoted in the README fairness section.
Usage: python eval/fairness_breakdown.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")
import joblib
import numpy as np
import pandas as pd
import shap

from common import ARTIFACTS, FAIRNESS_COLUMNS, MODELS, PROCESSED

CHECKPOINTS = ["week_4", "week_8", "week_12"]


def latest_pooled():
    """Prefer the fixed production artifact; fall back to newest timestamped."""
    prod = MODELS / "lgb_pooled_production.joblib"
    if prod.exists():
        mp = prod
        ts = "production"
    else:
        cands = sorted(MODELS.glob("lgb_pooled_*.joblib"))
        mp = cands[-1]
        ts = mp.stem.replace("lgb_pooled_", "")
    feat = json.load(open(MODELS / f"lgb_pooled_{ts}_features.json"))
    return joblib.load(mp), feat, ts


def imd_low(s):
    s = str(s)
    if s == "?":
        return None
    try:
        return float(s.replace("%", "").split("-")[0])
    except Exception:
        return None


def main():
    model, feature_cols, ts = latest_pooled()
    features = pd.read_parquet(PROCESSED / "features.parquet")
    test = features[features["split"] == "test"].copy()
    fair = pd.read_parquet(PROCESSED / "fairness_eval.parquet")
    metrics = json.load(open(ARTIFACTS / "metrics.json"))
    thr_map = {c["checkpoint"]: c["operating_threshold"] for c in metrics["checkpoints"]}

    # ---- True base-rate prevalence by group ----
    print("=== TRUE at-risk PREVALENCE by group (held-out test) ===")
    te_prev = test.drop(columns=[c for c in FAIRNESS_COLUMNS if c in test.columns], errors="ignore")
    te_prev = te_prev.merge(fair, on=["id_student", "module_presentation"], how="left")
    for col in ["gender", "imd_band"]:
        g = te_prev.groupby(col)["at_risk"].agg(["count", "mean"]).sort_values("mean")
        print(f"\n--- {col} ---")
        for val, r in g.iterrows():
            print(f"  {str(val):10s} n={int(r['count']):5d} at_risk_prevalence={r['mean']:.4f}")
    te_prev["imd_low"] = te_prev["imd_band"].map(imd_low)
    lo = te_prev[te_prev["imd_low"] <= 20]
    hi = te_prev[te_prev["imd_low"] >= 70]
    print("\nmost deprived (imd 0-20%%): n=%d prevalence=%.4f" % (len(lo), lo["at_risk"].mean()))
    print("least deprived (imd 70-100%%): n=%d prevalence=%.4f" % (len(hi), hi["at_risk"].mean()))
    print("prevalence gap (pp): %.1f" % ((lo["at_risk"].mean() - hi["at_risk"].mean()) * 100))
    print("gender: %s" % te_prev.groupby("gender")["at_risk"].mean().round(4).to_dict())

    # ---- Full per-checkpoint fairness breakdown ----
    print("\n\n=== FULL FAIRNESS BREAKDOWN (per checkpoint operating point) ===")
    for cp in CHECKPOINTS:
        te = test[test["checkpoint_label"] == cp].copy()
        X = te[feature_cols].copy()
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0)
        score = model.predict_proba(X)[:, 1]
        thr = thr_map[cp]
        pred = (score >= thr).astype(int)
        te = te.drop(columns=[c for c in FAIRNESS_COLUMNS if c in te.columns], errors="ignore")
        te = te.merge(fair, on=["id_student", "module_presentation"], how="left")
        te["pred"] = pred
        te["prob"] = score
        print(f"\n--- {cp} (threshold={thr}, overall flag-rate={pred.mean():.3f}, at-risk prevalence={te['at_risk'].mean():.3f}) ---")
        rows = []
        for col in ["gender", "imd_band"]:
            for val, grp in te.groupby(col, sort=True):
                y = grp["at_risk"].values.astype(int)
                p = grp["pred"].values.astype(int)
                tp = int(((p == 1) & (y == 1)).sum()); fn = int(((p == 0) & (y == 1)).sum())
                fp = int(((p == 1) & (y == 0)).sum()); tn = int(((p == 0) & (y == 0)).sum())
                rows.append({
                    "attribute": col, "group": val, "n": len(grp),
                    "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
                    "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
                    "flag_rate": round(p.mean(), 4),
                    "true_prevalence": round(y.mean(), 4),
                    "meanP_given_neg": round(float(grp.loc[y == 0, "prob"].mean()), 4) if (y == 0).sum() else None,
                })
        print(pd.DataFrame(rows).to_string(index=False))
        # Effective-threshold diagnostic: mean predicted P among true negatives.
        print("\n  effective-threshold check (mean predicted P(at-risk) among NOT-at-risk students):")
        for col in ["gender", "imd_band"]:
            parts = []
            for val, grp in te.groupby(col, sort=True):
                neg = grp[grp["at_risk"] == 0]
                if len(neg):
                    parts.append(f"{val}={neg['prob'].mean():.4f} (n={len(neg)})")
            print(f"    {col:10s}: " + ", ".join(parts))

    # ---- SHAP standing of no_assessment_closed ----
    print("\n\n=== no_assessment_closed SHAP STANDING ===")
    te = test[test["checkpoint_label"] == "week_4"].copy()
    X = te[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.iloc[:2000])
    imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    r = imp[imp["feature"] == "no_assessment_closed"].iloc[0]
    print(f"rank: {int(r['rank'])} of {len(imp)} | mean|SHAP| = {r['mean_abs_shap']:.5f} "
          f"| features ranking below it: {int(len(imp) - r['rank'])}")
    print("\nTop 15 global SHAP features (week_4 sample):")
    print(imp.head(15)[["rank", "feature", "mean_abs_shap"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
