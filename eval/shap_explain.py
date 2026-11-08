"""Step 7 — SHAP explainability on the winning (pooled LightGBM) model.

Global feature-importance bar chart + 4 worked individual examples with a
per-student SHAP breakdown. Outputs to artifacts/plots/ and prints a
top-5 SHAP breakdown per example.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from common import ARTIFACTS, MODELS, PROCESSED

PLOTS = ARTIFACTS / "plots"
N_EXAMPLES = 4


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
        feature_cols = json.load(fh)
    return joblib.load(model_path), feature_cols, ts


def main():
    model, feature_cols, ts = latest_pooled_model()
    PLOTS.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(PROCESSED / "features.parquet")
    test = features[features["split"] == "test"].copy()

    X = test[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    y = test["at_risk"].values.astype(int)
    prob = model.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(model)

    # Global importance on a sample for speed.
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(1500, len(X)), replace=False)
    Xs = X.iloc[idx]
    sv = explainer.shap_values(Xs)
    prob_s = model.predict_proba(Xs)[:, 1]
    y_s = y[idx]
    test_s = test.loc[Xs.index].copy()
    test_s["prob"] = prob_s
    test_s["at_risk"] = y_s

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("Global feature importance (SHAP, pooled LightGBM)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS / "shap_global_importance.png", dpi=150); plt.close(fig)
    print("Wrote", PLOTS / "shap_global_importance.png")
    print("\nTop 10 global SHAP features:")
    print(imp.head(10).round(5).to_string(index=False))

    # Individual examples: 2 highest-prob flagged + 2 lowest-prob (representative).
    pos = test_s[test_s["at_risk"] == 1].nlargest(2, "prob")
    neg = test_s[test_s["at_risk"] == 0].nsmallest(2, "prob")
    examples = pd.concat([pos, neg])

    shap_df = pd.DataFrame(sv, columns=feature_cols, index=Xs.index)
    print("\n=== Worked individual SHAP examples (top features) ===")
    for _, ex in examples.iterrows():
        row = Xs.loc[ex.name]
        srow = shap_df.loc[ex.name].sort_values(key=np.abs, ascending=False)
        top5 = srow.head(5)
        print(f"\n--- student {int(ex['id_student'])} | {ex['module_presentation']} | {ex['checkpoint_label']}"
              f" | prob_at_risk={ex['prob']:.3f} | true_at_risk={int(ex['at_risk'])} | base_value={explainer.expected_value:.3f} ---")
        for fname, sval in top5.items():
            fval = row[fname]
            print(f"   {fname:45s} shap={sval:+.4f}  value={fval}")
        # waterfall plot
        exp = shap.Explanation(
            values=shap_df.loc[ex.name].values,
            base_values=explainer.expected_value,
            data=row.values,
            feature_names=feature_cols,
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp, max_display=8, show=False)
        ax.set_title(f"student {int(ex['id_student'])} {ex['checkpoint_label']} prob={ex['prob']:.3f}")
        fig.tight_layout(); fig.savefig(PLOTS / f"shap_student_{int(ex['id_student'])}_{ex['checkpoint_label']}.png", dpi=150); plt.close(fig)
    print("\nWrote individual waterfall plots to", PLOTS)


if __name__ == "__main__":
    main()
