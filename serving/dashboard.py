"""Step 8 — Streamlit dashboard.

Headline: per-checkpoint PR-AUC comparison (Step 5).
Cohort-level risk distribution, per-student drill-down with SHAP explanation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt

from common import MODELS, PROCESSED

st.set_page_config(page_title="OULAD At-Risk Prediction", layout="wide")

PLOTS = Path("artifacts/plots")


@st.cache_resource
def load_model():
    """Prefer the fixed production artifact; fall back to newest timestamped."""
    prod = MODELS / "lgb_pooled_production.joblib"
    if prod.exists():
        model_path = prod
        ts = "production"
    else:
        cands = sorted(MODELS.glob("lgb_pooled_*.joblib"))
        if not cands:
            return None, [], None, 0.0
        model_path = cands[-1]
        ts = model_path.stem.replace("lgb_pooled_", "")
    feat_path = MODELS / f"lgb_pooled_{ts}_features.json"
    with open(feat_path) as fh:
        feature_cols = json.load(fh)
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    ev = explainer.expected_value
    ev = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
    return model, feature_cols, explainer, ev


@st.cache_data
def load_data():
    features = pd.read_parquet(PROCESSED / "features.parquet")
    test = features[features["split"] == "test"].copy()
    metrics = json.load(open(Path("artifacts/metrics.json")))
    comparison = json.load(open(Path("artifacts/comparison.json")))
    return test, metrics, comparison


def _to_frame(model, feature_cols, features_dict):
    row = {c: 0 for c in feature_cols}
    for k, v in features_dict.items():
        if k in row:
            row[k] = v
    df = pd.DataFrame([row], columns=feature_cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)


def main():
    model, feature_cols, explainer, expected_value = load_model()
    test, metrics, comparison = load_data()

    st.title("OULAD Student At-Risk Prediction")
    st.caption("Binary at-risk (Withdrawn/Fail) prediction at week 4 / 8 / 12 checkpoints. Held-out test: 2014B.")

    if model is None:
        st.error("No model artifact found. Run `python models/train.py` first.")
        return

    # ---- Headline: per-checkpoint PR-AUC comparison ----
    st.subheader("Headline: per-checkpoint PR-AUC (Step 5 comparison)")
    rows = comparison["rows"]
    comp = pd.DataFrame(rows)
    pivot = comp.pivot(index="model", columns="checkpoint", values="pr_auc")
    ckpts = ["week_4", "week_8", "week_12"]
    plot_df = pivot.reindex(index=["lgb_pooled (baseline)"] if False else None)
    # Build grouped values: pooled + separate where available
    chart_rows = []
    for cp in ckpts:
        pooled = comp[(comp["model"] == "lgb_pooled") & (comp["checkpoint"] == cp)]["pr_auc"]
        sep = comp[comp["model"] == f"lgb_{cp} (separate)"]["pr_auc"]
        chart_rows.append({"checkpoint": cp, "model": "Pooled LightGBM",
                           "pr_auc": float(pooled.iloc[0]) if len(pooled) else np.nan})
        if len(sep):
            chart_rows.append({"checkpoint": cp, "model": f"Separate ({cp})",
                               "pr_auc": float(sep.iloc[0])})
    chart = pd.DataFrame(chart_rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ckpts)); w = 0.35
    pooled_vals = [chart[(chart.checkpoint == cp) & (chart.model == "Pooled LightGBM")]["pr_auc"].iloc[0] for cp in ckpts]
    sep_vals = [chart[(chart.checkpoint == cp) & (chart.model == f"Separate ({cp})")]["pr_auc"].iloc[0]
                if len(chart[(chart.checkpoint == cp) & (chart.model == f"Separate ({cp})")]) else np.nan
                for cp in ckpts]
    ax.bar(x - w / 2, pooled_vals, w, label="Pooled LightGBM", color="#1f77b4")
    ax.bar(x + w / 2, sep_vals, w, label="Separate per-checkpoint", color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["Week 4", "Week 8", "Week 12"])
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(pooled_vals):
        ax.text(i - w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(sep_vals):
        if not np.isnan(v):
            ax.text(i + w / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    st.pyplot(fig, use_container_width=True)
    ci_rows = metrics.get("checkpoints", [])
    ci_tbl = pd.DataFrame(ci_rows)
    if not ci_tbl.empty:
        st.write("Pooled model PR-AUC with bootstrap 95% CI (held-out test 2014B):")
        st.dataframe(ci_tbl[["checkpoint", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi", "n"]].round(4),
                     use_container_width=True, hide_index=True)
    st.caption("Baseline LogisticRegression PR-AUC: week 4 = 0.870, week 8 = 0.903, week 12 = 0.906. "
               "Pooled and separate LightGBM are statistically indistinguishable (differences < 0.002); "
               "the pooled model is kept as the production artifact.")

    st.markdown("---")

    # ---- Cohort risk distribution ----
    st.subheader("Cohort-level risk distribution (held-out test)")
    cp_choice = st.selectbox("Checkpoint", ckpts, index=2)
    te = test[test["checkpoint_label"] == cp_choice].copy()
    X = te[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    probs = model.predict_proba(X)[:, 1]
    te["prob"] = probs

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs, bins=30, color="#1f77b4", alpha=0.8, edgecolor="white")
    ax.axvline(probs.mean(), color="red", linestyle="--", label=f"mean={probs.mean():.3f}")
    ax.set_xlabel("P(at_risk)"); ax.set_ylabel("Number of students")
    ax.set_title(f"Risk-score distribution at {cp_choice}")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    st.write(f"n={len(te)}, at-risk (true)={int(te['at_risk'].sum())} ({te['at_risk'].mean():.1%}), "
             f"mean predicted P(at_risk)={probs.mean():.3f}")

    st.markdown("---")

    # ---- Per-student drill-down ----
    st.subheader("Per-student drill-down with SHAP explanation")
    stu_ids = te["id_student"].tolist()
    stu = st.selectbox("Select student", stu_ids, index=0)
    srow = te[te["id_student"] == stu].iloc[0]
    X_one = srow[feature_cols].to_frame().T
    for c in X_one.columns:
        X_one[c] = pd.to_numeric(X_one[c], errors="coerce")
    X_one = X_one.fillna(0)
    prob = float(model.predict_proba(X_one)[:, 1][0])
    true_label = srow["final_result"]
    true_at_risk = int(srow["at_risk"])

    c1, c2, c3 = st.columns(3)
    c1.metric("P(at_risk)", f"{prob:.3f}")
    c2.metric("True label", true_label)
    c3.metric("True at_risk", f"{true_at_risk} (1=Withdrawn/Fail)")

    sv = explainer.shap_values(X_one)
    if isinstance(sv, list):
        sv = sv[0]
    shap_row = sv[0]
    order = np.argsort(-np.abs(shap_row))[:10]
    shap_df = pd.DataFrame({
        "feature": [feature_cols[i] for i in order],
        "shap": [float(shap_row[i]) for i in order],
        "value": [float(X_one.iloc[0, i]) for i in order],
    }).sort_values("shap", ascending=False)

    st.write("Top SHAP drivers (positive = pushes toward at-risk):")
    st.dataframe(shap_df.round(4), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    exp = shap.Explanation(values=shap_row, base_values=expected_value,
                           data=X_one.iloc[0].values, feature_names=feature_cols)
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(fig, use_container_width=True)


if __name__ == "__main__":
    main()
