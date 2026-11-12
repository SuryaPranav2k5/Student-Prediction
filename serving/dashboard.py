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

# The score is decision SUPPORT, never a decision.
HUMAN_REVIEW_NOTE = (
    "Advisory only. Every flag must be reviewed by a human advisor before any "
    "action, and must never trigger automated grading, punishment or admission "
    "decisions."
)


def risk_band(prob: float, threshold: float) -> str:
    """Advisory band anchored to the validation-selected operating threshold."""
    if prob >= threshold:
        return "High"
    if prob >= 0.5 * threshold:
        return "Moderate"
    return "Low"


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
    st.caption("Binary at-risk (Withdrawn/Fail) prediction at week 4 / 8 / 12 checkpoints. Held-out test: 2014J.")
    st.info(HUMAN_REVIEW_NOTE)

    if model is None:
        st.error("No model artifact found. Run `python models/train.py` first.")
        return

    # ---- Headline: per-checkpoint PR-AUC comparison ----
    st.subheader("Test-set model comparison by checkpoint")
    rows = comparison["rows"]
    comp = pd.DataFrame(rows)
    test_comp = comp[comp["split"] == "test"]
    ckpts = ["week_4", "week_8", "week_12"]
    # Build grouped values: pooled LightGBM, RandomForest, and the separate
    # per-checkpoint LightGBM (which only has a value for its own checkpoint).
    def _pr(model: str, cp: str) -> float:
        s = test_comp[(test_comp["model"] == model) & (test_comp["checkpoint"] == cp)]["pr_auc"]
        return float(s.iloc[0]) if len(s) else np.nan

    series = [
        ("Pooled LightGBM", "#1f77b4", lambda cp: _pr("lgb_pooled", cp)),
        ("RandomForest", "#2ca02c", lambda cp: _pr("rf_pooled (random forest)", cp)),
        ("Separate per-checkpoint", "#ff7f0e", lambda cp: _pr(f"lgb_{cp} (separate)", cp)),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(ckpts)); w = 0.26
    for i, (name, color, value_of) in enumerate(series):
        vals = [value_of(cp) for cp in ckpts]
        offset = (i - 1) * w
        ax.bar(x + offset, vals, w, label=name, color=color, alpha=0.9)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi + offset, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Week 4", "Week 8", "Week 12"])
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig, use_container_width=True)

    # Baseline LR row — read from comparison.json so it can never go stale.
    lr_rows = test_comp[test_comp["model"] == "lr_pooled (baseline)"]
    if len(lr_rows):
        lr_map = dict(zip(lr_rows["checkpoint"], lr_rows["pr_auc"]))
        lr_txt = ", ".join(
            f"{c.replace('week_', 'week ')} = {lr_map.get(c, float('nan')):.3f}" for c in ckpts
        )
        st.caption(f"Baseline LogisticRegression PR-AUC on test data: {lr_txt}. "
                   "The pooled LightGBM is the production model.")

    ci_rows = metrics.get("checkpoints", [])
    ci_tbl = pd.DataFrame(ci_rows)
    if not ci_tbl.empty:
        st.write("Pooled model PR-AUC with bootstrap 95% CI (held-out test 2014J):")
        st.dataframe(ci_tbl[["checkpoint", "pr_auc", "pr_auc_ci_lo", "pr_auc_ci_hi", "n"]].round(4),
                     use_container_width=True, hide_index=True)

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
    cp_thr = next(r["operating_threshold"] for r in metrics["checkpoints"] if r["checkpoint"] == cp_choice)
    st.write(f"Advisory risk band: **{risk_band(prob, cp_thr)}** "
             f"(anchored to the {cp_choice} operating threshold {cp_thr:.3f}).")

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

    st.markdown("---")

    # ---- Live what-if: type feature values and watch the score move ----
    st.subheader("Try it yourself: enter feature values and get a live prediction")
    st.caption("Any feature you leave at 0 is treated as absent. This is the same model and the same "
               "SHAP explainer as the FastAPI /predict endpoint — the form is just a friendlier front end.")

    # Defaults come from the currently selected student, so the form starts from a real case.
    default_row = X_one.iloc[0]
    KEY_FEATURES = [
        ("n_missed", "Assessments missed (count)", 0.0, None, 1.0),
        ("n_assessments_due", "Assessments due by cutoff", 0.0, None, 1.0),
        ("n_assessments_taken", "Assessments taken", 0.0, None, 1.0),
        ("frac_missed", "Fraction of assessments missed", 0.0, 1.0, 0.05),
        ("first_assessment_score", "First assessment score (0-100)", 0.0, 100.0, 1.0),
        ("second_assessment_score", "Second assessment score (0-100)", 0.0, 100.0, 1.0),
        ("mean_earliness_days", "Mean days early/late (negative = early)", -60.0, 60.0, 1.0),
        ("homepage_clicks_total", "VLE homepage clicks (all time)", 0.0, None, 1.0),
        ("homepage_clicks_last7", "VLE homepage clicks (last 7 days)", 0.0, None, 1.0),
        ("studied_credits", "Studied credits", 0.0, None, 1.0),
        ("num_of_prev_attempts", "Previous attempts", 0.0, None, 1.0),
        ("date_registration", "Registration day (0 = course start)", -200.0, 0.0, 1.0),
        ("no_assessment_closed", "No assessment closed yet (0/1)", 0.0, 1.0, 1.0),
    ]
    preset = st.radio("Preset", ["Selected student", "High-risk example", "Low-risk example"],
                      horizontal=True)

    hi = {"n_missed": 5.0, "n_assessments_due": 5.0, "n_assessments_taken": 0.0, "frac_missed": 1.0,
          "first_assessment_score": 0.0, "second_assessment_score": 0.0, "mean_earliness_days": 0.0,
          "homepage_clicks_total": 0.0, "homepage_clicks_last7": 0.0, "studied_credits": 240.0,
          "num_of_prev_attempts": 1.0, "date_registration": -120.0, "no_assessment_closed": 0.0}
    lo = {"n_missed": 0.0, "n_assessments_due": 3.0, "n_assessments_taken": 3.0, "frac_missed": 0.0,
          "first_assessment_score": 92.0, "second_assessment_score": 88.0, "mean_earliness_days": -4.0,
          "homepage_clicks_total": 250.0, "homepage_clicks_last7": 40.0, "studied_credits": 60.0,
          "num_of_prev_attempts": 0.0, "date_registration": -20.0, "no_assessment_closed": 0.0}

    base = default_row if preset == "Selected student" else (hi if preset == "High-risk example" else lo)

    inputs = {}
    cols = st.columns(3)
    # Widget keys are scoped to the preset so switching presets re-seeds the
    # fields with the new defaults. Without this, Streamlit's keyed widget state
    # persists across the rerun and the inputs silently keep their old values.
    for i, (key, label, mn, mx, step) in enumerate(KEY_FEATURES):
        # `base` is a pandas Series for "Selected student" or a dict for the examples.
        if isinstance(base, dict):
            val = float(base.get(key, 0.0))
        else:
            val = float(base[key]) if key in base.index else 0.0
        wkey = f"in_{preset}_{key}"
        with cols[i % 3]:
            if key == "no_assessment_closed":
                inputs[key] = 1.0 if st.checkbox(label, value=bool(val), key=wkey) else 0.0
            else:
                inputs[key] = st.number_input(label, value=val, min_value=mn, max_value=mx,
                                              step=step, key=wkey)

    # Everything the form does not expose: for the named examples we start from
    # zeros so the example is self-contained; for "Selected student" we carry the
    # student's full profile, otherwise the example features are drowned out.
    fill_rest = st.checkbox("Start from ALL of the selected student's other features "
                            "(region, education, engagement lags...) instead of zeros",
                           value=(preset == "Selected student"))

    if fill_rest:
        row_vals = [float(default_row[c]) for c in feature_cols]
    else:
        row_vals = [0.0 for _ in feature_cols]
    for k, v in inputs.items():
        row_vals[feature_cols.index(k)] = float(v)

    X_new = pd.DataFrame([row_vals], columns=feature_cols)
    X_new = X_new.apply(pd.to_numeric, errors="coerce").fillna(0)
    p_new = float(model.predict_proba(X_new)[:, 1][0])
    threshold_row = next(
        row for row in metrics["checkpoints"] if row["checkpoint"] == cp_choice
    )
    p_thr = threshold_row["operating_threshold"]

    m1, m2, m3 = st.columns(3)
    m1.metric("P(at_risk)", f"{p_new:.3f}")
    m2.metric(f"Flag at operating threshold ({p_thr:.3f})?", "YES" if p_new >= p_thr else "no")
    m3.metric("Advisory risk band", risk_band(p_new, p_thr))

    sv_new = explainer.shap_values(X_new)
    if isinstance(sv_new, list):
        sv_new = sv_new[0]
    shap_new = sv_new[0]
    order_new = np.argsort(-np.abs(shap_new))[:10]
    st.write("Top SHAP drivers for YOUR input (positive = pushes toward at-risk):")
    st.dataframe(pd.DataFrame({
        "feature": [feature_cols[i] for i in order_new],
        "shap": [float(shap_new[i]) for i in order_new],
        "value": [float(X_new.iloc[0, i]) for i in order_new],
    }).round(4), use_container_width=True, hide_index=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(values=shap_new, base_values=expected_value,
                         data=X_new.iloc[0].values, feature_names=feature_cols),
        max_display=10, show=False,
    )
    st.pyplot(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
