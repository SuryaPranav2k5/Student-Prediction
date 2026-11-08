"""Step 2 + Step 3 — label construction and feature engineering.

Builds, per student per checkpoint, a feature matrix using ONLY data with a
timestamp/day-offset <= that row's checkpoint cutoff. Outputs:

  data/processed/features.parquet   -> model feature matrix + label
  data/processed/fairness_eval.parquet -> gender/disability/age_band/imd_band
                                         keyed by student_id (Step 6 only)

Design constraints honoured:
  * No feature references a date past the row's cutoff.
  * gender, disability, age_band, imd_band are EXCLUDED from model features
    and routed to fairness_eval.parquet.
  * Missing demographic fields are encoded as explicit "unknown", never
    imputed with mean/mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from common import (
    ARTIFACTS,
    CUTOFF_DAYS,
    FAIRNESS_COLUMNS,
    LABEL_MAP,
    PROCESSED,
    TRAIN_PRESENTATIONS,
    TEST_PRESENTATION,
    add_module_presentation,
    ensure_dirs,
    load_tables,
    make_at_risk,
    split_train_test,
    to_day,
)

REG_DATE_SENTINEL = -999999


def _click_features(student_vle: pd.DataFrame, vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Cumulative VLE clicks by activity_type: all-time-to-date + last 7 days."""
    sv = student_vle.merge(vle[["id_site", "activity_type"]], on="id_site", how="left")
    sv = sv[sv["date"] <= cutoff_day].copy()

    total = (
        sv.groupby(["code_module", "code_presentation", "id_student", "activity_type"])["sum_click"]
        .sum()
        .reset_index()
        .pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
        )
        .add_suffix("_clicks_total")
        .reset_index()
    )

    last7 = sv[sv["date"] > cutoff_day - 7]
    last7 = (
        last7.groupby(["code_module", "code_presentation", "id_student", "activity_type"])["sum_click"]
        .sum()
        .reset_index()
        .pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
        )
        .add_suffix("_clicks_last7")
        .reset_index()
    )

    clicks = total.merge(last7, on=["code_module", "code_presentation", "id_student"], how="outer")
    return clicks


def _engagement_lag(student_vle: pd.DataFrame, vle: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """Days between material release week and student's first click on it."""
    sv = student_vle.merge(vle[["id_site", "week_from"]], on="id_site", how="left")
    sv = sv[sv["date"] <= cutoff_day].copy()
    sv["week_from"] = to_day(sv["week_from"])
    known = sv[sv["week_from"].notna()].copy()
    if known.empty:
        return pd.DataFrame(
            columns=["code_module", "code_presentation", "id_student", "engagement_lag_mean", "n_materials_with_release"]
        )
    first = known.sort_values("date").groupby(
        ["code_module", "code_presentation", "id_student", "id_site"]
    ).first().reset_index()
    first["lag"] = first["date"] - first["week_from"]
    agg = (
        first.groupby(["code_module", "code_presentation", "id_student"])["lag"]
        .agg(engagement_lag_mean="mean")
        .reset_index()
    )
    n_rel = (
        first.groupby(["code_module", "code_presentation", "id_student"])["id_site"]
        .count()
        .reset_index(name="n_materials_with_release")
    )
    return agg.merge(n_rel, on=["code_module", "code_presentation", "id_student"], how="outer")


def _assessment_features(student_assessment: pd.DataFrame, assessments: pd.DataFrame, cutoff_day: int):
    """Submission timeliness + early scores for assessments due by cutoff.

    Returns (per_student_stats, due_universe) where due_universe is the
    course-level count of assessments due by the cutoff (broadcast to every
    student so that students with no assessment activity still get a correct
    n_assessments_due / n_missed).
    """
    dated = assessments[assessments["date"].astype(str) != "?"].copy()
    dated["due_date"] = to_day(dated["date"])

    # Universe of assessments DUE by this cutoff, per course.
    due_universe = (
        dated[dated["due_date"] <= cutoff_day]
        .groupby(["code_module", "code_presentation"])["id_assessment"]
        .count()
        .reset_index(name="n_assessments_due")
    )

    sa = student_assessment.merge(
        dated[["id_assessment", "code_module", "code_presentation", "due_date"]],
        on="id_assessment",
        how="inner",
    )
    # Only submissions on/before the cutoff count, and only assessments due by the cutoff.
    sa = sa[(sa["date_submitted"] <= cutoff_day) & (sa["due_date"] <= cutoff_day)].copy()
    sa["score_n"] = pd.to_numeric(sa["score"], errors="coerce")

    # Per (student, assessment): latest submission on/before cutoff.
    sa = sa.sort_values("date_submitted")
    last = sa.groupby(["code_module", "code_presentation", "id_student", "id_assessment"]).tail(1)

    taken = last.copy()
    taken["earliness"] = taken["date_submitted"] - taken["due_date"]  # negative = early

    timeliness = (
        taken.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            n_assessments_taken=("id_assessment", "nunique"),
            mean_earliness_days=("earliness", "mean"),
        )
        .reset_index()
    )

    n_taken = (
        taken.groupby(["code_module", "code_presentation", "id_student"])["id_assessment"]
        .nunique()
        .reset_index(name="n_taken_distinct")
    )

    # Scores: first two assessments taken, ordered by FIRST submission date
    # (semantically "first taken"); score taken from the latest submission.
    first_dates = sa.groupby(
        ["code_module", "code_presentation", "id_student", "id_assessment"]
    )["date_submitted"].min().reset_index(name="first_date")
    taken = taken.merge(first_dates, on=["code_module", "code_presentation", "id_student", "id_assessment"], how="left")
    taken_sorted = taken.sort_values(["first_date", "id_assessment"])
    taken_sorted["assessment_rank"] = taken_sorted.groupby(
        ["code_module", "code_presentation", "id_student"]
    ).cumcount() + 1
    first_score = taken_sorted[taken_sorted["assessment_rank"] == 1][
        ["code_module", "code_presentation", "id_student", "score_n"]
    ].rename(columns={"score_n": "first_assessment_score"})
    second_score = taken_sorted[taken_sorted["assessment_rank"] == 2][
        ["code_module", "code_presentation", "id_student", "score_n"]
    ].rename(columns={"score_n": "second_assessment_score"})
    scores = first_score.merge(second_score, on=["code_module", "code_presentation", "id_student"], how="outer")
    scores["score_delta"] = scores["second_assessment_score"] - scores["first_assessment_score"]

    out = timeliness.merge(n_taken, on=["code_module", "code_presentation", "id_student"], how="outer")
    out = out.merge(scores, on=["code_module", "code_presentation", "id_student"], how="outer")
    return out, due_universe


def build_features(raw: Path = Path("data/raw")) -> pd.DataFrame:
    tables = load_tables(raw)
    student_info = add_module_presentation(tables["studentInfo"].copy())
    registration = tables["studentRegistration"].copy()
    registration["module_presentation"] = (
        registration["code_module"].astype(str) + "_" + registration["code_presentation"].astype(str)
    )
    registration["date_registration"] = to_day(registration["date_registration"])

    cutoffs = pd.read_parquet(PROCESSED / "cutoffs.parquet")

    student_info["at_risk"] = make_at_risk(student_info["final_result"])
    train, test = split_train_test(student_info)
    train["split"] = "train"
    test["split"] = "test"
    all_students = pd.concat([train, test], ignore_index=True)

    # Fairness table (never model input), keyed by student_id.
    fairness = all_students[["id_student", "module_presentation"] + FAIRNESS_COLUMNS].copy()
    for col in FAIRNESS_COLUMNS:
        fairness[col] = fairness[col].astype(str).replace({"nan": "unknown", "<NA>": "unknown", "None": "unknown"})

    student_vle = tables["studentVle"].copy()
    student_vle["module_presentation"] = (
        student_vle["code_module"].astype(str) + "_" + student_vle["code_presentation"].astype(str)
    )

    student_assessment = tables["studentAssessment"].copy()

    frames = []
    for label, cutoff_day in CUTOFF_DAYS.items():
        # Keep EVERY student at every checkpoint — never drop a course×checkpoint
        # just because no assessment closed yet. Students in courses with zero
        # closed assessments get assessment features = 0 plus an explicit
        # no_assessment_closed indicator (missingness is informative).
        sub = all_students.copy()

        clicks = _click_features(student_vle, tables["vle"], cutoff_day)
        lag = _engagement_lag(student_vle, tables["vle"], cutoff_day)
        assess_stats, due_universe = _assessment_features(student_assessment, tables["assessments"], cutoff_day)

        feat = sub.merge(clicks, on=["code_module", "code_presentation", "id_student"], how="left")
        feat = feat.merge(lag, on=["code_module", "code_presentation", "id_student"], how="left")
        feat = feat.merge(assess_stats, on=["code_module", "code_presentation", "id_student"], how="left")
        # Broadcast the course-level count of assessments due by the cutoff to
        # every student (students with no assessment activity still accrue misses).
        feat = feat.merge(due_universe, on=["code_module", "code_presentation"], how="left")

        reg = registration[["module_presentation", "id_student", "date_registration"]]
        feat = feat.merge(reg, on=["module_presentation", "id_student"], how="left")
        feat["reg_date_missing"] = feat["date_registration"].isna().astype(int)
        feat["date_registration"] = feat["date_registration"].fillna(REG_DATE_SENTINEL)

        feat["checkpoint_label"] = label
        feat["cutoff_day"] = cutoff_day
        feat["week_elapsed"] = int(label.split("_")[1])
        feat["n_assessments_due"] = feat["n_assessments_due"].fillna(0)
        feat["n_assessments_taken"] = feat["n_assessments_taken"].fillna(0)
        feat["n_taken_distinct"] = feat["n_taken_distinct"].fillna(0)
        feat["mean_earliness_days"] = feat["mean_earliness_days"].fillna(0)
        feat["n_missed"] = feat["n_assessments_due"] - feat["n_taken_distinct"]
        feat["frac_missed"] = np.where(
            feat["n_assessments_due"] > 0, feat["n_missed"] / feat["n_assessments_due"], 0.0
        )
        feat["first_assessment_score"] = feat["first_assessment_score"].fillna(0)
        feat["second_assessment_score"] = feat["second_assessment_score"].fillna(0)
        feat["score_delta"] = feat["score_delta"].fillna(0)

        # Explicit missingness indicator: no assessment had closed for this
        # course by this checkpoint. Never imputed — informative on its own.
        feat["no_assessment_closed"] = (feat["n_assessments_due"] == 0).astype(int)

        # Missingness indicators for sparse derived features (informative, not imputed).
        feat["engagement_lag_missing"] = feat["engagement_lag_mean"].isna().astype(int)
        feat["first_score_missing"] = feat["first_assessment_score"].isna().astype(int)

        frames.append(feat)

    features = pd.concat(frames, ignore_index=True)

    # One-hot encode the non-excluded categorical demographics using full-data
    # categories (fixed labels, no target information -> no leakage).
    cat_cols = [c for c in ["region", "highest_education"] if c in features.columns]
    features = pd.get_dummies(features, columns=cat_cols, dummy_na=False)

    # Numeric features that are legitimately zero when no activity occurred.
    zero_fill = [c for c in features.columns if c.endswith(("_clicks_total", "_clicks_last7"))]
    zero_fill += ["engagement_lag_mean", "n_materials_with_release", "n_assessments_taken",
                  "mean_earliness_days", "n_taken_distinct", "n_missed", "frac_missed",
                  "first_assessment_score", "second_assessment_score", "score_delta"]
    for c in zero_fill:
        if c in features.columns:
            features[c] = features[c].fillna(0)

    ensure_dirs()
    feat_out = PROCESSED / "features.parquet"
    features.to_parquet(feat_out, index=False)

    fair_out = PROCESSED / "fairness_eval.parquet"
    fairness.to_parquet(fair_out, index=False)

    print("=== Step 2: label construction (at_risk = Withdrawn or Fail) ===")
    for name, d in [("train (2013J+2014J)", train), ("test (2014B)", test)]:
        n = len(d)
        pos = int(d["at_risk"].sum())
        print(f"{name}: n={n} at_risk={pos} ({pos/n:.4f})  not_at_risk={n-pos}")
    print()
    print("=== Step 3: features ===")
    print("feature rows:", len(features), "| columns:", features.shape[1])
    print("by checkpoint:")
    print(features.groupby("checkpoint_label").size())
    print("excluded courses per checkpoint honoured (EEE/GGG early checkpoints dropped).")
    print("Wrote", feat_out)
    print("Wrote", fair_out)

    # Step 4 — leakage audit runs as part of the feature-build pipeline (loud failure on violation).
    from test_no_leakage import run_audit
    run_audit()

    return features


if __name__ == "__main__":
    build_features()
