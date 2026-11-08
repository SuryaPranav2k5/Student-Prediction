"""Step 4 — leakage audit.

Independently re-derives every date-sensitive feature from the raw OULAD tables,
strictly filtering each source record to day-offsets <= the row's checkpoint
cutoff, and asserts element-wise equality with the values stored in
features.parquet for the FULL feature set (not a sample).

Raises AssertionError (loud failure, not a warning) on any violation. Run with
pytest or directly:  python features/test_no_leakage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from common import CUTOFF_DAYS, PROCESSED, add_module_presentation, load_tables, to_day

# Feature columns that are derived from time-bounded sources and must be audited.
CLICK_SUFFIXES = ("_clicks_total", "_clicks_last7")
ASSESS_COLS = [
    "n_assessments_taken", "mean_earliness_days", "n_missed", "n_assessments_due",
    "first_assessment_score", "second_assessment_score", "score_delta", "frac_missed",
]
LAG_COLS = ["engagement_lag_mean", "n_materials_with_release"]


def _assert_close(stored: pd.Series, expected: pd.Series, col: str, ctx: str) -> None:
    stored = stored.reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    assert len(stored) == len(expected), f"{ctx}: length mismatch for {col}"
    mask = ~(np.isclose(stored.astype(float), expected.astype(float), equal_nan=True))
    if mask.any():
        bad = stored[mask].head(5)
        exp = expected[mask].head(5)
        raise AssertionError(
            f"LEAKAGE/{ctx} mismatch for {col} ({int(mask.sum())} rows). "
            f"First mismatches stored={bad.tolist()} expected={exp.tolist()}"
        )


def _compare_col(sub: pd.DataFrame, recomputed: pd.DataFrame, col: str, ctx: str, join_keys) -> None:
    m = sub[list(join_keys) + [col]].merge(
        recomputed[list(join_keys) + [col]], on=list(join_keys), suffixes=("_s", "_r")
    )
    assert len(m) == len(sub), f"{ctx}: row count changed on merge for {col}"
    _assert_close(m[f"{col}_s"], m[f"{col}_r"], col, ctx)


def run_audit() -> None:
    tables = load_tables()
    tables["studentInfo"] = add_module_presentation(tables["studentInfo"].copy())
    tables["studentRegistration"]["module_presentation"] = (
        tables["studentRegistration"]["code_module"].astype(str)
        + "_"
        + tables["studentRegistration"]["code_presentation"].astype(str)
    )
    tables["studentVle"]["module_presentation"] = (
        tables["studentVle"]["code_module"].astype(str) + "_" + tables["studentVle"]["code_presentation"].astype(str)
    )

    features = pd.read_parquet(PROCESSED / "features.parquet")
    n_rows = len(features)
    print(f"Leakage audit: auditing all {n_rows} feature rows.")

    assessments = tables["assessments"].copy()
    assessments["due_date"] = to_day(assessments["date"])
    dated = assessments[assessments["due_date"].notna()].copy()

    student_vle = tables["studentVle"].merge(
        tables["vle"][["id_site", "activity_type", "week_from"]], on="id_site", how="left"
    )

    for label, cutoff_day in CUTOFF_DAYS.items():
        ctx = f"{label} (cutoff day {cutoff_day})"
        sub = features[features["checkpoint_label"] == label].copy()
        if sub.empty:
            continue

        # ---- Clicks: all-time-to-date + last 7 days, independently recomputed ----
        sv = student_vle[student_vle["date"] <= cutoff_day].copy()
        click_frames = []
        for at, grp in sv.groupby("activity_type"):
            agg = grp.groupby(["module_presentation", "id_student"])["sum_click"].sum().reset_index()
            agg = agg.rename(columns={"sum_click": f"{at}_clicks_total"})
            click_frames.append(agg)
        clicks_total = click_frames[0]
        for df in click_frames[1:]:
            clicks_total = clicks_total.merge(df, on=["module_presentation", "id_student"], how="outer")

        sv7 = sv[sv["date"] > cutoff_day - 7]
        click7_frames = []
        for at, grp in sv7.groupby("activity_type"):
            agg = grp.groupby(["module_presentation", "id_student"])["sum_click"].sum().reset_index()
            agg = agg.rename(columns={"sum_click": f"{at}_clicks_last7"})
            click7_frames.append(agg)
        clicks_last7 = click7_frames[0]
        for df in click7_frames[1:]:
            clicks_last7 = clicks_last7.merge(df, on=["module_presentation", "id_student"], how="outer")

        clicks = clicks_total.merge(clicks_last7, on=["module_presentation", "id_student"], how="outer")

        # ---- Assessments: only assessments DUE by cutoff, submissions by cutoff ----
        sa = tables["studentAssessment"].merge(
            dated[["id_assessment", "code_module", "code_presentation", "due_date"]],
            on="id_assessment",
            how="inner",
        )
        sa = sa[(sa["date_submitted"] <= cutoff_day) & (sa["due_date"] <= cutoff_day)].copy()
        sa["score_n"] = pd.to_numeric(sa["score"], errors="coerce")

        # Universe of assessments due by cutoff, per course.
        due_universe = (
            dated[dated["due_date"] <= cutoff_day]
            .groupby(["code_module", "code_presentation"])["id_assessment"]
            .count()
            .reset_index(name="n_assessments_due")
        )
        due_universe["module_presentation"] = (
            due_universe["code_module"].astype(str) + "_" + due_universe["code_presentation"].astype(str)
        )

        # Latest submission per (student, assessment).
        sa = sa.sort_values("date_submitted")
        last = sa.groupby(["code_module", "code_presentation", "id_student", "id_assessment"]).tail(1).copy()
        last["earliness"] = last["date_submitted"] - last["due_date"]

        assess_agg = (
            last.groupby(["code_module", "code_presentation", "id_student"])
            .agg(
                n_assessments_taken=("id_assessment", "nunique"),
                mean_earliness_days=("earliness", "mean"),
            )
            .reset_index()
        )
        n_taken = (
            last.groupby(["code_module", "code_presentation", "id_student"])["id_assessment"]
            .nunique()
            .reset_index(name="n_taken_distinct")
        )

        # First two scores ordered by first submission date.
        first_dates = (
            sa.groupby(["code_module", "code_presentation", "id_student", "id_assessment"])["date_submitted"]
            .min()
            .reset_index(name="first_date")
        )
        last = last.merge(first_dates, on=["code_module", "code_presentation", "id_student", "id_assessment"], how="left")
        last_sorted = last.sort_values(["first_date", "id_assessment"])
        last_sorted["assessment_rank"] = last_sorted.groupby(
            ["code_module", "code_presentation", "id_student"]
        ).cumcount() + 1
        first_score = last_sorted[last_sorted["assessment_rank"] == 1][
            ["code_module", "code_presentation", "id_student", "score_n"]
        ].rename(columns={"score_n": "first_assessment_score"})
        second_score = last_sorted[last_sorted["assessment_rank"] == 2][
            ["code_module", "code_presentation", "id_student", "score_n"]
        ].rename(columns={"score_n": "second_assessment_score"})
        scores = first_score.merge(second_score, on=["code_module", "code_presentation", "id_student"], how="outer")
        scores["score_delta"] = scores["second_assessment_score"] - scores["first_assessment_score"]

        assess = assess_agg.merge(n_taken, on=["code_module", "code_presentation", "id_student"], how="outer")
        assess = assess.merge(scores, on=["code_module", "code_presentation", "id_student"], how="outer")
        assess["module_presentation"] = (
            assess["code_module"].astype(str) + "_" + assess["code_presentation"].astype(str)
        )
        # n_assessments_due / n_missed / frac_missed / no_assessment_closed are
        # computed in recomputed (below) using the course-level due count broadcast
        # to every student, so zero-assessment students are covered too.

        # ---- Engagement lag ----
        svl = student_vle[student_vle["date"] <= cutoff_day].copy()
        svl["week_from"] = to_day(svl["week_from"])
        known = svl[svl["week_from"].notna()].copy()
        lag_rows = []
        if not known.empty:
            first = (
                known.sort_values("date")
                .groupby(["module_presentation", "id_student", "id_site"])
                .first()
                .reset_index()
            )
            first["lag"] = first["date"] - first["week_from"]
            lag = first.groupby(["module_presentation", "id_student"])["lag"].mean().reset_index(name="engagement_lag_mean")
            n_rel = (
                first.groupby(["module_presentation", "id_student"])["id_site"]
                .count()
                .reset_index(name="n_materials_with_release")
            )
            lag_rows = [lag, n_rel]

        # ---- Join recomputed features to stored rows and compare ----
        join_keys = ["module_presentation", "id_student"]
        recomputed = sub[join_keys].copy()
        recomputed = recomputed.merge(clicks, on=join_keys, how="left")
        recomputed = recomputed.merge(assess, on=join_keys, how="left")
        # Broadcast course-level assessments-due-by-cutoff to every student.
        recomputed = recomputed.merge(
            due_universe[["module_presentation", "n_assessments_due"]], on="module_presentation", how="left"
        )
        recomputed["n_assessments_due"] = recomputed["n_assessments_due"].fillna(0)
        recomputed["n_taken_distinct"] = recomputed["n_taken_distinct"].fillna(0)
        recomputed["n_missed"] = recomputed["n_assessments_due"] - recomputed["n_taken_distinct"]
        recomputed["frac_missed"] = np.where(
            recomputed["n_assessments_due"] > 0,
            recomputed["n_missed"] / recomputed["n_assessments_due"],
            0.0,
        )
        recomputed["no_assessment_closed"] = (recomputed["n_assessments_due"] == 0).astype(int)

        if lag_rows:
            lag_all = lag_rows[0]
            for df in lag_rows[1:]:
                lag_all = lag_all.merge(df, on=join_keys, how="outer")
            recomputed = recomputed.merge(lag_all, on=join_keys, how="left")

        # Only numeric date-sensitive columns are compared element-wise.
        compare_cols = []
        for col in features.columns:
            if col in recomputed.columns and pd.api.types.is_numeric_dtype(features[col]):
                compare_cols.append(col)
        compare_cols = sorted(set(compare_cols) - {"id_student", "cutoff_day", "week_elapsed", "at_risk", "date_registration"})

        # Build features zero-fills numeric date-sensitive columns; mirror that
        # convention in the recomputed frame before comparing.
        recomputed[compare_cols] = recomputed[compare_cols].fillna(0)

        for col in compare_cols:
            _compare_col(sub, recomputed, col, ctx, join_keys)

        # no_assessment_closed must equal (n_assessments_due == 0)
        if "no_assessment_closed" in features.columns:
            _compare_col(sub, recomputed, "no_assessment_closed", ctx, join_keys)

        # registration date is static pre-course info
        reg = tables["studentRegistration"]
        reg_sub = reg[reg["module_presentation"].isin(sub["module_presentation"].unique())].copy()
        reg_map = reg_sub.set_index(["module_presentation", "id_student"])["date_registration"]
        reg_date = sub.set_index(["module_presentation", "id_student"]).index.map(reg_map)
        reg_date = pd.Series(pd.to_numeric(reg_date, errors="coerce"), index=sub.index)
        stored_reg = sub["date_registration"].reset_index(drop=True)
        # Only compare rows where registration is known in both.
        known = reg_date.notna() & stored_reg.notna()
        if known.any():
            _assert_close(
                stored_reg[known].reset_index(drop=True),
                reg_date[known].reset_index(drop=True),
                "date_registration",
                ctx,
            )

        print(f"  {ctx}: {len(sub)} rows audited — PASSED")

    print(f"\nFull leakage audit PASSED — all {n_rows} rows derived exclusively from pre-cutoff data.")


def test_leakage_audit():
    run_audit()


if __name__ == "__main__":
    run_audit()
