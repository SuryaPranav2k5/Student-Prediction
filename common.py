"""Shared helpers for loading and preparing the OULAD dataset.

All date fields in OULAD are day-offsets relative to the start of the
module presentation (day 0 = first day of the presentation). There are no
absolute calendar dates, so every checkpoint cutoff is expressed as a day
offset and every feature derivation is anchored to that offset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
ARTIFACTS = Path("artifacts")
MODELS = ARTIFACTS / "models"

# Checkpoint cutoffs, expressed as day-offsets from module start.
# Fixed calendar cutoffs agreed with the user (see decision log):
#   week 4 -> day 28, week 8 -> day 56, week 12 -> day 84.
# Rows for courses where zero assessments have closed by a cutoff are
# excluded from that checkpoint's feature set (see Step 1 / cutoff build).
CUTOFF_DAYS = {"week_4": 28, "week_8": 56, "week_12": 84}

TRAIN_PRESENTATIONS = ["2013J", "2014J"]
TEST_PRESENTATION = "2014B"

LABEL_MAP = {
    "Pass": 0,
    "Distinction": 0,
    "Withdrawn": 1,
    "Fail": 1,
}

# Demographics routed to fairness_eval only, never used as model inputs.
FAIRNESS_COLUMNS = ["gender", "disability", "age_band", "imd_band"]


def ensure_dirs() -> None:
    for d in (PROCESSED, ARTIFACTS, MODELS):
        d.mkdir(parents=True, exist_ok=True)


def load_tables(raw: Path = RAW) -> Dict[str, pd.DataFrame]:
    """Load the seven OULAD CSV tables."""
    return {
        "studentInfo": pd.read_csv(raw / "studentInfo.csv"),
        "studentRegistration": pd.read_csv(raw / "studentRegistration.csv"),
        "studentAssessment": pd.read_csv(raw / "studentAssessment.csv"),
        "assessments": pd.read_csv(raw / "assessments.csv"),
        "vle": pd.read_csv(raw / "vle.csv"),
        "studentVle": pd.read_csv(raw / "studentVle.csv"),
        "courses": pd.read_csv(raw / "courses.csv"),
    }


def to_day(series: pd.Series) -> pd.Series:
    """Coerce a day-offset column (may contain '?' or non-numeric) to Int64."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def add_module_presentation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["module_presentation"] = df["code_module"].astype(str) + "_" + df["code_presentation"].astype(str)
    return df


def make_at_risk(final_result: pd.Series) -> pd.Series:
    return final_result.isin(["Withdrawn", "Fail"]).astype(int)


def split_train_test(student_info: pd.DataFrame):
    train = student_info[student_info["code_presentation"].isin(TRAIN_PRESENTATIONS)].copy()
    test = student_info[student_info["code_presentation"] == TEST_PRESENTATION].copy()
    return train, test
