"""EDA and data-quality profiling (Step 2).

Profiles the raw OULAD tables and the built feature matrix: volumes, class
balance by split, missing values, duplicate keys, date ranges, the heavy tail
in VLE clicks (why tree models are used instead of scaling) and residual
missingness in the feature matrix.

Usage: python eval/eda_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import PROCESSED, RAW


def main():
    info = pd.read_csv(RAW / "studentInfo.csv")
    assess = pd.read_csv(RAW / "assessments.csv")
    sa = pd.read_csv(RAW / "studentAssessment.csv")
    svle = pd.read_csv(RAW / "studentVle.csv", usecols=["id_student", "date", "sum_click"])
    reg = pd.read_csv(RAW / "studentRegistration.csv")

    print("=== Table volumes ===")
    for name, df in [("studentInfo", info), ("studentRegistration", reg),
                     ("studentAssessment", sa), ("assessments", assess),
                     ("studentVle", svle)]:
        print(f"  {name:22s} rows={len(df):>9,}  cols={df.shape[1]:>2}")

    print("\n=== Duplicate keys / integrity ===")
    key = ["id_student", "code_module", "code_presentation"]
    print(f"  studentInfo rows per student-presentation (max) = "
          f"{info.groupby(key).size().max()}")
    print(f"  students with >1 final_result row              = "
          f"{(info.groupby(key)['final_result'].nunique() > 1).sum()}")
    print(f"  studentVle rows with negative date             = {(svle['date'] < 0).mean():.2%}")

    print("\n=== Class balance (target) ===")
    print("  final_result:", info["final_result"].value_counts().to_dict())
    at_risk = info["final_result"].isin(["Withdrawn", "Fail"])
    print(f"  at_risk = Withdrawn+Fail: {at_risk.mean():.2%} of students "
          f"(near-balanced, NOT a rare-event problem)")

    print("\n=== Missing values (raw) ===")
    for col in ["imd_band", "gender", "disability", "age_band"]:
        miss = info[col].isna() | (info[col].astype(str) == "?")
        print(f"  studentInfo.{col:12s} missing/unknown = {miss.mean():.2%}")
    print(f"  studentAssessment.score  missing = {sa['score'].isna().mean():.2%} "
          f"({int(sa['score'].isna().sum())} rows)")
    print("  -> missing demographics are kept as an explicit 'unknown' category; "
          "they are never model inputs.")

    print("\n=== Date ranges (day offsets from presentation start) ===")
    due = pd.to_numeric(assess["date"], errors="coerce")
    print(f"  studentVle.date  min={svle['date'].min()}  max={svle['date'].max()} "
          f"(negative = pre-course access)")
    print(f"  assessments due  min={due.min():.0f}  max={due.max():.0f}  "
          f"missing('?')={int(due.isna().sum())}")

    print("\n=== Heavy tail: total VLE clicks per student-presentation ===")
    clicks = svle.groupby(["id_student"])["sum_click"].sum()
    q = clicks.quantile([0.5, 0.9, 0.99, 0.999]).round(0)
    print(f"  median={q[0.5]:.0f}  p90={q[0.9]:.0f}  p99={q[0.99]:.0f}  "
          f"max={clicks.max():.0f}  (max/median = {clicks.max()/max(q[0.5],1):.0f}x)")
    print("  -> click counts span 4+ orders of magnitude next to 0-100 scores. "
          "Left untouched (they are real behaviour, not corruption); trees are "
          "scale-invariant and the LR baseline uses StandardScaler.")

    print("\n=== Built feature matrix (data quality) ===")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    print(f"  rows={len(feats):,}  cols={feats.shape[1]}  "
          f"students={feats['id_student'].nunique():,}  "
          f"checkpoints={feats['checkpoint_label'].nunique()}")
    print("  rows by split:", feats["split"].value_counts().to_dict())
    print(f"  NaN cells in feature matrix = {int(feats.isna().sum().sum())}")
    print(f"  rows with no assessment closed by cutoff = "
          f"{(feats['no_assessment_closed'] == 1).sum():,} "
          f"({(feats['no_assessment_closed'] == 1).mean():.1%})")
    print(f"  at_risk rate in matrix = {feats['at_risk'].mean():.4f}")


if __name__ == "__main__":
    main()
