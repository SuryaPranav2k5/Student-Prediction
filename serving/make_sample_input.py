"""Export a real feature vector to JSON so the API can be tested by hand.

Writes two files:
  artifacts/sample_input.json        -> one real student's full 74-feature vector
  artifacts/sample_input_min.json    -> minimal partial payload (API fills the rest with 0)

Usage: python serving/make_sample_input.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import ARTIFACTS, MODELS, PROCESSED

feat_cols = json.load(open(MODELS / "lgb_pooled_production_features.json"))
feats = pd.read_parquet(PROCESSED / "features.parquet")
te = feats[(feats["split"] == "test") & (feats["checkpoint_label"] == "week_12")]
row = te.iloc[0]

full = {c: float(row[c]) for c in feat_cols}
ARTIFACTS.mkdir(parents=True, exist_ok=True)

with open(ARTIFACTS / "sample_input.json", "w") as fh:
    json.dump(full, fh, indent=2)

# A small, human-editable subset that is enough to move the prediction.
subset = {
    "n_assessments_due": float(row["n_assessments_due"]),
    "n_missed": float(row["n_missed"]),
    "frac_missed": float(row["frac_missed"]),
    "n_assessments_taken": float(row["n_assessments_taken"]),
    "first_assessment_score": float(row["first_assessment_score"]),
    "second_assessment_score": float(row["second_assessment_score"]),
    "mean_earliness_days": float(row["mean_earliness_days"]),
    "homepage_clicks_total": float(row.get("homepage_clicks_total", 0)),
    "homepage_clicks_last7": float(row.get("homepage_clicks_last7", 0)),
    "studied_credits": float(row["studied_credits"]),
    "num_of_prev_attempts": float(row.get("num_of_prev_attempts", 0)),
    "date_registration": float(row["date_registration"]),
    "no_assessment_closed": float(row["no_assessment_closed"]),
    "engagement_lag_mean": float(row.get("engagement_lag_mean", 0)),
    "week_elapsed": 12.0,
}
with open(ARTIFACTS / "sample_input_min.json", "w") as fh:
    json.dump(subset, fh, indent=2)

print(f"student {int(row['id_student'])} | {row['module_presentation']} | {row['checkpoint_label']}")
print("true label:", int(row["at_risk"]), f"({row['final_result']})")
print("Wrote", ARTIFACTS / "sample_input.json", f"({len(full)} features)")
print("Wrote", ARTIFACTS / "sample_input_min.json", f"({len(subset)} features)")
