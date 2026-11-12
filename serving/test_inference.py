"""Inference test — the API must load the saved artifact and return real scores.

Runs without a live server (no httpx needed): calls the endpoint functions
directly, so it exercises the same model load, feature framing, SHAP path and
response contract the HTTP layer uses.

Usage: python -m pytest serving/test_inference.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serving.api as api

LOW = {"n_missed": 0, "n_assessments_due": 3, "n_assessments_taken": 3,
       "frac_missed": 0.0, "first_assessment_score": 92.0,
       "second_assessment_score": 88.0, "mean_earliness_days": -4.0,
       "homepage_clicks_total": 250.0, "homepage_clicks_last7": 40.0,
       "studied_credits": 60.0, "num_of_prev_attempts": 0.0,
       "date_registration": -20.0, "no_assessment_closed": 0, "week_elapsed": 0}
HIGH = {"n_missed": 5, "frac_missed": 1.0, "first_assessment_score": 0.0,
        "homepage_clicks_last7": 0.0, "studied_credits": 240.0,
        "n_assessments_taken": 0, "n_assessments_due": 5,
        "no_assessment_closed": 0, "week_elapsed": 12}


@pytest.fixture(scope="module", autouse=True)
def loaded():
    api._startup()
    return api


def _predict(features):
    return api.predict(api.PredictRequest(features=features, checkpoint_label="week_12"))


def test_model_is_the_saved_production_artifact():
    assert api._MODEL_META["artifact"] == "lgb_pooled_production.joblib"
    assert api._MODEL_META["n_features"] == 74


def test_predictions_are_real_and_ordered():
    low, high = _predict(LOW), _predict(HIGH)
    assert 0.0 <= low.at_risk_probability <= 1.0
    assert high.at_risk_probability > low.at_risk_probability + 0.3
    assert high.risk_band == "high"
    assert low.risk_band in {"low", "moderate"}


def test_response_carries_band_threshold_and_human_review_note():
    r = _predict(HIGH)
    assert r.operating_threshold > 0
    assert r.risk_band in {"low", "moderate", "high"}
    note = r.human_review_note.lower()
    assert "human" in note and "never" in note and "automated" in note


def test_protected_attributes_cannot_change_the_prediction():
    base = _predict(HIGH).at_risk_probability
    tampered = _predict({**HIGH, "gender": "F", "imd_band": "0-10%"}).at_risk_probability
    assert abs(base - tampered) < 1e-12
