"""Step 8 — FastAPI serving for the at-risk prediction model.

Loads the latest pooled LightGBM artifact at startup and exposes:
  POST /predict   -> at_risk probability + SHAP explanation
  GET  /health
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common import MODELS, PROCESSED

app = FastAPI(title="OULAD At-Risk Prediction", version="1.0.0")

_MODEL = None
_FEATURE_COLS: List[str] = []
_EXPLAINER = None
_EXPECTED_VALUE = 0.0
_MODEL_META: Dict[str, Any] = {}


def _load_latest() -> Tuple[object, List[str]]:
    """Prefer the fixed production artifact; fall back to newest timestamped."""
    prod = MODELS / "lgb_pooled_production.joblib"
    if prod.exists():
        model_path = prod
        ts = "production"
    else:
        cands = sorted(MODELS.glob("lgb_pooled_*.joblib"))
        if not cands:
            raise RuntimeError("no lgb_pooled model artifact found; run models/train.py first")
        model_path = cands[-1]
        ts = model_path.stem.replace("lgb_pooled_", "")
    feat_path = MODELS / f"lgb_pooled_{ts}_features.json"
    with open(feat_path) as fh:
        feature_cols = json.load(fh)
    return joblib.load(model_path), feature_cols, ts


@app.on_event("startup")
def _startup():
    global _MODEL, _FEATURE_COLS, _EXPLAINER, _EXPECTED_VALUE, _MODEL_META
    _MODEL, _FEATURE_COLS, ts = _load_latest()
    _EXPLAINER = shap.TreeExplainer(_MODEL)
    ev = _EXPLAINER.expected_value
    _EXPECTED_VALUE = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)
    meta_path = MODELS / f"lgb_pooled_{ts}_features.json"
    _MODEL_META = {"artifact": f"lgb_pooled_{ts}.joblib", "timestamp": ts, "n_features": len(_FEATURE_COLS)}


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    checkpoint_label: str | None = "week_12"


class PredictResponse(BaseModel):
    at_risk_probability: float
    checkpoint_label: str | None
    expected_value: float
    top_shap_features: List[Dict[str, Any]]
    model: Dict[str, Any]


def _to_frame(features: Dict[str, Any]) -> pd.DataFrame:
    row = {c: 0 for c in _FEATURE_COLS}
    for k, v in features.items():
        if k in row:
            row[k] = v
    df = pd.DataFrame([row], columns=_FEATURE_COLS)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.fillna(0)


@app.get("/health")
def health():
    return {"status": "ok", "model": _MODEL_META}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    X = _to_frame(req.features)
    prob = float(_MODEL.predict_proba(X)[:, 1][0])
    sv = _EXPLAINER.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    shap_row = sv[0]
    order = np.argsort(-np.abs(shap_row))[:10]
    top = []
    for i in order:
        fname = _FEATURE_COLS[i]
        top.append(
            {
                "feature": fname,
                "shap_value": float(shap_row[i]),
                "value": float(X.iloc[0, i]),
            }
        )
    return PredictResponse(
        at_risk_probability=round(prob, 4),
        checkpoint_label=req.checkpoint_label,
        expected_value=round(_EXPECTED_VALUE, 4),
        top_shap_features=top,
        model=_MODEL_META,
    )
