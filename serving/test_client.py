"""Test client for the FastAPI service — shows how to feed the model input.

Usage:
    python serving/test_client.py            # minimal editable payload
    python serving/test_client.py --full     # full 74-feature payload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from common import ARTIFACTS

BASE = "http://127.0.0.1:8000"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="use the full 74-feature payload")
    ap.add_argument("--url", default=BASE)
    args = ap.parse_args()

    h = requests.get(f"{args.url}/health", timeout=10)
    print("GET /health ->", h.status_code, h.json())

    name = "sample_input.json" if args.full else "sample_input_min.json"
    features = json.load(open(ARTIFACTS / name))

    r = requests.post(
        f"{args.url}/predict",
        json={"features": features, "checkpoint_label": "week_12"},
        timeout=30,
    )
    print(f"\nPOST /predict ({name}, {len(features)} features) -> {r.status_code}")
    out = r.json()
    print("  probability:", out["at_risk_probability"])
    print("  base value :", out["expected_value"])
    print("  top SHAP drivers:")
    for t in out["top_shap_features"][:6]:
        direction = "toward at-risk" if t["shap_value"] > 0 else "protective"
        print(f"    {t['feature']:34s} shap={t['shap_value']:+.4f}  value={t['value']:.2f}  ({direction})")


if __name__ == "__main__":
    main()
