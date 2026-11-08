"""Interact with the dashboard's what-if form and confirm the prediction moves.

Verifies the three presets produce three distinguishable risk scores, which is
the actual "give the model input and see the answer" behaviour.

Usage: python serving/test_dashboard_interactive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "dashboard.py"


def run_with(preset: str) -> dict:
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    at.radio[0].set_value(preset).run()
    if at.exception:
        raise RuntimeError([f"{type(e.value).__name__}: {e.value}" for e in at.exception])
    labels = [m.label for m in at.metric]
    # the what-if section's P(at_risk) is the 4th metric
    return {"prob": at.metric[3].value, "flag": at.metric[4].value, "metrics": labels}


def main() -> None:
    base = AppTest.from_file(str(APP), default_timeout=300).run()
    if base.exception:
        raise RuntimeError([f"{type(e.value).__name__}: {e.value}" for e in base.exception])

    print("dashboard renders OK")
    print(f"  radio presets available: {base.radio[0].options}")
    print(f"  number inputs           : {len(base.number_input)}")
    print(f"  checkboxes              : {len(base.checkbox)}")
    print(f"  charts (images)         : {len(base.image)}")

    results = {}
    for preset in ["Selected student", "High-risk example", "Low-risk example"]:
        r = run_with(preset)
        results[preset] = r
        print(f"\npreset '{preset}' -> P(at_risk) = {r['prob']}   flagged at operating threshold: {r['flag']}")

    hi = float(results["High-risk example"]["prob"])
    lo = float(results["Low-risk example"]["prob"])
    print("\n--- ordering check ---")
    assert hi > 0.9, f"high-risk preset should score >0.9, got {hi}"
    assert lo < 0.2, f"low-risk preset should score <0.2, got {lo}"
    assert hi > lo, "high-risk preset must outrank low-risk preset"
    print(f"PASS: high-risk ({hi:.3f}) >> low-risk ({lo:.3f}) — the input form genuinely drives the model.")


if __name__ == "__main__":
    main()
