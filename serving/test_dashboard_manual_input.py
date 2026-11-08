"""Type directly into the what-if fields and confirm the score moves.

Complements test_dashboard_interactive.py (which only flips presets): this one
sets an individual number input, which is what a user actually does.

Usage: python serving/test_dashboard_manual_input.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "dashboard.py"


def run() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    if at.exception:
        raise RuntimeError([f"{type(e.value).__name__}: {e.value}" for e in at.exception])
    return at


def main() -> None:
    at = run()
    labels = [n.label for n in at.number_input]
    print("what-if fields:")
    for i, l in enumerate(labels):
        print(f"  [{i}] {l}")

    def whatif_prob(a: AppTest) -> float:
        return float(a.metric[3].value)

    # Start from the low-risk preset so every feature is dominated by the inputs.
    at.radio[0].set_value("Low-risk example").run()
    before = whatif_prob(at)
    print(f"\nbaseline (Low-risk example): P(at_risk) = {before:.4f}")

    # Miss 5 of 5 assessments -> should push toward at-risk.
    at.number_input(key="in_Low-risk example_n_missed").set_value(5.0).run()
    after_missed = whatif_prob(at)
    print(f"after setting n_missed = 5 : P(at_risk) = {after_missed:.4f}")

    # Then submit perfect scores -> should pull it back down.
    at.number_input(key="in_Low-risk example_first_assessment_score").set_value(100.0).run()
    after_score = whatif_prob(at)
    print(f"after first_assessment_score = 100 : P(at_risk) = {after_score:.4f}")

    assert after_missed > before, "missing assessments must raise the score"
    assert after_score < after_missed, "a high first score must pull the score back down"
    print(f"\nPASS: typing into the fields genuinely drives the model "
          f"({before:.3f} -> {after_missed:.3f} -> {after_score:.3f}).")


if __name__ == "__main__":
    main()
