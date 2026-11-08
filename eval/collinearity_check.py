"""Collinearity check for the `no_assessment_closed` indicator.

The SHAP audit shows the flag has ~zero importance (rank 62/78). This script
verifies *why*: it tests whether the flag is exactly, symmetrically collinear
with an existing feature (so a tree split already covers it), and checks the
tempting-but-wrong "the n_missed / frac_missed / first_assessment_score triple
already encodes it" explanation.

Usage: python eval/collinearity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import PROCESSED


def main() -> None:
    f = pd.read_parquet(PROCESSED / "features.parquet")

    print("=== no_assessment_closed collinearity ===")
    n1 = int((f["no_assessment_closed"] == 1).sum())
    print(f"rows with flag=1: {n1} of {len(f)} ({n1/len(f):.2%})")

    # When the flag fires, every assessment-derived feature is pinned.
    flag = f[f["no_assessment_closed"] == 1]
    for col in ["n_assessments_due", "n_missed", "frac_missed", "n_assessments_taken",
                "first_assessment_score", "second_assessment_score", "score_delta"]:
        vals = flag[col].unique()
        print(f"  flag=1 -> {col:26s} unique values: {sorted(vals)[:6]}")

    # EXACT symmetric collinearity with n_assessments_due == 0 ?
    eq = (f["no_assessment_closed"] == 1) == (f["n_assessments_due"] == 0)
    print("\nEXACT symmetric equality: (flag==1) <-> (n_assessments_due==0)")
    print(f"  agreement: {eq.mean():.6%}  ({int(eq.sum())} of {len(f)} rows)")
    print(f"  => flag is a deterministic function of n_assessments_due: {bool(eq.all())}")

    # The tempting-but-wrong explanation: is the (n_missed, frac_missed,
    # first_score) triple signature unique to flag=1 rows?
    sig = (f["n_missed"] == 0) & (f["frac_missed"] == 0) & (f["first_assessment_score"] == 0)
    amb = (sig & (f["no_assessment_closed"] == 0))
    print(f"\nAmbiguity check: rows with same triple signature but flag=0: {int(amb.sum())}")
    if amb.sum():
        a = f[amb]
        print(f"  first_score_missing==0 (score genuinely recorded as 0): {int((a['first_score_missing']==0).sum())} of {len(a)}")
        print(f"  their at-risk rate: {a['at_risk'].mean():.4f}  vs flag=1 rows: {flag['at_risk'].mean():.4f}")
        print("  => the triple signature IS ambiguous; redundancy runs through n_assessments_due, not the triple.")


if __name__ == "__main__":
    main()
