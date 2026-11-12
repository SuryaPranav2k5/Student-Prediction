"""Error analysis — false positives and false negatives at the operating point.

The brief requires explaining false positives and false negatives, not just
reporting a confusion matrix. This script profiles both error types on the
held-out test presentation (2014J) at each checkpoint's validation-selected
threshold, and prints two worked examples (a high-confidence false positive
and a false negative).

Usage: python eval/error_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from common import ARTIFACTS, MODELS, PROCESSED

CHECKPOINTS = ["week_4", "week_8", "week_12"]
PROFILE = ["n_missed", "frac_missed", "first_assessment_score",
           "homepage_clicks_last7", "mean_earliness_days"]


def latest_pooled():
    prod = MODELS / "lgb_pooled_production.joblib"
    ts = "production" if prod.exists() else "latest"
    mp = prod if prod.exists() else sorted(MODELS.glob("lgb_pooled_*.joblib"))[-1]
    feat = json.load(open(MODELS / f"lgb_pooled_{ts}_features.json"))
    return joblib.load(mp), feat


def main():
    model, feature_cols = latest_pooled()
    metrics = json.load(open(ARTIFACTS / "metrics.json"))
    thr = {c["checkpoint"]: c["operating_threshold"] for c in metrics["checkpoints"]}
    test = pd.read_parquet(PROCESSED / "features.parquet")
    test = test[test["split"] == "test"]

    for cp in CHECKPOINTS:
        te = test[test["checkpoint_label"] == cp].copy()
        X = te[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        te["prob"] = model.predict_proba(X)[:, 1]
        te["pred"] = (te["prob"] >= thr[cp]).astype(int)

        tp = te[(te.pred == 1) & (te.at_risk == 1)]
        fp = te[(te.pred == 1) & (te.at_risk == 0)]
        fn = te[(te.pred == 0) & (te.at_risk == 1)]
        tn = te[(te.pred == 0) & (te.at_risk == 0)]

        print(f"\n=== {cp} — threshold {thr[cp]:.4f} ===")
        print(f"  TP={len(tp):5d}  FP={len(fp):5d}  FN={len(fn):5d}  TN={len(tn):5d}")
        print(f"  error rate: FP={len(fp)/len(te):.1%} of all students, "
              f"FN={len(fn)/len(te):.1%} of all students")

        print("\n  FALSE POSITIVES (flagged, actually Pass/Distinction):")
        print(f"    mean predicted P(at_risk) = {fp['prob'].mean():.3f}")
        print(f"    missed no assessments        = {(fp['n_missed'] == 0).mean():.1%}")
        print(f"    have a first assessment score= {(fp['first_assessment_score'] > 0).mean():.1%}")
        print(f"    mean first assessment score  = {fp['first_assessment_score'].mean():.1f}"
              f"  (TP: {tp['first_assessment_score'].mean():.1f})")
        print(f"    mean last-7-day VLE clicks   = {fp['homepage_clicks_last7'].mean():.1f}"
              f"  (TP: {tp['homepage_clicks_last7'].mean():.1f})")

        print("\n  FALSE NEGATIVES (not flagged, actually Withdrawn/Fail):")
        print(f"    mean predicted P(at_risk) = {fn['prob'].mean():.3f}")
        print(f"    are Withdrawn vs Fail       = {(fn['final_result'] == 'Withdrawn').mean():.1%}"
              f" / {(fn['final_result'] == 'Fail').mean():.1%}")
        print(f"    mean missed assessments     = {fn['n_missed'].mean():.2f}"
              f"  (TP: {tp['n_missed'].mean():.2f})")
        print(f"    mean last-7-day VLE clicks  = {fn['homepage_clicks_last7'].mean():.1f}"
              f"  (TP: {tp['homepage_clicks_last7'].mean():.1f})")

        if cp == "week_12":
            worst_fp = fp.sort_values("prob", ascending=False).iloc[0]
            worst_fn = fn.sort_values("prob").iloc[0]
            print("\n  Worked examples (week 12):")
            for name, r in [("highest-confidence FP", worst_fp), ("deepest FN", worst_fn)]:
                print(f"    {name}: student {int(r['id_student'])} "
                      f"({r['code_module']}_{r['code_presentation']}) "
                      f"P={r['prob']:.3f} true={r['final_result']} "
                      f"n_missed={int(r['n_missed'])} "
                      f"first_score={r['first_assessment_score']:.0f} "
                      f"homepage_last7={r['homepage_clicks_last7']:.0f}")


if __name__ == "__main__":
    main()
