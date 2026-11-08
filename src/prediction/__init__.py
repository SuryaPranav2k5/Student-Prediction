"""OULAD student at-risk prediction system.

Note on layout: the working code lives at the repo root (``common.py`` plus the
``features/``, ``models/``, ``eval/`` and ``serving/`` packages) and is run
directly from there::

    python features/build_features.py   # features + leakage audit
    python models/train.py              # LR baseline + pooled/per-checkpoint LightGBM
    python eval/run_eval.py             # PR-AUC CI, operating point, fairness, metrics.json
    python eval/shap_explain.py         # global + per-student SHAP
    python eval/fairness_breakdown.py   # fairness deep-dive
    python eval/collinearity_check.py   # no_assessment_closed collinearity proof
    uvicorn serving.api:app --reload    # FastAPI service
    streamlit run serving/dashboard.py # dashboard

This module is a convenience entry point only; ``python -m prediction`` prints
the same run order.
"""

from __future__ import annotations


def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()