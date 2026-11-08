"""Step 1 — checkpoint cutoff dates.

Loads the assessments table and, for each course (module_presentation),
computes the actual assessment due-date distribution. Cutoffs are fixed
calendar offsets from module start (week 4 -> day 28, week 8 -> day 56,
week 12 -> day 84) and are only accepted where at least one assessment
has closed by that cutoff for that course.

Courses where zero assessments close by a cutoff are flagged
(exclude_from_checkpoint) and their rows are dropped from that
checkpoint's feature set downstream (decision: fixed calendar cutoffs +
exclude affected rows).

Output: data/processed/cutoffs.parquet keyed by module_presentation +
checkpoint_label.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common import CUTOFF_DAYS, ensure_dirs, load_tables, to_day


def build_cutoffs(raw: Path = Path("data/raw"), out: Path | None = None) -> pd.DataFrame:
    tables = load_tables(raw)
    assessments = tables["assessments"].copy()

    # Only dated assessments have a real due date; '?' (exams with no fixed date)
    # are excluded from the due-date distribution.
    dated = assessments[assessments["date"].astype(str) != "?"].copy()
    dated["date"] = to_day(dated["date"])

    # Per-course due-date distribution (day-offsets from module start).
    dist = (
        dated.groupby(["code_module", "code_presentation"])["date"]
        .agg(due_dates=list, n_dated="size")
        .reset_index()
    )
    dist["module_presentation"] = dist["code_module"].astype(str) + "_" + dist["code_presentation"].astype(str)

    rows = []
    for _, r in dist.iterrows():
        dates = sorted(int(d) for d in r["due_dates"] if pd.notna(d))
        cp = r["module_presentation"]
        for label, cutoff_day in CUTOFF_DAYS.items():
            n_closed = sum(1 for d in dates if d <= cutoff_day)
            rows.append(
                {
                    "module_presentation": cp,
                    "code_module": r["code_module"],
                    "code_presentation": r["code_presentation"],
                    "checkpoint_label": label,
                    "cutoff_day": cutoff_day,
                    "n_assessments_closed": int(n_closed),
                    "first_assessment_day": int(dates[0]) if dates else None,
                    "exclude_from_checkpoint": n_closed == 0,
                }
            )

    cutoffs = pd.DataFrame(rows)
    ensure_dirs()
    out = out or Path("data/processed/cutoffs.parquet")
    cutoffs.to_parquet(out, index=False)

    print("=== Step 1: cutoff dates ===")
    print(cutoffs.to_string(index=False))
    print()
    print(
        "Rows excluded (no assessment closed by cutoff):",
        int(cutoffs["exclude_from_checkpoint"].sum()),
        "of", len(cutoffs),
    )
    print("Wrote", out)
    return cutoffs


if __name__ == "__main__":
    build_cutoffs()
