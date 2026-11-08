"""Render the dashboard headlessly and capture any exception.

Uses Streamlit's official AppTest harness: it executes the app script in a
sandbox and surfaces errors exactly as the browser would.

Usage: python serving/test_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent / "dashboard.py"


def main() -> None:
    at = AppTest.from_file(str(APP), default_timeout=180).run()

    if at.exception:
        print("DASHBOARD RAISED:")
        for e in at.exception:
            print("  ", type(e.value).__name__, e.value)
        raise SystemExit(1)

    print("DASHBOARD RENDERED OK — no exceptions")
    print(f"  titles      : {len(at.title)}")
    print(f"  subheaders  : {len(at.subheader)}")
    for s in at.subheader:
        print(f"     - {s.value}")
    print(f"  selectboxes: {len(at.selectbox)}")
    print(f"  number_inputs : {len(at.number_input)}")
    print(f"  buttons     : {len(at.button)}")
    print(f"  dataframes  : {len(at.dataframe)}")
    print(f"  images (charts) : {len(at.image)}")
    print(f"  metrics     : {len(at.metric)}")
    for m in at.metric:
        print(f"     - {m.label}: {m.value}")


if __name__ == "__main__":
    main()
