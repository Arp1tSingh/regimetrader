"""Print what each dashboard screen actually renders, for manual inspection."""

from __future__ import annotations

import os
import sys

os.environ["RT_DASH_CONFIG"] = sys.argv[1] if len(sys.argv) > 1 else "config/config.smoke.yaml"

import sys as _sys  # noqa: E402
from pathlib import Path  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.dashboard.app import SCREENS  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")

for screen in SCREENS:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if at.exception:
        print(f"{screen}: EXCEPTION {at.exception}")
        continue
    at.sidebar.radio[0].set_value(screen).run()

    if at.exception:
        print(f"{screen}: EXCEPTION {at.exception}")
        continue

    charts = len(at.get("plotly_chart") or [])
    tables = len(at.get("dataframe") or [])
    titles = [
        str(getattr(x, "value", "")).strip()
        for x in list(at.subheader) + list(at.markdown)
        if str(getattr(x, "value", "")).strip().startswith("#")
    ]
    headers = [t for t in titles if t.startswith("##")]
    print(f"{screen:22} charts={charts} tables={tables} sections={headers}")
    for note in (
        list(at.warning) + list(at.error) + list(at.info)
    ):
        value = str(getattr(note, "value", "")).strip().replace("\n", " ")
        print(f"    {value[:110]}")
print("\nno exceptions on any screen.")
