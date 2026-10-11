"""Check the per-year table in RESULTS.md against the derived CSV.

Anchored on the derived block's own heading, so it cannot accidentally match the
`yearly_test.csv` table that sits above it. Read-only; exits non-zero on a mismatch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ORDER = [
    "V2 strategy (frozen)",
    "B1 buy and hold ^NSEI",
    "B2 equal weight monthly",
    "B3 trend sma_20 > sma_50",
    "V1 regime only (no classifier)",
    "V2 without costs",
]

MINUS = "−"  # the typographic minus RESULTS.md uses


def main() -> int:
    results = Path("reports/results")
    frame = pd.read_csv(results / "yearly_test_all_runs_DERIVED.csv")
    pivot = frame.pivot(index="year", columns="label", values="return")[ORDER]

    text = (results / "RESULTS.md").read_text(encoding="utf-8")
    anchor = text.index("Per-year returns for every run, derived")
    block = text[anchor : text.index("## 2.", anchor)]

    rows = {}
    for line in block.splitlines():
        match = re.match(r"^\|\s*(\d{4})(?:\s*\(to 10-09\))?\s*\|(.+)\|\s*$", line)
        if match:
            rows[int(match.group(1))] = [
                c.strip().replace("**", "") for c in match.group(2).split("|") if c.strip()
            ]

    problems: list[str] = []
    for year, expected in pivot.iterrows():
        cells = rows.get(int(year))
        if cells is None:
            problems.append(f"{year}: no row found in RESULTS.md")
            continue
        if len(cells) != len(ORDER):
            problems.append(f"{year}: {len(cells)} cells, expected {len(ORDER)}")
            continue
        for label, cell in zip(ORDER, cells, strict=False):
            shown = float(cell.replace(MINUS, "-").replace("%", "").replace("+", ""))
            want = float(expected[label]) * 100.0
            if abs(shown - want) > 0.005:
                problems.append(f"{year} {label}: RESULTS.md says {shown}, CSV says {want:.4f}")

    if problems:
        print("MISMATCHES:")
        for line in problems:
            print("  " + line)
        return 1

    print("RESULTS.md per-year table matches yearly_test_all_runs_DERIVED.csv exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
