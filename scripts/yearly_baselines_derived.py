"""Read-only: per-year baseline returns for the test period, from stored equity curves.

This recomputes **nothing**. It reads the `fact_portfolio_daily` equity series that the
authorised Phase 6 run already persisted, splits it by calendar year, and reports each
year's simple return.

Before writing anything it re-derives each run's **full-period** CAGR and Sharpe from the
same stored equity and compares them with `summary_test.csv`. If any of them disagrees by
more than 1e-9 the script exits non-zero and writes no file: a per-year split that does
not reconcile with the reported whole-period number would be worse than no table at all.

Nothing here writes to the warehouse or touches any existing test output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.metrics import cagr, daily_returns, max_drawdown, sharpe
from src.common.config import load_config
from src.common.db import get_engine, read_sql

#: How close a re-derived full-period figure must be to the reported one.
TOLERANCE = 1e-9

#: Runs to report per-year for.
RUNS = ("backtest_main", "test_B1", "test_B2", "test_B3", "test_V1", "test_V2_nocost")

LABELS = {
    "backtest_main": "V2 strategy (frozen)",
    "test_B1": "B1 buy and hold ^NSEI",
    "test_B2": "B2 equal weight monthly",
    "test_B3": "B3 trend sma_20 > sma_50",
    "test_V1": "V1 regime only (no classifier)",
    "test_V2_nocost": "V2 without costs",
}


def equity_series(engine: Any, run_id: str) -> pd.Series:
    """The stored daily equity for one run, indexed by date."""
    frame = read_sql(
        engine,
        "SELECT date_key, equity FROM fact_portfolio_daily WHERE run_id = :r ORDER BY date_key",
        {"r": run_id},
    )
    if frame.empty:
        raise ValueError(f"no stored equity for run_id {run_id!r}")
    index = pd.to_datetime([str(int(k)) for k in frame["date_key"]], format="%Y%m%d")
    return pd.Series(
        frame["equity"].to_numpy(dtype=float), index=pd.DatetimeIndex(index, name="date")
    )


def yearly(equity: pd.Series) -> list[dict[str, Any]]:
    """Simple return for each calendar year present."""
    rows = []
    for year, group in equity.groupby(equity.index.year):
        series = group
        rows.append(
            {
                "year": int(year),
                "n_days": int(len(series)),
                "start_equity": float(series.iloc[0]),
                "end_equity": float(series.iloc[-1]),
                "return": float(series.iloc[-1] / series.iloc[0] - 1.0),
            }
        )
    return rows


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)
    results = Path(cfg["paths"]["results"])

    summary = pd.read_csv(results / "summary_test.csv").set_index("run_id")
    if summary.empty:
        print("summary_test.csv is empty; run final_test first", file=sys.stderr)
        return 1

    # ---------------------------------------------------------------- tie-out
    print("Tie-out of full-period figures re-derived from stored equity")
    print(f"{'run':16}{'metric':8}{'reported':>16}{'re-derived':>16}{'delta':>14}")
    print("-" * 72)
    failures: list[str] = []
    curves: dict[str, pd.Series] = {}
    rf = float(cfg["backtest"]["risk_free_annual"])

    for run_id in RUNS:
        if run_id not in summary.index:
            continue
        equity = equity_series(engine, run_id)
        curves[run_id] = equity
        returns = daily_returns(equity)
        checks = {
            "sharpe": (float(summary.loc[run_id, "sharpe"]), sharpe(returns, rf)),
            "cagr": (float(summary.loc[run_id, "cagr"]), cagr(equity)),
            "max_dd": (float(summary.loc[run_id, "max_drawdown"]), max_drawdown(equity)),
        }
        for name, (reported, derived) in checks.items():
            delta = abs(reported - derived)
            print(f"{run_id:16}{name:8}{reported:16.10f}{derived:16.10f}{delta:14.2e}")
            if delta > TOLERANCE:
                failures.append(f"{run_id}.{name}: reported {reported} vs derived {derived}")

    if failures:
        print("\nTIE-OUT FAILED; nothing written:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 2
    print("\nAll re-derived full-period figures match summary_test.csv to 1e-9.")

    # ---------------------------------------------------------------- per year
    rows = []
    for run_id, equity in curves.items():
        for row in yearly(equity):
            rows.append({"run_id": run_id, "label": LABELS.get(run_id, run_id), **row})

    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="year", columns="label", values="return")
    print("\nPer-year simple return (derived from stored equity curves)")
    print(pivot.map(lambda v: f"{v:+.2%}").to_string())

    out = results / "yearly_test_all_runs_DERIVED.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("NOTE: derived from stored equity curves; no metric was recomputed or rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
