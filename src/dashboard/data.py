"""Shared, read-only data access for the dashboard (`spec 02` section 6).

Every query here is a `SELECT`. Nothing in this module writes to the warehouse, and the
dashboard never recomputes a metric: it reads the CSV files the authorised runs already
produced. That is deliberate. If the dashboard calculated its own Sharpe, the number on
screen could drift from the number in `RESULTS.md`, and the two would both look right.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import event

from src.common.config import load_config
from src.common.db import read_sql

#: Shown at the top of every screen.
DISCLAIMER = (
    "Academic prototype. Not investment advice. No real money, no broker connection. "
    "The out-of-sample test found no demonstrated edge; see RESULTS.md."
)

PAPER_DISCLAIMER = (
    "Paper trading demonstrates that the system operates. It does not demonstrate that "
    "it makes money. Paper P&L is not evidence of performance."
)

#: Run ids to plot, always with the baselines beside the strategy.
#: `{index}` is filled from `cfg["data"]["index_symbol"]` by `result_labels`, so the
#: label cannot go stale if the configured index changes.
RESULT_RUNS = {
    "backtest_main": "V2 strategy (frozen)",
    "test_B1": "B1 buy and hold {index}",
    "test_B2": "B2 equal weight monthly",
    "test_B3": "B3 trend sma_20 > sma_50",
    "test_V1": "V1 regime only (no classifier)",
    "test_V2_nocost": "V2 without costs",
}


def result_labels(cfg: dict[str, Any]) -> dict[str, str]:
    """`RESULT_RUNS` with the configured index symbol substituted in."""
    index = str(cfg["data"]["index_symbol"])
    return {run: label.format(index=index) for run, label in RESULT_RUNS.items()}


def cfg_and_engine(config: str | None = None) -> tuple[dict[str, Any], Any]:
    """Load the config and open a **read-only** connection.

    The app used to open a writable engine here even though every query is a SELECT.
    Opening the SQLite file with `mode=ro` makes the read-only promise mechanical rather
    than a convention: a mistake in a future screen fails at the database instead of
    silently writing to the warehouse.
    """
    cfg = load_config(config or "config/config.yaml")
    return cfg, readonly_engine(cfg)


def readonly_engine(cfg: Mapping[str, Any]) -> Any:
    """Open the warehouse read-only via SQLite `mode=ro`.

    **This is what makes a publicly exposed dashboard safe.** The page must never be
    able to write to the real warehouse, and the cheapest way to guarantee that is to
    make the database itself refuse. Any write raises `sqlite3.OperationalError`
    ("attempt to write a readonly database"); `tests/test_bot_demo.py` asserts that.

    WAL is deliberately *not* enabled here: switching journal mode is itself a write.
    """
    from pathlib import Path

    from sqlalchemy import create_engine

    db_path = Path(cfg["paths"]["db"]).resolve()
    uri = f"sqlite:///file:{db_path.as_posix()}?mode=ro&uri=true"
    engine = create_engine(uri, future=True)

    @event.listens_for(engine, "connect")
    def _readonly(connection: Any, _record: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA query_only = ON")
        finally:
            cursor.close()

    return engine


def empty(message: str) -> None:
    """Every screen shows a message rather than raising when its tables are empty."""
    import streamlit as st

    st.info(message)


# ---------------------------------------------------------------- market


def index_and_regimes(
    cfg: dict[str, Any], engine: Any, limit_days: int | None = None
) -> pd.DataFrame:
    """Index close with the regime label, oldest first.

    `limit_days=None` returns the whole stored history. The previous default of 1,200
    silently cut a 2,900-day warehouse while the chart title claimed "whole stored
    history"; a limit is still available for callers that explicitly want a window.
    """
    sql = (
        "SELECT p.date_key AS date_key, p.close_price AS close, r.regime_label AS regime "
        "FROM fact_price p "
        "JOIN dim_stock s ON s.stock_key = p.stock_key "
        "LEFT JOIN fact_market_regime m ON m.date_key = p.date_key "
        "LEFT JOIN dim_regime r ON r.regime_key = m.regime_key "
        "WHERE s.symbol = :sym"
    )
    params: dict[str, Any] = {"sym": cfg["data"]["index_symbol"]}
    if limit_days is None:
        sql += " ORDER BY p.date_key"
    else:
        sql += " ORDER BY p.date_key DESC LIMIT :n"
        params["n"] = int(limit_days)
    frame = read_sql(engine, sql, params)
    return frame if limit_days is None else frame.iloc[::-1]


def vix_series(cfg: dict[str, Any], engine: Any, limit_days: int | None = None) -> pd.DataFrame:
    """India VIX close, oldest first. `limit_days=None` means the full history."""
    sql = (
        "SELECT p.date_key AS date_key, p.close_price AS close FROM fact_price p "
        "JOIN dim_stock s ON s.stock_key = p.stock_key WHERE s.symbol = :sym"
    )
    params: dict[str, Any] = {"sym": cfg["data"]["vix_symbol"]}
    if limit_days is None:
        sql += " ORDER BY p.date_key"
    else:
        sql += " ORDER BY p.date_key DESC LIMIT :n"
        params["n"] = int(limit_days)
    frame = read_sql(engine, sql, params)
    return frame if limit_days is None else frame.iloc[::-1]


def latest_data_key(engine: Any) -> int | None:
    """The most recent date with any market data."""
    frame = read_sql(engine, "SELECT MAX(date_key) AS m FROM fact_price")
    if frame.empty or frame["m"].iloc[0] is None:
        return None
    return int(frame["m"].iloc[0])


def is_stale(cfg: dict[str, Any], last_key: int | None, today_key: int) -> bool:
    """True when the market data is older than `paper.stale_banner_days`."""
    from datetime import datetime

    if not last_key:
        return True
    start = datetime.strptime(str(int(last_key)), "%Y%m%d")
    end = datetime.strptime(str(int(today_key)), "%Y%m%d")
    return (end - start).days > int(cfg["paper"]["stale_banner_days"])


# ---------------------------------------------------------------- OLAP explorer


def regime_labels(engine: Any) -> list[str]:
    """The regime labels present in the warehouse, for the OLAP filter."""
    frame = read_sql(
        engine,
        "SELECT DISTINCT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key ORDER BY r.regime_label",
    )
    return [str(v) for v in frame["regime"].dropna().tolist()]


def available_years(engine: Any) -> list[int]:
    """Years that actually have equity indicator rows, for the OLAP filter."""
    frame = read_sql(
        engine,
        "SELECT DISTINCT d.year AS year FROM fact_indicator i "
        "JOIN dim_date d ON d.date_key = i.date_key "
        "JOIN dim_stock s ON s.stock_key = i.stock_key "
        "WHERE s.instrument_type = 'EQUITY' ORDER BY d.year",
    )
    return [int(v) for v in frame["year"].dropna().tolist()]


def sector_quarter_returns(
    engine: Any,
    sector: str | None = None,
    year: int | None = None,
    quarter: int | None = None,
    regime: str | None = None,
) -> pd.DataFrame:
    """Mean daily return (%) per sector and year-quarter.

    A **return**, not a price level: a sector whose average close is high is not
    thereby a good sector, so the heat map must not rank on price.
    """
    sql = (
        "SELECT d.year AS year, d.quarter AS quarter, s.sector AS sector, "
        "       AVG(i.return_1d) * 100.0 AS avg_daily_return_pct, "
        "       COUNT(*) AS n "
        "FROM fact_indicator i "
        "JOIN dim_date d ON d.date_key = i.date_key "
        "JOIN dim_stock s ON s.stock_key = i.stock_key "
        "JOIN fact_market_regime m ON m.date_key = i.date_key "
        "JOIN dim_regime r ON r.regime_key = m.regime_key "
        "WHERE s.instrument_type = 'EQUITY'"
    )
    params: dict[str, Any] = {}
    if sector is not None:
        sql += " AND s.sector = :sector"
        params["sector"] = sector
    if year is not None:
        sql += " AND d.year = :year"
        params["year"] = int(year)
    if quarter is not None:
        sql += " AND d.quarter = :quarter"
        params["quarter"] = int(quarter)
    if regime is not None:
        sql += " AND r.regime_label = :regime"
        params["regime"] = regime
    sql += " GROUP BY d.year, d.quarter, s.sector ORDER BY d.year, d.quarter, s.sector"
    return read_sql(engine, sql, params)


def quarter_label(year: Any, quarter: Any) -> str:
    """`2024`, `3` -> `2024-Q3`, the column heading on the heat map."""
    return f"{int(year)}-Q{int(quarter)}"


def stocks_in_sector_quarter(engine: Any, sector: str, year: int, quarter: int) -> pd.DataFrame:
    """Drill-down: every stock in one sector, with its return over that quarter."""
    return read_sql(
        engine,
        "SELECT s.symbol AS symbol, s.company_name AS company, "
        "       SUM(i.return_1d) * 100.0 AS quarter_return_pct, "
        "       COUNT(*) AS days "
        "FROM fact_indicator i "
        "JOIN dim_date d ON d.date_key = i.date_key "
        "JOIN dim_stock s ON s.stock_key = i.stock_key "
        "WHERE s.instrument_type = 'EQUITY' AND s.sector = :sector "
        "AND d.year = :year AND d.quarter = :quarter "
        "GROUP BY s.symbol, s.company_name "
        "ORDER BY quarter_return_pct DESC",
        {"sector": sector, "year": int(year), "quarter": int(quarter)},
    )


# ---------------------------------------------------------------- signals

#: The paper run whose stored decision the dashboard shows.
PAPER_RUN_ID = "paper_live"

_SIGNAL_COLUMNS = ["date_key", "symbol", "target_weight", "prob_up"]


def latest_signals_date(engine: Any, run_id: str = PAPER_RUN_ID) -> int | None:
    """The latest date a paper run actually processed.

    `fact_portfolio_daily` is the authoritative marker, not `pending_target`: a flat
    decision deliberately stores **no** pending-target rows, so a MAX over that table
    would report "no decision" for a real decision to hold cash.
    """
    frame = read_sql(
        engine,
        "SELECT MAX(date_key) AS m FROM fact_portfolio_daily WHERE run_id = :r",
        {"r": run_id},
    )
    if frame.empty or frame["m"].iloc[0] is None:
        return None
    return int(frame["m"].iloc[0])


def latest_signals(
    engine: Any,
    date_key: int | None = None,
    model: str | None = None,
    run_id: str = PAPER_RUN_ID,
) -> pd.DataFrame:
    """Stored probabilities and target weights for one paper decision.

    Returns one row per stock the paper job scored. A name the strategy did not select
    has `target_weight` 0.0 rather than disappearing, so a flat decision is visible
    instead of looking like missing data. `prob_up` comes from `fact_signal`; the target
    weight comes from `pending_target` for the same decision date.

    Args:
        engine: Read-only warehouse engine.
        date_key: Decision date. Defaults to the latest processed paper date.
        model: Model name to prefer (usually `strategy.selected_model`). When omitted,
            the model with the most rows for that date is used.
        run_id: Paper run identifier.
    """
    if date_key is None:
        date_key = latest_signals_date(engine, run_id)
    if date_key is None:
        return pd.DataFrame(columns=_SIGNAL_COLUMNS)

    model_key = _signal_model_key(engine, int(date_key), model)
    if model_key is not None:
        frame = read_sql(
            engine,
            "SELECT f.date_key AS date_key, s.symbol AS symbol, "
            "       COALESCE(p.target_weight, 0.0) AS target_weight, "
            "       f.prob_up AS prob_up "
            "FROM fact_signal f "
            "JOIN dim_stock s ON s.stock_key = f.stock_key "
            "LEFT JOIN pending_target p "
            "  ON p.run_id = :r AND p.decision_date_key = f.date_key "
            " AND p.stock_key = f.stock_key "
            "WHERE f.date_key = :d AND f.model_key = :k "
            "ORDER BY target_weight DESC, f.prob_up DESC, s.symbol",
            {"r": run_id, "d": int(date_key), "k": int(model_key)},
        )
        if not frame.empty:
            return frame

    # Fallback for an older warehouse that stored targets but not probabilities.
    return read_sql(
        engine,
        "SELECT p.decision_date_key AS date_key, s.symbol AS symbol, "
        "       p.target_weight AS target_weight, p.prob_up AS prob_up "
        "FROM pending_target p JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE p.run_id = :r AND p.decision_date_key = :d "
        "ORDER BY p.target_weight DESC, s.symbol",
        {"r": run_id, "d": int(date_key)},
    )


def _signal_model_key(engine: Any, date_key: int, model: str | None) -> int | None:
    """Pick the stored model version with the most rows for one decision date.

    The configured model is preferred. If that model has no rows for the date (for
    example a smoke config whose selected model is not the one the paper job stored),
    fall back to whichever stored model has the most rows rather than showing a false
    "no signals" message.
    """
    if model:
        frame = read_sql(
            engine,
            "SELECT f.model_key AS k FROM fact_signal f "
            "JOIN dim_model m ON m.model_key = f.model_key "
            "WHERE f.date_key = :d AND m.model_name = :m "
            "GROUP BY f.model_key ORDER BY COUNT(*) DESC, f.model_key LIMIT 1",
            {"d": int(date_key), "m": str(model)},
        )
        if not frame.empty and frame["k"].iloc[0] is not None:
            return int(frame["k"].iloc[0])

    frame = read_sql(
        engine,
        "SELECT model_key AS k FROM fact_signal WHERE date_key = :d "
        "GROUP BY model_key ORDER BY COUNT(*) DESC, model_key LIMIT 1",
        {"d": int(date_key)},
    )
    if frame.empty or frame["k"].iloc[0] is None:
        return None
    return int(frame["k"].iloc[0])


def regime_for_date(engine: Any, date_key: int) -> str | None:
    """The smoothed regime label stored for one date, or None when absent."""
    frame = read_sql(
        engine,
        "SELECT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key WHERE m.date_key = :d",
        {"d": int(date_key)},
    )
    if frame.empty:
        return None
    return str(frame["regime"].iloc[0])


# ---------------------------------------------------------------- results


def results_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["results"])


def summary(cfg: dict[str, Any], period: str) -> pd.DataFrame:
    """The stored `summary_<period>.csv`. Read, never recomputed."""
    path = results_dir(cfg) / f"summary_{period}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def equity_curves(engine: Any, run_ids: list[str]) -> pd.DataFrame:
    """Stored equity for the given runs, long format."""
    if not run_ids:
        return pd.DataFrame()
    keys = ", ".join(f"'{r}'" for r in run_ids)
    return read_sql(
        engine,
        f"SELECT run_id, date_key, equity, drawdown FROM fact_portfolio_daily "
        f"WHERE run_id IN ({keys}) ORDER BY date_key",
    )


def derived_yearly(cfg: dict[str, Any]) -> pd.DataFrame:
    """Per-year returns derived from the stored equity curves, if the file exists."""
    path = results_dir(cfg) / "yearly_test_all_runs_DERIVED.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------- diagnostics


def classifier_metrics(cfg: dict[str, Any]) -> pd.DataFrame:
    path = results_dir(cfg) / "classifier_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def prob_histogram(engine: Any, model: str, period_start: int, period_end: int) -> pd.DataFrame:
    """`prob_up` distribution for one model over a window. Read from stored signals."""
    return read_sql(
        engine,
        "SELECT f.prob_up AS prob_up FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b "
        "AND f.prob_up IS NOT NULL",
        {"model": model, "a": int(period_start), "b": int(period_end)},
    )


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """2x2 confusion matrix from stored labels and predictions."""
    return pd.crosstab(
        pd.Series(y_true, name="actual"),
        pd.Series(y_pred, name="predicted"),
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)


def training_feature_names(cfg: Any, engine: Any) -> list[str]:
    """The feature columns the models were trained on, in training order."""
    from src.features.panel import get_feature_cols, use_vix_in_regime

    return list(get_feature_cols(use_vix_in_regime(cfg, engine)))


def feature_importance(
    model_name: str, models_dir: Path, feature_names: list[str] | None = None
) -> pd.DataFrame:
    """Feature importance from a saved model bundle, when it exposes one.

    The trainer writes <model>_wf<year>.joblib, so the newest such file is used.
    Names must be supplied: the pipeline's first step (the winsorizer) cannot report
    feature names, so get_feature_names_out raises and cannot be relied on.
    """
    import joblib

    path = next(
        (p for p in sorted(models_dir.glob(f"{model_name}_wf*.joblib")) if p.exists()), None
    )
    if path is None:
        return pd.DataFrame()

    bundle = joblib.load(path)

    # Pipeline.steps holds (name, estimator) pairs, so the model is the second
    # element of the last pair.
    steps = getattr(bundle, "steps", None)
    estimator = bundle
    if steps:
        last = list(steps)[-1]
        estimator = last[1] if isinstance(last, tuple) else last

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return pd.DataFrame()
        importances = abs(coefficients).ravel()

    values = [float(v) for v in importances]
    if feature_names is None or len(feature_names) != len(values):
        return pd.DataFrame()
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def regime_counts(engine: Any) -> pd.DataFrame:
    return read_sql(
        engine,
        "SELECT r.regime_label AS regime, COUNT(*) AS days FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key GROUP BY r.regime_label",
    )
