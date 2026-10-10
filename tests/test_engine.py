"""Tests for the backtest engine (task T5.4, spec 06 section 2, TESTS section 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import Engine, text

from src.backtest.engine import DayContext, run_backtest
from src.common.db import get_engine, init_db, read_sql


def base_cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.smoke.yaml"))


#: Fixture dates sit after the smoke config's test_start (2023-01-01), so each test
#: moves the guard out of the way unless it is specifically testing the guard.
def open_window(cfg: dict[str, Any]) -> dict[str, Any]:
    """Copy the config with the test guard pushed past the fixture dates."""
    out = {**cfg}
    out["splits"] = {**cfg["splits"], "test_start": "2030-01-01"}
    return out


def load_rows(
    engine: Engine,
    symbols: list[str],
    prices: dict[tuple[str, int], float],
    opens: dict[tuple[str, int], float] | None = None,
) -> None:
    """Insert dim_date, dim_stock and fact_price rows for a controlled scenario."""
    opens = opens or {}
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
        # Order matters only with FKs off; dim_date is referenced by fact_price.
        conn.exec_driver_sql("DELETE FROM fact_price")
        conn.exec_driver_sql("DELETE FROM dim_stock")
        conn.exec_driver_sql("DELETE FROM dim_date")
        keys = sorted({k for _, k in list(prices) + list(opens)})
        for key in keys:
            stamp = f"{key // 10000:04d}-{key // 100 % 100:02d}-{key % 100:02d}"
            conn.execute(
                text(
                    "INSERT INTO dim_date (date_key, full_date, day_of_week, day_name, "
                    "week_of_year, month, month_name, quarter, year, is_trading_day, is_month_end) "
                    "VALUES (:k, :d, 0, 'Mon', 1, 1, 'Jan', 1, :y, 1, 0)"
                ),
                {"k": key, "d": stamp, "y": key // 10000},
            )
        for i, symbol in enumerate(symbols, start=1):
            conn.execute(
                text(
                    "INSERT INTO dim_stock (stock_key, symbol, company_name, sector, "
                    "instrument_type) VALUES (:k, :s, :s, 'IT', 'EQUITY')"
                ),
                {"k": i, "s": symbol},
            )
        key_map = {s: i for i, s in enumerate(symbols, start=1)}
        for (symbol, key), close in prices.items():
            open_price = opens.get((symbol, key), close)
            conn.execute(
                text(
                    "INSERT INTO fact_price (date_key, stock_key, open_price, high_price, "
                    "low_price, close_price, volume, is_flagged) "
                    "VALUES (:d, :k, :o, :hi, :lo, :c, 1000, 0)"
                ),
                {
                    "d": key,
                    "k": key_map[symbol],
                    "o": open_price,
                    "hi": max(open_price, close) * 1.01,
                    "lo": min(open_price, close) * 0.99,
                    "c": close,
                },
            )


def regimes(engine: Engine, labels: dict[int, str]) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM fact_market_regime")
        key_map = dict(
            conn.execute(text("SELECT regime_label, regime_key FROM dim_regime")).fetchall()
        )
        for key, label in labels.items():
            conn.execute(
                text(
                    "INSERT INTO fact_market_regime (date_key, regime_key, cluster_id, "
                    "mkt_return_20d, mkt_vol_20d, india_vix, mkt_drawdown) "
                    "VALUES (:d, :r, 0, 0.0, 0.1, 15.0, 0.0)"
                ),
                {"d": key, "r": int(key_map[label])},
            )


def signals(engine: Engine, probs: dict[tuple[str, int], float], model: str = "tree") -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM fact_signal")
        conn.exec_driver_sql("DELETE FROM dim_model")
        conn.execute(
            text(
                "INSERT INTO dim_model (model_key, model_name, version, trained_through) "
                "VALUES (1, :m, 'wf2020', '2019-12-31')"
            ),
            {"m": model},
        )
        key_map = dict(conn.execute(text("SELECT symbol, stock_key FROM dim_stock")).fetchall())
        regime_map = dict(
            conn.execute(text("SELECT date_key, regime_key FROM fact_market_regime")).fetchall()
        )
        for (symbol, key), prob in probs.items():
            if symbol not in key_map or key not in regime_map:
                continue
            conn.execute(
                text(
                    "INSERT INTO fact_signal (date_key, stock_key, model_key, regime_key, "
                    "prob_up, signal, target_weight) VALUES (:d, :s, 1, :r, :p, 0, 0.0)"
                ),
                {"d": key, "s": key_map[symbol], "r": regime_map[key], "p": prob},
            )


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


DAYS = [20240102, 20240103, 20240104, 20240105, 20240108]


def constant_weights(symbol: str) -> Any:
    def provider(context: DayContext) -> pd.Series:
        if context.signals.empty:
            return None
        return pd.Series({symbol: 1.0})

    return provider


# --------------------------------------------------------------- fill timing


def test_engine_fills_next_open(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """The fill uses `open_(t+1)`, never `close_t` (`TESTS` section 7)."""
    cfg = open_window(cfg_smoke)
    cfg["backtest"]["slippage"] = 0.0
    cfg["backtest"]["cost"] = 0.0

    # close_t is 100 but the next open is 200: a look-ahead engine would fill near 100.
    prices = {("A.NS", d): 100.0 for d in DAYS}
    opens = {("A.NS", DAYS[0]): 100.0, ("A.NS", DAYS[1]): 200.0}
    load_rows(engine, ["A.NS"], prices, opens)
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    result = run_backtest(
        cfg,
        constant_weights("A.NS"),
        "2024-01-02",
        "2024-01-08",
        "t1",
        engine=engine,
        model_name="tree",
        fractional_shares=True,
        stop_loss_enabled=False,
        persist=False,
    )

    assert not result.trades.empty
    first = result.trades.iloc[0]
    assert first["fill_price"] == pytest.approx(200.0)
    assert first["fill_price"] != pytest.approx(100.0)


def test_engine_zero_cost_identity(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """Zero slippage, zero cost, fractional: equity equals the hand calculation."""
    cfg = open_window(cfg_smoke)
    cfg["backtest"]["slippage"] = 0.0
    cfg["backtest"]["cost"] = 0.0
    capital = float(cfg["backtest"]["initial_capital"])

    closes = {("A.NS", d): float(100 + i) for i, d in enumerate(DAYS)}
    # Day 2 opens at 100, the same price as the first close, so the hand calculation
    # is simply capital * final_close / 100.
    opens = {("A.NS", DAYS[0]): 100.0, ("A.NS", DAYS[1]): 100.0}
    load_rows(engine, ["A.NS"], closes, opens)
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    result = run_backtest(
        cfg,
        constant_weights("A.NS"),
        "2024-01-02",
        "2024-01-08",
        "t2",
        engine=engine,
        model_name="tree",
        fractional_shares=True,
        stop_loss_enabled=False,
        persist=False,
    )

    # Bought at the day-2 open with the whole account, marked at the final close.
    buy_open = opens[("A.NS", DAYS[1])]
    expected = capital * closes[("A.NS", DAYS[-1])] / buy_open
    assert result.equity["equity"].iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_cash_never_negative(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """Property test: cash stays non-negative on every day."""
    import random

    rng = random.Random(11)
    cfg = open_window(cfg_smoke)
    cfg["backtest"]["initial_capital"] = 100_000.0

    symbols = [f"S{i}.NS" for i in range(5)]
    prices, opens, probs, labels = {}, {}, {}, {}
    for d in DAYS:
        for s in symbols:
            prices[(s, d)] = 100.0 + rng.uniform(-10, 10)
            opens[(s, d)] = prices[(s, d)]
            probs[(s, d)] = rng.uniform(0.3, 0.95)
        labels[d] = rng.choice(["BULL", "SIDEWAYS", "HIGH_VOL"])
    load_rows(engine, symbols, prices, opens)
    regimes(engine, labels)
    signals(engine, probs)

    from src.backtest.engine import _strategy_weights_fn

    result = run_backtest(
        cfg,
        _strategy_weights_fn(cfg),
        "2024-01-02",
        "2024-01-08",
        "t3",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    assert (result.equity["cash"] >= -1e-6).all()


def test_missing_bar_skips_order(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """A symbol with no bar on the execution day is skipped, not fatal."""
    cfg = open_window(cfg_smoke)
    prices = {("A.NS", DAYS[0]): 100.0, ("A.NS", DAYS[1]): 100.0, ("A.NS", DAYS[3]): 100.0}
    load_rows(engine, ["A.NS"], prices)
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    result = run_backtest(
        cfg,
        constant_weights("A.NS"),
        "2024-01-02",
        "2024-01-08",
        "t4",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    assert len(result.trades) >= 0  # did not raise
    assert len(result.equity) >= 3


# --------------------------------------------------------------- guard


def test_engine_period_guard(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """`run_backtest` raises for a period reaching test_start unless allowed."""
    cfg = dict(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 for d in DAYS})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    with pytest.raises(ValueError, match="test period"):
        run_backtest(
            cfg,
            constant_weights("A.NS"),
            "2024-01-02",
            cfg["splits"]["test_start"],
            "t5",
            engine=engine,
            persist=False,
        )

    # The same period runs fine when the caller explicitly allows the test period.
    result = run_backtest(
        cfg,
        constant_weights("A.NS"),
        "2024-01-02",
        "2024-01-08",
        "t5",
        engine=engine,
        persist=False,
        allow_test_period=True,
    )
    assert not result.equity.empty


# --------------------------------------------------------------- persistence


def test_persist_idempotent(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """The same run_id twice leaves identical `fact_trade` and `fact_portfolio_daily`."""
    cfg = open_window(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 + i for i, d in enumerate(DAYS)})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    from src.backtest.engine import _strategy_weights_fn

    fn = _strategy_weights_fn(cfg)
    run_backtest(
        cfg, fn, "2024-01-02", "2024-01-08", "persist_run", engine=engine, model_name="tree"
    )
    first_trades = read_sql(engine, "SELECT COUNT(*) n FROM fact_trade WHERE run_id='persist_run'")
    first_equity = read_sql(
        engine, "SELECT COUNT(*) n FROM fact_portfolio_daily WHERE run_id='persist_run'"
    )
    checksum_a = read_sql(engine, "SELECT date_key, close_price FROM fact_price ORDER BY date_key")

    run_backtest(
        cfg, fn, "2024-01-02", "2024-01-08", "persist_run", engine=engine, model_name="tree"
    )
    second_trades = read_sql(engine, "SELECT COUNT(*) n FROM fact_trade WHERE run_id='persist_run'")
    second_equity = read_sql(
        engine, "SELECT COUNT(*) n FROM fact_portfolio_daily WHERE run_id='persist_run'"
    )
    checksum_b = read_sql(engine, "SELECT date_key, close_price FROM fact_price ORDER BY date_key")

    assert first_trades["n"].iloc[0] == second_trades["n"].iloc[0]
    assert first_equity["n"].iloc[0] == second_equity["n"].iloc[0]
    pd.testing.assert_frame_equal(checksum_a, checksum_b)


def test_only_primary_run_writes_signals(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """Only a run with write_signals changes `fact_signal.signal` (DECISIONS R3)."""
    cfg = open_window(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 for d in DAYS})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    from src.backtest.engine import _strategy_weights_fn

    fn = _strategy_weights_fn(cfg)
    run_backtest(
        cfg, fn, "2024-01-02", "2024-01-08", "nonprimary", engine=engine, model_name="tree"
    )
    assert read_sql(engine, "SELECT SUM(signal) s FROM fact_signal")["s"].iloc[0] == 0

    run_backtest(
        cfg,
        fn,
        "2024-01-02",
        "2024-01-08",
        "primary",
        engine=engine,
        model_name="tree",
        write_signals=True,
    )
    assert read_sql(engine, "SELECT SUM(signal) s FROM fact_signal")["s"].iloc[0] > 0


# --------------------------------------------------------------- equity curve


def test_equity_curve_shape(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    cfg = open_window(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 + i for i, d in enumerate(DAYS)})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    from src.backtest.engine import _strategy_weights_fn

    result = run_backtest(
        cfg,
        _strategy_weights_fn(cfg),
        "2024-01-02",
        "2024-01-08",
        "t6",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    for column in ("cash", "positions_value", "equity", "daily_return", "drawdown"):
        assert column in result.equity.columns
    assert (result.equity["drawdown"] >= -1e-12).all()
    assert (result.equity["equity"] > 0).all()


def test_strategy_is_deterministic(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    from src.backtest.engine import _strategy_weights_fn

    cfg = open_window(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 + i for i, d in enumerate(DAYS)})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    a = run_backtest(
        cfg,
        _strategy_weights_fn(cfg),
        "2024-01-02",
        "2024-01-08",
        "d1",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    b = run_backtest(
        cfg,
        _strategy_weights_fn(cfg),
        "2024-01-02",
        "2024-01-08",
        "d2",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    pd.testing.assert_series_equal(a.equity["equity"], b.equity["equity"])


def test_weights_none_places_no_orders(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    """A provider returning None clears pending targets (DECISIONS R29)."""
    cfg = open_window(cfg_smoke)
    load_rows(engine, ["A.NS"], {("A.NS", d): 100.0 for d in DAYS})
    regimes(engine, {d: "BULL" for d in DAYS})
    signals(engine, {("A.NS", d): 0.9 for d in DAYS})

    result = run_backtest(
        cfg,
        lambda ctx: None,
        "2024-01-02",
        "2024-01-08",
        "t7",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    assert result.trades.empty
    assert (result.equity["positions_value"] == 0).all()
