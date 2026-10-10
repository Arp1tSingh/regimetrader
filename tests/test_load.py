"""Tests for the warehouse load (task T1.5, spec 03 section 4.3, TESTS section 3)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import DataError, IntegrityError, StatementError

from src.common.db import get_engine, init_db, read_sql
from src.etl.load import load, price_checksum
from src.etl.synthetic import generate_market
from src.etl.transform import attach_instrument_types, transform

INDEX = "^NSEI"


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


@pytest.fixture
def market(cfg_smoke: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """A small synthetic market, transformed and ready to load."""
    cfg_smoke["synthetic"]["n_stocks"] = 3
    cfg_smoke["data"]["start_date"] = "2023-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2023-06-30"
    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    raw = attach_instrument_types(built.prices, types)
    clean, rejected = transform(raw, cfg_smoke)
    return (
        clean,
        built.universe,
        {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)},
    )


# --------------------------------------------------------------- basic load


def test_load_writes_prices(
    engine: Engine,
    market: tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]],
    cfg_smoke: dict[str, Any],
) -> None:
    clean, universe, stats = market
    result = load(engine, clean, universe, "FULL", stats, cfg_smoke)

    assert result["rows_loaded"] == len(clean)
    stored = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")
    assert stored["n"].iloc[0] == len(clean)


def test_load_writes_dim_stock(engine: Engine, market: tuple, cfg_smoke: dict[str, Any]) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    stored = read_sql(engine, "SELECT symbol, instrument_type FROM dim_stock ORDER BY symbol")
    assert len(stored) == len(universe)
    assert set(stored["instrument_type"]) == {"EQUITY", "INDEX", "VOLATILITY"}


def test_load_writes_dim_date(engine: Engine, market: tuple, cfg_smoke: dict[str, Any]) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    dim = read_sql(engine, "SELECT date_key, is_trading_day, is_month_end FROM dim_date")
    assert not dim.empty
    assert (dim["is_month_end"].sum()) > 0


def test_month_end_is_last_trading_day(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    """DECISIONS R20: is_month_end is the last trading day present in each month."""
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    rows = read_sql(
        engine,
        "SELECT full_date FROM dim_date "
        "WHERE is_trading_day = 1 AND is_month_end = 1 ORDER BY full_date",
    )
    trading = read_sql(
        engine, "SELECT full_date FROM dim_date WHERE is_trading_day = 1 ORDER BY full_date"
    )
    for date in rows["full_date"]:
        month = date[:7]
        later = trading[trading["full_date"] > date]
        assert later.empty or later["full_date"].iloc[0][:7] != month


def test_run_log_records_success(engine: Engine, market: tuple, cfg_smoke: dict[str, Any]) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    log = read_sql(engine, "SELECT mode, status, rows_loaded FROM etl_run_log")
    assert len(log) == 1
    assert log["status"].iloc[0] == "SUCCESS"
    assert log["mode"].iloc[0] == "FULL"


# --------------------------------------------------------------- idempotency


def test_load_idempotent(engine: Engine, market: tuple, cfg_smoke: dict[str, Any]) -> None:
    """Loading the same clean frame twice leaves identical counts and checksum."""
    clean, universe, stats = market

    load(engine, clean, universe, "FULL", stats, cfg_smoke)
    first_count = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")["n"].iloc[0]
    first_sum = price_checksum(engine)

    load(engine, clean, universe, "FULL", stats, cfg_smoke)
    second_count = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")["n"].iloc[0]
    second_sum = price_checksum(engine)

    assert first_count == second_count
    assert first_sum == second_sum


def test_incremental_and_full_agree(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "INCREMENTAL", stats, cfg_smoke)
    a = price_checksum(engine)

    load(engine, clean, universe, "FULL", stats, cfg_smoke)
    assert price_checksum(engine) == a


def test_updated_prices_are_overwritten(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    """The upsert updates an existing (date_key, stock_key) rather than duplicating it."""
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    changed = clean.copy()
    key = (changed["symbol"].iloc[0], changed["date_key"].iloc[0])
    changed.loc[(changed["symbol"] == key[0]) & (changed["date_key"] == key[1]), "close"] = 999.0
    load(engine, clean=changed, universe=universe, mode="FULL", stats=stats, cfg=cfg_smoke)

    row = read_sql(
        engine,
        "SELECT close_price FROM fact_price WHERE stock_key = "
        "(SELECT stock_key FROM dim_stock WHERE symbol = :s) AND date_key = :d",
        {"s": key[0], "d": int(key[1])},
    )
    assert row["close_price"].iloc[0] == pytest.approx(999.0)


# --------------------------------------------------------------- rollback


def test_load_rollback(engine: Engine, market: tuple, cfg_smoke: dict[str, Any]) -> None:
    """An injected error rolls back all rows and still logs FAILED."""
    clean, universe, stats = market

    broken = clean.copy()
    broken["close"] = "not-a-number"  # fails conversion inside the transaction

    with pytest.raises((StatementError, DataError, IntegrityError, ValueError)):
        load(engine, broken, universe, "FULL", stats, cfg_smoke)

    stored = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")
    assert stored["n"].iloc[0] == 0, "the failed load must leave no rows"

    log = read_sql(engine, "SELECT status FROM etl_run_log")
    assert log["status"].iloc[-1] == "FAILED"


def test_rollback_after_partial_success(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    """Dimensions written before the failure are rolled back too."""
    clean, universe, stats = market
    # A duplicate primary key on a NOT NULL column cannot be inserted.
    bad = clean.head(50).copy()
    bad["date_key"] = bad["date_key"].astype(object)
    bad.loc[bad.index[-1], "date_key"] = None

    with pytest.raises((StatementError, DataError, IntegrityError, ValueError)):
        load(engine, bad, universe, "FULL", stats, cfg_smoke)

    stocks = read_sql(engine, "SELECT COUNT(*) AS n FROM dim_stock")
    assert stocks["n"].iloc[0] == 0


# --------------------------------------------------------------- guards


def test_invalid_mode_raises(engine: Engine, market: tuple) -> None:
    clean, universe, stats = market
    with pytest.raises(ValueError, match="FULL or INCREMENTAL"):
        load(engine, clean, universe, "SOMETIMES", stats)


def test_unknown_symbol_is_skipped_not_fatal(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    """A price row for a symbol absent from dim_stock is skipped, not inserted."""
    clean, universe, stats = market
    stray = clean.copy()
    stray.loc[stray.index[0], "symbol"] = "GHOST.NS"

    result = load(engine, stray, universe, "FULL", stats, cfg_smoke)
    assert result["rows_loaded"] == len(clean) - 1


def test_flagged_rows_are_persisted(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    clean, universe, stats = market
    clean = clean.copy()
    clean.loc[clean.index[0], "is_flagged"] = 1
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    flagged = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price WHERE is_flagged = 1")
    assert flagged["n"].iloc[0] >= 1


def test_checksum_changes_when_data_changes(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)
    before = price_checksum(engine)

    clean = clean.copy()
    clean.loc[clean.index[0], "close"] = float(clean.loc[clean.index[0], "close"]) + 1.0
    load(engine, clean, universe, "FULL", stats, cfg_smoke)

    assert price_checksum(engine) != before


def test_foreign_keys_hold_after_load(
    engine: Engine, market: tuple, cfg_smoke: dict[str, Any]
) -> None:
    clean, universe, stats = market
    load(engine, clean, universe, "FULL", stats, cfg_smoke)
    orphans = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM fact_price f "
        "LEFT JOIN dim_stock s ON f.stock_key = s.stock_key "
        "LEFT JOIN dim_date d ON f.date_key = d.date_key "
        "WHERE s.stock_key IS NULL OR d.date_key IS NULL",
    )
    assert orphans["n"].iloc[0] == 0
