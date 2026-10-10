"""Warehouse schema conformance (task T0.5, spec 03 section 2).

Every table documented in `03` section 2 must exist with the documented grain and
primary key, and `dim_regime` must carry its four seed rows.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import Engine, inspect

from src.common.db import get_engine, init_db

# (table, kind, primary key columns) exactly as documented in 03 section 2.
EXPECTED_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("stg_price_raw", "staging", ()),
    ("etl_run_log", "audit", ("run_id",)),
    ("dim_date", "dimension", ("date_key",)),
    ("dim_stock", "dimension", ("stock_key",)),
    ("dim_regime", "dimension", ("regime_key",)),
    ("dim_model", "dimension", ("model_key",)),
    ("fact_price", "fact", ("date_key", "stock_key")),
    ("fact_indicator", "fact", ("date_key", "stock_key")),
    ("fact_market_regime", "fact", ("date_key",)),
    ("fact_signal", "fact", ("date_key", "stock_key", "model_key")),
    ("fact_trade", "fact", ("trade_id",)),
    ("fact_portfolio_daily", "fact", ("run_id", "date_key")),
    ("pending_target", "state", ("run_id", "decision_date_key", "stock_key")),
)

#: Tables that must not carry a primary key.
UNKEYED = ("stg_price_raw",)


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


@pytest.mark.parametrize(
    ("table", "_kind", "_pk"), EXPECTED_TABLES, ids=[t[0] for t in EXPECTED_TABLES]
)
def test_table_exists(engine: Engine, table: str, _kind: str, _pk: tuple[str, ...]) -> None:
    assert table in inspect(engine).get_table_names()


@pytest.mark.parametrize(
    ("table", "_kind", "pk"), EXPECTED_TABLES, ids=[t[0] for t in EXPECTED_TABLES]
)
def test_primary_key_matches_spec(
    engine: Engine, table: str, _kind: str, pk: tuple[str, ...]
) -> None:
    inspector = inspect(engine)
    actual = tuple(inspector.get_pk_constraint(table)["constrained_columns"])
    if table in UNKEYED:
        assert actual == ()
    else:
        assert actual == pk


def test_no_extra_tables(engine: Engine) -> None:
    """The warehouse holds exactly the documented tables, nothing accumulated by accident."""
    names = set(inspect(engine).get_table_names())
    assert names == {table for table, _, _ in EXPECTED_TABLES}


def test_dim_regime_seed_rows(engine: Engine) -> None:
    from src.common.db import read_sql

    df = read_sql(
        engine,
        "SELECT regime_key, regime_label, description FROM dim_regime ORDER BY regime_key",
    )
    assert list(df["regime_label"]) == ["BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"]
    assert list(df["regime_key"]) == [1, 2, 3, 4]
    assert df["description"].notna().all()


def test_dim_stock_instrument_type_check(engine: Engine) -> None:
    """`03` section 2 constrains instrument_type to EQUITY, INDEX, VOLATILITY."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO dim_stock (symbol, company_name, sector, instrument_type) "
                "VALUES ('X.NS', 'X', 'IT', 'BOND')"
            )


def test_fact_trade_reason_check(engine: Engine) -> None:
    """`DECISIONS` R25: every trade carries a reason from the documented set."""
    from sqlalchemy.exc import IntegrityError

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO dim_date VALUES "
            "(20240101, '2024-01-01', 0, 'Mon', 1, 1, 'Jan', 1, 2024, 1, 0)"
        )
        conn.exec_driver_sql(
            "INSERT INTO dim_stock (symbol, company_name, sector, instrument_type) "
            "VALUES ('A.NS', 'A', 'IT', 'EQUITY')"
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO fact_trade "
                "(run_id, date_key, stock_key, side, quantity, fill_price, cost, reason) "
                "VALUES ('r1', 20240101, 1, 'BUY', 10, 100.0, 1.0, 'HAPPENED')"
            )


def test_date_key_is_integer(engine: Engine) -> None:
    """Conventions: date_key is an integer YYYYMMDD, never a float or string."""
    from sqlalchemy import text

    with engine.connect() as conn:
        assert conn.execute(text("SELECT typeof(date_key) FROM dim_date")).scalar() != "real"


def test_indexes_created(engine: Engine) -> None:
    names = {idx["name"] for idx in inspect(engine).get_indexes("fact_price")}
    assert "idx_price_stock_date" in names
    signal_idx = {idx["name"] for idx in inspect(engine).get_indexes("fact_signal")}
    assert signal_idx == {"idx_signal_date"}
