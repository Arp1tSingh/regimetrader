"""Tests for the data-quality checks (task T1.6, spec 03 section 5, TESTS section 3).

`test_dq_checks_pass_and_fail` requires every DQ-01 to DQ-07 to pass on clean data
and fail on injected bad data; hard ones must raise through `run_checks`.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from sqlalchemy import Engine, text

from src.common.db import get_engine, init_db
from src.etl.load import load
from src.etl.quality import (
    DataQualityError,
    check_date_references,
    check_duplicates,
    check_flagged_share,
    check_price_validity,
    check_rejected_share,
    check_row_coverage,
    check_staleness,
    run_checks,
)
from src.etl.synthetic import generate_market
from src.etl.transform import attach_instrument_types, transform


@pytest.fixture
def loaded(
    engine: Engine, cfg_smoke: dict[str, Any]
) -> tuple[Engine, dict[str, Any], dict[str, int]]:
    """A warehouse holding a small synthetic market, and the load stats."""
    cfg_smoke["synthetic"]["n_stocks"] = 4
    cfg_smoke["data"]["start_date"] = "2023-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2023-06-30"
    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    clean, rejected = transform(attach_instrument_types(built.prices, types), cfg_smoke)
    stats = {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)}
    load(engine, clean, built.universe, "FULL", stats, cfg_smoke)
    return engine, cfg_smoke, stats


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


# ---------------------------------------------------------------- DQ-01


def test_dq01_passes_on_clean_data(loaded: tuple) -> None:
    engine, _, _ = loaded
    assert check_duplicates(engine).passed


def test_dq01_fails_on_duplicate(loaded: tuple) -> None:
    """The primary key makes duplicates impossible through the normal path, so DQ-01
    is a safety net. Rebuild the table without the key on this scratch database to
    prove the check still detects one."""
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = OFF")
        conn.exec_driver_sql("ALTER TABLE fact_price RENAME TO fact_price_backup")
        conn.exec_driver_sql(
            """
            CREATE TABLE fact_price (
                date_key    INTEGER NOT NULL,
                stock_key   INTEGER NOT NULL,
                open_price  REAL, high_price REAL, low_price REAL, close_price REAL,
                volume      INTEGER,
                is_flagged  INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.exec_driver_sql("INSERT INTO fact_price SELECT * FROM fact_price_backup")
        conn.exec_driver_sql("INSERT INTO fact_price SELECT * FROM fact_price_backup LIMIT 1")
        conn.exec_driver_sql("DROP TABLE fact_price_backup")
    assert not check_duplicates(engine).passed


# ---------------------------------------------------------------- DQ-02


def test_dq02_passes_on_clean_data(loaded: tuple) -> None:
    engine, _, _ = loaded
    assert check_price_validity(engine).passed


def test_dq02_fails_on_non_positive_price(loaded: tuple) -> None:
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE fact_price SET close_price = 0 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    assert not check_price_validity(engine).passed


def test_dq02_fails_on_inverted_range(loaded: tuple) -> None:
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE fact_price SET high_price = low_price - 1 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    assert not check_price_validity(engine).passed


def test_dq02_fails_when_high_below_close(loaded: tuple) -> None:
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE fact_price SET high_price = close_price * 0.5 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    assert not check_price_validity(engine).passed


# ---------------------------------------------------------------- DQ-03


def test_dq03_passes_on_clean_data(loaded: tuple) -> None:
    engine, _, _ = loaded
    assert check_date_references(engine).passed


def test_dq03_fails_on_non_trading_date(loaded: tuple) -> None:
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE dim_date SET is_trading_day = 0 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    assert not check_date_references(engine).passed


def test_dq03_fails_on_unknown_date_key(loaded: tuple) -> None:
    """A price row whose date_key is absent from dim_date. The foreign key blocks
    this normally, so it is injected with FKs off."""
    engine, _, _ = loaded
    with engine.begin() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(
            text(
                "INSERT INTO fact_price "
                "SELECT 20991231, stock_key, open_price, high_price, low_price, "
                "       close_price, volume, is_flagged FROM fact_price "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price) LIMIT 1"
            )
        )
    assert not check_date_references(engine).passed


# ---------------------------------------------------------------- DQ-04


def test_dq04_passes_on_clean_data(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    assert check_row_coverage(engine, cfg).passed


def test_dq04_fails_when_a_stock_is_missing_bars(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM fact_price WHERE stock_key = "
                "(SELECT stock_key FROM dim_stock WHERE symbol != :index_symbol LIMIT 1) "
                "AND date_key > (SELECT MIN(date_key) FROM fact_price)"
            ),
            {"index_symbol": cfg["data"]["index_symbol"]},
        )
    assert not check_row_coverage(engine, cfg).passed


def test_dq04_tolerates_short_history_stocks(loaded: tuple) -> None:
    """DECISIONS R10: a stock listed later is not penalised, counted from its own first bar."""
    engine, cfg, _ = loaded
    result = check_row_coverage(engine, cfg)
    assert "under 10 years" in result.detail


# ---------------------------------------------------------------- DQ-05


def test_dq05_passes_on_clean_data(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    assert check_flagged_share(engine, cfg).passed


def test_dq05_fails_when_too_many_flagged(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    cfg["quality"]["max_flagged_share"] = 0.0
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE fact_price SET is_flagged = 1 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    assert not check_flagged_share(engine, cfg).passed


# ---------------------------------------------------------------- DQ-06


def test_dq06_skipped_for_synthetic(loaded: tuple) -> None:
    """Synthetic data has no relationship to today, so the check is skipped."""
    engine, cfg, _ = loaded
    result = check_staleness(engine, cfg)
    assert result.passed
    assert "synthetic" in result.detail


def test_dq06_passes_when_fresh(loaded: tuple) -> None:
    """The synthetic data ends 2023-06-30, so a reference date near it is fresh."""
    engine, cfg, _ = loaded
    cfg["data"]["source"] = "yfinance"
    cfg["data"]["stale_after_days"] = 5

    result = check_staleness(engine, cfg, today=date(2023, 7, 3))
    assert result.passed
    assert "old" in result.detail


def test_dq06_fails_when_stale(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    cfg["data"]["source"] = "yfinance"
    result = check_staleness(engine, cfg, today=date(2030, 1, 1))
    assert not result.passed
    assert "old" in result.detail


def test_dq06_fails_on_empty_warehouse(engine: Engine, cfg_smoke: dict[str, Any]) -> None:
    cfg_smoke["data"]["source"] = "yfinance"
    assert not check_staleness(engine, cfg_smoke).passed


# ---------------------------------------------------------------- DQ-07


def test_dq07_passes_on_clean_data(loaded: tuple) -> None:
    engine, cfg, stats = loaded
    assert check_rejected_share(engine, cfg, stats).passed


def test_dq07_fails_when_too_many_rejected(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    cfg["quality"]["max_rejected_share"] = 0.0
    assert not check_rejected_share(engine, cfg, {"rows_extracted": 100, "rows_rejected": 5}).passed


def test_dq07_with_no_stats_falls_back_to_row_count(loaded: tuple) -> None:
    engine, cfg, _ = loaded
    assert check_rejected_share(engine, cfg, None).passed


# ---------------------------------------------------------------- runner


def test_run_checks_returns_all_seven(loaded: tuple) -> None:
    engine, cfg, stats = loaded
    results = run_checks(engine, cfg, stats)
    assert [r.id for r in results] == [f"DQ-0{i}" for i in range(1, 8)]
    assert all(r.passed for r in results)


def test_run_checks_raises_on_hard_failure(loaded: tuple) -> None:
    engine, cfg, stats = loaded
    with engine.begin() as conn:
        conn.execute(text("UPDATE fact_price SET close_price = -1"))
    with pytest.raises(DataQualityError, match="DQ-02"):
        run_checks(engine, cfg, stats)


def test_run_checks_does_not_raise_on_soft_failure(loaded: tuple) -> None:
    """Soft failures are warnings; only hard ones stop the pipeline."""
    engine, cfg, stats = loaded
    cfg["quality"]["max_flagged_share"] = 0.0
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE fact_price SET is_flagged = 1 "
                "WHERE date_key = (SELECT MIN(date_key) FROM fact_price)"
            )
        )
    # run_checks must not raise even though DQ-05 now fails.
    results = run_checks(engine, cfg, stats)
    flagged = next(r for r in results if r.id == "DQ-05")
    assert not flagged.passed
    assert flagged.severity == "SOFT"


def test_check_result_as_row(loaded: tuple) -> None:
    engine, cfg, stats = loaded
    row = run_checks(engine, cfg, stats)[0].as_row()
    assert set(row) == {"id", "severity", "passed", "detail"}
    assert row["passed"] == "PASS"
    assert row["severity"] in ("HARD", "SOFT")


def test_severity_matches_spec() -> None:
    from src.etl.quality import SEVERITY

    assert SEVERITY["DQ-01"] == "HARD"
    assert SEVERITY["DQ-03"] == "HARD"
    assert SEVERITY["DQ-04"] == "SOFT"
    assert all(SEVERITY[f"DQ-0{i}"] == "SOFT" for i in range(4, 8))
