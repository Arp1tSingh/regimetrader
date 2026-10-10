"""Tests for the cross-cutting utilities in `src.common` (task T0.4, spec 07 section 4)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect, text

from src.common import dates, seeds
from src.common.config import config_hash, load_config
from src.common.db import execute_many, get_engine, init_db, read_sql
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.common.trial_log import count_trials, log_trial

from .conftest import BASE_CONFIG

# ---------------------------------------------------------------- dates


@pytest.mark.parametrize(
    ("ts", "expected"),
    [
        (pd.Timestamp("2024-01-05"), 20240105),
        (pd.Timestamp("2015-12-31 15:30:00"), 20151231),
        (np.datetime64("2019-03-01"), 20190301),
        ("2020-06-15", 20200615),
        (np.datetime64("2021-01-01", "D"), 20210101),
    ],
)
def test_to_date_key(ts: Any, expected: int) -> None:
    """Date keys are int(YYYYMMDD), never floats or strings (AGENTS.md section 6)."""
    key = dates.to_date_key(ts)
    assert key == expected
    assert isinstance(key, int) and not isinstance(key, bool)


def test_to_date_key_ignores_time_of_day() -> None:
    """yfinance is tz-aware; tz_localize(None) then the key must not shift a day."""
    aware = pd.Timestamp("2024-01-05 18:15:00+05:30")
    naive = aware.tz_localize(None)
    assert dates.to_date_key(aware) == dates.to_date_key(naive) == 20240105


@pytest.mark.parametrize("key", [20240105, "20240105", 19991231])
def test_from_date_key(key: int) -> None:
    ts = dates.from_date_key(key)
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("2024-01-05") if key == 20240105 else True


def test_from_date_key_accepts_string() -> None:
    assert dates.from_date_key("20240105") == pd.Timestamp("2024-01-05")


def test_date_key_roundtrip() -> None:
    for ts in pd.date_range("2015-01-01", "2025-06-30", freq="ME"):
        assert dates.from_date_key(dates.to_date_key(ts)) == ts


# ---------------------------------------------------------------- db


def test_init_db_is_idempotent(cfg_smoke: dict[str, Any]) -> None:
    """Applying schema.sql twice must not error."""
    engine = get_engine(cfg_smoke)
    init_db(engine)
    init_db(engine)

    tables = set(inspect(engine).get_table_names())
    for expected in (
        "stg_price_raw", "etl_run_log", "dim_date", "dim_stock", "dim_regime",
        "dim_model", "fact_price", "fact_indicator", "fact_market_regime",
        "fact_signal", "fact_trade", "fact_portfolio_daily", "pending_target",
    ):
        assert expected in tables


def test_dim_regime_has_four_seed_rows(cfg_smoke: dict[str, Any]) -> None:
    engine = get_engine(cfg_smoke)
    init_db(engine)
    df = read_sql(engine, "SELECT regime_key, regime_label FROM dim_regime ORDER BY regime_key")
    assert list(df["regime_label"]) == ["BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"]


def test_foreign_keys_are_on(cfg_smoke: dict[str, Any]) -> None:
    """SQLite has foreign keys off by default; the pragma must be applied (AGENTS.md 6)."""
    engine = get_engine(cfg_smoke)
    init_db(engine)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1


def test_foreign_key_violation_is_rejected(cfg_smoke: dict[str, Any]) -> None:
    engine = get_engine(cfg_smoke)
    init_db(engine)
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO fact_price (date_key, stock_key, close_price) "
                    "VALUES (:d, :s, 1.0)"
                ),
                {"d": 20991231, "s": 999},  # neither dim row exists
            )


def test_wal_mode_is_set(cfg_smoke: dict[str, Any]) -> None:
    engine = get_engine(cfg_smoke)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar().lower() == "wal"


def test_read_sql_and_execute_many(cfg_smoke: dict[str, Any]) -> None:
    engine = get_engine(cfg_smoke)
    init_db(engine)
    with engine.begin() as conn:
        written = execute_many(
            conn,
            "INSERT INTO dim_stock (symbol, company_name, sector, instrument_type) "
            "VALUES (:symbol, :name, :sector, :type)",
            [
                {"symbol": "AAA.NS", "name": "Alpha", "sector": "IT", "type": "EQUITY"},
                {"symbol": "BBB.NS", "name": "Beta", "sector": "Banking", "type": "EQUITY"},
            ],
        )
    assert written == 2
    df = read_sql(
        engine,
        "SELECT symbol FROM dim_stock WHERE sector = :s ORDER BY symbol",
        {"s": "IT"},
    )
    assert list(df["symbol"]) == ["AAA.NS"]


def test_read_sql_empty_result_is_empty_frame(cfg_smoke: dict[str, Any]) -> None:
    engine = get_engine(cfg_smoke)
    init_db(engine)
    df = read_sql(engine, "SELECT symbol FROM dim_stock")
    assert df.empty and "symbol" in df.columns


# ---------------------------------------------------------------- seeds


def test_seed_everything_is_reproducible() -> None:
    cfg = load_config(str(BASE_CONFIG))
    seeds.seed_everything(cfg)
    a = (random.random(), float(np.random.rand()), pd.Series(np.random.rand(3)).tolist())
    seeds.seed_everything(cfg)
    b = (random.random(), float(np.random.rand()), pd.Series(np.random.rand(3)).tolist())
    assert a == b


def test_seed_everything_uses_config_seed() -> None:
    cfg = load_config(str(BASE_CONFIG))
    cfg["project"]["seed"] = 123
    seeds.seed_everything(cfg)
    expected = random.Random(123).random()
    seeds.seed_everything(cfg)
    assert random.random() == expected


# ---------------------------------------------------------------- trial_log


def test_log_trial_writes_and_upserts(cfg_smoke: dict[str, Any]) -> None:
    """Rows are upserted on the key, so reruns never double-count (DECISIONS R28)."""
    params = {"entry_base": 0.55, "slots": 10}
    log_trial(cfg_smoke, phase="dev", kind="strategy_grid", run_id="r1", model="rf",
              params=params, period="validation", metric="sharpe", value=0.9)
    log_trial(cfg_smoke, phase="dev", kind="strategy_grid", run_id="r1", model="rf",
              params=params, period="validation", metric="sharpe", value=0.9)

    df = read_sql_path(cfg_smoke, "trial_log.csv")
    assert len(df) == 1
    assert df.iloc[0]["value"] == 0.9
    assert count_trials(cfg_smoke) == 1


def test_log_trial_distinct_configs_count_separately(cfg_smoke: dict[str, Any]) -> None:
    log_trial(cfg_smoke, phase="dev", kind="strategy_grid", run_id="r1", model="rf",
              params={"entry_base": 0.55}, period="validation", metric="sharpe", value=0.9)
    log_trial(cfg_smoke, phase="dev", kind="strategy_grid", run_id="r1", model="rf",
              params={"entry_base": 0.60}, period="validation", metric="sharpe", value=0.8)
    assert count_trials(cfg_smoke) == 2


def test_calibration_rows_are_not_counted_as_trials(cfg_smoke: dict[str, Any]) -> None:
    """R28: the reported experiment count excludes calibration diagnostics."""
    log_trial(cfg_smoke, phase="dev", kind="strategy_grid", run_id="r1", model="rf",
              period="validation", metric="sharpe", value=1.0)
    log_trial(cfg_smoke, phase="dev", kind="calibration", run_id="r1", model="rf",
              period="validation", metric="calibrated_accuracy", value=0.5)
    assert count_trials(cfg_smoke) == 1


def test_log_trial_updates_value_on_rerun(cfg_smoke: dict[str, Any]) -> None:
    for value in (0.5, 0.7):
        log_trial(cfg_smoke, phase="dev", kind="model_year", run_id="wf2023", model="xgb",
                  period="validation", metric="accuracy", value=value)
    df = read_sql_path(cfg_smoke, "trial_log.csv")
    assert len(df) == 1
    assert df.iloc[0]["value"] == 0.7


# ---------------------------------------------------------------- manifest


def test_write_manifest_contains_required_fields(cfg_smoke: dict[str, Any]) -> None:
    init_db(get_engine(cfg_smoke))
    write_manifest(cfg_smoke, "backtest_main", extra={"period": "validation"})

    import json

    path = Path(cfg_smoke["paths"]["results"]) / "manifests" / "backtest_main.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    for key in (
        "run_id", "timestamp", "git_commit", "config_hash", "seed",
        "command_line", "package_versions", "data_range", "models_used", "extra",
    ):
        assert key in data
    assert data["run_id"] == "backtest_main"
    assert data["extra"]["period"] == "validation"
    for pkg in ("python", "pandas", "numpy", "scikit-learn", "xgboost"):
        assert pkg in data["package_versions"]


def test_manifest_config_hash_matches(cfg_smoke: dict[str, Any]) -> None:
    import json

    init_db(get_engine(cfg_smoke))
    write_manifest(cfg_smoke, "r1")
    path = Path(cfg_smoke["paths"]["results"]) / "manifests" / "r1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["config_hash"] == config_hash(cfg_smoke)


# ---------------------------------------------------------------- log


def test_get_logger_writes_to_file(cfg_smoke: dict[str, Any]) -> None:
    log = get_logger("probe", cfg=cfg_smoke)
    log.warning("hello from the test")
    assert (Path(cfg_smoke["paths"]["logs"]) / "probe.log").exists()


def read_sql_path(cfg: dict[str, Any], filename: str) -> pd.DataFrame:
    """Read one CSV written under `paths.results`."""
    return pd.read_csv(Path(cfg["paths"]["results"]) / filename)
