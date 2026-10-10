"""Shared pytest fixtures.

Unit tests use the synthetic generator and temporary SQLite files only:
never the real warehouse, never the network (AGENTS.md section 7).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_CONFIG = REPO_ROOT / "config" / "config.yaml"
SMOKE_CONFIG = REPO_ROOT / "config" / "config.smoke.yaml"

#: Keys under `paths:` that name a directory rather than a file.
DIR_PATH_KEYS = ("raw", "external", "processed", "models", "results", "figures", "daily", "logs")
#: Keys under `paths:` that name a file.
FILE_PATH_KEYS = ("db", "frozen_config")


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]):
    """An initialised warehouse in tmp_path, shared by the integration fixtures."""
    from src.common.db import get_engine, init_db

    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


@pytest.fixture
def loaded(engine, cfg_smoke: dict[str, Any], repo_root: Path) -> tuple:
    """Synthetic warehouse with prices and indicators loaded."""
    from src.etl.load import load
    from src.etl.synthetic import generate_market
    from src.etl.transform import attach_instrument_types, transform
    from src.features.indicators import build_indicators

    cfg_smoke["synthetic"]["n_stocks"] = 8
    cfg_smoke["data"]["start_date"] = "2017-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2022-12-30"
    cfg_smoke["data"]["universe_file"] = str(repo_root / "config" / "nifty50_symbols.csv")

    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    clean, rejected = transform(attach_instrument_types(built.prices, types), cfg_smoke)
    stats = {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)}
    load(engine, clean, built.universe, "FULL", stats, cfg_smoke)
    build_indicators(engine, cfg_smoke)
    return engine, cfg_smoke


@pytest.fixture
def panel_ready(loaded: tuple, tmp_path: Path) -> tuple:
    """`loaded` plus fitted regimes, so the regime one-hot features exist.

    The panel drops rows with NaN in an active feature, so without regimes every row
    would be discarded.
    """
    import yaml

    from src.common.config import load_config
    from src.regime.regime_model import main as regime_main

    engine, cfg = loaded
    base = yaml.safe_load((repo_root_cfg()).read_text(encoding="utf-8"))
    base["paths"] = dict(cfg["paths"])
    base["data"]["start_date"] = cfg["data"]["start_date"]
    base["synthetic"]["end_date"] = cfg["synthetic"]["end_date"]
    base["synthetic"]["n_stocks"] = cfg["synthetic"]["n_stocks"]
    config_path = tmp_path / "models_smoke.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")

    assert regime_main(["--fit", "--config", str(config_path)]) == 0
    return load_config(str(config_path)), engine


@pytest.fixture
def signals_ready(panel_ready: tuple) -> tuple:
    """`panel_ready` plus a `tree` model row and `fact_signal` probabilities.

    The backtest needs stored probabilities; without them every weight provider
    returns None and every strategy looks identical.
    """
    import random
    from datetime import date

    from sqlalchemy import text

    from src.common.db import execute_many

    cfg, engine = panel_ready
    rng = random.Random(7)

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT OR IGNORE INTO dim_model (model_key, model_name, version, "
                "trained_through) VALUES (1, 'tree', 'wf2016', '2015-12-31')"
            )
        )
        stocks = conn.execute(
            text("SELECT stock_key FROM dim_stock WHERE instrument_type = 'EQUITY'")
        ).fetchall()
        regimes = conn.execute(
            text("SELECT date_key, regime_key FROM fact_market_regime")
        ).fetchall()
        rows = [
            {
                "d": int(date_key),
                "s": int(stock_key),
                "r": int(regime_key),
                "p": rng.uniform(0.45, 0.80),
            }
            for date_key, regime_key in regimes
            for (stock_key,) in stocks
        ]
        execute_many(
            conn,
            "INSERT OR REPLACE INTO fact_signal (date_key, stock_key, model_key, "
            "regime_key, prob_up, signal, target_weight) "
            "VALUES (:d, :s, 1, :r, :p, 0, 0.0)",
            rows,
        )
    del date
    return cfg, engine


def repo_root_cfg() -> Path:
    """Path of the shipped smoke config."""
    return REPO_ROOT / "config" / "config.smoke.yaml"


@pytest.fixture
def cfg_smoke(tmp_path: Path) -> dict[str, Any]:
    """The smoke configuration, redirected so every path lives under `tmp_path`.

    Used by any test that must not touch the real warehouse, models or reports.
    """
    from src.common.config import load_config

    cfg = load_config(str(SMOKE_CONFIG))
    for key in DIR_PATH_KEYS:
        cfg["paths"][key] = str(tmp_path / key)
    for key in FILE_PATH_KEYS:
        cfg["paths"][key] = str(tmp_path / Path(cfg["paths"][key]).name)
    cfg["data"]["universe_file"] = str(tmp_path / "nifty50_symbols.csv")
    cfg["data"]["vix_csv"] = str(tmp_path / "external" / "india_vix.csv")
    cfg["data"]["csv_dir"] = str(tmp_path / "external" / "prices")
    return cfg


@pytest.fixture
def cfg_base() -> dict[str, Any]:
    """The unmodified base configuration, read straight from disk."""
    from src.common.config import load_config

    return load_config(str(BASE_CONFIG))
