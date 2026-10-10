"""Tests for the ETL pipeline CLI (task T1.7, spec 03 section 4, 07 section 5).

Marked slow: each test runs the whole pipeline on the synthetic generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.common.db import get_engine, read_sql
from src.etl.load import price_checksum
from src.etl.pipeline import (
    EXIT_DATA_FAILURE,
    EXIT_OK,
    EXIT_QUALITY,
    main,
)

from .conftest import SMOKE_CONFIG


def smoke_argv() -> list[str]:
    return ["--full-refresh", "--config", str(SMOKE_CONFIG)]


@pytest.fixture
def smoke_cfg(cfg_smoke: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """A trimmed smoke config: 4 stocks over ~1 year keeps the test fast.

    The synthetic source generates its own symbols, but the pipeline still reads the
    universe file, so point it at the real one rather than a tmp_path copy.
    """
    cfg_smoke["synthetic"]["n_stocks"] = 4
    cfg_smoke["data"]["start_date"] = "2022-01-03"
    cfg_smoke["synthetic"]["end_date"] = "2022-12-30"
    cfg_smoke["data"]["universe_file"] = str(repo_root / "config" / "nifty50_symbols.csv")
    return cfg_smoke


@pytest.fixture
def smoke_config_file(smoke_cfg: dict[str, Any], tmp_path: Path) -> str:
    """Write the trimmed config to disk and return its path for the CLI."""
    import yaml

    path = tmp_path / "smoke_fast.yaml"
    base = yaml.safe_load(Path(SMOKE_CONFIG).read_text(encoding="utf-8"))
    base["synthetic"]["n_stocks"] = 4
    base["data"]["start_date"] = "2022-01-03"
    base["synthetic"]["end_date"] = "2022-12-30"
    for key, value in smoke_cfg["paths"].items():
        base["paths"][key] = value
    for key in ("universe_file", "vix_csv", "csv_dir"):
        base["data"][key] = smoke_cfg["data"][key]
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


@pytest.mark.slow
def test_pipeline_full_refresh_succeeds(smoke_config_file: str) -> None:
    assert main(["--full-refresh", "--config", smoke_config_file]) == EXIT_OK


@pytest.mark.slow
def test_pipeline_loads_all_synthetic_symbols(smoke_config_file: str) -> None:
    """Regression: the synthetic universe must be used, not the real Nifty 50 file."""
    from src.common.config import load_config

    main(["--full-refresh", "--config", smoke_config_file])
    cfg = load_config(smoke_config_file)
    df = read_sql(
        get_engine(cfg),
        "SELECT COUNT(DISTINCT stock_key) AS n FROM fact_price",
    )
    # 4 synthetic stocks + index + VIX.
    assert df["n"].iloc[0] == 6

    symbols = read_sql(
        get_engine(cfg),
        "SELECT s.symbol FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "GROUP BY s.symbol ORDER BY s.symbol",
    )
    assert set(symbols["symbol"]) == {
        "^INDIAVIX",
        "^NSEI",
        "SYN01.NS",
        "SYN02.NS",
        "SYN03.NS",
        "SYN04.NS",
    }


@pytest.mark.slow
def test_pipeline_rerun_is_a_no_op(smoke_config_file: str) -> None:
    """A second identical run must not change the warehouse (AGENTS.md rule 7)."""
    from src.common.config import load_config

    main(["--full-refresh", "--config", smoke_config_file])
    cfg = load_config(smoke_config_file)
    engine = get_engine(cfg)
    rows_before = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")["n"].iloc[0]
    checksum_before = price_checksum(engine)

    main(["--full-refresh", "--config", smoke_config_file])
    assert read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")["n"].iloc[0] == rows_before
    assert price_checksum(engine) == checksum_before


@pytest.mark.slow
def test_incremental_after_full(smoke_config_file: str) -> None:
    """The second run is incremental and still leaves the data identical."""
    from src.common.config import load_config

    main(["--full-refresh", "--config", smoke_config_file])
    cfg = load_config(smoke_config_file)
    engine = get_engine(cfg)
    checksum_before = price_checksum(engine)

    assert main(["--config", smoke_config_file]) == EXIT_OK
    assert price_checksum(engine) == checksum_before

    log = read_sql(engine, "SELECT mode FROM etl_run_log ORDER BY run_id")
    assert list(log["mode"]) == ["FULL", "INCREMENTAL"]


@pytest.mark.slow
def test_run_log_records_success(smoke_config_file: str) -> None:
    from src.common.config import load_config

    main(["--full-refresh", "--config", smoke_config_file])
    log = read_sql(get_engine(load_config(smoke_config_file)), "SELECT status FROM etl_run_log")
    assert set(log["status"]) == {"SUCCESS"}


@pytest.mark.slow
def test_limit_symbols_restricts_equities(smoke_config_file: str) -> None:
    from src.common.config import load_config

    main(["--full-refresh", "--limit-symbols", "2", "--config", smoke_config_file])
    df = read_sql(
        get_engine(load_config(smoke_config_file)),
        "SELECT COUNT(DISTINCT stock_key) AS n FROM fact_price",
    )
    assert df["n"].iloc[0] == 4  # 2 equities + index + VIX


@pytest.mark.slow
def test_incremental_preserves_trading_calendar(smoke_config_file: str) -> None:
    """Regression: an incremental load must not clear `is_trading_day` on older dates.

    The calendar comes from the index's bars in the warehouse, not from the current
    batch. Deriving it from the batch would flip every earlier trading day to 0 and
    then fail DQ-03 on 29,000 rows.
    """
    from src.common.config import load_config

    main(["--full-refresh", "--config", smoke_config_file])
    cfg = load_config(smoke_config_file)
    engine = get_engine(cfg)

    before = read_sql(engine, "SELECT COUNT(*) AS n FROM dim_date WHERE is_trading_day = 1")[
        "n"
    ].iloc[0]
    assert before > 0

    main(["--config", smoke_config_file])
    after = read_sql(engine, "SELECT COUNT(*) AS n FROM dim_date WHERE is_trading_day = 1")[
        "n"
    ].iloc[0]
    assert after == before

    bad = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM fact_price f LEFT JOIN dim_date d "
        "ON f.date_key = d.date_key WHERE d.date_key IS NULL OR d.is_trading_day <> 1",
    )["n"].iloc[0]
    assert bad == 0


@pytest.mark.slow
def test_incremental_then_quality_passes(smoke_config_file: str) -> None:
    """The incremental run must return 0, meaning DQ-01..07 all passed."""

    main(["--full-refresh", "--config", smoke_config_file])
    assert main(["--config", smoke_config_file]) == EXIT_OK


# --------------------------------------------------------------- exit codes


def test_bad_config_returns_error(tmp_path: Path) -> None:
    assert main(["--config", str(tmp_path / "nope.yaml")]) != EXIT_OK


def test_missing_universe_returns_data_failure(tmp_path: Path) -> None:
    """A missing universe file must not be papered over (AGENTS.md rule 11)."""
    import yaml

    base = yaml.safe_load(Path(SMOKE_CONFIG).read_text(encoding="utf-8"))
    base["data"]["source"] = "synthetic"
    base["data"]["universe_file"] = str(tmp_path / "absent.csv")
    path = tmp_path / "missing_universe.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")

    assert main(["--full-refresh", "--config", str(path)]) == EXIT_DATA_FAILURE


def test_unknown_source_rejected_by_argparse() -> None:
    with pytest.raises(SystemExit):
        main(["--source", "ftp", "--config", str(SMOKE_CONFIG)])


def test_exit_codes_are_the_documented_values() -> None:
    """`07` section 5: 0 success, 1 unexpected, 2 data failure, 3 guard, 4 quality."""
    from src.etl import pipeline

    assert (pipeline.EXIT_OK, pipeline.EXIT_ERROR) == (0, 1)
    assert (pipeline.EXIT_DATA_FAILURE, pipeline.EXIT_GUARD, pipeline.EXIT_QUALITY) == (2, 3, 4)


def test_help_works() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.slow
def test_quality_failure_returns_four(
    smoke_config_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard data-quality failure must exit 4, not 0."""
    from src.etl import pipeline
    from src.etl.quality import DataQualityError

    def boom(engine: Any, cfg: Any, stats: Any = None) -> Any:
        raise DataQualityError("forced failure")

    monkeypatch.setattr(pipeline, "run_checks", boom)
    assert pipeline.main(["--full-refresh", "--config", smoke_config_file]) == EXIT_QUALITY
