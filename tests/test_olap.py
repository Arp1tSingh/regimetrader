"""Tests for the OLAP runner (task T2.2, spec 03 section 6, TESTS section 4).

`test_olap_queries_run` requires every query in `sql/olap/` to execute without SQL
errors, and the roll-up (02) to equal a hand aggregation of query 01's inputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from sqlalchemy import Engine

from src.features.indicators import build_indicators
from src.olap.run import (
    OLAP_DIR,
    build_pivot_sql,
    main,
    parse_overrides,
    parse_params,
    run_query,
)

QUERY_STEMS = sorted(p.stem for p in OLAP_DIR.glob("*.sql"))

#: The smoke config file, used by the CLI tests.
SMOKE_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "config.smoke.yaml")


@pytest.fixture
def loaded(engine: Engine, cfg_smoke: dict[str, Any]) -> tuple:
    """A warehouse with synthetic prices and indicators loaded."""
    from src.etl.load import load
    from src.etl.synthetic import generate_market
    from src.etl.transform import attach_instrument_types, transform

    cfg_smoke["synthetic"]["n_stocks"] = 8
    cfg_smoke["data"]["start_date"] = "2022-01-03"
    cfg_smoke["synthetic"]["end_date"] = "2023-12-29"
    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    clean, rejected = transform(attach_instrument_types(built.prices, types), cfg_smoke)
    stats = {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)}
    load(engine, clean, built.universe, "FULL", stats, cfg_smoke)
    build_indicators(engine, cfg_smoke)
    return engine, cfg_smoke


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    from src.common.db import get_engine, init_db

    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


# --------------------------------------------------------------- parsing


def test_all_nine_queries_present() -> None:
    assert len(QUERY_STEMS) == 9
    assert QUERY_STEMS[0].startswith("01_")
    assert QUERY_STEMS[-1].startswith("09_")


def test_parse_params_reads_defaults() -> None:
    sql = (OLAP_DIR / "03_drilldown_sector_to_stock.sql").read_text(encoding="utf-8")
    params = parse_params(sql)
    assert params["sector"] == "IT"
    assert params["year"] == 2025


def test_parse_params_casts_numbers() -> None:
    sql = "-- param: a = 2023\n-- param: b = 1.5\n-- param: c = hello\n"
    params = parse_params(sql)
    assert params["a"] == 2023
    assert params["b"] == 1.5
    assert params["c"] == "hello"


def test_parse_overrides_typed() -> None:
    out = parse_overrides(["sector=Banking", "year=2022", "ratio=0.5"])
    assert out == {"sector": "Banking", "year": 2022, "ratio": 0.5}


def test_parse_overrides_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="name=value"):
        parse_overrides(["sector"])


def test_build_pivot_sql_uses_configured_years() -> None:
    sql = build_pivot_sql([2021, 2022, 2023])
    assert "y2021" in sql and "y2022" in sql and "y2023" in sql
    assert "d.year = 2021" in sql


def test_build_pivot_sql_changes_with_years() -> None:
    assert "y2030" in build_pivot_sql([2030])
    assert "y2022" not in build_pivot_sql([2030])


# --------------------------------------------------------------- execution


def test_olap_queries_run(loaded: tuple, tmp_path: Path) -> None:
    """Every query executes without a SQL error; empty is allowed."""
    engine, cfg = loaded
    out = tmp_path / "olap"
    for stem in QUERY_STEMS:
        path = next(p for p in OLAP_DIR.glob("*.sql") if p.stem == stem)
        df = run_query(engine, path, {}, cfg, out)
        assert isinstance(df, pd.DataFrame)
        assert (out / f"{stem}.csv").exists()


def test_rollup_02_matches_hand_aggregation_of_01(loaded: tuple, tmp_path: Path) -> None:
    """Query 02 (sector by year) equals a hand aggregation of query 01's inputs."""
    engine, cfg = loaded
    out = tmp_path / "olap"

    q01 = run_query(engine, OLAP_DIR / "01_rollup_sector_quarter.sql", {}, cfg, out)
    q02 = run_query(engine, OLAP_DIR / "02_rollup_sector_year.sql", {}, cfg, out)
    assert not q01.empty and not q02.empty

    # Aggregate 01's quarterly means is not valid (means of means are not means), so
    # recompute 02 straight from the source tables instead.
    from src.common.db import read_sql

    hand = read_sql(
        engine,
        "SELECT d.year AS year, s.sector AS sector, "
        "       ROUND(AVG(i.return_1d) * 100, 3) AS avg_daily_return_pct "
        "FROM fact_indicator i "
        "JOIN dim_date  d ON d.date_key = i.date_key "
        "JOIN dim_stock s ON s.stock_key = i.stock_key "
        "WHERE s.instrument_type = 'EQUITY' "
        "GROUP BY d.year, s.sector ORDER BY d.year, s.sector",
    )
    pd.testing.assert_frame_equal(
        q02.reset_index(drop=True), hand.reset_index(drop=True), check_dtype=False
    )


def test_query_01_result_is_hand_checkable(loaded: tuple, tmp_path: Path) -> None:
    """Spot-check one (year, quarter, sector) cell of query 01."""
    engine, cfg = loaded
    from src.common.db import read_sql

    q01 = run_query(engine, OLAP_DIR / "01_rollup_sector_quarter.sql", {}, cfg, tmp_path / "olap")
    row = q01.iloc[0]

    hand = read_sql(
        engine,
        "SELECT COUNT(*) AS n, ROUND(AVG(i.return_1d) * 100, 3) AS pct "
        "FROM fact_indicator i "
        "JOIN dim_date  d ON d.date_key = i.date_key "
        "JOIN dim_stock s ON s.stock_key = i.stock_key "
        "WHERE s.instrument_type = 'EQUITY' AND d.year = :y AND d.quarter = :q "
        "  AND s.sector = :s",
        {"y": int(row["year"]), "q": int(row["quarter"]), "s": row["sector"]},
    )
    assert int(hand["n"].iloc[0]) == int(row["observations"])
    assert hand["pct"].iloc[0] == pytest.approx(float(row["avg_daily_return_pct"]))


def test_parameter_override_is_honoured(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    df = run_query(
        engine,
        OLAP_DIR / "03_drilldown_sector_to_stock.sql",
        {"sector": "Banking", "year": 2023},
        cfg,
        tmp_path / "olap",
    )
    assert not df.empty


def test_default_year_falls_back_to_latest(loaded: tuple, tmp_path: Path) -> None:
    """A data-dependent default with no rows falls back to the latest year."""
    engine, cfg = loaded
    df = run_query(
        engine, OLAP_DIR / "03_drilldown_sector_to_stock.sql", {}, cfg, tmp_path / "olap"
    )
    assert not df.empty, "the default year should have fallen back to a year with data"


def test_query_07_returns_at_most_five(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    df = run_query(engine, OLAP_DIR / "07_window_top5_momentum.sql", {}, cfg, tmp_path / "olap")
    assert len(df) <= 5


def test_query_06_uses_configured_pivot_years(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    cfg["olap"]["pivot_years"] = [2022, 2023]
    df = run_query(engine, OLAP_DIR / "06_pivot_sector_by_year.sql", {}, cfg, tmp_path / "olap")
    assert "y2022" in df.columns and "y2023" in df.columns
    assert "y2024" not in df.columns


def test_empty_result_is_not_an_error(loaded: tuple, tmp_path: Path) -> None:
    """Query 08 is empty before regimes exist and that must not raise."""
    engine, cfg = loaded
    df = run_query(engine, OLAP_DIR / "08_regime_profile.sql", {}, cfg, tmp_path / "olap")
    assert df.empty


def test_csv_written_for_every_query(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    out = tmp_path / "olap"
    for stem in QUERY_STEMS:
        path = next(p for p in OLAP_DIR.glob("*.sql") if p.stem == stem)
        run_query(engine, path, {}, cfg, out)
    assert len(list(out.glob("*.csv"))) == 9


# --------------------------------------------------------------- CLI


def _tmp_config_file(cfg: dict, tmp_path: Path) -> str:
    """Write the tmp-redirected config to disk so the CLI can be given a path.

    The CLI takes a config *path*, not a config object, so the fixture's already
    redirected config has to be serialised before it can be handed to `main`.
    """
    path = Path(tmp_path) / "olap_cli_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(path)


def test_the_cli_tests_do_not_depend_on_the_untracked_smoke_warehouse() -> None:
    """Regression for a clean-clone failure found by the T10.2 check.

    These CLI tests used to pass `config/config.smoke.yaml`, whose `db` points at
    `data/warehouse/smoke.db`. That file is gitignored and only exists after
    `make smoke` has run, so on a fresh clone all three CLI tests failed with
    `9 failure(s)` / exit 1: the tables did not exist.

    They passed in the working directory purely because repeated smoke runs left
    a 16 MB database behind. A test that passes only because of untracked local
    state is not a test, so this asserts the source no longer points the CLI at
    the real smoke config.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    body = source[
        source.index("# --------------------------------------------------------------- CLI") :
    ]
    # Match the call pattern, not the bare name, so this test's own prose about
    # SMOKE_CONFIG_PATH cannot trip it.
    offenders = re.findall(r'"--config",\s*SMOKE_CONFIG_PATH', body)
    assert not offenders, (
        "an OLAP CLI call is still pointed at the gitignored smoke warehouse "
        f"instead of the tmp one: {offenders}"
    )


def test_cli_all(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    from src.olap.run import EXIT_OK

    assert main(["--all", "--config", _tmp_config_file(cfg, tmp_path)]) == EXIT_OK


def test_cli_single_query(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    from src.olap.run import EXIT_OK

    code = main(["--query", "01_", "--config", _tmp_config_file(cfg, tmp_path)])
    assert code == EXIT_OK


def test_cli_unknown_query_fails(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    from src.olap.run import EXIT_DATA_FAILURE

    code = main(["--query", "99_", "--config", _tmp_config_file(cfg, tmp_path)])
    assert code == EXIT_DATA_FAILURE


def test_cli_param_override(loaded: tuple, tmp_path: Path) -> None:
    engine, cfg = loaded
    from src.olap.run import EXIT_OK

    code = main(
        [
            "--query",
            "03_",
            "--param",
            "sector=IT",
            "--param",
            "year=2023",
            "--config",
            _tmp_config_file(cfg, tmp_path),
        ]
    )
    assert code == EXIT_OK
