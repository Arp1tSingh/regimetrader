"""Tests for extraction (task T1.3, spec 03 section 4.1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.etl.extract import ExtractError, _normalise, _retry_waits, extract


@pytest.fixture
def no_sleep() -> Any:
    """Retry sleeps are instant so the retry tests stay fast."""
    return lambda _seconds: None


@pytest.fixture
def good_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


# --------------------------------------------------------------- retry policy


def test_extract_retry(cfg_smoke: dict[str, Any], no_sleep: Any) -> None:
    """A yfinance that fails twice then succeeds is retried (TESTS section 3)."""
    cfg_smoke["data"]["retries"] = 3
    cfg_smoke["data"]["retry_wait_seconds"] = [0, 0, 0]
    calls = {"n": 0}

    frame = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="Date"),
    )

    def flaky(symbol: str, start: str) -> pd.DataFrame:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ConnectionError("boom")
        return frame

    result = extract(
        cfg_smoke, ["AAA.NS"], "2024-01-01", "yfinance", sleep=no_sleep, yf_fetcher=flaky
    )

    assert calls["n"] == 3
    assert not result.empty
    assert result["symbol"].iloc[0] == "AAA.NS"


def test_symbol_that_always_fails_is_skipped(
    cfg_smoke: dict[str, Any], no_sleep: Any, good_frame: pd.DataFrame
) -> None:
    """One dead symbol is logged and skipped while the others still load (TESTS section 3)."""
    cfg_smoke["data"]["retries"] = 3
    cfg_smoke["data"]["retry_wait_seconds"] = [0, 0, 0]

    def fetch(symbol: str, start: str) -> pd.DataFrame:
        if symbol == "DEAD.NS":
            raise ConnectionError("no such ticker")
        return good_frame

    result = extract(
        cfg_smoke,
        ["AAA.NS", "DEAD.NS", "BBB.NS"],
        "2024-01-01",
        "yfinance",
        sleep=no_sleep,
        yf_fetcher=fetch,
    )

    assert set(result["symbol"]) == {"AAA.NS", "BBB.NS"}
    assert "DEAD.NS" not in set(result["symbol"])


def test_retry_count_respects_config(cfg_smoke: dict[str, Any], no_sleep: Any) -> None:
    cfg_smoke["data"]["retries"] = 2
    cfg_smoke["data"]["retry_wait_seconds"] = [0, 0]
    calls = {"n": 0}

    def always_fail(symbol: str, start: str) -> pd.DataFrame:
        calls["n"] += 1
        raise ConnectionError("boom")

    extract(cfg_smoke, ["AAA.NS"], "2024-01-01", "yfinance", sleep=no_sleep, yf_fetcher=always_fail)
    assert calls["n"] == 2


def test_retry_waits_from_config(cfg_smoke: dict[str, Any]) -> None:
    assert _retry_waits(cfg_smoke) == [2.0, 5.0, 10.0]


# --------------------------------------------------------------- normalisation


def test_output_columns(cfg_smoke: dict[str, Any], no_sleep: Any, good_frame: pd.DataFrame) -> None:
    result = extract(
        cfg_smoke,
        ["AAA.NS"],
        "2024-01-01",
        "yfinance",
        sleep=no_sleep,
        yf_fetcher=lambda s, st: good_frame,
    )
    assert list(result.columns) == [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_timezone_not_shifted_by_normalise() -> None:
    """`2024-03-01 00:00+05:30` stays 2024-03-01, never 2024-02-29 (TESTS section 3)."""
    idx = pd.DatetimeIndex(["2024-03-01 00:00:00+05:30"])
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]}, index=idx
    )
    frame.index.name = "Date"

    out = _normalise(frame, "AAA.NS")
    assert out["trade_date"].iloc[0] == pd.Timestamp("2024-03-01")
    assert out["trade_date"].dt.tz is None


def test_empty_frame_returns_empty_long_frame(cfg_smoke: dict[str, Any], no_sleep: Any) -> None:
    empty = pd.DataFrame()
    result = extract(
        cfg_smoke,
        ["AAA.NS"],
        "2024-01-01",
        "yfinance",
        sleep=no_sleep,
        yf_fetcher=lambda s, st: empty,
    )
    assert result.empty
    assert list(result.columns) == [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


# --------------------------------------------------------------- sources


def test_synthetic_source_returns_expected_columns(
    cfg_smoke: dict[str, Any], no_sleep: Any
) -> None:
    result = extract(
        cfg_smoke, ["SYN01.NS"], cfg_smoke["data"]["start_date"], "synthetic", sleep=no_sleep
    )
    assert list(result.columns) == [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert not result.empty


def test_csv_source_reads_file(cfg_smoke: dict[str, Any], no_sleep: Any, tmp_path: Path) -> None:
    csv_dir = Path(cfg_smoke["data"]["csv_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10, 20, 30],
        }
    ).to_csv(csv_dir / "AAA.NS.csv", index=False)

    result = extract(cfg_smoke, ["AAA.NS"], "2024-01-01", "csv", sleep=no_sleep)
    assert len(result) == 3
    assert result["close"].tolist() == [1.5, 2.5, 3.5]


def test_csv_missing_file_is_skipped(cfg_smoke: dict[str, Any], no_sleep: Any) -> None:
    result = extract(cfg_smoke, ["GHOST.NS"], "2024-01-01", "csv", sleep=no_sleep)
    assert result.empty


def test_vix_csv_fallback(cfg_smoke: dict[str, Any], no_sleep: Any, tmp_path: Path) -> None:
    """`03` section 4.1 step 7: fall back to data.vix_csv when ^INDIAVIX is empty."""
    cfg_smoke["data"]["retries"] = 1
    vix_path = Path(cfg_smoke["data"]["vix_csv"])
    vix_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=3, freq="B"), "close": [13.0, 14.0, 15.0]}
    ).to_csv(vix_path, index=False)

    def fail_vix(symbol: str, start: str) -> pd.DataFrame:
        if symbol == cfg_smoke["data"]["vix_symbol"]:
            return pd.DataFrame()
        return pd.DataFrame(
            {"Close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="Date")
        )

    result = extract(
        cfg_smoke,
        [cfg_smoke["data"]["vix_symbol"]],
        "2024-01-01",
        "yfinance",
        sleep=no_sleep,
        yf_fetcher=fail_vix,
    )
    assert not result.empty
    assert set(result["symbol"]) == {cfg_smoke["data"]["vix_symbol"]}


# --------------------------------------------------------------- raw cache & staging


def test_raw_csv_cache_written(
    cfg_smoke: dict[str, Any], no_sleep: Any, good_frame: pd.DataFrame
) -> None:
    extract(
        cfg_smoke,
        ["AAA.NS"],
        "2024-01-01",
        "yfinance",
        run_date="2026-10-10",
        sleep=no_sleep,
        yf_fetcher=lambda s, st: good_frame,
    )
    assert (Path(cfg_smoke["paths"]["raw"]) / "AAA.NS_2026-10-10.csv").exists()


def test_staging_table_filled(
    cfg_smoke: dict[str, Any], no_sleep: Any, good_frame: pd.DataFrame
) -> None:
    from src.common.db import get_engine, init_db, read_sql

    engine = get_engine(cfg_smoke)
    init_db(engine)
    extract(
        cfg_smoke,
        ["AAA.NS"],
        "2024-01-01",
        "yfinance",
        engine=engine,
        sleep=no_sleep,
        yf_fetcher=lambda s, st: good_frame,
    )

    staged = read_sql(
        engine, "SELECT symbol, trade_date, close_price FROM stg_price_raw ORDER BY trade_date"
    )
    assert len(staged) == 5
    assert set(staged["symbol"]) == {"AAA.NS"}
    assert staged["trade_date"].iloc[0] == "2024-01-01"


def test_staging_is_truncated_each_run(
    cfg_smoke: dict[str, Any], no_sleep: Any, good_frame: pd.DataFrame
) -> None:
    from src.common.db import get_engine, init_db, read_sql

    engine = get_engine(cfg_smoke)
    init_db(engine)
    for _ in range(2):
        extract(
            cfg_smoke,
            ["AAA.NS"],
            "2024-01-01",
            "yfinance",
            engine=engine,
            sleep=no_sleep,
            yf_fetcher=lambda s, st: good_frame,
        )

    staged = read_sql(engine, "SELECT COUNT(*) AS n FROM stg_price_raw")
    assert staged["n"].iloc[0] == 5


# --------------------------------------------------------------- guards


def test_unknown_source_raises(cfg_smoke: dict[str, Any]) -> None:
    with pytest.raises(ExtractError, match="unknown source"):
        extract(cfg_smoke, ["AAA.NS"], "2024-01-01", "ftp")


def test_empty_symbol_list_raises(cfg_smoke: dict[str, Any]) -> None:
    with pytest.raises(ExtractError, match="no symbols"):
        extract(cfg_smoke, [], "2024-01-01", "synthetic")
