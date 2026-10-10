"""Tests for the transform stage (task T1.4, spec 03 section 4.2, TESTS section 3)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.etl.transform import (
    TransformError,
    attach_instrument_types,
    build_dim_date,
    mark_month_end,
    transform,
)

INDEX = "^NSEI"
VIX = "^INDIAVIX"


def frame(rows: list[tuple[str, str, float, float, float, float, int]]) -> pd.DataFrame:
    """Build a long raw frame from (symbol, date, open, high, low, close, volume)."""
    return pd.DataFrame(
        rows, columns=["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    )


def bar(symbol: str, date: str, close: float, volume: int = 1000) -> tuple:
    return (symbol, date, close, close + 1.0, close - 1.0, close, volume)


def index_bars(dates: list[str]) -> list[tuple]:
    return [bar(INDEX, d, 1000.0, 0) for d in dates]


def prepare(raw: pd.DataFrame, cfg_smoke: dict[str, Any]) -> pd.DataFrame:
    types = {INDEX: "INDEX", VIX: "VOLATILITY"}
    for symbol in raw["symbol"].unique():
        types.setdefault(symbol, "EQUITY")
    return attach_instrument_types(raw, types)


@pytest.fixture
def base_days() -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=10)]


# --------------------------------------------------------------- 1. normalise


def test_timezone_not_shifted(cfg_smoke: dict[str, Any]) -> None:
    """A bar at 2024-03-01 00:00+05:30 keeps date 2024-03-01, not 2024-02-29."""
    dates = [
        pd.Timestamp("2024-02-28 00:00:00+05:30"),
        pd.Timestamp("2024-02-29 00:00:00+05:30"),
        pd.Timestamp("2024-03-01 00:00:00+05:30"),
        pd.Timestamp("2024-03-04 00:00:00+05:30"),
    ]
    raw = pd.DataFrame(
        {
            "symbol": [INDEX] * 4,
            "trade_date": dates,
            "open": [1000.0] * 4,
            "high": [1001.0] * 4,
            "low": [999.0] * 4,
            "close": [1000.0] * 4,
            "volume": [0] * 4,
        }
    )
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)

    assert 20240301 in set(clean["date_key"])
    assert 20240229 in set(clean["date_key"])


def test_columns_are_lower_cased(cfg_smoke: dict[str, Any], base_days: list[str]) -> None:
    """Providers may deliver Open/High/Low/Close; transform normalises the names."""
    raw = frame(index_bars(base_days))
    raw = raw.rename(
        columns={
            "symbol": "Symbol",
            "trade_date": "Trade_Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    raw["Trade_Date"] = pd.to_datetime(raw["Trade_Date"])
    types = {INDEX: "INDEX", VIX: "VOLATILITY"}
    clean, _ = transform(attach_instrument_types(raw, types), cfg_smoke)
    assert set(clean.columns) == {
        "symbol",
        "date_key",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_flagged",
    }
    assert set(clean["symbol"]) == {INDEX}


def test_date_key_is_int(cfg_smoke: dict[str, Any], base_days: list[str]) -> None:
    clean, _ = transform(prepare(frame(index_bars(base_days)), cfg_smoke), cfg_smoke)
    assert pd.api.types.is_integer_dtype(clean["date_key"])


# --------------------------------------------------------------- 2. dedupe


def test_dedupe_keeps_last(cfg_smoke: dict[str, Any]) -> None:
    raw = frame(
        index_bars(["2024-01-01", "2024-01-02"])
        + [
            ("AAA.NS", "2024-01-01", 10.0, 11.0, 9.0, 10.5, 100),
            ("AAA.NS", "2024-01-01", 10.0, 11.0, 9.0, 99.0, 100),  # later duplicate wins
        ]
    )
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)
    aaa = clean[clean["symbol"] == "AAA.NS"]
    assert len(aaa) == 1
    assert aaa["close"].iloc[0] == 99.0


# --------------------------------------------------------------- 3. calendar


def test_trading_calendar_from_index(cfg_smoke: dict[str, Any]) -> None:
    """Rows of a stock on a date with no index bar are dropped."""
    raw = frame(
        index_bars(["2024-01-01", "2024-01-02", "2024-01-03"])
        + [
            bar("AAA.NS", "2024-01-01", 50.0),
            bar("AAA.NS", "2024-01-02", 51.0),
            bar("AAA.NS", "2024-01-04", 52.0),
        ]  # 2024-01-04 not a trading day
    )
    clean, rejected = transform(prepare(raw, cfg_smoke), cfg_smoke)

    assert 20240104 not in set(clean[clean["symbol"] == "AAA.NS"]["date_key"])
    assert "NON_TRADING_DAY" in set(rejected["reason"])


def test_missing_index_raises(cfg_smoke: dict[str, Any]) -> None:
    raw = frame([bar("AAA.NS", "2024-01-01", 50.0)])
    with pytest.raises(TransformError, match="index symbol"):
        transform(prepare(raw, cfg_smoke), cfg_smoke)


# --------------------------------------------------------------- 4. fill gaps


def test_ffill_limit_two_days(cfg_smoke: dict[str, Any]) -> None:
    """A 2-day gap is filled; a 3-day gap leaves the third day missing (TESTS 3)."""
    cfg_smoke["data"]["ffill_max_days"] = 2
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=8)]
    # AAA trades on day 1, then a 3-day gap (days 2,3,4), then resumes on day 5.
    raw = frame(
        index_bars(days)
        + [
            bar("AAA.NS", days[0], 100.0),
            bar("AAA.NS", days[4], 110.0),
        ]
    )
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)
    aaa = clean[clean["symbol"] == "AAA.NS"].set_index("date_key")

    filled_keys = [int(d.replace("-", "")) for d in days[1:3]]
    for key in filled_keys:
        assert key in aaa.index
        assert aaa.loc[key, "close"] == 100.0
        assert aaa.loc[key, "open"] == 100.0
        assert aaa.loc[key, "volume"] == 0
        assert aaa.loc[key, "is_flagged"] == 1

    third = int(days[3].replace("-", ""))
    assert third not in aaa.index, "the third missing day must stay missing"


def test_no_backfill_before_first_bar(cfg_smoke: dict[str, Any]) -> None:
    """Nothing is ever created before a symbol's first bar."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=6)]
    raw = frame(index_bars(days) + [bar("AAA.NS", days[4], 100.0)])
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)

    aaa = clean[clean["symbol"] == "AAA.NS"]
    assert min(aaa["date_key"]) == int(days[4].replace("-", ""))


def test_trailing_gap_is_filled_up_to_the_limit(cfg_smoke: dict[str, Any]) -> None:
    """`03` 4.2 step 4 reindexes to the last calendar date, so a trailing gap fills."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=5)]
    raw = frame(index_bars(days) + [bar("AAA.NS", days[0], 100.0), bar("AAA.NS", days[1], 101.0)])
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)
    aaa = clean[clean["symbol"] == "AAA.NS"].set_index("date_key")

    # The last two calendar days are trailing gaps, both filled and flagged.
    for day in days[2:4]:
        key = int(day.replace("-", ""))
        assert key in aaa.index
        assert aaa.loc[key, "is_flagged"] == 1
    # The fifth day is beyond ffill_max_days and stays missing.
    assert int(days[4].replace("-", "")) not in aaa.index


def test_index_is_not_filled(cfg_smoke: dict[str, Any]) -> None:
    """Only EQUITY and VOLATILITY are reindexed; the index defines the calendar."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=5)]
    # The index only trades the first three days.
    raw = frame(index_bars(days[:3]) + [bar("AAA.NS", days[0], 10.0), bar("AAA.NS", days[4], 12.0)])
    clean, rejected = transform(prepare(raw, cfg_smoke), cfg_smoke)

    # AAA's bar on the fourth day is not a trading day, so it is dropped, not filled.
    assert "NON_TRADING_DAY" in set(rejected["reason"])
    aaa = clean[clean["symbol"] == "AAA.NS"]
    assert set(aaa["date_key"]) <= {int(d.replace("-", "")) for d in days[:3]}
    # The index itself is never reindexed, so it has exactly its own bars.
    assert len(clean[clean["symbol"] == INDEX]) == 3


def test_vix_is_filled(cfg_smoke: dict[str, Any]) -> None:
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=5)]
    raw = frame(index_bars(days) + [bar(VIX, days[0], 13.0, 0), bar(VIX, days[3], 15.0, 0)])
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)
    vix = clean[clean["symbol"] == VIX]
    assert len(vix) >= 2


# --------------------------------------------------------------- 5. hard rejects


def test_hard_rejects(cfg_smoke: dict[str, Any]) -> None:
    """price <= 0, high < low, high below close, or volume < 0 are rejected."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=5)]
    raw = frame(
        index_bars(days)
        + [
            ("AAA.NS", days[1], 0.0, 0.0, 0.0, 0.0, 100),  # non-positive price
            ("BBB.NS", days[1], 10.0, 9.0, 11.0, 10.0, 100),  # high < low
            ("CCC.NS", days[1], 10.0, 10.0, 9.0, 50.0, 100),  # high below close
            ("DDD.NS", days[1], 10.0, 11.0, 9.0, 10.0, -5),  # negative volume
        ]
    )
    clean, rejected = transform(prepare(raw, cfg_smoke), cfg_smoke)

    assert set(rejected["symbol"]) == {"AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"}
    assert rejected["reason"].notna().all()
    # The offending rows are gone from the clean frame. Other rows of the same symbol
    # may survive: gap fill runs before validation (`03` 4.2 order), so a symbol can
    # still have filled days either side of a rejected bar.
    target = int(days[1].replace("-", ""))
    kept = clean[
        (clean["symbol"].isin(["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS"]))
        & (clean["date_key"] == target)
    ]
    assert kept.empty


def test_valid_rows_are_kept(cfg_smoke: dict[str, Any]) -> None:
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=3)]
    raw = frame(index_bars(days) + [bar("AAA.NS", days[1], 50.0)])
    clean, rejected = transform(prepare(raw, cfg_smoke), cfg_smoke)
    assert rejected.empty
    assert "AAA.NS" in set(clean["symbol"])


# --------------------------------------------------------------- 6. soft flags


def test_soft_flags(cfg_smoke: dict[str, Any]) -> None:
    """A 25% move is flagged; zero volume is flagged for EQUITY, never INDEX or VIX."""
    cfg_smoke["data"]["anomaly_abs_return"] = 0.20
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=4)]
    raw = frame(
        index_bars(days)
        + [
            bar("AAA.NS", days[0], 100.0, 1000),
            bar("AAA.NS", days[1], 125.0, 1000),  # +25% anomaly
            bar("BBB.NS", days[0], 50.0, 0),  # zero volume EQUITY
            bar("BBB.NS", days[1], 51.0, 0),
        ]
    )
    clean, rejected = transform(prepare(raw, cfg_smoke), cfg_smoke)

    assert rejected.empty, "soft issues flag, never reject"
    aaa = clean[clean["symbol"] == "AAA.NS"].set_index("date_key")
    bbb = clean[clean["symbol"] == "BBB.NS"].set_index("date_key")

    assert aaa.loc[int(days[1].replace("-", "")), "is_flagged"] == 1
    assert bbb.loc[int(days[1].replace("-", "")), "is_flagged"] == 1
    # The index has volume 0 by definition and must never be flagged for it.
    idx = clean[clean["symbol"] == INDEX]
    assert (idx["is_flagged"] == 0).all()


def test_small_move_is_not_flagged(cfg_smoke: dict[str, Any]) -> None:
    cfg_smoke["data"]["anomaly_abs_return"] = 0.20
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=3)]
    raw = frame(
        index_bars(days)
        + [bar("AAA.NS", days[0], 100.0, 1000), bar("AAA.NS", days[1], 105.0, 1000)]
    )
    clean, _ = transform(prepare(raw, cfg_smoke), cfg_smoke)
    aaa = clean[clean["symbol"] == "AAA.NS"]
    # Only the two real bars are unfilled; a trailing gap row is legitimately flagged.
    real = aaa[aaa["volume"] != 0]
    assert len(real) == 2
    assert (real["is_flagged"] == 0).all()


# --------------------------------------------------------------- 7. dim_date


def test_dim_date_trading_flag(cfg_smoke: dict[str, Any]) -> None:
    """`is_trading_day` equals "index has a bar"; `is_month_end` is the last trading day."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=40)]
    trading = pd.DatetimeIndex(pd.to_datetime(days))
    dim = mark_month_end(build_dim_date(trading, cfg_smoke))

    trading_keys = {int(d.replace("-", "")) for d in days}
    for _, row in dim.iterrows():
        assert row["is_trading_day"] == int(row["date_key"] in trading_keys)

    # Exactly one month end per calendar month that has trading days.
    month_ends = dim[dim["is_month_end"] == 1]
    assert len(month_ends) == len({d[:7] for d in days})
    for _, row in month_ends.iterrows():
        assert row["date_key"] in trading_keys


def test_dim_date_starts_at_configured_start_date(cfg_smoke: dict[str, Any]) -> None:
    """`dim_date` spans every calendar day from `data.start_date` to the last bar."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=10)]
    dim = build_dim_date(pd.DatetimeIndex(pd.to_datetime(days)), cfg_smoke)

    assert dim["full_date"].iloc[0] == str(cfg_smoke["data"]["start_date"])
    assert dim["full_date"].iloc[-1] == days[-1]
    assert (
        len(dim)
        == (pd.Timestamp(days[-1]) - pd.Timestamp(cfg_smoke["data"]["start_date"])).days + 1
    )


def test_dim_date_day_of_week_is_monday_zero(cfg_smoke: dict[str, Any]) -> None:
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=5)]
    dim = build_dim_date(pd.DatetimeIndex(days), cfg_smoke)
    first = dim[dim["full_date"] == "2024-01-01"].iloc[0]
    assert first["day_of_week"] == 0
    assert first["day_name"] == "Monday"


def test_month_end_marks_last_trading_day_not_last_calendar_day(cfg_smoke: dict[str, Any]) -> None:
    """DECISIONS R20: month end is the last trading day present, not the 31st."""
    trading = pd.DatetimeIndex(["2024-01-29", "2024-01-30", "2024-02-01", "2024-02-02"])
    dim = mark_month_end(build_dim_date(trading, cfg_smoke))

    ends = set(dim[dim["is_month_end"] == 1]["full_date"])
    assert ends == {"2024-01-30", "2024-02-02"}


# --------------------------------------------------------------- guards


def test_empty_input_raises(cfg_smoke: dict[str, Any]) -> None:
    with pytest.raises(TransformError, match="empty"):
        transform(pd.DataFrame(), cfg_smoke)


def test_transform_is_deterministic(cfg_smoke: dict[str, Any]) -> None:
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2024-01-01", periods=10)]
    raw = prepare(
        frame(index_bars(days) + [bar("AAA.NS", d, 10.0 + i) for i, d in enumerate(days[:5])]),
        cfg_smoke,
    )
    a, _ = transform(raw, cfg_smoke)
    b, _ = transform(raw, cfg_smoke)
    pd.testing.assert_frame_equal(a, b)
