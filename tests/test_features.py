"""Tests for the indicators (task T2.1, spec 04 section 2, TESTS section 4).

The leak-free tests here are the backbone of the project's honesty: every feature for
day *t* must use only data up to the close of *t*.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    INDICATOR_COLUMNS,
    annualized_vol,
    atr,
    bollinger_pct_b,
    build_indicators,
    compute_indicators,
)


def frame(close: list[float], *, high=None, low=None, volume=None) -> pd.DataFrame:
    """Build a daily frame from a close series.

    Optional `volume` is accepted as a plain array so callers do not have to align a
    Series with the generated date index.
    """
    idx = pd.bdate_range("2023-01-02", periods=len(close), name="date")
    close_s = pd.Series(close, index=idx, dtype="float64")
    volume_s = (
        pd.Series(volume, index=idx, dtype="float64")
        if volume is not None
        else pd.Series(1000.0, index=idx)
    )
    return pd.DataFrame(
        {
            "open": close_s,
            "high": high if high is not None else close_s + 1.0,
            "low": low if low is not None else close_s - 1.0,
            "close": close_s,
            "volume": volume_s,
        },
        index=idx,
    )


# --------------------------------------------------------------- known values


def test_indicator_columns_exact() -> None:
    out = compute_indicators(frame([1.0, 2.0, 3.0]))
    assert list(out.columns) == list(INDICATOR_COLUMNS)


def test_indicator_known_values() -> None:
    """`TESTS` section 4: sma_20, return_5d, return_20d equal manual values."""
    n = 60
    close = [100.0 + i for i in range(n)]
    out = compute_indicators(frame(close))

    # sma_20 at index 19 is the mean of the first 20 closes.
    expected_sma20 = np.mean(close[0:20])
    assert out["sma_20"].iloc[19] == pytest.approx(expected_sma20, rel=1e-12)

    # return_5d at index 5 is close[5] / close[0] - 1.
    assert out["return_5d"].iloc[5] == pytest.approx(close[5] / close[0] - 1, rel=1e-12)

    # return_20d at index 20 is close[20] / close[0] - 1.
    assert out["return_20d"].iloc[20] == pytest.approx(close[20] / close[0] - 1, rel=1e-12)

    # return_1d on the first bar is undefined.
    assert pd.isna(out["return_1d"].iloc[0])


def test_rsi_of_rising_series_is_100() -> None:
    out = compute_indicators(frame([100.0 + i for i in range(40)]))
    assert out["rsi_14"].iloc[-1] == pytest.approx(100.0)


def test_rsi_of_flat_series_is_50() -> None:
    out = compute_indicators(frame([100.0] * 40))
    assert out["rsi_14"].iloc[-1] == pytest.approx(50.0)


def test_rsi_bounded() -> None:
    rng = np.random.default_rng(7)
    out = compute_indicators(frame(list(rng.normal(100, 2, 200).cumsum())))
    valid = out["rsi_14"].dropna()
    assert valid.between(0, 100).all()


def test_macd_of_constant_series_is_zero() -> None:
    out = compute_indicators(frame([100.0] * 60))
    assert out["macd"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["macd_signal"].iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_warmup_values_are_nan() -> None:
    """Warm-up stays NaN and is stored as NULL, never back-filled (03 section 2.2)."""
    out = compute_indicators(frame([100.0 + i * 0.5 for i in range(60)]))
    assert pd.isna(out["sma_20"].iloc[18])
    assert pd.isna(out["sma_50"].iloc[48])
    assert pd.isna(out["return_20d"].iloc[19])


def test_bollinger_pct_b_between_zero_and_one_at_the_mean() -> None:
    close = pd.Series([100.0 + ((i % 7) - 3) for i in range(60)], dtype="float64")
    pct_b = bollinger_pct_b(close)
    valid = pct_b.dropna()
    assert valid.between(0, 1).all()


def test_bollinger_zero_width_is_nan() -> None:
    pct_b = bollinger_pct_b(pd.Series([100.0] * 40, dtype="float64"))
    assert pct_b.dropna().empty or np.isnan(pct_b.iloc[-1])


def test_atr_positive_on_rising_series() -> None:
    f = frame([100.0 + i for i in range(40)])
    values = atr(f["high"], f["low"], f["close"])
    assert values.iloc[-1] > 0


def test_annualized_vol_scales_by_sqrt_252() -> None:
    returns = pd.Series([0.01, -0.01] * 30, dtype="float64")
    vol = annualized_vol(returns)
    expected = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)


def test_volume_ratio_nan_when_average_zero() -> None:
    """Index and VIX have volume 0 throughout, so the ratio is undefined (04 2.2)."""
    f = frame([100.0 + i for i in range(40)], volume=np.zeros(40))
    out = compute_indicators(f)
    assert out["volume_ratio_20"].dropna().empty


def test_volume_ratio_value() -> None:
    closes = [100.0 + i for i in range(40)]
    volumes = [1000.0] * 39 + [2000.0]
    f = frame(closes, volume=np.array(volumes, dtype="float64"))
    out = compute_indicators(f)
    expected = 2000.0 / np.mean(volumes[-20:])
    assert out["volume_ratio_20"].iloc[-1] == pytest.approx(expected)


# --------------------------------------------------------------- leakage


@pytest.mark.parametrize("n", [30, 45, 60, 80])
def test_features_no_lookahead(n: int) -> None:
    """`compute_indicators(prices.iloc[:n])` equals `compute_indicators(prices).iloc[:n]`.

    This is the core no-look-ahead property: appending future bars must never change a
    past indicator.
    """
    rng = np.random.default_rng(11)
    closes = list(rng.normal(100, 3, 200).cumsum())
    full = frame(closes)
    partial = full.iloc[:n]

    full_out = compute_indicators(full).iloc[:n]
    partial_out = compute_indicators(partial)

    pd.testing.assert_frame_equal(full_out, partial_out, rtol=1e-12, check_freq=False)


def test_no_negative_shift_in_source() -> None:
    """Indicators must never look forward."""
    from pathlib import Path

    import src.features.indicators as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "shift(-" not in source


def test_indicator_is_deterministic() -> None:
    f = frame([100.0 + i * 0.7 for i in range(80)])
    pd.testing.assert_frame_equal(compute_indicators(f), compute_indicators(f))


# --------------------------------------------------------------- builder


@pytest.mark.slow
def test_build_indicators_writes_rows(engine: Engine, loaded: tuple) -> None:
    """The builder writes one row per (date_key, stock_key) for EQUITY and INDEX."""
    written = build_indicators(engine, engine.cfg)
    assert written > 0

    from src.common.db import read_sql

    total = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_indicator")["n"].iloc[0]
    assert total == written


@pytest.mark.slow
def test_builder_is_idempotent(engine: Engine, loaded: tuple) -> None:
    from src.common.db import read_sql

    build_indicators(engine, engine.cfg)
    first = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_indicator")["n"].iloc[0]
    build_indicators(engine, engine.cfg)
    second = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_indicator")["n"].iloc[0]
    assert first == second


@pytest.mark.slow
def test_vix_has_no_indicators(engine: Engine, loaded: tuple) -> None:
    """DECISIONS R9: indicators for EQUITY and INDEX only."""
    from src.common.db import read_sql

    build_indicators(engine, engine.cfg)
    rows = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM fact_indicator f JOIN dim_stock s "
        "ON s.stock_key = f.stock_key WHERE s.instrument_type = 'VOLATILITY'",
    )
    assert rows["n"].iloc[0] == 0


@pytest.mark.slow
def test_warmup_stored_as_null(engine: Engine, loaded: tuple) -> None:
    """Warm-up NaN becomes NULL, not a number."""
    from src.common.db import read_sql

    build_indicators(engine, engine.cfg)
    rows = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM fact_indicator WHERE sma_50 IS NULL",
    )
    assert rows["n"].iloc[0] > 0


# --------------------------------------------------------------- fixtures
from sqlalchemy import Engine  # noqa: E402


@pytest.fixture
def loaded(engine: Engine, cfg_smoke: dict[str, Any]) -> tuple:
    """A warehouse with a small synthetic market already loaded."""
    from src.etl.load import load
    from src.etl.synthetic import generate_market
    from src.etl.transform import attach_instrument_types, transform

    cfg_smoke["synthetic"]["n_stocks"] = 3
    cfg_smoke["data"]["start_date"] = "2023-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2023-06-30"
    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    clean, rejected = transform(attach_instrument_types(built.prices, types), cfg_smoke)
    stats = {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)}
    load(engine, clean, built.universe, "FULL", stats, cfg_smoke)
    engine.cfg = cfg_smoke  # type: ignore[attr-defined]
    return engine, cfg_smoke


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    from src.common.db import get_engine, init_db

    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng
