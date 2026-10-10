"""Tests for the synthetic market generator (`TESTS` section 2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.etl.synthetic import (
    LONG_COLUMNS,
    REGIME_NAMES,
    UNIVERSE_COLUMNS,
    generate_market,
    make_synthetic_market,
)


@pytest.fixture
def market(cfg_smoke: dict[str, Any]) -> Any:
    return generate_market(cfg_smoke)


# ------------------------------------------------------------- determinism


def test_same_seed_gives_identical_frames(cfg_smoke: dict[str, Any]) -> None:
    p1, u1 = make_synthetic_market(cfg_smoke)
    p2, u2 = make_synthetic_market(cfg_smoke)

    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(u1, u2)


def test_different_seed_differs(cfg_smoke: dict[str, Any]) -> None:
    p1, _ = make_synthetic_market(cfg_smoke)
    cfg_smoke["synthetic"]["seed"] = int(cfg_smoke["synthetic"]["seed"]) + 1
    p2, _ = make_synthetic_market(cfg_smoke)

    assert not p1["close"].equals(p2["close"])


def test_regime_chain_is_deterministic(cfg_smoke: dict[str, Any]) -> None:
    pd.testing.assert_series_equal(
        generate_market(cfg_smoke).regimes, generate_market(cfg_smoke).regimes
    )


# ------------------------------------------------------------- structure


def test_output_columns(market: Any) -> None:
    assert list(market.prices.columns) == list(LONG_COLUMNS)
    assert list(market.universe.columns) == list(UNIVERSE_COLUMNS)


def test_every_instrument_present(market: Any, cfg_smoke: dict[str, Any]) -> None:
    symbols = set(market.universe["symbol"])
    assert symbols == set(market.prices["symbol"])
    equities = market.universe[market.universe["instrument_type"] == "EQUITY"]
    assert len(equities) == int(cfg_smoke["synthetic"]["n_stocks"])


def test_date_range_respected(market: Any, cfg_smoke: dict[str, Any]) -> None:
    dates = market.prices["trade_date"]
    assert dates.min() >= pd.Timestamp(cfg_smoke["data"]["start_date"])
    assert dates.max() <= pd.Timestamp(cfg_smoke["synthetic"]["end_date"])


def test_index_and_vix_have_zero_volume(market: Any, cfg_smoke: dict[str, Any]) -> None:
    """AGENTS.md section 6: index and VIX volume is 0, so volume checks skip them."""
    for symbol in (cfg_smoke["data"]["index_symbol"], cfg_smoke["data"]["vix_symbol"]):
        subset = market.prices[market.prices["symbol"] == symbol]
        assert (subset["volume"] == 0).all()


def test_equity_volume_is_positive(market: Any) -> None:
    equities = market.prices[market.prices["symbol"].str.startswith("SYN")]
    assert (equities["volume"] > 0).all()


# ------------------------------------------------------------- calendar


def test_no_weekend_dates(market: Any) -> None:
    assert (market.prices["trade_date"].dt.dayofweek < 5).all()


def test_calendar_has_no_weekends_in_regimes(market: Any) -> None:
    assert (market.regimes.index.dayofweek < 5).all()


def test_holidays_are_removed(market: Any) -> None:
    """Roughly nine weekday holidays per year are dropped from the business days."""
    all_days = pd.bdate_range(market.prices["trade_date"].min(), market.prices["trade_date"].max())
    removed = len(all_days) - market.prices["trade_date"].nunique()
    years = all_days.year.nunique()
    assert 0 < removed <= years * 20


# ------------------------------------------------------------- OHLC sanity


def test_ohlc_consistent(market: Any) -> None:
    """`low <= min(open, close)` and `high >= max(open, close)`."""
    assert (market.prices["low"] <= market.prices[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (market.prices["high"] >= market.prices[["open", "close"]].max(axis=1) - 1e-9).all()


def test_prices_are_positive(market: Any) -> None:
    assert (market.prices[["open", "high", "low", "close"]] > 0).all().all()


def test_close_never_exceeds_high(market: Any) -> None:
    assert (market.prices["close"] <= market.prices["high"] + 1e-9).all()


# ------------------------------------------------------------- regimes


def test_all_four_states_occur(market: Any) -> None:
    """All four hidden states occur at least once over 2017 to 2025."""
    observed = set(market.regimes)
    assert observed == set(REGIME_NAMES)


def test_states_last_long_enough_to_detect(market: Any) -> None:
    """Mean duration near 80 days, so each regime spans many trading days."""
    counts = market.regimes.value_counts()
    for name in REGIME_NAMES:
        assert counts[name] > 20, f"{name} appears only {counts[name]} times"


def test_vix_higher_in_volatile_states(market: Any, cfg_smoke: dict[str, Any]) -> None:
    """VIX mean is clearly higher in HIGH_VOL and BEAR than in BULL."""
    vix = market.prices[market.prices["symbol"] == cfg_smoke["data"]["vix_symbol"]]
    vix_by_date = vix.set_index("trade_date")["close"]
    aligned = pd.concat([market.regimes.rename("regime"), vix_by_date], axis=1).dropna()

    calm = aligned.loc[aligned["regime"] == "BULL", "close"].mean()
    hot_high = aligned.loc[aligned["regime"] == "HIGH_VOL", "close"].mean()
    hot_bear = aligned.loc[aligned["regime"] == "BEAR", "close"].mean()

    assert hot_high > calm
    assert hot_bear > calm


def test_vix_never_below_floor(market: Any, cfg_smoke: dict[str, Any]) -> None:
    vix = market.prices[market.prices["symbol"] == cfg_smoke["data"]["vix_symbol"]]
    assert (vix["close"] >= 5.0).all()


def test_index_more_volatile_in_high_vol_state(market: Any, cfg_smoke: dict[str, Any]) -> None:
    index = market.prices[market.prices["symbol"] == cfg_smoke["data"]["index_symbol"]]
    index = index.set_index("trade_date")["close"].pct_change()
    aligned = pd.concat([market.regimes.rename("regime"), index.rename("ret")], axis=1).dropna()

    calm = aligned.loc[aligned["regime"] == "BULL", "ret"].std()
    hot = aligned.loc[aligned["regime"] == "HIGH_VOL", "ret"].std()
    assert hot > calm


def test_bear_state_has_negative_mean_return(market: Any, cfg_smoke: dict[str, Any]) -> None:
    index = market.prices[market.prices["symbol"] == cfg_smoke["data"]["index_symbol"]]
    index = index.set_index("trade_date")["close"].pct_change()
    aligned = pd.concat([market.regimes.rename("regime"), index.rename("ret")], axis=1).dropna()
    bear = aligned.loc[aligned["regime"] == "BEAR", "ret"]
    assert bear.mean() < 0


# ------------------------------------------------------------- stocks vs index


def test_stocks_correlate_with_index(market: Any, cfg_smoke: dict[str, Any]) -> None:
    """Beta in [0.7, 1.3] means stock returns track the index."""
    index = market.prices[market.prices["symbol"] == cfg_smoke["data"]["index_symbol"]]
    index_ret = index.set_index("trade_date")["close"].pct_change()

    equities = market.prices[market.prices["symbol"].str.startswith("SYN")]
    for symbol, group in equities.groupby("symbol"):
        stock_ret = group.set_index("trade_date")["close"].pct_change()
        joined = pd.concat([stock_ret.rename("s"), index_ret.rename("m")], axis=1).dropna()
        assert joined["s"].corr(joined["m"]) > 0.3, f"{symbol} barely tracks the index"


def test_universe_sectors_include_banking_and_it(market: Any) -> None:
    sectors = set(market.universe["sector"])
    assert {"Banking", "IT"}.issubset(sectors)


def test_sector_list_without_banking_raises(cfg_smoke: dict[str, Any]) -> None:
    cfg_smoke["synthetic"]["sectors"] = ["Energy", "Auto"]
    with pytest.raises(ValueError, match="Banking"):
        make_synthetic_market(cfg_smoke)


def test_zero_stocks_is_handled(cfg_smoke: dict[str, Any]) -> None:
    cfg_smoke["synthetic"]["n_stocks"] = 0
    prices, universe = make_synthetic_market(cfg_smoke)
    # The index and VIX still exist.
    assert set(universe["instrument_type"]) == {"INDEX", "VOLATILITY"}
    assert len(prices) > 0


def test_generation_is_reasonably_fast(cfg_smoke: dict[str, Any]) -> None:
    """Smoke runs the whole pipeline repeatedly, so generation must stay quick."""
    import time

    start = time.perf_counter()
    make_synthetic_market(cfg_smoke)
    assert time.perf_counter() - start < 15.0


def test_returns_are_finite(market: Any) -> None:
    numeric = market.prices[["open", "high", "low", "close", "volume"]].to_numpy()
    assert np.isfinite(numeric).all()
