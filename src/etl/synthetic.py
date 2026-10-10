"""Seeded regime-switching synthetic market (`TESTS` section 2, `DECISIONS` R14).

Gives every test and `make smoke` an offline market with a known hidden regime
structure, so regime detection and strategy code can be exercised without the network.
Deterministic: the same seed always produces the same frames.

Design: a four-state Markov chain over BULL, SIDEWAYS, HIGH_VOL and BEAR drives the
index return. Stocks follow the index with a beta plus AR(1) idiosyncratic noise, so
cross-sectional dispersion and market correlation both exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Regime states and their daily index return parameters (mean, sigma) per `TESTS` 2.
REGIME_PARAMS: dict[str, tuple[float, float]] = {
    "BULL": (0.0009, 0.007),
    "SIDEWAYS": (0.0000, 0.006),
    "HIGH_VOL": (0.0000, 0.015),
    "BEAR": (-0.0020, 0.018),
}
REGIME_NAMES: tuple[str, ...] = ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR")

#: Mean regime duration in trading days; durations are geometric with this mean.
REGIME_MEAN_DURATION = 80
#: Random weekday holidays removed per year.
HOLIDAYS_PER_YEAR = 9

BETA_RANGE = (0.7, 1.3)
START_PRICE_RANGE = (100.0, 3000.0)
IDIO_SIGMA_RANGE = (0.008, 0.012)
IDIO_PHI = 0.05
OPEN_NOISE = 0.003
WICK_NOISE = 0.004
VOLUME_MEAN = 1e6
VIX_BASE = 10.0
VIX_SLOPE = 800.0
VIX_NOISE = 0.8
VIX_FLOOR = 5.0

#: Long-format output columns (`03` section 4.1).
LONG_COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume")

#: Universe frame columns, matching `dim_stock`.
UNIVERSE_COLUMNS = ("symbol", "company_name", "sector", "industry", "instrument_type")


@dataclass(frozen=True)
class SyntheticMarket:
    """Generated market: long OHLCV, the universe frame and the hidden regime chain."""

    prices: pd.DataFrame
    universe: pd.DataFrame
    regimes: pd.Series


def _trading_calendar(rng: np.random.Generator, start: str, end: str) -> pd.DatetimeIndex:
    """Business days with about nine random weekday holidays dropped per year."""
    days = pd.bdate_range(start=start, end=end, name="trade_date")
    if len(days) == 0:
        return days
    # Drop roughly HOLIDAYS_PER_YEAR weekdays per calendar year, seeded.
    years = np.asarray([ts.year for ts in days], dtype=int)
    keep = np.ones(len(days), dtype=bool)
    for year in np.unique(years):
        idx = np.flatnonzero(years == year)
        n_holiday = min(HOLIDAYS_PER_YEAR, max(0, len(idx) - 1))
        if n_holiday:
            drop = rng.choice(idx, size=n_holiday, replace=False)
            keep[drop] = False
    return days[keep]


def _regime_chain(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Sample the hidden regime chain: geometric durations, uniform next state.

    Each step either continues the current regime or switches to one of the other
    three with equal probability, which gives a mean duration near REGIME_MEAN_DURATION.
    """
    if n_days == 0:
        return np.array([], dtype=int)
    states = np.empty(n_days, dtype=int)
    current = int(rng.integers(len(REGIME_NAMES)))
    p_continue = 1.0 - 1.0 / REGIME_MEAN_DURATION
    for i in range(n_days):
        states[i] = current
        if rng.random() > p_continue:
            alternatives = [s for s in range(len(REGIME_NAMES)) if s != current]
            current = int(alternatives[rng.integers(len(alternatives))])
    return states


def _ohlc_from_close(
    prev_close: np.ndarray, close: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Build open/high/low around the close series (see `TESTS` 2)."""
    open_ = prev_close * (1.0 + rng.normal(0.0, OPEN_NOISE, size=close.size))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, WICK_NOISE, size=close.size)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, WICK_NOISE, size=close.size)))
    return {"open": open_, "high": high, "low": low, "close": close}


def _frame(
    symbol: str, dates: pd.DatetimeIndex, ohlc: dict[str, np.ndarray], volume: np.ndarray
) -> pd.DataFrame:
    """Assemble one instrument's long-format frame."""
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": dates,
            "open": ohlc["open"],
            "high": ohlc["high"],
            "low": ohlc["low"],
            "close": ohlc["close"],
            "volume": volume,
        }
    )


def generate_market(cfg: Mapping[str, Any]) -> SyntheticMarket:
    """Generate the full synthetic market including its hidden regime chain.

    Args:
        cfg: Loaded config. Uses `synthetic.seed`, `synthetic.n_stocks`,
            `synthetic.sectors`, `data.start_date`, `synthetic.end_date` and
            `data.index_symbol` / `data.vix_symbol`.

    Returns:
        A `SyntheticMarket` with the long OHLCV frame, the universe frame and the
        per-date hidden regime labels.

    Raises:
        ValueError: If the configured sector list omits Banking or IT.
    """
    seed = int(cfg["synthetic"].get("seed", 7))
    n_stocks = int(cfg["synthetic"]["n_stocks"])
    sectors = list(cfg["synthetic"]["sectors"])
    start = str(cfg["data"]["start_date"])
    end = str(cfg["synthetic"]["end_date"])
    index_symbol = cfg["data"]["index_symbol"]
    vix_symbol = cfg["data"]["vix_symbol"]

    for required in ("Banking", "IT"):
        if required not in sectors:
            raise ValueError(f"synthetic.sectors must include {required!r}; got {sectors}")

    rng = np.random.default_rng(seed)
    dates = _trading_calendar(rng, start, end)
    n_days = len(dates)

    # Hidden regime chain and index returns.
    states = _regime_chain(rng, n_days)
    regimes = pd.Series([REGIME_NAMES[s] for s in states], index=dates, name="regime")
    means = np.array([REGIME_PARAMS[REGIME_NAMES[s]][0] for s in states])
    sigmas = np.array([REGIME_PARAMS[REGIME_NAMES[s]][1] for s in states])
    index_returns = means + sigmas * rng.standard_normal(n_days)

    index_close = 1000.0 * np.cumprod(1.0 + index_returns)
    prev_index_close = np.concatenate([[index_close[0]], index_close[:-1]])
    index_ohlc = _ohlc_from_close(prev_index_close, index_close, rng)

    frames = [_frame(index_symbol, dates, index_ohlc, np.zeros(n_days, dtype=np.int64))]
    universe_rows = [
        {
            "symbol": index_symbol,
            "company_name": "Synthetic Index",
            "sector": "INDEX",
            "industry": "INDEX",
            "instrument_type": "INDEX",
        }
    ]

    # Stocks: beta to the index plus AR(1) idiosyncratic noise.
    betas = rng.uniform(*BETA_RANGE, size=n_stocks)
    idio_sigma = rng.uniform(*IDIO_SIGMA_RANGE, size=n_stocks)
    start_prices = rng.uniform(*START_PRICE_RANGE, size=n_stocks)

    for i in range(n_stocks):
        symbol = f"SYN{i + 1:02d}.NS"
        eps = np.zeros(n_days)
        prev_eps = 0.0
        for t in range(n_days):
            eps[t] = IDIO_PHI * prev_eps + rng.normal(0.0, idio_sigma[i])
            prev_eps = eps[t]
        stock_returns = betas[i] * index_returns + eps
        close = start_prices[i] * np.cumprod(1.0 + stock_returns)
        prev_close = np.concatenate([[close[0]], close[:-1]])
        ohlc = _ohlc_from_close(prev_close, close, rng)
        volume = rng.lognormal(np.log(VOLUME_MEAN), 0.5, size=n_days).astype(np.int64)
        frames.append(_frame(symbol, dates, ohlc, volume))
        universe_rows.append(
            {
                "symbol": symbol,
                "company_name": f"Synthetic Co {i + 1:02d}",
                "sector": sectors[i % len(sectors)],
                "industry": f"{sectors[i % len(sectors)]} Industry",
                "instrument_type": "EQUITY",
            }
        )

    # VIX: a deterministic function of the state sigma plus small noise.
    vix_values = VIX_BASE + VIX_SLOPE * sigmas + rng.normal(0.0, VIX_NOISE, size=n_days)
    vix_values = np.clip(vix_values, VIX_FLOOR, None)
    prev_vix = np.concatenate([[vix_values[0]], vix_values[:-1]])
    vix_ohlc = _ohlc_from_close(prev_vix, vix_values, rng)
    # Keep VIX OHLC tight around the close.
    for key in ("open", "high", "low"):
        vix_ohlc[key] = np.clip(vix_ohlc[key], vix_values * 0.99, vix_values * 1.01)
    frames.append(_frame(vix_symbol, dates, vix_ohlc, np.zeros(n_days, dtype=np.int64)))
    universe_rows.append(
        {
            "symbol": vix_symbol,
            "company_name": "Synthetic India VIX",
            "sector": "VOLATILITY",
            "industry": "VOLATILITY",
            "instrument_type": "VOLATILITY",
        }
    )

    prices = pd.concat(frames, ignore_index=True)[list(LONG_COLUMNS)]
    prices = prices.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)
    universe = pd.DataFrame(universe_rows)[list(UNIVERSE_COLUMNS)]
    universe = universe.sort_values("symbol", kind="stable").reset_index(drop=True)

    return SyntheticMarket(prices=prices, universe=universe, regimes=regimes)


def make_synthetic_market(cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a seeded, regime-switching market.

    Args:
        cfg: Loaded config, as for `generate_market`.

    Returns:
        `(prices, universe)`: long-format OHLCV with columns
        `symbol, trade_date, open, high, low, close, volume` covering every
        instrument, and the universe frame with columns `symbol, company_name,
        sector, industry, instrument_type`.

    Raises:
        ValueError: If the configured sector list omits Banking or IT.
    """
    market = generate_market(cfg)
    return market.prices, market.universe


def synthetic_regimes(cfg: Mapping[str, Any]) -> pd.Series:
    """Return the hidden regime chain, for tests that assert on regime detection.

    The chain is a property of the generator seed, not of the prices alone, so it is
    regenerated here rather than inferred.
    """
    return generate_market(cfg).regimes
