"""Performance metrics (`spec 06` section 4).

Daily returns are `equity.pct_change().dropna()` and `years` is
`(last date - first date).days / 365.25`. Every metric states its units; the
risk-free rate is 0.0 by default and must be reported (`DECISIONS` R21).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Trading days per year, used to annualise volatility and Sharpe.
TRADING_DAYS = 252
#: Days per year for CAGR, matching the spec's 365.25.
DAYS_PER_YEAR = 365.25


def daily_returns(equity: pd.Series) -> pd.Series:
    """Daily returns of an equity curve, dropping the first NaN."""
    return equity.pct_change().dropna()


def _years(equity: pd.Series) -> float:
    index = equity.index
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return float("nan")
    return (index[-1] - index[0]).days / DAYS_PER_YEAR


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate, as a decimal."""
    years = _years(equity)
    if not years or years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns, as a decimal."""
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio.

    Args:
        returns: Daily returns.
        rf_annual: Annual risk-free rate as a decimal, 0.0 by default.

    Returns:
        Mean excess return over volatility, annualised.
    """
    if len(returns) < 2:
        return float("nan")
    rf_daily = float(rf_annual) / TRADING_DAYS
    excess = returns - rf_daily
    volatility = excess.std(ddof=1)
    if volatility == 0 or not np.isfinite(volatility):
        return float("nan")
    return float(excess.mean() / volatility * math.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualised Sortino ratio: excess return over downside deviation."""
    if len(returns) < 2:
        return float("nan")
    rf_daily = float(rf_annual) / TRADING_DAYS
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if downside.empty:
        return float("nan")
    deviation = np.sqrt((downside**2).mean())
    if deviation == 0:
        return float("nan")
    return float(excess.mean() / deviation * math.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Deepest peak-to-trough decline, as a positive decimal."""
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    return float((running_max - equity).max() / running_max.max())


def calmar(equity: pd.Series) -> float:
    """CAGR divided by max drawdown."""
    drawdown = max_drawdown(equity)
    growth = cagr(equity)
    if not drawdown or drawdown <= 0 or not np.isfinite(growth):
        return float("nan")
    return float(growth / drawdown)


def win_rate(returns: pd.Series) -> float:
    """Share of days with a positive return."""
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Gross gains divided by gross losses."""
    if returns.empty:
        return float("nan")
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def turnover(trades: pd.DataFrame) -> float:
    """Total traded value as a fraction of average equity.

    Args:
        trades: Frame with `quantity`, `fill_price` and optional `equity` columns.

    Returns:
        Average daily turnover, as a decimal.
    """
    if trades is None or trades.empty:
        return float("nan")
    traded = float((trades["quantity"] * trades["fill_price"]).sum())
    equity = float(trades["equity"].mean()) if "equity" in trades.columns else float("nan")
    days = int(trades["date_key"].nunique()) if "date_key" in trades.columns else len(trades)
    if not equity or days <= 0 or not np.isfinite(equity):
        return float("nan")
    return traded / equity / days


def exposure(equity: pd.DataFrame) -> float:
    """Share of days with any position, as a decimal."""
    if equity is None or equity.empty:
        return float("nan")
    if "positions_value" not in equity.columns:
        return float("nan")
    return float((equity["positions_value"] > 0).mean())


def weight_exposure(equity: pd.DataFrame) -> float:
    """Mean share of capital actually invested, as a decimal.

    `exposure` counts days with any position and is 1.0 for a book that is fully
    invested half the time and flat the other half. This weights by size, so a
    strategy that cuts to 30% in HIGH_VOL and goes to cash in BEAR is correctly
    reported as less than fully invested. Needed to tell B4 nulls apart from the
    real run: both hold a position nearly every day, but at different sizes.
    """
    if equity is None or equity.empty or "equity" not in equity.columns:
        return float("nan")
    capital = equity["equity"].astype(float)
    if "positions_value" not in equity.columns:
        return float("nan")
    usable = capital > 0
    if not usable.any():
        return float("nan")
    return float((equity["positions_value"][usable] / capital[usable]).mean())


def block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    block_days: int,
    resamples: int,
    ci_low: float,
    ci_high: float,
    rf_annual: float = 0.0,
    seed: int = 42,
) -> dict[str, float]:
    """Moving-block bootstrap interval for Sharpe (`spec 06` section 7.2).

    Daily strategy returns are serially correlated, so blocks of `block_days` are
    resampled whole rather than single days. Start indices are uniform in
    `[0, n - block]`, blocks are concatenated and trimmed to length `n`.

    Args:
        returns: Daily returns, as fractions.
        block_days: Block length in trading days.
        resamples: Number of resamples.
        ci_low: Lower percentile, e.g. 5 for the spec's 5th percentile.
        ci_high: Upper percentile.
        rf_annual: Annual risk-free rate, as a fraction.
        seed: Seed, so the interval is reproducible.

    Returns:
        `sharpe`, `ci_low`, `ci_high`, `n`, `block_days` and `resamples`. The bounds
        are NaN when the return series is too short to form one block.
    """
    values = pd.Series(returns).to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    point = sharpe(pd.Series(values), rf_annual) if n > 1 else float("nan")

    block = int(block_days)
    if n < block or block < 1 or resamples < 1:
        return {
            "sharpe": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": int(n),
            "block_days": block,
            "resamples": int(resamples),
        }

    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block)
    starts_max = n - block  # inclusive, so a block never runs past the series
    samples = np.empty((int(resamples), n), dtype=float)
    for i in range(int(resamples)):
        starts = rng.integers(0, starts_max + 1, size=blocks_needed)
        chunks = [values[s : s + block] for s in starts]
        samples[i] = np.concatenate(chunks)[:n]

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = samples.mean(axis=1)
        std = samples.std(axis=1, ddof=1)
        scale = math.sqrt(TRADING_DAYS)
        rf_daily = rf_annual / TRADING_DAYS
        null = np.where(std > 0, (mean - rf_daily) / std * scale, np.nan)
    good = null[np.isfinite(null)]

    return {
        "sharpe": point,
        "ci_low": float(np.percentile(good, ci_low)) if good.size else float("nan"),
        "ci_high": float(np.percentile(good, ci_high)) if good.size else float("nan"),
        "n": int(n),
        "block_days": block,
        "resamples": int(resamples),
    }


def summarize(
    equity_df: pd.DataFrame, trades_df: pd.DataFrame, cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Full metric set for one run.

    Args:
        equity_df: Equity curve with an index of dates and an `equity` column.
        trades_df: Trades, possibly empty.
        cfg: Loaded config; `backtest.risk_free_annual` is the only input.

    Returns:
        A dict of metrics plus `risk_free_annual` and `start`/`end`, so every results
        file states the assumptions it used.
    """
    equity = equity_df["equity"].astype(float)
    returns = daily_returns(equity)
    rf = float(cfg["backtest"]["risk_free_annual"])
    return {
        "start": str(equity_df.index[0]) if len(equity_df) else None,
        "end": str(equity_df.index[-1]) if len(equity_df) else None,
        "n_days": int(len(equity_df)),
        "initial_equity": float(equity.iloc[0]) if len(equity) else float("nan"),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float("nan"),
        "cagr": cagr(equity),
        "annual_volatility": annual_volatility(returns),
        "sharpe": sharpe(returns, rf),
        "sortino": sortino(returns, rf),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "n_trades": int(len(trades_df)) if trades_df is not None else 0,
        "turnover": turnover(trades_df) if trades_df is not None else float("nan"),
        "exposure": exposure(equity_df),
        "risk_free_annual": rf,
    }
