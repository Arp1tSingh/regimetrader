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
