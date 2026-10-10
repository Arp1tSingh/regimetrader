"""Technical indicators for `fact_indicator` (`spec 04` section 2).

Every function here is **pure** and looks only backwards: `shift(1)`, trailing rolling
windows and `ewm(adjust=False)` all use the past and the present bar only. The
truncation test in `tests/test_features.py` proves that appending later rows never
changes an earlier row's value.

Indicators are computed for `EQUITY` and `INDEX` instruments only (`DECISIONS` R9):
the VIX has no meaningful price indicators and its volume is always 0.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Trading days per year, used to annualise volatility.
TRADING_DAYS = 252

#: Columns written to `fact_indicator`, in schema order.
INDICATOR_COLUMNS = (
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "rsi_14",
    "bb_pct_b",
    "atr_14",
    "vol_20d",
    "volume_ratio_20",
)

#: RSI window.
RSI_WINDOW = 14
#: Bollinger band window and width.
BB_WINDOW = 20
BB_N_STD = 2.0
#: ATR window.
ATR_WINDOW = 14
#: Volatility and volume windows.
VOL_WINDOW = 20
VOLUME_WINDOW = 20


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Wilder RSI. An all-gain window gives 100; a flat window gives 50.

    Args:
        close: Close price series.
        window: Lookback length.

    Returns:
        RSI in [0, 100], NaN during warm-up.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((loss == 0) & (gain == 0), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and signal line.

    Args:
        close: Close price series.
        fast, slow: EMA spans for the two legs.
        signal: EMA span of the signal line.

    Returns:
        `(macd, signal)`.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    return line, line.ewm(span=signal, adjust=False).mean()


def bollinger_pct_b(
    close: pd.Series, window: int = BB_WINDOW, n_std: float = BB_N_STD
) -> pd.Series:
    """Bollinger %B: where the close sits inside the bands, 0 at the lower, 1 at the upper.

    Args:
        close: Close price series.
        window: Rolling window.
        n_std: Band width in standard deviations.

    Returns:
        %B series, NaN when the band width is zero.
    """
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower, upper = mid - n_std * std, mid + n_std * std
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = ATR_WINDOW) -> pd.Series:
    """Average True Range, Wilder-smoothed.

    Args:
        high, low, close: Daily bars.
        window: Smoothing window.

    Returns:
        ATR series, NaN on the first bar (no previous close).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def annualized_vol(returns: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Annualised standard deviation of daily returns, as a decimal (0.18 = 18%)."""
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute every `fact_indicator` column for one instrument.

    Args:
        prices: One instrument's daily frame, indexed by trade date, with columns
            `open, high, low, close, volume`, ascending by date.

    Returns:
        A DataFrame indexed like `prices` with the `INDICATOR_COLUMNS` columns.
        Warm-up values stay NaN and are stored as NULL downstream.
    """
    close = prices["close"].astype("float64")
    high = prices["high"].astype("float64")
    low = prices["low"].astype("float64")
    volume = prices["volume"].astype("float64")

    out = pd.DataFrame(index=prices.index)
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_20d"] = close.pct_change(20)
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["ema_12"] = ema_12
    out["ema_26"] = ema_26
    out["macd"], out["macd_signal"] = macd(close)
    out["rsi_14"] = rsi(close)
    out["bb_pct_b"] = bollinger_pct_b(close)
    out["atr_14"] = atr(high, low, close)
    out["vol_20d"] = annualized_vol(out["return_1d"])

    volume_mean = volume.rolling(VOLUME_WINDOW).mean()
    # Index and VIX have volume 0 throughout, so the ratio is undefined, not zero.
    ratio = volume / volume_mean.replace(0, np.nan)
    out["volume_ratio_20"] = ratio

    return out[list(INDICATOR_COLUMNS)]


def build_indicators(engine: Any, cfg: Mapping[str, Any]) -> int:
    """Compute and upsert `fact_indicator` for every EQUITY and INDEX instrument.

    Indicators are always recomputed from the full history rather than incrementally:
    the data is small, and an incremental EWM would drift from the full-history value
    (`AGENTS.md` section 6).

    Args:
        engine: Warehouse engine.
        cfg: Loaded config.

    Returns:
        The number of rows upserted.
    """

    from src.common.db import execute_many, read_sql

    log_rows = read_sql(
        engine,
        "SELECT s.symbol, f.date_key, f.open_price, f.high_price, f.low_price, "
        "       f.close_price, f.volume "
        "FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE s.instrument_type IN ('EQUITY', 'INDEX') "
        "ORDER BY s.symbol, f.date_key",
    )
    if log_rows.empty:
        return 0

    keys = dict(
        read_sql(engine, "SELECT symbol, stock_key FROM dim_stock")[
            ["symbol", "stock_key"]
        ].itertuples(index=False)
    )

    upsert = """
        INSERT INTO fact_indicator (
            date_key, stock_key, return_1d, return_5d, return_20d, sma_20, sma_50,
            ema_12, ema_26, macd, macd_signal, rsi_14, bb_pct_b, atr_14, vol_20d,
            volume_ratio_20
        )
        VALUES (
            :date_key, :stock_key, :return_1d, :return_5d, :return_20d, :sma_20, :sma_50,
            :ema_12, :ema_26, :macd, :macd_signal, :rsi_14, :bb_pct_b, :atr_14, :vol_20d,
            :volume_ratio_20
        )
        ON CONFLICT (date_key, stock_key) DO UPDATE SET
            return_1d = excluded.return_1d, return_5d = excluded.return_5d,
            return_20d = excluded.return_20d, sma_20 = excluded.sma_20,
            sma_50 = excluded.sma_50, ema_12 = excluded.ema_12,
            ema_26 = excluded.ema_26, macd = excluded.macd,
            macd_signal = excluded.macd_signal, rsi_14 = excluded.rsi_14,
            bb_pct_b = excluded.bb_pct_b, atr_14 = excluded.atr_14,
            vol_20d = excluded.vol_20d, volume_ratio_20 = excluded.volume_ratio_20
    """

    written = 0
    with engine.begin() as conn:
        for symbol, group in log_rows.groupby("symbol", sort=True):
            frame = group.set_index(pd.Index(group["date_key"].astype(int), name="date_key"))
            frame = frame[["open_price", "high_price", "low_price", "close_price", "volume"]]
            frame.columns = ["open", "high", "low", "close", "volume"]
            frame = frame.sort_index()

            indicators = compute_indicators(frame)
            stock_key = int(keys[symbol])
            rows = [
                {
                    "date_key": int(date_key),
                    "stock_key": stock_key,
                    **{column: _num(value) for column, value in record.items()},
                }
                for date_key, record in indicators.iterrows()
            ]
            written += execute_many(conn, upsert, rows)

    return written


def _num(value: Any) -> float | None:
    """NaN to None so SQLite stores NULL rather than a sentinel float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)
