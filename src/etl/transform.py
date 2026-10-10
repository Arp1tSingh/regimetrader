"""Transform: normalise, deduplicate, calendar, gap fill, validation (`spec 03` section 4.2).

The seven steps are applied in the documented order because each depends on the last:
the trading calendar comes from the index, gaps are filled against it, and only then is
a row rejected or flagged.

Two rules are absolute and covered by tests: nothing is ever back-filled (a row before a
symbol's first bar is never invented), and a gap longer than `data.ffill_max_days` stays
missing beyond that limit rather than being filled with stale prices.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.common.dates import to_date_key

#: Output columns for the clean frame (`03` section 4.2).
CLEAN_COLUMNS = ("symbol", "date_key", "open", "high", "low", "close", "volume", "is_flagged")

#: Reasons a row is rejected outright.
REJECT_REASONS = ("NON_POSITIVE_PRICE", "HIGH_BELOW_LOW", "OUT_OF_RANGE", "NEGATIVE_VOLUME")

#: Relative tolerance for the high/low containment check.
RANGE_TOL = 1e-6


class TransformError(ValueError):
    """Raised when the input frame cannot be transformed at all."""


def _empty_clean() -> pd.DataFrame:
    return (
        pd.DataFrame({c: pd.Series(dtype="float64") for c in CLEAN_COLUMNS})
        .assign(symbol=pd.Series(dtype="object"), date_key=pd.Series(dtype="int64"))
        .reindex(columns=list(CLEAN_COLUMNS))
    )


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Step 1: lower-case columns, strip timezone, build the date key.

    The timezone is removed with `tz_localize(None)` on the local timestamps.
    Converting to UTC first would shift the calendar date back by one day.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["symbol", "trade_date", "open", "high", "low", "close", "volume"]
        )

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    if getattr(df["trade_date"].dt, "tz", None) is not None:
        df["trade_date"] = df["trade_date"].dt.tz_localize(None)
    df = df.dropna(subset=["trade_date"])
    df["date_key"] = df["trade_date"].map(to_date_key)
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: keep the last row per (symbol, date)."""
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["symbol", "date_key"], keep="last")
        .sort_values(["symbol", "date_key"], kind="stable")
        .reset_index(drop=True)
    )


def _trading_calendar(df: pd.DataFrame, index_symbol: str) -> pd.DatetimeIndex:
    """Step 3: the set of dates on which the index has a bar."""
    index_rows = df[df["symbol"] == index_symbol]
    if index_rows.empty:
        raise TransformError(
            f"index symbol {index_symbol!r} has no bars; the trading calendar cannot be built"
        )
    return pd.DatetimeIndex(sorted(index_rows["trade_date"].unique()))


def _fill_gaps(
    df: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    cfg: Mapping[str, Any],
    instrument_types: Mapping[str, str],
) -> pd.DataFrame:
    """Step 4: forward-fill short gaps against the trading calendar.

    A gap of up to `data.ffill_max_days` consecutive trading days is filled with the
    previous close (open = high = low = close), volume 0 and `is_flagged = 1`.
    Longer gaps are filled only up to that limit; the remainder stay missing. Nothing is
    ever created before a symbol's first bar, and nothing is back-filled.

    Args:
        df: Deduplicated long frame with `trade_date`, `date_key` and prices.
        calendar: Trading dates from the index.
        cfg: Loaded config.
        instrument_types: Symbol to instrument type, so only EQUITY and VOLATILITY
            are reindexed (the index defines the calendar itself).

    Returns:
        The frame with filled rows appended.
    """
    max_fill = int(cfg["data"]["ffill_max_days"])
    if df.empty:
        return df

    out = [df]
    calendar_keys = pd.Index([to_date_key(d) for d in calendar], name="date_key")

    for symbol, group in df.groupby("symbol", sort=True):
        if instrument_types.get(symbol) not in ("EQUITY", "VOLATILITY"):
            continue

        # Only from the symbol's first bar onwards: never invent earlier rows.
        first_key = group["date_key"].min()
        target_keys = calendar_keys[calendar_keys >= first_key]

        indexed = group.set_index("date_key").sort_index()
        full = indexed.reindex(target_keys)
        full.index.name = "date_key"
        # Reindexed rows are new slots; the ones that already existed keep their flag.
        full["is_flagged"] = full["is_flagged"].fillna(0)

        missing = full["close"].isna()
        if not missing.any():
            continue

        # Position of each row within its run of consecutive missing days, 0-based.
        # Group id advances at every observed value, so each group is one gap.
        group = (~missing).cumsum()
        run = missing.astype(int).groupby(group).cumsum() - 1
        fillable = missing & (run < max_fill)

        if not fillable.any():
            continue

        # ffill carries the last known close into each missing slot.
        previous_close = full["close"].ffill()
        filled = full[fillable].copy()
        for column in ("open", "high", "low", "close"):
            filled[column] = previous_close[fillable]
        filled["volume"] = 0
        filled["is_flagged"] = 1

        # Reconstruct trade_date from the key and carry the symbol through.
        filled = filled.reset_index()
        filled["symbol"] = symbol
        filled["trade_date"] = filled["date_key"].map(lambda k: pd.Timestamp(str(int(k))))
        out.append(
            filled[
                [
                    "symbol",
                    "trade_date",
                    "date_key",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "is_flagged",
                ]
            ]
        )

    result = pd.concat(out, ignore_index=True)
    return result.sort_values(["symbol", "date_key"], kind="stable").reset_index(drop=True)


def _hard_reject(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 5: split out rows that cannot be trusted."""
    if df.empty:
        return df, df.assign(reason=pd.Series(dtype="object"))

    price_cols = ["open", "high", "low", "close"]
    bad = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index, dtype="object")

    non_positive = (df[price_cols] <= 0).any(axis=1)
    reasons.loc[non_positive] = "NON_POSITIVE_PRICE"
    bad |= non_positive

    inverted = df["high"] < df["low"]
    reasons.loc[inverted & ~bad] = "HIGH_BELOW_LOW"
    bad |= inverted

    # high must cover max(open, close) and low must not exceed min(open, close).
    hi_ref = df[["open", "close"]].max(axis=1)
    lo_ref = df[["open", "close"]].min(axis=1)
    out_of_range = (df["high"] < hi_ref * (1 - RANGE_TOL)) | (df["low"] > lo_ref * (1 + RANGE_TOL))
    reasons.loc[out_of_range & ~bad] = "OUT_OF_RANGE"
    bad |= out_of_range

    negative_volume = df["volume"].notna() & (df["volume"] < 0)
    reasons.loc[negative_volume & ~bad] = "NEGATIVE_VOLUME"
    bad |= negative_volume

    rejected = df.loc[bad].assign(reason=reasons.loc[bad])
    return df.loc[~bad], rejected


def _soft_flag(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    instrument_types: Mapping[str, str],
) -> pd.DataFrame:
    """Step 6: flag suspicious but usable rows. EQUITY only.

    Flags a large one-day move (possible unadjusted corporate action) and zero volume
    on a trading day (possible suspension). Index and VIX are exempt because their
    volume is legitimately 0. Rows already filled in step 4 keep their flag but are
    not double-counted as suspensions.
    """
    if df.empty:
        return df

    df = df.copy()
    threshold = float(cfg["data"]["anomaly_abs_return"])
    df["is_flagged"] = df["is_flagged"].fillna(0).astype(int)

    equities = df["symbol"].map(lambda s: instrument_types.get(s) == "EQUITY")

    # Large one-day move, computed within each symbol on date order.
    df = df.sort_values(["symbol", "date_key"], kind="stable")
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (df["close"] - prev_close) / prev_close
    big_move = equities & (ret.abs() > threshold)

    zero_volume = equities & (df["volume"] == 0) & (df["is_flagged"] == 0)

    df["is_flagged"] = (df["is_flagged"] | big_move | zero_volume).astype(int)
    return df


def build_dim_date(
    trading_dates: pd.DatetimeIndex | list[pd.Timestamp],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Step 7: one `dim_date` row per calendar day from the start date to the last bar.

    Args:
        trading_dates: Dates on which the index has a bar, used for `is_trading_day`.
        cfg: Loaded config; `data.start_date` sets the first row.

    Returns:
        A DataFrame with `date_key, full_date, day_of_week, day_name, week_of_year,
        month, month_name, quarter, year, is_trading_day, is_month_end`. `day_of_week`
        is 0 for Monday. `is_month_end` is set on the last trading day of each month
        present in the data (`DECISIONS` R20) and recomputed after each load.
    """
    trading_set = {to_date_key(d) for d in trading_dates}
    start = pd.Timestamp(cfg["data"]["start_date"])
    if not trading_set:
        return pd.DataFrame(
            columns=[
                "date_key",
                "full_date",
                "day_of_week",
                "day_name",
                "week_of_year",
                "month",
                "month_name",
                "quarter",
                "year",
                "is_trading_day",
                "is_month_end",
            ]
        )

    end = pd.Timestamp(str(max(trading_set)))
    days = pd.date_range(start=start, end=end, freq="D")

    dim = pd.DataFrame(
        {
            "date_key": [to_date_key(d) for d in days],
            "full_date": days.strftime("%Y-%m-%d"),
            "day_of_week": days.dayofweek,
            # `day_name` and `month_name` are methods on the index in pandas 3.
            "day_name": days.day_name(),
            "week_of_year": days.isocalendar().week.astype(int),
            "month": days.month,
            "month_name": days.month_name(),
            "quarter": days.quarter,
            "year": days.year,
        }
    )
    dim["is_trading_day"] = dim["date_key"].isin(trading_set).astype(int)
    dim["is_month_end"] = 0
    return dim


def mark_month_end(dim_date: pd.DataFrame) -> pd.DataFrame:
    """Recompute `is_month_end`: 1 on the last trading day of each month present.

    `DECISIONS` R20. Months are keyed on the last *trading* day, not the last
    calendar day, because B2 rebalances on month ends.
    """
    dim = dim_date.copy()
    dim["is_month_end"] = 0
    trading = dim[dim["is_trading_day"] == 1]
    if trading.empty:
        return dim
    last_keys = trading.groupby(trading["full_date"].str[:7])["date_key"].max()
    dim["is_month_end"] = dim["date_key"].isin(set(last_keys)).astype(int)
    return dim


def transform(raw: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the documented transform pipeline.

    Args:
        raw: Long-format extracted bars with columns `symbol, trade_date, open, high,
            low, close, volume`.
        cfg: Loaded config.

    Returns:
        `(clean, rejected)`. `clean` has columns `symbol, date_key, open, high, low,
        close, volume, is_flagged`. `rejected` carries the same columns plus `reason`.

    Raises:
        TransformError: If the input is empty or the index has no bars.
    """
    if raw is None or raw.empty:
        raise TransformError("cannot transform an empty extract")

    index_symbol = cfg["data"]["index_symbol"]
    universe_types = raw.attrs.get("instrument_types", {})

    df = _normalise(raw)
    df = _dedupe(df)

    calendar = _trading_calendar(df, index_symbol)

    # Step 3: drop rows on dates the index did not trade.
    calendar_keys = {to_date_key(d) for d in calendar}
    non_trading = ~df["date_key"].isin(calendar_keys)
    if non_trading.any():
        dropped = df.loc[non_trading].assign(reason="NON_TRADING_DAY")
        df = df.loc[~non_trading]

    df["is_flagged"] = 0
    df = _fill_gaps(df, calendar, cfg, universe_types)

    clean, rejected = _hard_reject(df)
    rejected = pd.concat([rejected, dropped], ignore_index=True) if non_trading.any() else rejected
    clean = _soft_flag(clean, cfg, universe_types)

    clean = clean[list(CLEAN_COLUMNS)].sort_values(["symbol", "date_key"], kind="stable")
    clean = clean.reset_index(drop=True)

    if not rejected.empty:
        rejected = rejected.sort_values(["symbol", "date_key"], kind="stable").reset_index(
            drop=True
        )

    return clean, rejected


def attach_instrument_types(raw: pd.DataFrame, instrument_types: Mapping[str, str]) -> pd.DataFrame:
    """Carry the symbol-to-type map on the raw frame so `transform` can use it."""
    raw = raw.copy()
    raw.attrs["instrument_types"] = dict(instrument_types)
    return raw
