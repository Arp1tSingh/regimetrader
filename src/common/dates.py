"""Date-key helpers.

Warehouse date keys are `int(YYYYMMDD)` integers (AGENTS.md section 6), never floats
or strings. yfinance timestamps are timezone-aware in Asia/Kolkata: strip the tz with
`tz_localize(None)` and **do not** convert to UTC first, which would shift the date back
one day.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def to_date_key(ts: Any) -> int:
    """Convert a timestamp-like value to an `int(YYYYMMDD)` date key.

    Args:
        ts: `pd.Timestamp`, `np.datetime64`, `datetime.date`, or an ISO date string.
            Time of day and any timezone offset are discarded; the local calendar
            date in Asia/Kolkata is what the key represents.

    Returns:
        The date as an integer, e.g. `20240105`.
    """
    if isinstance(ts, pd.Timestamp):
        stamp = ts
    elif isinstance(ts, np.datetime64):
        stamp = pd.Timestamp(ts)
    elif isinstance(ts, str):
        stamp = pd.Timestamp(ts)
    else:
        stamp = pd.Timestamp(ts)

    if stamp.tzinfo is not None:
        # Localize away, never convert: converting shifts the calendar date.
        stamp = stamp.tz_localize(None)
    return int(stamp.strftime("%Y%m%d"))


def from_date_key(key: int | str) -> pd.Timestamp:
    """Convert an `int(YYYYMMDD)` date key back to a midnight `pd.Timestamp`.

    Args:
        key: Date key as an int, a numeric string or an ISO date string.

    Returns:
        A naive `pd.Timestamp` at midnight.
    """
    if isinstance(key, str) and not key.isdigit():
        return pd.Timestamp(key).normalize()
    text = str(int(key))
    if len(text) != 8:
        raise ValueError(f"date key must be 8 digits (YYYYMMDD), got {key!r}")
    return pd.Timestamp(year=int(text[:4]), month=int(text[4:6]), day=int(text[6:8]))
