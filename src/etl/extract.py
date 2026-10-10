"""Extraction from yfinance, CSV files or the synthetic generator (`spec 03` section 4.1).

Retry policy comes from `data.retries` and `data.retry_wait_seconds`: a symbol that
still fails after the last attempt is logged and skipped, never fatal, because one
delisted ticker must not stop the warehouse load (`03` section 3).

Raw responses are cached to `data/raw/<symbol>_<run_date>.csv` and staged into
`stg_price_raw`.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from sqlalchemy import Engine, text

from src.common.log import get_logger

from .synthetic import make_synthetic_market

#: Long-format output columns (`03` section 4.1).
LONG_COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume")

#: Accepted values for `data.source`.
SOURCES = ("yfinance", "csv", "synthetic")


class ExtractError(RuntimeError):
    """Raised when extraction cannot proceed at all (bad source name, no symbols)."""


def _retry_waits(cfg: Mapping[str, Any]) -> list[float]:
    waits = cfg["data"].get("retry_wait_seconds") or [0]
    return [float(w) for w in waits]


def _attempts(cfg: Mapping[str, Any]) -> int:
    """Total attempts: the configured retries, plus the first try."""
    return max(1, int(cfg["data"].get("retries", 3)))


def _with_retry(
    fetch: Callable[[], pd.DataFrame],
    *,
    cfg: Mapping[str, Any],
    symbol: str,
    sleep: Callable[[float], None],
) -> pd.DataFrame | None:
    """Call `fetch`, retrying per config. Returns None after the final failure."""
    waits = _retry_waits(cfg)
    log = get_logger(__name__)
    for attempt in range(_attempts(cfg)):
        try:
            return fetch()
        except Exception as exc:  # noqa: BLE001 - any provider failure is retried
            if attempt == _attempts(cfg) - 1:
                log.warning("giving up on %s after %d attempts: %s", symbol, attempt + 1, exc)
                return None
            delay = waits[min(attempt, len(waits) - 1)]
            log.warning(
                "retry %d/%d for %s in %.0fs: %s", attempt + 1, _attempts(cfg), symbol, delay, exc
            )
            if delay:
                sleep(delay)
    return None


def _empty(symbol: str) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LONG_COLUMNS}).assign(symbol=symbol)


def _normalise(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Coerce any provider frame to the documented long columns.

    Handles yfinance's tz-aware DatetimeIndex and the CSV fallback's date column.
    """
    if df is None or df.empty:
        return _empty(symbol)

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    # Date: either the index or a column.
    if "trade_date" not in out.columns:
        for candidate in ("date", "datetime", "timestamp"):
            if candidate in out.columns:
                out = out.rename(columns={candidate: "trade_date"})
                break
    if "trade_date" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "trade_date"})

    # yfinance returns Open/High/Low/Close/Volume; the CSV source uses open/... already.
    for column in ("open", "high", "low", "close", "volume"):
        if column not in out.columns:
            out[column] = pd.NA

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    # Strip any timezone without converting: converting would shift the date back a day.
    if getattr(out["trade_date"].dt, "tz", None) is not None:
        out["trade_date"] = out["trade_date"].dt.tz_localize(None)

    for column in ("open", "high", "low", "close", "volume"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["symbol"] = symbol
    out = out.dropna(subset=["trade_date"])
    return out[list(LONG_COLUMNS)].sort_values("trade_date", kind="stable").reset_index(drop=True)


def _yfinance_frame(symbol: str, start: str) -> pd.DataFrame:
    """Fetch one symbol from yfinance. Imported lazily so tests can monkeypatch."""
    import yfinance as yf

    return yf.Ticker(symbol).history(start=start, auto_adjust=True)


def _csv_frame(cfg: Mapping[str, Any], symbol: str, start: str) -> pd.DataFrame:
    """Read `data.csv_dir/<symbol>.csv`."""
    path = Path(cfg["data"]["csv_dir"]) / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found for {symbol}: {path}")
    return pd.read_csv(path)


def _save_raw(cfg: Mapping[str, Any], symbol: str, frame: pd.DataFrame, run_date: str) -> None:
    """Cache one raw response to `data/raw/<symbol>_<run_date>.csv`."""
    if frame is None or frame.empty:
        return
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw_dir / f"{symbol}_{run_date}.csv", index=False)


def _stage(engine: Engine, frame: pd.DataFrame) -> None:
    """Truncate and refill `stg_price_raw`."""
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM stg_price_raw"))
        if frame.empty:
            return
        conn.execute(
            text(
                "INSERT INTO stg_price_raw "
                "(symbol, trade_date, open_price, high_price, low_price, close_price, volume) "
                "VALUES (:symbol, :trade_date, :open, :high, :low, :close, :volume)"
            ),
            [
                {
                    "symbol": r["symbol"],
                    "trade_date": r["trade_date"].strftime("%Y-%m-%d"),
                    "open": _as_float(r["open"]),
                    "high": _as_float(r["high"]),
                    "low": _as_float(r["low"]),
                    "close": _as_float(r["close"]),
                    "volume": _as_float(r["volume"]),
                }
                for r in frame.to_dict("records")
            ],
        )


def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _vix_fallback(cfg: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    """Load the VIX from `data.vix_csv` (columns date, close; OHLC default to close)."""
    path = Path(cfg["data"]["vix_csv"])
    if not path.exists():
        return _empty(symbol)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "close" not in df.columns:
        return _empty(symbol)
    if "trade_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "trade_date"})
    for column in ("open", "high", "low"):
        if column not in df.columns:
            df[column] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0
    return df


def extract(
    cfg: Mapping[str, Any],
    symbols: list[str],
    start: str,
    source: str,
    engine: Engine | None = None,
    *,
    run_date: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    yf_fetcher: Callable[[str, str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Fetch bars for `symbols` and return them in long format.

    Args:
        cfg: Loaded config.
        symbols: Symbols to fetch (universe plus index and VIX).
        start: Inclusive start date, `YYYY-MM-DD`.
        source: One of `yfinance`, `csv`, `synthetic`.
        engine: Optional warehouse engine; when given, the result is staged into
            `stg_price_raw`.
        run_date: Date stamp for the raw cache files. Defaults to today.
        sleep: Injected for tests so retries do not wait.
        yf_fetcher: Injected for tests; defaults to the real yfinance call.

    Returns:
        Long-format DataFrame with columns `symbol, trade_date, open, high, low,
        close, volume`. Symbols that failed every retry are absent, not faked.

    Raises:
        ExtractError: If `source` is unknown or `symbols` is empty.
    """
    if source not in SOURCES:
        raise ExtractError(f"unknown source {source!r}; expected one of {SOURCES}")
    if not symbols:
        raise ExtractError("no symbols to extract")

    log = get_logger(__name__)
    run_date = run_date or date.today().isoformat()
    fetch_yf = yf_fetcher or _yfinance_frame

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    if source == "synthetic":
        # The generator returns the whole market at once; no retry loop is needed.
        # Its universe is carried through on `attrs` because the synthetic symbols
        # differ from the real Nifty 50 file and the loader must use these instead.
        prices, synthetic_universe = make_synthetic_market(cfg)
        prices = prices[prices["trade_date"] >= pd.Timestamp(start)]
        frames.append(prices)
        frames.append(prices)
        result = pd.concat(frames, ignore_index=True)[list(LONG_COLUMNS)]
        result = result.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)
        result.attrs["universe"] = synthetic_universe
        if engine is not None:
            _stage(engine, result)
        return result
    else:
        for symbol in symbols:
            if source == "yfinance":
                fetch = lambda s=symbol: fetch_yf(s, start)  # noqa: E731
            else:
                fetch = lambda s=symbol: _csv_frame(cfg, s, start)  # noqa: E731

            raw = _with_retry(fetch, cfg=cfg, symbol=symbol, sleep=sleep)
            if raw is None or (isinstance(raw, pd.DataFrame) and raw.empty):
                failed.append(symbol)
                log.warning("no data for %s from source %s", symbol, source)
                continue
            frame = _normalise(raw, symbol)
            _save_raw(cfg, symbol, raw, run_date)
            frames.append(frame)

        # VIX fallback when the primary VIX fetch came back empty.
        vix = cfg["data"]["vix_symbol"]
        if vix in failed:
            fallback = _vix_fallback(cfg, vix)
            if not fallback.empty:
                frame = _normalise(fallback, vix)
                frames.append(frame)
                failed.remove(vix)
                log.info("loaded VIX from fallback CSV")

    result = (
        pd.concat(frames, ignore_index=True)[list(LONG_COLUMNS)]
        if frames
        else pd.DataFrame({c: pd.Series(dtype="object") for c in LONG_COLUMNS})
    )
    result = result.sort_values(["symbol", "trade_date"], kind="stable").reset_index(drop=True)

    if engine is not None:
        _stage(engine, result)

    if failed:
        log.warning("%d symbol(s) produced no data: %s", len(failed), ", ".join(sorted(failed)))
    return result
