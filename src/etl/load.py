"""Warehouse load (`spec 03` section 4.3).

Everything happens in one transaction so a partial load can never be observed. On any
error the transaction rolls back and a `FAILED` row is written to `etl_run_log` in a
separate short transaction, so the failure is still recorded (`03` section 4.3 step 4).

Re-running a load for the same dates changes nothing: every write is an upsert keyed on
the primary keys (AGENTS.md rule 7).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import Engine, text

from src.common.db import execute_many
from src.common.log import get_logger

from .transform import build_dim_date

#: Column order of the clean frame consumed by `load`.
LOAD_COLUMNS = ("symbol", "date_key", "open", "high", "low", "close", "volume", "is_flagged")

_UPSERT_STOCK = """
INSERT INTO dim_stock (symbol, company_name, sector, industry, instrument_type)
VALUES (:symbol, :company_name, :sector, :industry, :instrument_type)
ON CONFLICT (symbol) DO UPDATE SET
    company_name    = excluded.company_name,
    sector          = excluded.sector,
    industry        = excluded.industry,
    instrument_type = excluded.instrument_type
"""

_UPSERT_DATE = """
INSERT INTO dim_date (
    date_key, full_date, day_of_week, day_name, week_of_year,
    month, month_name, quarter, year, is_trading_day, is_month_end
)
VALUES (
    :date_key, :full_date, :day_of_week, :day_name, :week_of_year,
    :month, :month_name, :quarter, :year, :is_trading_day, 0
)
ON CONFLICT (date_key) DO UPDATE SET
    full_date    = excluded.full_date,
    day_of_week  = excluded.day_of_week,
    day_name     = excluded.day_name,
    week_of_year = excluded.week_of_year,
    month        = excluded.month,
    month_name   = excluded.month_name,
    quarter      = excluded.quarter,
    year         = excluded.year,
    is_trading_day = excluded.is_trading_day
"""

_UPSERT_PRICE = """
INSERT INTO fact_price (
    date_key, stock_key, open_price, high_price, low_price, close_price, volume, is_flagged
)
VALUES (:date_key, :stock_key, :open, :high, :low, :close, :volume, :is_flagged)
ON CONFLICT (date_key, stock_key) DO UPDATE SET
    open_price  = excluded.open_price,
    high_price  = excluded.high_price,
    low_price   = excluded.low_price,
    close_price = excluded.close_price,
    volume      = excluded.volume,
    is_flagged  = excluded.is_flagged
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dim_date_rows(trading_dates: Any, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    dim = build_dim_date(trading_dates, cfg)
    if dim.empty:
        return []
    rows = dim.to_dict("records")
    for row in rows:
        row["is_month_end"] = 0
    return rows


def _stock_rows(universe: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["symbol", "company_name", "sector", "industry", "instrument_type"]
    return universe[columns].fillna("").to_dict("records")


def _stock_key_map(engine: Engine) -> dict[str, int]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT symbol, stock_key FROM dim_stock")).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def _price_rows(clean: pd.DataFrame, keys: Mapping[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in clean[list(LOAD_COLUMNS)].to_dict("records"):
        stock_key = keys.get(record["symbol"])
        if stock_key is None:
            continue
        rows.append(
            {
                "date_key": int(record["date_key"]),
                "stock_key": stock_key,
                "open": _num(record["open"]),
                "high": _num(record["high"]),
                "low": _num(record["low"]),
                "close": _num(record["close"]),
                "volume": _num(record["volume"]),
                "is_flagged": int(record["is_flagged"] or 0),
            }
        )
    return rows


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _recompute_date_flags(conn: Any, index_symbol: str) -> None:
    """Recompute `is_trading_day` and `is_month_end` from the index's bars in the
    warehouse, not from the current batch.

    An incremental load only sees the recent slice, so deriving the calendar from the
    incoming rows would clear `is_trading_day` on every earlier trading day. The index
    is the authority for the calendar (`03` section 4.2 step 3), so it is read back
    from `fact_price` here.

    `is_month_end` is the last trading day of each calendar month present
    (`DECISIONS` R20), because B2 rebalances on month ends.
    """
    conn.execute(text("UPDATE dim_date SET is_trading_day = 0"))
    conn.execute(
        text(
            "UPDATE dim_date SET is_trading_day = 1 WHERE date_key IN ("
            "  SELECT DISTINCT f.date_key FROM fact_price f "
            "  JOIN dim_stock s ON s.stock_key = f.stock_key "
            "  WHERE s.symbol = :index_symbol)"
        ),
        {"index_symbol": index_symbol},
    )
    conn.execute(text("UPDATE dim_date SET is_month_end = 0 WHERE is_trading_day = 1"))
    conn.execute(
        text(
            """
            UPDATE dim_date
            SET is_month_end = 1
            WHERE date_key IN (
                SELECT MAX(date_key) FROM dim_date
                WHERE is_trading_day = 1
                GROUP BY SUBSTR(full_date, 1, 7)
            )
            """
        )
    )


def _log_run(
    engine: Engine,
    mode: str,
    status: str,
    started_at: str,
    rows: Mapping[str, int],
) -> None:
    """Write one `etl_run_log` row in its own transaction, so FAILED still records."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO etl_run_log "
                "(started_at, finished_at, mode, rows_extracted, rows_loaded, "
                " rows_rejected, status) "
                "VALUES (:started_at, :finished_at, :mode, :extracted, :loaded, "
                "        :rejected, :status)"
            ),
            {
                "started_at": started_at,
                "finished_at": _now(),
                "mode": mode,
                "extracted": int(rows.get("rows_extracted", 0)),
                "loaded": int(rows.get("rows_loaded", 0)),
                "rejected": int(rows.get("rows_rejected", 0)),
                "status": status,
            },
        )


def load(
    engine: Engine,
    clean: pd.DataFrame,
    universe: pd.DataFrame,
    mode: str,
    stats: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Upsert dimensions and prices in one transaction.

    Args:
        engine: Warehouse engine.
        clean: Transformed frame with the `LOAD_COLUMNS`.
        universe: Universe frame with symbol, company_name, sector, industry and
            instrument_type.
        mode: `FULL` or `INCREMENTAL`.
        stats: Extracted/rejected counts to record in the run log.
        cfg: Loaded config, needed to build `dim_date`. When omitted, `dim_date` is
            built from the dates already present in `fact_price`.

    Returns:
        Counts of what was written: `rows_extracted`, `rows_loaded`, `rows_rejected`.

    Raises:
        ValueError: If `mode` is not FULL or INCREMENTAL.
        Exception: Anything raised inside the transaction propagates after rollback,
            with a FAILED row already written to `etl_run_log`.
    """
    if mode not in ("FULL", "INCREMENTAL"):
        raise ValueError(f"mode must be FULL or INCREMENTAL, got {mode!r}")

    log = get_logger(__name__)
    started_at = _now()
    mode_flag = mode

    try:
        with engine.begin() as conn:
            # Dimensions first: fact_price references both.
            execute_many(conn, _UPSERT_STOCK, _stock_rows(universe))
            # Read the keys inside this transaction so the uncommitted dim_stock
            # rows are visible.
            keys = {
                row[0]: int(row[1])
                for row in conn.execute(text("SELECT symbol, stock_key FROM dim_stock")).fetchall()
            }

            if cfg is not None and not clean.empty:
                # `build_dim_date` already spans data.start_date to the last loaded
                # date, so an incremental load still carries every calendar day.
                trading = pd.DatetimeIndex(
                    pd.to_datetime(clean["date_key"].astype(str), format="%Y%m%d").unique()
                ).sort_values()
                execute_many(conn, _UPSERT_DATE, _dim_date_rows(trading, cfg))

            loaded = execute_many(conn, _UPSERT_PRICE, _price_rows(clean, keys))
            _recompute_date_flags(conn, cfg["data"]["index_symbol"] if cfg else "^NSEI")

        result = {
            "rows_extracted": int(stats.get("rows_extracted", len(clean))),
            "rows_loaded": loaded,
            "rows_rejected": int(stats.get("rows_rejected", 0)),
            "mode": mode_flag,
        }
        _log_run(engine, mode_flag, "SUCCESS", started_at, result)
        return result

    except Exception as exc:
        # The transaction above has rolled back. Record the failure separately.
        log.error("load failed and was rolled back: %s", exc)
        failed = {
            "rows_extracted": int(stats.get("rows_extracted", 0)),
            "rows_loaded": 0,
            "rows_rejected": int(stats.get("rows_rejected", 0)),
        }
        _log_run(engine, mode_flag, "FAILED", started_at, failed)
        raise


def price_checksum(engine: Engine) -> str:
    """Stable checksum of `fact_price`, used to prove a reload changed nothing."""
    import hashlib

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT date_key, stock_key, close_price, volume, is_flagged "
                "FROM fact_price ORDER BY date_key, stock_key"
            )
        ).fetchall()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8"))
    return digest.hexdigest()[:16]
