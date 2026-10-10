"""ETL pipeline CLI (`spec 03` sections 4 and 5, `07` section 5).

Runs extract -> transform -> load -> quality checks and returns a process exit code:
0 success, 1 unexpected error, 2 data or network failure, 4 hard data-quality failure.

An incremental run starts from the last loaded index date minus
`data.incremental_overlap_days`; an empty warehouse is always a full refresh
(`03` section 4.1 step 6). Because every load is an upsert, re-running the same dates
changes nothing.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.common.config import ConfigError, load_config
from src.common.db import get_engine, init_db
from src.common.log import get_logger
from src.common.manifest import write_manifest

from .extract import ExtractError, extract
from .load import load
from .quality import DataQualityError, run_checks
from .transform import TransformError, attach_instrument_types, transform
from .universe import UniverseError, load_universe

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2
EXIT_GUARD = 3
EXIT_QUALITY = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.etl.pipeline",
        description="Load market data into the warehouse.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Reload from data.start_date. Default is incremental.",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "csv", "synthetic"],
        default=None,
        help="Override data.source.",
    )
    parser.add_argument(
        "--limit-symbols",
        type=int,
        default=None,
        help="Load only the first N symbols (debugging; not used by make smoke).",
    )
    parser.add_argument("--config", default=None, help="Override config file.")
    return parser


def _incremental_start(engine: Any, cfg: dict[str, Any]) -> str | None:
    """Start date for an incremental run, or None when a full refresh is required."""
    try:
        with engine.connect() as conn:
            last = conn.execute(
                text(
                    "SELECT MAX(f.date_key) FROM fact_price f "
                    "JOIN dim_stock s ON s.stock_key = f.stock_key "
                    "WHERE s.symbol = :index_symbol"
                ),
                {"index_symbol": cfg["data"]["index_symbol"]},
            ).scalar()
    except Exception:  # noqa: BLE001 - a missing table just means "empty warehouse"
        return None
    if last is None:
        return None

    from src.common.dates import from_date_key

    overlap = int(cfg["data"]["incremental_overlap_days"])
    start = from_date_key(int(last)) - timedelta(days=overlap)
    return start.strftime("%Y-%m-%d")


def run(
    cfg: dict[str, Any],
    *,
    full_refresh: bool,
    source: str | None,
    limit_symbols: int | None,
) -> int:
    """Execute the pipeline once and return an exit code."""
    log = get_logger("src.etl.pipeline", cfg)
    engine = get_engine(cfg)
    init_db(engine)

    source_name = source or cfg["data"]["source"]
    cfg["data"]["source"] = source_name

    universe = load_universe(cfg)
    if limit_symbols:
        equities = universe[universe["instrument_type"] == "EQUITY"].head(limit_symbols)
        seeds = universe[universe["instrument_type"] != "EQUITY"]
        universe = pd.concat([equities, seeds], ignore_index=True)
        universe = universe.sort_values("symbol", kind="stable").reset_index(drop=True)

    symbols = list(universe["symbol"])

    # An empty warehouse is always a full refresh (`03` section 4.1 step 6).
    start = cfg["data"]["start_date"]
    mode = "FULL"
    if not full_refresh:
        incremental = _incremental_start(engine, cfg)
        if incremental is not None:
            start = incremental
            mode = "INCREMENTAL"
            log.info("incremental run from %s", start)
        else:
            log.info("warehouse is empty, running a full refresh")

    run_id = f"etl_{date.today().isoformat()}"
    log.info("extracting %d symbol(s) from %s since %s", len(symbols), source_name, start)

    try:
        raw = extract(cfg, symbols, start, source_name, engine=engine)
    except ExtractError as exc:
        log.error("extract failed: %s", exc)
        return EXIT_DATA_FAILURE

    if raw.empty:
        log.error("extract returned no rows; refusing to load")
        return EXIT_DATA_FAILURE

    # The synthetic generator has its own symbols (SYN01..SYNn). The real Nifty 50
    # file must not be loaded alongside them, so `extract` carries the generated
    # universe through on `attrs` and it wins here.
    synthetic_universe = raw.attrs.get("universe")
    if synthetic_universe is not None:
        universe = synthetic_universe
        if limit_symbols:
            # Restrict the generated market to the first N equities, always keeping
            # the index and VIX: they define the trading calendar and the regime
            # features, so the load is meaningless without them.
            index_symbol = cfg["data"]["index_symbol"]
            vix_symbol = cfg["data"]["vix_symbol"]
            equities = universe[universe["instrument_type"] == "EQUITY"].head(limit_symbols)
            seeds = universe[
                (universe["symbol"] == index_symbol) | (universe["symbol"] == vix_symbol)
            ]
            universe = pd.concat([equities, seeds], ignore_index=True)
            universe = universe.sort_values("symbol", kind="stable").reset_index(drop=True)
            raw = raw[raw["symbol"].isin(set(universe["symbol"]))].reset_index(drop=True)
            log.info("restricted to %d synthetic equities", len(equities))

    types = dict(zip(universe["symbol"], universe["instrument_type"], strict=True))

    try:
        clean, rejected = transform(attach_instrument_types(raw, types), cfg)
    except TransformError as exc:
        log.error("transform failed: %s", exc)
        return EXIT_DATA_FAILURE

    rejected_path = None
    if not rejected.empty:
        raw_dir = cfg["paths"]["raw"]
        rejected_path = f"{raw_dir}/rejected_{date.today().isoformat()}.csv"
        rejected.to_csv(rejected_path, index=False)
        log.warning("%d row(s) rejected, written to %s", len(rejected), rejected_path)

    stats = {"rows_extracted": len(raw), "rows_rejected": len(rejected)}

    try:
        result = load(engine, clean, universe, mode, stats, cfg)
    except Exception as exc:  # noqa: BLE001 - load already logged and rolled back
        log.error("load failed: %s", exc)
        return EXIT_ERROR

    log.info("loaded %d row(s) in %s mode", result["rows_loaded"], result["mode"])

    try:
        results = run_checks(engine, cfg, result)
    except DataQualityError as exc:
        log.error("%s", exc)
        return EXIT_QUALITY

    failures = [r for r in results if not r.passed]
    if failures:
        log.warning("%d soft check(s) failed: %s", len(failures), ", ".join(r.id for r in failures))

    write_manifest(
        cfg,
        run_id,
        engine,
        extra={
            "mode": result["mode"],
            "source": source_name,
            "start_date": start,
            "symbols": len(symbols),
            **result,
        },
    )
    log.info("pipeline complete")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 unexpected, 2 data failure, 4 quality."""
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        return run(
            cfg,
            full_refresh=args.full_refresh,
            source=args.source,
            limit_symbols=args.limit_symbols,
        )
    except UniverseError as exc:
        # The universe is never fabricated; this must reach the human.
        get_logger("src.etl.pipeline", cfg).error("%s", exc)
        print(f"universe error: {exc}", file=sys.stderr)
        return EXIT_DATA_FAILURE
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the CLI must not traceback at the user
        get_logger("src.etl.pipeline", cfg).exception("unexpected pipeline failure")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
