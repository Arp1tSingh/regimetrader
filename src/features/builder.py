"""Feature builder CLI (`spec 07` section 5).

`python -m src.features.builder --indicators` computes `fact_indicator` for every
EQUITY and INDEX instrument from the full history and upserts it. Reruns are
idempotent because every write is an upsert keyed on the primary key.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.common.config import ConfigError, load_config
from src.common.db import get_engine, init_db
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.features.indicators import build_indicators

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.features.builder",
        description="Build derived features from the warehouse.",
    )
    parser.add_argument("--indicators", action="store_true", help="Build fact_indicator (default).")
    parser.add_argument("--panel", action="store_true", help="Build the modelling panel (Phase 4).")
    parser.add_argument("--all", action="store_true", help="Build indicators and the panel.")
    parser.add_argument("--config", default=None, help="Override config file.")
    return parser


def run(cfg: dict[str, Any], *, indicators: bool, panel: bool) -> int:
    """Build the requested artefacts and return an exit code."""
    log = get_logger("src.features.builder", cfg)
    engine = get_engine(cfg)
    init_db(engine)

    if indicators or not panel:
        written = build_indicators(engine, cfg)
        log.info("wrote %d indicator row(s)", written)
        write_manifest(cfg, "features_indicators", engine, extra={"rows": written})

    if panel:
        # The panel needs fact_market_regime, which Phase 3 produces. Until then this
        # is a clear message rather than a crash.
        log.warning("panel build requested but not yet available; run `make regime` first")

    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 unexpected, 2 data failure."""
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        return run(
            cfg,
            indicators=args.indicators or args.all,
            panel=args.panel or args.all,
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the CLI must not traceback at the user
        get_logger("src.features.builder", cfg).exception("builder failed")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
