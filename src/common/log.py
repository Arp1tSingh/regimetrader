"""Logging to console and `logs/<module>.log`.

WARNING is used for fallbacks so a degraded run is visible in both destinations
(AGENTS.md section 7: no printing, use logging).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_logs_dir(cfg: Mapping[str, Any] | None) -> Path | None:
    if not cfg:
        return None
    target = cfg.get("paths", {}).get("logs")
    return Path(target) if target else None


def get_logger(name: str, cfg: Mapping[str, Any] | None = None) -> logging.Logger:
    """Return a logger writing INFO and above to the console and to a file.

    Args:
        name: Usually the module's `__name__`; also the log file stem.
        cfg: Loaded config. When present, logs go to `paths.logs/<name>.log`.
            When absent, console logging only.

    Returns:
        A configured `logging.Logger`. Handlers are attached once per name.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_regimetrader_configured", False):
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    logs_dir = _resolve_logs_dir(cfg)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._regimetrader_configured = True  # type: ignore[attr-defined]
    return logger
