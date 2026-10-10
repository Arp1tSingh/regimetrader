"""Shared pytest fixtures.

Unit tests use the synthetic generator and temporary SQLite files only:
never the real warehouse, never the network (AGENTS.md section 7).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_CONFIG = REPO_ROOT / "config" / "config.yaml"
SMOKE_CONFIG = REPO_ROOT / "config" / "config.smoke.yaml"

#: Keys under `paths:` that name a directory rather than a file.
DIR_PATH_KEYS = ("raw", "external", "processed", "models", "results", "figures", "daily", "logs")
#: Keys under `paths:` that name a file.
FILE_PATH_KEYS = ("db", "frozen_config")


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def cfg_smoke(tmp_path: Path) -> dict[str, Any]:
    """The smoke configuration, redirected so every path lives under `tmp_path`.

    Used by any test that must not touch the real warehouse, models or reports.
    """
    from src.common.config import load_config

    cfg = load_config(str(SMOKE_CONFIG))
    for key in DIR_PATH_KEYS:
        cfg["paths"][key] = str(tmp_path / key)
    for key in FILE_PATH_KEYS:
        cfg["paths"][key] = str(tmp_path / Path(cfg["paths"][key]).name)
    cfg["data"]["universe_file"] = str(tmp_path / "nifty50_symbols.csv")
    cfg["data"]["vix_csv"] = str(tmp_path / "external" / "india_vix.csv")
    cfg["data"]["csv_dir"] = str(tmp_path / "external" / "prices")
    return cfg


@pytest.fixture
def cfg_base() -> dict[str, Any]:
    """The unmodified base configuration, read straight from disk."""
    from src.common.config import load_config

    return load_config(str(BASE_CONFIG))
