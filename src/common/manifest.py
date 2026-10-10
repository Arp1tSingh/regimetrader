"""Run manifests (`spec 07` section 4).

Every backtest and paper run writes `reports/results/manifests/<run_id>.json` so a
result can be traced to the code, config and data that produced it (`AGENTS.md` rule 8).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import Engine, text

from .config import config_hash


def _git_commit() -> str:
    """Current commit hash, or `unknown` when this is not a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _package_versions() -> dict[str, str]:
    """Versions of the packages that can change a number in the results."""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for label, module in (
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
    ):
        try:
            versions[label] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - a missing optional package must not break the run
            versions[label] = "not installed"
    return versions


def _data_range(engine: Engine) -> dict[str, Any]:
    """First and last date_key present in `fact_price`, or None when empty."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT MIN(date_key), MAX(date_key) FROM fact_price")).first()
    except Exception:  # noqa: BLE001 - the table may not exist yet on a fresh database
        return {"first_date_key": None, "last_date_key": None}
    if row is None or row[0] is None:
        return {"first_date_key": None, "last_date_key": None}
    return {"first_date_key": int(row[0]), "last_date_key": int(row[1])}


def write_manifest(
    cfg: Mapping[str, Any],
    run_id: str,
    engine: Engine | None = None,
    *,
    models_used: list[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the manifest for one run.

    Args:
        cfg: Loaded config.
        run_id: Run identifier, also the file stem.
        engine: Optional warehouse engine, used to record the loaded data range.
            When omitted, the range is recorded as unknown.
        models_used: Model names involved in the run.
        extra: Anything else worth recording, for example the period or the CLI flags.

    Returns:
        Path of the written manifest.
    """
    from .db import get_engine

    engine = engine or get_engine(cfg)
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config_hash": config_hash(dict(cfg)),
        "seed": int(cfg["project"]["seed"]),
        "command_line": " ".join(sys.argv),
        "package_versions": _package_versions(),
        "data_range": _data_range(engine),
        "models_used": list(models_used or []),
        "extra": dict(extra or {}),
    }

    out_dir = Path(cfg["paths"]["results"]) / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
