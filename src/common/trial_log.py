"""Experiment trial log (`spec 07` section 4, `DECISIONS` R28).

One row per configuration tried, in `reports/results/trial_log.csv`. Rows are
**upserted** on (`kind`, `run_id`, `model`, `params_json`, `period`) so that
re-running an idempotent step never inflates the reported experiment count. The
count reported in the write-up is the number of rows with `kind != 'calibration'`.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

#: Column order of the CSV. Fixed: downstream tables read it positionally by name.
COLUMNS = (
    "timestamp",
    "phase",
    "kind",
    "run_id",
    "model",
    "params_json",
    "period",
    "metric",
    "value",
    "extra_json",
    "selected",
    "note",
)

#: `kind` values. Calibration rows are diagnostics, not configurations.
KINDS = ("regime_k", "model_year", "strategy_grid", "calibration", "other")

#: Rows whose `kind` is calibration are excluded from the experiment count.
NON_TRIAL_KINDS = ("calibration",)


def _canonical_json(value: Any) -> str:
    """Stable JSON so the upsert key does not depend on dict ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _trial_path(cfg: Mapping[str, Any]) -> Path:
    path = Path(cfg["paths"]["results"]) / "trial_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _key_of(row: Mapping[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("kind", ""),
        row.get("run_id", ""),
        row.get("model", ""),
        row.get("params_json", ""),
        row.get("period", ""),
    )


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def log_trial(
    cfg: Mapping[str, Any],
    *,
    phase: str,
    kind: str,
    run_id: str,
    model: str | None = None,
    params: Mapping[str, Any] | None = None,
    period: str | None = None,
    metric: str | None = None,
    value: float | None = None,
    extra: Mapping[str, Any] | None = None,
    selected: bool = False,
    note: str = "",
) -> None:
    """Append or update one trial row.

    Args:
        cfg: Loaded config; the CSV lives at `paths.results/trial_log.csv`.
        phase: Build phase that produced the trial, for example `dev` or `tune`.
        kind: One of `KINDS`.
        run_id: Run the trial belongs to.
        model: Model name, or None for strategy-only trials.
        params: The configuration tried. Serialised canonically into the key.
        period: Period scored on, for example `validation`.
        metric: Headline metric name, for example `sharpe` or `accuracy`.
        value: Headline metric value.
        extra: Any other numbers measured; stored as JSON.
        selected: True when this configuration won the tuning grid.
        note: Free text.

    Raises:
        ValueError: If `kind` is not a known value.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown trial kind {kind!r}; expected one of {KINDS}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase,
        "kind": kind,
        "run_id": run_id,
        "model": model or "",
        "params_json": _canonical_json(dict(params or {})),
        "period": period or "",
        "metric": metric or "",
        "value": "" if value is None else repr(float(value)),
        "extra_json": _canonical_json(dict(extra or {})),
        "selected": "1" if selected else "0",
        "note": note,
    }

    path = _trial_path(cfg)
    rows = _read_rows(path)
    new_key = _key_of(row)
    for index, existing in enumerate(rows):
        if _key_of(existing) == new_key:
            # Upsert in place, keeping the original timestamp of first record.
            row["timestamp"] = existing.get("timestamp", row["timestamp"])
            rows[index] = row
            break
    else:
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_trials(cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return every logged trial row."""
    return _read_rows(_trial_path(cfg))


def count_trials(cfg: Mapping[str, Any]) -> int:
    """Count logged configurations, excluding calibration diagnostics (`DECISIONS` R28)."""
    return sum(1 for row in read_trials(cfg) if row.get("kind") not in NON_TRIAL_KINDS)
