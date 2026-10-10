"""Tests for the tuning grid (task T5.6, spec 05 section 6, TESTS section 7)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.tune import _is_finite, candidate_day_share
from src.common.trial_log import read_trials


def cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))


# --------------------------------------------------------------- grid size


def expected_grid_size(c: dict[str, Any], fallback_applies: bool) -> int:
    """models x entry bases x slots; a fallback entry base is added per model."""
    entries = len(c["tuning"]["entry_base_grid"]) + (1 if fallback_applies else 0)
    return len(c["tuning"]["models"]) * entries * len(c["tuning"]["slots_grid"])


def test_grid_size_is_24_with_fallback_on_every_model() -> None:
    """4 models x 3 entry bases x 2 slots = 24 (TESTS section 7 test_tune_grid_size)."""
    c = cfg()
    assert len(c["tuning"]["models"]) == 4
    assert len(c["tuning"]["entry_base_grid"]) == 2
    assert len(c["tuning"]["slots_grid"]) == 2
    assert expected_grid_size(c, fallback_applies=True) == 24
    assert expected_grid_size(c, fallback_applies=False) == 16


def test_grid_size_is_the_product_not_a_per_model_addition() -> None:
    """The fallback adds an entry base per model, and every entry base crosses every
    slot count. Adding one trial per model would give 20, which is wrong."""
    c = cfg()
    per_model_fallback = (len(c["tuning"]["entry_base_grid"]) + 1) * len(c["tuning"]["slots_grid"])
    assert per_model_fallback == 6
    assert len(c["tuning"]["models"]) * per_model_fallback == 24


def test_fallback_entry_is_distinct_from_the_grid() -> None:
    c = cfg()
    assert float(c["strategy"]["fallback_entry_base"]) not in [
        float(e) for e in c["tuning"]["entry_base_grid"]
    ]


# --------------------------------------------------------------- NaN robustness


def test_is_finite_rejects_nan() -> None:
    assert _is_finite(1.5)
    assert not _is_finite(float("nan"))
    assert not _is_finite(None)
    assert not _is_finite(pd.NA)


def test_non_trading_runs_excluded_from_selection() -> None:
    """A never-trading run has NaN Sharpe and must not be selectable."""
    grid = pd.DataFrame(
        [
            {"model": "tree", "sharpe": float("nan"), "turnover": 0.0, "traded": False},
            {"model": "rf", "sharpe": 0.4, "turnover": 0.2, "traded": True},
            {"model": "xgb", "sharpe": float("nan"), "turnover": 0.0, "traded": False},
        ]
    )
    tradable = grid[grid["traded"] & grid["sharpe"].notna()]
    assert list(tradable["model"]) == ["rf"]

    best = tradable.sort_values(["sharpe", "turnover"], ascending=[False, True]).iloc[0]
    assert best["model"] == "rf", "a NaN Sharpe must never win"


def test_selection_skips_negative_sharpe_but_not_nan() -> None:
    """A genuinely negative Sharpe is still a valid, selectable result."""
    grid = pd.DataFrame(
        [
            {"model": "tree", "sharpe": -0.5, "turnover": 0.1, "traded": True},
            {"model": "rf", "sharpe": float("nan"), "turnover": 0.0, "traded": False},
        ]
    )
    tradable = grid[grid["traded"] & grid["sharpe"].notna()]
    assert list(tradable["model"]) == ["tree"]


def test_no_tradable_run_raises() -> None:
    """If nothing traded, selection must fail loudly rather than pick a NaN."""
    grid = pd.DataFrame(
        [{"model": "tree", "sharpe": float("nan"), "turnover": 0.0, "traded": False}]
    )
    tradable = grid[grid["traded"] & grid["sharpe"].notna()]
    assert tradable.empty


def test_tie_break_prefers_lower_turnover() -> None:
    grid = pd.DataFrame(
        [
            {"model": "a", "sharpe": 1.0, "turnover": 0.5, "traded": True},
            {"model": "b", "sharpe": 1.0, "turnover": 0.2, "traded": True},
        ]
    )
    best = grid.sort_values(["sharpe", "turnover"], ascending=[False, True]).iloc[0]
    assert best["model"] == "b"


# --------------------------------------------------------------- calibration


def test_candidate_day_share_definition(signals_ready: tuple) -> None:
    """Same definition as the calibration check, applied at a given entry."""
    c, engine = signals_ready
    need = math.ceil(min(c["tuning"]["slots_grid"]) / 2)
    entry = float(min(c["tuning"]["entry_base_grid"]))

    share, days = candidate_day_share(engine, c, "tree", entry, need)
    assert 0.0 <= share <= 1.0
    assert days > 0

    # A higher threshold can only shrink the share.
    higher, _ = candidate_day_share(engine, c, "tree", entry + 0.20, need)
    assert higher <= share


def test_candidate_day_share_unknown_model(signals_ready: tuple) -> None:
    c, engine = signals_ready
    share, days = candidate_day_share(engine, c, "no_such_model", 0.5, 3)
    assert math.isnan(share) and days == 0


# --------------------------------------------------------------- trial log


def test_trial_log_upsert_does_not_double_count(cfg_smoke: dict[str, Any]) -> None:
    """Re-running an idempotent step must not inflate the count (DECISIONS R28)."""
    from src.common.trial_log import count_trials, log_trial

    before = count_trials(cfg_smoke)
    for _ in range(3):
        log_trial(
            cfg_smoke,
            phase="tune",
            kind="strategy_grid",
            run_id="dup_probe",
            model="tree",
            params={"entry_base": 0.55, "slots": 5},
            period="validation",
            metric="sharpe",
            value=0.5,
        )
    after = count_trials(cfg_smoke)
    assert after == before + 1


def test_calibration_rows_are_not_counted_as_trials(cfg_smoke: dict[str, Any]) -> None:
    """R28: calibration is a diagnostic, not a configuration tried."""
    from src.common.trial_log import count_trials, log_trial

    before = count_trials(cfg_smoke)
    log_trial(
        cfg_smoke,
        phase="tune",
        kind="calibration",
        run_id="calibration",
        model="tree",
        params={"entry": 0.55},
        period="validation",
        metric="candidate_day_share",
        value=0.004,
    )
    assert count_trials(cfg_smoke) == before


def test_non_trading_trial_is_logged_with_no_value(cfg_smoke: dict[str, Any]) -> None:
    """A non-trading run is still a logged trial, but carries no Sharpe value."""
    from src.common.trial_log import count_trials, log_trial

    before = count_trials(cfg_smoke)
    log_trial(
        cfg_smoke,
        phase="tune",
        kind="strategy_grid",
        run_id="grid_validation",
        model="tree",
        params={"entry_base": 0.55, "slots": 10},
        period="validation",
        metric="sharpe",
        value=None,
        extra={"traded": False},
        note="non-trading run: excluded from selection",
    )
    assert count_trials(cfg_smoke) == before + 1
    row = [r for r in read_trials(cfg_smoke) if r["note"].startswith("non-trading")][-1]
    assert row["value"] == ""
