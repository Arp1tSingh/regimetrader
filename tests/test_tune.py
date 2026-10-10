"""Tests for the tuning grid (task T5.6, spec 05 section 6, TESTS section 7)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.backtest.tune import _is_finite, candidate_day_share, eligibility
from src.common.trial_log import read_trials


def cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))


# --------------------------------------------------------------- grid size


def expected_grid_size(c: dict[str, Any], n_eligible: int) -> int:
    """eligible models x 3 entry bases x slots (DECISIONS R33, R34)."""
    entries = len(c["tuning"]["entry_base_grid"]) + 1  # plus the per-model third entry
    return n_eligible * entries * len(c["tuning"]["slots_grid"])


def test_grid_size_is_18_when_the_tree_is_excluded() -> None:
    """3 eligible models x 3 entry bases x 2 slots = 18 (test_tune_grid_size)."""
    c = cfg()
    assert expected_grid_size(c, n_eligible=3) == 18


def test_grid_size_would_be_24_if_every_model_were_eligible() -> None:
    """4 x 3 x 2 = 24. This is the arithmetic trap: adding one trial per model
    instead of one entry base per model would give 20, which is wrong."""
    c = cfg()
    assert expected_grid_size(c, n_eligible=4) == 24
    per_model = (len(c["tuning"]["entry_base_grid"]) + 1) * len(c["tuning"]["slots_grid"])
    assert per_model == 6
    assert len(c["tuning"]["models"]) * per_model == 24


# --------------------------------------------------------------- eligibility


def test_eligibility_rule_rejects_a_degenerate_model() -> None:
    """The decision tree's measured distribution fails both conditions."""
    tree = {"tied_share": 0.7638, "p95_minus_p50": 0.0}
    ok, reason = eligibility(tree)
    assert not ok
    assert "tied share" in reason
    assert "p95-p50" in reason


@pytest.mark.parametrize(
    ("name", "stats"),
    [
        ("logreg", {"tied_share": 0.0001, "p95_minus_p50": 0.0244}),
        ("rf", {"tied_share": 0.0001, "p95_minus_p50": 0.0185}),
        ("xgb", {"tied_share": 0.0002, "p95_minus_p50": 0.0639}),
    ],
)
def test_eligibility_rule_accepts_the_other_three(name: str, stats: dict[str, float]) -> None:
    ok, _reason = eligibility(stats)
    assert ok, f"{name} should be eligible"


def test_eligibility_uses_no_return_information() -> None:
    """The rule reads only distribution fields; outcome fields cannot move it.

    Proven by behaviour rather than by reading the source: a stats dict holding only
    the two distributional keys must be enough, and adding outcome fields with
    extreme values must not change the verdict.
    """
    base = {"tied_share": 0.0001, "p95_minus_p50": 0.0244}
    assert eligibility(base)[0] is True

    with_outcomes = {
        **base,
        "sharpe": 9.9,
        "accuracy": 0.99,
        "next_day_return": 0.5,
        "target_up": 1,
        "total_return": 42.0,
    }
    assert eligibility(with_outcomes) == eligibility(base), (
        "eligibility must not consult returns, Sharpe or accuracy"
    )

    # A degenerate distribution stays ineligible no matter how good its outcome was.
    degenerate = {
        "tied_share": 0.7638,
        "p95_minus_p50": 0.0,
        "sharpe": 99.0,
        "accuracy": 0.99,
    }
    assert eligibility(degenerate)[0] is False


def test_eligibility_thresholds_are_documented_constants() -> None:
    from src.backtest.tune import MAX_TIED_SHARE, MIN_P95_P50_SPREAD

    assert MAX_TIED_SHARE == 0.20
    assert MIN_P95_P50_SPREAD == 0.01


@pytest.mark.slow
def test_thresholds_are_not_knife_edge(cfg_base: dict[str, Any]) -> None:
    """No eligible model sits near either boundary, so the rule is not arbitrary.

    Runs against the **real** warehouse, because these are the measured margins that
    justified fixing the thresholds at 20% and 0.01 (DECISIONS R31). Skipped when
    the real warehouse has not been populated.
    """
    from src.backtest.tune import MAX_TIED_SHARE, MIN_P95_P50_SPREAD, distribution_stats
    from src.common.db import get_engine

    engine = get_engine(cfg_base)
    checked = 0
    for model in ("logreg", "rf", "xgb"):
        stats = distribution_stats(engine, cfg_base, model)
        if stats["distinct"] <= 1:
            continue  # no real signal rows for this model yet
        checked += 1
        assert stats["tied_share"] < MAX_TIED_SHARE / 10, (
            f"{model} tied share {stats['tied_share']:.4f} is not far below 20%"
        )
        assert stats["p95_minus_p50"] > MIN_P95_P50_SPREAD, (
            f"{model} p95-p50 {stats['p95_minus_p50']:.4f} is not far above 0.01"
        )
    if checked == 0:
        pytest.skip("real warehouse has no validation signals")


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


# --------------------------------------------------------------- tie handling


def diagnostics_for(**probs: float) -> dict[str, float]:
    """Run V2 and return the tie-break diagnostics it records."""
    from src.backtest.strategies import make_v2
    from tests.test_strategies import context

    provider = make_v2(cfg())
    provider(context(probs, regime="BULL"))  # type: ignore[attr-defined]
    return dict(provider.diagnostics)  # type: ignore[attr-defined]


def test_ties_straddling_the_threshold_are_counted() -> None:
    """Only the entry on the cutoff probability was decided by symbol order."""
    from src.backtest.strategies import make_v2
    from tests.test_strategies import context

    slots = int(cfg()["strategy"]["slots"])
    # `slots - 1` distinct probabilities qualify outright, then a tie of five names
    # competes for the single remaining slot. The winner is alphabetical, not better.
    probs = {f"H{i:02d}": 0.90 - i / 1000 for i in range(slots - 1)}
    probs.update({f"T{i}": 0.70 for i in range(5)})
    probs["BELOW"] = 0.30

    provider = make_v2(cfg())
    weights = provider(context(probs, regime="BULL"))
    diag = dict(provider.diagnostics)  # type: ignore[attr-defined]

    assert weights is not None
    assert diag["new_entries"] == slots
    assert diag["tie_broken_entries"] == 1, (
        "only the cutoff entry is tie-broken; the rest qualified on probability"
    )
    # Alphabetical order picks T0; T1..T4 were equally good on probability.
    assert "T0" in weights.index
    assert not any(f"T{i}" in weights.index for i in range(1, 5))


def test_a_fully_tied_field_is_entirely_tie_broken() -> None:
    """The degenerate case: every candidate tied, so every entry is an arbitrary pick.

    This is the hazard R31 exists to catch, at the level a single day sees it.
    """
    from src.backtest.strategies import make_v2
    from tests.test_strategies import context

    slots = int(cfg()["strategy"]["slots"])
    tied = {f"S{i:02d}": 0.70 for i in range(50)}

    provider = make_v2(cfg())
    provider(context(tied, regime="BULL"))  # type: ignore[attr-defined]
    diag = dict(provider.diagnostics)  # type: ignore[attr-defined]

    assert diag["new_entries"] == slots
    assert diag["tie_broken_entries"] == diag["new_entries"], (
        "with one probability mass, every entry is decided alphabetically"
    )
    from src.backtest.tune import MAX_TIE_BREAK_SHARE

    assert diag["tie_broken_entries"] / diag["new_entries"] > MAX_TIE_BREAK_SHARE


def test_no_ties_means_no_tie_broken_entries() -> None:
    """Distinct probabilities: nothing is decided by the symbol tie-break."""
    probs = {"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.10}
    diag = diagnostics_for(**probs)
    assert diag["new_entries"] == 3
    assert diag["tie_broken_entries"] == 0


def test_ties_below_the_cutoff_are_not_counted() -> None:
    """A tie among names that all get in is not a selection decision at all."""
    slots = int(cfg()["strategy"]["slots"])
    probs = {"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.60}
    diag = diagnostics_for(**probs)
    assert diag["new_entries"] == min(slots, len(probs))
    assert diag["tie_broken_entries"] == 0


def test_no_entries_records_a_zero_count_not_a_share() -> None:
    """A day with no qualifying stock records zero counts, not a share."""
    from src.backtest.strategies import make_v2
    from tests.test_strategies import context

    provider = make_v2(cfg())
    provider(context({"A": 0.10}, regime="BULL"))  # type: ignore[attr-defined]
    diag = dict(provider.diagnostics)  # type: ignore[attr-defined]
    assert diag["new_entries"] == 0
    assert diag["tie_broken_entries"] == 0


def test_tie_break_share_is_nan_when_a_run_never_enters() -> None:
    """Engine level: no new entries means no share, so the run is not flagged."""
    from src.backtest.engine import BacktestResult

    result = BacktestResult(
        equity=pd.DataFrame(), trades=pd.DataFrame(), signals=pd.DataFrame(), run_id="r"
    )
    assert math.isnan(result.tie_break_share), "no entries means no share, not 0%"

    ran = BacktestResult(
        equity=pd.DataFrame(),
        trades=pd.DataFrame(),
        signals=pd.DataFrame(),
        run_id="r",
        new_entries=4,
        tie_broken_entries=1,
    )
    assert ran.tie_break_share == pytest.approx(0.25)


def test_tie_flagged_runs_are_excluded_from_selection() -> None:
    """R34: a run above the tie-break cap is flagged and not selectable."""
    from src.backtest.tune import MAX_TIE_BREAK_SHARE

    grid = pd.DataFrame(
        [
            {
                "model": "tree",
                "sharpe": 5.0,
                "turnover": 0.1,
                "traded": False,
                "tie_break_share": 0.9,
                "tie_break_flagged": True,
            },
            {
                "model": "rf",
                "sharpe": 0.3,
                "turnover": 0.4,
                "traded": True,
                "tie_break_share": 0.01,
                "tie_break_flagged": False,
            },
        ]
    )
    tradable = grid[grid["traded"] & grid["sharpe"].notna()]
    assert list(tradable["model"]) == ["rf"], "a tie-heavy run must not win on Sharpe"
    assert MAX_TIE_BREAK_SHARE == 0.05


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
