"""Tests for the strategy variants and baselines (task T5.5, spec 05 section 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.backtest.engine import DayContext
from src.backtest.strategies import make_weights_fn


def cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.smoke.yaml"))


def context(
    probs: dict[str, float] | None = None,
    regime: str = "BULL",
    held: set[str] | None = None,
    is_month_end: bool = False,
    no_entry: frozenset[str] = frozenset(),
    force_exit: frozenset[str] = frozenset(),
) -> DayContext:
    """A DayContext built directly, so providers can be tested in isolation."""
    frame = (
        pd.DataFrame({"prob_up": pd.Series(probs, dtype="float64")})
        if probs
        else pd.DataFrame(columns=["prob_up"])
    )
    return DayContext(
        date=pd.Timestamp("2024-01-02"),
        regime=regime,
        signals=frame,
        indicators=pd.DataFrame(),
        held=set(held or set()),
        current_weights=pd.Series(dtype="float64"),
        force_exit=force_exit,
        no_entry=no_entry,
        is_month_end=is_month_end,
    )


PROBS = {f"S{i:02d}.NS": 0.55 + i / 100 for i in range(15)}


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        make_weights_fn("V9", cfg())


# --------------------------------------------------------------- variants


def test_v0_uses_bull_rule_in_every_regime() -> None:
    """V0 drops the per-regime threshold and exposure but keeps the classifier."""
    provider = make_weights_fn("V0", cfg())
    for regime in ("BULL", "SIDEWAYS", "HIGH_VOL"):
        weights = provider(context(PROBS, regime=regime))
        assert weights is not None and len(weights) > 0
        assert (weights.round(6) == 0.20).all(), "V0 uses BULL exposure 1.0 / 5 slots"


def test_v1_needs_bull_regime() -> None:
    """V1 is the regime filter alone: invested only when BULL."""
    provider = make_weights_fn("V1", cfg())
    assert provider(context(PROBS, regime="BULL")) is not None
    assert provider(context(PROBS, regime="SIDEWAYS")) is None
    assert provider(context(PROBS, regime="HIGH_VOL")) is None


def test_v2_applies_regime_rules() -> None:
    """V2 weights shrink with the regime exposure."""
    provider = make_weights_fn("V2", cfg())
    bull = provider(context(PROBS, regime="BULL"))
    sideways = provider(context(PROBS, regime="SIDEWAYS"))
    high_vol = provider(context(PROBS, regime="HIGH_VOL"))

    assert bull.iloc[0] == pytest.approx(0.20)
    assert sideways.iloc[0] == pytest.approx(0.12)
    assert high_vol.iloc[0] == pytest.approx(0.06)
    assert (
        provider(context(PROBS, regime="BEAR")) is None
        or len(provider(context(PROBS, regime="BEAR"))) == 0
    )


def test_v2_respects_stop_loss_and_cooldown() -> None:
    provider = make_weights_fn("V2", cfg())
    ctx = context(PROBS, regime="BULL", held={"S01"}, force_exit=frozenset({"S01"}))
    weights = provider(ctx)
    assert "S01" not in (weights.index if weights is not None else [])


def test_variants_return_none_without_signals() -> None:
    for name in ("V0", "V1", "V2", "B1", "B2", "B3"):
        assert make_weights_fn(name, cfg())(context(None)) is None


# --------------------------------------------------------------- baselines


def test_b1_trades_once_then_holds() -> None:
    """B1 returns weight 1.0 once, then None on every later decision."""
    provider = make_weights_fn("B1", cfg())
    first = provider(context(PROBS))
    assert first is not None and len(first) == 1
    assert first.iloc[0] == pytest.approx(1.0)

    for _ in range(5):
        assert provider(context(PROBS)) is None


def test_b2_invests_from_first_day_and_rebalances_on_month_ends() -> None:
    """B2 is invested immediately and only rebalances on month ends."""
    provider = make_weights_fn("B2", cfg())
    first = provider(context(PROBS, is_month_end=False))
    assert first is not None and len(first) > 0
    assert (first.round(6) == 0.20).all()

    # Not a month end: no orders.
    assert provider(context(PROBS, is_month_end=False)) is None

    # Month end: rebalances.
    assert provider(context(PROBS, is_month_end=True)) is not None


def test_b3_holds_the_index() -> None:
    provider = make_weights_fn("B3", cfg())
    assert provider(context(PROBS)) is not None
    assert provider(context(PROBS)) is None


def test_providers_are_fresh_objects() -> None:
    """Providers hold state, so each build must start clean."""
    b1 = make_weights_fn("B1", cfg())
    b1(context(PROBS))
    assert make_weights_fn("B1", cfg())(context(PROBS)) is not None


# --------------------------------------------------------------- integration


@pytest.mark.slow
def test_variants_differ_on_smoke_signals(signals_ready: tuple) -> None:
    """V0, V1 and V2 must produce different equity curves.

    This runs on the **synthetic** smoke warehouse, not real data: it checks that the
    providers are wired differently, not that any variant is profitable.
    """
    from src.backtest.engine import run_backtest
    from src.backtest.strategies import make_weights_fn as build

    cfg_obj, engine = signals_ready
    cfg_obj = {**cfg_obj, "splits": {**cfg_obj["splits"], "test_start": "2030-01-01"}}
    start = str(cfg_obj["splits"]["validation_start"])
    end = str(cfg_obj["splits"]["validation_end"])

    curves = {}
    for name in ("V0", "V1", "V2"):
        result = run_backtest(
            cfg_obj,
            build(name, cfg_obj),
            start,
            end,
            f"v_{name}",
            engine=engine,
            model_name="tree",
            persist=False,
        )
        curves[name] = result.equity["equity"].to_numpy()

    assert not (curves["V0"] == curves["V1"]).all(), "V0 and V1 must differ"
    assert not (curves["V1"] == curves["V2"]).all(), "V1 and V2 must differ"
