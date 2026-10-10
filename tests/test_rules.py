"""Tests for the trading rules (task T5.1, spec 05, TESTS section 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.trading.rules import build_regime_rules, is_stopped_out, target_weights


def cfg() -> dict[str, Any]:
    """The base config with the strategy keys pinned to the spec's starting settings.

    `paths.frozen_config` is applied as a last overlay on the base, so the live
    `entry_base` and `slots` change every time tuning runs. These tests assert the
    rule *logic* against fixed values, so the overlay is stripped here rather than
    letting every tuning run silently rewrite the expected numbers.
    """
    from src.common.config import load_config

    loaded = load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))
    return {
        **loaded,
        "strategy": {
            **loaded["strategy"],
            "selected_model": None,
            "entry_base": 0.55,
            "slots": 10,
        },
    }


def signals(**probs: float) -> pd.DataFrame:
    """Build a day_signals frame indexed by symbol with a prob_up column."""
    return pd.DataFrame({"prob_up": pd.Series(probs)})


RULES = build_regime_rules(cfg())


def test_rules_from_config() -> None:
    """Entries exactly 0.55, 0.60, 0.65 with no float noise; BEAR entry is None."""
    assert RULES["BULL"]["entry"] == 0.55
    assert RULES["SIDEWAYS"]["entry"] == 0.60
    assert RULES["HIGH_VOL"]["entry"] == 0.65
    assert RULES["BEAR"]["entry"] is None
    assert RULES["BEAR"]["exposure"] == 0.0
    assert RULES["BULL"]["exposure"] == 1.0


def test_bear_is_empty() -> None:
    out = target_weights(signals(A="0.90", B="0.80"), "BEAR", {"A", "B"}, RULES, slots=10)
    assert len(out) == 0


def test_runbook_worked_example() -> None:
    """`TESTS` section 6: keeps H1..H7, drops H8, adds N1..N3, 10 names at 0.10."""
    probs = {
        "H1": 0.52,
        "H2": 0.53,
        "H3": 0.54,
        "H4": 0.55,
        "H5": 0.56,
        "H6": 0.57,
        "H7": 0.60,
        "H8": 0.47,
        "N1": 0.70,
        "N2": 0.66,
        "N3": 0.61,
        "N4": 0.56,
        "N5": 0.54,
    }
    held = {f"H{i}" for i in range(1, 9)}
    out = target_weights(signals(**probs), "BULL", held, RULES, slots=10)

    assert len(out) == 10
    assert "H8" not in out.index
    for i in range(1, 8):
        assert f"H{i}" in out.index
    assert {"N1", "N2", "N3"}.issubset(set(out.index))
    assert "N4" not in out.index, "no free slot left"
    assert "N5" not in out.index, "0.54 is below the BULL entry of 0.55"
    assert (out.round(6) == 0.10).all()


def test_hysteresis() -> None:
    """A held stock at 0.52 is kept; an unheld one at 0.52 is not entered."""
    probs = {"HELD": 0.52, "NEW": 0.52}
    out = target_weights(signals(**probs), "BULL", {"HELD"}, RULES, slots=10)
    assert "HELD" in out.index
    assert "NEW" not in out.index


def test_sideways_and_high_vol_sizing() -> None:
    probs = {f"S{i}": 0.90 for i in range(10)}
    sideways = target_weights(signals(**probs), "SIDEWAYS", set(), RULES, slots=10)
    high_vol = target_weights(signals(**probs), "HIGH_VOL", set(), RULES, slots=10)

    assert (sideways.round(6) == 0.06).all()
    assert (high_vol.round(6) == 0.03).all()


def test_entry_thresholds_differ_by_regime() -> None:
    """0.61 qualifies in SIDEWAYS (entry 0.60) but not in HIGH_VOL (entry 0.65)."""
    probs = {"A": 0.61}
    assert len(target_weights(signals(**probs), "SIDEWAYS", set(), RULES, slots=10)) == 1
    assert len(target_weights(signals(**probs), "HIGH_VOL", set(), RULES, slots=10)) == 0


def test_force_exit() -> None:
    """A held stock at 0.90 in force_exit is neither kept nor re-entered."""
    probs = {"STOP": 0.90}
    out = target_weights(
        signals(**probs), "BULL", {"STOP"}, RULES, slots=10, force_exit=frozenset({"STOP"})
    )
    assert "STOP" not in out.index


def test_no_entry() -> None:
    """A candidate in no_entry is skipped; a held stock in no_entry stays."""
    probs = {"BLOCKED": 0.90, "KEPT": 0.52}
    out = target_weights(
        signals(**probs), "BULL", {"KEPT"}, RULES, slots=10, no_entry=frozenset({"BLOCKED", "KEPT"})
    )
    assert "BLOCKED" not in out.index
    assert "KEPT" in out.index


def test_determinism_and_ties() -> None:
    """Equal probabilities with shuffled input give the same result, ties A to Z."""
    probs = {"ZZ": 0.70, "AA": 0.70, "MM": 0.70, "BB": 0.70}
    first = target_weights(signals(**probs), "BULL", set(), RULES, slots=2)
    shuffled = signals(**{k: probs[k] for k in reversed(list(probs))})
    second = target_weights(shuffled, "BULL", set(), RULES, slots=2)

    assert list(first.index) == ["AA", "BB"]
    assert list(second.index) == list(first.index)


def test_nan_probs() -> None:
    """NaN prob_up fails the exit test for a held stock and never enters an unheld one."""
    frame = pd.DataFrame({"prob_up": pd.Series({"HELD": float("nan"), "NEW": float("nan")})})
    out = target_weights(frame, "BULL", {"HELD", "NEW"}, RULES, slots=10)
    assert len(out) == 0


def test_slots_are_respected() -> None:
    probs = {f"S{i:02d}": 0.9 - i / 100 for i in range(30)}
    out = target_weights(signals(**probs), "BULL", set(), RULES, slots=10)
    assert len(out) == 10


def test_held_names_keep_their_slot() -> None:
    """A held name above exit_prob occupies a slot even below the entry threshold."""
    probs = {f"H{i}": 0.51 for i in range(9)}
    probs.update({f"N{i}": 0.99 for i in range(5)})
    out = target_weights(signals(**probs), "BULL", {f"H{i}" for i in range(9)}, RULES, slots=10)
    assert len(out) == 10
    assert sum(1 for s in out.index if s.startswith("H")) == 9
    assert sum(1 for s in out.index if s.startswith("N")) == 1


def test_regime_transition_trims_weights() -> None:
    """BULL to SIDEWAYS lowers every target weight; the band does the trimming."""
    probs = {f"S{i}": 0.80 for i in range(10)}
    held = set(probs)
    bull = target_weights(signals(**probs), "BULL", held, RULES, slots=10)
    sideways = target_weights(signals(**probs), "SIDEWAYS", held, RULES, slots=10)

    assert set(bull.index) == set(sideways.index), "nothing is force-sold (DECISIONS R6)"
    assert (sideways < bull).all()


@pytest.mark.parametrize(
    ("close", "avg", "expected"),
    [(93.0, 100.0, True), (93.01, 100.0, False), (80.0, 100.0, True)],
)
def test_stop_loss(close: float, avg: float, expected: bool) -> None:
    assert is_stopped_out(close, avg, 0.07) is expected


def test_target_weights_is_pure() -> None:
    """Calling twice with the same inputs gives the same result; inputs untouched."""
    probs = {"A": 0.9, "B": 0.8}
    frame = signals(**probs)
    held = {"A"}
    first = target_weights(frame, "BULL", held, RULES, slots=10)
    second = target_weights(frame, "BULL", held, RULES, slots=10)
    pd.testing.assert_series_equal(first, second)
    assert frame["prob_up"].tolist() == [0.9, 0.8]
