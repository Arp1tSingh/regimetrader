"""Regime rules and target weights (`spec 05` sections 2 to 5).

Pure functions with no I/O, so the strategy logic can be tested exhaustively without
a warehouse. The backtest and the paper job both call these, which is what keeps a
single strategy code path (`AGENTS.md` rule 9).
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

#: Regime with zero exposure: no entries, everything exits at the next open.
NO_EXPOSURE = 0.0


def build_regime_rules(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Entry threshold and exposure per regime.

    Entry threshold is `entry_base + entry_offsets[regime]`, rounded to 6 decimals.
    A regime with zero exposure gets `entry = None`: nothing may be entered.

    Args:
        cfg: Loaded config; uses `strategy.entry_base`, `strategy.entry_offsets` and
            `strategy.exposure`.

    Returns:
        `{regime: {"entry": float | None, "exposure": float}}`.
    """
    strategy = cfg["strategy"]
    rules: dict[str, dict[str, Any]] = {}
    for regime, exposure in strategy["exposure"].items():
        if float(exposure) == NO_EXPOSURE:
            rules[regime] = {"entry": None, "exposure": NO_EXPOSURE}
        else:
            rules[regime] = {
                "entry": round(
                    float(strategy["entry_base"])
                    + float(strategy["entry_offsets"].get(regime, 0.0)),
                    6,
                ),
                "exposure": float(exposure),
            }
    return rules


def target_weights(
    day_signals: pd.DataFrame,
    regime: str,
    held: set[str],
    rules: Mapping[str, Mapping[str, Any]],
    slots: int = 10,
    exit_prob: float = 0.50,
    force_exit: frozenset[str] = frozenset(),
    no_entry: frozenset[str] = frozenset(),
) -> pd.Series:
    """Target weight per symbol for the next open.

    Pure and deterministic. Held names above `exit_prob` are kept and trimmed to
    `exposure / slots`; free slots go to the highest `prob_up` at or above the entry
    threshold, ties broken by symbol A to Z. Nothing is force-sold for having the
    lowest probability (`DECISIONS` R6).

    Args:
        day_signals: Frame indexed by symbol with a `prob_up` column, as of today's close.
        regime: Smoothed regime label for today.
        held: Symbols currently held.
        rules: Output of `build_regime_rules`.
        slots: Maximum number of positions.
        exit_prob: Hold while `prob_up >= exit_prob`.
        force_exit: Stop-loss names: dropped now and cannot be re-entered.
        no_entry: Names that cannot be newly entered but may be kept if held.

    Returns:
        A Series of target weights indexed by symbol. Empty in a zero-exposure regime.
    """
    rule = rules[regime]
    if float(rule["exposure"]) == NO_EXPOSURE:
        return pd.Series(dtype=float)

    probs = day_signals["prob_up"]

    # NaN prob_up never qualifies for entry and fails the exit test.
    keep = sorted(
        symbol
        for symbol in held
        if symbol not in force_exit
        and symbol in probs.index
        and pd.notna(probs[symbol])
        and float(probs[symbol]) >= exit_prob
    )
    blocked = set(keep) | set(force_exit) | set(no_entry)

    entry = float(rule["entry"])
    eligible = probs[(probs >= entry) & ~probs.index.isin(sorted(blocked))]
    # Sort by probability, then by symbol, so ties resolve A to Z.
    eligible = eligible.sort_index().sort_values(ascending=False, kind="stable")

    free_slots = max(int(slots) - len(keep), 0)
    new = eligible.head(free_slots).index.tolist()

    chosen = keep + list(new)
    if not chosen:
        return pd.Series(dtype=float)
    return pd.Series(float(rule["exposure"]) / int(slots), index=chosen, dtype=float)


def entry_threshold(regime: str, rules: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Entry threshold for one regime, or None when it accepts no entries."""
    return rules[regime]["entry"]


def is_stopped_out(close: float, avg_fill_price: float, pct: float) -> bool:
    """Whether a close breaches the stop-loss.

    Args:
        close: Today's closing price.
        avg_fill_price: Quantity-weighted average fill of the open position.
        pct: Stop-loss fraction, 0.07 by default.

    Returns:
        True when `close <= avg_fill_price * (1 - pct)`.
    """
    return close <= avg_fill_price * (1.0 - float(pct))
