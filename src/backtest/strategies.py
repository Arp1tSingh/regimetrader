"""Strategy variants and baselines as weight providers (`spec 05` section 7, `06` section 6).

Every variant and baseline is a `Callable[[DayContext], pd.Series | None]` consumed by
the same engine, so results are directly comparable. Returning `None` means "place no
orders at the next open" (`DECISIONS` R29).

Variants (ablations of the full system):
    V0 classifier only, BULL rule in every regime, always invested up to the slot limit
    V1 regime filter only, equal weight across every stock with a bar when BULL
    V2 the full strategy: classifier plus regime rules

Baselines:
    B1 buy and hold the index equivalent, once
    B2 equal weight, rebalanced on month ends
    B3 the index itself
    B4 the symbol-permutation null (`06` section 6)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

import pandas as pd

from src.backtest.engine import DayContext
from src.trading.rules import build_regime_rules, target_weights

#: Signature every provider matches.
WeightProvider = Callable[[DayContext], pd.Series | None]


def _regime_rules(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return build_regime_rules(cfg)


def make_v0(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """Classifier only: the BULL rule in every regime, always invested.

    Isolates how much the regime filter contributes, since V0 keeps the classifier
    but drops the per-regime thresholds and exposures.
    """
    strategy = cfg["strategy"]
    slots = int(strategy["slots"])
    exit_prob = float(strategy["exit_prob"])
    bull = _regime_rules(cfg)["BULL"]
    rules = {
        regime: {"entry": bull["entry"], "exposure": bull["exposure"]}
        for regime in ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR")
    }

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty:
            return None
        return target_weights(
            context.signals,
            "BULL",
            context.held,
            rules,
            slots=slots,
            exit_prob=exit_prob,
            force_exit=context.force_exit,
            no_entry=context.no_entry,
        )

    del kw
    return provider


def make_v1(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """Regime filter only: equal weight across every stock with a bar, when BULL.

    No classifier at all, so it measures what the regime detection alone earns.
    """
    slots = int(cfg["strategy"]["slots"])
    weight = 1.0 / slots

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty or context.regime != "BULL":
            return None
        held = set(context.held)
        available = [s for s in context.signals.index if s not in context.no_entry or s in held]
        if not available:
            return None
        return pd.Series(weight, index=sorted(available[:slots]), dtype=float)

    del kw
    return provider


def make_v2(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """The full strategy: classifier plus regime rules, with stop-loss and band."""
    strategy = cfg["strategy"]
    rules = _regime_rules(cfg)
    slots = int(strategy["slots"])
    exit_prob = float(strategy["exit_prob"])

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty:
            return None
        return target_weights(
            context.signals,
            context.regime,
            context.held,
            rules,
            slots=slots,
            exit_prob=exit_prob,
            force_exit=context.force_exit,
            no_entry=context.no_entry,
        )

    del kw
    return provider


def make_b1(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """Buy and hold: weight 1.0 on the first decision, then None (no more orders)."""
    state = {"done": False}

    def provider(context: DayContext) -> pd.Series | None:
        if state["done"] or context.signals.empty:
            return None
        state["done"] = True
        return pd.Series(1.0, index=sorted(context.signals.index)[:1], dtype=float)

    del cfg, kw
    return provider


def make_b2(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """Equal weight across the universe, invested from the first day and rebalanced
    on every month end (DECISIONS R29)."""
    slots = int(cfg["strategy"]["slots"])
    weight = 1.0 / slots
    started = {"done": False}

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty:
            return None
        if not started["done"]:
            started["done"] = True
        elif not context.is_month_end:
            return None  # hold between month ends
        available = sorted(context.signals.index)
        if not available:
            return None
        return pd.Series(weight, index=available[:slots], dtype=float)

    del cfg, kw
    return provider


def make_b3(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """The index itself, held for the whole period."""
    state = {"done": False}

    def provider(context: DayContext) -> pd.Series | None:
        if state["done"] or context.signals.empty:
            return None
        state["done"] = True
        return pd.Series(1.0, index=sorted(context.signals.index)[:1], dtype=float)

    del cfg, kw
    return provider


#: Registry used by `make_weights_fn`.
PROVIDERS: dict[str, Callable[..., WeightProvider]] = {
    "V0": make_v0,
    "V1": make_v1,
    "V2": make_v2,
    "B1": make_b1,
    "B2": make_b2,
    "B3": make_b3,
}


def make_weights_fn(name: str, cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """Build a weight provider by name.

    Args:
        name: `V0`, `V1`, `V2`, `B1`, `B2` or `B3`.
        cfg: Loaded config.
        kw: Forwarded to the provider.

    Returns:
        A fresh provider. Providers hold small amounts of state, so build a new one
        for each run rather than reusing one.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in PROVIDERS:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(PROVIDERS)}")
    return PROVIDERS[name](cfg, **kw)
