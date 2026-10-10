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
    """The full strategy: classifier plus regime rules, with stop-loss and band.

    The provider carries a `diagnostics` attribute updated on every call so the engine
    can report the share of entries decided by the symbol tie-break (`DECISIONS` R34).
    """
    strategy = cfg["strategy"]
    rules = _regime_rules(cfg)
    slots = int(strategy["slots"])
    exit_prob = float(strategy["exit_prob"])
    diagnostics = {"new_entries": 0, "tie_broken_entries": 0}

    def provider(context: DayContext) -> pd.Series | None:
        diagnostics.update({"new_entries": 0, "tie_broken_entries": 0})
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
            diagnostics=diagnostics,
        )

    provider.diagnostics = diagnostics  # type: ignore[attr-defined]
    del kw
    return provider


def make_b1(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """B1, buy and hold the index: weight 1.0 on `data.index_symbol` at the first
    decision, then None for the rest of the period (`spec 06` section 6).

    If the index has no bar on the first decision day, the run raises rather than
    silently substituting an equity: a "buy and hold" baseline that quietly held a
    stock would not be the benchmark the report claims.
    """
    index_symbol = str(cfg["data"]["index_symbol"])
    state = {"done": False}

    def provider(context: DayContext) -> pd.Series | None:
        if state["done"] or context.signals.empty:
            return None
        if index_symbol not in context.signals.index:
            raise ValueError(
                f"B1 needs a bar for {index_symbol} on the first decision day; "
                f"it is not in the tradable universe"
            )
        state["done"] = True
        return pd.Series(1.0, index=[index_symbol], dtype=float)

    del kw
    return provider


def make_b2(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """B2, equal-weight stocks rebalanced monthly (`spec 06` section 6).

    Weight is `1 / N` where `N` is the number of stocks with a bar that day, not the
    strategy's slot count: the baseline has to be able to hold the whole universe.
    """
    index_symbol = str(cfg["data"]["index_symbol"])
    state = {"done": False}

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty:
            return None
        if not state["done"]:
            state["done"] = True
        elif not context.is_month_end:
            return None  # hold between month ends
        available = sorted(s for s in context.signals.index if s != index_symbol)
        if not available:
            return None
        return pd.Series(1.0 / len(available), index=available, dtype=float)

    del kw
    return provider


def make_b3(cfg: Mapping[str, Any], **kw: Any) -> WeightProvider:
    """B3, the trend rule with no ML: hold every stock with `sma_20 > sma_50`,
    weight `1 / N_universe` each and rest in cash, recomputed daily
    (`spec 06` section 6)."""
    del cfg, kw

    def provider(context: DayContext) -> pd.Series | None:
        if context.signals.empty or context.indicators.empty:
            return None
        for column in ("sma_20", "sma_50"):
            if column not in context.indicators.columns:
                return None
        symbols = sorted(set(context.signals.index) & set(context.indicators.index))
        if not symbols:
            return None
        fast = context.indicators["sma_20"].reindex(symbols)
        slow = context.indicators["sma_50"].reindex(symbols)
        # A NaN SMA (still warming up) disqualifies the stock: no signal is not a hold.
        trending = sorted(
            symbol
            for symbol in symbols
            if pd.notna(fast[symbol]) and pd.notna(slow[symbol]) and fast[symbol] > slow[symbol]
        )
        if not trending:
            return None
        return pd.Series(1.0 / len(symbols), index=trending, dtype=float)

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
