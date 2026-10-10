"""Portfolio state and order execution (`spec 06` section 3).

Pure functions over a `Portfolio`; no warehouse access. The backtest engine and the
paper job both use these, which is what keeps one strategy code path (`AGENTS.md` rule 9).

Cost model, from `06` section 3:

    fill_price = open * (1 + slippage) for BUY, open * (1 - slippage) for SELL
    cost       = cost_pct * quantity * fill_price
    BUY  cash out = trade_value + cost
    SELL cash in  = trade_value - cost

Positions use the **average-cost** method, not FIFO (`DECISIONS` R25): `avg_fill` is
the quantity-weighted mean of the current position's buy fills, costs excluded, and is
reset only when the position is fully sold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: Trade reasons, matching the `fact_trade.reason` CHECK constraint.
REASONS = ("ENTRY", "REBALANCE", "EXIT", "STOP_LOSS")


@dataclass
class Position:
    """One open position.

    Attributes:
        quantity: Shares held.
        avg_fill: Quantity-weighted mean buy fill price, costs excluded.
        avg_cost: Quantity-weighted mean cash-out per share, costs included.
        entry_date: First bar the position was opened.
    """

    quantity: int | float
    avg_fill: float
    avg_cost: float
    entry_date: Any = None


@dataclass
class Order:
    """A planned order, decided at one close and filled at the next open."""

    symbol: str
    side: str
    quantity: int | float
    reason: str


@dataclass
class Trade:
    """A filled order."""

    symbol: str
    side: str
    quantity: int | float
    fill_price: float
    cost: float
    reason: str
    realized_pnl: float | None = None


@dataclass
class Portfolio:
    """Cash plus open positions."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


def fill_price_for(open_price: float, side: str, slippage: float) -> float:
    """Fill price with adverse slippage."""
    if side == "BUY":
        return float(open_price) * (1.0 + float(slippage))
    return float(open_price) * (1.0 - float(slippage))


def fill_order(
    order: Order, open_price: float, cfg: Mapping[str, Any], costs_enabled: bool = True
) -> Trade:
    """Fill one order at the next open.

    Args:
        order: Planned order.
        open_price: The execution day's opening price.
        cfg: Loaded config; `backtest.slippage` and `backtest.cost`.
        costs_enabled: When False, slippage and cost are both zero. Used by the
            zero-cost identity test in `TESTS` section 7.

    Returns:
        A filled `Trade` with `realized_pnl` still to be set by `apply_trade`.
    """
    slippage = float(cfg["backtest"]["slippage"]) if costs_enabled else 0.0
    cost_pct = float(cfg["backtest"]["cost"]) if costs_enabled else 0.0

    price = fill_price_for(open_price, order.side, slippage)
    trade_value = order.quantity * price
    cost = cost_pct * trade_value
    return Trade(
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        fill_price=price,
        cost=cost,
        reason=order.reason,
    )


def apply_trade(portfolio: Portfolio, trade: Trade) -> None:
    """Apply a filled trade to the portfolio, mutating it.

    Realised P&L on a sell is `(fill - avg_cost_per_share) * quantity - sell_cost`,
    where `avg_cost_per_share` already includes the buy-side cost (`DECISIONS` R25).

    Args:
        portfolio: Mutated in place.
        trade: A filled trade.
    """
    if trade.side == "BUY":
        value = trade.quantity * trade.fill_price
        cash_out = value + trade.cost
        existing = portfolio.positions.get(trade.symbol)
        if existing is None:
            portfolio.positions[trade.symbol] = Position(
                quantity=trade.quantity,
                avg_fill=trade.fill_price,
                avg_cost=cash_out / trade.quantity,
            )
        else:
            total = existing.quantity + trade.quantity
            existing.avg_fill = (
                existing.avg_fill * existing.quantity + trade.fill_price * trade.quantity
            ) / total
            existing.avg_cost = (existing.avg_cost * existing.quantity + cash_out) / total
            existing.quantity = total
        portfolio.cash -= cash_out
        return

    # SELL
    existing = portfolio.positions.get(trade.symbol)
    value = trade.quantity * trade.fill_price
    cash_in = value - trade.cost
    portfolio.cash += cash_in

    if existing is not None:
        realized = (trade.fill_price - existing.avg_cost) * trade.quantity - trade.cost
        trade.realized_pnl = float(realized)
        remaining = existing.quantity - trade.quantity
        if remaining <= 0:
            # Fully sold: reset, so a later entry starts a fresh average.
            del portfolio.positions[trade.symbol]
        else:
            existing.quantity = remaining
    else:
        trade.realized_pnl = 0.0


def mark_to_market(portfolio: Portfolio, closes: Mapping[str, float]) -> tuple[float, float]:
    """Value the portfolio at a close.

    Args:
        portfolio: Current state.
        closes: Close price per symbol for today. A held symbol missing from `closes`
            is valued at its last known close, which the engine maintains.

    Returns:
        `(positions_value, equity)`.
    """
    value = 0.0
    for symbol, position in portfolio.positions.items():
        price = closes.get(symbol)
        if price is None:
            price = position.avg_fill
        value += position.quantity * float(price)
    return float(value), float(portfolio.cash + value)


def _current_weight(
    symbol: str, portfolio: Portfolio, closes: Mapping[str, float], equity: float
) -> float:
    position = portfolio.positions.get(symbol)
    if position is None or equity <= 0:
        return 0.0
    return (position.quantity * float(closes.get(symbol, position.avg_fill))) / equity


def plan_orders(
    targets: Any,
    portfolio: Portfolio,
    equity_at_decision: float,
    opens: Mapping[str, float],
    prob_up: Mapping[str, float],
    cfg: Mapping[str, Any],
    band: float | None = None,
    fractional_shares: bool = False,
) -> list[Order]:
    """Plan the orders that turn current positions into the target weights.

    Sells are planned first so they free cash, then buys in descending `prob_up` with
    ties broken by symbol. A buy is capped by available cash so cash never goes
    negative. The rebalance band stops small adjustments on existing positions; new
    positions and exits always trade.

    Args:
        targets: Target weights by symbol, or None for "place no orders".
        portfolio: State at the decision close.
        equity_at_decision: Equity at the decision close.
        opens: Next day's open per symbol. A symbol with no bar is skipped, and the
            next decision recomputes it.
        prob_up: Probability per symbol, used only to order the buys.
        cfg: Loaded config.
        band: Rebalance band; defaults to `strategy.rebalance_band`.
        fractional_shares: Keep the fractional quantity instead of flooring.

    Returns:
        Planned orders, sells first.
    """
    if targets is None:
        return []

    band_value = float(cfg["strategy"]["rebalance_band"] if band is None else band)
    if len(targets):
        targets = targets[targets > 0]

    slippage = float(cfg["backtest"]["slippage"])
    cost_pct = float(cfg["backtest"]["cost"])

    current = set(portfolio.positions)
    desired = set(targets.index)

    sells: list[Order] = []
    buys: list[Order] = []

    # Positions leaving the target set are always sold.
    for symbol in sorted(current - desired):
        if symbol not in opens:
            continue
        sells.append(Order(symbol, "SELL", portfolio.positions[symbol].quantity, "EXIT"))

    # Existing positions: only trade when the rebalance band is breached.
    for symbol in sorted(current & desired):
        if symbol not in opens:
            continue
        position = portfolio.positions[symbol]
        target_weight = float(targets[symbol])
        weight_now = _current_weight(symbol, position, equity_at_decision)
        if abs(target_weight - weight_now) <= band_value * target_weight:
            continue
        open_price = float(opens[symbol])
        target_qty = _quantity(target_weight, equity_at_decision, open_price, fractional_shares)
        delta = target_qty - position.quantity
        if delta == 0:
            continue
        if delta > 0:
            buys.append(Order(symbol, "BUY", delta, "REBALANCE"))
        else:
            sells.append(Order(symbol, "SELL", abs(delta), "REBALANCE"))

    # New positions, best probability first, symbol A to Z on ties.
    for symbol in sorted(desired - current, key=lambda s: (-float(prob_up.get(s, 0.0)), s)):
        if symbol not in opens:
            continue
        qty = _quantity(
            float(targets[symbol]), equity_at_decision, float(opens[symbol]), fractional_shares
        )
        if qty > 0:
            buys.append(Order(symbol, "BUY", qty, "ENTRY"))

    # Cash released by the sells funds the buys.
    projected_cash = portfolio.cash
    for order in sells:
        price = fill_price_for(float(opens[order.symbol]), "SELL", slippage)
        projected_cash += order.quantity * price * (1.0 - cost_pct)

    planned: list[Order] = list(sells)
    for order in buys:
        price = fill_price_for(float(opens[order.symbol]), "BUY", slippage)
        quantity = _affordable(
            quantity=order.quantity,
            price=price,
            cash=projected_cash,
            cost_pct=cost_pct,
            fractional=fractional_shares,
        )
        if quantity <= 0:
            continue  # cannot afford a single share: skip rather than go negative
        planned.append(Order(order.symbol, "BUY", quantity, order.reason))
        projected_cash -= quantity * price * (1.0 + cost_pct)

    return planned


def _current_weight(symbol: str, position: Position, equity: float) -> float:
    """Weight of one position, valued at its average fill.

    The engine holds the decision-close price, which is the last known close for a
    symbol; `avg_fill` is the stable stand-in used when sizing the band test.
    """
    if equity <= 0:
        return 0.0
    return (position.quantity * float(position.avg_fill)) / float(equity)


def _affordable(
    quantity: float, price: float, cash: float, cost_pct: float, fractional: bool
) -> float:
    """Largest affordable quantity whose outlay fits in `cash`.

    Whole shares step down by one; fractional shares step down by 1% at a time so the
    loop always terminates. Cash therefore never goes negative (`06` section 3 rule 3).
    """
    if price <= 0 or quantity <= 0:
        return 0.0
    step = max(quantity * 0.01, 1e-9) if fractional else 1.0
    while quantity > 0:
        if quantity * price * (1.0 + cost_pct) <= cash:
            return quantity
        quantity -= step
    return 0.0


def _quantity(
    target_weight: float, equity: float, open_price: float, fractional: bool
) -> int | float:
    """Target share count for a weight, floored to whole shares unless fractional."""
    if open_price <= 0:
        return 0
    raw = float(target_weight) * float(equity) / float(open_price)
    return raw if fractional else float(int(raw))
