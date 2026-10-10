"""Tests for execution, costs and metrics (task T5.2-T5.3, TESTS section 7)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.backtest.metrics import (
    annual_volatility,
    cagr,
    calmar,
    daily_returns,
    max_drawdown,
    profit_factor,
    sharpe,
    sortino,
    summarize,
    turnover,
    win_rate,
)
from src.trading.execution import (
    Order,
    Portfolio,
    Position,
    Trade,
    apply_trade,
    fill_order,
    fill_price_for,
    mark_to_market,
    plan_orders,
)


def cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))


CONFIG = cfg()


# --------------------------------------------------------------- cost model


def test_cost_worked_example() -> None:
    """`TESTS` section 7: buy 100 at open 1,000 -> cash out 100,150.05."""
    order = Order("AAA.NS", "BUY", 100, "ENTRY")
    trade = fill_order(order, 1000.0, CONFIG)

    assert trade.fill_price == pytest.approx(1000.50)
    assert trade.quantity * trade.fill_price == pytest.approx(100050.00)
    assert trade.cost == pytest.approx(100.05)

    portfolio = Portfolio(cash=1_000_000.0)
    apply_trade(portfolio, trade)
    assert portfolio.cash == pytest.approx(1_000_000.0 - 100150.05)


def test_slippage_is_adverse_on_both_sides() -> None:
    assert fill_price_for(1000.0, "BUY", 0.0005) > 1000.0
    assert fill_price_for(1000.0, "SELL", 0.0005) < 1000.0


def test_costs_disabled_gives_clean_fill() -> None:
    trade = fill_order(Order("A", "BUY", 10, "ENTRY"), 100.0, CONFIG, costs_enabled=False)
    assert trade.fill_price == pytest.approx(100.0)
    assert trade.cost == pytest.approx(0.0)


def test_round_trip_cost_about_030_percent() -> None:
    """`06` section 3: a round trip costs about 0.30% of trade value.

    That is 0.10% cost per side (0.20%) plus 0.05% slippage per side (0.10%),
    measured against the raw notional.
    """
    quantity, notional_open = 100, 1000.0
    buy = fill_order(Order("A", "BUY", quantity, "ENTRY"), notional_open, CONFIG)
    sell = fill_order(Order("A", "SELL", quantity, "EXIT"), notional_open, CONFIG)

    notional = quantity * notional_open
    cost_drag = buy.cost + sell.cost
    # Slippage drag: bought 0.05% above and sold 0.05% below the open.
    slippage_drag = quantity * (buy.fill_price - notional_open) + quantity * (
        notional_open - sell.fill_price
    )
    assert (cost_drag + slippage_drag) / notional == pytest.approx(0.003, abs=1e-4)
    assert cost_drag / notional == pytest.approx(0.002, abs=1e-4)


def test_no_cost_zero_cost_identity() -> None:
    """With costs off, buying and selling the same price is exactly neutral."""
    portfolio = Portfolio(cash=100_000.0)
    apply_trade(portfolio, fill_order(Order("A", "BUY", 100, "ENTRY"), 10.0, CONFIG, False))
    apply_trade(portfolio, fill_order(Order("A", "SELL", 100, "EXIT"), 10.0, CONFIG, False))
    assert portfolio.cash == pytest.approx(100_000.0)
    assert portfolio.positions == {}


def test_realized_pnl_on_sell() -> None:
    portfolio = Portfolio(cash=0.0)
    buy = Trade("A", "BUY", 100, 10.0, 0.0, "ENTRY")
    apply_trade(portfolio, buy)
    sell = Trade("A", "SELL", 100, 12.0, 0.0, "EXIT")
    apply_trade(portfolio, sell)

    assert sell.realized_pnl == pytest.approx(200.0)
    assert portfolio.positions == {}, "a full sale resets the position"


def test_partial_sell_keeps_average_cost() -> None:
    """Average-cost, not FIFO: a partial sale must not change avg_cost."""
    portfolio = Portfolio(cash=0.0)
    apply_trade(portfolio, Trade("A", "BUY", 100, 10.0, 0.0, "ENTRY"))
    apply_trade(portfolio, Trade("A", "BUY", 100, 20.0, 0.0, "REBALANCE"))
    before = portfolio.positions["A"].avg_cost

    apply_trade(portfolio, Trade("A", "SELL", 50, 20.0, 0.0, "REBALANCE"))
    after = portfolio.positions["A"]

    assert after.quantity == 150
    assert after.avg_cost == pytest.approx(before)


def test_mark_to_market_uses_last_known_close() -> None:
    portfolio = Portfolio(cash=100.0, positions={"A": Position(10, 5.0, 5.0)})
    value, equity = mark_to_market(portfolio, {})  # no bar today
    assert value == pytest.approx(50.0)
    assert equity == pytest.approx(150.0)

    value, equity = mark_to_market(portfolio, {"A": 6.0})
    assert equity == pytest.approx(160.0)


# --------------------------------------------------------------- plan_orders


def _portfolio_with(weights: dict[str, float], equity: float, price: float = 100.0) -> Portfolio:
    """A portfolio holding `weights` of equity, each bought at `price`.

    Starts with the full equity as cash and spends it down, so cash never goes
    negative the way it would if the purchases were subtracted from zero.
    """
    portfolio = Portfolio(cash=float(equity))
    for symbol, weight in weights.items():
        quantity = int(weight * equity / price)
        portfolio.positions[symbol] = Position(quantity, price, price)
        portfolio.cash -= quantity * price
    return portfolio


def test_plan_orders_within_band_does_not_trade() -> None:
    """Target 0.10, current 0.12: diff 0.02 <= 0.25 * 0.10 = 0.025, so no trade."""
    equity = 100_000.0
    portfolio = _portfolio_with({"A": 0.12}, equity)
    targets = pd.Series({"A": 0.10})
    orders = plan_orders(targets, portfolio, equity, {"A": 100.0}, {"A": 0.9}, CONFIG)
    assert orders == []


def test_plan_orders_outside_band_trims() -> None:
    """Target 0.10, current 0.14: sell down to the target quantity."""
    equity = 100_000.0
    portfolio = _portfolio_with({"A": 0.14}, equity)
    targets = pd.Series({"A": 0.10})
    orders = plan_orders(targets, portfolio, equity, {"A": 100.0}, {"A": 0.9}, CONFIG)

    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert orders[0].reason == "REBALANCE"
    assert orders[0].quantity == pytest.approx(40.0)  # 140 - 100


def test_plan_orders_new_position_buys() -> None:
    equity = 100_000.0
    portfolio = Portfolio(cash=equity)
    targets = pd.Series({"A": 0.10})
    orders = plan_orders(targets, portfolio, equity, {"A": 100.0}, {"A": 0.9}, CONFIG)

    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].reason == "ENTRY"
    assert orders[0].quantity == pytest.approx(100.0)


def test_plan_orders_exit_sells_all() -> None:
    equity = 100_000.0
    portfolio = _portfolio_with({"A": 0.10}, equity)
    orders = plan_orders(pd.Series(dtype=float), portfolio, equity, {"A": 100.0}, {}, CONFIG)

    assert len(orders) == 1
    assert orders[0].side == "SELL"
    assert orders[0].reason == "EXIT"


def test_plan_orders_quantity_is_floored() -> None:
    equity = 100_000.0
    portfolio = Portfolio(cash=equity)
    targets = pd.Series({"A": 0.1015})  # 0.1015 * 100000 / 100 = 101.5
    orders = plan_orders(targets, portfolio, equity, {"A": 100.0}, {"A": 0.9}, CONFIG)
    assert orders[0].quantity == pytest.approx(101.0)


def test_plan_orders_sells_before_buys() -> None:
    equity = 100_000.0
    portfolio = _portfolio_with({"OLD": 0.10}, equity)
    targets = pd.Series({"NEW": 0.10})
    orders = plan_orders(
        targets,
        portfolio,
        equity,
        {"OLD": 100.0, "NEW": 100.0},
        {"NEW": 0.9, "OLD": 0.9},
        CONFIG,
    )
    assert orders[0].side == "SELL"
    assert orders[-1].side == "BUY"


def test_plan_orders_cash_never_negative() -> None:
    """Buys are trimmed until the cash outlay fits."""
    equity = 100_000.0
    portfolio = Portfolio(cash=5_000.0)
    targets = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05})
    orders = plan_orders(
        targets,
        portfolio,
        equity,
        {"A": 100.0, "B": 100.0, "C": 100.0},
        {"A": 0.9, "B": 0.8, "C": 0.7},
        CONFIG,
    )

    slippage = float(CONFIG["backtest"]["slippage"])
    cost_pct = float(CONFIG["backtest"]["cost"])
    outlay = sum(
        o.quantity * fill_price_for(100.0, "BUY", slippage) * (1 + cost_pct) for o in orders
    )
    assert outlay <= portfolio.cash + 1e-6, "cash must never go negative"


def test_plan_orders_skips_symbols_without_a_bar() -> None:
    equity = 100_000.0
    portfolio = Portfolio(cash=equity)
    targets = pd.Series({"A": 0.10, "GONE": 0.10})
    orders = plan_orders(targets, portfolio, equity, {"A": 100.0}, {"A": 0.9, "GONE": 0.9}, CONFIG)
    assert [o.symbol for o in orders] == ["A"]


def test_plan_orders_none_places_nothing() -> None:
    portfolio = Portfolio(cash=100_000.0)
    assert plan_orders(None, portfolio, 100_000.0, {"A": 100.0}, {}, CONFIG) == []


def test_plan_orders_buys_ordered_by_probability() -> None:
    equity = 300_000.0
    portfolio = Portfolio(cash=equity)
    targets = pd.Series({"LOW": 0.10, "HIGH": 0.10, "MID": 0.10})
    orders = plan_orders(
        targets,
        portfolio,
        equity,
        {"LOW": 100.0, "HIGH": 100.0, "MID": 100.0},
        {"LOW": 0.6, "HIGH": 0.9, "MID": 0.7},
        CONFIG,
    )
    assert [o.symbol for o in orders] == ["HIGH", "MID", "LOW"]


# --------------------------------------------------------------- metrics


def test_max_drawdown_example() -> None:
    """`TESTS` section 7: equity 100, 120, 90, 110 -> -0.25."""
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])
    assert max_drawdown(equity) == pytest.approx(0.25)


def test_sharpe_example() -> None:
    """`TESTS` section 7: mean 0.0005, std 0.009 -> sqrt(252) * 0.0005 / 0.009 = 0.8819."""
    mean, std, n = 0.0005, 0.009, 2000
    # z has exactly mean 0 and population std 1, so `mean + std * z` has the
    # mean and standard deviation the fixture calls for.
    z = pd.Series([(-1.0) ** i for i in range(n)])
    z = z - z.mean()
    z = z / z.std(ddof=1)
    returns = mean + std * z

    assert returns.mean() == pytest.approx(mean, abs=1e-12)
    assert returns.std(ddof=1) == pytest.approx(std, abs=1e-12)
    assert sharpe(returns, 0.0) == pytest.approx(math.sqrt(252) * mean / std, abs=1e-6)
    assert sharpe(returns, 0.0) == pytest.approx(0.8819, abs=1e-4)


def test_cagr_example() -> None:
    """Equity 100 on 2020-01-01 and 121 on 2022-01-01 -> about 0.0999."""
    equity = pd.Series([100.0, 121.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    assert cagr(equity) == pytest.approx(0.0999, abs=1e-3)


def test_sortino_calmar_turnover_exposure() -> None:
    equity = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    returns = daily_returns(equity)
    assert sortino(returns) > 0
    assert calmar(equity) == pytest.approx(cagr(equity) / max_drawdown(equity), rel=1e-9)

    trades = pd.DataFrame(
        {
            "quantity": [10, 10],
            "fill_price": [10.0, 10.0],
            "date_key": [1, 1],
            "equity": [1000.0, 1000.0],
        }
    )
    assert turnover(trades) == pytest.approx((200.0 / 1000.0) / 1)

    curve = pd.DataFrame(
        {"equity": [1.0, 2.0], "positions_value": [0.0, 1.0]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    assert summarize(curve, pd.DataFrame(), CONFIG)["exposure"] == pytest.approx(0.5)


def test_profit_factor_no_losses() -> None:
    """All positive returns: profit factor is inf, and nothing raises."""
    returns = pd.Series([0.01, 0.02, 0.03])
    assert profit_factor(returns) == float("inf")
    assert win_rate(returns) == pytest.approx(1.0)


def test_annual_volatility_scales() -> None:
    rng = pd.Series([0.001, -0.001, 0.002, -0.002])
    expected = rng.std(ddof=1) * math.sqrt(252)
    assert annual_volatility(rng) == pytest.approx(expected)


def test_summarize_states_risk_free_rate() -> None:
    """DECISIONS R21: the risk-free rate must appear in every results file."""
    curve = pd.DataFrame(
        {"equity": [1_000_000.0, 1_010_000.0], "positions_value": [0.0, 500_000.0]},
        index=pd.date_range("2023-01-02", periods=2),
    )
    summary = summarize(curve, pd.DataFrame(), CONFIG)
    assert summary["risk_free_annual"] == pytest.approx(0.0)
    assert summary["initial_equity"] == pytest.approx(1_000_000.0)
    assert summary["final_equity"] == pytest.approx(1_010_000.0)


def test_summarize_on_empty_trades_does_not_raise() -> None:
    curve = pd.DataFrame(
        {"equity": [1_000_000.0], "positions_value": [0.0]}, index=pd.to_datetime(["2023-01-02"])
    )
    summary = summarize(curve, pd.DataFrame(), CONFIG)
    assert summary["n_trades"] == 0
    assert summary["n_days"] == 1
