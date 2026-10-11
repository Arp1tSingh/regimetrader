"""Paper-trading state, derived from the warehouse (`spec 08` section 3.2; R25).

Nothing about a paper run lives in memory. Holdings and average cost come from replaying
`fact_trade` in order (average-cost method, not FIFO), cash and equity from
`fact_portfolio_daily`, pending decisions from `pending_target`, cooldowns from
`fact_trade.reason = 'STOP_LOSS'` inside the cooldown window, and a date counts as
processed when `fact_portfolio_daily` has a row for (`run_id`, date).

The point of deriving everything is crash recovery: a job killed mid-date and restarted
must rebuild byte-identical state, and a crash must never be mistaken for a decision to
go to cash. Because each date commits atomically, a processed date always has its
`pending_target` rows, so an empty pending set means a real decision to be flat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from src.trading.execution import Portfolio, Position, Trade, apply_trade, mark_to_market


@dataclass
class PaperState:
    """Everything needed to continue a paper run."""

    run_id: str
    #: Symbols sorted, for deterministic iteration.
    held: dict[str, Position] = field(default_factory=dict)
    cash: float = 0.0
    #: Last processed date key, or None when the run has never processed a date.
    last_date_key: int | None = None
    equity: float = 0.0

    @property
    def portfolio(self) -> Portfolio:
        """The engine's `Portfolio` view, so execution code is shared unchanged."""
        return Portfolio(positions=dict(self.held), cash=self.cash)

    @property
    def is_started(self) -> bool:
        return self.last_date_key is not None


def processed_dates(engine: Any, run_id: str) -> list[int]:
    """Date keys this run has already committed, ascending."""
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT date_key FROM fact_portfolio_daily WHERE run_id = :r ORDER BY date_key",
        {"r": run_id},
    )
    if frame.empty:
        return []
    return [int(x) for x in frame["date_key"]]


def last_processed(engine: Any, run_id: str) -> int | None:
    """The most recent committed date key, or None."""
    dates = processed_dates(engine, run_id)
    return dates[-1] if dates else None


def load_state(engine: Any, run_id: str) -> PaperState:
    """Rebuild the run's state by replaying its stored trades.

    Args:
        engine: Warehouse engine.
        run_id: Paper run identifier.

    Returns:
        A `PaperState`. A run with no trades starts flat with zero cash; the caller
        supplies the initial capital when it first processes a date.
    """
    from src.common.db import read_sql

    trades = read_sql(
        engine,
        "SELECT t.date_key AS date_key, s.symbol AS symbol, t.side AS side, "
        "       t.quantity AS quantity, t.fill_price AS fill_price, t.cost AS cost, "
        "       t.reason AS reason "
        "FROM fact_trade t JOIN dim_stock s ON s.stock_key = t.stock_key "
        "WHERE t.run_id = :r ORDER BY t.date_key, t.trade_id",
        {"r": run_id},
    )
    state = PaperState(run_id=run_id)

    # Holdings come from replaying the trades in order (average-cost method).
    if not trades.empty:
        portfolio = Portfolio(cash=0.0, positions={})
        for row in trades.itertuples(index=False):
            apply_trade(
                portfolio,
                Trade(
                    symbol=str(row.symbol),
                    side=str(row.side),
                    quantity=float(row.quantity),
                    fill_price=float(row.fill_price),
                    cost=float(row.cost),
                    reason=str(row.reason) if row.reason else "",
                ),
            )
        state.held = dict(portfolio.positions)

    # Cash and equity come from the last committed portfolio row, NOT from the trade
    # replay. Replaying trades cannot recover cash that was simply never spent, so a run
    # that placed no orders would otherwise come back the next day with zero cash.
    daily = read_sql(
        engine,
        "SELECT date_key, cash, equity FROM fact_portfolio_daily WHERE run_id = :r "
        "ORDER BY date_key",
        {"r": run_id},
    )
    if not daily.empty:
        state.last_date_key = int(daily["date_key"].iloc[-1])
        state.cash = float(daily["cash"].iloc[-1])
        state.equity = float(daily["equity"].iloc[-1])
    return state


def pending_targets(engine: Any, run_id: str, decision_date_key: int) -> dict[str, float]:
    """Target weights decided on `decision_date_key`, to be executed at the next open.

    An empty dict means the strategy deliberately decided to be flat. It is not the
    same as "nothing stored yet", which is why callers must only ask for a date they
    know was processed.
    """
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT s.symbol AS symbol, p.target_weight AS target_weight "
        "FROM pending_target p JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE p.run_id = :r AND p.decision_date_key = :d",
        {"r": run_id, "d": int(decision_date_key)},
    )
    if frame.empty:
        return {}
    return {str(r.symbol): float(r.target_weight) for r in frame.itertuples(index=False)}


def store_pending_targets(
    conn: Any,
    run_id: str,
    decision_date_key: int,
    targets: Mapping[str, float],
    prob_up: Mapping[str, float],
    stock_keys: Mapping[str, int],
) -> None:
    """Replace the pending targets for one decision date.

    Upsert semantics: rows for that date are deleted first so re-running a date cannot
    leave a stale order behind next to a fresh one.
    """
    from sqlalchemy import text

    from src.common.db import execute_many

    conn.execute(
        text("DELETE FROM pending_target WHERE run_id = :r AND decision_date_key = :d"),
        {"r": run_id, "d": int(decision_date_key)},
    )
    rows = [
        {
            "r": run_id,
            "d": int(decision_date_key),
            "s": stock_keys[symbol],
            "w": float(weight),
            "p": prob_up.get(symbol),
        }
        for symbol, weight in sorted(targets.items())
        if symbol in stock_keys
    ]
    if rows:
        execute_many(
            conn,
            "INSERT OR REPLACE INTO pending_target (run_id, decision_date_key, stock_key, "
            "target_weight, prob_up) VALUES (:r, :d, :s, :w, :p)",
            rows,
        )


def cooldown_symbols(engine: Any, run_id: str, as_of_key: int, cooldown_days: int) -> set[str]:
    """Symbols stopped out within the last `cooldown_days` trading days."""
    from src.common.db import read_sql

    dates = processed_dates(engine, run_id)
    window = set(dates[-int(cooldown_days) :]) if cooldown_days > 0 else set()
    window.add(int(as_of_key))
    if not window:
        return set()
    keys = ", ".join(f"{int(d)}" for d in sorted(window))
    frame = read_sql(
        engine,
        f"SELECT DISTINCT s.symbol AS symbol FROM fact_trade t "
        f"JOIN dim_stock s ON s.stock_key = t.stock_key "
        f"WHERE t.run_id = :r AND t.reason = 'STOP_LOSS' AND t.date_key IN ({keys})",
        {"r": run_id},
    )
    if frame.empty:
        return set()
    return {str(x) for x in frame["symbol"]}


def average_cost_trades(trades: pd.DataFrame) -> dict[str, float]:
    """Average cost per symbol from a trade frame, by the average-cost method.

    Used by the round-trip test to confirm the store's replay matches the engine.
    """
    out: dict[str, float] = {}
    for row in trades.sort_values(["symbol", "date_key"]).itertuples(index=False):
        side = str(row.side)
        price = float(row.fill_price)
        if side == "BUY":
            held = out.get(str(row.symbol), 0.0)
            quantity = float(row.quantity)
            total = held * quantity + price * quantity
            out[str(row.symbol)] = total / (quantity + held) if quantity + held else 0.0
        elif str(row.symbol) in out:
            out[str(row.symbol)] = out[str(row.symbol)]
    return out


def mark_state(state: PaperState, closes: Mapping[str, float]) -> tuple[float, float]:
    """Positions value and equity at the given closes, using the engine's own maths."""
    return mark_to_market(state.portfolio, closes)
