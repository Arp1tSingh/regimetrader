"""Backtest engine (`spec 06` sections 2.1 and 2.2).

The ordering is the whole point:

    EXECUTE (orders decided at the previous close, filled at today's open)
    MARK    (value at today's close; detect stop-outs)
    DECIDE  (build today's context from data up to today's close)
    STORE   (keep today's targets as tomorrow's pending orders)

Executing before deciding is what prevents look-ahead: an order decided after the
close of *t* is priced at the open of *t+1* and never at `close_t` (`06` section 2.2).

`run_backtest` refuses any period reaching `splits.test_start` unless
`allow_test_period=True`, which only `final_test.py` sets (section 8).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

import pandas as pd

from src.common.config import ConfigError, load_config
from src.common.dates import to_date_key
from src.common.db import get_engine, init_db
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.trading.execution import (
    Order,
    Portfolio,
    apply_trade,
    fill_order,
    mark_to_market,
    plan_orders,
)
from src.trading.rules import build_regime_rules, target_weights

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2
EXIT_GUARD = 3

#: Columns of the equity curve, matching `fact_portfolio_daily`.
EQUITY_COLUMNS = (
    "run_id",
    "date_key",
    "cash",
    "positions_value",
    "equity",
    "daily_return",
    "drawdown",
)


@dataclass
class DayContext:
    """Everything the weight provider may look at, using data up to today's close.

    Attributes:
        date: The decision date.
        regime: Smoothed regime label for this date.
        signals: Frame indexed by symbol with `prob_up`, equities with a bar today.
        indicators: Frame indexed by symbol with today's indicator columns.
        held: Symbols currently held.
        current_weights: Current weight per held symbol at today's close.
        force_exit: Stop-loss triggers found at today's close.
        no_entry: Cooldowns, flagged or forward-filled bars.
        is_month_end: Whether today is the last trading day of its month.
    """

    date: pd.Timestamp
    regime: str
    signals: pd.DataFrame
    indicators: pd.DataFrame
    held: set[str]
    current_weights: pd.Series
    force_exit: frozenset[str]
    no_entry: frozenset[str]
    is_month_end: bool


@dataclass
class BacktestResult:
    """Outcome of one run."""

    equity: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    run_id: str
    model_name: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _load_market(engine: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Prices, regimes, indicators and dim_date, all indexed by date_key."""
    from src.common.db import read_sql

    prices = read_sql(
        engine,
        "SELECT f.date_key AS date_key, s.symbol AS symbol, s.instrument_type AS itype, "
        "       f.open_price AS open, f.high_price AS high, f.low_price AS low, "
        "       f.close_price AS close, f.volume AS volume, f.is_flagged AS is_flagged "
        "FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE s.instrument_type = 'EQUITY' ORDER BY f.date_key, s.symbol",
    )
    regimes = read_sql(
        engine,
        "SELECT m.date_key AS date_key, r.regime_label AS regime "
        "FROM fact_market_regime m JOIN dim_regime r ON r.regime_key = m.regime_key",
    )
    indicators = read_sql(
        engine,
        "SELECT i.date_key AS date_key, s.symbol AS symbol, i.return_20d AS return_20d, "
        "       i.vol_20d AS vol_20d, i.rsi_14 AS rsi_14 "
        "FROM fact_indicator i JOIN dim_stock s ON s.stock_key = i.stock_key "
        "WHERE s.instrument_type = 'EQUITY'",
    )
    month_ends = read_sql(engine, "SELECT date_key FROM dim_date WHERE is_month_end = 1")

    return {
        "opens": _pivot(prices, "open"),
        "closes": _pivot(prices, "close"),
        "flagged": _pivot(prices, "is_flagged"),
        "regimes": dict(zip(regimes["date_key"], regimes["regime"], strict=True)),
        "indicators": indicators,
        "month_ends": set(month_ends["date_key"]),
    }


def _pivot(prices: pd.DataFrame, column: str) -> dict[int, dict[str, float]]:
    """date_key -> {symbol: value}."""
    if prices.empty:
        return {}
    frame = prices.pivot_table(index="date_key", columns="symbol", values=column, aggfunc="first")
    return {int(k): {c: float(v) for c, v in row.dropna().items()} for k, row in frame.iterrows()}


def _load_signals(
    engine: Any, model_name: str, from_key: int, to_key: int
) -> dict[int, dict[str, float]]:
    """date_key -> {symbol: prob_up} for one model's out-of-sample predictions."""
    from src.common.db import read_sql

    df = read_sql(
        engine,
        "SELECT f.date_key AS date_key, s.symbol AS symbol, f.prob_up AS prob_up "
        "FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b "
        "ORDER BY f.date_key, s.symbol",
        {"model": model_name, "a": int(from_key), "b": int(to_key)},
    )
    if df.empty:
        return {}
    return {
        int(key): dict(zip(group["symbol"], group["prob_up"], strict=True))
        for key, group in df.groupby("date_key")
    }


def run_backtest(
    cfg: Mapping[str, Any],
    weights_fn: Callable[[DayContext], pd.Series | None],
    start: str,
    end: str,
    run_id: str,
    *,
    engine: Any = None,
    persist: bool = True,
    write_signals: bool = False,
    model_name: str | None = None,
    costs_enabled: bool = True,
    stop_loss_enabled: bool = True,
    band: float | None = None,
    fractional_shares: bool = False,
    allow_test_period: bool = False,
) -> BacktestResult:
    """Simulate the strategy over a period.

    Args:
        cfg: Loaded config.
        weights_fn: Returns target weights for the next open, or None for "no orders".
        start: First decision date, `YYYY-MM-DD`.
        end: Last date to mark, `YYYY-MM-DD`.
        run_id: Identifier written to `fact_trade` and `fact_portfolio_daily`.
        engine: Optional warehouse engine; one is created when omitted.
        persist: Write results to the warehouse, replacing any existing rows for
            this `run_id` so reruns are idempotent.
        write_signals: Update `fact_signal.signal` and `target_weight`. Only the
            primary strategy run may do this (`DECISIONS` R3).
        model_name: Model whose `prob_up` drives the signals.
        costs_enabled: Apply slippage and costs.
        stop_loss_enabled: Check the stop-loss at each close.
        band: Rebalance band override.
        fractional_shares: Allow fractional quantities (index baseline B1).
        allow_test_period: Only `final_test.py` sets this.

    Returns:
        A `BacktestResult`.

    Raises:
        ValueError: If the period reaches `splits.test_start` and the test period has
            not been explicitly allowed.
    """
    test_start = str(cfg["splits"]["test_start"])
    if not allow_test_period and end >= test_start:
        raise ValueError(
            f"backtest end {end} reaches test_start {test_start}; the test period is "
            "evaluated once by final_test.py with allow_test_period=True"
        )

    log = get_logger("src.backtest.engine", cfg)
    engine = engine or get_engine(cfg)
    if persist:
        init_db(engine)

    start_key = to_date_key(start)
    end_key = to_date_key(end)
    market = _load_market(engine, cfg)
    signals = _load_signals(engine, model_name, start_key, end_key) if model_name else {}

    dates = sorted(k for k in market["closes"] if start_key <= k <= end_key)
    if not dates:
        raise ValueError(f"no trading dates between {start} and {end}")

    stop_cfg = cfg["strategy"]["stop_loss"]
    cooldown_days = int(stop_cfg.get("cooldown_days", 5))

    capital = float(cfg["backtest"]["initial_capital"])
    portfolio = Portfolio(cash=capital)
    pending: pd.Series | None = None
    pending_probs: dict[str, float] = {}

    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    cooldown_until: dict[str, int] = {}
    stop_pending: set[str] = set()
    last_close: dict[str, float] = {}

    for date_key in dates:
        opens_today = market["opens"].get(date_key, {})
        closes_today = market["closes"].get(date_key, {})
        flagged_today = market["flagged"].get(date_key, {})

        # A stop-out detected at the previous close exits now, at this open.
        if stop_pending:
            for symbol in sorted(stop_pending):
                if symbol in portfolio.positions:
                    pending = None if pending is None else pending
                    order = Order(symbol, "SELL", portfolio.positions[symbol].quantity, "STOP_LOSS")
                    if symbol in opens_today:
                        trade = fill_order(order, opens_today[symbol], cfg, costs_enabled)
                        apply_trade(portfolio, trade)
                        trade_rows.append(
                            {
                                "run_id": run_id,
                                "date_key": date_key,
                                "symbol": symbol,
                                "side": "SELL",
                                "quantity": trade.quantity,
                                "fill_price": trade.fill_price,
                                "cost": trade.cost,
                                "realized_pnl": trade.realized_pnl,
                                "reason": "STOP_LOSS",
                            }
                        )
                        # Cooldown counts trading days from this exit.
                        cooldown_until[symbol] = _nth_trading_day(dates, date_key, cooldown_days)
            stop_pending = set()

        # 1. EXECUTE: orders decided at the previous close, filled at today's open.
        if pending is not None:
            orders = plan_orders(
                pending,
                portfolio,
                capital,
                opens_today,
                pending_probs,
                cfg,
                band=band,
                fractional_shares=fractional_shares,
            )
            for order in orders:
                open_price = opens_today.get(order.symbol)
                if open_price is None:
                    continue  # no bar today: the next decision recomputes
                trade = fill_order(order, open_price, cfg, costs_enabled)
                apply_trade(portfolio, trade)
                trade_rows.append(
                    {
                        "run_id": run_id,
                        "date_key": date_key,
                        "symbol": order.symbol,
                        "side": trade.side,
                        "quantity": trade.quantity,
                        "fill_price": trade.fill_price,
                        "cost": trade.cost,
                        "realized_pnl": trade.realized_pnl,
                        "reason": trade.reason,
                    }
                )
            pending = None
            pending_probs = {}

        # 2. MARK: value at today's close; a missing bar uses the last known close.
        marks = dict(last_close)
        marks.update(closes_today)
        for symbol, close in closes_today.items():
            last_close[symbol] = close
        positions_value, equity = mark_to_market(portfolio, marks)
        # The running peak must include today, otherwise a new equity high looks
        # like a negative drawdown.
        running_max = max(max((r["equity"] for r in equity_rows), default=equity), equity)
        equity_rows.append(
            {
                "run_id": run_id,
                "date_key": date_key,
                "cash": portfolio.cash,
                "positions_value": positions_value,
                "equity": equity,
                "daily_return": float("nan"),
                "drawdown": (running_max - equity) / running_max if running_max else 0.0,
            }
        )
        capital = equity

        # Stop-loss detected at today's close; it forces tomorrow's exit.
        force_exit: frozenset[str] = frozenset()
        if stop_loss_enabled and bool(stop_cfg.get("enabled", True)):
            triggered = set()
            for symbol, position in list(portfolio.positions.items()):
                close = marks.get(symbol)
                if close is not None and close <= position.avg_fill * (
                    1.0 - float(stop_cfg["pct"])
                ):
                    triggered.add(symbol)
            force_exit = frozenset(triggered)

        # Cooldowns from earlier stop-outs, plus flagged or filled bars today.
        cooling = {symbol for symbol, until in cooldown_until.items() if date_key <= until}
        no_entry = frozenset(set(cooling) | {s for s, flag in flagged_today.items() if flag})

        # 3. DECIDE: only data up to today's close.
        regime = market["regimes"].get(date_key, "SIDEWAYS")
        probs_today = signals.get(date_key, {})
        signal_frame = pd.DataFrame(
            {"prob_up": pd.Series(probs_today, dtype="float64")}
            if probs_today
            else pd.DataFrame(columns=["prob_up"])
        )
        indicator_rows = market["indicators"]
        today_indicators = (
            indicator_rows[indicator_rows["date_key"] == date_key].set_index("symbol")
            if not indicator_rows.empty
            else pd.DataFrame()
        )

        _, equity_now = mark_to_market(portfolio, marks)
        current_weights = pd.Series(
            {
                symbol: (portfolio.positions[symbol].quantity * marks.get(symbol, 0.0)) / equity_now
                for symbol in portfolio.positions
                if equity_now
            },
            dtype="float64",
        )

        context = DayContext(
            date=pd.Timestamp(
                year=date_key // 10000, month=(date_key // 100) % 100, day=date_key % 100
            ),
            regime=regime,
            signals=signal_frame,
            indicators=today_indicators,
            held=set(portfolio.positions),
            current_weights=current_weights,
            force_exit=force_exit,
            no_entry=no_entry,
            is_month_end=date_key in market["month_ends"],
        )

        targets = weights_fn(context)

        # 4. STORE: today's targets become tomorrow's pending orders.
        if targets is not None and len(targets):
            pending = targets
            pending_probs = {s: float(v) for s, v in probs_today.items()}
            for symbol, weight in targets.items():
                decision_rows.append(
                    {
                        "run_id": run_id,
                        "date_key": date_key,
                        "symbol": symbol,
                        "signal": 1,
                        "target_weight": float(weight),
                    }
                )
        else:
            pending = None
            pending_probs = {}

        # A stop-out detected here exits at tomorrow's open; the cooldown is registered
        # then, when the trade actually fills.
        stop_pending = set(force_exit)

    equity_df = pd.DataFrame(equity_rows)
    equity_df["daily_return"] = equity_df["equity"].pct_change()
    trades_df = pd.DataFrame(trade_rows)
    signals_df = pd.DataFrame(decision_rows)

    if persist:
        _persist(engine, run_id, equity_df, trades_df, model_name, cfg)
    if write_signals and model_name:
        _write_signal_columns(engine, run_id, signals_df, model_name)

    log.info(
        "%s: %d day(s), %d trade(s), final equity %.2f",
        run_id,
        len(equity_df),
        len(trades_df),
        float(equity_df["equity"].iloc[-1]) if len(equity_df) else float("nan"),
    )
    return BacktestResult(
        equity=equity_df,
        trades=trades_df,
        signals=signals_df,
        run_id=run_id,
        model_name=model_name,
    )


def _persist(
    engine: Any,
    run_id: str,
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    model_name: str | None,
    cfg: Mapping[str, Any],
) -> None:
    """Write equity and trades, replacing any previous rows for this run_id."""
    from sqlalchemy import text

    from src.common.db import execute_many

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM fact_trade WHERE run_id = :run_id"), {"run_id": run_id})
        conn.execute(
            text("DELETE FROM fact_portfolio_daily WHERE run_id = :run_id"), {"run_id": run_id}
        )

        if not trades_df.empty:
            key_map = dict(conn.execute(text("SELECT symbol, stock_key FROM dim_stock")).fetchall())
            rows = [
                {
                    "run_id": run_id,
                    "date_key": int(r["date_key"]),
                    "stock_key": int(key_map[r["symbol"]]),
                    "side": r["side"],
                    "quantity": int(r["quantity"]),
                    "fill_price": float(r["fill_price"]),
                    "cost": float(r["cost"]),
                    "realized_pnl": None
                    if r["realized_pnl"] is None or pd.isna(r["realized_pnl"])
                    else float(r["realized_pnl"]),
                    "reason": r["reason"],
                }
                for r in trades_df.to_dict("records")
                if r["symbol"] in key_map
            ]
            execute_many(
                conn,
                "INSERT INTO fact_trade (run_id, date_key, stock_key, side, quantity, "
                "fill_price, cost, realized_pnl, reason) "
                "VALUES (:run_id, :date_key, :stock_key, :side, :quantity, :fill_price, "
                ":cost, :realized_pnl, :reason)",
                rows,
            )

        execute_many(
            conn,
            "INSERT INTO fact_portfolio_daily (run_id, date_key, cash, positions_value, "
            "equity, daily_return, drawdown) VALUES (:run_id, :date_key, :cash, "
            ":positions_value, :equity, :daily_return, :drawdown)",
            [
                {
                    "run_id": run_id,
                    "date_key": int(r["date_key"]),
                    "cash": float(r["cash"]),
                    "positions_value": float(r["positions_value"]),
                    "equity": float(r["equity"]),
                    "daily_return": None
                    if pd.isna(r["daily_return"])
                    else float(r["daily_return"]),
                    "drawdown": float(r["drawdown"]),
                }
                for r in equity_df.to_dict("records")
            ],
        )
    del model_name, cfg


def _write_signal_columns(
    engine: Any, run_id: str, signals_df: pd.DataFrame, model_name: str
) -> None:
    """Update `fact_signal.signal` and `target_weight` for the primary run only."""
    from sqlalchemy import text

    if signals_df.empty:
        return
    with engine.begin() as conn:
        key_map = dict(conn.execute(text("SELECT symbol, stock_key FROM dim_stock")).fetchall())
        model_rows = conn.execute(
            text("SELECT model_key, version FROM dim_model WHERE model_name = :m"),
            {"m": model_name},
        ).fetchall()
        for row in signals_df.to_dict("records"):
            stock_key = key_map.get(row["symbol"])
            if stock_key is None:
                continue
            # Match the model version whose training window ends before this date.
            for model_key, version in model_rows:
                year = int(str(version).replace("wf", "") or 0)
                if year <= int(row["date_key"]) // 10000:
                    conn.execute(
                        text(
                            "UPDATE fact_signal SET signal = :sig, target_weight = :w "
                            "WHERE date_key = :d AND stock_key = :s AND model_key = :m"
                        ),
                        {
                            "sig": int(row["signal"]),
                            "w": float(row["target_weight"]),
                            "d": int(row["date_key"]),
                            "s": int(stock_key),
                            "m": int(model_key),
                        },
                    )


def _nth_trading_day(dates: list[int], from_key: int, n: int) -> int:
    """The date `n` trading days after `from_key`, clamped to the last date."""
    future = [d for d in dates if d > from_key]
    if not future:
        return from_key
    return future[min(int(n), len(future)) - 1]


def _strategy_weights_fn(
    cfg: Mapping[str, Any], stop_loss_enabled: bool = True
) -> Callable[[DayContext], pd.Series | None]:
    """The V0 strategy: regime rules plus stop-loss, cooldown and the band."""
    rules = build_regime_rules(cfg)
    slots = int(cfg["strategy"]["slots"])
    exit_prob = float(cfg["strategy"]["exit_prob"])

    def weights_fn(context: DayContext) -> pd.Series | None:
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

    del stop_loss_enabled
    return weights_fn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.backtest.engine",
        description="Backtest the strategy on the validation period.",
    )
    parser.add_argument("--run-id", default="backtest_main")
    parser.add_argument("--model", default=None, help="Model whose prob_up to use.")
    parser.add_argument("--variant", default="V0", help="Strategy variant V0, V1 or V2.")
    parser.add_argument("--no-cost", action="store_true", help="Disable slippage and cost.")
    parser.add_argument("--period", default="validation", help="validation or test.")
    parser.add_argument("--config", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 3 guard refusal, 1 or 2 on failure."""
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    model = args.model or cfg["strategy"].get("selected_model")
    if not model:
        print("no model selected; pass --model or run `make backtest` first", file=sys.stderr)
        return EXIT_ERROR

    start = str(cfg["splits"]["validation_start"])
    end = str(cfg["splits"]["validation_end"])
    try:
        result = run_backtest(
            cfg,
            _strategy_weights_fn(cfg),
            start,
            end,
            args.run_id,
            model_name=model,
            costs_enabled=not args.no_cost,
        )
    except ValueError as exc:
        if "test period" in str(exc):
            print(str(exc), file=sys.stderr)
            return EXIT_GUARD
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    from src.backtest.metrics import summarize

    summary = summarize(result.equity, result.trades, cfg)
    write_manifest(
        cfg,
        args.run_id,
        get_engine(cfg),
        models_used=[model],
        extra={
            "variant": args.variant,
            "period": args.period,
            **{k: v for k, v in summary.items() if isinstance(v, (int, float, str))},
        },
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())


def _today() -> str:  # pragma: no cover - convenience for CLI runs
    return date.today().isoformat()
