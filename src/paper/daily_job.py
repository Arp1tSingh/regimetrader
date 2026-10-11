"""The daily paper-trading job (`spec 08` section 3; `DECISIONS` R25, R26).

One trading date at a time, in ascending order, with steps a to f of a date committed in
**one transaction or not at all**. That is what makes a crash safe: a date either has its
portfolio row, its trades and its pending targets, or it has none of them and will be
redone. Because a processed date always has its `pending_target` rows, an empty pending
set means a real decision to be flat, never an artefact of a half-finished run.

Guards (`R26`): a new `paper_live` run starts flat at the latest trading date and is
refused until `TEST_RUN_LOCK.json` exists; it writes `fact_signal` only for dates after
the lock's `data_end_date`; a `paper_replay` never touches `fact_signal`.

The strategy code is the same code the backtest used (`src/trading/`), so a replay
reproduces the backtest exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ConfigError, load_config
from src.common.db import get_engine
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.paper import alerts
from src.paper.store import (
    cooldown_symbols,
    last_processed,
    load_state,
    pending_targets,
    store_pending_targets,
)

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2
EXIT_REFUSED = 3


def trading_dates(
    engine: Any, after_key: int | None, from_key: int | None, to_key: int | None
) -> list[int]:
    """Trading dates the index has a bar for, ascending.

    A date counts as trading only when the index itself traded, which is what
    distinguishes a session from a holiday or a not-yet-published day.
    """
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT p.date_key AS date_key FROM fact_price p "
        "JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE s.instrument_type = 'INDEX' "
        "AND p.date_key >= :a AND p.date_key <= :b "
        "ORDER BY p.date_key",
        {"a": int(after_key or from_key or 0), "b": int(to_key or 99999999)},
    )
    return [int(x) for x in frame["date_key"]] if not frame.empty else []


def latest_trading_date(engine: Any) -> int | None:
    """The most recent date the index traded."""
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT MAX(p.date_key) AS d FROM fact_price p "
        "JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE s.instrument_type = 'INDEX'",
    )
    if frame.empty or frame["d"].iloc[0] is None:
        return None
    return int(frame["d"].iloc[0])


def guards(cfg: dict[str, Any], run_id: str, engine: Any) -> str | None:
    """Refusal message for this run, or None when it may proceed (R26)."""
    if run_id != str(cfg["paper"]["run_id"]):
        return None  # replays are not live runs

    lock_path = Path(cfg["paths"]["results"]) / "TEST_RUN_LOCK.json"
    if not lock_path.exists():
        return (
            f"refusing to start {run_id}: {lock_path} does not exist. The strategy must "
            "be frozen and the test period evaluated before live paper trading starts."
        )
    frozen = Path(cfg["paths"]["frozen_config"])
    if not frozen.exists():
        return f"refusing to start {run_id}: no frozen config at {frozen}"
    return None


def signal_dates_allowed(cfg: dict[str, Any], run_id: str, date_key: int) -> bool:
    """Whether this run may write `fact_signal` for this date (R26)."""
    if run_id != str(cfg["paper"]["run_id"]):
        return False  # a replay never writes fact_signal
    lock_path = Path(cfg["paths"]["results"]) / "TEST_RUN_LOCK.json"
    if not lock_path.exists():
        return False
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return int(date_key) > int(lock["data_end_date"])


def _market_snapshot(engine: Any, date_key: int) -> dict[str, dict[str, float]]:
    """Opens, closes and flagged flags for one date."""
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT s.symbol AS symbol, p.open_price AS open, p.close_price AS close, "
        "       p.is_flagged AS flagged "
        "FROM fact_price p JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE p.date_key = :d",
        {"d": int(date_key)},
    )
    if frame.empty:
        return {}
    return {
        "opens": {str(r.symbol): float(r.open) for r in frame.itertuples(index=False)},
        "closes": {str(r.symbol): float(r.close) for r in frame.itertuples(index=False)},
        "flagged": {
            str(r.symbol)
            for r in frame.itertuples(index=False)
            if r.flagged and pd.notna(r.flagged)
        },
    }


def _regime_for(engine: Any, date_key: int) -> str:
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key WHERE m.date_key = :d",
        {"d": int(date_key)},
    )
    if frame.empty:
        return "SIDEWAYS"
    return str(frame["regime"].iloc[0])


def _probabilities(engine: Any, model: str, date_key: int) -> dict[str, float]:
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT s.symbol AS symbol, f.prob_up AS prob_up FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE m.model_name = :model AND f.date_key = :d",
        {"model": model, "d": int(date_key)},
    )
    if frame.empty:
        return {}
    return {
        str(r.symbol): float(r.prob_up)
        for r in frame.itertuples(index=False)
        if pd.notna(r.prob_up)
    }


def process_date(
    cfg: dict[str, Any],
    engine: Any,
    run_id: str,
    date_key: int,
    log: Any,
) -> dict[str, Any]:
    """Process one trading date atomically: settle, mark, decide, store.

    Args:
        cfg: Loaded config.
        engine: Warehouse engine.
        run_id: Paper run identifier.
        date_key: The date to process.
        log: Logger.

    Returns:
        A dict describing what happened: trades filled, targets decided, equity.
    """
    from sqlalchemy import text

    from src.backtest.strategies import make_weights_fn
    from src.common.db import execute_many, read_sql
    from src.trading.execution import (
        Portfolio,
        apply_trade,
        fill_order,
        mark_to_market,
        plan_orders,
    )

    model = str(cfg["strategy"].get("selected_model") or cfg["tuning"]["models"][0])
    if not cfg["strategy"].get("selected_model"):
        log.warning(
            "no frozen selected_model; falling back to the first configured model %r", model
        )
    initial = float(cfg["backtest"]["initial_capital"])
    state = load_state(engine, run_id)
    stock_keys = dict(
        read_sql(engine, "SELECT symbol, stock_key FROM dim_stock").itertuples(
            index=False, name=None
        )
    )

    with engine.begin() as conn:
        previous = int(
            conn.execute(
                text("SELECT MAX(date_key) AS d FROM fact_portfolio_daily WHERE run_id = :r"),
                {"r": run_id},
            ).fetchone()[0]
            or 0
        )
        portfolio = Portfolio(cash=state.cash, positions=dict(state.held))
        if not previous:
            # First date of the run: start flat with the configured capital.
            portfolio = Portfolio(cash=initial, positions={})
            equity_at_decision = initial
        else:
            closes_prev = _market_snapshot(engine, previous)["closes"]
            _value, equity_at_decision = mark_to_market(portfolio, closes_prev)

        # (a) SETTLE the previous decision at today's open.
        snapshot = _market_snapshot(engine, date_key)
        targets_prev = pending_targets(engine, run_id, previous) if previous else {}
        model = str(cfg["strategy"].get("selected_model") or cfg["tuning"]["models"][0])
        probs = _probabilities(engine, model, date_key)
        targets_series = (
            pd.Series(targets_prev, dtype="float64") if targets_prev else pd.Series(dtype="float64")
        )
        orders = plan_orders(
            targets_series if previous else None,
            portfolio,
            equity_at_decision,
            snapshot["opens"],
            probs,
            cfg,
        )

        trade_rows = []
        for order in orders:
            # `fill_order` takes the execution day's open, not the portfolio.
            open_price = snapshot["opens"].get(order.symbol)
            if open_price is None:
                continue
            fill = fill_order(order, open_price, cfg)
            apply_trade(portfolio, fill)
            trade_rows.append(
                {
                    "run_id": run_id,
                    "date_key": int(date_key),
                    "stock_key": stock_keys.get(order.symbol),
                    "side": order.side,
                    "quantity": fill.quantity,
                    "fill_price": fill.fill_price,
                    "cost": fill.cost,
                    "reason": order.reason,
                    "realized_pnl": fill.realized_pnl,
                }
            )

        # (b) MARK at today's close.
        _value, equity = mark_to_market(portfolio, snapshot["closes"])
        peak = equity_at_decision if equity_at_decision > 0 else equity
        previous_equity = equity_at_decision
        if previous:
            from src.common.db import read_sql

            history = read_sql(
                engine,
                "SELECT MAX(equity) AS m FROM fact_portfolio_daily WHERE run_id = :r",
                {"r": run_id},
            )
            if not history.empty and history["m"].iloc[0] is not None:
                peak = max(float(history["m"].iloc[0]), equity)
        drawdown = (peak - equity) / peak if peak else 0.0

        conn.execute(
            text("DELETE FROM fact_portfolio_daily WHERE run_id = :r AND date_key = :d"),
            {"r": run_id, "d": int(date_key)},
        )
        conn.execute(
            text(
                "INSERT INTO fact_portfolio_daily (run_id, date_key, cash, positions_value, "
                "equity, daily_return, drawdown) VALUES (:r, :d, :c, :pv, :e, :dr, :dd)"
            ),
            {
                "r": run_id,
                "d": int(date_key),
                "c": float(portfolio.cash),
                "pv": float(equity - portfolio.cash),
                "e": float(equity),
                "dr": float(equity / previous_equity - 1.0) if previous_equity else float("nan"),
                "dd": float(drawdown),
            },
        )

        if trade_rows:
            execute_many(
                conn,
                "INSERT OR REPLACE INTO fact_trade (run_id, date_key, stock_key, side, "
                "quantity, fill_price, cost, realized_pnl, reason) "
                "VALUES (:run_id, :date_key, :stock_key, :side, :quantity, "
                ":fill_price, :cost, :realized_pnl, :reason)",
                trade_rows,
            )

        # (c,d,e) indicators, regime and probabilities must already exist for this date.
        regime = _regime_for(engine, date_key)
        probs = _probabilities(engine, model, date_key)
        if not probs:
            raise ValueError(
                f"no stored probabilities for {model} on {date_key}; run "
                "`python -m src.models.train` before the paper job"
            )

        # (f) DECIDE and store tomorrow's targets.
        force_exit = frozenset()
        cooldown = cooldown_symbols(
            engine,
            run_id,
            date_key,
            int(cfg["strategy"]["stop_loss"]["cooldown_days"]),
        )
        signals = pd.DataFrame({"prob_up": pd.Series(probs, dtype="float64")})
        provider = make_weights_fn("V2", cfg)
        context = _DayContext(
            date=pd.Timestamp(str(int(date_key))),
            regime=regime,
            signals=signals,
            held=set(portfolio.positions),
            current_weights=pd.Series(dtype="float64"),
            force_exit=force_exit,
            no_entry=frozenset(cooldown | snapshot["flagged"]),
            is_month_end=False,
        )
        tomorrow = provider(context)
        # `tomorrow` is a Series or None; `if tomorrow` would be ambiguous.
        weights = {str(k): float(v) for k, v in tomorrow.items()} if tomorrow is not None else {}

        store_pending_targets(conn, run_id, date_key, weights, probs, stock_keys)

    log.info(
        "%s %d: %d fill(s), equity %.2f, regime %s, %d target(s) for the next open",
        run_id,
        date_key,
        len(trade_rows),
        equity,
        regime,
        len(weights),
    )
    return {
        "date_key": int(date_key),
        "trades": len(trade_rows),
        "equity": float(equity),
        "regime": regime,
        "targets": len(weights),
        "held": sorted(portfolio.positions),
    }


class _DayContext:
    """Minimal `DayContext` carrying only what `target_weights` reads.

    The paper job reuses the trading rules, not the backtest engine, so it builds the
    context itself. Only `signals`, `regime`, `held`, `force_exit` and `no_entry` affect
    the decision, and all five come from the warehouse.
    """

    def __init__(
        self,
        *,
        date: pd.Timestamp,
        regime: str,
        signals: pd.DataFrame,
        held: set[str],
        current_weights: pd.Series,
        force_exit: frozenset[str],
        no_entry: frozenset[str],
        is_month_end: bool,
    ) -> None:
        self.date = date
        self.regime = regime
        self.signals = signals
        self.held = held
        self.current_weights = current_weights
        self.force_exit = force_exit
        self.no_entry = no_entry
        self.is_month_end = is_month_end


def run(
    cfg: dict[str, Any],
    *,
    run_id: str | None = None,
    from_key: int | None = None,
    to_key: int | None = None,
    no_etl: bool = True,
) -> dict[str, Any]:
    """Process every outstanding trading date in order.

    Args:
        cfg: Loaded config.
        run_id: Paper run id; defaults to `paper.run_id`.
        from_key: First date for a brand-new run. A run that has already processed a
            date continues from its last one and ignores this.
        to_key: Last date to process.
        no_etl: Skip the incremental load. The default here is True because the ETL is
            run separately; the CLI passes False unless `--no-etl` is given.

    Returns:
        A summary dict.
    """
    log = get_logger("src.paper.daily_job", cfg)
    engine = get_engine(cfg)
    identifier = str(run_id or cfg["paper"]["run_id"])

    refusal = guards(cfg, identifier, engine)
    if refusal:
        log.error("%s", refusal)
        return {"refused": True, "message": refusal, "processed": []}

    if not no_etl:
        from src.etl.pipeline import run as etl_run

        code = etl_run(cfg, full_refresh=False)
        if code != 0:
            message = "incremental ETL failed; refusing to trade on stale data"
            alerts.alert(cfg, "ERROR", message)
            return {"refused": True, "message": message, "processed": []}

    previous = last_processed(engine, identifier)
    dates = trading_dates(
        engine,
        after_key=previous,
        from_key=from_key,
        to_key=to_key,
    )
    if previous is None:
        # A new run starts flat with nothing to settle (spec 08 section 3.2).
        # `paper_live` always starts at the latest trading date, so it cannot back-fill a
        # fake live history (R26). A replay honours --from, because back-filling is
        # exactly what a replay is for.
        is_live = identifier == str(cfg["paper"]["run_id"])
        if is_live or from_key is None:
            latest = latest_trading_date(engine)
            dates = [latest] if latest is not None else []
            log.info(
                "%s is new: starting flat at the latest trading date %s (R26)",
                identifier,
                latest,
            )
        else:
            dates = trading_dates(engine, after_key=None, from_key=from_key, to_key=to_key)
            log.info(
                "%s is new: starting flat at %s (replay, --from honoured)",
                identifier,
                from_key,
            )

    if not dates:
        log.info("nothing to do (holiday or already processed)")
        return {"refused": False, "processed": [], "message": "nothing to do"}

    processed = [process_date(cfg, engine, identifier, key, log) for key in dates]

    # (g) REPORT, latest date only.
    from src.paper.report import write_report

    report_path = write_report(cfg, engine, identifier, processed[-1]["date_key"])
    log.info("wrote %s", report_path)

    write_manifest(
        cfg,
        f"paper_{identifier}",
        engine,
        extra={"run_id": identifier, "dates": [p["date_key"] for p in processed]},
    )
    return {"refused": False, "processed": processed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paper-trading daily job.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--from", dest="from_key", default=None, help="YYYYMMDD")
    parser.add_argument("--to", dest="to_key", default=None, help="YYYYMMDD")
    parser.add_argument("--no-etl", action="store_true", help="Skip the incremental load.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    def as_int(value: str | None) -> int | None:
        return int(str(value).replace("-", "")) if value else None

    outcome = run(
        cfg,
        run_id=args.run_id,
        from_key=as_int(args.from_key),
        to_key=as_int(args.to_key),
        no_etl=args.no_etl,
    )
    if outcome["refused"]:
        print(outcome["message"], file=sys.stderr)
        return EXIT_REFUSED
    if not outcome["processed"]:
        print(outcome.get("message", "nothing to do"))
        return EXIT_OK
    for item in outcome["processed"]:
        print(
            f"{item['date_key']}: {item['trades']} fill(s), equity {item['equity']:.2f}, "
            f"regime {item['regime']}, {item['targets']} target(s)"
        )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
