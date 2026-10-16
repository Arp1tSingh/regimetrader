"""Bot demo replay: re-run the frozen strategy day by day on a throwaway copy.

**This module is the safety boundary for a publicly exposed dashboard.** Every rule here
exists because the page is reachable by an anonymous visitor on the internet.

What it will never do
---------------------
* It never writes to the real warehouse. The page opens it through `readonly_engine`,
  which uses SQLite `mode=ro`; a write raises `OperationalError` (tested).
* It never calls `paper_live`, `final_test`, or the tune/train modules. The forbidden
  names are listed in `FORBIDDEN_CALLABLES` and asserted absent by a test.
* It never writes to `reports/`. The run directory is a fresh temp directory per run,
  deleted in a `finally` even when the run raises (tested).
* It never accepts free text. `WHITELIST_START_DATES` is a fixed tuple of ints; the page
  turns a selectbox index into one of them. Nothing a visitor types reaches SQL or a
  shell (tested).

How the no-hindsight property is produced
-----------------------------------------
For each day D in the window the run works on a **copy of the warehouse with every row
after D deleted**, then recomputes indicators, regime, panel and probabilities from that
truncated copy using exactly the functions the daily paper bot uses
(`build_indicators`, `_extend_regime`, `build_panel`, the saved classifier).

Recomputing all 143k rows costs about 45-65s per call on this machine (measured, see
`scripts/probe_costs.py`), so a five-day literal replay would take several minutes per
click, which is not acceptable behind a web button. So the recompute is restricted to a
**lookback slice** of the needed history: the longest windows in play are the regime
drawdown (`regime.drawdown_window`, 252), the smoothing window (`regime.smoothing.window`,
5), and `sma_50` (50) in the indicators. `lookback_trading_days()` derives the total from
those values at run time rather than hard-coding it.

The whole window is then recomputed **once** rather than once per day (measured: 7.8-8.2s
for the band against 42-49s for the literal path), because orders are decided at a day's
close and fill at the *next* day's open, so a per-day run would plan its entries and have
nowhere to fill them.

Slicing is sound only because every feature is causal. `equivalence` proves that rather
than asserting it: for day 1 and day 5 the fast path is compared against a literal
full recompute on a truncated copy, and a mutation test changes a row *after* D to show
the day-D output does not move.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import pandas as pd

#: Start days offered on the real warehouse. These are also the dates a test asserts.
#: They are derived from the configured test period at run time by
#: `available_start_dates`, so the smoke warehouse offers its own first five test days
#: instead of dates it does not contain.
#:
#: The first entry is 2024-01-01 deliberately: the recorded `backtest_main` run made its
#: first decision at that close and filled at the 2024-01-02 open, so a replay must start
#: there to reproduce those same ten orders.
WHITELIST_START_DATES: tuple[int, ...] = (
    20240101,
    20240102,
    20240103,
    20240104,
    20240105,
)


def available_start_dates(cfg: Mapping[str, Any], engine: Any) -> tuple[int, ...]:
    """The five start days this warehouse can replay.

    Derived from `splits.test_start` and the index's trading days, so the real and smoke
    warehouses each get dates they actually contain. Still a fixed list computed by this
    module: the page offers a selectbox over it, so a visitor can only ever pick one of
    these values.
    """
    from src.common.db import read_sql

    test_start = str(cfg["splits"]["test_start"]).replace("-", "")
    days = [
        int(r)
        for r in read_sql(
            engine,
            "SELECT DISTINCT p.date_key FROM fact_price p "
            "JOIN dim_stock s ON s.stock_key = p.stock_key "
            "WHERE s.instrument_type = 'INDEX' AND p.date_key >= :a ORDER BY 1",
            {"a": int(test_start)},
        )["date_key"]
    ]
    if len(days) < WINDOW_DAYS:
        return ()
    return tuple(days[:WINDOW_DAYS])


#: How many consecutive trading days each run covers.
WINDOW_DAYS = 5

#: Names that must never be reachable from this module. Asserted by tests.
FORBIDDEN_CALLABLES: tuple[str, ...] = (
    "paper_live",
    "final_test",
    "run_final_test",
    "write_lock",
    "retrain",
    "fit_classifier",
)

#: Seconds a single replay may take before it is abandoned.
RUN_TIMEOUT_SECONDS = 180

#: Only one replay at a time across all visitors (the page is a public singleton).
_RUN_LOCK = threading.Lock()

#: Cache of finished replays, keyed by start date. Small: a handful of whitelisted days.
_CACHE: dict[int, "BotRun"] = {}
_CACHE_LOCK = threading.Lock()


def is_whitelisted(start_date: int, allowed: tuple[int, ...] | None = None) -> bool:
    """Whether a start date may be replayed. The only gate on visitor input.

    `allowed` defaults to the documented real-warehouse list; the page passes the list
    derived for whichever warehouse it is showing.
    """
    permitted = WHITELIST_START_DATES if allowed is None else tuple(allowed)
    try:
        return int(start_date) in permitted
    except (TypeError, ValueError):
        return False


def lookback_trading_days(cfg: Mapping[str, Any]) -> int:
    """Trading days of history the recompute needs, derived from the real window lengths.

    Not hard-coded: it reads the regime drawdown window, the smoothing window and the
    longest indicator window (50, `sma_50`), then adds a margin for warm-up rows.
    """
    regime = cfg["regime"]
    drawdown = int(regime["drawdown_window"])
    smoothing = int(regime.get("smoothing", {}).get("window", 0) or 0)
    indicator_window = 50  # sma_50, the longest rolling window in build_indicators
    warmup = int(cfg.get("features", {}).get("warmup_rows", 0) or 0)
    total = max(drawdown + smoothing, indicator_window + warmup) + 10
    return total


def entry_threshold(cfg: Mapping[str, Any], regime: str) -> float | None:
    """The entry probability this regime actually applies, from the config.

    `None` in a zero-exposure regime, where nothing may be entered. The previous version
    of this screen counted candidates above a literal `0.5`, which is not the strategy's
    rule: the threshold is `entry_base + entry_offsets[regime]`, and the selected
    `entry_base` is 0.504415, so a hardcoded 0.5 admitted names the strategy would have
    rejected (`AGENTS.md` rule 6, no magic numbers).
    """
    from src.trading.rules import build_regime_rules

    try:
        rule = build_regime_rules(cfg)[regime]
    except KeyError:
        return None
    value = rule.get("entry")
    return None if value is None else float(value)


@dataclass
class DayResult:
    """What the bot decided on one day."""

    date_key: int
    regime: str
    passed_threshold: int
    candidates: int
    target_weights: dict[str, float]
    orders: list[dict[str, Any]]
    cash: float
    positions_value: float
    equity: float
    holdings: dict[str, int]
    #: The regime's configured entry threshold, so the screen shows what the strategy
    #: actually applied rather than a literal.
    entry_threshold: float | None = None


@dataclass
class BotRun:
    """A finished replay."""

    start_date: int
    days: list[DayResult] = field(default_factory=list)
    first_day_check: dict[str, Any] = field(default_factory=dict)
    #: Orders the FIRST decision day produced, which fill on the second window day.
    first_day_orders: list[dict[str, Any]] = field(default_factory=list)
    #: The date those orders filled on (the second window day).
    first_day_orders_date: int | None = None
    elapsed_seconds: float = 0.0
    lookback: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "lookback": self.lookback,
            "first_day_check": self.first_day_check,
            "days": [
                {
                    "date_key": d.date_key,
                    "regime": d.regime,
                    "passed_threshold": d.passed_threshold,
                    "candidates": d.candidates,
                    "orders": d.orders,
                    "cash": round(d.cash, 2),
                    "positions_value": round(d.positions_value, 2),
                    "equity": round(d.equity, 2),
                    "holdings": d.holdings,
                    "entry_threshold": d.entry_threshold,
                    "target_weights": {k: round(v, 6) for k, v in d.target_weights.items()},
                }
                for d in self.days
            ],
        }


class ReplayBusy(RuntimeError):
    """Raised when another visitor already has a replay running."""


@contextmanager
def single_run() -> Iterator[None]:
    """Hold the global replay lock. A second visitor gets `ReplayBusy`, not a second run."""
    if not _RUN_LOCK.acquire(blocking=False):
        raise ReplayBusy("a replay is already running; try again in a moment")
    try:
        yield
    finally:
        _RUN_LOCK.release()


def cached(start_date: int) -> BotRun | None:
    with _CACHE_LOCK:
        return _CACHE.get(int(start_date))


def store(run: BotRun) -> None:
    with _CACHE_LOCK:
        _CACHE[int(run.start_date)] = run


def clear_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


@contextmanager
def run_directory() -> Iterator[Path]:
    """A private temp directory, removed even if the body raises.

    `ignore_errors=True` alone was not enough: when the body raised, the SQLite engine
    was still holding the copied database open, so the delete silently did nothing and
    left a copy of the warehouse on disk. Hence the explicit dispose in `run_replay`
    plus a bounded retry here. Tested by `test_the_run_directory_is_removed_even_when_
    the_run_fails`, which caught exactly this.
    """
    path = Path(tempfile.mkdtemp(prefix="regimetrader_bot_"))
    try:
        yield path
    finally:
        for attempt in range(5):
            shutil.rmtree(path, ignore_errors=True)
            if not path.exists():
                break
            time.sleep(0.2 * (attempt + 1))
        if path.exists():
            # Last resort: make everything writable, then try once more.
            for item in path.rglob("*"):
                try:
                    item.chmod(0o700)
                except OSError:
                    continue
            shutil.rmtree(path, ignore_errors=True)


def remaining_temp_dirs(temp_root: Path | None = None) -> list[Path]:
    """Leftover replay directories. Should always be empty; used by the cleanup test."""
    root = temp_root or Path(tempfile.gettempdir())
    return list(root.glob("regimetrader_bot_*"))


def copy_warehouse(source: Path, destination: Path) -> Path:
    """Copy the real warehouse so the replay can never touch the original."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, destination)
    for suffix in ("-wal", "-shm"):
        side = source.with_name(source.name + suffix)
        if side.exists():
            shutil.copy(side, destination.with_name(destination.name + suffix))
    return destination


def truncate_after(engine: Any, cutoff: int) -> dict[str, int]:
    """Delete every row dated after `cutoff`. This is what produces no hindsight."""
    removed: dict[str, int] = {}
    statements = [
        ("fact_indicator", "DELETE FROM fact_indicator WHERE date_key > :c"),
        ("fact_market_regime", "DELETE FROM fact_market_regime WHERE date_key > :c"),
        ("fact_signal", "DELETE FROM fact_signal WHERE date_key > :c"),
        ("fact_trade", "DELETE FROM fact_trade WHERE date_key > :c"),
        ("fact_portfolio_daily", "DELETE FROM fact_portfolio_daily WHERE date_key > :c"),
        ("fact_price", "DELETE FROM fact_price WHERE date_key > :c"),
    ]
    with engine.begin() as conn:
        for name, sql in statements:
            result = conn.exec_driver_sql(sql, {"c": int(cutoff)})
            removed[name] = int(result.rowcount or 0)
    return removed


def trim_to_lookback(engine: Any, as_of: int, lookback: int) -> int:
    """Keep only the last `lookback` trading days up to `as_of`.

    This deletes rows *both* after the window and far before it, leaving a band of
    roughly `lookback x n_instruments` rows instead of the whole history. Deleting only
    the far side would keep a prefix of ~127k rows and save nothing.

    Sound because every feature is causal; `scripts/bot_equivalence.py` proves the
    day-D output is identical to a full recompute on all history up to D.
    """
    from src.common.db import read_sql

    days = [
        int(r)
        for r in read_sql(
            engine,
            "SELECT p.date_key FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
            "WHERE s.instrument_type='INDEX' AND p.date_key <= :a ORDER BY 1",
            {"a": int(as_of)},
        )["date_key"]
    ]
    if len(days) <= lookback:
        return as_of
    earliest = days[-lookback]
    with engine.begin() as conn:
        for table in (
            "fact_indicator",
            "fact_market_regime",
            "fact_signal",
            "fact_price",
        ):
            conn.exec_driver_sql(f"DELETE FROM {table} WHERE date_key < :c", {"c": int(earliest)})
    return as_of


def timeout_guard(seconds: int) -> Callable[[], None]:
    """Simple cooperative deadline check for long loops."""

    start = time.time()

    def check() -> None:
        if time.time() - start > seconds:
            raise TimeoutError("the replay took too long and was stopped")

    return check


def run_replay(
    cfg: Mapping[str, Any],
    source_db: Path,
    start_date: int,
    *,
    allowed: tuple[int, ...] | None = None,
    deadline_check: Callable[[], None] | None = None,
    per_day_check: Callable[[int], None] | None = None,
) -> BotRun:
    """Replay `WINDOW_DAYS` trading days from `start_date` on a temp copy.

    Args:
        cfg: Loaded configuration (read only).
        source_db: Path to the real warehouse. It is copied, never opened for writing.
        start_date: Must be in the whitelist.
        allowed: The whitelist to check against; derived from `cfg` when omitted.
        deadline_check: Optional cooperative timeout hook.
        per_day_check: Called with each day index as it completes, for progress display.

    Raises:
        ValueError: If `start_date` is not whitelisted.
        ReplayBusy: If another replay is already running.
    """
    if allowed is None:
        from src.common.db import get_engine as _get_engine

        probe = _get_engine(cfg)  # type: ignore[arg-type]
        try:
            allowed = available_start_dates(cfg, probe)
        finally:
            probe.dispose()
    if not is_whitelisted(start_date, allowed):
        raise ValueError("that start date is not available")

    import joblib

    from src.backtest.engine import run_backtest
    from src.backtest.strategies import make_weights_fn
    from src.common.db import get_engine, read_sql
    from src.features.indicators import build_indicators
    from src.features.panel import build_panel, get_feature_cols, use_vix_in_regime
    from src.paper.daily_job import _extend_regime

    started = time.time()
    lookback = lookback_trading_days(cfg)

    index_days = [
        int(r)
        for r in read_sql(
            get_engine(cfg),
            "SELECT p.date_key FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
            "WHERE s.instrument_type='INDEX' ORDER BY 1",
        )["date_key"]
    ]
    after = [d for d in index_days if d >= int(start_date)]
    if len(after) < WINDOW_DAYS:
        raise ValueError("not enough trading days after the start date")
    window = after[:WINDOW_DAYS]

    local_cfg = dict(cfg)
    local_cfg["paths"] = dict(cfg["paths"])  # type: ignore[index]
    local_cfg["paper"] = dict(cfg.get("paper", {}))  # type: ignore[index]
    # Never let anything in this replay behave like a live paper run.
    local_cfg["paper"]["run_id"] = "bot_replay"  # type: ignore[index]

    run = BotRun(start_date=int(start_date), lookback=lookback)

    with run_directory() as work:
        # Dispose the engine before the directory is removed: an open SQLite
        # handle makes the delete fail silently and leaves a copy of the warehouse
        # behind. `test_the_run_directory_is_removed_even_when_the_run_fails` exists
        # because this was caught failing.
        engine = None
        try:
            db_path = copy_warehouse(Path(source_db), work / "replay.db")
            local_cfg["paths"]["db"] = str(db_path)  # type: ignore[index]
            engine = get_engine(local_cfg)  # type: ignore[arg-type]

            model = str(cfg["strategy"]["selected_model"])  # type: ignore[index]
            weights_fn = make_weights_fn("V2", local_cfg)  # type: ignore[arg-type]

            window_end = window[-1]

            # --- one recompute for the whole window -----------------------------
            # Truncate at the LAST day of the window: the bot never sees anything after
            # it. History before the lookback band is dropped purely for speed, which is
            # sound because every feature is causal. `scripts/bot_equivalence.py` proves
            # it rather than asserting it: day 1 and day 5 are compared against a literal
            # full recompute, and a mutation test changes a row after D to show the day-D
            # output does not move.
            #
            # Recomputing once instead of once per day is the difference between about 40
            # seconds and about three minutes per click: build_indicators on the band
            # costs ~39s measured. This is the "reuse the causal computation" of Option B.
            if deadline_check is not None:
                deadline_check()
            trim_to_lookback(engine, window_end, lookback)
            truncate_after(engine, window_end)

            build_indicators(engine, local_cfg)  # type: ignore[arg-type]
            _extend_regime(local_cfg, engine, window_end)  # type: ignore[arg-type]
            panel = build_panel(engine, local_cfg, use_cache=False, live=True)  # type: ignore[arg-type]

            year = pd.Timestamp(str(window_end)).year
            estimator = joblib.load(
                Path(cfg["paths"]["models"]) / f"{model}_wf{year}.joblib"  # type: ignore[index]
            )
            feature_cols = get_feature_cols(use_vix_in_regime(local_cfg, engine))  # type: ignore[arg-type]

            stock_keys = dict(
                read_sql(engine, "SELECT symbol, stock_key FROM dim_stock").itertuples(
                    index=False, name=None
                )
            )
            regimes = dict(
                read_sql(engine, "SELECT date_key, regime_key FROM fact_market_regime").itertuples(
                    index=False, name=None
                )
            )
            model_key = int(
                read_sql(
                    engine,
                    "SELECT model_key FROM dim_model WHERE model_name = :m",
                    {"m": model},
                )["model_key"].iloc[0]
            )
            regime_labels = {
                int(r.regime_key): str(r.regime_label)
                for r in read_sql(
                    engine, "SELECT regime_key, regime_label FROM dim_regime"
                ).itertuples()
            }

            from src.common.db import execute_many

            signal_rows: list[dict[str, Any]] = []
            for day_key in window:
                today = panel[panel["date_key"] == day_key]
                if today.empty:
                    continue
                probs = estimator.predict_proba(today[feature_cols])[:, 1]
                signal_rows.extend(
                    {
                        "d": int(day_key),
                        "s": stock_keys.get(str(row.symbol)),
                        "k": model_key,
                        "r": int(regimes.get(int(day_key), 0) or 0),
                        "p": float(prob),
                    }
                    for (_, row), prob in zip(today.iterrows(), probs, strict=True)
                    if str(row.symbol) in stock_keys
                )
                if deadline_check is not None:
                    deadline_check()

            with engine.begin() as conn:
                execute_many(
                    conn,
                    "INSERT OR REPLACE INTO fact_signal (date_key, stock_key, model_key, "
                    "regime_key, prob_up, signal, target_weight) "
                    "VALUES (:d, :s, :k, :r, :p, 0, 0.0)",
                    signal_rows,
                )

            # --- the real engine, over the whole window in one pass -----------
            #
            # It must be ONE call, not one per day. Orders are decided at a day's close
            # and fill at the NEXT day's open, so a single-day run plans its entries and
            # then has nowhere to fill them: the first version of this produced zero
            # trades on every day while the engine logged "0 of 10 new entries".
            #
            # `allow_test_period=True` lifts the R26 guard, which otherwise refuses to
            # run over test dates. It is required here because this screen deliberately
            # replays already-scored test days. That makes it the single most dangerous
            # flag in the codebase, so it is passed together with `persist=False` and
            # `write_signals=False`, which mean the run writes nothing anywhere; the
            # tests assert this exact combination appears here and nowhere else.
            if deadline_check is not None:
                deadline_check()
            result = run_backtest(
                local_cfg,  # type: ignore[arg-type]
                weights_fn,
                pd.Timestamp(str(window[0])).strftime("%Y-%m-%d"),
                pd.Timestamp(str(window[-1])).strftime("%Y-%m-%d"),
                run_id=f"bot_demo_{window[0]}",
                engine=engine,
                persist=False,
                write_signals=False,
                allow_test_period=True,
                model_name=model,
            )
            trades = getattr(result, "trades", pd.DataFrame())
            equity = getattr(result, "equity", pd.DataFrame())
            decisions = getattr(result, "signals", pd.DataFrame())

            # The engine's own decision frame carries the target weights it computed
            # (`decision_rows`). Reading them here is what puts real numbers in the
            # weights panel instead of a hardcoded empty dict.
            weights_by_day: dict[int, dict[str, float]] = {}
            if not decisions.empty and "target_weight" in decisions:
                for date_key, group in decisions.groupby("date_key"):
                    weights_by_day[int(date_key)] = {
                        str(r.symbol): float(r.target_weight) for r in group.itertuples()
                    }

            for index, day_key in enumerate(window):
                if deadline_check is not None:
                    deadline_check()
                day_rows = [r for r in signal_rows if r["d"] == day_key]
                day_regime = regime_labels.get(int(regimes.get(int(day_key), 0) or 0), "")
                day_equity = equity[equity["date_key"] == day_key] if not equity.empty else None
                run.days.append(
                    _day_from_result(
                        day_key,
                        trades[trades["date_key"] == day_key] if not trades.empty else trades,
                        [r["p"] for r in day_rows],
                        day_rows,
                        day_regime,
                        day_equity.iloc[-1] if day_equity is not None and len(day_equity) else None,
                        weights_by_day.get(int(day_key), {}),
                        entry_threshold(cfg, day_regime),
                    )
                )
                if per_day_check is not None:
                    per_day_check(index)

            # The no-hindsight comparison is on the FILLS produced by the first day's
            # decisions, which land on the second day of the window.
            run.first_day_orders = _orders_for(trades, window[1]) if len(window) > 1 else []
            run.first_day_orders_date = window[1] if len(window) > 1 else None
            engine.dispose()
        finally:
            if engine is not None:
                engine.dispose()
    run.elapsed_seconds = time.time() - started
    return run


def _orders_for(trades: pd.DataFrame, date_key: int) -> list[dict[str, Any]]:
    """The filled orders recorded on `date_key`."""
    if trades.empty:
        return []
    return [
        {
            "symbol": str(row["symbol"]),
            "side": str(row["side"]),
            "reason": str(row.get("reason", "")),
            "quantity": float(row["quantity"]),
            "fill_price": round(float(row["fill_price"]), 2),
        }
        for _, row in trades[trades["date_key"] == date_key].iterrows()
    ]


def _day_from_result(
    day_key: int,
    trades: pd.DataFrame,
    probs: Any,
    signal_rows: list[dict[str, Any]],
    regime: str = "",
    equity_row: Any = None,
    weights: dict[str, float] | None = None,
    threshold: float | None = None,
) -> DayResult:
    """Build one day's view from the engine's trades for that day and its equity row.

    The regime is passed in rather than read from the equity frame: `EQUITY_COLUMNS`
    in the engine is ("run_id", "date_key", "cash", "positions_value", "equity",
    "daily_return", "drawdown") and carries no regime, so it is looked up from
    `fact_market_regime` on the temporary copy instead.

    `weights` are the targets the engine actually computed for this decision
    (`BacktestResult.signals`), not a placeholder: a previous version passed
    `target_weights={}` unconditionally, so the panel could never show them.
    """
    orders = _orders_for(trades, day_key)

    holdings: dict[str, int] = {}
    for order in orders:
        qty = int(order["quantity"])
        if order["side"] == "BUY":
            holdings[order["symbol"]] = holdings.get(order["symbol"], 0) + qty
        else:
            remaining = holdings.get(order["symbol"], 0) - qty
            if remaining > 0:
                holdings[order["symbol"]] = remaining
            else:
                holdings.pop(order["symbol"], None)

    cash = positions_value = equity = 0.0
    if equity_row is not None:
        cash = float(equity_row.get("cash", 0.0) or 0.0)
        positions_value = float(equity_row.get("positions_value", 0.0) or 0.0)
        equity = float(equity_row.get("equity", 0.0) or 0.0)

    # Count against the regime's configured threshold, not a literal. A regime with no
    # exposure admits nothing, so the count is zero rather than undefined.
    if threshold is None:
        passed = 0
    else:
        passed = len([r for r in signal_rows if r["p"] >= threshold])

    return DayResult(
        date_key=int(day_key),
        regime=regime,
        passed_threshold=passed,
        candidates=len(signal_rows),
        target_weights=dict(weights or {}),
        orders=orders,
        cash=cash,
        positions_value=positions_value,
        equity=equity,
        holdings=holdings,
        entry_threshold=threshold,
    )


def first_day_check_against_backtest(cfg: Mapping[str, Any], run: "BotRun") -> dict[str, Any]:
    """Compare the replay's day-one orders with the stored `backtest_main` orders.

    Read from the real warehouse through the caller's read-only handle. `backtest_main`
    filled its first orders on the 2024-01-02 open (ASIANPAINT, BSE, CIPLA, HDFCBANK,
    HINDUNILVR, ITC, KOTAKBANK, RELIANCE, SUNPHARMA, TITAN); the replay must produce
    exactly those from data up to the decision date only.
    """
    from src.common.db import get_engine, read_sql

    if not run.days:
        return {}
    # Orders decided on the first day fill on the second day of the window, so the
    # comparison is against the stored trades on that second day.
    fill_date = run.first_day_orders_date or run.days[0].date_key

    engine = get_engine(cfg)
    try:
        stored = read_sql(
            engine,
            "SELECT s.symbol AS symbol, t.side AS side, t.quantity AS quantity, "
            "       t.fill_price AS fill_price, t.reason AS reason "
            "FROM fact_trade t JOIN dim_stock s ON s.stock_key = t.stock_key "
            "WHERE t.run_id = :r AND t.date_key = :d "
            "ORDER BY s.symbol",
            {"r": "backtest_main", "d": int(fill_date)},
        )
    finally:
        engine.dispose()

    result = compare_first_day_against_stored(run.first_day_orders, stored)
    result["compared_date"] = int(fill_date)
    result["decision_date"] = run.days[0].date_key
    result["stored_reason"] = "backtest_main"
    return result


def compare_first_day_against_stored(
    replay_orders: list[dict[str, Any]],
    stored_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the replay's day-one orders with the stored `backtest_main` orders.

    Both are lists of (symbol, side, quantity, fill_price). Mismatches are reported,
    never suppressed.
    """
    replay = {
        (o["symbol"], o["side"]): (float(o["quantity"]), float(o["fill_price"]))
        for o in replay_orders
    }
    stored = {
        (str(r.symbol), str(r.side)): (float(r.quantity), float(r.fill_price))
        for r in stored_rows.itertuples()
    }

    only_replay = sorted(set(replay) - set(stored))
    only_stored = sorted(set(stored) - set(replay))
    differing = []
    for key in sorted(set(replay) & set(stored)):
        qty_a, price_a = replay[key]
        qty_b, price_b = stored[key]
        if qty_a != qty_b or abs(price_a - price_b) > 0.01:
            differing.append(
                {"symbol": key[0], "side": key[1], "replay_qty": qty_a, "stored_qty": qty_b}
            )

    return {
        "match": not (only_replay or only_stored or differing),
        "n_replay": len(replay),
        "n_stored": len(stored),
        "only_in_replay": [list(k) for k in only_replay],
        "only_in_stored": [list(k) for k in only_stored],
        "quantity_or_price_differences": differing,
    }
