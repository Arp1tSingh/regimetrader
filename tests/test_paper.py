"""Paper trading: warehouse-derived state, guards and crash recovery.

The property that matters most is that a job killed mid-date and restarted rebuilds
**identical** state. Everything below exists to prove that, or to prove that the guards
that protect the test period still hold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.common.log import get_logger
from src.paper.alerts import alert, is_stale, stale_days
from src.paper.daily_job import (
    guards,
    latest_trading_date,
    signal_dates_allowed,
    trading_dates,
)
from src.paper.store import (
    cooldown_symbols,
    last_processed,
    load_state,
    pending_targets,
    processed_dates,
)

# --------------------------------------------------------------- guards (R26)


def test_paper_live_is_refused_without_the_test_lock(cfg_smoke: dict[str, Any]) -> None:
    """R26: the strategy must be frozen and tested before live trading starts."""
    results = Path(cfg_smoke["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    for name in ("TEST_RUN_LOCK.json",):
        (results / name).unlink(missing_ok=True)
    frozen = Path(cfg_smoke["paths"]["frozen_config"])
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text("strategy:\n  selected_model: logreg\n", encoding="utf-8")

    message = guards(cfg_smoke, "paper_live", None)
    assert message is not None
    assert "TEST_RUN_LOCK.json" in message


def test_paper_live_is_refused_without_a_frozen_config(cfg_smoke: dict[str, Any]) -> None:
    results = Path(cfg_smoke["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "TEST_RUN_LOCK.json").write_text(
        json.dumps({"run_count": 1, "data_end_date": "20221230"}), encoding="utf-8"
    )
    Path(cfg_smoke["paths"]["frozen_config"]).unlink(missing_ok=True)

    message = guards(cfg_smoke, "paper_live", None)
    assert message is not None
    assert "frozen config" in message


def test_a_replay_is_not_gated_on_the_lock(cfg_smoke: dict[str, Any]) -> None:
    """Replays are research runs, not live money; the lock does not apply."""
    results = Path(cfg_smoke["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "TEST_RUN_LOCK.json").unlink(missing_ok=True)
    assert guards(cfg_smoke, "paper_replay", None) is None


def test_only_paper_live_may_write_fact_signal(cfg_smoke: dict[str, Any]) -> None:
    """R26: a replay never touches `fact_signal`."""
    results = Path(cfg_smoke["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "TEST_RUN_LOCK.json").write_text(
        json.dumps({"run_count": 1, "data_end_date": "20221230"}), encoding="utf-8"
    )
    assert signal_dates_allowed(cfg_smoke, "paper_replay", 20230103) is False
    assert signal_dates_allowed(cfg_smoke, "paper_live", 20221230) is False, (
        "dates at or before the lock's data_end_date must not be written"
    )
    assert signal_dates_allowed(cfg_smoke, "paper_live", 20230103) is True


@pytest.mark.slow
def test_the_daily_job_never_writes_fact_signal_up_to_the_lock_date(
    signals_ready: tuple,
) -> None:
    """Regression for the Phase 7 bug: paper_live wrote fact_signal at data_end_date.

    On the first real run the job extended indicators, regime and probabilities to the
    latest trading date, which happened to be exactly the lock's `data_end_date` — the
    one date R26 forbids a live run from writing. This runs the job against that
    situation and asserts the row was not created.
    """
    import src.paper.daily_job as job
    from src.common.db import read_sql

    cfg_obj, engine = signals_ready
    lock_date = 20230103
    results = Path(cfg_obj["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "TEST_RUN_LOCK.json").write_text(
        json.dumps({"run_count": 1, "data_end_date": str(lock_date)}), encoding="utf-8"
    )
    cfg_obj = {
        **cfg_obj,
        "paper": {**cfg_obj["paper"], "run_id": "paper_live"},
        "strategy": {**cfg_obj["strategy"], "selected_model": "tree"},
    }

    def max_signal_date() -> int:
        return int(read_sql(engine, "SELECT MAX(date_key) AS m FROM fact_signal")["m"].iloc[0])

    before = max_signal_date()

    # A date on or before the lock must be refused outright.
    assert signal_dates_allowed(cfg_obj, "paper_live", lock_date) is False

    outcome = job.run(cfg_obj, run_id="paper_live", no_etl=True)
    assert outcome["processed"] == [], "no date may be processed at or before the lock"
    assert max_signal_date() == before, (
        "paper_live must not extend fact_signal to the lock's data_end_date"
    )

    # Even when asked to process that date explicitly, nothing is written.
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM fact_portfolio_daily WHERE run_id = 'paper_live'")
    job.run(cfg_obj, run_id="paper_live", from_key=lock_date, to_key=lock_date, no_etl=True)
    assert max_signal_date() == before, "asking for the locked date must not cause a write either"

    (results / "TEST_RUN_LOCK.json").unlink(missing_ok=True)
    with engine.begin() as conn:
        for table in ("fact_trade", "fact_portfolio_daily", "pending_target"):
            conn.exec_driver_sql("DELETE FROM {} WHERE run_id='paper_live'".format(table))


def test_ensure_inputs_refuses_a_date_it_may_not_write(
    signals_ready: tuple,
) -> None:
    """`ensure_inputs` is the step that writes signals, so it must check R26 first."""
    import src.paper.daily_job as job

    cfg_obj, engine = signals_ready
    results = Path(cfg_obj["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    (results / "TEST_RUN_LOCK.json").write_text(
        json.dumps({"run_count": 1, "data_end_date": "20991231"}), encoding="utf-8"
    )
    # A lock far in the future makes every date disallowed.
    assert signal_dates_allowed(cfg_obj, "paper_live", 20230103) is False

    source = Path(job.__file__).read_text(encoding="utf-8")
    assert "signal_dates_allowed" in source, (
        "ensure_inputs must consult signal_dates_allowed before writing fact_signal"
    )
    (results / "TEST_RUN_LOCK.json").unlink(missing_ok=True)


# --------------------------------------------------------------- store round trip


def test_store_rebuilds_positions_from_fact_trade(signals_ready: tuple) -> None:
    """T7.1: the derived holdings equal the engine's own state after a backtest."""
    from src.backtest.engine import run_backtest
    from src.backtest.strategies import make_weights_fn as build
    from src.common.db import read_sql

    cfg_obj, engine = signals_ready
    cfg_obj = {**cfg_obj, "splits": {**cfg_obj["splits"], "test_start": "2030-01-01"}}
    start = str(cfg_obj["splits"]["validation_start"])
    end = str(cfg_obj["splits"]["validation_end"])

    result = run_backtest(
        cfg_obj,
        build("V2", cfg_obj),
        start,
        end,
        "store_probe",
        engine=engine,
        model_name="tree",
        persist=True,
    )

    state = load_state(engine, "store_probe")
    assert state.is_started

    # Expected open positions: net quantity per symbol across the trade log.
    net: dict[str, float] = {}
    for row in result.trades.itertuples(index=False):
        delta = float(row.quantity) * (1 if str(row.side) == "BUY" else -1)
        net[str(row.symbol)] = net.get(str(row.symbol), 0.0) + delta
    expected_open = {s: q for s, q in net.items() if abs(q) > 1e-9}

    assert sorted(state.held) == sorted(expected_open), (
        "the derived holdings must equal the symbols the engine left open"
    )
    for symbol, quantity in expected_open.items():
        assert float(state.held[symbol].quantity) == pytest.approx(quantity), symbol

    stored = read_sql(
        engine,
        "SELECT COUNT(*) AS n FROM fact_portfolio_daily WHERE run_id = 'store_probe'",
    )
    assert int(stored["n"].iloc[0]) == len(result.equity)

    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM fact_trade WHERE run_id = 'store_probe'")
        conn.exec_driver_sql("DELETE FROM fact_portfolio_daily WHERE run_id = 'store_probe'")


def test_a_run_with_no_rows_has_never_started(signals_ready: tuple) -> None:
    _cfg, engine = signals_ready
    assert last_processed(engine, "never_ran") is None
    assert processed_dates(engine, "never_ran") == []
    assert load_state(engine, "never_ran").is_started is False


def _a_date(engine: Any) -> int:
    """A real `date_key` that satisfies the foreign keys to `dim_date`."""
    from src.common.db import read_sql

    frame = read_sql(
        engine, "SELECT date_key AS d FROM dim_date WHERE date_key >= 20180102 ORDER BY date_key"
    )
    assert not frame.empty, "the fixture must have dim_date rows"
    return int(frame["d"].iloc[0])


def test_pending_targets_round_trip(signals_ready: tuple) -> None:
    from sqlalchemy import text

    from src.common.db import read_sql
    from src.paper.store import store_pending_targets

    cfg_obj, engine = signals_ready
    keys = dict(
        read_sql(engine, "SELECT symbol, stock_key FROM dim_stock").itertuples(
            index=False, name=None
        )
    )
    day = _a_date(engine)
    symbols = sorted(keys)[:3]
    with engine.begin() as conn:
        store_pending_targets(
            conn, "paper_probe", day, {s: 0.1 for s in symbols}, {s: 0.6 for s in symbols}, keys
        )
    stored = pending_targets(engine, "paper_probe", day)
    assert sorted(stored) == symbols
    assert all(v == pytest.approx(0.1) for v in stored.values())

    # Re-storing must replace, not accumulate (crash-and-retry safety).
    with engine.begin() as conn:
        store_pending_targets(conn, "paper_probe", day, {symbols[0]: 0.2}, {}, keys)
    again = pending_targets(engine, "paper_probe", day)
    assert list(again) == [symbols[0]], "stale targets must not survive a re-store"

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM pending_target WHERE run_id = 'paper_probe'"))


def test_empty_pending_set_is_a_real_decision_to_be_flat(signals_ready: tuple) -> None:
    """The distinction R25 exists for: no rows means "go to cash", not "crashed"."""
    _cfg, engine = signals_ready
    assert pending_targets(engine, "no_such_run", _a_date(engine)) == {}


# --------------------------------------------------------------- crash recovery


def _signal_dates(engine: Any, limit: int = 4) -> list[int]:
    """Trading dates that actually carry stored probabilities, ascending.

    The paper job needs `prob_up` for a date to decide, so any test that processes
    dates must pick dates the trainer actually wrote.
    """
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT DISTINCT date_key AS d FROM fact_signal ORDER BY date_key LIMIT :n",
        {"n": int(limit)},
    )
    return [int(x) for x in frame["d"]]


@pytest.mark.slow
def test_restart_after_a_mid_date_crash_reproduces_the_same_state(
    signals_ready: tuple,
) -> None:
    """Kill the job part-way through a date and prove a restart is identical.

    The simulated crash happens inside the date transaction, after the portfolio row
    has been written but before it commits. Restarting must redo the date from the
    warehouse and produce exactly the same holdings, cash, equity and trades, with no
    duplicates and no silent move to cash.
    """
    import src.paper.daily_job as job
    from src.common.db import read_sql

    cfg_obj, engine = signals_ready
    cfg_obj = {
        **cfg_obj,
        "splits": {**cfg_obj["splits"], "test_start": "2030-01-01"},
        "paper": {**cfg_obj["paper"], "run_id": "paper_crash"},
        "strategy": {**cfg_obj["strategy"], "selected_model": "tree"},
    }
    run_id = "paper_crash"

    dates = _signal_dates(engine, limit=3)
    assert len(dates) >= 3, "need at least three signal dates for the test"

    log = get_logger("test", cfg_obj)
    baseline = [job.process_date(cfg_obj, engine, run_id, key, log) for key in dates[:3]]

    def portfolio_rows(identifier: str) -> list[tuple[int, float, float, float]]:
        """(date, cash, equity, drawdown) for every committed date, ascending."""
        frame = read_sql(
            engine,
            "SELECT date_key AS d, cash AS c, equity AS e, drawdown AS dd "
            "FROM fact_portfolio_daily WHERE run_id = :r ORDER BY date_key",
            {"r": identifier},
        )
        return [
            (int(r.d), float(r.c), float(r.e), float(r.dd)) for r in frame.itertuples(index=False)
        ]

    snapshot = {
        "state": {
            "cash": load_state(engine, run_id).cash,
            "equity": load_state(engine, run_id).equity,
            "held": {s: float(p.quantity) for s, p in load_state(engine, run_id).held.items()},
        },
        "portfolio": portfolio_rows(run_id),
        "trades": int(
            read_sql(engine, f"SELECT COUNT(*) n FROM fact_trade WHERE run_id = '{run_id}'")[
                "n"
            ].iloc[0]
        ),
        "pending": pending_targets(engine, run_id, dates[-1]),
    }

    # Simulate a crash during the last date: roll that date back to nothing.
    last = dates[2]
    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"DELETE FROM fact_portfolio_daily WHERE run_id = '{run_id}' AND date_key = {last}"
        )
        conn.exec_driver_sql(
            f"DELETE FROM fact_trade WHERE run_id = '{run_id}' AND date_key = {last}"
        )
        conn.exec_driver_sql(
            f"DELETE FROM pending_target WHERE run_id = '{run_id}' AND decision_date_key = {last}"
        )

    # Nothing may have been "decided" in the gap: holdings survive because they were
    # committed on an earlier date, not because of the rolled-back one. The rolled-back
    # date's effect must be fully undone, so state reverts to the previous committed row.
    mid = load_state(engine, run_id)
    prior_row = snapshot["portfolio"][-2]  # (date, cash, equity, drawdown)
    assert mid.last_date_key == prior_row[0], (
        "the last committed date must fall back to the one before the crash"
    )
    assert mid.cash == pytest.approx(prior_row[1]), (
        "a rolled-back date must not leave cash from its own trades behind"
    )
    assert mid.equity == pytest.approx(prior_row[2])
    assert mid.cash != pytest.approx(snapshot["state"]["cash"]), (
        "otherwise this test would prove nothing: the rollback changed nothing"
    )

    # Restart: redo the interrupted date.
    job.process_date(cfg_obj, engine, run_id, last, log)

    after = load_state(engine, run_id)
    assert after.cash == pytest.approx(snapshot["state"]["cash"], rel=1e-12)
    assert after.equity == pytest.approx(snapshot["state"]["equity"], rel=1e-12), (
        "the restart must land on the same equity, not merely the same cash"
    )
    assert {s: float(p.quantity) for s, p in after.held.items()} == pytest.approx(
        snapshot["state"]["held"]
    ), "the restart must reproduce the same holdings"
    assert portfolio_rows(run_id) == pytest.approx(snapshot["portfolio"]), (
        "every committed date's cash, equity and drawdown must be identical"
    )
    trades_after = int(
        read_sql(engine, f"SELECT COUNT(*) n FROM fact_trade WHERE run_id = '{run_id}'")["n"].iloc[
            0
        ]
    )
    assert trades_after == snapshot["trades"], "the restart must not duplicate trades"
    assert pending_targets(engine, run_id, last) == pytest.approx(snapshot["pending"]), (
        "the restart must reach the same decision, not go to cash"
    )

    # And the whole three-date sequence replays identically from scratch.
    run_id2 = "paper_crash2"
    cfg2 = {**cfg_obj, "paper": {**cfg_obj["paper"], "run_id": run_id2}}
    repeat = [job.process_date(cfg2, engine, run_id2, key, log) for key in dates[:3]]
    assert [r["equity"] for r in repeat] == [r["equity"] for r in baseline], (
        "a fresh run over the same dates must reproduce the same equity"
    )
    assert [r["held"] for r in repeat] == [r["held"] for r in baseline]
    assert portfolio_rows(run_id2) == pytest.approx(snapshot["portfolio"]), (
        "a fresh run over the same dates must reproduce the same equity path"
    )

    # No duplicates within a single run.
    dupes = read_sql(
        engine,
        f"SELECT t.date_key AS date_key, s.symbol AS symbol, COUNT(*) AS n "
        f"FROM fact_trade t JOIN dim_stock s ON s.stock_key = t.stock_key "
        f"WHERE t.run_id = '{run_id}' GROUP BY t.date_key, s.symbol HAVING n > 1",
    )
    assert dupes.empty, f"duplicate trades found: {dupes.to_dict('records')}"

    for name in (run_id, run_id2):
        with engine.begin() as conn:
            for table in ("fact_trade", "fact_portfolio_daily", "pending_target"):
                conn.exec_driver_sql(f"DELETE FROM {table} WHERE run_id = '{name}'")


@pytest.mark.slow
def test_cash_survives_a_date_with_no_trades(signals_ready: tuple) -> None:
    """Regression: cash must not be lost when a day places no orders.

    Cash used to be rebuilt by replaying `fact_trade`, so a run that held nothing came
    back the next day with zero cash and an equity of 0.
    """
    import src.paper.daily_job as job
    from src.common.db import read_sql

    cfg_obj, engine = signals_ready
    cfg_obj = {
        **cfg_obj,
        "paper": {**cfg_obj["paper"], "run_id": "paper_cash"},
        "strategy": {**cfg_obj["strategy"], "selected_model": "tree"},
    }
    initial = float(cfg_obj["backtest"]["initial_capital"])
    dates = _signal_dates(engine, limit=3)
    assert len(dates) >= 2
    log = get_logger("test", cfg_obj)

    for key in dates:
        job.process_date(cfg_obj, engine, "paper_cash", key, log)

    state = load_state(engine, "paper_cash")
    assert state.equity > 0, "equity must never collapse to zero"
    assert state.cash > 0, "cash must survive a day with no orders"

    rows = read_sql(
        engine,
        "SELECT equity FROM fact_portfolio_daily WHERE run_id = 'paper_cash' ORDER BY date_key",
    )
    assert (rows["equity"] > 0).all(), "every committed date must hold a positive equity"
    assert rows["equity"].iloc[0] == pytest.approx(initial, rel=0.5), (
        "the first date must start from the configured initial capital"
    )

    with engine.begin() as conn:
        for table in ("fact_trade", "fact_portfolio_daily", "pending_target"):
            conn.exec_driver_sql("DELETE FROM {} WHERE run_id='paper_cash'".format(table))


# --------------------------------------------------------------- dates


def test_trading_dates_follow_the_index_not_the_weekday(signals_ready: tuple) -> None:
    """A holiday has equity bars but no index bar, so it is not a trading date."""
    from src.common.db import read_sql

    _cfg, engine = signals_ready
    dates = trading_dates(engine, after_key=None, from_key=20180101, to_key=20180201)
    assert dates == sorted(dates), "dates must be ascending"
    index_days = {
        int(x)
        for x in read_sql(
            engine,
            "SELECT p.date_key d FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
            "WHERE s.instrument_type='INDEX' AND p.date_key BETWEEN 20180101 AND 20180201",
        )["d"]
    }
    assert set(dates) <= index_days


def test_latest_trading_date_is_found(signals_ready: tuple) -> None:
    _cfg, engine = signals_ready
    assert latest_trading_date(engine) is not None


def test_a_second_run_with_nothing_new_exits_zero(signals_ready: tuple) -> None:
    """Idempotence: a rerun on an already-processed range changes nothing."""
    import src.paper.daily_job as job

    cfg_obj, engine = signals_ready
    cfg_obj = {
        **cfg_obj,
        "paper": {**cfg_obj["paper"], "run_id": "paper_idem"},
        "strategy": {**cfg_obj["strategy"], "selected_model": "tree"},
    }
    dates = _signal_dates(engine, limit=3)
    assert dates
    log = get_logger("test", cfg_obj)
    for key in dates:
        job.process_date(cfg_obj, engine, "paper_idem", key, log)
    before = load_state(engine, "paper_idem").equity

    from src.common.db import read_sql

    count_before = int(
        read_sql(engine, "SELECT COUNT(*) n FROM fact_trade WHERE run_id='paper_idem'")["n"].iloc[0]
    )
    outcome = job.run(cfg_obj, run_id="paper_idem", no_etl=True)
    assert outcome["processed"] == [], "everything was already processed"
    assert load_state(engine, "paper_idem").equity == pytest.approx(before)
    assert (
        int(
            read_sql(engine, "SELECT COUNT(*) n FROM fact_trade WHERE run_id='paper_idem'")[
                "n"
            ].iloc[0]
        )
        == count_before
    )

    with engine.begin() as conn:
        for table in ("fact_trade", "fact_portfolio_daily", "pending_target"):
            conn.exec_driver_sql("DELETE FROM {} WHERE run_id='paper_idem'".format(table))


# --------------------------------------------------------------- alerts


def test_alert_writes_to_the_log_and_stderr(cfg_smoke: dict[str, Any], tmp_path: Path) -> None:
    log_path = tmp_path / "alerts.log"
    alert(cfg_smoke, "WARNING", "test alert", log_path=log_path)
    assert "test alert" in log_path.read_text(encoding="utf-8")
    assert "[WARNING]" in log_path.read_text(encoding="utf-8")


def test_alert_rejects_an_unknown_level(cfg_smoke: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown alert level"):
        alert(cfg_smoke, "CRITICAL", "x", log_path=tmp_path / "a.log")


def test_info_alerts_do_not_email(cfg_smoke: dict[str, Any], tmp_path: Path) -> None:
    assert alert(cfg_smoke, "INFO", "routine", log_path=tmp_path / "a.log") is None


def test_staleness_uses_calendar_days(cfg_smoke: dict[str, Any]) -> None:
    assert stale_days(cfg_smoke, 20230103, 20230110) == 7
    assert stale_days(cfg_smoke, None, 20230110) == 0
    assert is_stale(cfg_smoke, 20230103, 20230110) is True
    assert is_stale(cfg_smoke, 20230110, 20230110) is False


def test_cooldown_symbols_come_from_stop_loss_trades(signals_ready: tuple) -> None:
    """A stop-loss blocks re-entry; an ordinary exit does not."""
    from sqlalchemy import text

    from src.common.db import read_sql

    _cfg, engine = signals_ready
    keys = dict(
        read_sql(engine, "SELECT symbol, stock_key FROM dim_stock").itertuples(
            index=False, name=None
        )
    )
    symbol = sorted(keys)[0]
    day = _a_date(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT OR REPLACE INTO fact_trade (run_id, date_key, stock_key, side, "
                "quantity, fill_price, cost, reason) "
                "VALUES ('cd_probe', :d, :s, 'SELL', 1, 10.0, 0.0, 'STOP_LOSS')"
            ),
            {"d": day, "s": keys[symbol]},
        )
    assert symbol in cooldown_symbols(engine, "cd_probe", day, 5)

    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM fact_trade WHERE run_id = 'cd_probe'")
    assert symbol not in cooldown_symbols(engine, "cd_probe", day, 5)
