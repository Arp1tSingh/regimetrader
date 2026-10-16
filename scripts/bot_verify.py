"""Verify the Bot demo end to end and measure every stage.

1. Remaining stage costs on the lookback band (what the page pays).
2. A real replay of 2024-01-02 -> 5 days, with the first-day no-hindsight comparison.
3. Proof the real warehouse is byte-identical afterwards.

Never touches the real warehouse: everything runs on a temp copy, and the final check
compares a hash taken before and after.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.dashboard import bot

#: Where the transcript is written. Derived from the config's `paths.logs` rather than
#: hardcoded, so the repository-hygiene test that forbids a source file naming a
#: gitignored directory stays satisfied (`AGENTS.md`: all paths come from config).
LOG_NAME = "bot_verify.log"


def say(msg: str, log: Path) -> None:
    print(msg, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    cfg = load_config("config/config.yaml")
    source = Path(cfg["paths"]["db"])
    log = Path(cfg["paths"]["logs"]) / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    before = sha(source)
    say(f"warehouse sha256 before : {before[:24]}...", log)

    model = str(cfg["strategy"]["selected_model"])
    lookback = bot.lookback_trading_days(cfg)
    say(f"model {model}, lookback {lookback} trading days", log)

    # ---- stage costs on the band -----------------------------------------
    from src.common.db import get_engine, read_sql
    from src.features.indicators import build_indicators
    from src.features.panel import build_panel, get_feature_cols, use_vix_in_regime
    from src.paper.daily_job import _extend_regime

    work = Path(tempfile.mkdtemp(prefix="regimetrader_verify_"))
    try:
        db = bot.copy_warehouse(source, work / "band.db")
        local = dict(cfg)
        local["paths"] = dict(cfg["paths"])
        local["paths"]["db"] = str(db)
        eng = get_engine(local)

        days = [
            int(r)
            for r in read_sql(
                eng,
                "SELECT p.date_key FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
                "WHERE s.instrument_type='INDEX' AND p.date_key <= 20240108 ORDER BY 1",
            )["date_key"]
        ]
        bot.trim_to_lookback(eng, 20240108, lookback)
        bot.truncate_after(eng, 20240108)
        band_rows = int(read_sql(eng, "SELECT COUNT(*) n FROM fact_price")["n"].iloc[0])
        say(f"\nband rows after trim   : {band_rows:,}", log)

        t = time.time()
        build_indicators(eng, local)
        say(f"  build_indicators  : {time.time() - t:7.1f}s", log)

        t = time.time()
        _extend_regime(local, eng, 20240108)
        say(f"  _extend_regime    : {time.time() - t:7.1f}s", log)

        t = time.time()
        panel = build_panel(eng, local, use_cache=False, live=True)
        say(f"  build_panel       : {time.time() - t:7.1f}s  rows={len(panel)}", log)

        import joblib

        estimator = joblib.load(Path(cfg["paths"]["models"]) / f"{model}_wf2024.joblib")
        cols = get_feature_cols(use_vix_in_regime(local, eng))
        t = time.time()
        for d in days[-5:]:
            sub = panel[panel["date_key"] == d]
            if not sub.empty:
                estimator.predict_proba(sub[cols])[:, 1]
        say(f"  predict x5 days   : {time.time() - t:7.1f}s", log)
        eng.dispose()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    # ---- the real replay --------------------------------------------------
    say("\n=== full replay, start 20240101 ===", log)
    t = time.time()

    def on_day(index: int) -> None:
        say(f"  day {index + 1}/{bot.WINDOW_DAYS} complete", log)

    run = bot.run_replay(cfg, source, bot.WHITELIST_START_DATES[0], per_day_check=on_day)
    say(
        f"replay wall time      : {time.time() - t:.1f}s   (reported {run.elapsed_seconds:.1f}s)",
        log,
    )

    for day in run.days:
        threshold = "n/a" if day.entry_threshold is None else f"{day.entry_threshold:.6f}"
        say(
            f"  {day.date_key}  regime={day.regime or 'n/a':9} "
            f"entry>={threshold}  cands={day.candidates:3} passed={day.passed_threshold:3} "
            f"weights={len(day.target_weights):2} orders={len(day.orders):2} "
            f"equity={day.equity:,.2f} cash={day.cash:,.2f}",
            log,
        )
        for order in day.orders[:4]:
            say(
                f"      {order['side']:4} {order['symbol']:14} {order['quantity']:>7} "
                f"@ {order['fill_price']:>10,.2f}  {order['reason']}",
                log,
            )
        if len(day.orders) > 4:
            say(f"      ... and {len(day.orders) - 4} more", log)
        if day.target_weights:
            top = sorted(day.target_weights.items(), key=lambda kv: -kv[1])[:3]
            say(
                "      weights: " + ", ".join(f"{s} {w:.6f}" for s, w in top),
                log,
            )

    check = bot.first_day_check_against_backtest(cfg, run)
    say(f"\nNO-HINDSIGHT CHECK (day 1, {check.get('compared_date')})", log)
    say(f"  match             : {check.get('match')}", log)
    say(f"  replay orders     : {check.get('n_replay')}", log)
    say(f"  backtest_main     : {check.get('n_stored')}", log)
    say(f"  only in replay    : {check.get('only_in_replay')}", log)
    say(f"  only in backtest  : {check.get('only_in_stored')}", log)
    say(f"  qty/price diffs   : {check.get('quantity_or_price_differences')}", log)

    after = sha(source)
    say(f"\nwarehouse sha256 after  : {after[:24]}...", log)
    say(f"WAREHOUSE UNCHANGED     : {before == after}", log)
    say(f"leftover temp dirs      : {bot.remaining_temp_dirs()}", log)
    say("DONE", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
