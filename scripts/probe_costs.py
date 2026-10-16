"""Measure the two costs that decide the Bot demo design.

1. build_indicators on a LOOKBACK SLICE  -> sizes the fast path
2. build_indicators on the FULL warehouse -> answers "how long does tonight's daily
   bot run take?"

Runs entirely on a THROWAWAY COPY. Never touches the real warehouse.
Writes each timing to logs/cost_probe.txt as it completes, so progress is visible.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.db import get_engine, read_sql

#: Transcript filename. The directory comes from the config's `paths.logs`, never a
#: literal: the repository-hygiene test forbids a source file naming a gitignored
#: directory (`AGENTS.md`: all paths come from config).
LOG_NAME = "cost_probe.txt"


def say(msg: str, log: Path) -> None:
    print(msg, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def fresh_copy(cfg: dict, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in workdir.glob("*.db*"):
        stale.unlink()
    target = workdir / "work.db"
    shutil.copy(Path(cfg["paths"]["db"]), target)
    return target


def main() -> int:
    cfg = load_config("config/config.yaml")
    log = Path(cfg["paths"]["logs"]) / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    workdir = Path(tempfile.gettempdir()) / "regimetrader_costprobe"

    target = fresh_copy(cfg, workdir)
    local = dict(cfg)
    local["paths"] = dict(cfg["paths"])
    local["paths"]["db"] = str(target)
    eng = get_engine(local)

    from src.features.indicators import build_indicators

    # ---- 1. slice cost -----------------------------------------------------
    days = [
        int(r)
        for r in read_sql(
            eng,
            "SELECT p.date_key FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
            "WHERE s.instrument_type='INDEX' ORDER BY 1",
        )["date_key"]
    ]
    window = days[-320:]
    cutoff = window[0]
    say(f"index trading days: {len(days)}; slice keeps last {len(window)} (from {cutoff})", log)

    t = time.time()
    with eng.begin() as conn:
        for tbl in ("fact_indicator", "fact_market_regime", "fact_price"):
            conn.exec_driver_sql(
                f"DELETE FROM {tbl} WHERE date_key < :c OR date_key > :h",
                {"c": cutoff, "h": days[-1]},
            )
    rows = int(read_sql(eng, "SELECT COUNT(*) n FROM fact_price")["n"].iloc[0])
    say(f"band rows         : {rows:,}  (was {len(days)} index days, keeping {len(window)})", log)
    t = time.time()
    build_indicators(eng, local)
    slice_s = time.time() - t
    say(f"build_indicators SLICE: {slice_s:6.1f}s  ({rows:,} rows)", log)
    eng.dispose()

    # ---- 2. full cost ------------------------------------------------------
    say("now the full warehouse (this is what tonight's daily bot pays)...", log)
    target2 = fresh_copy(cfg, workdir)
    local2 = dict(cfg)
    local2["paths"] = dict(cfg["paths"])
    local2["paths"]["db"] = str(target2)
    eng2 = get_engine(local2)
    full_rows = int(read_sql(eng2, "SELECT COUNT(*) n FROM fact_price")["n"].iloc[0])

    t = time.time()
    build_indicators(eng2, local2)
    full_s = time.time() - t
    say(f"build_indicators FULL : {full_s:6.1f}s  ({full_rows:,} rows)", log)
    say(f"speed-up from slicing : {full_s / max(slice_s, 0.01):.1f}x", log)
    eng2.dispose()
    shutil.rmtree(workdir, ignore_errors=True)
    say("DONE", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
