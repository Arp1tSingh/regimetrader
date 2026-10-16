"""Time the live paper job's per-day feature path, on a throwaway copy.

Answers a narrow question: when a new trading day arrives, how long does `make paper`
actually take? The cost lives in `ensure_inputs`, which recomputes indicators over the
whole warehouse for every date it processes (`AGENTS.md` section 6 requires the
full-history recompute, so this is by design and not a defect).

This measures the real function on a temporary copy of the real warehouse. It changes no
code and writes nothing outside the temp directory. The real warehouse is never opened
for writing, and no trading date is decided: this times the feature path only.

Usage:  python scripts/paper_rehearsal.py [--dates N]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config

LOG_NAME = "paper_rehearsal.log"


def say(msg: str, log: Path) -> None:
    print(msg, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def stage_times(cfg: dict[str, Any], engine: Any, date_key: int, model: str) -> dict[str, float]:
    """Time each stage of `ensure_inputs` on the copy. Returns seconds per stage."""
    from src.features.indicators import build_indicators
    from src.features.panel import build_panel, get_feature_cols, use_vix_in_regime
    from src.paper.daily_job import _extend_regime

    times: dict[str, float] = {}

    t = time.time()
    rows = build_indicators(engine, cfg)
    times["build_indicators"] = time.time() - t

    t = time.time()
    _extend_regime(cfg, engine, date_key)
    times["extend_regime"] = time.time() - t

    t = time.time()
    panel = build_panel(engine, cfg, use_cache=False, live=True)
    times["build_panel"] = time.time() - t

    t = time.time()
    import joblib

    year = int(str(date_key)[:4])
    estimator = joblib.load(Path(cfg["paths"]["models"]) / f"{model}_wf{year}.joblib")
    cols = get_feature_cols(use_vix_in_regime(cfg, engine))
    today = panel[panel["date_key"] == date_key]
    probs = estimator.predict_proba(today[cols])[:, 1] if not today.empty else []
    times["predict"] = time.time() - t

    times["_rows"] = float(rows)
    times["_panel_rows"] = float(len(panel))
    times["_probs"] = float(len(probs))
    return times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        type=int,
        default=2,
        help="how many dates to time; the job repeats the same work per date",
    )
    args = parser.parse_args(argv)

    cfg = load_config("config/config.yaml")
    log = Path(cfg["paths"]["logs"]) / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    model = str(cfg["strategy"]["selected_model"])
    work = Path(tempfile.mkdtemp(prefix="regimetrader_rehearsal_"))
    try:
        from src.common.db import get_engine, read_sql

        target = work / "rehearsal.db"
        shutil.copy(Path(cfg["paths"]["db"]), target)
        local = dict(cfg)
        local["paths"] = dict(cfg["paths"])
        local["paths"]["db"] = str(target)
        engine = get_engine(local)

        price_rows = int(read_sql(engine, "SELECT COUNT(*) n FROM fact_price")["n"].iloc[0])
        index_days = [
            int(r)
            for r in read_sql(
                engine,
                "SELECT p.date_key FROM fact_price p JOIN dim_stock s ON s.stock_key=p.stock_key "
                "WHERE s.instrument_type='INDEX' ORDER BY 1",
            )["date_key"]
        ]
        latest = index_days[-1]
        say(f"warehouse copy: {price_rows:,} price rows, {len(index_days)} index bars", log)
        say(f"model {model}; timing {args.dates} date(s) from the latest bar {latest}", log)
        say(
            "this is the feature path only: no date is decided, no trade is placed, and "
            "the real warehouse is never opened for writing\n",
            log,
        )

        per_date: list[float] = []
        for n in range(1, args.dates + 1):
            say(f"--- pass {n} (date_key={latest}) ---", log)
            t0 = time.time()
            times = stage_times(local, engine, latest, model)
            total = time.time() - t0
            per_date.append(total)
            for stage in ("build_indicators", "extend_regime", "build_panel", "predict"):
                say(f"  {stage:18} {times[stage]:8.1f}s", log)
            say(f"  {'TOTAL':18} {total:8.1f}s", log)
            say(
                f"  (indicator rows {int(times['_rows']):,}, "
                f"panel rows {int(times['_panel_rows']):,}, "
                f"probabilities {int(times['_probs'])})\n",
                log,
            )

        mean = sum(per_date) / len(per_date)
        say(f"mean per date : {mean:.1f}s  ({mean / 60:.1f} min)", log)
        if args.dates > 1:
            first, rest = per_date[0], per_date[1:]
            say(
                f"note: pass 1 was {first:.1f}s, later passes "
                f"{min(rest):.1f}-{max(rest):.1f}s; the job does the same work each time",
                log,
            )
        say(
            "extrapolation: 1 new trading day ~ "
            f"{mean / 60:.1f} min; 5 new days back-filled ~ "
            f"{mean * 5 / 60:.1f} min",
            log,
        )
        say("DONE", log)
        engine.dispose()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
