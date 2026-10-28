"""Prove the fast path equals a literal recompute, then write the proof to JSON.

The fast path in `src/dashboard/bot.py` recomputes features on a *lookback band* rather
than the full history, because a literal full recompute costs about 6 minutes per click
and this runs behind a public web button. That optimisation is only legitimate if every
feature is causal.

This script proves it rather than asserting it:

* **literal** - copy the warehouse, delete every row after day D, run the full
  `build_indicators` / `_extend_regime` / `build_panel` / classifier / `target_weights`
  over all remaining history, and read off day D;
* **fast** - the same, but with history before the lookback window also removed;
* **mutation** - perturb a row *after* day D, recompute over the **full untruncated**
  history, and confirm the day-D output does not move. That is what "no hindsight"
  actually means.

Everything is recomputed from truncated raw data. Nothing here reads a stored
`fact_signal` value: the previous version read `target_weight` straight out of
`fact_signal`, which `truncate_after` never deletes, so the literal and fast paths
compared the *same stored rows* against each other and the comparison could not fail.
Weights now come from calling `src.trading.rules.target_weights` -- the same function
the backtest and the paper bot use -- on the freshly computed probabilities. The test
`test_the_comparison_fails_if_a_stored_weight_is_corrupted` proves the difference.

Writes `reports/results/bot_equivalence.json` for the dashboard to display. If anything
does not match, the mismatch is recorded and reported rather than smoothed over.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config
from src.common.db import get_engine, read_sql

OUTPUT_NAME = "bot_equivalence.json"

#: Floating-point comparisons use this absolute tolerance. Labels (regime names) and
#: target weights are compared exactly; only continuous values get a tolerance.
FLOAT_TOL = 1e-9


def _day_output(engine: Any, cfg: dict[str, Any], day_key: int, model: str) -> dict[str, Any]:
    """Regime label, prob_up and target weights for one day, recomputed from raw data.

    The weights are produced by `target_weights`, the pure function the engine and the
    paper job both call, so this is the real strategy rather than a reimplementation
    (`AGENTS.md` rule 9). No stored signal value is consulted.
    """
    import joblib

    from src.features.panel import build_panel, get_feature_cols, use_vix_in_regime
    from src.trading.rules import build_regime_rules, target_weights

    regime_row = read_sql(
        engine,
        "SELECT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key WHERE m.date_key = :d",
        {"d": int(day_key)},
    )
    regime = str(regime_row["regime"].iloc[0]) if not regime_row.empty else ""

    panel = build_panel(engine, cfg, use_cache=False, live=True)
    cols = get_feature_cols(use_vix_in_regime(cfg, engine))
    today = panel[panel["date_key"] == day_key]
    if today.empty:
        return {"regime": regime, "prob_up": {}, "weights": {}, "rows": 0}

    year = pd.Timestamp(str(day_key)).year
    estimator = joblib.load(Path(cfg["paths"]["models"]) / f"{model}_wf{year}.joblib")
    probs = estimator.predict_proba(today[cols])[:, 1]

    signal_frame = pd.DataFrame({"prob_up": probs}, index=today["symbol"].astype(str))

    # The real strategy, on the freshly computed probabilities.
    strategy = cfg["strategy"]
    weights_series = target_weights(
        signal_frame,
        regime,
        held=set(),
        rules=build_regime_rules(cfg),
        slots=int(strategy["slots"]),
        exit_prob=float(strategy["exit_prob"]),
    )
    weights = {str(symbol): float(weight) for symbol, weight in weights_series.items()}

    return {
        "regime": regime,
        "prob_up": {str(s): float(p) for s, p in zip(today["symbol"], probs, strict=True)},
        "weights": weights,
        "rows": int(len(today)),
    }


def _prepare(cfg: dict[str, Any], source: Path, day_key: int, lookback: int | None) -> Any:
    """Copy the warehouse, cut it to `day_key` (and optionally trim far history)."""
    from src.dashboard.bot import copy_warehouse, trim_to_lookback, truncate_after

    work = Path(tempfile.mkdtemp(prefix="regimetrader_equiv_"))
    db_path = copy_warehouse(source, work / "equiv.db")
    local = dict(cfg)
    local["paths"] = dict(cfg["paths"])
    local["paths"]["db"] = str(db_path)
    engine = get_engine(local)
    if lookback is not None:
        trim_to_lookback(engine, day_key, lookback)
    truncate_after(engine, day_key)
    return local, engine, work


def literal(cfg: dict[str, Any], source: Path, day_key: int, model: str) -> dict[str, Any]:
    """Full recompute: all history up to D, nothing trimmed."""
    from src.features.indicators import build_indicators
    from src.paper.daily_job import _extend_regime

    local, engine, work = _prepare(cfg, source, day_key, lookback=None)
    try:
        build_indicators(engine, local)
        _extend_regime(local, engine, day_key)
        return _day_output(engine, local, day_key, model)
    finally:
        engine.dispose()
        shutil.rmtree(work, ignore_errors=True)


def fast(
    cfg: dict[str, Any], source: Path, day_key: int, model: str, lookback: int
) -> dict[str, Any]:
    """Band recompute: the fast path the page uses."""
    from src.features.indicators import build_indicators
    from src.paper.daily_job import _extend_regime

    local, engine, work = _prepare(cfg, source, day_key, lookback=lookback)
    try:
        build_indicators(engine, local)
        _extend_regime(local, engine, day_key)
        return _day_output(engine, local, day_key, model)
    finally:
        engine.dispose()
        shutil.rmtree(work, ignore_errors=True)


def mutated(
    cfg: dict[str, Any], source: Path, day_key: int, model: str, lookback: int
) -> dict[str, Any]:
    """Change a row AFTER day D, recompute over the FULL history, and see if D moves.

    This is the direct test of look-ahead: if any future row influenced the day-D
    output, changing it would change the answer.

    The whole history is deliberately **kept**, unlike the other two modes. The previous
    version mutated a future row and then called `truncate_after`, which deleted the very
    rows it had just changed before recomputing, so the comparison could not fail no
    matter what the features did. Here the future row is present throughout, and only the
    observed day-D output is read back.
    """
    from src.dashboard.bot import copy_warehouse
    from src.features.indicators import build_indicators
    from src.paper.daily_job import _extend_regime

    work = Path(tempfile.mkdtemp(prefix="regimetrader_equiv_"))
    db_path = copy_warehouse(source, work / "equiv.db")
    local = dict(cfg)
    local["paths"] = dict(cfg["paths"])
    local["paths"]["db"] = str(db_path)
    engine = get_engine(local)
    try:
        later = [
            int(r)
            for r in read_sql(
                engine,
                "SELECT DISTINCT date_key FROM fact_price WHERE date_key > :d ORDER BY 1",
                {"d": int(day_key)},
            )["date_key"]
        ]
        if not later:
            return {"mutated": False, "reason": "no rows after D to mutate"}
        target_day = int(later[0])
        # A large, unmistakable change to a price on a future date.
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "UPDATE fact_price SET close_price = close_price * 7.77 + 1000.0, "
                "open_price = open_price * 7.77 + 1000.0, high_price = high_price * 7.77 + 1000.0, "
                "low_price = low_price * 7.77 + 1000.0 WHERE date_key = :d",
                {"d": target_day},
            )
        # No `truncate_after` here: the mutated rows must survive into the recompute.
        build_indicators(engine, local)
        _extend_regime(local, engine, day_key)
        return {
            "mutated": True,
            "mutated_date": target_day,
            "rows_after_day_retained": True,
            "output": _day_output(engine, local, day_key, model),
        }
    finally:
        engine.dispose()
        shutil.rmtree(work, ignore_errors=True)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Exact for labels and weights, `FLOAT_TOL` for continuous values."""
    mismatches: list[dict[str, Any]] = []

    if a["regime"] != b["regime"]:
        mismatches.append({"kind": "regime", "literal": a["regime"], "fast": b["regime"]})

    for symbol in sorted(set(a["weights"]) | set(b["weights"])):
        wa = a["weights"].get(symbol)
        wb = b["weights"].get(symbol)
        if wa is None or wb is None:
            mismatches.append(
                {"kind": "weight_presence", "symbol": symbol, "literal": wa, "fast": wb}
            )
        elif abs(float(wa) - float(wb)) > FLOAT_TOL:
            mismatches.append(
                {
                    "kind": "target_weight",
                    "symbol": symbol,
                    "literal": wa,
                    "fast": wb,
                    "delta": abs(float(wa) - float(wb)),
                }
            )

    prob_values_compared = 0
    worst = 0.0
    for symbol in sorted(set(a["prob_up"]) & set(b["prob_up"])):
        prob_values_compared += 1
        delta = abs(a["prob_up"][symbol] - b["prob_up"][symbol])
        worst = max(worst, delta)
        if delta > FLOAT_TOL:
            mismatches.append(
                {
                    "kind": "prob_up",
                    "symbol": symbol,
                    "literal": a["prob_up"][symbol],
                    "fast": b["prob_up"][symbol],
                    "delta": delta,
                }
            )

    return {
        "match": not mismatches,
        "prob_values_compared": prob_values_compared,
        "weights_compared": len(set(a["weights"]) | set(b["weights"])),
        "max_prob_delta": worst,
        "tolerance": FLOAT_TOL,
        "mismatches": mismatches[:25],
        "mismatch_count": len(mismatches),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        type=int,
        nargs="+",
        default=None,
        help="decision days to compare; by default day 1, day 30 and the last day of "
        "the 60-day playback window, derived from the warehouse",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path; defaults to <paths.results>/bot_equivalence.json from the config",
    )
    args = parser.parse_args(argv)

    cfg = load_config("config/config.yaml")
    source = Path(cfg["paths"]["db"])
    model = str(cfg["strategy"]["selected_model"])
    out = Path(args.out) if args.out else Path(cfg["paths"]["results"]) / OUTPUT_NAME

    from src.dashboard.bot import (
        WINDOW_DAYS,
        available_start_dates,
        lookback_trading_days,
    )

    lookback = lookback_trading_days(cfg)
    print(f"lookback derived from config: {lookback} trading days")
    print(f"model: {model}   source: {source.name}")

    if args.days is None:
        probe = get_engine(cfg)
        try:
            starts = available_start_dates(cfg, probe)
            index_days = [
                int(r)
                for r in read_sql(
                    probe,
                    "SELECT p.date_key FROM fact_price p "
                    "JOIN dim_stock s ON s.stock_key = p.stock_key "
                    "WHERE s.instrument_type = 'INDEX' AND p.date_key >= :a ORDER BY 1",
                    {"a": int(starts[0])},
                )["date_key"]
            ]
        finally:
            probe.dispose()
        if not starts or len(index_days) < WINDOW_DAYS:
            parser.error("the warehouse does not contain a full playback window")
        args.days = [
            int(index_days[0]),
            int(index_days[WINDOW_DAYS // 2 - 1]),
            int(index_days[WINDOW_DAYS - 1]),
        ]

    print(f"start options: {list(starts) if 'starts' in locals() else 'provided by --days'}")
    print(f"playback window: {WINDOW_DAYS} trading days")
    print(
        "every mode recomputes from truncated raw data through the real pipeline; "
        "no stored fact_signal value is read"
    )

    results: list[dict[str, Any]] = []
    for day_key in args.days:
        print(f"\n=== day {day_key} ===", flush=True)

        t = time.time()
        lit = literal(cfg, source, day_key, model)
        lit_s = time.time() - t
        print(f"  literal recompute : {lit_s:6.1f}s", flush=True)

        t = time.time()
        fst = fast(cfg, source, day_key, model, lookback)
        fast_s = time.time() - t
        print(f"  fast recompute     : {fast_s:6.1f}s", flush=True)

        cmp_result = compare(lit, fst)
        print(
            f"  MATCH: {cmp_result['match']}  "
            f"(probs compared {cmp_result['prob_values_compared']}, "
            f"max delta {cmp_result['max_prob_delta']:.3e})",
            flush=True,
        )

        t = time.time()
        mut = mutated(cfg, source, day_key, model, lookback)
        mut_s = time.time() - t
        mutation_result: dict[str, Any] = {"performed": bool(mut.get("mutated"))}
        if mut.get("mutated"):
            after = mut["output"]
            stable_regime = after["regime"] == fst["regime"]
            stable_probs = all(
                abs(after["prob_up"].get(s, float("nan")) - fst["prob_up"].get(s, float("nan")))
                <= FLOAT_TOL
                for s in set(fst["prob_up"]) & set(after["prob_up"])
            )
            stable_weights = all(
                abs(after["weights"].get(s, float("nan")) - fst["weights"].get(s, float("nan")))
                <= FLOAT_TOL
                for s in set(fst["weights"]) & set(after["weights"])
            )
            mutation_result = {
                "performed": True,
                "mutated_date": mut.get("mutated_date"),
                "regime_unchanged": stable_regime,
                "probs_unchanged": stable_probs,
                "weights_unchanged": stable_weights,
                "passed": bool(stable_regime and stable_probs and stable_weights),
            }
            print(
                f"  mutation ({mut.get('mutated_date')}): "
                f"passed={mutation_result['passed']}  ({mut_s:.1f}s)",
                flush=True,
            )

        results.append(
            {
                "day": int(day_key),
                "regime": lit["regime"],
                "literal_seconds": round(lit_s, 1),
                "fast_seconds": round(fast_s, 1),
                "speedup": round(lit_s / max(fast_s, 0.01), 1),
                "equivalence": cmp_result,
                "lookahead_mutation": mutation_result,
                "rows": lit["rows"],
            }
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_trading_days": lookback,
        "model": model,
        "float_tolerance": FLOAT_TOL,
        "days_tested": [int(d) for d in args.days],
        "method": (
            "every mode recomputes indicators, regime, prob_up and target weights from "
            "truncated raw fact_price through the real pipeline "
            "(build_indicators -> _extend_regime -> build_panel -> classifier -> "
            "src.trading.rules.target_weights). No stored fact_signal value is read."
        ),
        "results": results,
        "all_match": all(r["equivalence"]["match"] for r in results),
        "all_lookahead_safe": all(r["lookahead_mutation"].get("passed", False) for r in results),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"all_match={payload['all_match']}  all_lookahead_safe={payload['all_lookahead_safe']}")
    if not payload["all_match"]:
        print("\nMISMATCH FOUND. The proof file records the details; do not present it as a pass.")
        return 1
    if not payload["all_lookahead_safe"]:
        print("\nLOOK-AHEAD MUTATION DETECTED. Stop and report.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
