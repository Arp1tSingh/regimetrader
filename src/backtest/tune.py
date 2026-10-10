"""Validation tuning grid (`spec 05` section 6, `06` section 5).

Protocol, in order:

1. **Calibration check** per model on validation-year probabilities. `candidate_day_share`
   is the share of validation days on which at least `ceil(min(slots_grid) / 2)` stocks
   have `prob_up >= min(entry_base_grid)`. Logged as a `calibration` row, which is **not**
   counted as a trial (`DECISIONS` R28).
2. **Fallback.** If a model's share is below `strategy.calibration_min_candidate_day_share`,
   append `strategy.fallback_entry_base` to that model's `entry_base` grid (rank-based
   selection) and log that the fallback was used. This changes the trial count.
3. **Grid.** models x entry_base x slots, strategy V2 on the **validation period only**,
   not persisted. One trial per combination, `metric = sharpe`, with CAGR, max drawdown
   and turnover in `extra`.
4. **Selection.** Highest validation Sharpe, tie-break lower turnover. Written to
   `paths.frozen_config` with `meta.method`.

The test period is never touched here (`AGENTS.md` rules 4 and 12).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.common.config import ConfigError, load_config
from src.common.db import get_engine
from src.common.log import get_logger
from src.common.trial_log import count_trials, log_trial

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1


def candidate_day_share(
    engine: Any,
    cfg: dict[str, Any],
    model_name: str,
    entry: float,
    need: int,
) -> tuple[float, int]:
    """Share of validation days with at least `need` stocks at or above `entry`.

    Args:
        engine: Warehouse engine.
        cfg: Loaded config.
        model_name: Model to measure.
        entry: Entry threshold.
        need: Minimum number of qualifying stocks.

    Returns:
        `(share, days_measured)`.
    """
    from src.common.db import read_sql

    start = int(str(cfg["splits"]["validation_start"]).replace("-", ""))
    end = int(str(cfg["splits"]["validation_end"]).replace("-", ""))
    df = read_sql(
        engine,
        "SELECT f.date_key AS date_key, f.prob_up AS prob_up "
        "FROM fact_signal f JOIN dim_model m ON m.model_key = f.model_key "
        "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b",
        {"model": model_name, "a": start, "b": end},
    )
    if df.empty:
        return float("nan"), 0
    per_day = df.assign(ok=df["prob_up"] >= entry).groupby("date_key")["ok"].sum()
    return float((per_day >= need).mean()), int(len(per_day))


def calibration_rows(engine: Any, cfg: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    """Step 1: measure and log `candidate_day_share` for every model."""
    need = math.ceil(min(cfg["tuning"]["slots_grid"]) / 2)
    entry = float(min(cfg["tuning"]["entry_base_grid"]))
    limit = float(cfg["strategy"]["calibration_min_candidate_day_share"])
    log = get_logger("src.backtest.tune", cfg)

    rows = []
    for model in models:
        share, days = candidate_day_share(engine, cfg, model, entry, need)
        triggered = bool(share < limit) if share == share else True
        log.info(
            "%s: candidate_day_share = %.4f over %d day(s) -> %s",
            model,
            share,
            days,
            "fallback" if triggered else "ok",
        )
        log_trial(
            cfg,
            phase="tune",
            kind="calibration",
            run_id="calibration",
            model=model,
            params={"entry": entry, "need": need},
            period="validation",
            metric="candidate_day_share",
            value=share,
            extra={"days": days, "threshold": limit, "fallback": triggered},
            note="diagnostic, not a trial (DECISIONS R28)",
        )
        rows.append({"model": model, "share": share, "days": days, "fallback": triggered})
    return rows


def run(cfg: dict[str, Any], *, apply_fallback: bool = True) -> dict[str, Any]:
    """Run the calibration check and the grid, then write the frozen config.

    Args:
        cfg: Loaded config.
        apply_fallback: Whether to extend the entry_base grid for models whose
            calibration share is too low.

    Returns:
        A summary dict: trials run, the selected configuration and the method used.
    """
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import summarize
    from src.common.manifest import write_manifest

    log = get_logger("src.backtest.tune", cfg)
    engine = get_engine(cfg)

    models = list(cfg["tuning"]["models"])
    entries = [float(e) for e in cfg["tuning"]["entry_base_grid"]]
    slots_grid = [int(s) for s in cfg["tuning"]["slots_grid"]]

    # Steps 1 and 2: calibration, then the fallback.
    calibration = calibration_rows(engine, cfg, models)
    fallback_models = {r["model"] for r in calibration if r["fallback"]}
    if not apply_fallback:
        fallback_models = set()

    fallback_entry = float(cfg["strategy"]["fallback_entry_base"])
    method = "threshold"
    if fallback_models:
        method = "rank_fallback"
        log.info(
            "fallback entry_base %s for: %s", fallback_entry, ", ".join(sorted(fallback_models))
        )

    start = str(cfg["splits"]["validation_start"])
    end = str(cfg["splits"]["validation_end"])

    # The frozen config must not bias the grid it is choosing (`DECISIONS` R24).
    tuning_cfg = {**cfg, "strategy": {**cfg["strategy"], "selected_model": None}}

    results: list[dict[str, Any]] = []
    for model in models:
        model_entries = list(entries)
        if model in fallback_models and fallback_entry not in model_entries:
            model_entries.append(fallback_entry)
        for entry in model_entries:
            for slots in slots_grid:
                variant_cfg = {
                    **tuning_cfg,
                    "strategy": {**tuning_cfg["strategy"], "entry_base": entry, "slots": slots},
                }
                rules_cfg = {**variant_cfg, "strategy": {**variant_cfg["strategy"]}}
                provider = _variant_provider(rules_cfg)
                outcome = run_backtest(
                    rules_cfg,
                    provider,
                    start,
                    end,
                    "tune_tmp",
                    engine=engine,
                    model_name=model,
                    persist=False,
                )
                metrics = summarize(outcome.equity, outcome.trades, rules_cfg)
                record = {
                    "model": model,
                    "entry_base": entry,
                    "slots": slots,
                    "sharpe": metrics["sharpe"],
                    "cagr": metrics["cagr"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "n_trades": metrics["n_trades"],
                    "method": "rank_fallback" if model in fallback_models else "threshold",
                }
                results.append(record)
                log_trial(
                    cfg,
                    phase="tune",
                    kind="strategy_grid",
                    run_id="grid_validation",
                    model=model,
                    params={"entry_base": entry, "slots": slots},
                    period="validation",
                    metric="sharpe",
                    value=metrics["sharpe"],
                    extra={
                        "cagr": metrics["cagr"],
                        "max_drawdown": metrics["max_drawdown"],
                        "turnover": metrics["turnover"],
                        "n_trades": metrics["n_trades"],
                        "method": record["method"],
                    },
                )
                log.info(
                    "%s entry=%.2f slots=%d -> sharpe %.4f",
                    model,
                    entry,
                    slots,
                    metrics["sharpe"],
                )

    grid = pd.DataFrame(results)
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    grid.sort_values(["sharpe", "turnover"], ascending=[False, True]).to_csv(
        results_dir / "tune_grid_validation.csv", index=False
    )

    # Step 4: highest Sharpe, tie-break lower turnover.
    best = grid.sort_values(["sharpe", "turnover"], ascending=[False, True]).iloc[0]

    frozen_path = Path(cfg["paths"]["frozen_config"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(
        yaml.safe_dump(
            {
                "strategy": {
                    "selected_model": str(best["model"]),
                    "entry_base": float(best["entry_base"]),
                    "slots": int(best["slots"]),
                },
                "meta": {
                    "tuned_at": pd.Timestamp.utcnow().isoformat(),
                    "validation_sharpe": float(best["sharpe"]),
                    "trials": int(len(grid)),
                    "method": method,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.info(
        "selected %s entry=%.2f slots=%d (sharpe %.4f) from %d trial(s)",
        best["model"],
        best["entry_base"],
        best["slots"],
        best["sharpe"],
        len(grid),
    )

    write_manifest(
        cfg,
        "tune",
        engine,
        extra={
            "trials": int(len(grid)),
            "selected_model": str(best["model"]),
            "entry_base": float(best["entry_base"]),
            "slots": int(best["slots"]),
            "method": method,
        },
    )

    return {
        "trials": int(len(grid)),
        "total_logged_trials": count_trials(cfg),
        "selected_model": str(best["model"]),
        "entry_base": float(best["entry_base"]),
        "slots": int(best["slots"]),
        "validation_sharpe": float(best["sharpe"]),
        "method": method,
        "fallback_models": sorted(fallback_models),
    }


def _variant_provider(cfg: dict[str, Any]) -> Any:
    """The full strategy V2, built from the candidate entry_base and slots."""
    from src.backtest.strategies import make_v2

    return make_v2(cfg)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.backtest.tune",
        description="Tune model, entry_base and slots on the validation period.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not extend the grid when the calibration check fails.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    summary = run(cfg, apply_fallback=not args.no_fallback)
    print(
        f"{summary['trials']} trial(s); selected {summary['selected_model']} "
        f"entry_base={summary['entry_base']} slots={summary['slots']} "
        f"(validation sharpe {summary['validation_sharpe']:.4f}, method {summary['method']})"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
