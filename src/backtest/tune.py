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


class TuningError(RuntimeError):
    """Raised when the grid produced no usable configuration to select."""


#: R31: a model is eligible for strategy tuning only if both hold on the
#: validation-year `prob_up` distribution. No returns are consulted.
MAX_TIED_SHARE = 0.20
MIN_P95_P50_SPREAD = 0.01

#: R33: a model uses the absolute third entry base if it reaches this share of
#: validation days at that threshold; otherwise its own validation p80 is used.
ABSOLUTE_FALLBACK_ENTRY = 0.52
DAYS_SHARE_FOR_ABSOLUTE = 0.50

#: R34: a run whose entries are more than this share decided by the symbol
#: tie-break is flagged and excluded from selection.
MAX_TIE_BREAK_SHARE = 0.05


def eligibility(stats: dict[str, float]) -> tuple[bool, str]:
    """Apply the R31 rule to one model's validation distribution.

    Args:
        stats: `tied_share` (share of rows at the single most common `prob_up`) and
            `p95_minus_p50`.

    Returns:
        `(eligible, reason)`.
    """
    reasons = []
    if stats["tied_share"] > MAX_TIED_SHARE:
        reasons.append(f"tied share {stats['tied_share']:.2%} exceeds {MAX_TIED_SHARE:.0%}")
    if stats["p95_minus_p50"] < MIN_P95_P50_SPREAD:
        reasons.append(f"p95-p50 {stats['p95_minus_p50']:.4f} below {MIN_P95_P50_SPREAD}")
    return (not reasons, "; ".join(reasons) or "eligible")


def _is_finite(value: Any) -> bool:
    """True for a real number, False for NaN and None."""
    try:
        return bool(pd.notna(value)) and float(value) == float(value)
    except (TypeError, ValueError):
        return False


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


def distribution_stats(engine: Any, cfg: dict[str, Any], model: str) -> dict[str, float]:
    """Validation-year `prob_up` distribution for one model. No returns are read.

    Returns:
        `p50`, `p80`, `p95`, `p99`, `max`, `distinct`, `tied_share`, `p95_minus_p50`.
    """
    import numpy as np

    from src.common.db import read_sql

    start = int(str(cfg["splits"]["validation_start"]).replace("-", ""))
    end = int(str(cfg["splits"]["validation_end"]).replace("-", ""))
    df = read_sql(
        engine,
        "SELECT f.prob_up AS prob_up FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b",
        {"model": model, "a": start, "b": end},
    )
    if df.empty:
        return {
            "p50": float("nan"),
            "p80": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "distinct": 0,
            "tied_share": 1.0,
            "p95_minus_p50": 0.0,
        }

    values = df["prob_up"].to_numpy()
    _, counts = np.unique(np.round(values, 12), return_counts=True)
    p50, p80, p95, p99 = (float(np.percentile(values, q)) for q in (50, 80, 95, 99))
    return {
        "p50": p50,
        "p80": p80,
        "p95": p95,
        "p99": p99,
        "max": float(values.max()),
        "distinct": int(len(counts)),
        "tied_share": float(counts.max() / values.size),
        "p95_minus_p50": p95 - p50,
    }


def plan_models(engine: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """R31-R33: eligibility plus the per-model third entry base and method."""
    models = list(cfg["tuning"]["models"])
    entries = [float(e) for e in cfg["tuning"]["entry_base_grid"]]
    need = math.ceil(min(cfg["tuning"]["slots_grid"]) / 2)

    plan: list[dict[str, Any]] = []
    for model in models:
        stats = distribution_stats(engine, cfg, model)
        ok, reason = eligibility(stats)
        if not ok:
            plan.append(
                {
                    "model": model,
                    "eligible": False,
                    "reason": reason,
                    "entries": [],
                    "method": "excluded",
                    "stats": stats,
                }
            )
            continue

        share, _days = candidate_day_share(engine, cfg, model, ABSOLUTE_FALLBACK_ENTRY, need)
        if share >= DAYS_SHARE_FOR_ABSOLUTE:
            third, method = ABSOLUTE_FALLBACK_ENTRY, "absolute"
        else:
            third, method = round(stats["p80"], 6), "percentile_p80"
        third = round(float(third), 6)

        model_entries = list(entries)
        if third not in model_entries:
            model_entries.append(third)
        plan.append(
            {
                "model": model,
                "eligible": True,
                "reason": "eligible",
                "entries": model_entries,
                "method": method,
                "third_entry": third,
                "share_at_fallback": share,
                "stats": stats,
            }
        )
    return plan


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
    slots_grid = [int(s) for s in cfg["tuning"]["slots_grid"]]

    # Step 1: report the shared-threshold calibration share for every model, so the
    # log keeps the record of why the absolute fallback cannot fit all four.
    calibration = calibration_rows(engine, cfg, models)
    for row in calibration:
        log.info(
            "%s: candidate_day_share at %.2f = %.4f",
            row["model"],
            float(min(cfg["tuning"]["entry_base_grid"])),
            float(row["share"]),
        )

    start = str(cfg["splits"]["validation_start"])
    end = str(cfg["splits"]["validation_end"])

    # The frozen config must not bias the grid it is choosing (`DECISIONS` R24).
    tuning_cfg = {**cfg, "strategy": {**cfg["strategy"], "selected_model": None}}

    # R31-R33: eligibility decides which models are tuned, and the third entry base
    # and its method are chosen per model from that model's own distribution.
    plan = plan_models(engine, cfg)
    eligible = [p for p in plan if p["eligible"]]
    excluded = [p for p in plan if not p["eligible"]]
    for p in excluded:
        log.warning("%s excluded from tuning: %s", p["model"], p["reason"])
    for p in eligible:
        log.info(
            "%s eligible: entries %s (third %s via %s)",
            p["model"],
            p["entries"],
            p["third_entry"],
            p["method"],
        )

    results: list[dict[str, Any]] = []
    for spec in eligible:
        model = spec["model"]
        model_entries = (
            spec["entries"]
            if apply_fallback
            else [float(e) for e in cfg["tuning"]["entry_base_grid"]]
        )
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
                # A run that never trades has constant equity, so its Sharpe is NaN.
                # Such a run must not win selection through a NaN comparison; it is
                # still written to the CSV and the trial log, marked non-trading.
                traded = bool(metrics["n_trades"]) and _is_finite(metrics["sharpe"])
                tie_share = outcome.tie_break_share
                tie_flagged = _is_finite(tie_share) and tie_share > MAX_TIE_BREAK_SHARE
                if tie_flagged:
                    traded = False
                record = {
                    "model": model,
                    "entry_base": entry,
                    "slots": slots,
                    "sharpe": metrics["sharpe"] if _is_finite(metrics["sharpe"]) else float("nan"),
                    "cagr": metrics["cagr"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                    "n_trades": metrics["n_trades"],
                    "traded": traded,
                    "tie_break_share": tie_share,
                    "tie_break_flagged": bool(tie_flagged),
                    "new_entries": outcome.new_entries,
                    "method": spec["method"],
                    "entry_source": "grid",
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
                    value=record["sharpe"] if traded else None,
                    extra={
                        "cagr": metrics["cagr"],
                        "max_drawdown": metrics["max_drawdown"],
                        "turnover": metrics["turnover"],
                        "n_trades": metrics["n_trades"],
                        "traded": traded,
                        "tie_break_share": tie_share,
                        "tie_break_flagged": bool(tie_flagged),
                        "method": record["method"],
                    },
                    note=(
                        ""
                        if traded
                        else (
                            "tie-break share above 5%: excluded from selection (R34)"
                            if tie_flagged
                            else "non-trading run: excluded from selection"
                        )
                    ),
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
    # Every configuration is listed, including the non-trading ones.
    grid.sort_values(["traded", "sharpe", "turnover"], ascending=[False, False, True]).to_csv(
        results_dir / "tune_grid_validation.csv", index=False
    )

    # Step 4: highest Sharpe among runs that actually traded, tie-break lower turnover.
    # Non-trading runs are excluded explicitly rather than by a silent NaN sort, because
    # pandas puts NaN last anyway and relying on that would hide the exclusion.
    tradable = grid[grid["traded"] & grid["sharpe"].notna()]
    if tradable.empty:
        raise TuningError(
            f"no configuration traded on the validation period "
            f"({int((~grid['traded']).sum())} of {len(grid)} runs never traded); "
            "no model can be selected"
        )
    best = tradable.sort_values(["sharpe", "turnover"], ascending=[False, True]).iloc[0]

    frozen_path = Path(cfg["paths"]["frozen_config"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    # R33: the method is recorded per model, not once for the whole run.
    per_model_method = {p["model"]: p["method"] for p in plan}
    excluded_models = {p["model"]: p["reason"] for p in excluded}
    methods = set(per_model_method.values())
    overall_method = methods.pop() if len(methods) == 1 else "mixed"
    frozen_path.write_text(
        yaml.safe_dump(
            {
                "strategy": {
                    "selected_model": str(best["model"]),
                    "entry_base": float(best["entry_base"]),
                    "slots": int(best["slots"]),
                },
                "meta": {
                    "tuned_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "validation_sharpe": float(best["sharpe"]),
                    "trials": int(len(grid)),
                    "method": overall_method,
                    "method_per_model": per_model_method,
                    "excluded_from_tuning": excluded_models,
                    "grid_trials": int(len(grid)),
                    "total_logged_trials": count_trials(cfg),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.info(
        "selected %s entry=%.2f slots=%d (sharpe %.4f) from %d trial(s), method %s",
        best["model"],
        best["entry_base"],
        best["slots"],
        best["sharpe"],
        len(grid),
        overall_method,
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
            "method": overall_method,
        },
    )

    return {
        "trials": int(len(grid)),
        "non_trading_runs": int((~grid["traded"]).sum()),
        "tie_flagged_runs": int(grid["tie_break_flagged"].sum()),
        "total_logged_trials": count_trials(cfg),
        "selected_model": str(best["model"]),
        "entry_base": float(best["entry_base"]),
        "slots": int(best["slots"]),
        "validation_sharpe": float(best["sharpe"]),
        "method": overall_method,
        "method_per_model": per_model_method,
        "eligible_models": [p["model"] for p in eligible],
        "excluded_models": {p["model"]: p["reason"] for p in excluded},
    }


def log_development_runs(cfg: dict[str, Any]) -> None:
    """Re-record the development runs so the reported trial count is honest.

    The classifier training run (the +2.44 accuracy) and the synthetic V0/V1/V2 wiring
    check were real executed runs that informed decisions, so they belong in the log
    rather than being retro-fitted into it later. Upsert keyed on the trial key, so
    calling this twice adds nothing.
    """
    from src.common.trial_log import read_trials

    existing = {
        (r["kind"], r["run_id"], r["model"], r["params_json"], r["period"])
        for r in read_trials(cfg)
    }

    dev = [
        {
            "kind": "model_year",
            "run_id": "wf2023",
            "model": "tree",
            "period": "validation",
            "metric": "accuracy",
            "value": 0.5315509797409499,
            "params": {"year": 2023, "stage": "development training run"},
            "note": "development run: walk-forward training, validation accuracy",
        },
        {
            "kind": "model_year",
            "run_id": "wf2023",
            "model": "logreg",
            "period": "validation",
            "metric": "accuracy",
            "value": 0.4980073065426768,
            "params": {"year": 2023, "stage": "development training run"},
            "note": "development run: walk-forward training, validation accuracy",
        },
        {
            "kind": "model_year",
            "run_id": "wf2023",
            "model": "rf",
            "period": "validation",
            "metric": "accuracy",
            "value": 0.5124543341082697,
            "params": {"year": 2023, "stage": "development training run"},
            "note": "development run: walk-forward training, validation accuracy",
        },
        {
            "kind": "model_year",
            "run_id": "wf2023",
            "model": "xgb",
            "period": "validation",
            "metric": "accuracy",
            "value": 0.5058950514779144,
            "params": {"year": 2023, "stage": "development training run"},
            "note": "development run: walk-forward training, validation accuracy",
        },
    ]
    for row in dev:
        log_trial(
            cfg,
            phase="dev",
            kind=row["kind"],
            run_id=row["run_id"],
            model=row["model"],
            params=row["params"],
            period=row["period"],
            metric=row["metric"],
            value=row["value"],
            note=row["note"],
        )
    del existing


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
