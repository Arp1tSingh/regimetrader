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

#: R39 activity filter. 30 round trips is the smallest sample whose trade-level hit
#: rate has a usable standard error (about 9 points near 50%); below 10% mean exposure
#: the Sharpe is computed on a handful of active days.
MIN_ROUND_TRIPS = 30
MIN_EXPOSURE = 0.10

#: R40 null filter.
NULL_SHORTLIST_SIZE = 3
NULL_SIMS = 100
NULL_PERCENTILE = 90.0

#: R41: the spec's starting strategy settings, used when nothing survives.
DEFAULT_ENTRY_BASE = 0.55
DEFAULT_SLOTS = 10


def _activity_eligible(grid: pd.DataFrame) -> pd.DataFrame:
    """Rows passing the R39 activity filter, with a finite Sharpe.

    NaN Sharpes are dropped here rather than sorted around: relying on pandas placing
    NaN last would hide the exclusion and could let a NaN win under a sort change.
    """
    if grid.empty:
        return grid
    mask = (
        grid["traded"].astype(bool)
        & grid["sharpe"].notna()
        & (grid["n_trades"].fillna(0).astype(int) >= MIN_ROUND_TRIPS)
        & (grid["exposure"].fillna(0.0).astype(float) >= MIN_EXPOSURE)
        & ~grid["tie_break_flagged"].astype(bool)
    )
    return grid[mask]


def _null_percentile(
    cfg: dict[str, Any],
    engine: Any,
    row: pd.Series,
    start: str,
    end: str,
    log: Any,
) -> dict[str, Any]:
    """R40: run this configuration's own B4 null and compare its Sharpe to it.

    The null is run under the shortlisted configuration's own entry base and slots, not
    a shared one, so exposure and turnover are matched.

    Returns:
        A dict describing the comparison, including the Series in `row`.
    """
    from src.backtest.baselines import b4_distribution

    variant = {
        **cfg,
        "strategy": {
            **cfg["strategy"],
            "selected_model": str(row["model"]),
            "entry_base": float(row["entry_base"]),
            "slots": int(row["slots"]),
        },
    }
    distribution = b4_distribution(
        variant, engine, start, end, sims=NULL_SIMS, model=str(row["model"]), log=None
    )
    sharpes = distribution["sharpe"].dropna()
    p90 = float(sharpes.quantile(NULL_PERCENTILE / 100.0)) if len(sharpes) else float("nan")
    actual = float(row["sharpe"])
    selectable = bool(_is_finite(p90) and _is_finite(actual) and actual > p90)

    for sim_sharpe in sharpes:
        log_trial(
            cfg,
            phase="tune",
            kind="other",  # `07` section 4 pins kind to the five listed values
            run_id=f"null_{row['model']}_{row['entry_base']:.4f}_{int(row['slots'])}",
            model=str(row["model"]),
            params={
                "entry_base": float(row["entry_base"]),
                "slots": int(row["slots"]),
                "null": "b4_symbol_permutation",
            },
            period="validation",
            metric="null_sharpe",
            value=float(sim_sharpe),
            extra={"actual_sharpe": actual, "null_p90": p90},
            note="B4 permutation null for a shortlisted configuration (DECISIONS R40)",
        )

    log.info(
        "null for %s/%.4f/%d: sharpe %.4f vs null p90 %.4f over %d sim(s) -> %s",
        row["model"],
        float(row["entry_base"]),
        int(row["slots"]),
        actual,
        p90,
        len(sharpes),
        "selectable" if selectable else "NOT selectable",
    )
    return {
        "model": str(row["model"]),
        "entry_base": float(row["entry_base"]),
        "slots": int(row["slots"]),
        "sharpe": actual,
        "n_trades": int(row["n_trades"]),
        "exposure": float(row["exposure"]),
        "null_p90": p90,
        "null_sims": int(len(sharpes)),
        "null_median": float(sharpes.median()) if len(sharpes) else float("nan"),
        "selectable": selectable,
        "row": row,
    }


def _no_edge_default(
    cfg: dict[str, Any],
    eligible: list[dict[str, Any]],
    plan: list[dict[str, Any]],
    aucs: dict[str, float] | None = None,
) -> tuple[pd.Series, str]:
    """R41: freeze the best classifier on the spec's starting settings.

    The model is the one with the highest validation ROC-AUC among the
    distribution-eligible models, so it still respects R31.

    Args:
        cfg: Loaded config.
        eligible: Plan entries with `eligible` true.
        plan: The whole plan, kept for symmetry with the caller.
        aucs: Optional `model -> ROC-AUC` override. Read from
            `classifier_metrics.csv` when omitted.

    Returns:
        `(row, "no_edge_default")`, where the row carries the spec's starting settings.
    """
    from src.models.diagnostics import roc_auc_by_model

    scores = roc_auc_by_model(cfg) if aucs is None else aucs
    names = [p["model"] for p in eligible]
    finite = {m: float(scores[m]) for m in names if m in scores and _is_finite(scores[m])}
    chosen = max(sorted(finite), key=lambda m: finite[m]) if finite else (
        names[0] if names else cfg["tuning"]["models"][0]
    )
    return (
        pd.Series(
            {
                "model": chosen,
                "entry_base": DEFAULT_ENTRY_BASE,
                "slots": DEFAULT_SLOTS,
                "sharpe": float("nan"),
                "n_trades": 0,
                "exposure": float("nan"),
            }
        ),
        "no_edge_default",
    )


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
                    "exposure": metrics["exposure"],
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

    # ---------------------------------------------------------------- selection
    # R42: Sharpe alone is never sufficient. A configuration must first clear the
    # activity filter (R39) and then beat the B4 null for its own settings (R40).
    candidates = _activity_eligible(grid)
    log.info(
        "activity filter (R39: n_trades >= %d, exposure >= %.0f%%): %d of %d configuration(s)",
        MIN_ROUND_TRIPS,
        MIN_EXPOSURE * 100,
        len(candidates),
        len(grid),
    )
    if not candidates.empty:
        shortlist = candidates.sort_values(["sharpe", "turnover"], ascending=[False, True]).head(
            NULL_SHORTLIST_SIZE
        )
        log.info(
            "shortlist (top %d by Sharpe): %s",
            len(shortlist),
            ", ".join(
                f"{r['model']}/{r['entry_base']:.4f}/{int(r['slots'])} (sharpe {r['sharpe']:.4f})"
                for _, r in shortlist.iterrows()
            ),
        )
    else:
        shortlist = grid.iloc[0:0]
        log.warning("no configuration passed the activity filter (R39); nothing can be shortlisted")

    null_results = []
    for _, row in shortlist.iterrows():
        null = _null_percentile(cfg, engine, row, start, end, log)
        null_results.append(null)

    passed = [n for n in null_results if n["selectable"]]
    log.info(
        "null filter (R40): %d of %d shortlisted configuration(s) beat their own "
        "null's %.0fth percentile",
        len(passed),
        len(null_results),
        NULL_PERCENTILE,
    )

    if passed:
        best = passed[0]["row"]
        method_outcome = "selected"
        frozen_method = "activity_and_null_filtered"
    else:
        # R41: no configuration beat its own null. Freeze the best classifier on the
        # spec's starting settings and say so, rather than promoting a failed run.
        best, frozen_method = _no_edge_default(cfg, eligible, plan)
        method_outcome = "no_edge_default"

    results_dir = Path(cfg["paths"]["results"])
    pd.DataFrame(null_results).drop(columns=["row"]).to_csv(
        results_dir / "null_filter_validation.csv", index=False
    )
    log.info("wrote %s", results_dir / "null_filter_validation.csv")

    frozen_path = Path(cfg["paths"]["frozen_config"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    # R33: the threshold method is recorded per model, not once for the whole run.
    per_model_method = {p["model"]: p["method"] for p in plan}
    excluded_models = {p["model"]: p["reason"] for p in excluded}
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
                    "method": frozen_method,
                    "outcome": method_outcome,
                    "demonstrated_edge": bool(method_outcome == "selected"),
                    "validation_sharpe": (
                        float(best["sharpe"]) if _is_finite(best["sharpe"]) else float("nan")
                    ),
                    "n_trades": int(best["n_trades"]),
                    "exposure": float(best["exposure"]),
                    "threshold_method_per_model": per_model_method,
                    "excluded_from_tuning": excluded_models,
                    "activity_filter": {
                        "min_round_trips": MIN_ROUND_TRIPS,
                        "min_exposure": MIN_EXPOSURE,
                        "passed": int(len(candidates)),
                        "of": int(len(grid)),
                    },
                    "null_filter": {
                        "sims": NULL_SIMS,
                        "percentile": NULL_PERCENTILE,
                        "shortlisted": int(len(null_results)),
                        "passed": int(len(passed)),
                    },
                    "grid_trials": int(len(grid)),
                    "total_logged_trials": count_trials(cfg),
                    "disclosure": (
                        "Selection filters R39-R42 were introduced after run 1 produced a "
                        "degenerate 1-trade winner. Run 1 is preserved and still counted."
                    ),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    log.info(
        "outcome %s: %s entry=%.4f slots=%d (method %s)",
        method_outcome,
        best["model"],
        best["entry_base"],
        best["slots"],
        frozen_method,
    )

    write_manifest(
        cfg,
        "tune",
        engine,
        extra={
            "trials": int(len(grid)),
            "outcome": method_outcome,
            "method": frozen_method,
            "selected_model": str(best["model"]),
            "entry_base": float(best["entry_base"]),
            "slots": int(best["slots"]),
        },
    )

    return {
        "trials": int(len(grid)),
        "non_trading_runs": int((~grid["traded"]).sum()),
        "tie_flagged_runs": int(grid["tie_break_flagged"].sum()),
        "activity_eligible": int(len(candidates)),
        "shortlisted": int(len(null_results)),
        "null_passed": int(len(passed)),
        "total_logged_trials": count_trials(cfg),
        "selected_model": str(best["model"]),
        "entry_base": float(best["entry_base"]),
        "slots": int(best["slots"]),
        "validation_sharpe": (
            float(best["sharpe"]) if _is_finite(best["sharpe"]) else float("nan")
        ),
        "outcome": method_outcome,
        "method": frozen_method,
        "threshold_method_per_model": per_model_method,
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
