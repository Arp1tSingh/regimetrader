"""The one-time test-period evaluation (`spec 06` section 8; `DECISIONS` R13, R23, R27).

This is the only production caller of `run_backtest(..., allow_test_period=True)`. It
runs once. The lock file makes that mechanical rather than a matter of memory: a second
invocation exits with code 3 and explains itself, and `--allow-rerun "<reason>"` is the
only way past it, which increments the run count and appends to `test_run_log.jsonl`.

Preconditions it refuses on: the frozen config is missing or has no
`strategy.selected_model`, or `--confirm` was not given.

Every run here uses the same engine, costs, slippage and period convention as
validation. Nothing is tuned in this module, and nothing it reads may be changed by its
output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ConfigError, load_config
from src.common.db import get_engine
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.common.trial_log import log_trial

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2
EXIT_ALREADY_RUN = 3

LOCK_NAME = "TEST_RUN_LOCK.json"
RERUN_LOG_NAME = "test_run_log.jsonl"
FROZEN_COPY_NAME = "frozen_config_used.yaml"


# ---------------------------------------------------------------- lock file


def _sha256(path: Path) -> str:
    """SHA-256 of a file, as lowercase hex."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    """The current commit, or `unknown` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - defensive
        return "unknown"


def lock_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["results"]) / LOCK_NAME


def read_lock(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The lock contents, or None when the test period has never been evaluated."""
    path = lock_path(cfg)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_lock(cfg: dict[str, Any], *, reason: str | None, count: int) -> dict[str, Any]:
    """Create or update the lock file and return it.

    Args:
        cfg: Loaded config.
        reason: The `--allow-rerun` reason, when this is a rerun.
        count: Run count to record.
    """
    from sqlalchemy import text

    engine = get_engine(cfg)
    with engine.begin() as conn:
        data_end = int(
            conn.execute(text("SELECT MAX(date_key) AS d FROM fact_price")).fetchone()[0] or 0
        )

    frozen = Path(cfg["paths"]["frozen_config"])
    record = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": _git_commit(),
        "config_hash": _sha256(Path(cfg["paths"]["db"])),
        "frozen_config_path": str(frozen),
        "frozen_config_hash": _sha256(frozen),
        "run_count": int(count),
        "data_end_date": str(data_end),
        "rerun_reason": reason,
    }
    path = lock_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    if reason:
        log_path = path.parent / RERUN_LOG_NAME
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


# ---------------------------------------------------------------- guards


def preconditions(cfg: dict[str, Any], *, confirmed: bool) -> str | None:
    """Return a refusal message, or None when the run may proceed."""
    frozen = Path(cfg["paths"]["frozen_config"])
    if not frozen.exists():
        return (
            f"frozen config not found at {frozen}; run `python -m src.backtest.tune` "
            "and settle the strategy before evaluating the test period"
        )
    import yaml

    data = yaml.safe_load(frozen.read_text(encoding="utf-8")) or {}
    if not (data.get("strategy") or {}).get("selected_model"):
        return f"frozen config at {frozen} has no strategy.selected_model"
    if not confirmed:
        return "--confirm was not given; the test period is evaluated exactly once"
    return None


# ---------------------------------------------------------------- runs


def _run_one(
    cfg: dict[str, Any],
    engine: Any,
    run_id: str,
    provider_name: str,
    start: str,
    end: str,
    model_name: str | None,
    **kwargs: Any,
) -> Any:
    """Run one strategy/ablation/baseline on the test period."""
    from src.backtest.engine import run_backtest
    from src.backtest.strategies import make_weights_fn

    return run_backtest(
        cfg,
        make_weights_fn(provider_name, cfg),
        start,
        end,
        run_id,
        engine=engine,
        model_name=model_name,
        allow_test_period=True,
        **kwargs,
    )


def yearly_metrics(equity: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Metrics for each calendar year in an equity curve (`06` section 7.3)."""
    from src.backtest.metrics import cagr, daily_returns, max_drawdown, sharpe

    frame = equity.copy()
    frame["year"] = frame.index.year
    rows = []
    for year, group in frame.groupby("year"):
        series = group["equity"].astype(float)
        returns = daily_returns(series)
        rows.append(
            {
                "year": int(year),
                "n_days": int(len(series)),
                "start_equity": float(series.iloc[0]),
                "end_equity": float(series.iloc[-1]),
                "return": float(series.iloc[-1] / series.iloc[0] - 1.0),
                "sharpe": sharpe(returns, float(cfg["backtest"]["risk_free_annual"])),
                "max_drawdown": max_drawdown(series),
                "cagr": cagr(series),
            }
        )
    return rows


def regime_breakdown(
    engine: Any, cfg: dict[str, Any], start_key: int, end_key: int
) -> pd.DataFrame:
    """Return by regime for the test period, from the persisted portfolio (`06` section 7.4)."""
    from src.common.db import read_sql

    return read_sql(
        engine,
        "SELECT r.regime_label AS regime, "
        "       AVG(p.daily_return) AS mean_daily_return, "
        "       COUNT(*) AS n_days, "
        "       SUM(CASE WHEN p.daily_return > 0 THEN 1 ELSE 0 END) AS n_up_days "
        "FROM fact_portfolio_daily p "
        "JOIN fact_market_regime m ON m.date_key = p.date_key "
        "JOIN dim_regime r ON r.regime_key = m.regime_key "
        "WHERE p.run_id = :run AND p.date_key BETWEEN :a AND :b "
        "GROUP BY r.regime_label ORDER BY r.regime_label",
        {"run": "backtest_main", "a": start_key, "b": end_key},
    )


def score_classifiers(cfg: dict[str, Any], start_key: int, end_key: int) -> list[dict[str, Any]]:
    """Append test-year rows to `classifier_metrics.csv` (`04` section 7; R23)."""
    from src.common.db import get_engine as _engine
    from src.features.panel import TARGET_COL, build_panel
    from src.models.evaluate import write_classifier_metrics

    engine = _engine(cfg)
    panel = build_panel(engine, cfg).dropna(subset=[TARGET_COL])
    results = []
    for model in cfg["tuning"]["models"]:
        window = panel[(panel["date_key"] >= start_key) & (panel["date_key"] <= end_key)]
        if window.empty:
            continue
        from src.common.db import read_sql

        frame = read_sql(
            engine,
            "SELECT f.date_key AS date_key, s.symbol AS symbol, f.prob_up AS prob_up "
            "FROM fact_signal f "
            "JOIN dim_model m ON m.model_key = f.model_key "
            "JOIN dim_stock s ON s.stock_key = f.stock_key "
            "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b",
            {"model": model, "a": start_key, "b": end_key},
        )
        if frame.empty:
            continue
        joined = window.merge(frame, on=["date_key", "symbol"], how="inner")
        if joined.empty:
            continue
        year = int(str(joined["date_key"].min())[:4])
        record = write_classifier_metrics(
            Path(cfg["paths"]["results"]) / "classifier_metrics.csv",
            joined[[TARGET_COL, "prob_up"]],
            model,
            "test",
            year,
        )
        results.append(record)
    return results


def run_b4_test(cfg: dict[str, Any], engine: Any, start: str, end: str) -> pd.DataFrame:
    """The B4 permutation null on the test period (`06` section 6)."""
    from src.backtest.baselines import b4_distribution

    model = str(cfg["strategy"]["selected_model"])
    return b4_distribution(
        cfg,
        engine,
        start,
        end,
        sims=int(cfg["backtest"]["random_baseline_sims"]),
        model=model,
    )


def run(
    cfg: dict[str, Any],
    *,
    confirmed: bool,
    allow_rerun: str | None,
    with_b4: bool = True,
) -> dict[str, Any]:
    """Execute the one-time test evaluation.

    Args:
        cfg: Loaded config.
        confirmed: `--confirm` was given.
        allow_rerun: Reason for overriding an existing lock.
        with_b4: Run the B4 null on the test period.

    Returns:
        A summary dict; the CLI maps it to an exit code.
    """
    log = get_logger("src.backtest.final_test", cfg)
    engine = get_engine(cfg)

    existing = read_lock(cfg)
    if existing is not None and not allow_rerun:
        return {
            "refused": True,
            "message": (
                f"the test period has already been evaluated "
                f"(run_count {existing.get('run_count')}, at {existing.get('timestamp')}). "
                'It is evaluated exactly once. Pass --allow-rerun "<reason>" to '
                "override, which increments the count and is recorded in "
                f"{RERUN_LOG_NAME}."
            ),
        }

    refusal = preconditions(cfg, confirmed=confirmed)
    if refusal:
        return {"refused": True, "message": refusal}

    frozen = Path(cfg["paths"]["frozen_config"])
    frozen_used = Path(cfg["paths"]["results"]) / FROZEN_COPY_NAME
    frozen_used.parent.mkdir(parents=True, exist_ok=True)
    frozen_used.write_text(frozen.read_text(encoding="utf-8"), encoding="utf-8")

    lock = write_lock(
        cfg,
        reason=allow_rerun,
        count=int(existing.get("run_count", 0)) + 1 if existing else 1,
    )
    log.info(
        "test-period lock written: run_count %d, commit %s",
        lock["run_count"],
        lock["git_commit"],
    )

    start = str(cfg["splits"]["test_start"])
    end = pd.to_datetime(str(lock["data_end_date"])).strftime("%Y-%m-%d")
    model = str(cfg["strategy"]["selected_model"])
    log.info("test period %s to %s, selected model %s", start, end, model)

    # The primary run is the only one allowed to write signal and target_weight (R3).
    main = _run_one(
        cfg,
        engine,
        "backtest_main",
        "V2",
        start,
        end,
        model,
        persist=True,
        write_signals=True,
    )

    from src.backtest.metrics import block_bootstrap_sharpe, daily_returns, summarize

    others = {
        "test_V0": ("V0", {}),
        "test_V1": ("V1", {}),
        "test_V2_nocost": ("V2", {"costs_enabled": False}),
        "test_B1": ("B1", {"fractional_shares": True, "band": 0.0, "stop_loss_enabled": False}),
        "test_B2": ("B2", {"band": 0.0, "stop_loss_enabled": False}),
        "test_B3": ("B3", {"stop_loss_enabled": False}),
    }
    results = {"backtest_main": main}
    for run_id, (provider, kwargs) in others.items():
        log.info("running %s", run_id)
        results[run_id] = _run_one(
            cfg,
            engine,
            run_id,
            provider,
            start,
            end,
            model if provider.startswith("V") else None,
            persist=True,
            **kwargs,
        )

    rows = []
    boot = cfg["backtest"]["bootstrap"]
    for run_id, outcome in results.items():
        metrics = summarize(outcome.equity, outcome.trades, cfg)
        interval = block_bootstrap_sharpe(
            daily_returns(outcome.equity["equity"]),
            block_days=int(boot["block_days"]),
            resamples=int(boot["resamples"]),
            ci_low=float(boot["ci_low"]),
            ci_high=float(boot["ci_high"]),
            rf_annual=float(cfg["backtest"]["risk_free_annual"]),
            seed=int(cfg["project"]["seed"]),
        )
        rows.append(
            {
                "run_id": run_id,
                "period": "test",
                "model": model if run_id.startswith(("backtest", "test_V")) else "",
                "sharpe": metrics["sharpe"],
                "sharpe_ci_low": interval["ci_low"],
                "sharpe_ci_high": interval["ci_high"],
                "cagr": metrics["cagr"],
                "max_drawdown": metrics["max_drawdown"],
                "sortino": metrics["sortino"],
                "calmar": metrics["calmar"],
                "turnover": metrics["turnover"],
                "n_trades": metrics["n_trades"],
                "exposure": metrics["exposure"],
                "final_equity": metrics["final_equity"],
                "tie_break_share": outcome.tie_break_share,
            }
        )

    results_dir = Path(cfg["paths"]["results"])
    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "summary_test.csv", index=False)

    pd.DataFrame(yearly_metrics(main.equity, cfg)).to_csv(
        results_dir / "yearly_test.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "run_id": r["run_id"],
                "sharpe": r["sharpe"],
                "ci_low": r["sharpe_ci_low"],
                "ci_high": r["sharpe_ci_high"],
                "block_days": int(boot["block_days"]),
                "resamples": int(boot["resamples"]),
            }
            for r in rows
        ]
    ).to_csv(results_dir / "bootstrap_sharpe.csv", index=False)

    start_key = int(start.replace("-", ""))
    end_key = int(lock["data_end_date"])

    breakdown = regime_breakdown(engine, cfg, start_key, end_key)
    breakdown.to_csv(results_dir / "regime_breakdown.csv", index=False)

    classifier_rows = score_classifiers(cfg, start_key, end_key)

    distribution = None
    if with_b4:
        log.info("running the B4 permutation null on the test period")
        distribution = run_b4_test(cfg, engine, start, end)
        distribution.to_csv(results_dir / "b4_sharpe_distribution_test.csv", index=False)

    for row in rows:
        log_trial(
            cfg,
            phase="final_test",
            kind="other",
            run_id=row["run_id"],
            model=row["model"] or row["run_id"],
            params={"period": "test", "start": start, "end": end},
            period="test",
            metric="sharpe",
            value=row["sharpe"],
            extra={
                "cagr": row["cagr"],
                "max_drawdown": row["max_drawdown"],
                "n_trades": row["n_trades"],
                "sharpe_ci": [row["sharpe_ci_low"], row["sharpe_ci_high"]],
            },
            note="final test-period evaluation (DECISIONS R13)",
        )

    write_manifest(
        cfg,
        "final_test",
        engine,
        extra={
            "period": "test",
            "start": start,
            "end": end,
            "run_count": lock["run_count"],
            "runs": list(results),
        },
    )

    log.info(
        "wrote summary_test.csv, yearly_test.csv, bootstrap_sharpe.csv, regime_breakdown.csv%s",
        ", b4_sharpe_distribution_test.csv" if with_b4 else "",
    )

    return {
        "refused": False,
        "summary": summary,
        "yearly": yearly_metrics(main.equity, cfg),
        "classifier_rows": classifier_rows,
        "b4": distribution,
        "lock": lock,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the test period once. This cannot be repeated casually."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--confirm", action="store_true", help="Required. Confirms the one-time evaluation."
    )
    parser.add_argument(
        "--allow-rerun",
        default=None,
        metavar="REASON",
        help="Override the lock after a completed run. The reason is recorded.",
    )
    parser.add_argument("--no-b4", action="store_true", help="Skip the B4 test-period null.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    outcome = run(cfg, confirmed=args.confirm, allow_rerun=args.allow_rerun, with_b4=not args.no_b4)
    if outcome["refused"]:
        print(outcome["message"], file=sys.stderr)
        return EXIT_ALREADY_RUN if read_lock(cfg) is not None else EXIT_DATA_FAILURE

    summary: pd.DataFrame = outcome["summary"]
    print()
    print(
        summary[
            [
                "run_id",
                "sharpe",
                "sharpe_ci_low",
                "sharpe_ci_high",
                "cagr",
                "max_drawdown",
                "n_trades",
            ]
        ].to_string(index=False)
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
