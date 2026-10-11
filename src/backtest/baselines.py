"""Baselines and the permutation null (`spec 06` section 6).

B1 buy-and-hold, B2 equal-weight monthly, B3 the trend rule, and B4 the
**symbol-permutation null**: each stock keeps another stock's `prob_up` series, so
exposure, regime behaviour and turnover stay roughly equal while the link between a
prediction and its own stock's returns is broken. B4 answers the only question that
matters for a Sharpe number: would a strategy that means nothing produce this?

Everything here runs on the period it is given, under the same engine, costs and
slippage as the strategy. Nothing is persisted.
"""

from __future__ import annotations

import argparse
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

#: `spec 06` section 6 engine settings per baseline.
BASELINE_SETTINGS: dict[str, dict[str, Any]] = {
    "B1": {"fractional_shares": True, "band": 0.0, "stop_loss_enabled": False},
    "B2": {"band": 0.0, "stop_loss_enabled": False},
    "B3": {"stop_loss_enabled": False},
}


def permuted_signals(signals: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Relabel each stock's `prob_up` series with another stock's.

    One fixed permutation per simulation, seeded, so the null is reproducible. The
    permutation is applied within each day across symbols, so every day keeps the same
    multiset of probabilities and only the stock each one belongs to changes.

    Args:
        signals: Rows carrying `date_key`, `symbol` and `prob_up`.
        seed: Seed for the permutation.

    Returns:
        A copy with `prob_up` permuted across symbols within each day, in the same row
        order as the input.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out = signals.copy()
    # A copy is required: `to_numpy` returns a read-only view under pandas 3.
    values = out["prob_up"].to_numpy(dtype=float).copy()

    for _day, index in out.groupby("date_key", sort=True).groups.items():
        positions = out.index.get_indexer(index)
        values[positions] = values[rng.permutation(positions)]

    out["prob_up"] = values
    return out


def b4_distribution(
    cfg: dict[str, Any],
    engine: Any,
    start: str,
    end: str,
    *,
    sims: int | None = None,
    model: str,
    log: Any = None,
) -> pd.DataFrame:
    """Run B4 and return one row per simulation.

    Each simulation writes the permuted probabilities under a **dedicated model key**
    (`<model>_b4perm`), never under the real model's key, so the strategy's own
    `fact_signal` rows cannot be disturbed. The temporary rows are removed in a
    `finally` block whatever happens.

    Args:
        cfg: Loaded config.
        engine: Warehouse engine.
        start: Period start, `YYYY-MM-DD`.
        end: Period end, `YYYY-MM-DD`.
        sims: Override for `backtest.random_baseline_sims`.
        model: The model whose probabilities are permuted.
        log: Optional logger.

    Returns:
        Columns `sim`, `sharpe`, `cagr`, `max_drawdown`, `n_trades`, `turnover`,
        `exposure`, `tie_break_share`, `traded`.
    """
    from sqlalchemy import text

    from src.backtest.engine import run_backtest
    from src.backtest.metrics import summarize, weight_exposure
    from src.backtest.strategies import make_weights_fn
    from src.common.db import execute_many, read_sql
    from src.features.panel import TARGET_COL, build_panel

    count = int(sims if sims is not None else cfg["backtest"]["random_baseline_sims"])
    seed = int(cfg["project"]["seed"])
    perm_model = f"{model}_b4perm"

    start_key = int(str(start).replace("-", ""))
    end_key = int(str(end).replace("-", ""))

    # Probabilities live in `fact_signal`, not in the feature panel.
    frame = read_sql(
        engine,
        "SELECT f.date_key AS date_key, s.symbol AS symbol, f.prob_up AS prob_up "
        "FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE m.model_name = :model AND f.date_key BETWEEN :a AND :b "
        "ORDER BY f.date_key, s.symbol",
        {"model": model, "a": start_key, "b": end_key},
    )
    if frame.empty:
        raise ValueError(f"no stored probabilities for model {model}; run `make train` first")

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT OR IGNORE INTO dim_model (model_name, version, trained_through, "
                "params_json) VALUES (:name, 'b4perm', 'permutation-null', '{}')"
            ),
            {"name": perm_model},
        )
        perm_key = int(
            conn.execute(
                text("SELECT model_key FROM dim_model WHERE model_name = :n"),
                {"n": perm_model},
            ).fetchone()[0]
        )
        stocks = dict(
            conn.execute(
                text("SELECT symbol, stock_key FROM dim_stock WHERE instrument_type = 'EQUITY'")
            ).fetchall()
        )
        regimes = dict(
            conn.execute(text("SELECT date_key, regime_key FROM fact_market_regime")).fetchall()
        )

    rows = []
    try:
        for sim in range(count):
            shuffled = permuted_signals(frame, seed + sim)
            payload = [
                {
                    "d": int(row.date_key),
                    "s": stocks[str(row.symbol)],
                    "r": int(regimes.get(int(row.date_key), 0) or 0),
                    "p": float(row.prob_up) if pd.notna(row.prob_up) else None,
                }
                for row in shuffled.itertuples(index=False)
                if str(row.symbol) in stocks
            ]

            with engine.begin() as conn:
                conn.execute(text("DELETE FROM fact_signal WHERE model_key = :k"), {"k": perm_key})
                execute_many(
                    conn,
                    "INSERT OR REPLACE INTO fact_signal (date_key, stock_key, model_key, "
                    "regime_key, prob_up, signal, target_weight) "
                    "VALUES (:d, :s, :k, :r, :p, 0, 0.0)",
                    [{**row, "k": perm_key} for row in payload],
                )

            provider = make_weights_fn("V2", cfg)
            outcome = run_backtest(
                cfg,
                provider,
                start,
                end,
                "b4_perm",
                engine=engine,
                model_name=perm_model,
                persist=False,
            )
            metrics = summarize(outcome.equity, outcome.trades, cfg)
            rows.append(
                {
                    "sim": sim,
                    "sharpe": metrics["sharpe"],
                    "cagr": metrics["cagr"],
                    "max_drawdown": metrics["max_drawdown"],
                    "n_trades": metrics["n_trades"],
                    "turnover": metrics["turnover"],
                    "exposure": metrics["exposure"],
                    "weight_exposure": weight_exposure(outcome.equity),
                    "tie_break_share": outcome.tie_break_share,
                    "traded": bool(metrics["n_trades"]),
                }
            )
            if log is not None:
                log.info(
                    "B4 sim %d/%d: sharpe %.4f, %d trade(s)",
                    sim + 1,
                    count,
                    metrics["sharpe"],
                    metrics["n_trades"],
                )
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_signal WHERE model_key = :k"), {"k": perm_key})
            conn.execute(text("DELETE FROM dim_model WHERE model_key = :k"), {"k": perm_key})

    return pd.DataFrame(rows)


def run(
    cfg: dict[str, Any],
    *,
    start: str,
    end: str,
    period: str,
    sims: int | None = None,
    with_b4: bool = True,
) -> dict[str, Any]:
    """Run B1 to B3, write `summary_<period>.csv`, and optionally run B4.

    Args:
        cfg: Loaded config.
        start: Period start, `YYYY-MM-DD`.
        end: Period end, `YYYY-MM-DD`.
        period: `validation` or `test`; names the output file.
        sims: Override for the B4 simulation count.
        with_b4: Whether to run the permutation null.

    Returns:
        A summary dict of the rows written.
    """
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import block_bootstrap_sharpe, daily_returns, summarize
    from src.backtest.strategies import make_weights_fn

    log = get_logger("src.backtest.baselines", cfg)
    engine = get_engine(cfg)
    boot = cfg["backtest"]["bootstrap"]
    seed = int(cfg["project"]["seed"])
    model = str(cfg["strategy"].get("selected_model") or cfg["tuning"]["models"][0])

    results: list[dict[str, Any]] = []
    for name in ("B1", "B2", "B3"):
        settings = BASELINE_SETTINGS[name]
        provider = make_weights_fn(name, cfg)
        outcome = run_backtest(
            cfg,
            provider,
            start,
            end,
            f"{period}_{name}",
            engine=engine,
            model_name=None,
            persist=False,
            **settings,
        )
        metrics = summarize(outcome.equity, outcome.trades, cfg)
        interval = block_bootstrap_sharpe(
            daily_returns(outcome.equity["equity"]),
            block_days=int(boot["block_days"]),
            resamples=int(boot["resamples"]),
            ci_low=float(boot["ci_low"]),
            ci_high=float(boot["ci_high"]),
            rf_annual=float(cfg["backtest"]["risk_free_annual"]),
            seed=seed,
        )
        row = {
            "run_id": f"{period}_{name}",
            "period": period,
            "baseline": name,
            "model": "",
            "sharpe": metrics["sharpe"],
            "sharpe_ci_low": interval["ci_low"],
            "sharpe_ci_high": interval["ci_high"],
            "cagr": metrics["cagr"],
            "max_drawdown": metrics["max_drawdown"],
            "calmar": metrics["calmar"],
            "sortino": metrics["sortino"],
            "turnover": metrics["turnover"],
            "n_trades": metrics["n_trades"],
            "exposure": metrics["exposure"],
            "final_equity": metrics["final_equity"],
            "tie_break_share": outcome.tie_break_share,
        }
        results.append(row)
        log_trial(
            cfg,
            phase="backtest",
            kind="other",  # `07` section 4 pins kind to the five listed values
            run_id=f"{period}_{name}",
            model=name,
            params=settings,
            period=period,
            metric="sharpe",
            value=metrics["sharpe"],
            extra={"n_trades": metrics["n_trades"], "cagr": metrics["cagr"]},
            note=f"{name} baseline ({period})",
        )
        log.info(
            "%s: sharpe %.4f [%.4f, %.4f], cagr %.4f, %d trade(s)",
            name,
            metrics["sharpe"],
            interval["ci_low"],
            interval["ci_high"],
            metrics["cagr"],
            metrics["n_trades"],
        )

    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / f"summary_{period}.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path)

    distribution = None
    if with_b4:
        distribution = b4_distribution(cfg, engine, start, end, sims=sims, model=model, log=log)
        distribution_path = results_dir / f"b4_sharpe_distribution_{period}.csv"
        distribution.to_csv(distribution_path, index=False)
        log.info("wrote %s", distribution_path)

    write_manifest(
        cfg,
        f"baselines_{period}",
        engine,
        extra={"period": period, "rows": len(results), "model": model},
    )
    return {"rows": len(results), "path": str(summary_path), "b4": distribution}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baselines B1-B4.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--period", default="validation", choices=["validation", "test"])
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--sims", type=int, default=None)
    parser.add_argument("--no-b4", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    start = args.start or str(cfg["splits"][f"{args.period}_start"])
    end = args.end or str(cfg["splits"][f"{args.period}_end"])
    summary = run(
        cfg,
        start=start,
        end=end,
        period=args.period,
        sims=args.sims,
        with_b4=not args.no_b4,
    )
    print(f"{summary['rows']} baseline row(s) -> {summary['path']}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
