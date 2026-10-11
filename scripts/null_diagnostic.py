"""Read-only diagnostic: the full B4 null distribution for the frozen configuration.

Recomputes the null for rf / entry_base 0.504415 / slots 10 so exposure, trade count
and CAGR can be reported alongside Sharpe. Those columns were never persisted per
simulation, and `b4_sharpe_distribution_validation.csv` was overwritten by a timing
run, so they have to be recomputed.

This logs **no** trials and writes no frozen config: it calls `b4_distribution`
directly rather than `tune`, so the trial log and the freeze are untouched. The
permutation is seeded per simulation, so the recomputed Sharpes must match the ones
already in `trial_log.csv` exactly; that equality is checked by parsing the simulation
index out of `run_id` and is what makes this a diagnostic rather than a re-run.

Writes `null_diagnostic_frozen.csv` and `null_diagnostic_frozen_summary.csv`.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.backtest.baselines import b4_distribution
from src.common.config import load_config
from src.common.db import get_engine

SIM = re.compile(r"_sim(\d+)$")


def spread(values: pd.Series) -> dict[str, float]:
    """Median and p10/p90 for one column, ignoring NaN."""
    clean = pd.Series(values).dropna()
    if clean.empty:
        return {"p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "p10": float(np.percentile(clean, 10)),
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
    }


def reference_runs(cfg, engine, model: str, entry: float, slots: int) -> pd.DataFrame:
    """The real run and the three baselines, on the same period and settings."""
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import summarize, weight_exposure
    from src.backtest.strategies import make_weights_fn as build

    rows = []
    labels = [
        ("V2 real (frozen)", "V2", model, 1.0),
        ("B1 index buy-and-hold", "B1", None, None),
        ("B2 equal weight", "B2", None, None),
        ("B3 trend sma20>sma50", "B3", None, None),
    ]
    for label, provider_name, mdl, _ in labels:
        provider = build(provider_name, cfg)
        kwargs = (
            {"fractional_shares": True, "band": 0.0, "stop_loss_enabled": False}
            if provider_name == "B1"
            else (
                {"band": 0.0, "stop_loss_enabled": False}
                if provider_name == "B2"
                else ({"stop_loss_enabled": False} if provider_name == "B3" else {})
            )
        )
        outcome = run_backtest(
            cfg,
            provider,
            "2023-01-01",
            "2023-12-31",
            f"null_diag_{provider_name}",
            engine=engine,
            model_name=mdl,
            persist=False,
            **kwargs,
        )
        m = summarize(outcome.equity, outcome.trades, cfg)
        rows.append(
            {
                "run": label,
                "sharpe": m["sharpe"],
                "cagr": m["cagr"],
                "n_trades": m["n_trades"],
                "days_in_position": m["exposure"],
                "weight_exposure": weight_exposure(outcome.equity),
            }
        )
    del entry, slots
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)

    model = str(cfg["strategy"]["selected_model"])
    entry = float(cfg["strategy"]["entry_base"])
    slots = int(cfg["strategy"]["slots"])
    sims = 100
    seed = int(cfg["project"]["seed"])

    print(f"frozen configuration: {model} / entry_base {entry:.6f} / slots {slots}")
    print(f"{sims} simulations, permutation seed = project.seed({seed}) + sim\n")

    print("computing reference runs (real V2 and the three baselines)...")
    references = reference_runs(cfg, engine, model, entry, slots)
    print(references.to_string(index=False))
    print()

    distribution = b4_distribution(cfg, engine, "2023-01-01", "2023-12-31", sims=sims, model=model)

    # Determinism check: match logged rows to simulations by the parsed index, NOT by
    # a lexicographic sort of run_id, which puts sim10 before sim2.
    trials = pd.read_csv(f"{cfg['paths']['results']}/trial_log.csv")
    prefix = f"null_{model}_{entry:.4f}_{slots}_sim"
    logged = trials[
        (trials["metric"] == "null_sharpe") & trials["run_id"].astype(str).str.startswith(prefix)
    ].copy()
    match = "no logged rows to compare"
    if not logged.empty:
        logged["sim"] = logged["run_id"].astype(str).str.extract(SIM).astype(int)
        logged = logged.set_index("sim").loc[distribution.index]
        delta = float(
            np.nanmax(np.abs(logged["value"].to_numpy() - distribution["sharpe"].to_numpy()))
        )
        match = f"max |recomputed - trial_log| = {delta:.3e}"
        if delta > 1e-9:
            match += "   MISMATCH"

    rows = []
    for column in (
        "sharpe",
        "weight_exposure",
        "exposure",
        "n_trades",
        "cagr",
        "turnover",
        "max_drawdown",
    ):
        rows.append({"metric": column, **spread(distribution[column])})
    frame = pd.DataFrame(rows)

    print(f"B4 null distribution over {len(distribution)} simulations: {match}")
    print()
    header = f"{'metric':16}{'p10':>12}{'p50':>12}{'p90':>12}"
    print(header)
    print("-" * 52)
    for _, row in frame.iterrows():
        print(f"{row['metric']:16}{row['p10']:12.4f}{row['p50']:12.4f}{row['p90']:12.4f}")

    print()
    print(
        f"simulations with 0 trades: {int((distribution['n_trades'] == 0).sum())} / "
        f"{len(distribution)}"
    )
    print(
        f"simulations with 1 trade : {int((distribution['n_trades'] == 1).sum())} / "
        f"{len(distribution)}"
    )

    results = cfg["paths"]["results"]
    distribution.to_csv(f"{results}/null_diagnostic_frozen.csv", index=False)
    frame.to_csv(f"{results}/null_diagnostic_frozen_summary.csv", index=False)
    references.to_csv(f"{results}/null_diagnostic_reference.csv", index=False)
    print(f"\nwrote {results}/null_diagnostic_frozen.csv and _reference.csv")

    from src.backtest.tune import count_trials

    print(f"counted trials (must still be 331): {count_trials(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
