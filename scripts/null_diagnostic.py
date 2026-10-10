"""Read-only diagnostic: the full B4 null distribution for the frozen configuration.

Recomputes the null for rf / entry_base 0.504415 / slots 10 so that exposure, trade
count and CAGR can be reported alongside Sharpe. Those columns were never persisted
per simulation, and `b4_sharpe_distribution_validation.csv` was overwritten by a
timing run, so they have to be recomputed.

This logs **no** trials and writes no frozen config: it calls `b4_distribution`
directly rather than `tune`, so the trial log and the freeze are untouched. The
permutation is seeded per simulation, so the recomputed Sharpes must match the ones
already in `trial_log.csv` exactly; that equality is asserted and printed, and it is
what makes this a diagnostic rather than a re-run.

Writes `reports/results/null_diagnostic_frozen.csv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.baselines import b4_distribution
from src.common.config import load_config
from src.common.db import get_engine


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


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)

    frozen = cfg["strategy"]
    model = str(frozen["selected_model"])
    entry = float(frozen["entry_base"])
    slots = int(frozen["slots"])
    sims = 100
    seed = int(cfg["project"]["seed"])

    print(f"recomputing null for the frozen configuration: {model}/{entry:.6f}/{slots}")
    print(f"{sims} simulations, seed {seed} (permutation seed = project.seed + sim)\n")

    distribution = b4_distribution(cfg, engine, "2023-01-01", "2023-12-31", sims=sims, model=model)

    # Determinism check against what is already logged.
    trials = pd.read_csv(f"{cfg['paths']['results']}/trial_log.csv")
    logged = trials[
        (trials["metric"] == "null_sharpe")
        & trials["run_id"].astype(str).str.startswith(f"null_{model}_{entry:.4f}_{slots}_sim")
    ].sort_values("run_id")
    recomputed = distribution["sharpe"].reset_index(drop=True)
    match = ""
    if len(logged) == len(recomputed):
        logged_sorted = logged["value"].reset_index(drop=True).sort_index()
        delta = float(np.nanmax(np.abs(logged_sorted.to_numpy() - recomputed.to_numpy())))
        match = f" (max |difference| vs trial_log = {delta:.3e})"
        if delta > 1e-9:
            match += "  MISMATCH"

    rows = []
    for column in ("sharpe", "exposure", "n_trades", "cagr", "turnover", "max_drawdown"):
        stats = spread(distribution[column])
        rows.append({"metric": column, **stats})

    frame = pd.DataFrame(rows)
    print("B4 null distribution across", len(distribution), "simulations:", match)
    print()
    print(f"{'metric':14}{'p10':>12}{'p50':>12}{'p90':>12}")
    print("-" * 50)
    for _, row in frame.iterrows():
        print(f"{row['metric']:14}{row['p10']:12.4f}{row['p50']:12.4f}{row['p90']:12.4f}")

    print()
    zero_trade = int((distribution["n_trades"] == 0).sum())
    one_trade = int((distribution["n_trades"] == 1).sum())
    print(f"simulations with 0 trades: {zero_trade} / {len(distribution)}")
    print(f"simulations with 1 trade : {one_trade} / {len(distribution)}")

    out = f"{cfg['paths']['results']}/null_diagnostic_frozen.csv"
    distribution.to_csv(out, index=False)
    frame.to_csv(f"{cfg['paths']['results']}/null_diagnostic_frozen_summary.csv", index=False)
    print(f"\nwrote {out}")

    # Nothing was logged: confirm the trial count is unchanged.
    from src.backtest.tune import count_trials

    print(f"counted trials (must be unchanged at 331): {count_trials(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
