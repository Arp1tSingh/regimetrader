"""One-off: validation-year prob_up distributions for all four models.

Uses only the distribution of predictions. No returns, Sharpe or accuracy are read
here, so nothing in this script can leak outcome information into a threshold choice.

Reported per model:
  p50, p80, p95, p99, max
  distinct prob_up values
  share of rows tied at the single most common value
  share of validation days with >= ceil(min(slots_grid)/2) stocks at or above an entry
"""

from __future__ import annotations

import math

import numpy as np

from src.common.config import load_config
from src.common.db import get_engine, read_sql


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)
    need = math.ceil(min(cfg["tuning"]["slots_grid"]) / 2)
    entries = [0.55, 0.60, 0.52]

    start = int(str(cfg["splits"]["validation_start"]).replace("-", ""))
    end = int(str(cfg["splits"]["validation_end"]).replace("-", ""))

    signals = read_sql(
        engine,
        "SELECT m.model_name AS model, f.date_key AS date_key, s.symbol AS symbol, "
        "       f.prob_up AS prob_up "
        "FROM fact_signal f JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE f.date_key BETWEEN :a AND :b ORDER BY m.model_name",
        {"a": start, "b": end},
    )

    print(f"validation year {start}..{end}; {len(signals):,} signal rows")
    print(f"qualifying stocks needed on a day: {need}")
    print()
    header = (
        f"{'model':8}{'p50':>8}{'p80':>8}{'p95':>8}{'p99':>8}{'max':>8}"
        f"{'distinct':>10}{'tied%':>8}{'p95-p50':>9}  "
        + "".join(f"{'days>=' + str(e):>10}" for e in entries)
    )
    print(header)
    print("-" * len(header))

    for model, group in signals.groupby("model"):
        values = group["prob_up"].to_numpy()
        most_common = pd_value_counts_top(values)
        distinct = int(len(np.unique(np.round(values, 12))))
        p50, p80, p95, p99 = (np.percentile(values, q) for q in (50, 80, 95, 99))

        per_day = {}
        for d, _s, p in group[["date_key", "symbol", "prob_up"]].itertuples(index=False):
            per_day.setdefault(d, []).append(p)

        day_shares = []
        for entry in entries:
            share = float(
                np.mean([sum(1 for p in ps if p >= entry) >= need for ps in per_day.values()])
            )
            day_shares.append(share)

        print(
            f"{model:8}{p50:8.4f}{p80:8.4f}{p95:8.4f}{p99:8.4f}{values.max():8.4f}"
            f"{distinct:10d}{most_common * 100:7.2f}%{p95 - p50:9.4f}  "
            + "".join(f"{s:10.4f}" for s in day_shares)
        )
    return 0


def pd_value_counts_top(values: np.ndarray) -> float:
    """Share of rows holding the single most common prob_up value."""
    _, counts = np.unique(np.round(values, 12), return_counts=True)
    return float(counts.max() / values.size)


if __name__ == "__main__":
    raise SystemExit(main())
