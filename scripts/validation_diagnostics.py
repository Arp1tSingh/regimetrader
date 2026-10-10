"""One-off: the R35 validation diagnostics for all four models.

Validation year only. Nothing here is read by `tune`, so it cannot influence
eligibility or selection.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import load_config
from src.common.db import get_engine, read_sql
from src.models.diagnostics import model_diagnostics


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)
    seed = int(cfg["project"]["seed"])

    start = int(str(cfg["splits"]["validation_start"]).replace("-", ""))
    end = int(str(cfg["splits"]["validation_end"]).replace("-", ""))

    rows = read_sql(
        engine,
        "SELECT m.model_name AS model, f.date_key AS date_key, s.symbol AS symbol, "
        "       f.prob_up AS prob_up, f.next_day_return AS fwd_return "
        "FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE f.date_key BETWEEN :a AND :b "
        "ORDER BY m.model_name, s.symbol",
        {"a": start, "b": end},
    )
    rows["target_up"] = (rows["fwd_return"] > 0).astype(int)
    print(f"validation {start}..{end}: {len(rows):,} rows\n")

    out = []
    for model, group in rows.groupby("model", sort=True):
        stats = group["prob_up"]
        result = model_diagnostics(group, seed=seed)
        result["model"] = model
        # Degeneracy alongside the diagnostics, so the tree's numbers are never
        # read without its distribution.
        result["distinct_prob"] = int(stats.round(12).nunique())
        result["tied_share"] = float(stats.round(12).value_counts(normalize=True).iloc[0])
        out.append(result)

    frame = pd.DataFrame(out).set_index("model")

    print("ROC-AUC (higher is better; 0.5 is coin-flip):")
    for m, v in frame["roc_auc"].items():
        print(f"  {m:8}{v:.4f}")

    print("\nAccuracy edge over majority baseline, day-clustered 95% CI:")
    for m in frame.index:
        print(
            f"  {m:8}{frame.loc[m, 'edge_over_baseline']:+.4f} "
            f"[{frame.loc[m, 'edge_ci_low']:+.4f}, {frame.loc[m, 'edge_ci_high']:+.4f}]"
            f"   (baseline {frame.loc[m, 'majority_baseline']:.4f})"
        )

    print("\nPer-day top-quintile minus bottom-quintile next-day return, day-clustered 95% CI:")
    for m in frame.index:
        print(
            f"  {m:8}{frame.loc[m, 'quintile_spread']:+.6f} "
            f"[{frame.loc[m, 'quintile_ci_low']:+.6f}, {frame.loc[m, 'quintile_ci_high']:+.6f}]"
        )

    print("\nDegeneracy of prob_up alongside:")
    print(f"  {'model':8}{'distinct':>10}{'tied share':>14}")
    for m in frame.index:
        print(f"  {m:8}{frame.loc[m, 'distinct_prob']:10d}{frame.loc[m, 'tied_share']:13.2%}")

    out_path = cfg["paths"]["results"]
    frame.to_csv(f"{out_path}/validation_diagnostics.csv")
    print(f"\nwrote {out_path}/validation_diagnostics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
