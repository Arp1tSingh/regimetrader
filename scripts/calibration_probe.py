"""One-off: report the calibration check before the tuning grid is run (T5.6).

`05` section 6 step 1-2: log `candidate_day_share` per model on the validation year.
If it falls below `strategy.calibration_min_candidate_day_share`, that model's
`entry_base` grid gains the rank-based fallback and the trial count changes.
"""

from __future__ import annotations

import math

from src.common.config import load_config
from src.common.db import get_engine, read_sql


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)

    min_slots = min(cfg["tuning"]["slots_grid"])
    need = math.ceil(min_slots / 2)
    min_entry = min(cfg["tuning"]["entry_base_grid"])
    limit = float(cfg["strategy"]["calibration_min_candidate_day_share"])

    models = list(cfg["tuning"]["models"])
    entries = list(cfg["tuning"]["entry_base_grid"])
    slots = list(cfg["tuning"]["slots_grid"])
    base_grid = len(models) * len(entries) * len(slots)

    print("tuning grid")
    print(f"  models         : {models}")
    print(f"  entry_base_grid: {entries}")
    print(f"  slots_grid     : {slots}")
    print(f"  base grid size : {base_grid} trials (before any fallback)")
    print()
    print(f"calibration: need >= {need} stocks with prob_up >= {min_entry} on a day;")
    print(f"            fallback triggers below {limit}")
    print()

    start = int(str(cfg["splits"]["validation_start"]).replace("-", ""))
    end = int(str(cfg["splits"]["validation_end"]).replace("-", ""))
    df = read_sql(
        engine,
        "SELECT m.model_name AS model_name, f.date_key AS date_key, f.prob_up AS prob_up "
        "FROM fact_signal f JOIN dim_model m ON m.model_key = f.model_key "
        "WHERE f.date_key BETWEEN :a AND :b ORDER BY m.model_name, f.date_key",
        {"a": start, "b": end},
    )
    print(f"validation rows: {len(df)}")
    print()

    fallback = []
    for model, group in df.groupby("model_name"):
        per_day = group.assign(ok=group["prob_up"] >= min_entry).groupby("date_key")["ok"].sum()
        share = float((per_day >= need).mean())
        triggered = share < limit
        if triggered:
            fallback.append(model)
        flag = "FALLBACK" if triggered else "ok"
        print(f"  {model:7} candidate_day_share = {share:.4f}  {flag}")

    print()
    print(f"base grid        : {base_grid} trials")
    if fallback:
        print(f"fallback for     : {', '.join(fallback)}")
        print(f"TOTAL TRIALS     : {base_grid + len(fallback)}")
        print("-> the trial count changes; the human must approve before tuning")
    else:
        print(f"TOTAL TRIALS     : {base_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())