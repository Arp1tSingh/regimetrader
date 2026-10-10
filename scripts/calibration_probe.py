"""One-off: report the calibration check before the tuning grid is run (T5.6).

`05` section 6 step 1-2: log `candidate_day_share` per model on the validation year.
If it falls below `strategy.calibration_min_candidate_day_share`, that model's
`entry_base` grid gains the rank-based fallback and the trial count changes.
"""

from __future__ import annotations

import math
from typing import Any

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

    models_list = sorted(df["model_name"].unique())

    def _share(engine: Any, cfg: Any, model: str, entry: float, need: int) -> tuple[float, int]:
        subset = df[df["model_name"] == model]
        if subset.empty:
            return float("nan"), 0
        per_day = subset.assign(ok=subset["prob_up"] >= entry).groupby("date_key")["ok"].sum()
        return float((per_day >= need).mean()), int(len(per_day))

    print()
    print(
        "candidate_day_share at the fallback threshold "
        f"{cfg['strategy']['fallback_entry_base']} (same definition, fallback entry):"
    )
    fallback_entry = float(cfg["strategy"]["fallback_entry_base"])
    still_zero = []
    for model in models_list:
        share, _days = _share(engine, cfg, model, fallback_entry, need)
        near_zero = share < 0.10
        if near_zero:
            still_zero.append(model)
        flag = "NEAR ZERO - STOP" if near_zero else "usable"
        print(f"  {model:7} share@0.52 = {share:.4f}  {flag}")
    print()
    if still_zero:
        print(f"STOP: still near zero at 0.52 for {', '.join(still_zero)}.")
        print("The fallback is not doing its job. A percentile threshold on validation")
        print("prob_up (e.g. the 80th) is the documented alternative; it changes the")
        print("grid, so record it in DECISIONS.md BEFORE running and recount.")
        return 1

    print("All models reach 0.52 on a meaningful share of days: the fallback is working.")
    print()
    print(f"base grid          : {base_grid} trials")
    if fallback:
        # The fallback adds an entry base to each affected model, and every entry
        # base crosses every slot count: models x (entries+1) x slots.
        per_model = (len(entries) + 1) * len(slots)
        total = len(models) * per_model
        print(f"fallback for       : {', '.join(fallback)}")
        per_model_entries = len(entries) + 1
        print(
            f"per model w/ fb    : {per_model_entries} entry bases x "
            f"{len(slots)} slots = {per_model}"
        )
        print(f"TOTAL TRIALS       : {total}")
        print("-> the trial count changes; the human must approve before tuning")
    else:
        print(f"TOTAL TRIALS       : {base_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
