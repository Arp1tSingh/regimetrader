"""One-off: the R31-R33 tuning plan, and the step-5 stop condition.

Reads only probability distributions. No returns, no Sharpe, no backtest.
"""

from __future__ import annotations

from src.backtest.tune import plan_models
from src.common.config import load_config
from src.common.db import get_engine


def main() -> int:
    cfg = load_config("config/config.yaml")
    engine = get_engine(cfg)
    plan = plan_models(engine, cfg)

    header = f"{'model':8}{'eligible':10}{'method':18}{'entry bases':30}reason"
    print(header)
    print("-" * 118)
    for p in plan:
        entries = ",".join(f"{e:.4f}" for e in p["entries"]) or "-"
        print(f"{p['model']:8}{str(p['eligible']):10}{p['method']:18}{entries:30}{p['reason']}")

    eligible = [p for p in plan if p["eligible"]]
    excluded = [p for p in plan if not p["eligible"]]
    slots = len(cfg["tuning"]["slots_grid"])
    trials = len(eligible) * 3 * slots

    print()
    print(f"eligible: {len(eligible)}   excluded: {len(excluded)}")
    print(f"grid trials = {len(eligible)} x 3 x {slots} = {trials}")

    # Step 5: stop if more than one model is ineligible, or fewer than two are eligible.
    triggered = len(excluded) > 1 or len(eligible) < 2
    print()
    verdict = "TRIGGERED, do not run" if triggered else "clear, may run"
    print(f"stop condition (>1 ineligible or <2 eligible): {verdict}")
    return 1 if triggered else 0


if __name__ == "__main__":
    raise SystemExit(main())
