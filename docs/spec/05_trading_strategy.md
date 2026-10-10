# 05. Trading Strategy

How predictions and regimes become positions. Every number is a configurable starting value in `config/config.yaml` (`strategy.*`), not a recommendation.

---

## 1. Principles

1. **Long-only.** The bot holds a stock or does not.
2. **Daily decisions, next-open execution.** Signals computed after the close of day *t* are executed at the open of day *t+1*.
3. **The regime sets the risk budget.** The model picks stocks; the regime decides how much capital may be invested.
4. **Fewer trades.** Costs destroy small edges (`06` section 3), so rules use hysteresis, a rebalance band and a stop-out cooldown.

---

## 2. Regime rules

Entry threshold = `strategy.entry_base + strategy.entry_offsets[regime]`, rounded to 6 decimals. Maximum equity invested = `strategy.exposure[regime]`.

| Regime | Entry threshold on `prob_up` (defaults) | Maximum equity invested | Behaviour |
|---|---|---|---|
| BULL | 0.55 | 100% | Fully active |
| SIDEWAYS | 0.60 | 60% | Higher conviction, smaller size |
| HIGH_VOL | 0.65 | 30% | Very selective, small size |
| BEAR | none | 0% | No new entries; exit everything at the next open |

---

## 3. Position rules

| Rule | Value (config key) |
|---|---|
| Position slots | 10 (`slots`) |
| Weight per slot | `exposure[regime] / slots` (BULL: 10% each) |
| Selection | Rank by `prob_up` (ties: symbol A to Z), take the best at or above the entry threshold until slots are full |
| Hold while | `prob_up >= exit_prob` (0.50) and regime is not BEAR |
| Exit when | `prob_up < exit_prob`, or regime BEAR, or `force_exit` (stop-loss) |
| Stop-loss | If `close <= avg_fill_price * (1 - pct)` (7%) at a close, the symbol is in `force_exit` for tomorrow's decision |
| Stop-out cooldown | After a stop-loss exit, the symbol is in `no_entry` for `cooldown_days` (5) trading days (derived from `fact_trade` rows with `reason = STOP_LOSS`) |
| No new entry | Symbols with a flagged or forward-filled bar today are in `no_entry` (held ones may stay) |
| Rebalance band | Trade an existing position only if `|target_weight - current_weight| > rebalance_band * target_weight` (25%). A new position (current 0) and an exit (target 0) always trade |
| Cash | Not invested means cash with zero return |

**Hysteresis:** exit threshold (0.50) below entry threshold (0.55 or more) stops repeated buying and selling when `prob_up` hovers near a level.

Stop-loss, cooldown and the band are part of the strategy variants V0 to V2 and not of the baselines B1 to B3 (`06` section 6).

---

## 4. Reference implementation

```python
import pandas as pd


def build_regime_rules(cfg: dict) -> dict:
    """{'BULL': {'entry': 0.55, 'exposure': 1.0}, ..., 'BEAR': {'entry': None, 'exposure': 0.0}}"""
    s = cfg["strategy"]
    rules = {}
    for regime, exposure in s["exposure"].items():
        if exposure == 0.0:
            rules[regime] = {"entry": None, "exposure": 0.0}
        else:
            rules[regime] = {
                "entry": round(s["entry_base"] + s["entry_offsets"][regime], 6),
                "exposure": float(exposure),
            }
    return rules


def target_weights(
    day_signals: pd.DataFrame,      # index = symbol, column 'prob_up' (after today's close)
    regime: str,
    held: set,                      # symbols currently held
    rules: dict,                    # from build_regime_rules
    slots: int = 10,
    exit_prob: float = 0.50,
    force_exit: frozenset = frozenset(),   # stop-loss: leave now, cannot be kept or re-entered
    no_entry: frozenset = frozenset(),     # may be kept if already held, cannot be newly entered
) -> pd.Series:
    """Target weight per symbol for the next open. Pure function, deterministic."""
    rule = rules[regime]
    if rule["exposure"] == 0.0:
        return pd.Series(dtype=float)

    probs = day_signals["prob_up"]
    keep = sorted(
        s for s in held
        if s not in force_exit and s in probs.index and probs[s] >= exit_prob
    )
    blocked = set(keep) | set(force_exit) | set(no_entry)
    cand = probs[(probs >= rule["entry"]) & ~probs.index.isin(list(blocked))]
    cand = cand.sort_index().sort_values(ascending=False, kind="stable")
    free_slots = max(slots - len(keep), 0)
    new = cand.head(free_slots).index.tolist()

    chosen = keep + new
    return pd.Series(rule["exposure"] / slots, index=chosen, dtype=float)
```

The function returns weights only. The execution core (`06` section 3) converts weights to share quantities, applies the rebalance band and costs. NaN `prob_up` never qualifies for entry and fails the exit test.

---

## 5. Regime transitions

| Change | What happens (follows from the code above) |
|---|---|
| BULL to SIDEWAYS or HIGH_VOL | Every held stock still above `exit_prob` is kept, but each target weight falls to `exposure / slots`; the 25% band then trims the positions. Nothing is force-sold for having the lowest probability (`DECISIONS` R6) |
| Any regime to BEAR | Empty target; all positions sold at the next open |
| BEAR to any other regime | Buying restarts, subject to entry thresholds, free slots and cooldowns |

Regime smoothing (`04` section 5.3) limits flip-flopping.

---

## 6. Calibration and tuning protocol

Classifier probabilities for daily direction cluster near 0.5, so thresholds can be too high to ever trigger. They can also be **degenerate** — a large mass of identical values — which no absolute threshold can serve.

1. **Distribution table** (per model, on validation-year probabilities, no returns read): `p50`, `p80`, `p95`, `p99`, `max`, the number of distinct `prob_up` values, the share of rows tied at the single most common value, `p95 - p50`, and the `candidate_day_share` at 0.52. Reported, not counted as trials.

2. **Eligibility rule** (`DECISIONS` R31). A model is eligible for strategy tuning only if **both** hold on that distribution: no single `prob_up` value holds more than **20%** of rows, **and** `p95 - p50 >= 0.01`. The rule reads only the distribution of predictions — never returns, Sharpe or accuracy. It was fixed after seeing the decision tree's distribution and before any strategy outcome for it.

   An ineligible model is excluded from the grid **only**. It is still trained, written to `fact_signal`, scored, and reported in the classifier comparison, `classifier_metrics.csv`, the trial log's `model_year` rows and the validation diagnostics (R32).

3. **Per-model third entry base** (`DECISIONS` R33). Every eligible model keeps `0.55` and `0.60`. The third entry base is `strategy.fallback_entry_base` (0.52) when the model reaches it on at least 50% of validation days (`method: absolute`); otherwise it is that model's own validation **p80** of `prob_up` (`method: percentile_p80`). Measured: xgb 60.0% → absolute; logreg 15.5% and rf 13.9% → p80.

4. **Grid** (validation period only): for each **eligible** model, each of its 3 entry bases, each `slots` in `tuning.slots_grid`: run the full strategy V2 on the validation period (not persisted), log one trial (`metric = sharpe`; CAGR, max drawdown and turnover in `extra`) and write all trials to `reports/results/tune_grid_validation.csv`. Grid size = **eligible models x 3 entry bases x 2 slots** = 3 x 3 x 2 = **18**, plus the 9 prior development trials = **27 total**.

5. **Tie-break reporting** (`DECISIONS` R34). Every tuning and baseline run reports `tie_break_share` — the share of new entries sitting on the cutoff probability, which were therefore chosen by symbol order rather than by a higher probability. A run above **5%** is flagged `tie_break_flagged` in `tune_grid_validation.csv` and excluded from selection with a note in the trial log. Flagged and non-trading runs both stay in the CSV and the log.

6. **Selection (`DECISIONS` R39-R42).** Selection by Sharpe **alone is not used anywhere.** A configuration must clear two filters in order; survival is a precondition, not a filter applied afterwards.

   - **Activity filter (R39).** Eligible only with at least **30 completed round trips** and **mean exposure at least 10%** on the validation period. 30 round trips is the smallest sample whose trade-level hit rate has a usable standard error (about 9 points near 50%).
   - **Null filter (R40).** Take the **top 3 by Sharpe** among activity-eligible configurations and run the B4 symbol-permutation null for **each**, 100 simulations, seed from config. A configuration is selectable only if its Sharpe **exceeds the 90th percentile of its own null**. Each null is run under that configuration's own entry base and slots, not a shared one.
   - **Survivors** are ordered by Sharpe with the turnover tie-break, as before.

7. **No demonstrated edge (R41).** If **no** configuration passes both filters, the frozen config is recorded as **no demonstrated edge**: `selected_model` is the **highest-validation-ROC-AUC model among the distribution-eligible models**, with the spec's starting strategy settings (`entry_base` 0.55 with its fallback rule, `slots` 10), and `meta.method = no_edge_default`.

8. Report the number of configurations tried (trial log), how many passed each filter, and which path was taken. Every B4 null simulation is logged as `kind = "other"`.

9. **Trial accounting across grids (R43).** If a grid is re-run under a revised selection rule, the earlier run's output is preserved under a separate filename, its trial rows remain counted, and a `kind = "other"` row records the revision. A revised protocol that silently replaces the first attempt is indistinguishable from tuning until something looked good.

Discipline: tune only on the validation year; keep the grid small; never touch the test period; after the final test do not change the strategy and re-run (`06` section 8).

---

## 7. Strategy variants (ablations)

Each variant is a weight provider for the shared engine (`06` section 2).

| Variant | Description |
|---|---|
| V0 | Classifier only: BULL rule for every regime (same threshold, exposure 100%), always invested up to the slot limit. Stop-loss and band on |
| V1 | Regime filter only: equal weight across all stocks with a bar today when the regime is BULL, cash otherwise; no classifier. Band on, stop-loss off |
| V2 | **Full strategy**: classifier plus regime rules (the main system) |
| V3 (stretch) | V2 with the relative-return target (`target.kind: relative`; models named `<name>_rel`) |
| V4 (stretch) | V2 with one classifier per regime, if each regime has enough rows |
