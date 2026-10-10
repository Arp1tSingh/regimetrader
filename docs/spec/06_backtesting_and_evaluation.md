# 06. Backtesting and Evaluation

How the strategy is simulated and judged. The rules here exist to keep results honest.

---

## 1. Principles

1. **Only past information at decision time.** A decision after the close of day *t* uses data through *t*.
2. **Realistic execution.** Orders fill at the next day's open, with slippage and costs.
3. **Out-of-sample only.** Headline results come from the test period (`splits.test_start` onwards), produced by models that never saw it.
4. **Compare against baselines.** A return number without a benchmark means little.
5. **Report everything tried.** Selection bias is the main way trading projects mislead their authors.

---

## 2. Engine

### 2.1 Interfaces (`src/backtest/engine.py`, `src/trading/`)

```python
@dataclass
class DayContext:
    date: pd.Timestamp
    regime: str                      # smoothed label for this date
    signals: pd.DataFrame            # index = symbol, column prob_up (selected model), equities with a bar today
    indicators: pd.DataFrame         # index = symbol, fact_indicator columns for today (for rule baselines)
    held: set                        # symbols currently held
    current_weights: pd.Series       # current weight per held symbol at today's close
    force_exit: frozenset            # stop-loss triggers found at today's close
    no_entry: frozenset              # cooldowns, flagged or filled bars
    is_month_end: bool

WeightProvider = Callable[[DayContext], Optional[pd.Series]]
# Returns target weights (symbol -> weight). Returning None means "place no orders at the next open": it clears pending targets and does NOT repeat the previous ones.

def run_backtest(cfg, weights_fn, start, end, run_id, *, persist=True, write_signals=False,
                 model_name=None, costs_enabled=True, stop_loss_enabled=True,
                 band=None, fractional_shares=False, allow_test_period=False) -> BacktestResult
# BacktestResult: equity (DataFrame: cash, positions_value, equity, daily_return, drawdown),
#                 trades (DataFrame like fact_trade), signals (DataFrame of decisions)
```

- `run_backtest` MUST raise `ValueError` if `end >= splits.test_start` and `allow_test_period` is False. Only `final_test.py` sets it (section 8).
- `persist=True` writes `fact_trade` and `fact_portfolio_daily` (delete existing rows for that `run_id` first, so reruns are idempotent). `write_signals=True` (primary run only) updates `fact_signal.signal` and `target_weight` for `model_name`.
- `band=None` uses `strategy.rebalance_band`.

Strategies V0, V1, V2 and baselines B1 to B3 are `WeightProvider`s (`src/backtest/baselines.py`, `src/trading/rules.py`). Backtest and paper trading share `src/trading/`.

### 2.2 Algorithm

State: `cash = initial_capital`, empty positions, empty pending targets. The period starts flat (`DECISIONS` R18).

```text
for each trading day t in [start, end]:
    1. EXECUTE : if pending targets exist (decided at the close of t-1), plan orders against current
                 positions using equity_at_decision (equity at close t-1); sells first, then buys in
                 descending prob_up; fill at open_t adjusted for slippage; deduct costs; update cash.
                 Symbols with no bar on t are skipped (the next decision recomputes).
    2. MARK    : value positions at close_t (a held symbol with no bar on t is valued at its last known close);
                 record cash, positions_value, equity, daily_return, drawdown.
                 Stop-loss: any held symbol with close_t <= avg_fill_price * (1 - pct) joins force_exit
                 and starts its cooldown (when stop_loss_enabled).
    3. DECIDE  : build DayContext from data up to close_t; targets = weights_fn(ctx).
    4. STORE   : pending = targets (None means no orders at the next open); if write_signals,
                 record signal = 1 and target_weight for chosen symbols, 0 otherwise.
    5. LOG     : buffer trades and portfolio rows; write them to the warehouse at the end (persist).
```

Steps 1 and 3 are separated by the overnight gap: the order for tomorrow is decided today but priced tomorrow. That ordering prevents look-ahead. The last day's decision has no next open and is ignored; open positions are marked at the last close, not liquidated.

**Position sizing:** `target_qty = floor(target_weight * equity_at_decision / open_t)` (whole shares; `fractional_shares=True` keeps the float, used for the index baseline B1). The leftover stays in cash.

**Inputs (all from the warehouse):** prices, regime labels (`fact_market_regime`), `prob_up` for the selected model and the model year that predicts each date (`fact_signal` joined to `dim_model`), indicators, `dim_date`.

---

## 3. Execution and cost model (`src/trading/execution.py`)

| Component | Default (config `backtest.*`) | Note |
|---|---|---|
| Slippage | 0.05% per side, adverse (buys fill higher, sells lower) | Gap between the open price and what you can get |
| All-in cost | 0.10% of trade value per side | Brokerage, exchange charges, taxes, stamp duty combined. Rates change; adjust in config |
| Initial capital | 1,000,000 | Virtual rupees |
| Risk-free rate | 0.0 | Used only in Sharpe and Sortino; state it in results |

Formulas:

```text
fill_price  = open * (1 + slippage)   for BUY;   open * (1 - slippage)   for SELL
trade_value = quantity * fill_price
cost        = cost_pct * trade_value
BUY  cash out = trade_value + cost          SELL cash in = trade_value - cost
```

**Worked example (test fixture):** buy 100 shares at open 1,000: fill 1,000.50; trade value 100,050.00; cost 100.05; cash out **100,150.05**. A round trip costs about 0.30% of trade value.

**Order planning rules**

1. Current weight = position value at the decision close divided by `equity_at_decision`.
2. Apply the rebalance band to existing positions; new positions and exits always trade.
3. Sells first (they free cash), then buys in descending `prob_up` (ties by symbol). A buy is capped by available cash: reduce quantity (floor) until `trade_value + cost <= cash`; skip if zero. Cash never goes negative.
4. `realized_pnl` on a SELL = `(fill_price - avg_cost_per_share) * quantity - sell_cost`, where `avg_cost_per_share` includes buy costs (buy cash out divided by quantity bought). BUY rows have `realized_pnl = NULL`.
5. **Average-cost method (not FIFO):** `Position.avg_fill` is the quantity-weighted mean of the buy fill prices of the current position (costs excluded). It is reset when the position is fully sold; partial sells do not change it. The stop-loss test and the paper job's state recovery from `fact_trade` use it. Every trade row carries `reason`: `ENTRY` (new position), `REBALANCE` (band trade on an existing position), `EXIT` (full sale for a signal or regime reason), `STOP_LOSS`.

**Why costs matter (fixture for intuition):** a 55% hit rate with symmetric 1% wins and losses earns +0.10% per trade gross, minus 0.30% round-trip cost, giving -0.20% net. Always report results after costs.

---

## 4. Metrics (`src/backtest/metrics.py`)

Daily returns = `equity.pct_change().dropna()`. `years = (last date - first date).days / 365.25`.

| Metric | Definition |
|---|---|
| CAGR | `(final / initial) ** (1 / years) - 1` |
| Annualised volatility | `std(daily, ddof=1) * sqrt(252)` |
| Sharpe | `sqrt(252) * mean(excess) / std(excess, ddof=1)`, `excess = daily - rf / 252` |
| Sortino | `sqrt(252) * mean(excess) / sqrt(mean(min(excess, 0) ** 2))` |
| Max drawdown | Largest peak-to-trough fall of equity (a negative number) |
| Calmar | `CAGR / abs(max drawdown)` (NaN if drawdown is 0) |
| Win rate | Share of SELL trades with `realized_pnl > 0` |
| Profit factor | Sum of positive `realized_pnl` / abs(sum of negative); NaN if there are no losses |
| Turnover | `(total buy value + total sell value) / 2 / mean(equity) / years` |
| Exposure | Mean of `positions_value / equity` over days |

```python
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = daily_returns - risk_free_annual / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(daily_returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = daily_returns - risk_free_annual / TRADING_DAYS
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean())
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside) if downside > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    return float((equity / running_peak - 1).min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
```

`metrics.summarize(equity_df, trades_df, cfg) -> dict` returns every metric above and is the only place summary rows are built. The reference code hard-codes 252 for readability; the implementation reads `backtest.trading_days`.

---

## 5. Run IDs and period convention

| Purpose | `run_id` | Period | Persisted |
|---|---|---|---|
| Development and tuning trials | `tune_<model>_<base>_<slots>` | validation | no (metrics go to the trial log) |
| Validation results | `val_V0`, `val_V1`, `val_V2`, `val_B1`, `val_B2`, `val_B3` | validation | yes; created by `baselines.py --all --period validation` with the frozen selected model |
| Final test (written once, by `final_test.py`) | `backtest_main` (V2, selected model; the only run with `write_signals=True`), `test_V0`, `test_V1`, `test_B1`, `test_B2`, `test_B3`, `test_V2_nocost` | test | yes |
| Random null (B4) | none (200 simulations) | validation and test | no (distribution saved to CSV) |
| Paper trading | `paper_live`; replays use `paper_replay` | live | yes |

Every run writes a manifest (`07` section 4).

---

## 6. Baselines and ablations

All baselines use the same engine, costs, slippage and period convention (flat start, first decision at the close of the first day).

| ID | Definition | Engine settings |
|---|---|---|
| B1 | Buy and hold the Nifty 50 index: weight 1.0 on `^NSEI` on the first decision, `None` afterwards | `fractional_shares=True`, `band=0`, stop-loss off |
| B2 | Equal-weight stocks, rebalanced monthly: on the first decision day and on every `is_month_end` day return `1/N` for each stock with a bar today (N = their count), otherwise `None` | `band=0`, stop-loss off |
| B3 | Trend rule, no ML: hold each stock with `sma_20 > sma_50`; weight `1/N_universe` each (rest in cash), recomputed daily | default band, stop-loss off |
| B4 | **Symbol-permutation null**: for each of `backtest.random_baseline_sims` simulations, relabel each stock's `prob_up` series to a different stock (one fixed random permutation per simulation, seeded), then run V2 unchanged. Keeps exposure, regime behaviour and turnover roughly equal, breaks the link between a stock's predictions and its own returns. Save the Sharpe and CAGR distribution; show where the real V2 falls | persist off |

| Ablation | Compares | Run |
|---|---|---|
| A1 | V0 (classifier without regime rules) against V2 | `val_V0`, `test_V0` |
| A2 | V1 (regime filter without classifier) against V2 | `val_V1`, `test_V1` |
| A3 | LR vs Tree vs RF vs XGB | validation Sharpe from the tuning trials and `classifier_metrics.csv` |
| A4 | V2 with versus without costs and slippage | `backtest_main` vs `test_V2_nocost` (`costs_enabled=False`) |

---

## 7. Statistical honesty (`src/backtest/evaluate.py`)

1. **Trial log.** Every model and parameter combination run is logged. The final report states the total count (rows of `trial_log.csv` with `kind != 'calibration'`, `07` section 4).
2. **Bootstrap interval for Sharpe.** Moving-block bootstrap of the daily strategy returns (`backtest.bootstrap.block_days` = 10, `resamples` = 1000): pick `ceil(n / block)` start indices uniformly in `[0, n - block]`, concatenate the blocks, trim to length n, compute Sharpe; report the 5th and 95th percentiles (`ci_low`, `ci_high`) with the point estimate, seed from config. A Sharpe of 0.6 with an interval of -0.3 to 1.5 is weak evidence and `RESULTS.md` must say so.
3. **Subperiod analysis.** Metrics for each calendar year of the test period (`yearly_test.csv`).
4. **Regime breakdown.** Hit rate and return by regime (OLAP query 09) and strategy return by regime (`regime_breakdown.csv`).
5. **Negative results are acceptable.** If the strategy does not beat B1 on return, check drawdown and Sharpe. Report what is found; `RESULTS.md` is generated from the numbers, with no wording that claims success.

---

## 8. Test-period guard (`src/backtest/final_test.py`)

- The test period is evaluated once. This module is the only production caller of `run_backtest(..., allow_test_period=True)` (tests on the smoke warehouse may also call it).
- Until this module runs, nothing in development scores test-year rows: not the strategy, not the classifier metrics (`04` section 7), not the figures.
- Preconditions it checks and refuses on: `paths.frozen_config` exists and contains `strategy.selected_model`; `--confirm` was given. (The fast test suite is a prerequisite of the `make final-test` target, not something this module runs; smoke runs and tests call the module directly.)
- On first run it: writes `reports/results/TEST_RUN_LOCK.json` (timestamp, git commit or `unknown`, config hash, path of the frozen config, run count 1, `data_end_date` = last loaded trading date); copies the frozen config to `reports/results/frozen_config_used.yaml`; runs all test-period runs in section 5; writes `summary_test.csv`, `yearly_test.csv`, `bootstrap_sharpe.csv`, `regime_breakdown.csv`, `b4_sharpe_distribution_test.csv`; appends the test-year rows (period `test`) to `classifier_metrics.csv`.
- A second invocation exits with code 3 and an explanatory message. `--allow-rerun "<reason>"` overrides it, increments the count in the lock file, and appends `{timestamp, reason, commit}` to `reports/results/test_run_log.jsonl`. `RESULTS.md` prints the run count.
- Smoke runs use their own `paths.results`, so they never touch the real lock.

---

## 9. Leakage checklist (`reports/results/leakage_checklist.md`)

Generated by `report.py` at the end of Phase 6. Each item has a status: `VERIFIED` (an artifact check passed, for example the lock count), `TEST` (proved by the named pytest tests; the agent runs `make test` and records the result in `PROGRESS.md`) or `STATEMENT` (text that must appear in `RESULTS.md`; `report.py` checks that it does). Do not mark an item without evidence.

- [ ] Features at day *t* use only days up to and including *t* (`test_features_no_lookahead`, `test_panel_no_lookahead`).
- [ ] The target uses day *t+1* only as the label (`test_forbidden_patterns`).
- [ ] Winsorizer, scaler and regime model were fit on training data only (`test_winsorizer_train_only`, `test_regime_fit_window`).
- [ ] Train, validation and test blocks are consecutive and non-overlapping, with an embargo before each test year (`test_walk_forward_embargo`).
- [ ] Orders decided at the close of *t* fill at the open of *t+1* (`test_engine_fills_next_open`).
- [ ] Hyperparameters and strategy settings were chosen on train and validation data only (`trial_log.csv` periods; `frozen_config_used.yaml`).
- [ ] The test period was evaluated once with final settings (`TEST_RUN_LOCK.json` count = 1, or each rerun explained).
- [ ] Costs and slippage are included in all reported results (A4 shows the no-cost version separately).
- [ ] Survivorship bias (frozen Nifty 50 list) is stated in `RESULTS.md`.

---

## 10. Outputs

**Tables** in `reports/results/`: `summary_validation.csv`, `summary_test.csv` (columns: `strategy, cagr, volatility, sharpe, sortino, max_drawdown, calmar, win_rate, profit_factor, turnover, exposure, final_equity, rf`), `yearly_test.csv`, `bootstrap_sharpe.csv`, `regime_breakdown.csv`, `classifier_metrics.csv`, `regime_profile.csv`, `regime_counts.csv`, `regime_k_selection.csv`, `b4_sharpe_distribution_validation.csv`, `b4_sharpe_distribution_test.csv`, `tune_grid_validation.csv`, `trial_log.csv`, `TEST_RUN_LOCK.json`, `test_run_log.jsonl`, `manifests/*.json`, `leakage_checklist.md`, `RESULTS.md`.

`summary_validation.csv` is written by `baselines.py` (rows B1 to B3 and V0 to V2; the V2 row also carries `b4_sharpe_p05`, `b4_sharpe_p50`, `b4_sharpe_p95` from the null distribution).

Results template (in `RESULTS.md`, filled from `summary_test.csv`):

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Calmar | Turnover | Exposure |
|---|---|---|---|---|---|---|---|
| B1 Nifty 50 buy and hold | | | | | | | |
| B2 Equal-weight buy and hold | | | | | | | |
| B3 SMA trend rule | | | | | | | |
| A1 Classifier, no regime (V0) | | | | | | | |
| A2 Regime only (V1) | | | | | | | |
| **Full strategy (V2)** | | | | | | | |

**Figures** in `reports/figures/` (matplotlib, saved PNG, titled, axes labelled):

| File | Content |
|---|---|
| `equity_curves.png` | Strategy and baselines, log scale, regime bands shaded behind |
| `drawdown_vs_b1.png` | Drawdown of V2 versus B1 |
| `regime_k_selection.png` | Elbow and silhouette for k |
| `regime_bands.png` | Index close with regime bands |
| `confusion_matrix_<model>.png`, `roc_<model>.png` | Per model |
| `feature_importance_<model>.png` | Permutation importance |
| `prob_up_histogram.png` | With entry thresholds marked |
| `b4_sharpe_distribution.png` | Random-null Sharpe distribution with the real V2 marked |
| `olap_sector_quarter_heatmap.png` | Sector by quarter from OLAP query 01 |
