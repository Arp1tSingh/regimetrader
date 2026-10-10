# 08. Runbook: Research Pipeline and Daily Operation

What happens end to end, in research and in daily operation, how to schedule it, and how it fails.

---

## 1. Two modes

| Mode | When | Goal |
|---|---|---|
| A. Research (offline) | Once, and again when models or rules change | Build the warehouse, train, backtest, produce results |
| B. Daily operation | Every trading day after the close | Update data, classify the regime, produce signals, run the paper portfolio |

Both use the same feature, regime, prediction and strategy code. What you backtest is what you paper-trade.

---

## 2. Mode A: research pipeline

| Step | Command | Output |
|---|---|---|
| 1 | `python -m src.etl.pipeline --full-refresh` | `fact_price`, `dim_*`, `etl_run_log` |
| 2 | `python -m src.features.builder --indicators` | `fact_indicator` |
| 3 | `python -m src.regime.regime_model --fit` (after reviewing `--k-select`) | `fact_market_regime`, regime model file |
| 4 | `python -m src.features.builder --panel` | `data/processed/panel.parquet` |
| 5 | `python -m src.models.train` | `fact_signal` (`prob_up`), `dim_model`, model files, classifier metrics |
| 6 | `python -m src.backtest.tune` then `python -m src.backtest.baselines --all --period validation` | frozen config, `val_*` runs, `summary_validation.csv` |
| 7 | `python -m src.backtest.final_test --confirm` (once) then `python -m src.backtest.report` | `backtest_main`, `test_*` runs, results tables, figures, `RESULTS.md` |
| 8 | `python -m src.olap.run --all` | OLAP result files |

Typical laptop run time: ETL a few minutes; training a few minutes per model; one backtest under a minute.

---

## 3. Mode B: the daily job (`python -m src.paper.daily_job`)

### 3.1 Timeline (India Standard Time)

| Time | Event |
|---|---|
| 3:30 PM | NSE regular session closes |
| After 5:00 PM | Job starts (end-of-day data is published) |
| About 5:10 PM | Warehouse updated, regime classified, signals generated, tomorrow's targets stored |
| Next trading day, 9:15 AM | Virtual orders are considered filled at that day's open |

The open price is known only after the next session starts, so the job settles the **previous** decision at the **new** day's open when it processes the new day. Dates and the evening cut-off use `project.timezone` (Asia/Kolkata).

### 3.2 Algorithm

```text
daily_job(args):
  1. ETL (unless --no-etl): incremental load; retry per section 5; hard DQ failure or exhausted retries
     -> run FAILED, alert, exit 2 (or 4). Never trade on stale data.
  2. dates = trading dates (index has a bar) after the last processed date of run_id, within [--from, --to].
     A run_id with no processed date yet STARTS at --from if given, otherwise at the latest trading date in
     the warehouse (flat, capital = backtest.initial_capital, nothing to settle). It never back-fills history.
     Guards: a new paper.run_id on real data requires the frozen config and TEST_RUN_LOCK.json to exist
     (the strategy must be frozen and tested first). Only run_id == paper.run_id writes fact_signal, and only
     for dates after the lock's data_end_date.
     if none: log "nothing to do (holiday or already processed)" and exit 0.
  3. for d in dates, ascending (steps a to f of one date commit in ONE transaction or not at all):                                 # catch-up processes missed days in order
       a. SETTLE : load pending_target rows of the previous processed date p (empty set = go to cash);
                   equity_at_decision = equity of p; execute at open_d with slippage and costs
                   (same plan_orders / fill_order as the backtest); write fact_trade.
       b. MARK   : value positions at close_d; write fact_portfolio_daily (run_id, d);
                   stop-loss triggers -> force_exit; update cooldowns.
       c. FEATURES AND REGIME : recompute indicators from full history; classify d with the frozen regime
                   model and trailing smoothing from stored raw cluster ids; write fact_indicator,
                   fact_market_regime.
       d. MODEL  : ensure dim_model has (selected_model, 'wf<year(d)>'); if missing, train it
                   (src.models.train --year) on data before year start minus embargo.
       e. PREDICT: prob_up for every equity with complete features at d; upsert fact_signal.
       f. DECIDE : target_weights(...) with held, force_exit, no_entry (flagged or filled bars, cooldowns);
                   update fact_signal.signal / target_weight only when run_id == paper.run_id (paper_live;
                   replays never touch fact_signal); write pending_target for d.
       g. REPORT (latest date only): reports/daily/<d>.md
  4. Write alerts (section 6) and a manifest for the run.
```

State is derived from the warehouse: holdings and average fill (average-cost method, `06` section 3) by replaying `fact_trade` in order; cooldowns from `fact_trade.reason = 'STOP_LOSS'` within the last `cooldown_days` trading days; stop-loss triggers by comparing the last close with `avg_fill`; cash and equity from `fact_portfolio_daily`; pending decisions from `pending_target`. A date is "processed" when `fact_portfolio_daily` has a row for (`run_id`, date). Because each date commits atomically, a processed date always has its `pending_target` rows, so an empty set there means a real decision to go to cash. The first processed date of a `run_id` starts flat with `backtest.initial_capital`.

### 3.3 Worked example (also a test fixture in `TESTS.md` section 6)

| Item | Value |
|---|---|
| Regime | BULL |
| Rules for BULL | Entry 0.55, maximum exposure 100%, 10 slots, 10% per slot |
| Held | 8 stocks |
| Exit check | One held stock now has `prob_up` 0.47 (below 0.50): dropped; 7 kept |
| Free slots | 10 - 7 = 3 |
| New entries | The 3 highest-probability stocks not held, each at or above 0.55 |
| Orders for tomorrow | SELL 1, BUY 3; each target 10% of equity (subject to the rebalance band) |

If the regime became BEAR the target would be empty and all 7 positions would be sold at the next open.

---

## 4. Daily report and scheduling

`reports/daily/<date>.md` (one page): date, latest data date, regime and regime change flag, portfolio equity and drawdown, held positions with weight, today's fills, tomorrow's orders (BUY, SELL, HOLD with target weight and `prob_up`), warnings. Include the disclaimer line.

`deploy/cron.example` (server clock set to IST; do not install):

```bash
0 17 * * 1-5  cd /path/to/regimetrader && .venv/bin/python -m src.paper.daily_job >> logs/daily.log 2>&1
```

`deploy/github_actions_daily.yml` (cron is UTC; 17:00 IST = 11:30 UTC):

```yaml
on:
  schedule:
    - cron: "30 11 * * 1-5"
```

Runners start with a clean disk. A scheduled workflow must restore and save the SQLite file (artifact or commit) or use PostgreSQL. Put this warning in the file.

---

## 5. Failure handling (one test per row)

| Problem | Behaviour |
|---|---|
| API temporarily unavailable | Retry per config; if it still fails mark the ETL run FAILED, alert, do not trade, exit 2 |
| Market holiday, or end-of-day data not yet published | The index has no bar for the weekday: log INFO and exit 0 (the job cannot tell the two apart); the next run catches up. An alert is raised if the index has no new bar for `data.stale_after_days` consecutive calendar days |
| A few stocks missing today | Forward-fill up to 2 days and flag; do not open new positions in those stocks (`no_entry`); held ones may stay |
| ETL run twice for the same day | Upserts make it safe; no duplicates |
| Data anomaly (for example a 25% one-day jump) | Row flagged and logged; the symbol is added to `no_entry` for that day; alert for manual review |
| Model file missing or corrupt | Fall back to the previous `dim_model` version for the selected model; log WARNING; if none exists exit 2 |
| Job skipped for several days | Next run processes each missed trading date in order; the first one settles the pending targets at its open |
| Hard data-quality failure | Run FAILED, exit 4, no trading |

---

## 6. Monitoring, logging, alerts

- `etl_run_log` records every ETL run. `logs/daily.log`: INFO for steps, WARNING for fallbacks.
- Dashboard banner turns red if the latest data is older than `paper.stale_banner_days` calendar days.
- `src/paper/alerts.py`: `alert(level, message)` writes to `logs/alerts.log` and stderr; sends email only if `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `ALERT_TO` are set. Alert for: failed ETL, no new index bar for `data.stale_after_days` days, regime change, paper drawdown above `paper.alert_drawdown`.

---

## 7. Reproducibility and retraining

- Seeds from config for K-Means, Random Forest, XGBoost; `requirements.lock`; every model has a `dim_model` row with `trained_through` and parameters; every signal links to its `model_key`.
- Every backtest or paper run has a `run_id`, a manifest with config hash and commit.
- **Retraining policy:** once a year, at the first daily run of a new calendar year (the job trains `wf<year>` as in step 3d); keep the previous model for comparison. Never retrain reactively after a few bad weeks: short windows are dominated by noise and reacting reintroduces overfitting. The regime model is not refit (`regime.refit: none`).
