# TESTS: required tests and numeric fixtures

Tests are the executable form of the spec. Each fixture value below was worked out by hand in the original blueprint and is the expected answer. Write tests before the code they cover. Names in `code font` are mandatory names (referenced from `06` section 9 and `AGENTS.md`).

---

## 1. Conventions

- Fast tests run in `make check` (`-m "not slow"`). Mark integration and smoke tests `@pytest.mark.slow`.
- Use `pytest.approx` for floats (`rel=1e-6` unless stated).
- No network, no real warehouse. Use fixtures from `07` section 8.

---

## 2. Synthetic market generator (`tests/test_synthetic.py`)

Spec for `make_synthetic_market(cfg)` (seed `synthetic.seed`, stocks `synthetic.n_stocks`, range `data.start_date` to `synthetic.end_date`):

- **Calendar:** business days (Mon to Fri) minus about 9 random weekday holidays per year (seeded). The index has a bar on every trading day.
- **Hidden regime chain:** four states BULL, SIDEWAYS, HIGH_VOL, BEAR. State durations are geometric with mean 80 days; the next state is drawn uniformly from the other three. Daily index return = mean + sigma * N(0,1) with (mean, sigma): BULL (0.0009, 0.007), SIDEWAYS (0.0, 0.006), HIGH_VOL (0.0, 0.015), BEAR (-0.002, 0.018).
- **Stocks:** `beta ~ U(0.7, 1.3)`; return = beta * index return + idiosyncratic noise (sigma uniform 0.008 to 0.012, AR(1) with phi 0.05). Start prices 100 to 3000.
- **OHLC:** open = previous close * (1 + N(0, 0.003)); close from the return; high = max(open, close) * (1 + |N(0, 0.004)|); low = min(open, close) * (1 - |N(0, 0.004)|); volume lognormal (mean 1e6). Index and VIX volume 0.
- **VIX:** `10 + 800 * sigma_state + N(0, 0.8)` clipped at 5; OHLC all near the close.
- **Universe:** symbols `SYN01.NS` upward; sectors cycle through `synthetic.sectors` (includes `Banking` and `IT`); company names `Synthetic Co 01`.

Tests: same seed gives identical frames; different seed differs; no weekend dates; `low <= min(open, close)` and `high >= max(open, close)`; VIX has a clearly higher mean in HIGH_VOL and BEAR days than in BULL days; all four hidden states occur at least once over 2017 to 2025.

---

## 3. ETL (`tests/test_etl_*.py`)

| Test | Assertion |
|---|---|
| `test_timezone_not_shifted` | A bar stamped `2024-03-01 00:00+05:30` becomes date `2024-03-01` (not `2024-02-29`) |
| `test_dedupe_keeps_last` | Duplicate (symbol, date) rows collapse to the later row |
| `test_trading_calendar_from_index` | Rows of a stock on a date with no index bar are dropped |
| `test_ffill_limit_two_days` | A 2-day gap is filled (open = high = low = close = previous close, volume 0, flagged); a 3-day gap leaves the third day missing; nothing is filled before a stock's first bar; no back-fill |
| `test_hard_rejects` | Rows with price <= 0, high < low, high below close by more than 1e-6 relative, or volume < 0 land in `rejected` with a reason |
| `test_soft_flags` | A 25% one-day move is flagged for an EQUITY; zero volume is flagged for EQUITY but never for INDEX or VIX |
| `test_load_idempotent` | Loading the same clean frame twice leaves identical row counts and an identical checksum of `fact_price` |
| `test_load_rollback` | An injected error mid-load leaves no new rows and an `etl_run_log` row with status FAILED |
| `test_dq_checks_pass_and_fail` | Each DQ-01 to DQ-07 passes on clean data and fails (hard ones raise) on injected bad data |
| `test_dim_date_trading_flag` | `is_trading_day` equals "index has a bar"; `is_month_end` is the last trading day of each month |
| `test_extract_retry` | A mocked yfinance that fails twice then succeeds is retried; a symbol that always fails is logged and skipped while others load |

---

## 4. Features and leakage (`tests/test_features.py`)

| Test | Assertion |
|---|---|
| `test_indicator_known_values` | On a hand-built 60-day series: `sma_20`, `return_5d`, `return_20d` equal manual values; RSI of a strictly rising series is 100; RSI of a flat series is 50; MACD of a constant series is 0 |
| `test_features_no_lookahead` | `compute_indicators(prices.iloc[:n])` equals `compute_indicators(prices).iloc[:n]` (rtol 1e-12) for several n |
| `test_panel_no_lookahead` | Build the panel from the full warehouse and from a copy truncated at date T; all feature columns (including regime one-hots) are identical for rows up to T minus one trading day |
| `test_target_is_only_forward_column` | Changing a stock's close on day t+1 changes `target_up` at t and no feature at t |
| `test_forbidden_patterns` | Grep over `src/`: no `shift(-` except a line marked `# label-only` in `panel.py`; no `.bfill(`; no `train_test_split(`; no `shuffle=True` |
| `test_warmup_rows_dropped` | A stock's first 50 panel rows are excluded; no NaN in `get_feature_cols(...)` for training rows |
| `test_olap_queries_run` | Each query in `sql/olap/` executes on the smoke warehouse without SQL errors; the roll-up (02) equals a hand aggregation of query 01's inputs for one sector and year |

---

## 5. Regimes and classifiers (`tests/test_regime.py`, `tests/test_models.py`)

**K-Means worked example** (`test_nearest_centre_example`): centres Bull (1.0, -0.7), Sideways (0.0, -0.3), Bear (-1.4, 1.5); point (1.2, -0.6). Distances: **0.224**, **1.237**, **3.342**; nearest is Bull.

**Smoothing** (`test_trailing_mode_smoothing`): window 5, labels `[A, A, A, B, A, A, B, B, B]` give `[A, A, A, A, A, A, A, B, B]`. Tie rule: window 2 on `[A, B]` gives B at index 1 (most recent tied label wins). The smoothed label at day t never depends on days after t (compare truncated and full runs).

| Test | Assertion |
|---|---|
| `test_regime_fit_window` | The scaler and KMeans are fit only on rows with date <= `train_end` (inspect `n_samples_seen_`) |
| `test_regime_deterministic` | Same seed gives identical cluster labels |
| `test_regime_profile_sanity` | On smoke data, training period: BULL mean 20-day return above 0 and above SIDEWAYS; mean volatility of BEAR and HIGH_VOL above BULL and SIDEWAYS; BEAR mean drawdown below BULL's |
| `test_vix_auto_drop` | With VIX coverage below the threshold, VIX is excluded from regime features and from `get_feature_cols(...)` |
| `test_winsorizer_train_only` | `Winsorizer` limits equal the 1st and 99th percentiles of the fit data; transforming other data does not change them; one-hot columns untouched |
| `test_walk_forward_embargo` | For every test year, `max(train index) < year_start - embargo_days`, and no test-year row is in the training set |
| `test_models_fit_predict` | All four models fit on a small sample and return probabilities in [0, 1] |
| `test_signals_written` | After training, `fact_signal` has `prob_up` for every labelled equity-day with complete features (the panel rows) from `validation_start`; `signal` and `target_weight` are 0; rerun is idempotent; a primary-run update of `signal` survives a retrain |

**Classification metrics worked example** (`test_classification_metrics_example`): TP=280, FP=230, FN=210, TN=280. Accuracy **0.560**; precision 280/510 = **0.549**; recall 280/490 = **0.571**; F1 **0.560**. Actual up = 490 of 1000, so majority baseline (always down) = **0.51**; accuracy minus baseline = 5 points.

**Apriori worked example (stretch, `test_apriori_example`):** 10 weeks, Banking up 6, IT up 5, both 4: support **0.40**, confidence **0.667**, lift **1.33**.

---

## 6. Strategy rules (`tests/test_rules.py`)

All with default rules: BULL entry 0.55 / exposure 1.0, SIDEWAYS 0.60 / 0.6, HIGH_VOL 0.65 / 0.3, BEAR exposure 0; slots 10; exit 0.50.

| Test | Setup | Expected |
|---|---|---|
| `test_rules_from_config` | `build_regime_rules(default cfg)` | entries exactly 0.55, 0.60, 0.65 (no float noise); BEAR entry None |
| `test_bear_is_empty` | BEAR, any signals, any held | empty Series |
| `test_runbook_worked_example` | BULL; held H1..H8 with `prob_up` H1..H7 in {0.52..0.60} and H8 = **0.47**; candidates N1 0.70, N2 0.66, N3 0.61, N4 0.56 (not held), N5 0.54 | keeps H1..H7; drops H8; adds N1, N2, N3 (3 free slots); N4 not added (no slot); N5 below entry; 10 names, each weight **0.10** |
| `test_hysteresis` | BULL; held stock at 0.52; unheld stock at 0.52 | held kept; unheld not entered |
| `test_sideways_and_high_vol_sizing` | 10 qualifying stocks | SIDEWAYS weights 0.06 each (entry 0.60); HIGH_VOL weights 0.03 each (entry 0.65) |
| `test_force_exit` | Held stock at 0.90 in `force_exit` | not kept and not re-entered as a candidate |
| `test_no_entry` | Candidate in `no_entry`; held stock in `no_entry` | candidate skipped; held stock stays |
| `test_determinism_and_ties` | Equal `prob_up` for several candidates, shuffled input order | same result every time, ties by symbol A to Z |
| `test_nan_probs` | NaN `prob_up` for a held and an unheld stock | held fails exit (dropped); unheld never entered |

---

## 7. Execution, metrics, engine (`tests/test_execution.py`, `test_metrics.py`, `test_engine.py`)

**Cost example** (`test_cost_worked_example`): buy 100 at open 1,000 with slippage 0.0005 and cost 0.0010: fill **1,000.50**, trade value **100,050.00**, cost **100.05**, cash out **100,150.05**. A round trip costs about **0.30%** of trade value.

**Metrics fixtures:**

| Test | Fixture | Expected |
|---|---|---|
| `test_max_drawdown_example` | equity 100, 120, 90, 110 | **-0.25** |
| `test_sharpe_example` | daily excess returns constructed to have mean 0.0005 and `std(ddof=1)` 0.009 | `sqrt(252) * 0.0005 / 0.009` = **0.8819** (the blueprint rounds to 0.88) |
| `test_cagr_example` | equity 100 on 2020-01-01 and 121 on 2022-01-01 | about 0.0999 (tolerance 1e-3) |
| `test_sortino_calmar_turnover_exposure` | small hand-built equity and trade tables | match hand calculations |
| `test_profit_factor_no_losses` | all SELLs profitable | NaN, no exception |

**Order planning** (`test_plan_orders`):

| Case | Expected |
|---|---|
| Target 0.10, current 0.12 (diff 0.02 <= 0.25 * 0.10 = 0.025) | no trade |
| Target 0.10, current 0.14 | SELL down to target quantity |
| New position (current 0) | BUY |
| Exit (target absent) | SELL all |
| Quantity | `floor(target * equity / open)` |
| Insufficient cash | sells execute first; buys in descending `prob_up`; the last buy is reduced to what cash allows; cash never negative |

**Engine:**

| Test | Assertion |
|---|---|
| `test_engine_fills_next_open` | Build prices where `open_(t+1)` differs strongly from `close_t`; the fill price equals `open_(t+1) * (1 + slippage)`, never `close_t` |
| `test_engine_zero_cost_identity` | One stock, constant target weight, zero slippage and cost, fractional shares: final equity equals `initial * close_end / open_(day 2)` computed by hand, within 1e-9 |
| `test_cash_never_negative` | Property test over random synthetic signals: `cash >= 0` on every day |
| `test_stop_loss_and_cooldown` | After a crash below 93% of the average fill, the symbol is sold at the next open and not re-bought for `cooldown_days` trading days even with `prob_up` 0.9 |
| `test_average_cost_and_reason` | Two buys at different prices give the quantity-weighted `avg_fill`; a partial sell leaves it unchanged; a full sell resets it; trade rows carry `ENTRY`, `REBALANCE`, `EXIT` or `STOP_LOSS` as appropriate |
| `test_engine_period_guard` | `run_backtest` raises `ValueError` for `end >= test_start` unless `allow_test_period=True` |
| `test_missing_bar_skips_order` | A symbol with no bar on the execution day is skipped without error |
| `test_persist_idempotent` | Running the same `run_id` twice leaves identical `fact_trade` and `fact_portfolio_daily` rows |
| `test_baselines_run` | B1, B2, B3, B4 and V0 to V2 run on smoke data; V0, V1, V2 equity curves differ; B4 is reproducible from its seed |
| `test_primary_run_writes_signals` | Only a run with `write_signals=True` changes `fact_signal.signal` and `target_weight` |
| `test_bootstrap_reproducible` | Same seed, same interval; the point estimate lies inside the interval on a long synthetic return series |
| `test_test_lock` | First `final_test` run writes `TEST_RUN_LOCK.json` (including `data_end_date`); a second invocation exits 3; `--allow-rerun "reason"` runs and appends to `test_run_log.jsonl` with count 2 |
| `test_tune_grid_size` | Number of grid trials equals **eligible** models x 3 entry bases x slots = 3 x 3 x 2 = 18 (`test_grid_size_is_18_when_the_tree_is_excluded`); the arithmetic trap of adding one trial per model instead of one entry base per model is covered by `test_grid_size_would_be_24_if_every_model_were_eligible`; each trial is in the trial log; re-running `tune` adds no rows; `calibration` rows are excluded from the count |
| `test_eligibility_rule` | The R31 rule rejects the decision tree's measured distribution (76.4% tied, `p95-p50` 0.0000) and accepts logreg, rf and xgb; the thresholds are the documented constants 0.20 and 0.01; the rule reads only distribution fields, so outcome fields in its input cannot change the verdict; no eligible model sits within an order of magnitude of either boundary |
| `test_tie_break_reporting` | A new entry on the cutoff probability counts as tie-broken while entries above it do not; a fully tied candidate field makes every entry tie-broken, which is the hazard R31 catches; a run with no new entries reports NaN share, not zero; a run above the 5% cap is flagged and excluded from selection |

---

## 8. Paper trading (`tests/test_paper.py`, mostly `@pytest.mark.slow`)

| Test | Assertion |
|---|---|
| `test_paper_replay_equals_backtest` | On the smoke warehouse, `daily_job --no-etl --run-id paper_replay --from D1 --to D2` produces `fact_portfolio_daily` equity equal (1e-6) to a backtest of the same period and model, starting flat. This proves backtest and paper use the same code |
| `test_daily_job_idempotent` | Running the job twice for the same dates leaves all tables unchanged |
| `test_holiday_exit_zero` | A date range with no index bar exits 0 and writes nothing |
| `test_catch_up_in_order` | Skipping several days then running processes them in order; the first day settles the previous pending targets at its open |
| `test_model_fallback` | Corrupting the current model file makes the job use the previous `dim_model` version and log a WARNING; with no previous version the job exits 2 |
| `test_new_year_trains_model` | On the first processed date of a year with no `wf<year>` model, the job trains one and records it in `dim_model` |
| `test_missing_stock_no_new_entry` | A forward-filled stock is never newly entered that day, a held one may stay |
| `test_etl_failure_no_trading` | A failing ETL exits 2, writes a FAILED run, and no trades or portfolio rows are added |
| `test_alerts` | Drawdown above `paper.alert_drawdown` writes an alert line |
| `test_anomaly_no_entry_and_alert` | A 25% one-day jump in a stock: the symbol is in `no_entry` that day and an alert line is written |
| `test_hard_dq_failure_exit_4` | An injected hard data-quality failure makes the job exit 4 with no trades or portfolio rows added |
| `test_date_commits_atomically` | An exception injected after the MARK step of a date leaves no `fact_portfolio_daily`, `fact_trade` or `pending_target` rows for that date; the next run reprocesses it |
| `test_new_run_starts_at_latest_date` | A new `paper_live` run with no `--from` processes only the latest trading date, flat, with `backtest.initial_capital` |
| `test_paper_requires_lock` | On a config without `TEST_RUN_LOCK.json`, a new `paper_live` run is refused (exit 3); a `paper_replay` run is allowed; only `paper_live` writes `fact_signal`, and only for dates after the lock's `data_end_date` |

---

## 9. Integration and dashboard (`@pytest.mark.slow`)

| Test | Assertion |
|---|---|
| `test_smoke_pipeline` | `make smoke` completes in under 5 minutes on a laptop with no network; afterwards: `fact_price`, `fact_indicator`, `fact_market_regime`, `fact_signal` non-empty; no NaN in key columns; all `fact_trade` rows have a valid `run_id`; `summary_test.csv`, `RESULTS.md`, `leakage_checklist.md`, trial log exist |
| `test_results_md_matches_csv` | Every number in the `RESULTS.md` results table equals its source CSV value |
| `test_dashboard_renders` | `streamlit.testing.v1.AppTest` runs `src/dashboard/app.py` against the smoke warehouse and against an empty database with no exception |
| `test_reproducible_run` | Two runs of the smoke pipeline from scratch produce identical `summary_validation.csv` |
