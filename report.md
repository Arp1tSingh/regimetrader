# RegimeTrader — Project Report

**A regime-aware Nifty 50 direction predictor, data warehouse, backtester and paper-trading console.**

**Academic Data Warehousing and Mining project. Not investment advice. No real money. No broker integration.**

| Item | Value |
|---|---|
| Repository | `https://github.com/Arp1tSingh/regimetrader` |
| Report date | 2026-10-28 |
| Code state | `main`, HEAD `83409e6`, tracked tree clean, `origin/main` matches; `report.md` is new/untracked |
| Frozen test period | 2024-01-01 to 2026-10-09 (685 trading days) |
| Current warehouse data | 2015-01-02 to 2026-10-28 (2,910 trading dates, 52 instruments) |
| Current paper run | `paper_live`, 8 processed dates, 2026-10-16 to 2026-10-28, currently flat |
| Last full test run | 565 passed; Ruff clean |

---

## 0. How to read this report

This report describes the project as it exists on 2026-10-28. It is built from the
repository's own artifacts: the README, the specification and decisions, the frozen
configuration, the result CSVs, the run manifests, the test lock, and the current SQLite
warehouse. **No new model training, tuning, backtest or final test was run to produce this
report.** The frozen test result is therefore exactly the result already recorded by the
authorised run; the current paper-trading state is reported separately and is **not** a new
performance result.

Three dates matter:

1. **2022-12-31** — the end of the training period. Models, scalers, winsorizers and the
   regime model are fitted only on data through this date.
2. **2023-12-31** — the end of the validation period. Model selection, strategy tuning,
   activity filtering and the B4 null filter use this year only.
3. **2026-10-09** — the frozen test-period end. The one-time test evaluation is locked at
   this date. Data after it is used by the paper job, not by a new test.

A fourth date is operational: **2026-10-28**, the latest market bar and the latest
`paper_live` decision in the current warehouse.

Two warnings apply throughout:

- **The frozen test found no demonstrated edge.** The frozen strategy's Sharpe confidence
  interval crosses zero, it lost to a plain equal-weight basket and to a simple trend rule,
  and its 2026 return was negative. The full result is in §8.
- **The stored V1 ablation is not a clean “regime filter only, cash otherwise” test.** It
  stayed invested on 684 of 685 test days and held 10 names rather than the whole universe.
  This weakens the README/RESULTS claim that “removing the classifier improved the system.”
  The evidence and the exact scope of the problem are in §6.4 and §8.9. The frozen V2
  result and the B1/B2/B3 comparisons are unaffected.

The report uses **trade rows** for the `n_trades` column. `RESULTS.md` labels that column
“round trips,” but it is the number of individual buy/sell rows, not completed round trips.

---

## 1. Executive summary

### 1.1 What the project is

RegimeTrader is an end-to-end, laptop-scale quantitative research system for the Indian
large-cap market. It:

1. downloads and validates daily prices for the current Nifty 50 universe, the Nifty 50
   index (`^NSEI`) and India VIX (`^INDIAVIX`);
2. loads them into a SQLite star schema with dimensions, fact tables, an ETL audit log and
   idempotent upserts;
3. computes technical indicators and a leakage-safe feature panel;
4. clusters market days into **BULL**, **SIDEWAYS**, **BEAR** and (if selected)
   **HIGH_VOL** regimes with K-Means;
5. trains four next-day direction classifiers — logistic regression, decision tree,
   random forest and XGBoost — in a yearly walk-forward protocol;
6. converts probabilities and regimes into long-only target weights with thresholds,
   exposure caps, hysteresis, a rebalance band and a stop-loss;
7. backtests the strategy against buy-and-hold, equal-weight and trend baselines, with
   transaction costs and next-open execution;
8. runs the same strategy code path in a crash-safe daily paper-trading job;
9. exposes six read-only dashboard screens, including an honest “Today's signals” view and
   a fake-money Bot demo replay;
10. is covered by a 565-test suite, a leakage checklist, run manifests, a trial log and a
    one-time test lock.

### 1.2 The frozen test result

The frozen configuration is random forest, `entry_base = 0.504415`, `slots = 10`. On the
locked test period (2024-01-01 to 2026-10-09, 685 trading days), the headline result is:

| Run | Sharpe | 95% bootstrap CI | CAGR | Max drawdown | Trade rows |
|---|---:|---:|---:|---:|---:|
| **V2 frozen strategy (rf)** | **0.4848** | **[−0.4561, 1.5437]** | **5.48%** | **19.32%** | **1,004** |
| B1 buy and hold `^NSEI` | 0.1576 | [−0.7746, 1.1081] | 1.20% | 15.59% | 1 |
| B2 equal weight, monthly | 0.7017 | [−0.1866, 1.6856] | 8.82% | 15.06% | 1,076 |
| B3 trend `sma_20 > sma_50` | 0.6117 | [−0.2256, 1.4971] | 5.02% | 9.75% | 819 |
| V0 classifier only, no regime rules | 0.4481 | [−0.5046, 1.4946] | 5.96% | 22.37% | 1,142 |
| V1 regime filter only (not a clean ablation) | 0.6738 | [−0.2337, 1.6914] | 10.29% | 16.37% | 322 |
| V2 with zero costs | 0.8949 | [−0.0516, 1.9391] | 11.15% | 15.66% | 1,000 |

The honest reading:

- V2 **beat buy-and-hold** on Sharpe and CAGR, but **lost on drawdown** (19.32% versus
  15.59%). The confidence interval on the difference is wide enough to include “no
  difference.”
- V2 **lost to the equal-weight basket** on Sharpe (0.4848 versus 0.7017), CAGR (5.48%
  versus 8.82%) and drawdown (19.32% versus 15.06%).
- V2 **lost to the 20/50-day trend rule** on Sharpe (0.4848 versus 0.6117) and drawdown
  (19.32% versus 9.75%), while having a slightly higher CAGR (5.48% versus 5.02%).
- V2 was slightly better than V0 on Sharpe and drawdown, but V0 had a slightly higher CAGR.
- **Every 95% confidence interval crosses zero.** There is no statistically reliable
  outperformance in this sample.
- **2026 was the bad year:** V2 returned −17.52% in 2026, versus −13.77% for buy-and-hold,
  −6.39% for equal weight and −6.65% for the trend rule.
- **Costs matter:** with zero costs, V2's Sharpe would be 0.8949 and CAGR 11.15%; costs
  remove about 0.41 Sharpe and 5.7 percentage points of CAGR. The no-cost confidence
  interval still crosses zero.
- **The classifier showed no edge:** test AUCs were 0.514–0.520 and the best test edge over
  the majority class was +0.0094 (XGBoost); validation's best edge was +0.0001 (decision
  tree). See §5.4.
- **The B4 null is strong but handicapped:** V2's 0.4848 is above all 200 test-period
  permutations, whose 90th percentile was about −0.227 and whose best was 0.1198. But the
  null traded 2,478–2,735 trade rows against V2's 1,004 (about 2.6×), so it paid much more
  in costs. “Beat every permutation” overstates the evidence. See §8.7.
- **The V1 ablation is not clean:** the stored V1 run is a 10-name, mostly invested basket
  with regime-gated entries, not the spec's “equal-weight all stocks in BULL, cash
  otherwise, no stop-loss” variant. Its 0.6738 Sharpe therefore does not establish that
  removing the classifier improved the system. See §6.4 and §8.9.

### 1.3 Current operational state

The frozen test ended on 2026-10-09. The warehouse and the paper job have since moved
forward to **2026-10-28**:

- the latest ETL run is an incremental load on 2026-10-28: 624 rows extracted, 624 loaded,
  0 rejected;
- the latest stored Nifty 50 close is **22,698.85** and India VIX is **16.245**;
- the latest regime is **SIDEWAYS**;
- `paper_live` has **8 processed dates**, 2026-10-16 through 2026-10-28;
- every paper decision is **flat**: equity and cash are ₹10,00,000, there are 0 positions,
  0 trades and 0 pending targets;
- on 2026-10-28, 50 names were scored and all target weights were 0.0; the maximum
  `prob_up` was **0.5217796**, below the SIDEWAYS entry threshold of **0.554415** by about
  **0.0326**;
- the dashboard is running locally at `http://127.0.0.1:8501`; its health endpoint returned
  `ok` when this report was compiled.

**This flat paper state is an operational state, not performance evidence.** It shows that
the daily job, warehouse and dashboard work; it says nothing about whether the strategy
makes money.

### 1.4 What was delivered

| Area | Delivered |
|---|---|
| Warehouse | SQLite star schema, 13 tables, 144,105 price rows, 52 instruments, 2015-01-02 to 2026-10-28 |
| ETL | yfinance/NSE/synthetic sources, idempotent upserts, data-quality checks, audit log |
| OLAP | Roll-up, drill-down, slice, dice, pivot, window query and mining-linked queries |
| Mining | 19-feature panel, K-Means regimes, four classifiers, walk-forward evaluation |
| Strategy | Long-only regime rules, 10 slots, thresholds, exposure caps, band, stop-loss, cooldown |
| Backtest | Next-open engine, costs, slippage, bootstrap CIs, six baselines/variants, B4 null |
| Paper trading | Crash-safe daily job, one transaction per date, pending targets, daily reports |
| Dashboard | Six read-only screens, honest flat-signal view, fake-money replay |
| Tests | 565 passed, 26 test modules (28 Python files under `tests/`), Ruff clean, leakage checklist, test lock |
| Documentation | README, spec, decisions, build plan, progress log, results, run manifests |

---

## 2. Problem statement, objectives and scope

### 2.1 The problem

A retail investor or small advisory desk faces two questions every day:

1. **Which stocks should I hold tomorrow?** Fifty large-cap stocks produce noisy price
   histories; manual chart reading does not scale.
2. **How much risk should I take right now?** A strategy tuned in a calm uptrend can lose
   badly in a crash. Most simple bots use one model and trade the same way in every market
   condition.

RegimeTrader addresses both by combining a **market-regime layer** (what kind of market is
this?) with a **stock-direction layer** (which stocks are likely to rise?) and a **risk
layer** (how much capital may be invested in this regime?).

### 2.2 Objectives

| ID | Objective | Measurable outcome | Status |
|---|---|---|---|
| O1 | Build a star-schema warehouse | ≥10 years of daily data for 50 stocks, the index and VIX; dimensions and facts; idempotent loads | Met: 2015-01-02 to 2026-10-28; 52 instruments; 144,105 price rows |
| O2 | Build a reproducible ETL pipeline | Extract, validate, load, audit; reject bad rows; rerun without duplicates | Met: 6 audit runs; latest incremental 624/624, 0 rejected; 0 duplicate `(date, stock)` rows |
| O3 | Support OLAP analysis | Roll-up, drill-down, slice, dice, pivot, window functions | Met: seven populated OLAP outputs; two mining-linked outputs are stale/empty (see §4.6) |
| O4 | Discover market regimes | K-Means on market features; interpretable labels; frozen model | Met: k=3 BULL/SIDEWAYS/BEAR; 2,886 regime rows; no HIGH_VOL in k=3 |
| O5 | Predict next-day direction | Four classifiers; chronological walk-forward; honest evaluation | Met: logreg, tree, rf, xgb; yearly wf2023–wf2026; test AUC 0.514–0.520 |
| O6 | Build and evaluate a trading strategy | Long-only rules, costs, baselines, ablations, confidence intervals | Met as an evaluation: no demonstrated edge; V1 ablation is not clean |
| O7 | Paper-trade the strategy | Daily job, same code path, crash-safe state, dashboard | Met operationally: 8 flat dates; not a performance claim |
| O8 | Document and test | README, spec, tests, manifests, leakage checklist, report | Met: 565 tests; Ruff clean; report present |

The original brief's objectives are O1–O6 in `docs/human/01_problem_and_use_case.md`;
the table above restates them as eight engineering objectives, separating ETL, paper
operation and documentation. The project is an **academic prototype**, not a production
trading system.

### 2.3 Scope

**In scope:**

- daily, end-of-day data for the current Nifty 50 universe plus the index and VIX;
- a single Indian market, 2015–2026;
- long-only, cash-or-stock positions;
- one pooled classifier across stocks, not one model per stock;
- one regime model, not a regime ensemble;
- simplified but explicit transaction costs and next-open fills;
- local SQLite, local files, local dashboard;
- paper trading only, with no broker.

**Out of scope:**

- intraday data, order-book data, futures, options, short selling or leverage;
- real-money execution, broker integration, SEBI compliance, tax accounting;
- market-impact and liquidity modelling beyond fixed slippage;
- a point-in-time historical index membership (the universe is the current list);
- live news, fundamentals, sentiment or alternative data;
- a production scheduler or high-availability deployment (kits exist, but nothing is
  installed by default);
- association-rule mining (a documented stretch goal, not implemented);
- V3/V4 strategy variants and per-regime models (documented, not implemented).

### 2.4 What “success” means here

Success is **not** “the strategy made money.” The project's honesty rules define success as:

- the warehouse and ETL are correct and reproducible;
- the mining pipeline is leakage-safe;
- the backtest is realistic enough to be informative;
- the baselines are explicit;
- the test period is evaluated once;
- negative results are reported as negative;
- the paper job and dashboard operate end to end.

By those criteria the engineering is largely successful. By the market criterion — a
reliable, deployable edge — the result is negative.

---

## 3. System architecture

### 3.1 Layers

```text
SOURCE        yfinance API        NSE CSV fallback        synthetic generator (tests)
                    \                  |                    /
INTEGRATION   Extract  ->  Transform and validate  ->  Load        (src/etl)
                                   |
WAREHOUSE     dimensions + fact tables in SQLite (single source of truth)
                 |            |             |                |
ANALYTICS     OLAP queries  Feature builder  Regime clustering  Direction classifiers
              (src/olap)    (src/features)   (src/regime)       (src/models)
                                   \             |             /
APPLICATION   Strategy rules -> Backtester | Paper broker | Dashboard
              (src/trading, src/backtest, src/paper, src/dashboard)
```

The warehouse is the single source of truth. Each module reads inputs from warehouse tables
and writes outputs back to warehouse tables or to versioned files. The backtester and the
paper job call the same functions in `src/trading/`; there is no separate “paper strategy.”

### 3.2 Components

| Component | Main modules | Responsibility |
|---|---|---|
| Common | `src/common/config.py`, `db.py`, `log.py`, `dates.py`, `manifest.py` | Config loading, SQLite connections, logging, date keys, run manifests |
| ETL | `src/etl/pipeline.py`, `extract.py`, `transform.py`, `load.py`, `quality.py`, `universe.py`, `synthetic.py` | Download, validate, transform, upsert, audit, synthetic data |
| Warehouse | `sql/schema.sql`, `src/common/db.py` | DDL, indexes, foreign keys, views |
| OLAP | `src/olap/runner.py`, `sql/olap/*.sql` | Named analytical queries and CSV outputs |
| Features | `src/features/indicators.py`, `panel.py`, `builder.py` | Technical indicators and leakage-safe model panel |
| Regime | `src/regime/regime_model.py` | K-Means fit, label naming, smoothing, persistence, prediction |
| Models | `src/models/train.py`, `factory.py`, `preprocessing.py`, `evaluate.py`, `diagnostics.py` | Walk-forward training, metrics, diagnostics, model files |
| Trading | `src/trading/rules.py`, `execution.py` | Regime rules, target weights, order planning, fills |
| Backtest | `src/backtest/engine.py`, `metrics.py`, `strategies.py`, `baselines.py`, `tune.py`, `final_test.py` | Simulation, metrics, variants, tuning, one-time test guard |
| Paper | `src/paper/daily_job.py`, `store.py`, `report.py`, `alerts.py` | Daily paper run, crash-safe state, reports, alerts |
| Dashboard | `src/dashboard/app.py`, `data.py`, `ui.py`, `bot.py` | Six read-only screens, queries, replay |

### 3.3 Research data flow

```text
1. ETL full refresh
   yfinance/NSE -> staging -> validation -> dimensions + fact_price

2. Feature panel
   fact_price -> indicators (fact_indicator) -> features for day t -> label for t+1

3. Regime model
   market features through 2022-12-31 -> scaler + K-Means (k=3) -> labels for all dates
   -> trailing 5-day mode smoothing -> fact_market_regime

4. Classifiers
   panel + regime one-hot -> yearly walk-forward training (2023, 2024, 2025, 2026)
   -> prob_up per stock/day -> fact_signal

5. Strategy and backtest
   prob_up + regime + prices -> target weights -> next-open fills -> fact_trade
   -> daily portfolio -> fact_portfolio_daily -> summary metrics -> reports/results

6. Paper trading
   latest warehouse data -> same strategy code -> pending_target -> next-open fills
   -> fact_trade + fact_portfolio_daily + reports/daily

7. Dashboard
   warehouse + result CSVs -> six read-only screens
```

### 3.4 Daily paper data flow

```text
latest market bar (t)
  -> ensure inputs: prices, indicators, regime, signals
  -> strategy decides target weights after close of t
  -> write pending_target for t+1 open
  -> next run: fill pending targets at t+1 open, mark to market, write trades
  -> write fact_portfolio_daily
  -> after the run, write a daily markdown report for the latest processed date
```

The paper job processes **one trading date per transaction**. A date is committed by its
`fact_portfolio_daily` row; the transaction also writes any trade and pending-target rows
that date produced. If no portfolio row exists, the date is redone. This is what makes a
crash safe.

### 3.5 Technology stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ (the recorded test environment used Python 3.14.8) |
| Data | pandas, NumPy |
| Storage | SQLite (single-file warehouse) |
| ML | scikit-learn (logistic regression, decision tree, random forest, K-Means, scalers), XGBoost |
| Plots | matplotlib (static figures), Plotly (dashboard) |
| Dashboard | Streamlit |
| Tests | pytest, pytest-cov, Hypothesis |
| Lint/format | Ruff |
| Packaging | `pyproject.toml`, `requirements.txt`, `requirements.lock` |
| Scheduling examples | cron, GitHub Actions, systemd timer, Windows Task Scheduler scripts |
| Hosting examples | Cloudflare quick tunnel, free-tier VPS kit, laptop-as-host kit |

### 3.6 Design principles and invariants

1. **No look-ahead.** Features for day *t* use data through the close of *t*. Only the
   label uses day *t+1*. Orders decided at the close of *t* fill at the open of *t+1*.
2. **Chronological everything.** No shuffling. Train through 2022-12-31, validate 2023,
   test from 2024-01-01.
3. **Fit on training data only.** Winsorization limits, scalers and K-Means are fitted on
   training data and frozen for later dates.
4. **Test once.** The test period is guarded by a lock file and a confirmation flag. The
   recorded lock has `run_count: 2` because the first run crashed in B4 and an authorised
   rerun was needed; see §7.6.
5. **Log every experiment.** `trial_log.csv` records 342 rows, of which 338 count as
   experiments (4 calibration rows excluded).
6. **Config-driven.** Dates, thresholds, costs, seeds and model settings live in
   `config/config.yaml`; the frozen config may only override model, entry base and slots.
7. **Idempotent loads.** Re-running an ETL or job for the same dates changes nothing.
8. **Reproducible.** Seeds, config hash, git commit and run manifests are recorded.
9. **One strategy code path.** Backtest and paper trading call the same `src/trading/`
   functions.
10. **Warehouse is the source of truth.** Modules read inputs from and write outputs to
    warehouse tables, plus model files.
11. **No invented data.** If a source fails, use the documented fallback or stop.
12. **Honest reporting.** Never tune to beat buy-and-hold; report negative results as they
    are.

---

## 4. Data warehouse, ETL and OLAP

### 4.1 Universe and coverage

The warehouse covers **52 instruments**:

- 50 current Nifty 50 stocks;
- the Nifty 50 index, `^NSEI`;
- India VIX, `^INDIAVIX`.

The universe file is `config/nifty50_symbols.csv`, marked “as of 2026-10-10.” It has 50
equity rows across 15 sectors:

| Sector | Stocks |
|---|---:|
| Financial Services | 7 |
| Pharma and Healthcare | 5 |
| Banking | 5 |
| Auto | 5 |
| Metals and Mining | 4 |
| IT | 4 |
| FMCG | 4 |
| Energy | 3 |
| Services | 2 |
| Consumer Durables | 2 |
| Infrastructure | 2 |
| Consumer Services | 2 |
| Materials | 2 |
| Power | 2 |
| Telecom | 1 |

The current universe is a **static current list**, not a point-in-time historical index
membership. This creates survivorship bias: stocks that were in the Nifty 50 in 2015 but
are not in it today are absent, and stocks added recently have shorter histories. Seven of
the 50 stocks start after 2015-01-02; six of those are under 10 years as of the test end
(INDIGO.NS starts in November 2015 and is the exception). `BSE.NS` is a recent addition to
the index. The report returns to this in §14.

The warehouse's price coverage is **2015-01-02 to 2026-10-28**, with **2,910 distinct
trading dates** and **144,105 price rows**. The full Cartesian product would be
52 × 2,910 = 151,320 rows; the 7,215-row difference is exactly the shorter histories of the
seven stocks that start after 2015-01-02. The index and VIX each have all 2,910 bars.

### 4.2 Schema

The canonical DDL is `sql/schema.sql`. The warehouse is a star schema:

| Table | Kind | One row is | Primary key | Written by |
|---|---|---|---|---|
| `stg_price_raw` | staging | one raw bar from the latest extract | none | extract; truncated at the start of each run |
| `etl_run_log` | audit | one ETL run | `run_id` | load |
| `dim_date` | dimension | one calendar day | `date_key` (YYYYMMDD integer) | load |
| `dim_stock` | dimension | one instrument (stock, index or VIX) | `stock_key`, unique `symbol` | universe/load |
| `dim_regime` | dimension | one regime label | `regime_key`, unique `regime_label` | regime model |
| `dim_model` | dimension | one model version | `model_key`, unique `(model_name, version)` | train |
| `fact_price` | fact | one instrument/day OHLCV bar | `(date_key, stock_key)` | load |
| `fact_indicator` | fact | one instrument/day indicator row | `(date_key, stock_key)` | features |
| `fact_market_regime` | fact | one market day's regime | `date_key` | regime model |
| `fact_signal` | fact | one stock/day/model probability | `(date_key, stock_key, model_key)` | train/backtest/paper |
| `fact_trade` | fact | one simulated fill | `trade_id` | backtest/paper |
| `fact_portfolio_daily` | fact | one run/day portfolio mark | `(run_id, date_key)` | backtest/paper |
| `pending_target` | fact | one run/day/stock next-open order | `(run_id, date_key, stock_key)` | paper |

Important schema conventions:

- **Date keys are integers** in `YYYYMMDD` form, never floats or strings.
- **Foreign keys are enabled** on every connection (`PRAGMA foreign_keys = ON`).
- **Prices are adjusted close** where the source provides it; OHLC columns are stored for
  sanity checks and execution approximation.
- **`fact_signal` semantics:** training writes `prob_up` for every model with `signal = 0`
  and `target_weight = 0.0`; only the primary strategy run (`backtest_main`, then
  `paper_live`) updates `signal` and `target_weight` for its selected model rows. Ablations
  and baselines do not overwrite the primary signal. This is decision R3 in
  `DECISIONS.md`. (The paper job stores its rows under the first `rf` row by name; see
  §14.5.)
- **No separate positions table.** Holdings and cash are derived from `fact_trade` and
  `fact_portfolio_daily`. `pending_target` stores next-open orders. This is decision R4.

### 4.3 Current warehouse snapshot

Read-only query against `data/warehouse/regimetrader.db` on 2026-10-28:

| Table | Rows | Notes |
|---|---:|---|
| `dim_date` | 4,319 | 2015-01-01 to 2026-10-28 calendar days |
| `dim_stock` | 52 | 50 equities + index + VIX |
| `dim_regime` | 4 | BULL, SIDEWAYS, BEAR, HIGH_VOL |
| `dim_model` | 16 | 4 model families × 4 walk-forward years |
| `fact_price` | 144,105 | 2015-01-02 to 2026-10-28; 2,910 dates |
| `fact_indicator` | 141,195 | 50 equities + index; VIX has no indicator rows |
| `fact_market_regime` | 2,886 | 2015-02-02 to 2026-10-28 |
| `fact_signal` | 185,376 | 2023-01-02 to 2026-10-28; 16 model versions (4 families × 4 years) |
| `fact_trade` | 5,364 | seven test runs; paper has 0 |
| `fact_portfolio_daily` | 4,803 | seven test runs × 685 days + 8 paper days |
| `pending_target` | 0 | paper book is flat |
| `etl_run_log` | 6 | latest run 2026-10-28 |
| `stg_price_raw` | 624 | latest extract batch |

Regime counts in the current warehouse:

| Regime | Current all rows | Training through 2022-12-31 | Test 2024-01-01 to 2026-10-09 |
|---|---:|---:|---:|
| BULL | 1,889 | 1,167 | 495 |
| SIDEWAYS | 970 | 754 | 190 |
| BEAR | 27 | 27 | 0 |
| HIGH_VOL | 0 | 0 | 0 |
| **Total** | **2,886** | **1,948** | **685** |

The current SIDEWAYS count is 970 rather than the 962 recorded in
`reports/results/regime_counts.csv` because the CSV is a snapshot through 2026-10-09 and
the current warehouse has eight more SIDEWAYS paper dates. There are no BEAR days in the
test period, which is an important limitation for interpreting the BEAR rules.

### 4.4 ETL pipeline

The ETL is implemented in `src/etl/` and is designed to be idempotent.

**Extract**

- Primary source: `yfinance` daily bars, with the configured start date (2015-01-01) and
  today's date.
- Fallback: a local NSE CSV if the network/source fails, as documented in the spec.
- Test/smoke source: a deterministic synthetic generator; tests never need network.
- The index and VIX are extracted alongside the stocks.
- The local timestamp is made timezone-naive **without converting to UTC first**, so the
  trading date does not shift back one day.

**Transform and validate**

- Normalize column names and types; parse dates; enforce `YYYYMMDD` date keys.
- Validate OHLC relationships, non-negative volume, duplicate `(date, symbol)` rows and
  sufficient coverage.
- Flag suspicious bars (for example, extreme adjusted-close jumps) rather than silently
  accepting them.
- Volume checks apply to equities only; the index and VIX have zero volume by construction.
- Forward-fill at most two days for a missing stock bar; never back-fill.
- Drop rows that fail hard validation; count rejected rows in the audit log.

**Load**

- Upsert dimensions and facts with `INSERT ... ON CONFLICT DO UPDATE`.
- Wrap each load in a transaction; enable foreign keys.
- Write one `etl_run_log` row with mode (`FULL` or `INCREMENTAL`), extracted/loaded/rejected
  counts, timestamps and status.
- Truncate `stg_price_raw` at the start of each run so it always represents the latest
  extract batch.

**Indicators**

- Indicators are recomputed from full history, not incrementally, to avoid EWM drift.
- The indicator builder writes `fact_indicator` for equities and the index.
- The feature panel joins prices, indicators, VIX and regime one-hots.

### 4.5 Data-quality checks

The project defines hard checks that must pass before a real-data load is accepted, plus
soft flags that are recorded but do not stop the load. The categories are:

| Check | What it verifies | Failure behavior |
|---|---|---|
| Schema and types | Required columns exist; dates and numerics parse | Hard fail |
| Duplicate bars | No duplicate `(date, stock)` in the incoming batch or the warehouse | Hard fail |
| OHLC sanity | High ≥ low; open/close within a plausible range; non-negative prices | Hard fail for invalid rows |
| Volume sanity | Equity volume non-negative; index/VIX volume exempt | Hard fail for invalid rows |
| Date coverage | Enough sessions per instrument; no impossible gaps | Hard fail or flag |
| Price jumps | Extreme adjusted-close moves are flagged for review | Soft flag |
| Universe coverage | All required instruments present; sector requirements for the full 50 | Hard fail for missing required names |

The final full ETL run on 2026-10-10 extracted 143,883 rows, loaded 143,481 and rejected 419
(0.29%), within the documented 1% rejection limit. The latest incremental run on
2026-10-28 extracted 624 rows, loaded 624 and rejected 0. The warehouse has **zero duplicate
`(date, stock)` price rows**. It has **103 flagged price rows out of 144,105 (0.0715%)**,
well below the 0.5% soft-flag threshold.

### 4.6 OLAP analysis

The OLAP layer is a set of named SQL queries in `sql/olap/`, run by
`src/olap/runner.py`. The committed outputs are in `reports/results/olap/`.

| Query | Operation | Output | Rows |
|---|---|---|---:|
| `01_rollup_sector_quarter` | Roll-up | Average daily return by year, quarter, sector | 720 |
| `02_rollup_sector_year` | Roll-up | Average daily return by year, sector | 180 |
| `03_drilldown_sector_to_stock` | Drill-down | Sector → month → stock average daily return | 40 |
| `04_slice_year` | Slice | One year's sector returns and volume | 15 |
| `05_dice_sectors_years_quarters` | Dice | Selected sectors, years and quarters | 12 |
| `06_pivot_sector_by_year` | Pivot | Sector × year average daily return | 15 × 3 (2022–2024) |
| `07_window_top5_momentum` | Window function | Top five 20-day momentum stocks on a date | 5 |
| `08_regime_profile` | Mining-linked OLAP | Market profile by regime | 0 in committed CSV |
| `09_classifier_by_regime` | Mining-linked OLAP | Classifier edge by regime | 0 in committed CSV |

The OLAP outputs show the expected sector rotation:

- In the **2024** pivot, Consumer Services was the strongest sector at **+0.365% average
  daily return**, followed by Telecom (+0.188%) and Services (+0.151%). Consumer Durables
  was the weakest at **−0.093%**, followed by FMCG (−0.031%) and Metals and Mining (+0.014%).
- In **2022**, Metals and Mining was strongest (+0.138%) and IT weakest (−0.090%).
- In **2023**, Consumer Services was strongest (+0.339%) and Banking weakest (+0.044%).
- In the committed **2026** OLAP output (a 2026-10-10 snapshot), Services was strongest
  (+0.052% average daily return), Metals and Mining second (+0.043%), while IT was weakest
  (−0.123%), FMCG −0.089% and Auto −0.077%. These are average daily returns in percent, not
  annual returns.

The test-period regime breakdown is:

| Regime | Days | Mean daily return | Up days |
|---|---:|---:|---:|
| BULL | 495 | +0.0408% | 253 |
| SIDEWAYS | 190 | −0.0161% | 94 |
| BEAR | 0 | — | — |

The training-period regime profile is more informative because it includes the 27 BEAR
days:

| Regime | Days | Mean 20-day return | Mean 20-day volatility | Mean VIX | Mean drawdown |
|---|---:|---:|---:|---:|---:|
| BULL | 1,167 | +2.56% | 11.55% | 15.42 | −2.47% |
| SIDEWAYS | 754 | −1.04% | 18.54% | 20.23 | −10.37% |
| BEAR | 27 | −15.33% | 72.08% | 57.16 | −29.32% |

The BEAR profile is dominated by the COVID crash and has only 27 observations. It is not a
reliable estimate of “average bear-market behavior.”

**Committed OLAP caveat:** the committed `08_regime_profile.csv` and
`09_classifier_by_regime.csv` outputs are empty. The regime profile is available in
`reports/results/regime_profile.csv`, and the dashboard reads the warehouse directly. The
empty CSVs are a stale-output/documentation issue, not a missing warehouse table.

### 4.7 Warehouse and OLAP caveats

1. **Survivorship bias.** The universe is the current Nifty 50 list. Historical returns are
   measured on today's survivors, which flatters long-run performance.
2. **Short histories.** Seven stocks start after 2015-01-02; six are under 10 years as of
   the test end (INDIGO.NS starts in November 2015 and is the exception). The panel and
   classifier handle this by dropping missing rows, but the effective sample is smaller for
   those names.
3. **Adjusted prices.** Corporate actions are handled through adjusted close where the
   source provides it; the execution model is still a simplification.
4. **Source quality.** yfinance is an unofficial interface. Gaps, revisions and occasional
   bad prints are possible; the ETL flags and rejects some, but cannot guarantee vendor
   quality.
5. **VIX fallback.** If India VIX is unavailable, the documented fallback is used; the
   regime model records whether VIX was available in its manifest.
6. **Single-writer SQLite.** Fine for a laptop and a single daily job; not a multi-user
   warehouse.
7. **Static regime model.** K-Means is fitted once on training data and frozen; it does not
   adapt to structural change after 2022.
8. **OLAP outputs are snapshots.** Several CSVs are generated at specific dates; the
   dashboard should be preferred for current values.

---

## 5. Data mining: features, regimes and classifiers

### 5.1 Feature catalogue

The model panel uses 19 features. All features for day *t* use data through the close of
*t*; only the label uses *t+1*.

| # | Feature | Formula / meaning | Source |
|---|---|---|---|
| 1 | `ret_1d` | 1-day close-to-close return | `fact_indicator.return_1d` |
| 2 | `ret_5d` | 5-day close-to-close return | `fact_indicator.return_5d` |
| 3 | `ret_20d` | 20-day close-to-close return | `fact_indicator.return_20d` |
| 4 | `px_to_sma20` | `close / SMA20 − 1` | derived |
| 5 | `px_to_sma50` | `close / SMA50 − 1` | derived |
| 6 | `sma20_to_sma50` | `SMA20 / SMA50 − 1` | derived |
| 7 | `rsi_14` | 14-day Wilder RSI, 0–100 | derived |
| 8 | `macd_hist_pct` | `(MACD − signal) / close` | derived |
| 9 | `bb_pct_b` | Position inside 20-day, 2σ Bollinger Bands | derived |
| 10 | `atr_pct` | 14-day ATR / close | derived |
| 11 | `vol_20d` | 20-day return standard deviation × √252 | derived |
| 12 | `volume_ratio_20` | Volume / 20-day average volume | `fact_price.volume` |
| 13 | `rel_strength_20` | Stock 20-day return − index 20-day return | derived |
| 14 | `mkt_ret_20d` | Nifty 50 20-day return | index |
| 15 | `india_vix` | India VIX close | `fact_price` |
| 16 | `regime_bull` | One-hot regime indicator | `fact_market_regime` |
| 17 | `regime_sideways` | One-hot regime indicator | `fact_market_regime` |
| 18 | `regime_high_vol` | One-hot regime indicator | `fact_market_regime` |
| 19 | `regime_bear` | One-hot regime indicator | `fact_market_regime` |

The technical indicators are computed in `src/features/indicators.py`; the derived features
and the panel are built in `src/features/panel.py` and `src/features/builder.py`.

### 5.2 Preprocessing and target

- **Warm-up:** the first 50 rows per stock are dropped because the 50-day SMA and other
  indicators are not yet valid.
- **Missing values:** forward-fill at most two days; never back-fill. Missing VIX is handled
  by the documented fallback; the regime manifest records whether VIX was used.
- **Winsorization:** 1st/99th percentile limits fitted on the training period only and then
  applied frozen to validation and test.
- **Scaling:** logistic regression and K-Means use a fitted scaler; tree models do not need
  scaling. All preprocessing objects are fitted on training data only.
- **Regime encoding:** one-hot columns for BULL, SIDEWAYS, HIGH_VOL and BEAR. With k=3 the
  HIGH_VOL column is always zero, but the schema and feature list keep it for completeness.
- **Target:** `y = 1` if the next trading day's close is above today's close, else 0. The
  label is the only place `shift(-1)` is allowed.
- **Last row:** the most recent row per stock has no next-day label; it is dropped from
  training/evaluation but kept for live prediction.
- **No shuffling:** every split is chronological.
- **No resampling:** the up share is roughly 50–53%; the majority-class baseline is
  reported explicitly.

### 5.3 Market regimes

#### 5.3.1 Market features

The K-Means regime model uses four market-level features:

- `mkt_return_20d` — Nifty 50 20-day return;
- `mkt_vol_20d` — Nifty 50 20-day return volatility;
- `india_vix` — India VIX close;
- `mkt_drawdown` — Nifty 50 drawdown from its 252-day high.

These are market-wide, not stock-specific. The model is fitted on rows through
**2022-12-31** only, then frozen (`regime.refit: none`).

#### 5.3.2 Choosing k

The k-selection output is:

| k | Inertia | Silhouette | Training days |
|---:|---:|---:|---:|
| 2 | 4,809.36 | 0.7706 | 1,948 |
| 3 | 3,287.96 | 0.3619 | 1,948 |
| 4 | 2,582.12 | 0.3635 | 1,948 |
| 5 | 2,145.84 | 0.3203 | 1,948 |
| 6 | 1,844.26 | 0.3162 | 1,948 |

k=2 has the highest silhouette, but two clusters are too coarse for a regime overlay.
k=3 and k=4 have nearly identical silhouette scores; k=4 split two clusters that each
contained under 3% of training days, which is too fragile. **k=3 was chosen** for
interpretability and stability, producing BULL, SIDEWAYS and BEAR. The `HIGH_VOL` label
remains in `dim_regime` but has zero rows under k=3.

#### 5.3.3 Fitting, naming and smoothing

1. Drop rows with missing market features.
2. Fit the scaler and K-Means on rows with date ≤ 2022-12-31.
3. Name clusters by their market profile: the cluster with the strongest returns and lowest
   VIX is BULL; the cluster with the worst returns, highest volatility and highest VIX is
   BEAR; the remaining cluster is SIDEWAYS.
4. Predict raw cluster ids for all dates with the frozen scaler and centres.
5. Smooth with a **trailing 5-day mode**: the label for day *t* is the most common label in
   the last five labels, with ties going to the most recent tied label. This removes
   one-day flicker without using future data.
6. Write `fact_market_regime`.

The regime model manifest records `k=3`, `use_vix: true`, four features, 2,878 regime rows
written and no sanity failures. The model itself was fitted on the 1,948 training days
through 2022-12-31. The current `fact_market_regime` table has 2,886 rows because eight
paper dates were appended after the fit.

#### 5.3.4 Regime interpretation

The training-period profile is the clearest summary:

| Regime | Days | Mean 20-day return | Mean 20-day volatility | Mean VIX | Mean drawdown |
|---|---:|---:|---:|---:|---:|
| BULL | 1,167 | +2.56% | 11.55% | 15.42 | −2.47% |
| SIDEWAYS | 754 | −1.04% | 18.54% | 20.23 | −10.37% |
| BEAR | 27 | −15.33% | 72.08% | 57.16 | −29.32% |

The labels are intuitive: BULL is calm and rising; SIDEWAYS is choppier and more volatile;
BEAR is the COVID-style crash cluster. The BEAR estimate is based on only 27 days, so it
should not be treated as a general bear-market model.

The test period contains **495 BULL days and 190 SIDEWAYS days, and no BEAR days**. The
BEAR rule (exit everything, zero exposure) is therefore **not tested out of sample**. This
is a material limitation: the strategy's most important risk rule is exercised only in
training.

### 5.4 Direction classifiers

#### 5.4.1 Models and settings

| Model | Settings |
|---|---|
| Logistic regression (`logreg`) | `C=1.0`, `max_iter=1000` |
| Decision tree (`tree`) | `max_depth=4`, `min_samples_leaf=200` |
| Random forest (`rf`) | 300 trees, `max_depth=6`, `min_samples_leaf=50` |
| XGBoost (`xgb`) | 300 rounds, learning rate 0.03, depth 3, subsample 0.8, colsample 0.8 |

One **pooled** model is trained across all stocks, with stock-specific features and regime
one-hots. The model predicts `prob_up` for each stock/day. This is simpler and more stable
than 50 separate models, but it cannot learn stock-specific idiosyncrasies beyond the
features.

#### 5.4.2 Walk-forward protocol

The training protocol is yearly walk-forward:

| Model version | Training data ends | Prediction year |
|---|---|---|
| `wf2023` | 2022-12-31 minus 5-day embargo | 2023 (validation) |
| `wf2024` | 2023-12-31 minus 5-day embargo | 2024 |
| `wf2025` | 2024-12-31 minus 5-day embargo | 2025 |
| `wf2026` | 2025-12-31 minus 5-day embargo | 2026 (through 2026-10-09 for the test) |

The 5-day embargo prevents the label window at the end of the training year from overlapping
the first prediction days. The training window expands each year. Scalers, winsorizers and
the regime model are fitted on training data only.

The `dim_model` table has 16 rows: four model families × four walk-forward years. The model
files are in `models/` (`logreg_wf2023.joblib`, ..., `xgb_wf2026.joblib`, plus
`regime_model.joblib`).

#### 5.4.3 Classifier metrics

Validation (2023) and test (2024-01-01 to 2026-10-09) metrics from
`reports/results/classifier_metrics.csv`:

| Split | Model | Accuracy | Majority accuracy | Edge | AUC | Rows |
|---|---|---:|---:|---:|---:|---:|
| Validation | logreg | 0.498007 | 0.531468 | −0.033461 | 0.522161 | 12,044 |
| Validation | tree | 0.531551 | 0.531468 | +0.000083 | 0.514624 | 12,044 |
| Validation | rf | 0.512454 | 0.531468 | −0.019014 | 0.526831 | 12,044 |
| Validation | xgb | 0.505895 | 0.531468 | −0.025573 | 0.522220 | 12,044 |
| Test | logreg | 0.506637 | 0.501345 | +0.005292 | 0.514024 | 34,200 |
| Test | tree | 0.498187 | 0.501345 | −0.003158 | 0.515682 | 34,200 |
| Test | rf | 0.505789 | 0.501345 | +0.004444 | 0.515684 | 34,200 |
| Test | xgb | 0.510760 | 0.501345 | +0.009415 | 0.520112 | 34,200 |

The validation edges are negative for logreg, rf and xgb, and essentially zero for the
decision tree (+0.000083). The test edges are small positives for logreg, rf and xgb, and a
small negative for the tree. Test AUCs are **0.514–0.520**, barely above random. There is no
evidence of a useful directional classifier.

#### 5.4.4 The decision-tree problem

The decision tree is a degenerate ranker on validation: it produces only **3 distinct
probabilities**, and **76.38%** of validation rows are tied at one value; the p95−p50 spread
is 0.0. A model that gives almost every stock the same score cannot rank a cross-section.
It was therefore excluded from strategy tuning, although it was still trained, stored and
reported. This is a documented decision, not a silent omission.

#### 5.4.5 Why random forest was selected

The frozen strategy uses **random forest**, not the model with the best test AUC. The
selection was made on 2023 validation only, using the tuning grid, the activity filter and
the B4 null filter. XGBoost had a slightly higher test AUC (0.5201 versus 0.5157) and a
higher test edge (+0.0094 versus +0.0044), but that is test-period information and was not
used to select the model. Selecting on test performance would violate the project's
test-once rule.

### 5.5 What the mining results mean

- The regime layer produces interpretable, persistent labels, but the BEAR label is rare
  and absent from the test period.
- The classifiers are close to random. Test accuracy edges of about −0.3 to +0.9
  percentage points, with AUC around 0.52, are not a basis for a trading edge; validation
  edges are negative or zero.
- The random forest was selected by validation filters, not because it was the best test
  classifier.
- The strategy's backtest result therefore cannot be attributed to demonstrated predictive
  skill. A plausible alternative is that its low-churn, regime-gated holdings capture
  momentum or exposure effects. That alternative was never tested directly. This is
  discussed again in §8.9 and §14.

---

## 6. Trading strategy

### 6.1 Principles

1. **Long-only.** The bot holds a stock or holds cash; it does not short.
2. **Daily decisions, next-open execution.** Signals are computed after the close of day
   *t* and filled at the open of day *t+1*.
3. **The regime sets the risk budget.** The classifier ranks stocks; the regime decides how
   much capital may be invested.
4. **Fewer trades.** Hysteresis, a rebalance band and a stop-out cooldown reduce churn,
   because costs destroy small edges.
5. **One code path.** The backtester and the paper job call the same `src/trading/`
   functions.

### 6.2 Regime rules

Entry threshold = `strategy.entry_base + strategy.entry_offsets[regime]`, rounded to six
decimals. Maximum equity invested = `strategy.exposure[regime]`.

With the frozen `entry_base = 0.504415`:

| Regime | Entry threshold on `prob_up` | Maximum equity invested | Behaviour |
|---|---:|---:|---|
| BULL | 0.504415 | 100% | Fully active |
| SIDEWAYS | 0.554415 | 60% | Higher conviction, smaller size |
| HIGH_VOL | 0.604415 | 30% | Very selective, small size |
| BEAR | none | 0% | No entries; exit everything at the next open |

The default spec starting values are 0.55 / 0.60 / 0.65 with 100% / 60% / 30% / 0%
exposure. The frozen config uses a lower BULL threshold because the random-forest
probabilities are compressed; the threshold was selected on validation, not test.

### 6.3 Position and order rules

- **Slots:** 10. Per-slot target weight is `exposure / slots`:
  - BULL: 10% per slot;
  - SIDEWAYS: 6% per slot;
  - HIGH_VOL: 3% per slot;
  - BEAR: 0%.
- **Entry:** a stock may be entered only if `prob_up >= entry threshold` for the current
  regime, it has a bar today, it is not in the no-entry set, and it is not in stop-out
  cooldown.
- **Selection:** rank eligible stocks by `prob_up` descending; ties break by symbol
  ascending (A–Z) so results are deterministic. Take at most the available slots.
- **Held names:** a held name is kept while `prob_up >= exit_prob` (0.50) and the regime is
  not BEAR, even if it is below the entry threshold. This is the hysteresis that avoids
  entering and exiting on small probability changes.
- **Exit:** exit a held name if `prob_up < exit_prob` or the regime is BEAR. BEAR exits
  everything at the next open.
- **Rebalance band:** 25%. Small differences between target and current weight are left
  alone; the position is rebalanced only when the difference is material. This reduces
  turnover.
- **Stop-loss:** 7% below entry, with a 5-day cooldown before re-entry. The stop-loss is
  enabled for V2.
- **No-entry set:** a stock with a flagged bar or a forward-filled bar cannot be newly
  entered; a held position may stay. This prevents trading on suspect data.
- **Cash:** uninvested cash earns zero. The backtest is in nominal INR; there is no interest
  on cash.
- **Whole shares:** the engine buys whole shares and leaves residual cash. B1 uses
  fractional shares because it is an index-equivalent buy-and-hold baseline.
- **Order sequence:** sells first, then buys in descending `prob_up`; buys that would make
  cash negative are skipped. This avoids implicit leverage.

### 6.4 The V1 ablation is not the spec's clean regime-only variant

This is the most important implementation caveat in the report. The spec defines V1 as:

> equal weight across all stocks with a bar today when the regime is BULL, cash otherwise;
> no classifier; band on, stop-loss off.

The stored V1 run does **not** implement that definition cleanly. Three deviations are
visible in the code and in the warehouse:

1. **It does not go to cash outside BULL.** `src/backtest/strategies.py::make_v1` returns
   `None` outside BULL. The engine interprets `None` as “place no orders at the next open,”
   not “liquidate.” Existing positions are therefore held through SIDEWAYS and other
   non-BULL regimes. Stored `test_V1` has **1 flat day out of 685** and a mean invested
   share of **0.9704**; the summary's `exposure` column is 0.99854 because it measures the
   share of days with any position (684/685). It has **zero `EXIT` trades**.
2. **It holds 10 names, not all 50.** The provider slices `available[:slots]`, and `slots`
   is 10. The first day, 2024-01-02, has exactly 10 `ENTRY` rows. The spec's V1 is an
   equal-weight basket of every stock with a bar; the stored V1 is a 10-stock basket.
3. **It has stop-loss trades.** The spec's V1 row says stop-loss off, but stored `test_V1`
   has **15 `STOP_LOSS` trades**, so stop-loss was active in the run.

The spec is internally inconsistent here: §3 of `05_trading_strategy.md` says
stop-loss/cooldown/band are part of V0–V2, while the V1 row in §7 says stop-loss off. The
stored run follows the §3 sentence, not the V1 row. Either way, it does not implement the
V1 row's “cash otherwise” definition.

The stored V1 trade mix is:

| Reason | Trades |
|---|---:|
| ENTRY | 25 |
| REBALANCE | 282 |
| STOP_LOSS | 15 |
| EXIT | 0 |
| **Total** | **322** |

**Consequence:** V1 is a legitimate result for the strategy it actually ran — a 10-name,
regime-gated, almost-always-invested basket with band rebalancing and stop-loss — but it is
**not** a clean “regime filter only, cash otherwise” ablation. Its 0.6738 Sharpe and 10.29%
CAGR cannot be used to conclude that removing the classifier improved the system. The
README/RESULTS claim to that effect is therefore weaker than stated. The frozen V2 result
and the B1/B2/B3 comparisons are unaffected. No code or result was changed while preparing
this report; the issue is documented, not fixed.

### 6.5 Strategy variants and baselines

| ID | Name | What it does | Status |
|---|---|---|---|
| V0 | Classifier only | Applies the BULL rule in every regime; always invested up to the slot limit; no regime exposure cap | Implemented, test result |
| V1 | Regime filter only | Intended: equal weight all stocks in BULL, cash otherwise. Actual: 10 names, no liquidation outside BULL, stop-loss active | Implemented, but not a clean ablation (see §6.4) |
| V2 | Full strategy | Classifier + regime thresholds/exposure + band + stop-loss | Frozen strategy, test result |
| V3 | Per-regime models | Separate classifier per regime | Not implemented |
| V4 | Alternative regime model | GMM/HMM or different k | Not implemented |
| B1 | Buy and hold `^NSEI` | Buy the index once, fractional shares, no costs beyond entry | Implemented |
| B2 | Equal weight, monthly | Hold all 50 stocks equal-weight, rebalance at month ends | Implemented |
| B3 | Trend `sma_20 > sma_50` | Long the index when the 20-day SMA is above the 50-day SMA, else cash | Implemented |
| B4 | Daily cross-sectional permutation null | Within each day, permute the stock probabilities across symbols; same rules; 200 test simulations | Implemented |

B1 has 1 trade row; B2 has 1,076 (50 entries + 1,026 rebalances); B3 has 819 (404 entries,
399 exits, 16 rebalances). V0 has 1,142 (576 entries, 470 exits, 96 stop-losses). V2 has
1,004 (463 entries, 381 exits, 83 rebalances, 77 stop-losses). V1 has 322 as shown above.
These are trade rows, not round trips.

### 6.6 Strategy tuning and selection

The tuning protocol is validation-only (2023). The test period is never touched.

1. **Calibration check.** For each model, compute the share of validation days on which at
   least `ceil(min(slots_grid)/2)` stocks have `prob_up >= min(entry_base_grid)`. These
   four rows are logged as `calibration` and are **not** counted as trials.
2. **Model eligibility.** A model is eligible only if its validation probabilities are not
   degenerate: tied share ≤ 20% and p95−p50 ≥ 0.01. The decision tree fails both tests
   (76.38% tied, p95−p50 = 0) and is excluded from tuning.
3. **Per-model third entry base.** For each eligible model, the grid uses 0.55, 0.60 and a
   third base:
   - logistic regression: validation p80;
   - random forest: validation p80 = **0.504415**;
   - XGBoost: absolute 0.52 if at least 50% of validation days have enough candidates,
     otherwise p80.
4. **Grid.** 3 eligible models × 3 entry bases × 2 slot counts (5, 10) = **18 strategy-grid
   trials**. Each trial runs V2 on validation only and logs Sharpe, CAGR, drawdown, turnover
   and trade count.
5. **Activity filter (R39).** Require at least 30 trade rows and at least 10% mean exposure.
   **10 of 18** configurations pass.
6. **Null filter (R40).** Shortlist the top three by validation Sharpe; for each, run its own
   100-simulation B4 null and require the actual Sharpe to exceed the null's 90th
   percentile. **3 shortlisted, 1 passed.**
7. **Selection.** The passing configuration is random forest, `entry_base = 0.504415`,
   `slots = 10`, with validation Sharpe **1.832750**, **655** trade rows and
   `days_in_position` **0.995918** (the share of validation days with any position).
8. **Disclosure.** Selection filters R39–R42 were introduced after run 1 produced a
   degenerate one-trade winner. Run 1 is preserved and still counted. The frozen config
   records this disclosure.

The frozen config's `demonstrated_edge: true` means **“this configuration passed the
validation activity and null filters.”** It is not a claim that the test period showed an
edge. The test period says the opposite: the Sharpe confidence interval crosses zero, and
the strategy lost to B2 and B3.

### 6.7 Execution and costs

| Parameter | Value |
|---|---:|
| Initial capital | ₹10,00,000 |
| Slippage per side | 0.05% |
| Commission/fee per side | 0.10% |
| All-in cost per side | 0.15% |
| Approximate round-trip cost | 0.30% |
| Risk-free annual rate | 0.0% |
| Trading days per year | 252 |
| Execution price | Next day's open |
| Share granularity | Whole shares (B1: fractional) |
| Cash return | 0% |
| Leverage | None |
| Short selling | None |

The cost model is deliberately simple: fixed slippage plus a fixed commission/fee. It does
not model market impact, bid–ask spread variation, liquidity, taxes, brokerage tiers or
borrow costs. For a 50-stock large-cap universe this is a reasonable first approximation,
but it is not a production execution model.

---

## 7. Backtesting and evaluation methodology

### 7.1 Engine

The backtest engine is `src/backtest/engine.py`. Its daily loop is:

1. **Fill pending targets at the open of day *t*.** Sells execute first, then buys in
   descending `prob_up`. Each fill pays slippage and cost. Buys that would make cash
   negative are skipped.
2. **Mark the portfolio at the close of day *t*.** Write equity, cash, positions value and
   drawdown to `fact_portfolio_daily`.
3. **Decide target weights after the close of day *t*.** The strategy provider receives a
   `DayContext` with the date, regime, signals, indicators, prices, current positions and
   cash.
4. **Store pending targets for the open of day *t+1*.**
5. **Repeat.**

The engine supports `None` from a provider as “place no orders at the next open: clear
pending targets and do not repeat the previous ones.” This is the behavior that makes the
V1 ablation hold existing positions outside BULL; see §6.4.

### 7.2 Run IDs and period convention

| Purpose | Run ID | Period | Persisted |
|---|---|---|---|
| Tuning trials | `grid_validation` / null run IDs | Validation | Trial log only |
| Validation results | `validation_B1`, `validation_B2`, `validation_B3` | 2023 | Persisted; V0/V1/V2 validation variants are not in the committed summary |
| Final test, primary | `backtest_main` | 2024-01-01 to 2026-10-09 | Yes; only run with `write_signals=True` |
| Final test, variants | `test_V0`, `test_V1`, `test_V2_nocost`, `test_B1`, `test_B2`, `test_B3` | Same | Yes |
| Paper trading | `paper_live` | 2026-10-16 onward | Yes |

The test period is **2024-01-01 to 2026-10-09**, 685 trading days. The current warehouse
extends to 2026-10-28, but that extension is **not** part of the frozen test and is not
re-scored.

### 7.3 Metrics

The summary metrics are computed by `src/backtest/metrics.py`:

- **CAGR:** geometric annual growth using calendar days / 365.25.
- **Sharpe:** mean daily excess return / standard deviation × √252, risk-free 0.
- **Sortino:** mean daily excess return / downside deviation × √252.
- **Max drawdown:** maximum peak-to-trough decline, stored as a positive magnitude.
- **Calmar:** CAGR / max drawdown.
- **Turnover:** not populated in `summary_test.csv`; the headline table does not use it.
- **Exposure:** in `summary_test.csv`, the share of days with any position, not mean
  capital invested. This report does not use that column as “mean exposure.”
- **Trade rows:** `n_trades`, the number of individual buy/sell rows.
- **Final equity:** portfolio value at the last test date.

The `RESULTS.md` column labelled “round trips” is actually `n_trades`. This report calls it
“trade rows” to avoid the implication of completed round trips.

### 7.4 Confidence intervals

Confidence intervals are **moving-block bootstrap** of daily returns:

- block length: 10 trading days;
- resamples: 1,000;
- interval: 5th to 95th percentile;
- seed: 42.

The block bootstrap preserves some serial dependence that an i.i.d. bootstrap would
destroy. The intervals are still wide because the test period is only 685 days.

### 7.5 Baselines and null

- **B1** is the index buy-and-hold. It is the “do nothing active” benchmark.
- **B2** is the equal-weight basket. It removes the cap-weighting of the index and is a
  strong simple benchmark for a 50-stock universe.
- **B3** is the 20/50-day trend rule. It is a simple, transparent market-timing benchmark.
- **B4** is a symbol-permutation null. Each day's cross-section of `prob_up` is permuted
  across symbols, so the multiset of scores is preserved but the stock-to-score mapping is
  destroyed; because the permutation is redrawn every day, a stock's assigned score can
  change daily. The strategy is then run on the permuted signals with the same regime rules
  and costs. The test uses 200 simulations.

The B4 null is matched to the real run on capital/weight exposure on validation (null mean
0.7471, median 0.7468, versus real 0.7487) but **not on turnover**. On test, the null executes 2,478–2,735
trade rows against the real run's 1,004, about 2.6× as many, because permuted rankings
reshuffle daily while the real model's autocorrelated probabilities give it stable
holdings. The null therefore pays roughly two and a half times the trading costs. A null
handicapped that way is easier to beat. See §8.7.

**Implementation note:** `src/backtest/baselines.py::permuted_signals` permutes within each
day, despite a docstring sentence about one fixed permutation per simulation. The daily
reshuffle is why the null's turnover is much higher than the real run's.

### 7.6 The one-time test guard

The test period is protected by `reports/results/TEST_RUN_LOCK.json` and
`src/backtest/final_test.py`:

- `run_count: 2`
- `data_end_date: 20261009`
- `config_hash: dba6b13862d1...`
- `frozen_config_hash: 0b8c88c64abf...`
- `git_commit: afefc4b2e5d1...`
- `rerun_reason: "b4_distribution called run_backtest without allow_test_period=True;
  crash fix to complete B4 and the trial-log/manifest entries."`

What happened:

1. The first authorised test run crashed in the B4 distribution because a call to
   `run_backtest` omitted `allow_test_period=True`.
2. A second authorised run completed B4 and the trial-log/manifest entries.
3. The six pre-existing outputs were bit-identical before and after the rerun, recorded in
   `reports/results/PRE_RERUN_CHECKSUMS.txt`.
4. The lock now records `run_count: 2`. This is disclosed, not hidden.

The guard refuses a second test unless `--allow-rerun` is supplied. The project treats that
flag as an emergency, not a convenience. **This report does not re-run the test.**

### 7.7 Leakage checklist

`reports/results/leakage_checklist.md` records 14 numbered checks; the main ones are:

- features use data through the close of *t* only;
- the label is the only `shift(-1)`;
- no back-fill;
- chronological splits;
- scalers/winsorizers fitted on training only;
- K-Means fitted on training only and frozen;
- yearly walk-forward with a 5-day embargo;
- no test data in tuning or model selection;
- test lock and run count;
- paper job refuses dates inside the locked test;
- one strategy code path for backtest and paper;
- dashboard is read-only;
- bot replay is a replay, not a live run.

The checklist is a document, not a proof; the tests in §11 exercise the same invariants.

### 7.8 Trial log

`reports/results/trial_log.csv` has **342 rows**:

| Kind | Rows | Counted as experiment? |
|---|---:|---|
| `strategy_grid` | 18 | Yes |
| `regime_k` | 5 | Yes |
| `model_year` | 4 | Yes |
| `other` | 311 | Yes |
| `calibration` | 4 | No |
| **Total** | **342** | **338 counted** |

The 311 `other` rows include 300 B4 validation null simulations, 7 final-test runs, 3
validation baseline rows and 1 selection-rule revision. Periods: 330 validation, 7 test,
5 train. The frozen config records `total_logged_trials: 331` at freeze time; the final
test added 7, bringing the current count to **338**.

### 7.9 Run manifests

Every significant run writes a JSON manifest under `reports/results/manifests/`:

- `final_test.json` — test period, run IDs, config hash, git commit, package versions;
- `train.json` — walk-forward training;
- `tune.json` — validation grid, selected configuration, trial count;
- `regime_model.json` — k, features, VIX usage, regime rows written;
- `features_indicators.json` — indicator refresh;
- `baselines_validation.json` — validation baseline runs;
- `etl_2026-10-28.json` — latest ETL;
- `paper_paper_live.json` — latest paper run.

Manifests record the seed (42), config hash, data range, git commit and package versions.
This is what makes the results traceable.

---

## 8. Results

### 8.1 Headline test table

Test period: **2024-01-01 to 2026-10-09**, 685 trading days. All figures are after the
stated costs except the `V2_nocost` row. Risk-free rate is 0.0. Confidence intervals are
moving-block bootstrap, 10-day blocks, 1,000 resamples, seed 42.

| Run | Sharpe | 95% CI | CAGR | Max drawdown | Trade rows | Final equity |
|---|---:|---:|---:|---:|---:|---:|
| **V2 frozen strategy (rf)** | **0.4848** | **[−0.4561, 1.5437]** | **5.48%** | **19.32%** | **1,004** | **₹11,59,447** |
| V0 classifier only | 0.4481 | [−0.5046, 1.4946] | 5.96% | 22.37% | 1,142 | ₹11,74,094 |
| V1 regime only (not clean) | 0.6738 | [−0.2337, 1.6914] | 10.29% | 16.37% | 322 | ₹13,11,860 |
| V2 no costs | 0.8949 | [−0.0516, 1.9391] | 11.15% | 15.66% | 1,000 | ₹13,40,368 |
| B1 buy and hold `^NSEI` | 0.1576 | [−0.7746, 1.1081] | 1.20% | 15.59% | 1 | ₹10,33,520 |
| B2 equal weight monthly | 0.7017 | [−0.1866, 1.6856] | 8.82% | 15.06% | 1,076 | ₹12,63,913 |
| B3 trend `sma_20 > sma_50` | 0.6117 | [−0.2256, 1.4971] | 5.02% | 9.75% | 819 | ₹11,45,472 |

Sortino and Calmar for completeness:

| Run | Sortino | Calmar |
|---|---:|---:|
| V2 | 0.4836 | 0.2840 |
| V0 | 0.4449 | 0.2666 |
| V1 | 0.6156 | 0.6287 |
| V2 no costs | 0.9020 | 0.7122 |
| B1 | 0.1531 | 0.0768 |
| B2 | 0.6726 | 0.5856 |
| B3 | 0.5683 | 0.5154 |

The headline is not “V2 won.” It is “V2 beat buy-and-hold on Sharpe and CAGR, lost on
drawdown, lost to equal-weight on all three metrics and to trend on Sharpe and drawdown, and
every confidence interval includes zero.”

### 8.2 V2 versus each benchmark

| Comparison | V2 | Benchmark | Winner |
|---|---:|---:|---|
| Sharpe vs B1 | 0.4848 | 0.1576 | V2 |
| CAGR vs B1 | 5.48% | 1.20% | V2 |
| Max drawdown vs B1 | 19.32% | 15.59% | B1 |
| Sharpe vs B2 | 0.4848 | 0.7017 | B2 |
| CAGR vs B2 | 5.48% | 8.82% | B2 |
| Max drawdown vs B2 | 19.32% | 15.06% | B2 |
| Sharpe vs B3 | 0.4848 | 0.6117 | B3 |
| CAGR vs B3 | 5.48% | 5.02% | V2 (slightly) |
| Max drawdown vs B3 | 19.32% | 9.75% | B3 |
| Sharpe vs V0 | 0.4848 | 0.4481 | V2 (slightly) |
| CAGR vs V0 | 5.48% | 5.96% | V0 (slightly) |
| Max drawdown vs V0 | 19.32% | 22.37% | V2 |
| Sharpe vs no-cost V2 | 0.4848 | 0.8949 | No-cost V2 |
| CAGR vs no-cost V2 | 5.48% | 11.15% | No-cost V2 |
| Max drawdown vs no-cost V2 | 19.32% | 15.66% | No-cost V2 |

The B2 equal-weight basket is the strongest simple benchmark on this test period: higher
Sharpe, higher CAGR and lower drawdown than V2. The B3 trend rule has a much lower drawdown
(9.75% versus 19.32%) and a higher Sharpe. A strategy that is supposed to add value over
simple alternatives does not do so here.

### 8.3 Year by year

| Run | 2024 | 2025 | 2026 (to 2026-10-09) |
|---|---:|---:|---:|
| V2 | +19.39% | +17.94% | **−17.52%** |
| B1 | +8.47% | +9.97% | −13.77% |
| B2 | +19.65% | +12.91% | −6.39% |
| B3 | +16.99% | +4.35% | −6.65% |
| V1 | +13.56% | +19.28% | −3.87% |
| V2 no costs | +27.41% | +21.99% | −13.66% |

V2's full-period result is a 2024–2025 story followed by a bad 2026. The 2026 drawdown is
the 19.32% maximum drawdown in the headline table. V2's 2026 Sharpe is **−1.7241**, versus
+1.3319 in 2024 and +1.7356 in 2025. A strategy that depends on two good years and then
loses 17.5% is not robust.

### 8.4 Validation results

The validation year is 2023. The frozen strategy's validation Sharpe is **1.832750** with
**655** trade rows. The three simple baselines had higher validation Sharpe than V2; B2 and
B3 also had higher CAGR, while B1 had a slightly lower CAGR (19.55% versus 20.70%) and a
lower drawdown (6.09% versus 6.79%):

| Run | Sharpe | 95% CI | CAGR | Max drawdown | Trade rows | Final equity |
|---|---:|---:|---:|---:|---:|---:|
| B1 buy and hold | 1.9195 | [0.0459, 3.9671] | 19.55% | 6.09% | 1 | ₹11,93,002 |
| B2 equal weight monthly | 3.4674 | [1.7097, 5.6420] | 38.89% | 5.50% | 386 | ₹13,83,624 |
| B3 trend | 3.8990 | [2.2588, 5.5670] | 32.00% | 3.12% | 269 | ₹13,15,772 |

V2's validation CAGR was **20.70%** (from `null_diagnostic_reference.csv`), above B1's
19.55% but below B2's 38.89% and B3's 32.00%. V2's validation Sharpe (1.8328) is below all
three. The B1/B2/B3 validation confidence intervals exclude zero; V2's validation interval
is not in the committed `summary_validation.csv`, which contains only B1–B3. This is a
documentation gap: the validation comparison is still clear from the point estimates, but
the V2 interval is not available in that file.

The selection process chose V2's configuration because it passed the activity and null
filters, not because it beat the baselines. It did not beat them on validation Sharpe. This
is consistent with the test result: the baselines are hard to beat, and the classifier adds
little.

### 8.5 Classifier results

The full classifier table is in §5.4.3. The summary:

- Validation: logreg −0.0335 edge, tree +0.000083, rf −0.0190, xgb −0.0256. The tree is
  degenerate (76.38% tied), so its near-zero edge is not meaningful.
- Test: logreg +0.0053, tree −0.0032, rf +0.0044, xgb +0.0094. Test AUCs 0.514–0.520.
- The selected rf is not the best test classifier; xgb has a slightly higher test AUC and
  edge. The rf was selected on validation.
- No model shows a reliable, economically meaningful directional edge.

### 8.6 Cost impact

The no-cost V2 row is the same strategy with slippage and cost set to zero:

| Metric | V2 with costs | V2 no costs | Cost impact |
|---|---:|---:|---:|
| Sharpe | 0.4848 | 0.8949 | −0.4102 |
| CAGR | 5.48% | 11.15% | −5.67 pp |
| Max drawdown | 19.32% | 15.66% | +3.66 pp |
| Trade rows | 1,004 | 1,000 | −4 |
| Final equity | ₹11,59,447 | ₹13,40,368 | −₹1,80,921 |

Costs remove about **0.41 Sharpe** and **5.7 percentage points of CAGR**. The no-cost
confidence interval is [−0.0516, 1.9391]; it still crosses zero, so even the impossible
zero-cost version is not statistically significant. The no-cost result is not a tradable
number; it is a diagnostic showing how much of the gross return is consumed by frictions.

### 8.7 B4 null

The test-period B4 distribution has 200 simulations:

| Statistic | Value |
|---|---:|
| Mean Sharpe | −0.5579 |
| Median Sharpe | −0.5606 |
| 90th percentile | about −0.227 |
| Best Sharpe | 0.1198 |
| Worst Sharpe | −1.2997 |
| Real V2 Sharpe | 0.4848 |
| Null mean CAGR | −7.41% |
| Null mean max drawdown | 27.76% |
| Null trade rows | 2,478–2,735 |
| Real trade rows | 1,004 |

V2's 0.4848 is above **all 200** permutations, whose best was 0.1198. On its face, that is
the strongest positive statement in the results. But the null is not turnover-matched:

- null mean weight exposure on test: 0.7661; real V2: 0.7773;
- validation match: null mean 0.7471 (median 0.7468) versus real 0.7487;
- test trade rows: null 2,478–2,735 versus real 1,004, about **2.6×**.

The null therefore pays roughly two and a half times the trading costs. A null that is
handicapped by higher turnover is easier to beat. The correct conclusion is not “the
strategy has skill”; it is “the strategy beat a cost-handicapped permutation null, and a
turnover-matched null was not run.” This is a known limitation, disclosed in the README and
RESULTS.

### 8.8 Regime breakdown

The test-period index behavior by regime:

| Regime | Days | Mean daily return | Up days |
|---|---:|---:|---:|
| BULL | 495 | +0.0408% | 253 |
| SIDEWAYS | 190 | −0.0161% | 94 |
| BEAR | 0 | — | — |

The training profile (including 27 BEAR days) is in §5.3.4. The test period had no BEAR
days, so the strategy's BEAR exit rule was not exercised out of sample. The test period did
have 190 SIDEWAYS days, but V2's mean invested capital was still 77.7% overall; the
SIDEWAYS 60% cap did not prevent a large drawdown in 2026.

### 8.9 What the results mean

1. **No demonstrated edge.** The Sharpe CI crosses zero; the strategy lost to B2 and B3; the
   2026 return was negative. The correct conclusion is “not proven,” not “profitable.”
2. **The classifier is weak.** Test AUC 0.514–0.520, test edges under 1 percentage point,
   validation edges negative or zero. The strategy cannot be attributed to predictive skill.
3. **Costs are large relative to the edge.** A 0.30% round trip consumes a substantial part
   of a small gross edge. The no-cost result is not achievable.
4. **The B4 null is positive but handicapped.** The turnover mismatch is a serious caveat.
5. **The V1 ablation is not clean.** The stored V1 is a 10-name, almost-always-invested
   basket with regime-gated entries and stop-loss, not “cash otherwise.” Its higher Sharpe
   does not establish that removing the classifier improved the system. See §6.4.
6. **The 2026 failure matters.** A strategy that loses 17.5% in the most recent year, with a
   19.3% drawdown, is not robust enough to deploy.
7. **An alternative explanation remains untested.** The strategy's low-churn holdings may
   capture momentum or exposure effects rather than classifier skill. A turnover-matched
   null, a randomized-regime control and a holding-period attribution would help separate
   these explanations. None was run.

The frozen config's `demonstrated_edge: true` should be read as “passed the validation
selection filters,” not “the test showed an edge.” The test period is the arbiter, and it
says no.

---

## 9. Paper trading and daily operation

### 9.1 What the paper job is

`src/paper/daily_job.py` is the daily operational path. It is **paper trading only**: no
broker, no real money, no order transmission. It uses the same strategy functions as the
backtester, so the paper decisions are the frozen strategy's decisions on stored data, not
a separate hand-written rule set.

The job:

1. reads the latest trading date from the warehouse;
2. ensures prices, indicators, regime and signals exist for that date;
3. loads the current paper portfolio state from `fact_trade` and
   `fact_portfolio_daily`;
4. applies the strategy to produce target weights after the close;
5. writes `pending_target` rows for the next open;
6. on the next run, fills those targets at the open, marks the portfolio at the close,
   writes `fact_trade` and `fact_portfolio_daily`, and writes a daily markdown report;
7. repeats for each unprocessed trading date.

### 9.2 Crash safety and idempotency

The job processes **one date per transaction**. A date is committed when:

- its `fact_portfolio_daily` row exists, together with any `fact_trade` and
  `pending_target` rows that date produced; or
- no `fact_portfolio_daily` row exists, in which case the date is redone.

This is what makes a crash safe. The job can be rerun without duplicating trades or
portfolio rows. It rebuilds holdings by replaying stored trades and takes cash/equity from
the latest `fact_portfolio_daily` row, not from an in-memory position object.
`pending_target` is the source of next-open orders; an empty set for a processed date means
the book is flat and there is nothing to fill.

### 9.3 Guards and start behavior

The paper job has guards:

- **Test-lock guard:** it refuses to process a date at or before the frozen test end
  (2026-10-09). The paper period starts after the locked test.
- **Start flat:** a new `paper_live` run starts flat at the latest trading date. It does not
  invent a historical position.
- **No backfill:** it does not backfill earlier sessions. The first **processed** paper date
  is 2026-10-16, so the paper history starts there; 2026-10-12 through 2026-10-15 were
  before that and are not part of `paper_live`. An earlier 2026-10-09 invocation started
  flat, wrote signals for the lock date, was rolled back under R26, and committed no date;
  it is not part of the processed history.
- **Idempotent dates:** already committed dates are skipped.
- **One code path:** the paper job calls the same `src/trading/` functions as the backtest.

### 9.4 Current paper state

As of 2026-10-28:

| Item | Value |
|---|---|
| Run ID | `paper_live` |
| Processed dates | 8: 2026-10-16, 10-19, 10-21, 10-22, 10-23, 10-26, 10-27, 10-28 |
| Regime on every date | SIDEWAYS |
| Equity | ₹10,00,000 |
| Cash | ₹10,00,000 |
| Positions | 0 |
| Trades | 0 |
| Pending targets | 0 |
| Drawdown | 0.00% |
| Scored names per date | 50 |
| Target weights | All 0.0 |
| Latest max `prob_up` (2026-10-28) | 0.5217796 |
| SIDEWAYS entry threshold | 0.554415 |
| Gap to threshold | about 0.0326 |

The flat decision is a **real decision**, not missing data. On 2026-10-28 the classifier
scored all 50 names, but the best score (0.5218) was below the SIDEWAYS entry threshold
(0.5544). The strategy therefore held cash. The same pattern repeated on all eight paper
dates; the closest date was 2026-10-26, when the maximum was about 0.5449, still below the
threshold.

The latest stored market bar is 2026-10-28:

- Nifty 50 close: **22,698.85**;
- India VIX: **16.245**;
- regime: **SIDEWAYS**.

These are stored end-of-day values, not live quotes.

### 9.5 Daily reports and alerts

The job writes one report per invocation, for the **latest processed date**, to
`reports/daily/YYYY-MM-DD.md` (spec 08 §3 step g). It does not write one report for every
date in a multi-date batch. The report contains:

- the run ID and latest data date;
- the regime and whether it changed;
- equity, cash, drawdown and positions;
- a positions table;
- fills today;
- orders for the next open;
- warnings.

`reports/daily/` currently contains `2026-10-16.md` and `2026-10-28.md`, plus a
`2026-10-09.md` left by the first real invocation that was rolled back under R26. The
2026-10-09 file is not a current processed date and is not part of the eight-date paper
history.

The 2026-10-28 report says:

- equity ₹10,00,000; cash ₹10,00,000; drawdown 0.00%; 0 positions;
- no fills;
- “flat: the strategy decided to hold no positions”;
- no warnings.

The job can alert on drawdown above 10% and on stale data older than 3 days. Neither
condition is active now. The alerts are operational notices, not trading signals.

### 9.6 What the paper state does and does not prove

It proves:

- the daily job runs end to end;
- the warehouse, regime, signal, strategy, pending-target and portfolio tables are
  connected;
- the job is idempotent and crash-safe;
- the dashboard can display a flat decision honestly;
- the paper path uses the same strategy code as the backtest.

It does **not** prove:

- that the strategy is profitable;
- that the frozen test result generalizes;
- that the classifier has an edge;
- that the paper period is a new out-of-sample test.

Eight flat dates are not a track record. They are an operational demonstration.

---

## 10. Dashboard

### 10.1 Six screens

The dashboard is a Streamlit app in `src/dashboard/app.py`. It has six screens:

| Screen | What it shows |
|---|---|
| **Market overview** | Nifty 50 with regime bands, India VIX, current regime, regime changes, stored-data banner |
| **OLAP explorer** | Sector × quarter heatmap, sector drill-down, year/sector filters, momentum table |
| **Today's signals** | Latest stored `paper_live` decision, all scored names, target weights, regime, threshold, flat callout |
| **Backtest results** | Summary table, equity/drawdown charts, confidence intervals, yearly returns, honesty notes |
| **Model diagnostics** | Classifier metrics, confusion matrix, probability histogram, feature importance, tree warning |
| **Bot demo (replay)** | 60 stored test days replayed with fake money, positions, fills and a buy-and-hold comparison |

### 10.2 Read-only, offline, no broker

The dashboard is deliberately constrained:

- every query is a `SELECT`;
- it does not write to the warehouse;
- it does not recompute reported metrics; it reads the result CSVs and warehouse;
- it does not connect to a broker;
- it does not fetch live prices;
- it shows a stale-data banner when the latest market bar is more than three calendar days
  old;
- tests assert that the dashboard is read-only.

The point is to make the dashboard a **viewer of stored evidence**, not a second research
pipeline that could disagree with the results.

### 10.3 Today's signals — the flat-decision fix

The “Today's signals” screen was fixed in commit `8d8fe61` to handle an honest flat state.
The problem: if the latest paper decision had no positions, an earlier implementation could
show an empty or misleading table, or fail to mark the date as processed. The fix:

- uses `fact_portfolio_daily` as the **processed-decision marker**;
- shows all scored names for the latest decision date, including names with zero target
  weight;
- uses **BUY/HOLD only** — no SELL row, because the book is long-only and a zero weight is
  a hold/cash decision, not a short;
- displays a clear flat callout when every target is zero;
- warns when the decision date is older than the latest market bar;
- does not invent a live signal.

On 2026-10-28, the decision date equals the latest market bar, so there is no stale warning.
The screen shows 50 scored names, all target 0.0, and the flat callout.

### 10.4 Bot demo (replay)

The Bot demo is the only screen with a P&L panel. It is a **replay**, not a live run:

- it replays **60 stored test days**;
- it uses a temporary copy of the warehouse and is read-only;
- it runs one replay at a time, with a time cap;
- it has fixed start dates and a cached result;
- it shows fake-money P&L against a buy-and-hold Nifty benchmark;
- it does **not** show Sharpe, CAGR, max drawdown, volatility or win rate;
- it is explicitly labelled as a demonstration, not a performance claim.

The replay exists to show that the frozen strategy's decisions can be reproduced step by
step. The equivalence proof (`reports/results/bot_equivalence.json`) compares the replay
with the stored backtest on three dates (2024-01-01, 2024-02-13, 2024-03-28): 50
probabilities each, 10, 7 and 10 weights respectively, a maximum probability difference of
**3.33e-16**, weights match, and a look-ahead mutation leaves the replay unchanged. That is
a floating-point-level match.

### 10.5 Current dashboard status

When this report was compiled:

- the dashboard was running at `http://127.0.0.1:8501`;
- the health endpoint `/_stcore/health` returned `ok`;
- the latest warehouse date was 2026-10-28;
- the latest paper decision was 2026-10-28;
- the paper book was flat.

The dashboard is not a hosted service. `make dashboard` starts it locally; the demo/hosting
kits are optional and additive.

---

## 11. Testing, reproducibility and code quality

### 11.1 Current test status

| Item | Value |
|---|---|
| Test modules | 26 (28 Python files including `conftest.py` and `__init__.py`) |
| Source files | 52 |
| Tests collected | 565 |
| Last full run | **565 passed**, 0 failed |
| Full-run time | 12m36s |
| Lint | Ruff: all checks passed |
| Collection time | 4.18s |

The README's “503 tests” and “59 minutes” are stale; the current suite is 565 tests and the
last full run took 12m36s. The README's “paper trading has never traded” is also stale: the
paper job has now processed eight dates, all flat.

### 11.2 Test categories

The 26 test modules cover:

- **Indicators:** formulas, warm-up, no look-ahead.
- **Features:** panel joins, winsorization, scaling, label shift.
- **ETL:** extraction, transform, validation, idempotent load, audit log.
- **Schema:** primary keys, foreign keys, date keys, views.
- **OLAP:** query outputs, dimensions, no write side effects.
- **Regime:** K-Means fit/freeze, label naming, smoothing, VIX fallback.
- **Models:** walk-forward splits, embargo, preprocessing, metrics, diagnostics.
- **Leakage:** look-ahead, split leakage, scaler leakage, test-lock.
- **Trading rules:** thresholds, exposure, hysteresis, band, stop-loss, cooldown,
  tie-breaks, no-entry set.
- **Execution:** next-open fills, sells before buys, cash constraint, costs.
- **Backtest engine:** daily loop, portfolio accounting, final equity.
- **Metrics:** CAGR, Sharpe, Sortino, drawdown, Calmar, bootstrap.
- **Baselines:** B1–B4 settings and null permutation.
- **Tuning:** grid, activity filter, null filter, fallback, no test access.
- **Final test:** lock, confirmation, rerun refusal.
- **Paper:** crash safety, idempotency, test-lock guard, flat state, daily report.
- **Dashboard:** read-only queries, six screens, flat-signal view, stale banner.
- **Bot demo:** replay equivalence, no writes, time cap, no performance metrics.
- **CLI:** exit codes, config overrides, manifests.

### 11.3 Leakage and test-lock tests

The test suite exercises the invariants that matter:

- a feature computed with future data changes the test result;
- a label shift in the wrong direction is caught;
- a scaler fitted on validation/test is caught;
- a K-Means refit after training is caught;
- a walk-forward split without embargo is caught;
- a second final-test run is refused;
- a paper date inside the locked test is refused;
- the dashboard has no write path;
- the bot replay has no write path.

The committed `leakage_checklist.md` is the human-readable companion to these tests.

### 11.4 Smoke and clean-clone checks

- `make smoke` runs the whole pipeline on synthetic data (`config/config.smoke.yaml`), with
  no network. It exercises ETL, features, regime, training, backtest, paper and dashboard
  plumbing on a small fixture.
- `make clean-clone-check` creates a fresh clone, installs dependencies and runs the fast
  suite. The README records a historical result of **475 passed, 35 deselected, 0 failed**
  in 32m42s; that count predates the latest dashboard/paper tests. The current full suite
  is 565.
- `make check` runs Ruff plus the fast tests and must pass before every commit.

### 11.5 Reproducibility

- **Seed:** 42 (`project.seed`), used for model randomness, bootstrap and null permutations.
- **Config hash:** recorded in every manifest.
- **Git commit:** recorded in every manifest.
- **Frozen config hash:** recorded in the test lock.
- **Run manifests:** JSON files with data range, package versions, seed, config hash and
  command line.
- **Model files:** 16 walk-forward model files plus the regime model, stored in `models/`.
- **Test lock:** `TEST_RUN_LOCK.json` with `run_count: 2`, `data_end_date: 20261009`.
- **Environment:** the recorded test manifest used Python 3.14.8, NumPy 2.5.3, pandas 3.0.6,
  scikit-learn 1.9.1 and XGBoost 3.4.1. The README states Python 3.10+; a fresh environment
  may resolve slightly different patch versions. `requirements.lock` pins 70 packages.

The combination of config hash, git commit, seed, manifests and model files is what makes
the result auditable. It does not make the result statistically significant; it makes it
reproducible.

### 11.6 What is not tested

- A full real-data end-to-end test on a fresh clone (the README documents this as deferred;
  the smoke test uses synthetic data).
- A live broker integration (out of scope).
- A production scheduler (kits exist, but nothing is installed by default).
- Market impact, liquidity and taxes.
- A turnover-matched null.
- A point-in-time universe.
- V3/V4 and association rules.

---

## 12. Project structure

```text
regimetrader_pack/
├── README.md                     # setup, commands, headline results, limitations
├── report.md                     # this report
├── AGENTS.md                     # build rules and non-negotiables
├── Makefile                      # setup, check, test, smoke, pipeline, dashboard
├── pyproject.toml                # Ruff, pytest, package metadata
├── requirements.txt              # direct dependencies
├── requirements.lock             # pinned environment (70 packages)
├── config/
│   ├── config.yaml               # main configuration
│   ├── config.smoke.yaml         # synthetic-data configuration
│   ├── frozen_config.yaml        # frozen model / entry base / slots
│   ├── frozen_config.smoke.yaml  # synthetic frozen config
│   ├── nifty50_symbols.csv       # current 50-stock universe
│   └── nifty50_symbols.FORMAT_EXAMPLE.csv
├── sql/
│   ├── schema.sql                # canonical DDL
│   └── olap/                     # named OLAP queries
├── src/
│   ├── common/                   # config, DB, dates, logging, manifests
│   ├── etl/                      # extract, transform, load, quality, synthetic
│   ├── features/                 # indicators, panel, builder
│   ├── olap/                     # OLAP runner
│   ├── regime/                   # K-Means regime model
│   ├── models/                   # classifiers, preprocessing, metrics, diagnostics
│   ├── trading/                  # rules and execution (shared by backtest/paper)
│   ├── backtest/                 # engine, metrics, strategies, baselines, tune, final test
│   ├── paper/                    # daily job, store, report, alerts
│   └── dashboard/                # Streamlit app, data access, UI, bot replay
├── tests/                        # 26 test modules (28 .py files), 565 tests
├── models/                       # 16 walk-forward model files + regime model
├── reports/
│   ├── results/                  # summary CSVs, trial log, manifests, lock, checklist
│   ├── results/olap/             # OLAP outputs
│   ├── figures/                  # regime_bands.png, regime_k_selection.png
│   └── daily/                    # daily paper reports
├── docs/
│   ├── spec/                     # 02–08 specs, DECISIONS, TESTS
│   ├── build/                    # BUILD_PLAN, PROGRESS
│   └── human/                    # problem, viva, timeline, report notes
├── deploy/                       # cron, GitHub Actions, VPS kit, host kit
├── scripts/                      # helper scripts
├── data/
│   ├── raw/                      # raw extracts
│   ├── external/                 # external files
│   ├── processed/                # processed panels
│   └── warehouse/regimetrader.db # SQLite warehouse
└── logs/                         # run logs
```

The repository has **52 Python source files** and **28 Python files under `tests/`** (26
test modules plus `conftest.py` and `__init__.py`). The main entry points are CLI modules
with `main(argv)` functions, exposed through the Makefile.

---

## 13. How to run the project

### 13.1 Setup and checks

```bash
make setup        # create .venv and install dependencies
make check        # Ruff + fast tests; must pass before every commit
make test         # full test suite (565 tests)
make smoke        # whole pipeline on synthetic data, no network
```

On Windows, the Makefile uses `.venv/Scripts/python.exe`; on POSIX it uses
`.venv/bin/python`. The commands are otherwise the same.

### 13.2 Real-data pipeline

```bash
make etl-full     # full refresh from yfinance/NSE fallback
make features     # recompute indicators into fact_indicator
make regime       # fit/freeze the K-Means regime model
make train        # walk-forward classifiers -> fact_signal + models/
make backtest     # tune on validation; validation baselines and variants
make final-test   # one-time guarded test evaluation
make paper        # daily paper job
make dashboard    # Streamlit dashboard
```

**Do not run `make train`, `make backtest` or `make final-test` on the current tree unless
you intend to regenerate models or frozen results.** The user/authorisation rules for this
report explicitly prohibit it. The frozen test has already been run twice (one crash plus
one authorised rerun) and is locked. Use a fresh clone and explicit authorisation for any
new experiment. For a paper refresh without re-downloading, use the job's `--no-etl` flag.

### 13.3 Paper trading

```bash
.venv/Scripts/python.exe -m src.paper.daily_job --no-etl --to 20261028
```

The job will process any unprocessed trading dates up to the requested date. It refuses
dates inside the locked test. The current paper run has processed 2026-10-16 through
2026-10-28 and is flat.

### 13.4 Dashboard

```bash
make dashboard
# open http://127.0.0.1:8501
```

The dashboard is read-only and offline. It reads the warehouse and the result CSVs. It
does not connect to a broker or a live price feed.

### 13.5 Demo and deployment

```bash
make demo-check   # verify tunnel prerequisites; launches nothing
make demo         # dashboard + Cloudflare quick tunnel
```

The demo URL is random and changes on every run. The tunnel is additive; the local
dashboard is the primary path. `deploy/` contains optional cron, GitHub Actions, VPS and
laptop-host kits. Nothing is installed by default.

### 13.6 Clean clone

```bash
make clean-clone-check
```

This creates a fresh clone, installs dependencies and runs the fast suite. It is the closest
thing to a “does this work from scratch?” check. The README's recorded clean-clone result
(475 passed, 35 deselected) predates the latest tests; the current full suite is 565.

---

## 14. Limitations, threats to validity and open issues

### 14.1 Statistical and economic limitations

1. **No demonstrated edge.** Every headline confidence interval crosses zero. The V2
   Sharpe is 0.4848 with CI [−0.4561, 1.5437]. A point estimate above zero is not evidence
   of skill.
2. **The classifier is weak.** Test AUC 0.514–0.520 and edges under 1 percentage point;
   validation edges are negative or zero. The strategy's result cannot be attributed to
   predictive skill.
3. **The baselines are strong.** B2 equal weight beat V2 on Sharpe, CAGR and drawdown; B3
   trend beat V2 on Sharpe and drawdown. A strategy that does not beat simple alternatives
   is not adding demonstrated value.
4. **2026 is bad.** V2 returned −17.52% in 2026, with a 19.32% max drawdown and a −1.7241
   Sharpe. The full-period result depends on 2024–2025.
5. **Costs dominate a small edge.** A 0.30% round trip removes about 0.41 Sharpe and 5.7
   percentage points of CAGR. The no-cost result is not achievable.
6. **The B4 null is turnover-handicapped.** The null trades about 2.6× as much as V2, so
   “beat every permutation” overstates the evidence. A turnover-matched null was not run.
7. **The V1 ablation is not clean.** The stored V1 holds 10 names, stays invested outside
   BULL and uses stop-loss, contrary to the spec. Its higher Sharpe does not establish that
   removing the classifier improved the system. See §6.4.
8. **The alternative explanation is untested.** Low-churn holdings may capture momentum or
   exposure effects rather than classifier skill. A holding-period attribution, randomized
   regime control or turnover-matched null would help.

### 14.2 Data limitations

1. **Survivorship bias.** The universe is the current Nifty 50 list, not point-in-time
   membership. Historical returns are measured on today's survivors.
2. **Short histories.** Seven stocks start after 2015-01-02; six are under 10 years as of
   the test end (INDIGO.NS starts in November 2015 and is the exception).
3. **One market, one period.** The study covers Indian large caps from 2015 to 2026. It does
   not test other markets, small caps, futures, options or a different macro regime.
4. **Few BEAR days.** The BEAR cluster has only 27 training days, dominated by the COVID
   crash, and **zero test days**. The BEAR exit rule is not tested out of sample.
5. **No HIGH_VOL regime.** k=3 produces no HIGH_VOL days; the 30% exposure rule is
   unexercised.
6. **K-Means limitations.** Hard labels, a single frozen fit, no uncertainty, and a 5-day
   trailing-mode smoothing. The regime model does not adapt after 2022.
7. **Data source quality.** yfinance is unofficial; gaps, revisions and bad prints are
   possible. The ETL flags/rejects some but cannot guarantee vendor quality.
8. **Adjusted prices and corporate actions.** The execution model is a simplification.

### 14.3 Execution and operational limitations

1. **Simplified costs.** Fixed slippage and a fixed commission/fee; no market impact,
   liquidity, taxes, brokerage tiers or borrow costs.
2. **Next-open approximation.** Orders fill at the next open with a fixed slippage; real
   opens can gap, be partial or be rejected.
3. **No broker.** Paper trading only; no real money, no order transmission, no SEBI
   compliance, no reconciliation.
4. **No live feed.** The dashboard and paper job use stored end-of-day data.
5. **Single-writer SQLite.** Fine for a laptop; not a multi-user warehouse.
6. **No production scheduler.** Cron/GitHub Actions/systemd/Windows Task Scheduler examples
   exist, but nothing is installed by default.

### 14.4 Selection and multiplicity

- The trial log counts **338 experiments** (342 rows minus 4 calibration rows), including
  300 B4 validation null simulations, 18 strategy-grid trials, 5 regime-k trials, 4
  model-year trials, 3 validation baselines, 1 selection-rule revision and 7 final-test
  runs.
- The test period was evaluated once, with one crash and one authorised rerun. The lock
  records `run_count: 2`; six pre-existing outputs were bit-identical.
- Selection filters R39–R42 were introduced after run 1 produced a degenerate one-trade
  winner. Run 1 is preserved and counted. This is disclosed, but it is still a form of
  researcher degrees of freedom.
- The frozen config's `demonstrated_edge: true` means “passed validation filters,” not
  “statistically significant test result.”

### 14.5 Metadata and documentation caveats

These do not change the frozen V2 result, but they matter for anyone auditing the files:

| Caveat | Evidence | Effect |
|---|---|---|
| `dim_model.trained_through` is `2022-12-31` for all 16 rows | Current warehouse; `leakage_checklist.md` | Actual walk-forward cutoffs are later for wf2024+; the training code enforces the cutoff by assertion, so this is a metadata issue, not a leakage issue |
| `classifier_metrics.csv` test rows use `year=2024` | File | The test window is 2024–2026; the year label is a placeholder, not the actual row year |
| `regime_counts.csv` is a snapshot through 2026-10-09 | File | Current DB has 8 more SIDEWAYS days (970 versus 962) |
| `summary_test.csv` `exposure` is the share of days with any position | File; all runs 0.99854 | It is not mean capital invested; actual mean invested V2 is 77.7% |
| `RESULTS.md` “round trips” column is `n_trades` | `summary_test.csv` | It counts individual trade rows, not completed round trips |
| README says 503 tests / 59 minutes / paper never traded | README | Current suite is 565 / 12m36s; paper has 8 flat dates |
| OLAP `08_regime_profile.csv` and `09_classifier_by_regime.csv` are empty | `reports/results/olap/` | Regime profile is in `regime_profile.csv`; dashboard reads the DB |
| Only two figures are committed | `reports/figures/` | `regime_bands.png` and `regime_k_selection.png`; other spec-listed static figures are absent |
| `summary_validation.csv` has only B1–B3 | File | V0/V1 validation rows are absent; V2 validation Sharpe is in the frozen config |
| Paper `fact_signal` rows are stored under `rf` `wf2023` | `fact_signal` for 2026-10-28 uses model_key 9 (`rf`, `wf2023`); `src/paper/daily_job.py::_model_key` selects the first row by model name | The prediction itself uses `rf_wf2026.joblib`; `_probabilities` reads by model name. This is a metadata/labeling issue, not a different model decision |
| V1 ablation is not spec-compliant | `src/backtest/strategies.py`, `fact_trade`, `fact_portfolio_daily` | See §6.4; the “classifier hurts” conclusion is not established by a clean ablation |

### 14.6 What was not done

- Association-rule mining (Apriori) — a documented stretch goal.
- V3 per-regime models and V4 alternative regime model.
- A turnover-matched B4 null.
- A point-in-time historical universe.
- A full real-data end-to-end test on a fresh clone (deferred; smoke uses synthetic data).
- A live broker sandbox or real-money execution.
- An installed production scheduler or high-availability deployment.
- A per-regime classifier or a calibrated probability model.
- A transaction-level tax/liquidity model.
- A direct test of the “momentum/holding behavior” alternative explanation.

### 14.7 Future work

If the project were continued, the highest-value next steps would be:

1. **Fix the V1 ablation** to match the spec: equal weight all stocks in BULL, liquidate to
   cash outside BULL, stop-loss off. Re-run it on validation and smoke only. Do **not**
   re-score the locked test; a new independent test period would be needed for a new claim.
2. **Run a turnover-matched B4 null** so the permutation comparison is not handicapped by
   higher trading costs.
3. **Build a point-in-time universe** to remove survivorship bias.
4. **Test the momentum/exposure explanation** with holding-period attribution, randomized
   regimes and a low-turnover control.
5. **Improve the classifier** with calibration, probability shrinkage, per-regime models or
   a better feature set; evaluate on validation, not test.
6. **Collect more BEAR/HIGH_VOL data** or use a regime model with uncertainty (GMM/HMM)
   rather than hard K-Means labels.
7. **Model execution better**: variable spreads, market impact, liquidity, taxes and
   partial fills.
8. **Run the paper job for longer** and compare its decisions with the backtest replay.
9. **Fix the metadata/documentation caveats** in §14.5.
10. **Add the missing static figures** if the report/viva requires them.

---

## 15. Conclusion

RegimeTrader is a complete, tested, end-to-end data-warehousing and data-mining project. It
builds a 52-instrument SQLite star schema, an idempotent ETL pipeline, a leakage-safe
feature panel, a K-Means regime model, four walk-forward classifiers, a long-only strategy,
a cost-aware backtester, a crash-safe paper job and a six-screen read-only dashboard. The
test suite has 565 passing tests, Ruff is clean, and the results are traceable through
manifests, a trial log, a leakage checklist and a one-time test lock.

The market result is negative. The frozen strategy's Sharpe is 0.4848 with a 95% confidence
interval of [−0.4561, 1.5437]; it had a lower Sharpe than a plain equal-weight basket and a
20/50-day trend rule, a lower CAGR than the equal-weight basket, and it lost 17.52% in 2026.
The classifier shows no reliable edge (test AUC
0.514–0.520). Costs remove about 0.41 Sharpe and 5.7 percentage points of CAGR. The B4 null
is positive but turnover-handicapped. The V1 ablation, which the README/RESULTS use to argue
that removing the classifier improved the system, is not a clean regime-only test: it holds
10 names, stays invested outside BULL and uses stop-loss.

The honest conclusion is therefore:

> **The engineering works; the strategy does not demonstrate a reliable edge.** The project
> is valuable as a reproducible negative result and as a template for honest evaluation, not
> as a deployable trading system.

The current paper state — eight flat dates through 2026-10-28, ₹10,00,000 cash, zero
positions — is an operational demonstration, not performance evidence. The frozen test ends
on 2026-10-09 and was not re-scored for this report. No code, model, frozen result or test
lock was changed while preparing it.

**Academic prototype. Not investment advice. No real money. No broker integration.**

---

## Appendix A. Key configuration

| Setting | Value |
|---|---|
| `project.seed` | 42 |
| `project.timezone` | Asia/Kolkata |
| `splits.train_end` | 2022-12-31 |
| `splits.validation_start` | 2023-01-01 |
| `splits.validation_end` | 2023-12-31 |
| `splits.test_start` | 2024-01-01 |
| `splits.embargo_days` | 5 |
| `regime.k` | 3 |
| `regime.k_range` | 2–6 |
| `regime.use_vix` | `auto` (resolved true) |
| `regime.features` | `mkt_return_20d`, `mkt_vol_20d`, `india_vix`, `mkt_drawdown` |
| `regime.smoothing` | trailing mode, window 5 |
| `regime.refit` | none |
| `strategy.selected_model` | `rf` |
| `strategy.slots` | 10 |
| `strategy.entry_base` | 0.504415 |
| `strategy.entry_offsets` | BULL 0.0, SIDEWAYS 0.05, HIGH_VOL 0.10 |
| `strategy.exposure` | BULL 1.0, SIDEWAYS 0.6, HIGH_VOL 0.3, BEAR 0.0 |
| `strategy.exit_prob` | 0.50 |
| `strategy.rebalance_band` | 0.25 |
| `strategy.stop_loss` | enabled, 7%, cooldown 5 days |
| `backtest.initial_capital` | 1,000,000 |
| `backtest.slippage` | 0.0005 per side |
| `backtest.cost` | 0.001 per side |
| `backtest.risk_free_annual` | 0.0 |
| `backtest.trading_days` | 252 |
| `backtest.bootstrap` | block 10, 1,000 resamples, 5th/95th |
| `backtest.random_baseline_sims` | 200 |
| `tuning.models` | logreg, tree, rf, xgb |
| `tuning.entry_base_grid` | 0.55, 0.60 |
| `tuning.slots_grid` | 5, 10 |
| `tuning.null_sims` | 100 |
| `tuning.min_round_trips` | 30 |
| `tuning.min_exposure` | 0.10 |
| `paper.run_id` | `paper_live` |
| `paper.alert_drawdown` | 0.10 |
| `paper.stale_banner_days` | 3 |
| `features.warmup_rows` | 50 |
| `features.winsor_lower_pct` | 1 |
| `features.winsor_upper_pct` | 99 |

The frozen config is `config/frozen_config.yaml`. It may override only
`strategy.selected_model`, `strategy.entry_base` and `strategy.slots`. The test lock hashes
both the main config and the frozen config.

## Appendix B. Current database snapshot

| Table | Rows | Date range / note |
|---|---:|---|
| `dim_date` | 4,319 | 2015-01-01 to 2026-10-28 |
| `dim_stock` | 52 | 50 equities + `^NSEI` + `^INDIAVIX` |
| `dim_regime` | 4 | BULL, SIDEWAYS, BEAR, HIGH_VOL |
| `dim_model` | 16 | 4 model families × 4 walk-forward years |
| `fact_price` | 144,105 | 2015-01-02 to 2026-10-28; 2,910 dates |
| `fact_indicator` | 141,195 | 50 equities + index; VIX has no indicator rows |
| `fact_market_regime` | 2,886 | 2015-02-02 to 2026-10-28 |
| `fact_signal` | 185,376 | 2023-01-02 to 2026-10-28 |
| `fact_trade` | 5,364 | 7 test runs; paper 0 |
| `fact_portfolio_daily` | 4,803 | 7 test runs + 8 paper days |
| `pending_target` | 0 | flat paper book |
| `etl_run_log` | 6 | latest 2026-10-28 |
| `stg_price_raw` | 624 | latest extract batch |

Latest market values: Nifty 50 **22,698.85**; India VIX **16.245**; regime **SIDEWAYS**;
paper equity/cash **₹10,00,000**; positions **0**; trades **0**.

## Appendix C. Evidence map

| Question | Where to look |
|---|---|
| Headline test numbers | `reports/results/summary_test.csv`, `reports/results/RESULTS.md` |
| Year-by-year returns | `reports/results/yearly_test.csv`, `yearly_test_all_runs_DERIVED.csv` |
| Classifier metrics | `reports/results/classifier_metrics.csv` |
| Regime counts/profile | `reports/results/regime_counts.csv`, `regime_profile.csv`, `regime_breakdown.csv` |
| Trial count | `reports/results/trial_log.csv`; 342 rows, 338 counted |
| Test lock | `reports/results/TEST_RUN_LOCK.json`, `test_run_log.jsonl` |
| Rerun checksums | `reports/results/PRE_RERUN_CHECKSUMS.txt` |
| Run manifests | `reports/results/manifests/` |
| Leakage checklist | `reports/results/leakage_checklist.md` |
| Bot equivalence | `reports/results/bot_equivalence.json` |
| Frozen configuration | `config/frozen_config.yaml` |
| Current warehouse | `data/warehouse/regimetrader.db` |
| Paper daily reports | `reports/daily/2026-10-28.md` |
| Dashboard | `src/dashboard/app.py`; `http://127.0.0.1:8501` |
| Specifications | `docs/spec/02_system_design.md` … `08_runbook.md`, `DECISIONS.md`, `TESTS.md` |
| Build history | `docs/build/BUILD_PLAN.md`, `docs/build/PROGRESS.md` |

## Appendix D. Glossary

| Term | Meaning |
|---|---|
| **Star schema** | A fact table surrounded by descriptive dimension tables |
| **Grain** | What one row in a fact table represents |
| **ETL** | Extract, transform, load |
| **OLAP** | Online analytical processing: roll-up, drill-down, slice, dice, pivot |
| **Regime** | A market state label (BULL, SIDEWAYS, BEAR, HIGH_VOL) |
| **K-Means** | Unsupervised clustering that assigns each day to the nearest centroid |
| **Walk-forward** | Train on the past, predict the next year, roll forward |
| **Embargo** | A gap between training and prediction to prevent label overlap |
| **Look-ahead** | Using information that would not have been available at decision time |
| **Sharpe ratio** | Mean excess return divided by return volatility, annualized |
| **CAGR** | Compound annual growth rate |
| **Max drawdown** | Largest peak-to-trough equity decline |
| **Bootstrap CI** | Resampled confidence interval; here a moving-block bootstrap |
| **Null distribution** | Outcomes under a deliberately uninformative randomized strategy |
| **Ablation** | Removing a component to see whether it matters |
| **Paper trading** | Simulated trading with no real money |
| **Test lock** | A guard that prevents repeated evaluation of the test period |

## Appendix E. Report provenance

- **Report date:** 2026-10-28.
- **Repository:** `C:\Users\Admin\Projects\DWM\regimetrader_pack`.
- **Branch:** `main`; HEAD `83409e6`; `origin/main` matches; tracked tree clean;
  `report.md` is the only new untracked file.
- **Recent commits:** `83409e6` paper refresh through 2026-10-28; `8d8fe61` dashboard
  flat-decision fix; `8fa40f1` ETL/indicator refresh to 2026-10-28; `ffbe27e` dashboard
  revamp; `43111e4` six-screen docs and bot replay; `477916a` run manifests from the
  authorised local test run; `afefc4b` B4 crash fix.
- **Test status at compilation:** 565 collected, last full run 565 passed in 12m36s, Ruff
  clean.
- **No new training, tuning, backtest or final test was run for this report.**
- **No code, model, frozen result, test lock or strategy logic was changed.**
- **Current paper state:** 8 flat dates through 2026-10-28; operational only.
- **Current dashboard health:** `ok` at `http://127.0.0.1:8501`.

**Academic prototype. Not investment advice. No real money. No broker integration.**

