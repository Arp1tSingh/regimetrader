# RegimeTrader

**A regime-aware Nifty 50 direction predictor, SQLite star-schema data warehouse, cost-aware backtester, crash-safe paper-trading engine, and read-only analytical console.**

> **Disclaimer:** Academic Data Warehousing and Mining (DWM) prototype. **Not investment advice. No real money. No live broker integration.**

| Attribute | Details |
| :--- | :--- |
| **Repository** | [`https://github.com/Arp1tSingh/regimetrader`](https://github.com/Arp1tSingh/regimetrader) |
| **Universe** | 52 Instruments: Current Nifty 50 Equities + Nifty 50 Index (`^NSEI`) + India VIX (`^INDIAVIX`) |
| **Warehouse Coverage** | `2015-01-02` to `2026-10-28` (2,910 trading dates; 144,105 daily OHLCV rows) |
| **Chronological Splits** | **Train:** $\le$ `2022-12-31` \| **Validation:** `2023-01-01` – `2023-12-31` \| **Frozen Test:** `2024-01-01` – `2026-10-09` (685 trading days) |
| **Paper Trading (`paper_live`)** | 8 processed dates (`2026-10-16` to `2026-10-28`); currently flat (₹10,00,000 cash, 0 positions) |
| **Verification & Quality** | **565 passing tests** across 26 test modules (12m36s full run); Ruff clean; 338 logged experimental trials |

---

## 📖 Table of Contents

1. [Overview & Core Philosophy](#-1-overview--core-philosophy)
2. [Honest Findings Summary (TL;DR)](#-2-honest-findings-summary-tldr)
3. [System Architecture & Data Flow](#-3-system-architecture--data-flow)
4. [Data Warehouse, ETL & OLAP](#-4-data-warehouse-etl--olap)
5. [Data Mining: Features, Regimes & Classifiers](#-5-data-mining-features-regimes--classifiers)
6. [Trading Strategy & Execution Engine](#-6-trading-strategy--execution-engine)
7. [Frozen Test Results & Benchmarks](#-7-frozen-test-results--benchmarks)
8. [Paper Trading & Crash-Safe Operations](#-8-paper-trading--crash-safe-operations)
9. [Read-Only Streamlit Dashboard](#-9-read-only-streamlit-dashboard)
10. [Engineering Rigor, Testing & Leakage Guards](#-10-engineering-rigor-testing--leakage-guards)
11. [Repository Structure](#-11-repository-structure)
12. [Setup, CLI Usage & Runbook](#-12-setup-cli-usage--runbook)
13. [Known Limitations, Caveats & Audit Notes](#-13-known-limitations-caveats--audit-notes)

---

## 🎯 1. Overview & Core Philosophy

Retail desks and systematic researchers face two coupled decisions at the end of every trading session:
1. **Cross-sectional selection:** *Which stocks are most likely to rise tomorrow?*
2. **Macro risk allocation:** *How much gross exposure should the portfolio carry given current market turbulence?*

Most toy quantitative bots fail because they apply a static classifier across all market environments, ignore transaction costs, leak future information across splits, or repeatedly tune parameters on out-of-sample data until a fluke backtest looks profitable.

**RegimeTrader** was built to solve the **engineering and methodological side** of this problem on a single laptop:
- **Single Source of Truth:** All raw bars, indicators, regimes, model probabilities, trades, and daily marks live in a relational SQLite star schema with foreign-key enforcement (`PRAGMA foreign_keys = ON`) and integer `YYYYMMDD` date keys.
- **Strict Chronological Discipline:** Preprocessing objects (1st/99th percentile winsorizers, standard scalers) and the unsupervised K-Means regime model are fitted **strictly on training data ($\le$ 2022-12-31)** and frozen thereafter.
- **Walk-Forward Direction Classifiers:** Pooled classifiers (`logreg`, `tree`, `rf`, `xgb`) are retrained annually (`wf2023` through `wf2026`) using an expanding training window with a **5-calendar-day embargo** to prevent label overlap.
- **Shared Execution Code Path:** Backtesting (`src/backtest/`) and daily paper trading (`src/paper/`) invoke the exact same signal-to-weight and order-planning functions in `src/trading/`.
- **One-Time Test Lock:** The out-of-sample test period (`2024-01-01` to `2026-10-09`) is guarded by `TEST_RUN_LOCK.json`, cryptographic config hashes, and manifest auditing.

---

## ⚠️ 2. Honest Findings Summary (TL;DR)

> **The engineering pipeline is reproducible, leakage-safe, and verified by 565 tests—but the trading strategy does NOT demonstrate a statistically reliable market edge.**

Rather than retrofitting parameters to manufacture a winning backtest, RegimeTrader reports its negative out-of-sample findings transparently:

1. **No Statistically Significant Alpha:** On the locked 685-day test period (`2024-01-01` to `2026-10-09`), the frozen **V2 strategy** (`rf`, `entry_base = 0.504415`, `slots = 10`) achieved a **Sharpe ratio of 0.4848** (`95% block-bootstrap CI: [-0.4561, 1.5437]`), **CAGR of 5.48%**, and **Max Drawdown of 19.32%** across **1,004 trade rows** (ending equity **₹11,59,447** on ₹10,00,000 initial capital). Every 95% confidence interval crosses zero.
2. **Underperformed Simple Baselines:** While V2 beat index buy-and-hold (**B1** `^NSEI`: Sharpe `0.1576`, CAGR `1.20%`) on Sharpe and CAGR, it suffered a worse drawdown (`19.32%` vs `15.59%`) and **had a lower Sharpe** than both a monthly equal-weight 50-stock basket (**B2**: Sharpe `0.7017`, CAGR `8.82%`, Max DD `15.06%`) and a simple 20/50-day SMA index trend rule (**B3**: Sharpe `0.6117`, CAGR `5.02%`, Max DD `9.75%`); B2 also beat it on CAGR, and B3 had a much smaller drawdown.
3. **2026 Regime Breakdown:** V2 delivered strong returns in 2024 (`+19.39%`) and 2025 (`+17.94%`), but suffered a **`-17.52%` loss in 2026** (Sharpe `-1.7241`), whereas B2 and B3 limited their 2026 losses to `-6.39%` and `-6.65%` respectively.
4. **Weak Directional Classifiers:** Across 34,200 test stock-days, classifier ROC-AUCs ranged between **0.5140 and 0.5201**—barely above a coin flip. Validation accuracy edges over the majority class were negative for `logreg`, `rf`, and `xgb`.
5. **Transaction Cost Drag:** Round-trip execution frictions (`0.05%` slippage + `0.10%` fee per side = `0.30%` round-trip) consumed **0.4102 Sharpe** and **5.67 percentage points of CAGR** (zero-cost V2 achieved Sharpe `0.8949` and CAGR `11.15%`, though its CI `[-0.0516, 1.9391]` still crosses zero).
6. **Important Caveat on B4 Null & V1 Ablation:**
   - **B4 Permutation Null Handicap:** V2's Sharpe (`0.4848`) beat all 200 daily symbol-permutation null runs (best null Sharpe `0.1198`, mean `-0.5579`), *but* the null reshuffled rankings daily, incurring **2,478–2,735 trade rows (~2.6× V2's turnover)** and paying 2.6× the transaction costs.
   - **V1 Ablation Deviation:** The stored `test_V1` run (Sharpe `0.6738`, CAGR `10.29%`) is **not** a clean implementation of the spec's *"equal-weight all 50 stocks in BULL, cash otherwise, no stop-loss"* ablation. Due to `make_v1` returning `None` outside BULL (which the engine interprets as "place no new orders" rather than liquidate) and slicing `available[:slots]`, `test_V1` held 10 stocks, stayed invested on 684 of 685 test days, and executed 15 `STOP_LOSS` trades. Thus, `test_V1` cannot be used to claim that removing the classifier improved performance.

---

## 🏗️ 3. System Architecture & Data Flow

```text
SOURCE LAYER       yfinance API          NSE CSV fallback        Synthetic generator (tests/smoke)
                         \                      |                       /
INTEGRATION        Extract  --->  Transform & Quality Checks  --->  Idempotent Upsert Load
(src/etl)                                       |
WAREHOUSE LAYER    SQLite Star Schema (data/warehouse/regimetrader.db — Single Source of Truth)
(sql/schema.sql)   ├── Dimensions: dim_date, dim_stock, dim_regime, dim_model
                   ├── Facts:      fact_price, fact_indicator, fact_market_regime,
                   │               fact_signal, fact_trade, fact_portfolio_daily, pending_target
                   └── Audit/Stg:  etl_run_log, stg_price_raw
                         |               |                 |                    |
ANALYTICS LAYER    OLAP Runner     Feature Builder   K-Means Regimes     Walk-Forward Classifiers
                   (src/olap)      (src/features)    (src/regime)        (src/models)
                                         \                 |                   /
APPLICATION LAYER                  Shared Trading Rules & Order Execution (src/trading)
                                         /                 |                   \
                             Backtest Engine      Crash-Safe Paper Job    Streamlit Dashboard
                             (src/backtest)          (src/paper)            (src/dashboard)
```

### Daily Decision & Next-Open Execution Timeline
To eliminate look-ahead bias, decision generation and order execution are strictly separated across trading sessions:
1. **After Market Close of Day $t$:**
   - Extract and validate OHLCV bars for day $t$; recompute full-history indicators.
   - Predict market regime for day $t$ (`BULL`, `SIDEWAYS`, `BEAR`, `HIGH_VOL`) using the frozen K-Means model and apply a **trailing 5-day mode filter**.
   - Score all eligible equities using the active walk-forward classifier to obtain `prob_up` for day $t+1$.
   - Compute target portfolio weights based on regime exposure caps, entry/exit thresholds, hysteresis, and the 25% rebalance band.
   - Persist target weights into `pending_target` for execution at the open of day $t+1$.
2. **At Market Open of Day $t+1$:**
   - Execute pending sells first, followed by buys ordered by descending `prob_up`.
   - Apply `0.05%` slippage and `0.10%` commission/fee per side; enforce whole-share quantities and non-negative cash constraints.
3. **At Market Close of Day $t+1$:**
   - Mark holdings to market close, record realized trades in `fact_trade`, and commit the daily portfolio snapshot to `fact_portfolio_daily`.

---

## 🗄️ 4. Data Warehouse, ETL & OLAP

### 4.1 Star Schema (`sql/schema.sql`)

The SQLite database (`data/warehouse/regimetrader.db`) comprises **13 tables**:

| Table | Role | Grain (One Row Represents) | Primary Key / Uniqueness | Current Rows |
| :--- | :--- | :--- | :--- | ---: |
| `dim_date` | Dimension | One calendar date (`2015-01-01` to `2026-10-28`) | `date_key` (`YYYYMMDD` int) | 4,319 |
| `dim_stock` | Dimension | One instrument (50 Nifty equities + `^NSEI` + `^INDIAVIX`) | `stock_key`, unique `symbol` | 52 |
| `dim_regime` | Dimension | One market regime label (`BULL`, `SIDEWAYS`, `BEAR`, `HIGH_VOL`) | `regime_key`, unique `regime_label` | 4 |
| `dim_model` | Dimension | One trained model artifact (4 families $\times$ 4 walk-forward years) | `model_key`, unique `(model_name, version)` | 16 |
| `fact_price` | Fact | One instrument's daily OHLCV bar + quality flag | `(date_key, stock_key)` | 144,105 |
| `fact_indicator` | Fact | One equity/index daily technical indicator vector | `(date_key, stock_key)` | 141,195 |
| `fact_market_regime` | Fact | One market trading day's smoothed regime assignment | `date_key` | 2,886 |
| `fact_signal` | Fact | One stock/day/model `prob_up` prediction + primary `target_weight` | `(date_key, stock_key, model_key)` | 185,376 |
| `fact_trade` | Fact | One simulated execution fill (buy/sell, shares, price, cost, reason) | `trade_id` | 5,364 |
| `fact_portfolio_daily` | Fact | One run/day mark-to-market equity, cash, positions value & drawdown | `(run_id, date_key)` | 4,803 |
| `pending_target` | Fact | One run/day/stock next-open target weight instruction | `(run_id, date_key, stock_key)` | 0 (flat) |
| `etl_run_log` | Audit | One ETL execution audit record (mode, extracted, loaded, rejected) | `run_id` | 6 |
| `stg_price_raw` | Staging | Raw batch from most recent extraction (truncated per run) | None | 624 |

### 4.2 Universe & Coverage (`config/nifty50_symbols.csv`)
The equity universe covers the 50 constituents of the Nifty 50 (as of `2026-10-10`) across **15 sectors**:
- **Financial Services (7)**, **Pharma & Healthcare (5)**, **Banking (5)**, **Auto (5)**, **Metals & Mining (4)**, **IT (4)**, **FMCG (4)**, **Energy (3)**, **Services (2)**, **Consumer Durables (2)**, **Infrastructure (2)**, **Consumer Services (2)**, **Materials (2)**, **Power (2)**, **Telecom (1)**.
- *Note on Survivorship & Short Histories:* The universe is a static current constituent list, introducing survivorship bias. Furthermore, 7 of the 50 equities listed after `2015-01-02` (accounting for the 7,215-row difference between the full $52 \times 2,910 = 151,320$ Cartesian product and the 144,105 rows in `fact_price`).

### 4.3 Idempotent ETL & Data Quality (`src/etl/`)
- **Timezone Safety:** Timestamps from `yfinance` are stripped of timezone metadata **without converting to UTC**, so Indian Standard Time (`Asia/Kolkata`) trading dates never shift backward by one calendar day.
- **Validation Rules:** Hard rejection checks enforce valid OHLC relationships ($\text{High} \ge \text{Low}$, positive prices), non-negative equity volume (index/VIX volume exempt), and zero duplicate `(date, symbol)` keys. Missing bars are forward-filled for at most **2 consecutive trading days** and never back-filled.
- **Soft Quality Flags:** Extreme single-day adjusted-close jumps are flagged in `fact_price` (103 flagged rows out of 144,105, or `0.0715%`). Stocks with a flagged or forward-filled bar on day $t$ are placed in the strategy's **no-entry set** for that day.
- **Full-History Indicator Recompute:** Technical indicators are recomputed over the full price history rather than incrementally to eliminate Exponential Weighted Moving Average (EWM) initialization drift.

### 4.4 OLAP Analytical Suite (`sql/olap/` & `src/olap/`)
Nine named SQL queries provide multidimensional warehouse views (outputs saved in `reports/results/olap/`):
1. `01_rollup_sector_quarter`: Roll-up of average daily return by `(year, quarter, sector)` (720 rows).
2. `02_rollup_sector_year`: Annual sector return aggregation (180 rows).
3. `03_drilldown_sector_to_stock`: Drill-down from sector $\rightarrow$ month $\rightarrow$ individual stock (40 rows).
4. `04_slice_year`: Single-year cross-sectional slice of sector returns and volume (15 rows).
5. `05_dice_sectors_years_quarters`: Multi-axis dice across selected sectors, years, and quarters (12 rows).
6. `06_pivot_sector_by_year`: Cross-tabulated sector $\times$ year return matrix (`2022–2024`, 15 $\times$ 3).
7. `07_window_top5_momentum`: SQL window-function ranking (`RANK() OVER (...)`) of top-5 20-day momentum equities on a target date.
8. `08_regime_profile` & `09_classifier_by_regime`: Mining-linked OLAP queries (note: committed CSV snapshots in `reports/results/olap/` are empty, though `reports/results/regime_profile.csv` and live dashboard SQL queries are fully populated).

---

## 🧠 5. Data Mining: Features, Regimes & Classifiers

### 5.1 19-Feature Leakage-Safe Panel (`src/features/`)
Every feature row for stock $i$ on day $t$ is computed strictly from information available at the close of day $t$. The first **50 warm-up bars** per stock are dropped. Continuous features are winsorized at the **1st and 99th percentiles** fitted on training data ($\le$ `2022-12-31`):

| # | Feature Name | Category | Formula / Definition |
| :---: | :--- | :--- | :--- |
| 1–3 | `ret_1d`, `ret_5d`, `ret_20d` | Momentum | 1-day, 5-day, and 20-day close-to-close returns |
| 4 | `px_to_sma20` | Trend | $\text{Close}_t / \text{SMA}_{20,t} - 1$ |
| 5 | `px_to_sma50` | Trend | $\text{Close}_t / \text{SMA}_{50,t} - 1$ |
| 6 | `sma20_to_sma50` | Trend | $\text{SMA}_{20,t} / \text{SMA}_{50,t} - 1$ |
| 7 | `rsi_14` | Oscillator | 14-day Wilder Relative Strength Index ($0\text{--}100$) |
| 8 | `macd_hist_pct` | Momentum | $(\text{MACD}_{12,26} - \text{Signal}_9) / \text{Close}_t$ |
| 9 | `bb_pct_b` | Mean Reversion | Position within 20-day, $2\sigma$ Bollinger Bands |
| 10 | `atr_pct` | Volatility | 14-day Average True Range divided by $\text{Close}_t$ |
| 11 | `vol_20d` | Volatility | 20-day daily return standard deviation $\times \sqrt{252}$ |
| 12 | `volume_ratio_20` | Activity | $\text{Volume}_t / \text{SMA}_{20}(\text{Volume}_t)$ |
| 13 | `rel_strength_20` | Cross-Sectional | Stock 20-day return minus `^NSEI` 20-day return |
| 14 | `mkt_ret_20d` | Macro / Index | Nifty 50 (`^NSEI`) 20-day return |
| 15 | `india_vix` | Macro / Vol | India VIX (`^INDIAVIX`) daily close |
| 16–19 | `regime_bull`, `regime_sideways`, `regime_high_vol`, `regime_bear` | Regime State | One-hot encoded market regime indicators from `fact_market_regime` |

- **Target Variable (`y`):** Binary next-day direction $y_{i,t} = \mathbb{I}(\text{Close}_{i,t+1} > \text{Close}_{i,t})$. This is the **only** place in the codebase where `shift(-1)` is permitted.

### 5.2 Unsupervised Market Regime Discovery (`src/regime/`)
Market days are clustered using **K-Means** fitted on 4 macro features (`mkt_return_20d`, `mkt_vol_20d`, `india_vix`, `mkt_drawdown` from 252-day high) across the **1,948 training days** ($\le$ `2022-12-31`):
- **Selecting $k=3$:** Silhouette scores across $k \in \{2..6\}$ were `0.7706` ($k=2$), `0.3619` ($k=3$), `0.3635` ($k=4$), `0.3203` ($k=5$), and `0.3162` ($k=6$). Because $k=2$ was too coarse and $k=4$ created two tiny clusters each holding $<3\%$ of days, **$k=3$ (`BULL`, `SIDEWAYS`, `BEAR`)** was selected and frozen (`regime.refit: none`).
- **Causal Smoothing:** Raw daily cluster predictions are smoothed with a **trailing 5-day mode filter** (ties go to the most recent label) to prevent single-day regime whip-saws.

| Regime Label | Training Days ($\le$ 2022) | Test Days (2024–2026) | Current All Rows | Train Mean 20d Return | Train Mean 20d Vol | Train Mean VIX | Train Mean Drawdown |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **BULL** | 1,167 | 495 | 1,889 | `+2.56%` | `11.55%` | `15.42` | `-2.47%` |
| **SIDEWAYS** | 754 | 190 | 970 | `-1.04%` | `18.54%` | `20.23` | `-10.37%` |
| **BEAR** | 27 | **0** | 27 | `-15.33%` | `72.08%` | `57.16` | `-29.32%` |
| **HIGH_VOL** | 0 | 0 | 0 | — | — | — | — |

> **Critical Regime Observation:** All 27 `BEAR` days occurred during the 2020 COVID crash in the training set; the test period (`2024-01-01` to `2026-10-09`) contains **495 BULL days, 190 SIDEWAYS days, and 0 BEAR days**. Consequently, the strategy's emergency `BEAR` liquidation rule was never exercised out-of-sample.

### 5.3 Walk-Forward Direction Classifiers (`src/models/`)
Four pooled model architectures are trained across all 50 equities using an expanding annual walk-forward schedule (`wf2023`, `wf2024`, `wf2025`, `wf2026`), where each training window ends **5 calendar days (embargo)** before the prediction year begins:

1. **Logistic Regression (`logreg`):** `C=1.0`, `max_iter=1000` (with standard scaler).
2. **Decision Tree (`tree`):** `max_depth=4`, `min_samples_leaf=200`. *(Note: Excluded from strategy tuning because it produced degenerate cross-sectional rankings on validation—only 3 distinct probability values with `76.38%` of rows tied at a single score).*
3. **Random Forest (`rf`):** 300 trees, `max_depth=6`, `min_samples_leaf=50` *(Selected frozen model)*.
4. **XGBoost (`xgb`):** 300 boosting rounds, `learning_rate=0.03`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`.

#### Out-of-Sample Classifier Performance (`reports/results/classifier_metrics.csv`)

| Split | Window | Model | Accuracy | Majority Baseline | Accuracy Edge | ROC-AUC | Scored Rows |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Validation** | 2023 | `logreg` | `0.4980` | `0.5315` | `-0.0335` | `0.5222` | 12,044 |
| **Validation** | 2023 | `tree` | `0.5316` | `0.5315` | `+0.0001` | `0.5146` | 12,044 |
| **Validation** | 2023 | `rf` | `0.5125` | `0.5315` | `-0.0190` | **`0.5268`** | 12,044 |
| **Validation** | 2023 | `xgb` | `0.5059` | `0.5315` | `-0.0256` | `0.5222` | 12,044 |
| **Test** | 2024–2026 | `logreg` | `0.5066` | `0.5013` | `+0.0053` | `0.5140` | 34,200 |
| **Test** | 2024–2026 | `tree` | `0.4982` | `0.5013` | `-0.0032` | `0.5157` | 34,200 |
| **Test** | 2024–2026 | `rf` | `0.5058` | `0.5013` | `+0.0044` | `0.5157` | 34,200 |
| **Test** | 2024–2026 | `xgb` | `0.5108` | `0.5013` | **`+0.0094`** | **`0.5201`** | 34,200 |

---

## 📈 6. Trading Strategy & Execution Engine

### 6.1 Regime-Adaptive Portfolio Rules (`src/trading/rules.py`)
The frozen configuration (`config/frozen_config.yaml`) uses **Random Forest (`rf`)**, **`slots = 10`**, and **`entry_base = 0.504415`** (the 80th percentile of `rf` validation probabilities, selected via validation tuning):

| Market Regime | Entry Threshold (`prob_up`) | Max Gross Exposure | Per-Slot Target Weight (`10` slots) | Rule Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **BULL** | $\ge 0.504415$ (`+0.00`) | `100%` | `10.0%` | Fully active cross-sectional selection |
| **SIDEWAYS** | $\ge 0.554415$ (`+0.05`) | `60%` | `6.0%` | Higher conviction required; reduced position sizing |
| **HIGH_VOL** | $\ge 0.604415$ (`+0.10`) | `30%` | `3.0%` | Defensive posture *(0 days under $k=3$)* |
| **BEAR** | None | `0%` | `0.0%` | No new entries; liquidate all holdings at next open |

### 6.2 Churn Control & Risk Management
- **Hysteresis (`exit_prob = 0.50`):** Existing holdings are retained as long as `prob_up >= 0.50` and the regime is not `BEAR`, preventing flip-flopping when probabilities hover near the entry threshold.
- **Rebalance Band (`25%`):** Held positions whose current weight drifts by less than 25% relative to their target slot weight are left untouched to save transaction costs.
- **Stop-Loss & Cooldown:** Individual positions are exited at the next open if price falls **7% below entry**, triggering a mandatory **5-trading-day re-entry cooldown** for that symbol.
- **Deterministic Tie-Breaking:** Candidates are ranked by `prob_up` descending, with ties broken alphabetically (`symbol` A–Z).

### 6.3 Validation Tuning & Selection Protocol (`src/backtest/tune.py`)
To prevent overfitting on the test set, hyperparameter selection was performed exclusively on **2023 validation data**:
1. **Grid Search:** 3 non-degenerate models (`logreg`, `rf`, `xgb`) $\times$ 3 entry bases (`0.55`, `0.60`, plus a model-specific 80th percentile / `0.52` threshold) $\times$ 2 slot counts (`5`, `10`) = **18 validation grid trials**.
2. **Activity Filter (Decision R39):** Required $\ge 30$ trade rows and $\ge 10\%$ exposure on validation (10 of 18 passed).
3. **B4 Null Filter (Decision R40):** Shortlisted the top 3 configurations by validation Sharpe and ran 100 symbol-permutation simulations per candidate; only **`rf` / `entry_base=0.504415` / `slots=10`** exceeded its 90th-percentile validation null (achieving validation Sharpe `1.8328`, CAGR `20.70%`, 655 trade rows).

---

## 📊 7. Frozen Test Results & Benchmarks

### 7.1 Headline Out-of-Sample Performance (`2024-01-01` to `2026-10-09`, 685 Trading Days)
*Initial capital: ₹10,00,000. Frictions: 0.05% slippage + 0.10% commission per side (0.15% one-way / ~0.30% round-trip). Confidence intervals: 1,000-sample moving-block bootstrap (block length = 10 days, seed = 42).*

| Run ID / Strategy Variant | Sharpe | 95% Bootstrap CI | Sortino | CAGR | Max Drawdown | Calmar | Trade Rows (`n_trades`) | Final Equity (₹) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`backtest_main` (V2 Frozen RF Strategy)** | **`0.4848`** | **`[-0.4561, 1.5437]`** | **`0.4836`** | **`5.48%`** | **`19.32%`** | **`0.2840`** | **1,004** | **₹11,59,447** |
| `test_V0` (Classifier Only, No Regime Rules) | `0.4481` | `[-0.5046, 1.4946]` | `0.4449` | `5.96%` | `22.37%` | `0.2666` | 1,142 | ₹11,74,094 |
| `test_V1` (Regime Filter Only — *see §2 caveat*) | `0.6738` | `[-0.2337, 1.6914]` | `0.6156` | `10.29%` | `16.37%` | `0.6287` | 322 | ₹13,11,860 |
| `test_V2_nocost` (V2 with Zero Slippage/Costs) | `0.8949` | `[-0.0516, 1.9391]` | `0.9020` | `11.15%` | `15.66%` | `0.7122` | 1,000 | ₹13,40,368 |
| **`test_B1` (Buy & Hold `^NSEI` Index)** | `0.1576` | `[-0.7746, 1.1081]` | `0.1531` | `1.20%` | `15.59%` | `0.0768` | 1 | ₹10,33,520 |
| **`test_B2` (Equal-Weight 50 Stocks, Monthly)** | **`0.7017`** | `[-0.1866, 1.6856]` | **`0.6726`** | **`8.82%`** | `15.06%` | **`0.5856`** | 1,076 | **₹12,63,913** |
| **`test_B3` (Index Trend `SMA20 > SMA50`)** | `0.6117` | `[-0.2256, 1.4971]` | `0.5683` | `5.02%` | **`9.75%`** | `0.5154` | 819 | ₹11,45,472 |

### 7.2 Year-by-Year Test Return Decomposition

| Strategy / Benchmark | 2024 Return | 2025 Return | 2026 Return (to Oct 9) | Trade Breakdown (Test Period) |
| :--- | ---: | ---: | ---: | :--- |
| **V2 Frozen Strategy (`rf`)** | `+19.39%` | `+17.94%` | **`-17.52%`** | 463 Entries, 381 Exits, 83 Rebalances, 77 Stop-Losses |
| **V0 Classifier Only** | — | — | — | 576 Entries, 470 Exits, 0 Rebalances, 96 Stop-Losses |
| **V1 Regime Only (10-stock)** | `+13.56%` | `+19.28%` | `-3.87%` | 25 Entries, 0 Exits, 282 Rebalances, 15 Stop-Losses |
| **V2 Zero Costs** | `+27.41%` | `+21.99%` | `-13.66%` | 1,000 total trade rows |
| **B1 Buy & Hold `^NSEI`** | `+8.47%` | `+9.97%` | `-13.77%` | 1 Entry |
| **B2 Equal-Weight Monthly** | **`+19.65%`** | `+12.91%` | `-6.39%` | 50 Entries, 1,026 Rebalances |
| **B3 Trend `SMA20 > SMA50`** | `+16.99%` | `+4.35%` | `-6.65%` | 404 Entries, 399 Exits, 16 Rebalances |

---

## 🛡️ 8. Paper Trading & Crash-Safe Operations

`src/paper/daily_job.py` transitions the frozen strategy into daily forward operation (`run_id = paper_live`):
- **Atomic Per-Date Transactions:** Each unprocessed trading date is committed in its own SQLite transaction. A date is marked complete if and only if its `fact_portfolio_daily` row is written alongside its `fact_trade` and `pending_target` rows. If a run crashes mid-execution, re-running the job cleanly rolls back and re-processes only uncommitted dates.
- **Test-Lock Boundary Guard:** The paper job refuses to process any date $\le$ `2026-10-09` (the locked test boundary).
- **Current Operational State (as of `2026-10-28`):**
  - Processed **8 trading dates** (`2026-10-16` through `2026-10-28`).
  - Market regime across all 8 sessions: **`SIDEWAYS`** (`^NSEI` close `22,698.85`, `^INDIAVIX` `16.245` on `2026-10-28`).
  - On `2026-10-28`, all 50 equities were scored, yielding a maximum `prob_up` of **`0.521780`**—below the `SIDEWAYS` entry threshold of **`0.554415`** by `0.0326`.
  - Consequently, the paper portfolio has remained **100% in cash (₹10,00,000 equity, 0 positions, 0 trades, 0% drawdown)** and writes a daily markdown report for the latest processed date in `reports/daily/YYYY-MM-DD.md`.

---

## 🖥️ 9. Read-Only Streamlit Dashboard

The Streamlit console (`src/dashboard/app.py`) is strictly **read-only**—every database access is verified by tests to be a `SELECT` query, and it never recomputes backtest metrics on the fly:

1. **Market Overview:** Nifty 50 price chart overlaid with color-coded regime bands, India VIX history, regime transition counters, and a stale-data warning banner if the latest bar is $>3$ calendar days old.
2. **OLAP Explorer:** Interactive sector $\times$ quarter heatmaps, sector-to-stock drill-downs, year/quarter slices, and top-5 momentum window tables.
3. **Today's Signals:** Displays the latest committed `paper_live` decision date (`fact_portfolio_daily` marker), all 50 scored equities with `prob_up` vs. active regime threshold, `BUY/HOLD` indicators, and an explicit flat-book callout when all target weights are `0.0`.
4. **Backtest Results:** Headline comparison table, equity and drawdown curves, bootstrap confidence interval plots, yearly return breakdowns, and methodology disclosures.
5. **Model Diagnostics:** Validation and test ROC-AUC / accuracy tables, confusion matrices, probability distribution histograms, feature importances, and the decision-tree degeneracy warning.
6. **Bot Demo (60-Day Replay):** Replays 60 stored test days step-by-step on an isolated temporary copy of the warehouse. Verified by `reports/results/bot_equivalence.json` to match stored backtest probabilities within $3.33 \times 10^{-16}$.

---

## 🧪 10. Engineering Rigor, Testing & Leakage Guards

- **565 Automated Tests (`tests/`):** 26 test modules (unit, integration, property-based Hypothesis tests, schema constraints, and CLI tests) passing in `12m36s`.
- **14-Point Leakage Checklist (`reports/results/leakage_checklist.md`):** Automated tests verify that mutating future prices does not alter day-$t$ features, scalers/winsorizers reject out-of-train fitting, walk-forward splits enforce the 5-calendar-day embargo, and the dashboard/replay cannot write to the warehouse.
- **One-Time Test Lock (`reports/results/TEST_RUN_LOCK.json`):** Records `run_count: 2`, config SHA-256 hashes, and git commit. *(Run 1 crashed inside `b4_distribution` because an internal call omitted `allow_test_period=True`; Run 2 completed B4 after the fix, and `PRE_RERUN_CHECKSUMS.txt` proves all 6 pre-existing outputs remained bit-for-bit identical).*
- **Complete Experiment Audit (`reports/results/trial_log.csv`):** Logs all 342 runs (338 counted experimental trials + 4 calibration rows) so multiple-testing degrees of freedom are fully disclosed.

---

## 📂 11. Repository Structure

```text
regimetrader/
├── README.md                     # Project overview, architecture, results, and runbook
├── report.md                     # Comprehensive technical & empirical audit report
├── AGENTS.md                     # Engineering invariants and non-negotiable rules
├── Makefile                      # Cross-platform build, pipeline, test, and dashboard targets
├── pyproject.toml                # Package metadata, Ruff linter, and pytest configuration
├── requirements.txt              # Direct Python dependencies
├── requirements.lock             # Pinned environment lockfile (70 packages)
├── config/
│   ├── config.yaml               # Master pipeline, split, regime, model, and cost config
│   ├── config.smoke.yaml         # Fast synthetic-data smoke test configuration
│   ├── frozen_config.yaml        # Locked validation winner (rf, entry_base=0.504415, slots=10)
│   └── nifty50_symbols.csv       # 50 Nifty equities + sector mappings
├── sql/
│   ├── schema.sql                # Canonical 13-table SQLite star schema DDL
│   └── olap/                     # 01–09 named analytical SQL queries
├── src/
│   ├── common/                   # Config loader, SQLite connection manager, dates, manifests
│   ├── etl/                      # yfinance/NSE extraction, quality validation, upsert loader
│   ├── features/                 # Full-history technical indicators & 19-feature panel builder
│   ├── olap/                     # SQL OLAP query runner and CSV exporter
│   ├── regime/                   # K-Means clustering, label naming, 5-day mode smoothing
│   ├── models/                   # Walk-forward training, preprocessing, metrics, diagnostics
│   ├── trading/                  # Shared regime rules, target weights, and next-open execution
│   ├── backtest/                 # Simulation engine, bootstrap CIs, B1–B4 baselines, tuning
│   ├── paper/                    # Crash-safe daily paper job, state reconstruction, reports
│   └── dashboard/                # 6-screen read-only Streamlit app and 60-day bot replay
├── tests/                        # 26 test modules (565 tests) + fixtures
├── models/                       # 16 walk-forward model joblibs + regime_model.joblib
├── reports/
│   ├── results/                  # CSVs, RESULTS.md, trial_log.csv, TEST_RUN_LOCK.json, manifests/
│   ├── figures/                  # regime_bands.png, regime_k_selection.png
│   └── daily/                    # Daily paper-trading markdown reports (e.g., 2026-10-28.md)
├── docs/                         # System specs (02–08), DECISIONS.md, BUILD_PLAN.md, viva notes
├── deploy/                       # Optional cron, GitHub Actions, systemd, and Cloudflare tunnel kits
└── data/warehouse/               # regimetrader.db (SQLite star-schema warehouse)
```

---

## 🚀 12. Setup, CLI Usage & Runbook

### 12.1 Environment Setup & Quality Checks
Requires **Python 3.10+**. The `Makefile` automatically detects Windows (`.venv/Scripts/python.exe`) vs. POSIX (`.venv/bin/python`):

```bash
make setup                # Create .venv and install dependencies
make check                # Run Ruff linter + fast test suite (pre-commit gate)
make test                 # Run full 565-test suite (~12.5 mins)
make smoke                # Run end-to-end pipeline on deterministic synthetic data (offline)
make clean-clone-check    # Verify fresh clone setup and fast test suite in temp directory
```

### 12.2 Running the Dashboard & Paper Trader
```bash
# Launch the 6-screen read-only Streamlit dashboard at http://127.0.0.1:8501
make dashboard

# Run daily paper trading up to a target date (using existing warehouse bars)
.venv/Scripts/python.exe -m src.paper.daily_job --no-etl --to 20261028

# Or run incremental ETL + daily paper job
make paper
```

### 12.3 Research Pipeline Commands (Protected)
> **Warning:** The frozen models (`models/*.joblib`) and locked test evaluation (`reports/results/TEST_RUN_LOCK.json`) are already committed. Do **not** re-run `make train`, `make backtest`, or `make final-test` unless intentionally regenerating artifacts in a separate branch.

```bash
make etl-full             # Full historical download (2015–present) + validation + warehouse load
make features             # Recompute full-history technical indicators into fact_indicator
make regime               # Fit K-Means (<= 2022-12-31) and write smoothed fact_market_regime
make train                # Train 16 walk-forward models (wf2023–wf2026) and write fact_signal
make backtest             # Run validation tuning grid, B4 null filter, and validation baselines
make final-test           # Guarded one-time test evaluation (refuses rerun without --allow-rerun)
```

---

## 🔍 13. Known Limitations, Caveats & Audit Notes

For anyone auditing the codebase or extending the research, the following caveats are documented in `report.md` and `DECISIONS.md`:

1. **V1 Ablation Implementation Gap (`src/backtest/strategies.py`):** As detailed in §2, `make_v1` returns `None` outside `BULL` (which clears pending orders rather than liquidating held positions), slices `available[:slots]` (10 stocks instead of all 50), and leaves stop-loss enabled. A spec-compliant V1 ablation would need to return `{}` outside `BULL` and use all available symbols.
2. **B4 Permutation Null Turnover Mismatch:** `src/backtest/baselines.py::permuted_signals` permutes probabilities cross-sectionally within each day. Because daily permutations destroy the temporal autocorrelation of scores, the null churns positions daily (~2,600 trade rows vs. V2's 1,004) and incurs ~2.6× higher transaction costs.
3. **Static Universe Survivorship Bias:** `config/nifty50_symbols.csv` reflects Nifty 50 membership as of October 2026 rather than point-in-time historical constituents.
4. **Zero Out-of-Sample `BEAR` Days:** Because $k=3$ K-Means calibrated its `BEAR` centroid on the extreme 2020 COVID crash (mean VIX `57.16`, mean 20d volatility `72.08%`), milder corrections in 2024–2026 were classified as `SIDEWAYS`.
5. **Minor Metadata & Artifact Quirks:**
   - `dim_model.trained_through` records `2022-12-31` across all 16 rows even though `wf2024–wf2026` expand their training cutoffs (enforced via code assertions in `src/models/train.py`).
   - `summary_test.csv` reports `exposure` as the fraction of days holding $\ge 1$ position (`0.99854`), whereas time-averaged invested capital weight for V2 is `77.73%`.
   - In `paper_live`, `fact_signal` rows are tagged with `model_key = 9` (`rf`, `wf2023`) because `_model_key()` looks up the first row matching `model_name = 'rf'`, even though actual scoring uses `rf_wf2026.joblib`.