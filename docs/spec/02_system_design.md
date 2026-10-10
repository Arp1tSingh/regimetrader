# 02. System Design

Architecture, components and principles. Terms: MUST / MUST NOT are binding; "should" is a strong default.

---

## 1. Layers

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
APPLICATION   Strategy rules -> Backtester | Paper broker | Dashboard   (src/trading, src/backtest, src/paper, src/dashboard)
```

**Key idea:** the warehouse is the single source of truth. Each module reads its inputs from warehouse tables and writes its outputs back to warehouse tables (plus model files in `models/`). Modules stay independent, every result is traceable, and the data-warehousing story of the project is intact.

---

## 2. Components

| Component | Module | Responsibility | Reads | Writes |
|---|---|---|---|---|
| Extractor | `src/etl/extract.py` | Download OHLCV for the symbol list and date range | Config, API, CSV, synthetic | `stg_price_raw`, `data/raw/*.csv` |
| Transformer | `src/etl/transform.py` | Clean, align to trading calendar, validate | Raw frames | Clean frame, rejected-rows CSV |
| Loader | `src/etl/load.py` | Upsert dimensions and facts in one transaction | Clean frame | `dim_*`, `fact_price`, `etl_run_log` |
| Quality | `src/etl/quality.py` | Hard and soft data-quality checks | Warehouse | Check results |
| Feature builder | `src/features/` | Indicators; model-ready panel | `fact_price` | `fact_indicator`, `data/processed/panel.parquet` |
| Regime engine | `src/regime/regime_model.py` | Fit K-Means on the training period; label every day | `fact_price` (index, VIX), `fact_indicator` | `fact_market_regime`, `models/regime_model.joblib` |
| Prediction engine | `src/models/` | Walk-forward training; probabilities | Panel | `fact_signal`, `dim_model`, `models/*.joblib` |
| Strategy rules | `src/trading/rules.py` | Probabilities and regime to target weights | Signals, regime | Target weights (no I/O) |
| Execution core | `src/trading/{portfolio,execution}.py` | Order planning, fills, costs, portfolio state | Targets, prices | Trades (in memory) |
| Backtester | `src/backtest/` | Simulate history with costs; metrics; baselines | Warehouse | `fact_trade`, `fact_portfolio_daily` (`run_id`), result files |
| Paper broker | `src/paper/` | Daily virtual execution with the same core | Warehouse | Same tables (`run_id = paper_live`), `pending_target` |
| Dashboard | `src/dashboard/app.py` | Visualise warehouse content and results | All facts | Nothing |

---

## 3. Data flow

- **Research flow:** raw history, ETL, warehouse, indicators, regime model, panel, walk-forward classifiers, signals, backtest, metrics (`08` section 2).
- **Daily flow:** new prices, incremental ETL, indicators, regime, probabilities, target weights, virtual orders for the next open (`08` section 3).

Both flows share the same feature, regime, prediction and strategy code. Live logic therefore equals what was backtested; a test proves it (`TESTS.md` section 8).

---

## 4. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Data | `pandas`, `numpy`, `pyarrow`, `yfinance` |
| Warehouse | SQLite through SQLAlchemy 2.x (raw SQL with named parameters); PostgreSQL optional |
| Mining | `scikit-learn`, `xgboost`; `mlxtend` for the optional Apriori task |
| Visualisation | `matplotlib` (report figures), `plotly` (dashboard) |
| Dashboard | `streamlit` |
| Scheduling | `cron` or GitHub Actions (example files only) |
| Testing and lint | `pytest`, `ruff` |

---

## 5. Design principles

1. **No look-ahead.** A feature for day *t* uses only data up to the close of *t*. Tested automatically.
2. **Chronological everything.** Train, validation, test are consecutive blocks. No shuffling.
3. **Idempotent loads and jobs.** Re-running changes nothing.
4. **Config-driven.** Dates, thresholds, costs and model settings live in `config/config.yaml`.
5. **Reproducible.** Fixed seeds, pinned dependencies, `run_id` and model version on every result, run manifests.
6. **Traceable.** Every signal row stores its model and regime; every trade stores its `run_id`.
7. **One strategy code path** for backtest and paper trading.

---

## 6. Dashboard screens (Phase 8)

| Screen | Content |
|---|---|
| Market overview | Index price with coloured regime bands; India VIX; current regime badge; stale-data banner (red if latest data is older than `paper.stale_banner_days` calendar days) |
| OLAP explorer | Sector by quarter heat map; drill-down from sector to stock; filters for year and regime |
| Today's signals | Per stock: `prob_up`, action (BUY, HOLD, SELL), target weight, from the latest `paper_live` decision |
| Backtest results | Equity curves of strategy and baselines; drawdown chart; metrics table from `summary_*.csv` |
| Model diagnostics | Confusion matrix; feature importance; `prob_up` histogram |

Each screen MUST show a clear message instead of an exception when its tables are empty. Queries are read-only.

---

## 7. Design choices that are fixed (do not revisit)

| Question | Choice | Rejected alternatives |
|---|---|---|
| Model structure | One pooled model for all stocks (about 100k training rows) | Per-stock, per-regime (per-regime is a stretch experiment, V4) |
| Regime method | K-Means | Gaussian Mixture, Hidden Markov Model (future work) |
| Target | Next-day direction | Return regression, multi-day horizon |
| Integration | ETL (validate before load) | ELT |
| Database | SQLite | PostgreSQL (SQL kept portable) |
| Schema | Star (sector and industry denormalised into `dim_stock`) | Snowflake |
