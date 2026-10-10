# 03. Data Warehouse, ETL and OLAP

The canonical DDL is **`sql/schema.sql`**. Apply it; do not retype it. This file explains what the tables mean and defines the ETL, checks and OLAP behaviour.

---

## 1. Goals and volume

- Store daily history once, in a form that supports both analysis (OLAP) and mining (features).
- Dimensions are small and descriptive; facts are numeric.
- Every table is loadable repeatedly without duplicates.
- Volume: about 52 instruments (50 stocks, the index, VIX) times about 2,900 trading days, roughly 150,000 rows per daily fact table. SQLite handles this.

---

## 2. Schema summary

| Table | Kind | Grain (one row is) | Primary key | Written by |
|---|---|---|---|---|
| `stg_price_raw` | staging | one raw bar from the latest extract | none | extract (truncated at the start of each run) |
| `etl_run_log` | audit | one ETL run | `run_id` | load |
| `dim_date` | dimension | one calendar day | `date_key` (YYYYMMDD) | load |
| `dim_stock` | dimension | one instrument (stock, index, VIX) | `stock_key`, unique `symbol` | universe seed |
| `dim_regime` | dimension | one regime label | `regime_key` | `schema.sql` seed |
| `dim_model` | dimension | one trained model version | `model_key`, unique (`model_name`, `version`) | training |
| `fact_price` | fact | one instrument on one trading day | (`date_key`, `stock_key`) | load |
| `fact_indicator` | fact | one instrument, one day (derived measures) | (`date_key`, `stock_key`) | feature builder |
| `fact_market_regime` | fact | one trading day (market level) | `date_key` | regime engine |
| `fact_signal` | fact | one instrument, one day, one model | (`date_key`, `stock_key`, `model_key`) | training (`prob_up`); primary run (`signal`, `target_weight`) |
| `fact_trade` | fact | one executed virtual trade (`reason`: ENTRY, REBALANCE, EXIT, STOP_LOSS) | `trade_id` | backtest, paper |
| `fact_portfolio_daily` | fact | one portfolio run on one day | (`run_id`, `date_key`) | backtest, paper |
| `pending_target` | state | one target weight decided at a close | (`run_id`, `decision_date_key`, `stock_key`) | paper |

**Concept hierarchies** (roll-up and drill-down): Date: day, week, month, quarter, year. Stock: stock, industry, sector, index. Regime: flat.

**Star versus snowflake:** sector and industry live inside `dim_stock` (denormalised): a true star with fewer joins. A snowflake would split out `dim_sector`, saving a little storage at the cost of a join.

Conventions: `date_key` is an integer `YYYYMMDD`. Prices are adjusted for splits and dividends. `fact_indicator.vol_20d` is an annualised decimal (0.18 = 18%). `india_vix` is in VIX points (13.9).

---

## 3. Universe and seed rows

**Universe file** `config/nifty50_symbols.csv`: columns `symbol, company_name, sector, industry`. Lines starting with `#` are comments (read with `comment="#"`). Symbols use the Yahoo form with `.NS` suffix (examples of awkward ones: `M&M.NS`, `BAJAJ-AUTO.NS`).

- Source: the current Nifty 50 constituent list from NSE (niftyindices.com `ind_nifty50list.csv`). Freeze it at project start and commit it. Put `# as of YYYY-MM-DD` in the first line.
- **Never fabricate this list.** If it is missing and cannot be downloaded, stop and ask the human.
- `industry` is NSE's industry text. `sector` is the project label below. NSE's "Financial Services" must be split by hand into `Banking` and `Financial Services`.
- The labels `Banking` and `IT` MUST exist (OLAP defaults use them). Suggested labels: `Banking`, `Financial Services`, `IT`, `Energy`, `Power`, `Auto`, `FMCG`, `Pharma and Healthcare`, `Metals and Mining`, `Infrastructure`, `Telecom`, `Consumer Durables`, `Consumer Services`, `Materials`, `Services`.
- Tickers can change after mergers or renames. The ETL warns when a symbol returns no data and continues.

**Seeds:**

- `dim_regime`: loaded by `schema.sql`.
- `dim_stock`: one row per universe symbol (`EQUITY`), plus `^NSEI` (`INDEX`, sector `INDEX`, company "Nifty 50 Index") and `^INDIAVIX` (`VOLATILITY`, sector `VOLATILITY`, company "India VIX"). The index and VIX reuse `fact_price`.

---

## 4. ETL rules

### 4.1 Extract

1. Symbols = universe plus `^NSEI` and `^INDIAVIX`.
2. Source `yfinance`: `yfinance.Ticker(symbol).history(start=..., auto_adjust=True)`. Returns adjusted OHLCV with a timezone-aware index.
3. Retry up to `data.retries` times with the waits in `data.retry_wait_seconds`. On final failure log the symbol, count it, and continue.
4. Save each raw response to `data/raw/<symbol>_<run_date>.csv`; insert into `stg_price_raw` (after truncating it).
5. Source `csv`: read `data.csv_dir/<symbol>.csv` (columns `date, open, high, low, close, volume`). Source `synthetic`: call `src/etl/synthetic.py`.
6. **Full refresh:** start = `data.start_date`. **Incremental:** start = last loaded index date minus `data.incremental_overlap_days`. An empty warehouse is always a full refresh.
7. VIX fallback: if `^INDIAVIX` is empty or too short, load `data.vix_csv` (columns `date, close`; OHLC default to close). If still unavailable, continue without VIX (`regime.use_vix: auto` handles it).

Extract output (long format): columns `symbol, trade_date, open, high, low, close, volume`.

### 4.2 Transform (apply in this order)

| Step | Rule |
|---|---|
| 1. Normalise | Lower-case column names. Remove the timezone with `tz_localize(None)` on the local timestamps (**never convert to UTC first**). Build `date_key`. |
| 2. Deduplicate | Keep the last row per (symbol, date). |
| 3. Trading calendar | A trading day is any date on which `^NSEI` has a bar. Rows of any symbol on other dates are dropped (reason `NON_TRADING_DAY`). |
| 4. Fill gaps | For each EQUITY and VIX symbol, reindex to the trading calendar from the symbol's first bar to the last calendar date. Forward-fill up to `data.ffill_max_days` consecutive missing trading days: filled rows get open = high = low = close = previous close, volume 0, `is_flagged = 1`. In a longer gap only the first `data.ffill_max_days` days are filled and the rest stay missing. Never back-fill. Never create rows before a symbol's first bar. |
| 5. Hard validation (reject row) | price <= 0; high < low; high < max(open, close) or low > min(open, close) by more than 1e-6 relative; volume < 0. Rejected rows go to `data/raw/rejected_<run_date>.csv` with a reason column. |
| 6. Soft validation (flag, keep) | EQUITY only: absolute 1-day return above `data.anomaly_abs_return` (possible unadjusted corporate action); zero volume on a trading day (possible suspension), excluding rows already filled in step 4. |
| 7. Date dimension | One `dim_date` row per calendar day from `data.start_date` to the last loaded date; `is_trading_day` from the index calendar. |

Return `(clean_df, rejected_df)`. `clean_df` has `symbol, date_key, open, high, low, close, volume, is_flagged`.

### 4.3 Load

1. Open one transaction (SQLAlchemy `engine.begin()`).
2. Upsert `dim_date`, `dim_stock`, `dim_regime` (regime already seeded), then `fact_price`. Use named parameters and `executemany`:

```sql
INSERT INTO fact_price (date_key, stock_key, open_price, high_price, low_price, close_price, volume, is_flagged)
VALUES (:date_key, :stock_key, :open, :high, :low, :close, :volume, :is_flagged)
ON CONFLICT (date_key, stock_key) DO UPDATE SET
    open_price  = excluded.open_price,
    high_price  = excluded.high_price,
    low_price   = excluded.low_price,
    close_price = excluded.close_price,
    volume      = excluded.volume,
    is_flagged  = excluded.is_flagged;
```

3. Recompute `dim_date.is_month_end` (last trading day of each calendar month present).
4. Write one `etl_run_log` row with rows extracted, loaded and rejected; commit. On any error roll back and record `FAILED` (in a separate short transaction so the failure is still logged).

Indicators are computed after the load by the feature builder (`04` section 2).

---

## 5. Data-quality checks

Run after every ETL (`src/etl/quality.py`). Return a list of `CheckResult(id, severity, passed, detail)`. Hard failures raise `DataQualityError` and the pipeline exits with code 4.

| ID | Check | Severity |
|---|---|---|
| DQ-01 | No duplicate (`date_key`, `stock_key`) in `fact_price` | Hard |
| DQ-02 | No non-positive prices; high >= open, close, low and low <= open, close (tolerance 1e-6 relative) | Hard |
| DQ-03 | Every `fact_price.date_key` exists in `dim_date` and `is_trading_day = 1` | Hard |
| DQ-04 | Row count per stock within `quality.row_count_tolerance` (2%) of the index's trading-day count, **counted from that stock's first bar** | Soft |
| DQ-05 | Share of flagged rows below `quality.max_flagged_share` (0.5%) | Soft |
| DQ-06 | Latest loaded date within `data.stale_after_days` calendar days of today (skipped for synthetic and replay runs) | Soft |
| DQ-07 | Rejected rows below `quality.max_rejected_share` (1%) of extracted rows | Soft |

Soft failures log a WARNING and are shown in the run summary. Report the count of stocks with fewer than 10 years of history (`DECISIONS` R10).

---

## 6. OLAP queries

Files in `sql/olap/` (SQLite, named parameters). `src/olap/run.py` runs them:

| File | Operation | Parameters (defaults in the file header as `-- param: name = value`) |
|---|---|---|
| `01_rollup_sector_quarter.sql` | Roll-up, detailed level | none |
| `02_rollup_sector_year.sql` | Roll-up, rolled-up level | none |
| `03_drilldown_sector_to_stock.sql` | Drill-down | `sector`, `year` |
| `04_slice_year.sql` | Slice | `year` |
| `05_dice_sectors_years_quarters.sql` | Dice | `sector_a`, `sector_b`, `year_from`, `year_to`, `quarter_a`, `quarter_b` |
| `06_pivot_sector_by_year.sql` | Pivot | years from `olap.pivot_years` (runner regenerates the `CASE` columns via `build_pivot_sql(years)`) |
| `07_window_top5_momentum.sql` | Window function (RANK) | `as_of_date` |
| `08_regime_profile.sql` | Mining-linked: market profile per regime | none |
| `09_classifier_by_regime.sql` | Mining-linked: hit rate per regime | `model_name` |

Runner behaviour:

- Parse `-- param:` lines for defaults; accept `--param name=value`; pass **only** the parameters named in the statement; cast numeric strings to numbers.
- Write each result to `reports/results/olap/<file stem>.csv` and print it.
- Defaults that depend on the data (year, `as_of_date`) should fall back to the latest available year or date when the default has no rows; log which was used.
- Query 08 returns rows once regimes exist (Phase 3). Query 09 needs `signal = 1` rows, which only the primary runs write (`backtest_main` in the final test, Phase 6, and `paper_live`), so it is empty before that. An empty result is not an error.
- PostgreSQL note: `GROUP BY ROLLUP (d.year, d.quarter, s.sector)` produces all subtotal levels in one query. Not available in SQLite.

---

## 7. Porting to PostgreSQL (optional, do not do unless asked)

| SQLite | PostgreSQL |
|---|---|
| `INTEGER PRIMARY KEY AUTOINCREMENT` | `GENERATED ALWAYS AS IDENTITY` |
| Dates as `TEXT` | `DATE` |
| `CURRENT_TIMESTAMP` default on `TEXT` | `TIMESTAMP` |
| `INSERT OR IGNORE` | `ON CONFLICT DO NOTHING` |
| No `ROLLUP` | `GROUP BY ROLLUP (...)` |
