-- RegimeTrader warehouse schema (SQLite). Canonical DDL: do not retype it elsewhere.
-- Apply with src.common.db.init_db(); every statement is idempotent (IF NOT EXISTS / OR IGNORE).
-- Connection must run: PRAGMA foreign_keys = ON;  PRAGMA journal_mode = WAL;
-- Lines marked "ADDED" differ from the original blueprint (see docs/spec/DECISIONS.md).

-- ---------- Staging and audit ----------
CREATE TABLE IF NOT EXISTS stg_price_raw (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    open_price  REAL,
    high_price  REAL,
    low_price   REAL,
    close_price REAL,
    volume      INTEGER,
    loaded_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS etl_run_log (
    run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    mode           TEXT NOT NULL CHECK (mode IN ('FULL', 'INCREMENTAL')),
    rows_extracted INTEGER,
    rows_loaded    INTEGER,
    rows_rejected  INTEGER,
    status         TEXT NOT NULL CHECK (status IN ('RUNNING', 'SUCCESS', 'FAILED'))
);

-- ---------- Dimensions ----------
CREATE TABLE IF NOT EXISTS dim_date (
    date_key       INTEGER PRIMARY KEY,          -- YYYYMMDD
    full_date      TEXT NOT NULL UNIQUE,         -- ISO date
    day_of_week    INTEGER NOT NULL,             -- 0 = Monday
    day_name       TEXT NOT NULL,
    week_of_year   INTEGER NOT NULL,
    month          INTEGER NOT NULL,
    month_name     TEXT NOT NULL,
    quarter        INTEGER NOT NULL,
    year           INTEGER NOT NULL,
    is_trading_day INTEGER NOT NULL DEFAULT 1,   -- 1 when ^NSEI has a bar that day
    is_month_end   INTEGER NOT NULL DEFAULT 0    -- 1 on the last TRADING day of the month (recomputed after each load)
);

CREATE TABLE IF NOT EXISTS dim_stock (
    stock_key       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL UNIQUE,        -- e.g. RELIANCE.NS, ^NSEI
    company_name    TEXT NOT NULL,
    sector          TEXT NOT NULL,               -- INDEX and VOLATILITY rows use 'INDEX' / 'VOLATILITY'
    industry        TEXT,
    instrument_type TEXT NOT NULL CHECK (instrument_type IN ('EQUITY', 'INDEX', 'VOLATILITY'))
);

CREATE TABLE IF NOT EXISTS dim_regime (
    regime_key   INTEGER PRIMARY KEY,
    regime_label TEXT NOT NULL UNIQUE,           -- BULL, SIDEWAYS, HIGH_VOL, BEAR
    description  TEXT
);

CREATE TABLE IF NOT EXISTS dim_model (
    model_key       INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,               -- logreg | tree | rf | xgb (suffix _rel for the relative-target variant)
    version         TEXT NOT NULL,               -- 'wf<year>' = trained to predict that calendar year
    trained_through TEXT NOT NULL,               -- last date in the training data
    params_json     TEXT,
    UNIQUE (model_name, version)
);

-- ---------- Facts ----------
-- Prices are adjusted for splits and dividends.
CREATE TABLE IF NOT EXISTS fact_price (
    date_key    INTEGER NOT NULL REFERENCES dim_date(date_key),
    stock_key   INTEGER NOT NULL REFERENCES dim_stock(stock_key),
    open_price  REAL,
    high_price  REAL,
    low_price   REAL,
    close_price REAL,
    volume      INTEGER,
    is_flagged  INTEGER NOT NULL DEFAULT 0,      -- 1 = failed a soft quality check or was forward-filled
    PRIMARY KEY (date_key, stock_key)
);

CREATE TABLE IF NOT EXISTS fact_indicator (
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    stock_key       INTEGER NOT NULL REFERENCES dim_stock(stock_key),
    return_1d       REAL,
    return_5d       REAL,
    return_20d      REAL,
    sma_20          REAL,
    sma_50          REAL,
    ema_12          REAL,
    ema_26          REAL,
    macd            REAL,
    macd_signal     REAL,
    rsi_14          REAL,
    bb_pct_b        REAL,
    atr_14          REAL,
    vol_20d         REAL,                        -- annualized, as a decimal (0.18 = 18%)
    volume_ratio_20 REAL,
    PRIMARY KEY (date_key, stock_key)
);

CREATE TABLE IF NOT EXISTS fact_market_regime (
    date_key       INTEGER PRIMARY KEY REFERENCES dim_date(date_key),
    regime_key     INTEGER NOT NULL REFERENCES dim_regime(regime_key),  -- smoothed label (used everywhere downstream)
    cluster_id     INTEGER NOT NULL,                                    -- raw K-Means cluster before smoothing
    mkt_return_20d REAL,
    mkt_vol_20d    REAL,
    india_vix      REAL,                                                -- VIX points (13.9), NULL when VIX is not used
    mkt_drawdown   REAL
);

-- One row per instrument, day and model. prob_up is written by training/prediction.
-- signal and target_weight are written ONLY by the primary strategy run (backtest_main, then paper_live);
-- until then they hold 0 / 0.0. Ablation and baseline runs do not write this table.
CREATE TABLE IF NOT EXISTS fact_signal (
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    stock_key       INTEGER NOT NULL REFERENCES dim_stock(stock_key),
    model_key       INTEGER NOT NULL REFERENCES dim_model(model_key),
    regime_key      INTEGER NOT NULL REFERENCES dim_regime(regime_key),
    prob_up         REAL NOT NULL,
    signal          INTEGER NOT NULL DEFAULT 0,  -- 1 = want a position, 0 = no position
    target_weight   REAL NOT NULL DEFAULT 0.0,
    next_day_return REAL,                        -- filled in once the next trading day is known
    PRIMARY KEY (date_key, stock_key, model_key)
);

CREATE TABLE IF NOT EXISTS fact_trade (
    trade_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,                  -- e.g. backtest_main, paper_live
    date_key     INTEGER NOT NULL REFERENCES dim_date(date_key),   -- execution date
    stock_key    INTEGER NOT NULL REFERENCES dim_stock(stock_key),
    side         TEXT NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity     INTEGER NOT NULL,
    fill_price   REAL NOT NULL,                  -- after slippage
    cost         REAL NOT NULL,                  -- cost_pct * quantity * fill_price
    realized_pnl REAL,                           -- SELL rows only; NULL for BUY
    reason       TEXT NOT NULL CHECK (reason IN ('ENTRY', 'REBALANCE', 'EXIT', 'STOP_LOSS'))  -- ADDED: why the trade happened
);

CREATE TABLE IF NOT EXISTS fact_portfolio_daily (
    run_id          TEXT NOT NULL,
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    cash            REAL NOT NULL,
    positions_value REAL NOT NULL,
    equity          REAL NOT NULL,
    daily_return    REAL,
    drawdown        REAL,
    PRIMARY KEY (run_id, date_key)
);

-- ADDED: target weights decided at the close of decision_date, executed at the next open.
-- Used by the paper broker. An empty set of rows for a processed date means "go to cash".
CREATE TABLE IF NOT EXISTS pending_target (
    run_id            TEXT NOT NULL,
    decision_date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    stock_key         INTEGER NOT NULL REFERENCES dim_stock(stock_key),
    target_weight     REAL NOT NULL,
    prob_up           REAL,
    PRIMARY KEY (run_id, decision_date_key, stock_key)
);

-- ---------- Indexes for common access paths ----------
CREATE INDEX IF NOT EXISTS idx_price_stock_date  ON fact_price (stock_key, date_key);
CREATE INDEX IF NOT EXISTS idx_ind_stock_date    ON fact_indicator (stock_key, date_key);
CREATE INDEX IF NOT EXISTS idx_signal_date       ON fact_signal (date_key);
CREATE INDEX IF NOT EXISTS idx_trade_run_date    ON fact_trade (run_id, date_key);

-- ---------- Seed rows ----------
INSERT OR IGNORE INTO dim_regime (regime_key, regime_label, description) VALUES
    (1, 'BULL',     'Rising market, low to moderate volatility'),
    (2, 'SIDEWAYS', 'Flat market, low volatility'),
    (3, 'HIGH_VOL', 'Turbulent market, high volatility, mixed direction'),
    (4, 'BEAR',     'Falling market, high volatility, deep drawdown');
