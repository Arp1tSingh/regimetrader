# 01. Problem Statement and Real-Life Use Case

**Purpose of this file:** justify the project: what problem it solves, for whom, within what limits, and how it maps to the DWM syllabus.

---

## 1. Problem statement

Retail investors and small advisory desks face two recurring problems:

1. **Which stocks should I hold tomorrow?** Price history is noisy, and manual chart reading does not scale to 50 stocks.
2. **How much risk should I take right now?** A strategy that works in a calm uptrend can lose badly in a crash. Most simple trading bots ignore this: they use one model and trade the same way in every market condition.

**RegimeTrader** addresses both. It stores market history in a data warehouse, mines the history to identify the current *market regime* (bull, bear, sideways, high volatility), predicts next-day stock direction, and scales exposure according to the regime.

---

## 2. Objectives

| ID | Objective | Measurable outcome |
|---|---|---|
| O1 | Build a star-schema warehouse | At least 10 years of daily data for 50 stocks, the index and VIX, loaded by a repeatable ETL with automated quality checks |
| O2 | Demonstrate OLAP | Roll-up, drill-down, slice, dice and pivot queries with results shown in the report |
| O3 | Detect market regimes | K-Means regimes with clearly different return and volatility profiles |
| O4 | Predict direction from history | At least 3 classifiers trained on past data and evaluated on later, unseen data |
| O5 | Evaluate a regime-aware strategy | Backtest with transaction costs against at least 3 baselines |
| O6 | Operate the bot | Daily paper trading with every signal and trade logged in the warehouse |

---

## 3. Real-life use cases

### Use case A (primary): daily decision support for a swing trader

| Item | Description |
|---|---|
| Actor | A retail investor who holds stocks for days to weeks |
| Trigger | After the market closes at 3:30 PM IST |
| Flow | 1. The system loads today's prices into the warehouse. 2. It classifies today's market regime. 3. It ranks stocks by predicted probability of rising tomorrow. 4. It outputs a short watchlist with suggested position sizes, scaled down or to zero when the regime is risky. |
| Output | "Market regime: BULL. Buy these 3, hold these 7, sell this 1." |
| Value | Replaces ad-hoc chart reading with a consistent, back-tested process, and reduces exposure automatically when markets turn dangerous |

### Use case B: risk overlay for a small advisory desk

The regime flag alone is useful. A small portfolio manager can reduce equity exposure when the system reports HIGH_VOL or BEAR, regardless of how the desk picks stocks.

### Use case C: "market mood" feature in a fintech app

The regime label and the OLAP views (sector performance by quarter, regime history) become an analytics page inside a retail investing app.

---

## 4. Requirement mapping

| Class requirement | How the project satisfies it | Evidence to show |
|---|---|---|
| Use of historical data for future prediction | Warehouse holds 10+ years of history; classifiers train on the past and predict next-day direction; regimes are learned from history and applied to new days | Walk-forward results table, confusion matrix, out-of-sample equity curve |
| Real-life use case | Daily decision-support tool for investors (Use case A) with a working paper-trading run | Daily signal report, paper-trading log, dashboard |

---

## 5. Scope

| In scope | Out of scope (listed as future work) |
|---|---|
| Daily end-of-day data | Intraday or tick data |
| NSE cash equities (Nifty 50 universe) | Futures, options and other derivatives |
| Long-only strategy | Short selling |
| Backtesting and paper trading | Live broker integration and real money |
| Technical-indicator features | News and social-media sentiment |
| K-Means regimes | Hidden Markov Models, deep learning |

---

## 6. Assumptions and constraints

- Free data (yfinance) is unofficial and can have gaps or revisions. NSE downloads are the fallback.
- The Nifty 50 constituent list is frozen at project start. Past index membership changed, so results carry **survivorship bias**. State this in the report.
- Prices are adjusted for splits and dividends.
- Orders fill at the next day's open price, plus slippage and costs (see `06_backtesting_and_evaluation.md`).
- Everything must run on an ordinary laptop.
- The project is academic. It is not investment advice.

---

## 7. DWM syllabus mapping

Syllabus names differ between universities. Check this table against your own syllabus and past papers, and add any missing topic as an extension.

| Syllabus area | Where it appears in the project | File |
|---|---|---|
| Data warehouse concepts: fact, dimension, grain, star schema, hierarchies | Warehouse design | `03` |
| ETL process | Extract from API, transform and validate, load | `03` |
| OLAP operations | Roll-up, drill-down, slice, dice, pivot queries | `03` |
| Data preprocessing: cleaning, transformation, normalization, discretization | ETL cleaning, feature scaling, up/down discretization | `03`, `04` |
| Classification: decision tree, ensemble methods, evaluation metrics | Direction classifiers, confusion matrix | `04` |
| Clustering: K-Means, cluster validity | Market-regime detection, elbow and silhouette | `04` |
| Association rules: Apriori, support, confidence, lift | Optional sector co-movement extension | `04` |
| Model evaluation and validation | Walk-forward validation, backtesting | `06` |
| Applications and case study | Use cases and report | `01`, `10` |

---

## 8. Success criteria

| Level | What it looks like |
|---|---|
| Minimum (pass) | End-to-end pipeline works; star schema and ETL; at least 3 OLAP queries; K-Means regimes; one classifier; backtest with one baseline; report |
| Good | Four classifiers compared; regime-aware version versus a regime-free ablation; costs included; dashboard |
| Excellent | Statistical honesty section (confidence intervals, number of experiments reported); leakage tests; paper-trading log; clear limitations |

Beating buy-and-hold on raw return is **not** required. A lower maximum drawdown, a better Sharpe ratio, or an honest and well-analysed negative result all count as valid findings.

---

## 9. User stories

- As a trader, I want a ranked watchlist after market close so that I can plan tomorrow's orders.
- As a risk-conscious investor, I want the system to cut exposure automatically in turbulent markets.
- As an evaluator, I want to re-run the whole pipeline from raw data and get the same results.
- As a student, I want every step mapped to a syllabus topic so I can explain it in the viva.
