# 09. Timeline, Team Roles and Risks

**Purpose of this file:** a realistic plan for getting the project built and presented. The plan assumes 6 weeks; compress it using the MVP cut line in section 4 if your deadline is shorter.

---

## 1. Six-week plan

| Week | Focus | Deliverables | Done when |
|---|---|---|---|
| 1 | Setup, data and warehouse | Repository, environment, `schema.sql`, ETL for prices, data-quality checks, warehouse populated | A full refresh runs end to end and all hard checks pass |
| 2 | Features, OLAP, regimes | Indicators in `fact_indicator`; EDA notebook; OLAP queries with screenshots; K-Means regimes with elbow and silhouette plots | Regime profile table looks sensible; 5 OLAP operations demonstrated |
| 3 | Classification | Feature matrix, target, 4 classifiers, walk-forward predictions, confusion matrices | Out-of-sample accuracy compared with the majority baseline |
| 4 | Strategy and backtest | Strategy rules, backtest engine with costs, baselines B1 to B4, metrics | Equity curves and results table for the validation year |
| 5 | Integration and paper trading | Test-period evaluation (once), ablations, paper-trading job, dashboard, leakage tests | Daily job runs by itself; dashboard shows signals and results |
| 6 | Report and viva | Report, slides, demo rehearsal, viva practice | Report reviewed by every team member; demo runs from a clean clone |

Keep roughly the last 3 to 4 days of week 6 as buffer. Something will slip.

---

## 2. Team roles

### Team of 4

| Role | Owns |
|---|---|
| A. Data and warehouse | Schema, ETL, data quality, OLAP queries, `03` |
| B. Mining | Features, regime clustering, classifiers, evaluation, `04` |
| C. Strategy and backtest | Strategy rules, backtest engine, baselines, metrics, `05`, `06` |
| D. Integration and communication | Repository setup, paper-trading job, dashboard, tests, report, slides |

### Smaller teams

| Team size | How to combine |
|---|---|
| 3 | D's work is split: A takes the dashboard, C takes the paper-trading job, everyone contributes to the report |
| 2 | Person 1 does A and D; person 2 does B and C |
| Solo | Follow the weeks in order and use the MVP cut line |

Everyone must be able to explain every part in the viva. Hold a 30-minute walkthrough at the end of each week where each owner explains their part to the others.

---

## 3. Milestone checklist

- [ ] Week 1: warehouse loaded, schema diagram drawn
- [ ] Week 2: OLAP screenshots; regime chart
- [ ] Week 3: model comparison table
- [ ] Week 4: first equity curve versus Nifty 50
- [ ] Week 5: test-period results; ablation table; dashboard demo
- [ ] Week 6: report, slides and a rehearsed demo

---

## 4. MVP cut line (3 to 4 weeks)

**Must keep:**

- Star schema, ETL and data-quality checks
- At least 5 OLAP queries (roll-up, drill-down, slice, dice, pivot)
- K-Means regimes
- Two classifiers (Logistic Regression and Random Forest)
- One backtest with costs and baseline B1
- Report with limitations

**Can drop:**

- Dashboard (use notebooks and saved figures)
- XGBoost and the Decision Tree comparison
- The scheduled paper-trading job (run the paper logic as a replay of the final weeks instead)
- Baselines B3 and B4, association rules, per-regime models

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| yfinance rate limits or format changes | Medium | High | Cache raw files in `data/raw`; keep the NSE CSV fallback; do the full download early in week 1 |
| Model accuracy is only about 50% | High | Medium | This is normal for daily direction. Frame it honestly, compare with the majority baseline, and emphasise regime risk control and the evaluation method |
| Strategy underperforms buy and hold | High | Medium | Report drawdown and Sharpe; present ablations; a well-explained negative result is acceptable |
| Look-ahead bug inflates results | Medium | High | The automated test; the leakage checklist; a second team member reviews feature code |
| Scope creep (sentiment, deep learning) | High | Medium | Stick to the MVP; list extras as future work |
| Ticker changes or missing history | Medium | Low | ETL warns on empty symbols; replace or drop the stock and document it |
| Time overrun | Medium | High | Weekly milestone checks; apply the MVP cut line by the end of week 3 if behind |

---

## 6. Definition of done

- The pipeline runs from a clean clone with the documented commands.
- `pytest` passes, including the look-ahead test.
- The results table in the report matches the saved CSV files.
- The leakage checklist in `06` is fully ticked.
- The report states survivorship bias, the number of experiments tried and the limitations.
- Every team member can answer the viva questions in `10`.
