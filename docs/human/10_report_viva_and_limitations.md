# 10. Report, Demo, Viva Preparation and Limitations

**Purpose of this file:** everything needed for the submission stage: report outline, demo script, likely viva questions with answers, how to align the project with past papers, limitations and future work.

---

## 1. Report outline

| Section | Content | Length |
|---|---|---|
| Title, abstract | Problem, method, headline result in about 150 words | 1/2 page |
| 1. Introduction | Motivation and the problem (see `01`) | 1 page |
| 2. Background | Efficient market hypothesis, technical indicators, market regimes, K-Means, classification. Cite 5 to 8 sources you have actually read | 1 to 2 pages |
| 3. Objectives and scope | Objectives O1 to O6; scope and assumptions | 1 page |
| 4. System design | Architecture diagram, components, technology stack (see `02`) | 2 pages |
| 5. Data warehouse | Star schema diagram, DDL highlights, ETL steps, data-quality checks (see `03`) | 3 pages |
| 6. OLAP analysis | Roll-up, drill-down, slice, dice, pivot results with 3 to 4 findings | 2 pages |
| 7. Data mining | Preprocessing, features, regime clustering, classifiers, evaluation (see `04`) | 4 pages |
| 8. Strategy and backtest | Rules, cost model, baselines, ablations (see `05`, `06`) | 3 pages |
| 9. Results and discussion | Results table, subperiods, regime breakdown, confidence intervals | 3 pages |
| 10. Limitations and threats to validity | See section 5 below | 1 page |
| 11. Conclusion and future work | What was achieved and what comes next | 1 page |
| References, appendix | DDL, config, repository structure, trial log | As needed |

The report can be written as you go: section 5 after week 1, 6 and 7 after weeks 2 and 3, and so on.

---

## 2. Demo script (7 minutes)

| Minute | Show | Say |
|---|---|---|
| 0 to 1 | Architecture diagram | The problem, and the one-sentence idea (regime-aware prediction) |
| 1 to 2 | Warehouse tables and schema diagram | Star schema, grain, why a warehouse and not just CSV files |
| 2 to 3 | Run two OLAP queries live (roll-up and drill-down) | What the sector and quarter patterns show |
| 3 to 4 | Regime chart over the Nifty 50 price | How K-Means identified calm, volatile and falling markets |
| 4 to 5 | Today's signals table from the dashboard | What the bot suggests and why the regime limits exposure |
| 5 to 6 | Backtest equity curves and results table | Strategy against baselines, after costs |
| 6 to 7 | Limitations slide | What the project does not prove |

Prepare a recorded fallback of the demo in case the network fails, and run the demo from a clean clone at least once before the presentation.

---

## 3. Viva questions and short answers

| # | Question | Short answer |
|---|---|---|
| 1 | Why a data warehouse instead of flat files? | A warehouse gives a structured, validated, queryable history with defined dimensions and grain. It supports OLAP, keeps every run traceable, and separates data preparation from analysis |
| 2 | What is the grain of `fact_price`? | One instrument on one trading day |
| 3 | Difference between OLTP and OLAP? Where is each here? | OLTP handles many small transactions on current data; OLAP supports analytical queries over historical data. The warehouse queries are OLAP; the source API is the operational system |
| 4 | Why a star schema rather than a snowflake? | Fewer joins and simpler, faster analytical queries. Sector and industry are denormalized into `dim_stock`. A snowflake would save a little storage but add a join |
| 5 | What are the ETL steps in your project? | Extract OHLCV from the API into staging; transform by cleaning, deduplicating, aligning to the trading calendar and validating; load with upserts and run quality checks |
| 6 | Name the OLAP operations and give an example from your project | Roll-up (quarter to year), drill-down (sector to stock), slice (year 2022), dice (Banking and IT, 2023 to 2025, quarters 1 and 2), pivot (sector by year) |
| 7 | Classification versus clustering in your project? | Clustering is unsupervised and labels market regimes with no target. Classification is supervised and predicts next-day direction using the known up/down label |
| 8 | Why standardize before K-Means? | K-Means uses Euclidean distance, so features with large numeric ranges would dominate. Standardizing gives each feature equal weight |
| 9 | How did you choose k? | Elbow plot of inertia and silhouette score for k = 2 to 6, plus whether the clusters have interpretable return and volatility profiles |
| 10 | Limitations of K-Means here? | Assumes roughly spherical clusters, depends on initialization (mitigated with multiple starts), needs k chosen in advance, and gives hard labels with no sense of regime persistence (mitigated by smoothing) |
| 11 | Why use Random Forest or XGBoost over Logistic Regression? | They capture non-linear relationships and feature interactions. Logistic Regression is kept as an interpretable baseline, and the Decision Tree shows explainable rules |
| 12 | Your accuracy is 56%. Is that useful? | It is above the 51% majority baseline, which is a realistic edge for daily direction. Whether it is useful depends on costs, which is why the project evaluates the trading strategy after costs rather than accuracy alone |
| 13 | What is look-ahead bias and how did you prevent it? | Using information that was not available at decision time. We use a chronological split, fit scalers and K-Means on training data only, execute at the next open, and have an automated test that checks features do not change when future rows are added |
| 14 | What is overfitting here and how did you control it? | A model or strategy that fits historical noise. We use walk-forward validation, a final test set used once, a small parameter grid, simple models, and we report every configuration tried |
| 15 | Explain Sharpe ratio and maximum drawdown | Sharpe is annualized excess return divided by volatility. Maximum drawdown is the largest peak-to-trough fall in equity |
| 16 | Why do transaction costs matter so much? | A 55% hit rate with symmetric 1% wins and losses earns 0.10% per trade before costs, but a round trip costs about 0.30%, so the net result is negative. The strategy therefore limits turnover |
| 17 | What is survivorship bias in your project? | The stock universe is today's Nifty 50 list, which excludes stocks that dropped out earlier. This flatters historical results, so we state it and avoid claiming future returns |
| 18 | What happens if the market behaves differently from the training period? | This is regime change or concept drift. The regime layer helps by reducing exposure in unfamiliar, volatile conditions, and yearly walk-forward retraining adapts the model. It cannot remove the risk |
| 19 | Why not use an LSTM or deep learning? | The dataset is small and noisy, the syllabus focus is classical mining, and simpler models are easier to explain and validate. It is listed as future work |
| 20 | How does this meet a real-life need? | It supports a daily investor decision workflow: a regime flag, a ranked watchlist and risk-scaled position sizes, all produced from a warehouse that retains every signal and trade |

---

## 4. Aligning with your past papers

Past-paper question styles vary by university. Use this table as a starting map, then check it against the actual papers your course has set and note the marks each type carries.

| Typical exam question type | Where it appears in this project | Practice material |
|---|---|---|
| Draw and explain a star schema for a given scenario | `fact_price` with `dim_date` and `dim_stock` | `03`, section 2 |
| Explain the ETL process with an example | The ETL steps for market data | `03`, section 4 |
| Differentiate OLTP and OLAP; list OLAP operations with examples | Viva questions 3 and 6 | `03`, section 6 |
| K-Means numerical | Regime assignment worked example | `04`, section 5 |
| Evaluate a classifier from a confusion matrix | Worked example with accuracy, precision, recall and F1 | `04`, section 7 |
| Apriori numerical: support, confidence, lift | Sector co-movement example | `04`, section 8 |
| Compare classification algorithms | Four-model comparison | `04`, section 6 |
| Data preprocessing techniques | Missing values, winsorizing, scaling, encoding, discretization | `04`, section 2 |
| Case study or application of data mining | The whole project | `01`, `10` |

Practice method: take 3 to 4 past questions per topic, solve them on paper, then re-solve them using this project's data as the scenario. If you share your syllabus and past papers, the question mapping and the viva list can be tailored to them.

---

## 5. Limitations and threats to validity

| Limitation | Effect | How the report handles it |
|---|---|---|
| Markets are close to efficient and noisy | Predictable edge is small | Report realistic accuracy and compare with baselines |
| Survivorship bias (frozen Nifty 50 list) | Results look better than a real investor would have seen | State it explicitly |
| Few bear-market episodes in the history | Bear regime and its rules are learned from little data | Show the regime sample counts; avoid strong claims |
| Regimes are labelled with hindsight-style clustering | There can be a lag in recognizing a regime change | Smoothing and frozen centres; discuss the lag |
| Free, unofficial data | Possible errors and revisions | Data-quality checks; NSE fallback |
| Adjusted prices used for fills | Fill prices differ slightly from real traded prices | Mention as a simplification |
| Simplified cost and slippage model | Real costs vary with order size and conditions | Make them configurable and report with and without costs |
| No market impact or liquidity limits | Fine for small capital in liquid stocks, not for large funds | State the capital assumption |
| Repeated experimentation | Selection bias | Trial log; test set used once; bootstrap intervals |
| Paper trading is not live trading | Real execution has delays and rejections | State it; no real-money claims |

---

## 6. Future work

- Add news and sentiment features (see use case C).
- Replace K-Means with Gaussian Mixture or Hidden Markov Models for probabilistic, persistent regimes.
- Train separate models per regime, with enough data.
- Portfolio construction beyond equal weights (risk parity, volatility targeting).
- Add pairs-trading or sector-rotation modules using the association-rule results.
- Explainability with SHAP values.
- Intraday data and a broker sandbox for realistic execution testing.

---

## 7. Ethics and disclaimer

State clearly in the report and the demo: this is an academic prototype, not investment advice; backtest results are not predictions of future returns; no real money is used; and any live use would require reviewing current SEBI and broker rules and obtaining proper authorization.
