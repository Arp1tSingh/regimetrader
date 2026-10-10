# DECISIONS

Authoritative list of locked decisions and of every place where this agent-ready pack resolves an ambiguity or adds something the original blueprint left open. **On any conflict between files, this file wins.** Append new decisions at the bottom with the next ID; never silently change a spec.

---

## 1. Locked decisions (from the original blueprint)

| ID | Decision | Choice |
|---|---|---|
| L1 | Warehouse engine | SQLite (PostgreSQL optional, port notes in `03`) |
| L2 | Data source | yfinance, NSE CSV downloads as fallback |
| L3 | Universe | Nifty 50 constituents frozen at project start, plus `^NSEI` and `^INDIAVIX` |
| L4 | Prediction target | Next-day direction (up or down) |
| L5 | Regime method | K-Means on market-level features, k = 4 (k = 3 if k = 4 fragments; report both) |
| L6 | Models | Logistic Regression, Decision Tree, Random Forest, XGBoost; one pooled model across stocks |
| L7 | Validation | Chronological split, yearly walk-forward with 5-day embargo, test used once |
| L8 | Trading | Long-only, daily decisions, next-open execution, paper trading only |
| L9 | Success | Better risk-adjusted results than baselines, or an honestly explained negative result |

---

## 2. Resolved ambiguities and additions

Format: **ID. Topic.** *Original:* what the blueprint said or left open. *Now:* what the spec requires. *Why.*

**R1. File `07_project_structure.md` was missing.** *Original:* referenced everywhere, not supplied. *Now:* rewritten from the references (folder layout, config, module contracts, CLI, tests) as `07_project_structure.md` plus `TESTS.md`. *Why:* the agent needs a single structure to build against.

**R2. Feature names versus warehouse columns.** *Original:* the feature catalogue (`ret_1d`, `px_to_sma20`, ...) does not match `fact_indicator` columns (`return_1d`, `sma_20`, ...). *Now:* `04` section 2 maps every model feature to its source columns and formula; derived features are computed in `src/features/panel.py`, not stored. *Why:* removes guesswork.

**R3. `fact_signal` semantics.** *Original:* PK has no `run_id`, yet backtest, ablations and paper trading all "write fact_signal". *Now:* training writes `prob_up` for every model (`signal = 0`, `target_weight = 0.0`); only the **primary strategy run** (`backtest_main`, then `paper_live`) updates `signal` and `target_weight` for the selected model; ablations and baselines do not touch the table. *Why:* keeps the original schema and PK, avoids collisions.

**R4. Pending orders for paper trading.** *Original:* "store pending orders", but no table. *Now:* new table `pending_target` (`sql/schema.sql`). Holdings and cash are derived from `fact_trade` and `fact_portfolio_daily`; no separate positions table. *Why:* the daily job runs as separate processes and needs state.

**R5. Stop-loss mechanics.** *Original:* "exit if close is 7% below entry". *Now:* entry price = quantity-weighted average fill price of the open position. Checked at each close; a breach forces an exit at the next open (`force_exit`). After a stop-out the symbol cannot be re-entered for `cooldown_days` (default 5 trading days, **added** to prevent same-week whipsaw). *Why:* needs a precise, testable rule.

**R6. Regime transitions.** *Original:* text says positions are "trimmed to the smaller weight; the lowest-probability stocks are sold first". The reference code does not do that: it keeps every held stock with `prob_up >= exit_prob` and lowers each slot weight to `exposure / slots`. *Now:* the **code behaviour is the spec** (proportional trim, no forced sale of the lowest-probability names). *Why:* simpler, matches the reference implementation, and the 25% band still avoids pointless trades.

**R7. Entry thresholds as base plus offset.** *Original:* thresholds 0.55 / 0.60 / 0.65 per regime, grid {0.55, 0.60}. *Now:* `entry(regime) = entry_base + offset(regime)` with offsets 0 / 0.05 / 0.10; the tuning grid varies `entry_base` in {0.55, 0.60} and `slots` in {5, 10}. If the calibration check in `05` section 6 fails, 0.52 (rank-based selection) is appended to that model's `entry_base` grid. *Why:* one parameter to tune, consistent regime ordering.

**R8. Order execution details.** *Original:* sizing rule only. *Now (`06` section 3):* sells execute first, then buys in descending `prob_up`; a buy is capped by available cash (floor to whole shares); a trade is skipped if the symbol has no bar that day (the next decision recomputes targets from scratch). *Why:* cash must never go negative.

**R9. Index and VIX handling.** *Original:* "indicators for every instrument", volume checks generic. *Now:* indicators are computed for `EQUITY` and `INDEX` only; volume-based soft checks apply to `EQUITY` only; `dim_stock.sector` for index and VIX is `'INDEX'` / `'VOLATILITY'`. *Why:* VIX has no meaningful indicators; index volume is 0.

**R10. Short-history stocks.** *Original:* objective O1 says 10+ years for 50 stocks. *Now:* some constituents listed recently; they are loaded from their first available bar, the panel is unbalanced, and DQ-04 measures coverage from each stock's first bar. Report the number of stocks with less than 10 years. *Why:* cannot fabricate history.

**R11. Regime model fitting and warm-up.** *Original:* fit on training period, centres frozen; drawdown over 252 days. *Now:* fit once on dates `<= splits.train_end`, centres frozen for all later dates (`regime.refit: none`). `mkt_drawdown` uses `rolling(252, min_periods=20)` so the regime exists from about day 21. *Why:* strict 252-day warm-up would drop the first year.

**R12. VIX availability.** *Original:* "if VIX unavailable, drop it". *Now:* `regime.use_vix: auto` uses VIX only if coverage on the training period is at least `min_vix_coverage`; the decision is logged. When dropped, drop it from the regime features **and** the classifier features (feature 15). *Why:* deterministic rule.

**R13. Test-period guard (added).** *Now:* the one-time test evaluation goes through `src.backtest.final_test`, which writes `reports/results/TEST_RUN_LOCK.json`; a second run is refused unless `--allow-rerun "<reason>"` is given, which appends to `test_run_log.jsonl`. All development commands default to the validation period. *Why:* enforces "test used once" mechanically.

**R14. Synthetic data mode (added).** *Now:* `src/etl/synthetic.py` generates a seeded, regime-switching market so every test and `make smoke` runs offline. *Why:* agents often have no network; tests must be deterministic.

**R15. Baseline definitions.** *Original:* one-line descriptions. *Now (`06` section 6):* B1 and B2 and B3 use the same execution engine and cost model as the strategy; B4 is a **symbol-permutation null**: relabel the probability series of each stock to a different stock (one fixed permutation per simulation), then run the full strategy. This preserves exposure, regime behaviour and turnover while breaking the link between a stock's predictions and its own returns. *Why:* "same exposure and turnover" has no other precise implementation.

**R16. Model selection and tuning.** *Original:* "pick the final model and strategy settings on validation". *Now:* grid = models x `entry_base` x `slots` (16 trials by default), each logged; select by validation Sharpe, tie-break lower turnover; write the result to `paths.frozen_config`. *Why:* explicit, reproducible, counted.

**R17. Model versions.** *Now:* `dim_model.model_name` in {logreg, tree, rf, xgb}; `version = 'wf<year>'` means "trained on data before that year minus the embargo, used to predict that year". The daily job trains the missing `wf<current year>` model on the first run of a new year. *Why:* ties yearly retraining to a stored row.

**R18. Period convention.** *Now:* every backtest starts flat; the first decision is made at the close of the first day of the period and filled at the next open. Open positions at the end are marked at the last close, not liquidated. Baselines use the same convention. *Why:* identical conventions make results comparable.

**R19. Database access.** *Now:* SQLAlchemy 2.x engine with raw SQL via `text()` and named parameters; foreign keys on, WAL mode. *Why:* original said "SQLite via sqlalchemy" without detail.

**R20. `is_month_end`.** *Now:* 1 on the last **trading** day of each calendar month present in the loaded data, recomputed after each load. *Why:* B2 rebalances monthly.

**R21. Risk-free rate.** *Now:* default `backtest.risk_free_annual: 0.0` (original suggested 0% as acceptable). Reported in every results file. *Why:* must be stated.

**R22. Scheduling.** *Now:* the agent writes `deploy/cron.example` and `deploy/github_actions_daily.yml`; it does **not** install a schedule. *Why:* deployment is the human's call.

**R23. Test-year predictions exist before the test, but are not scored.** *Original:* walk-forward trains and "evaluates" every year in one pass, which would show test-year accuracy during development and break "test used once". *Now:* `train.py` generates out-of-sample `prob_up` for all years but evaluates validation rows only; `final_test.py` scores the test years after the freeze (`04` section 7, `06` section 8). *Why:* the freeze must come before anyone sees test-period numbers.

**R24. Config precedence and the frozen config.** *Original:* not specified. *Now:* base, then the `--config` override (skipped when it is the base), then the frozen config last, limited to `strategy.selected_model`, `entry_base`, `slots` (`07` section 3). *Why:* otherwise the Makefile's `--config config/config.yaml` would silently undo the tuning result.

**R25. Paper-trading state is recoverable from the warehouse.** *Now:* `fact_trade.reason` (ENTRY, REBALANCE, EXIT, STOP_LOSS) added; average-cost method (not FIFO) defined; each paper date commits in one transaction. *Why:* cooldowns, stop-loss triggers and pending targets must survive a restart, and a crash must never look like a decision to go to cash.

**R26. Start and guards of `paper_live`.** *Now:* a new `paper_live` run starts at the latest trading date (flat), is refused until the final test has produced `TEST_RUN_LOCK.json`, and writes `fact_signal` only for dates after the lock's `data_end_date`. Replays (`paper_replay`) never touch `fact_signal`. *Why:* prevents back-filling a fake live history and collisions with `backtest_main`.

**R27. What the "test once" rule covers.** *Now:* it covers scoring of models and strategies (classifier metrics, importance, backtests, tuning). Market-only descriptive tables (sector returns, OLAP 01 to 07, `regime_counts.csv` `days_all`) may span all dates. Regime profile and k-selection use the training period. *Why:* a clear, enforceable line.

**R28. Trial-log semantics.** *Now:* upsert on (`kind`, `run_id`, `model`, `params_json`, `period`); calibration rows are not counted; only validation-year model trials are logged in development (`07` section 4). *Why:* idempotent reruns must not inflate the reported experiment count.

**R29. Weight-provider `None` and baseline start.** *Now:* `None` means "no orders at the next open" (pending is cleared); B1 returns weight 1.0 on the first decision then `None`; B2 returns `1/N` on the first decision and every month-end. *Why:* identical start convention for all baselines (R18) and no daily churn in B1.

**R30. Code layout additions.** *Now:* `src/backtest/strategies.py` holds the weight providers V0 to V2 and B1 to B3; `baselines.py` runs them and B4; `tune.py` writes `tune_grid_validation.csv`; `baselines.py` writes `summary_validation.csv`; `final_test.py` is not guarded by running pytest (the `make final-test` target depends on `make check`). *Why:* removes ownership overlaps and a recursion between the guard and its own test.

### Added during the build (Phase 5)

**R31. Strategy-tuning eligibility rule (distribution only).** *Original:* `05` section 6 assumed an absolute entry grid would work and that a rank fallback would rescue a poorly calibrated model. *Now:* a model is **eligible for strategy tuning** if and only if both hold on the validation year's `prob_up` distribution:
1. no single `prob_up` value holds more than **20%** of rows, and
2. `p95 - p50 >= 0.01`.

An ineligible model is excluded from the tuning grid only. It **remains** in the classifier comparison, `classifier_metrics.csv`, the trial log's `model_year` rows, and the validation diagnostics. *Written:* after seeing the decision tree's `prob_up` distribution and **before any strategy outcome for that model** — no backtest, Sharpe, return or accuracy was consulted when fixing these numbers. Returns were not used at any point: both conditions read only the distribution of predictions. *Why:* the rule exists because a model with a large mass of tied probabilities defeats any absolute threshold. With a tied mass at or below the threshold, nearly every stock qualifies on nearly every day, and `target_weights` then breaks the tie by symbol A to Z, producing an alphabetical portfolio that looks like a strategy but carries no information. Measured on the validation year, the tree has **3 distinct `prob_up` values across 12,044 rows with 76.4% tied at one value**; the other three models have ~12,000 distinct values with 0.01–0.02% tied. Both thresholds sit in wide empty gaps (no model is within an order of magnitude of either boundary), so the rule is not knife-edge.

**R32. Effect of ineligibility.** *Now:* the decision tree fails both conditions (76.4% tied, `p95 - p50 = 0.0000`) and is therefore **excluded from strategy tuning**. It is still trained, still written to `fact_signal`, still scored, and still reported in the classifier comparison and the validation diagnostics, because excluding it from tuning is not a claim that it is a poor classifier — it is a claim that its probabilities cannot drive a threshold-based portfolio. *Why:* keeps the honest negative result about the classifier separate from the practical limitation of its probability scale.

**R33. Per-model threshold method.** *Now:* for each **eligible** model, the third entry base is chosen from the model's own validation distribution, never from returns. Keep `0.55` and `0.60` for every eligible model. The third entry base is:
- **`0.52`** (the shared absolute fallback) when the model reaches `0.52` on at least **50%** of validation days; method recorded as `absolute`;
- otherwise the model's own validation **p80** of `prob_up`; method recorded as `percentile_p80`.

Measured: xgb reaches 0.52 on 60.0% of days, so it uses the absolute grid `{0.55, 0.60, 0.52}`. logreg (15.5%) and rf (13.9%) do not, so their third entry base is their validation p80 (0.5024 and 0.5044 respectively). `frozen_config`'s `meta.method` records the method **per model**, not once for the run. *Why:* a single absolute threshold cannot fit probability distributions on four different scales; p80 is a distributional quantile, so using it cannot leak outcome information.

**R34. Tie-break share must be reported and can disqualify a run.** *Now:* every tuning and baseline run reports the share of newly entered positions whose selection was decided by the symbol tie-break rather than by probability. If that share exceeds **5%**, the run is flagged in `tune_grid_validation.csv` (`tie_break_flagged = 1`) and **excluded from selection**, with a note in the trial log. Non-trading runs and tie-break-heavy runs are both excluded from selection but both remain in the CSV and the trial log. *Why:* makes the hazard in R31 measurable at run time instead of only being argued from the distribution.

**R35. Validation diagnostics, reported and not used.** *Now:* for all four models on the validation year only, `src/models/diagnostics.py` computes ROC-AUC, the per-day top-quintile minus bottom-quintile next-day return spread, and the accuracy edge over the majority baseline, each interval by a bootstrap **clustered on day**. These are written to `reports/results/validation_diagnostics.csv` and are **not** read by `tune`, so they cannot affect eligibility or selection. *Why:* the cross-sectional spread is the only evidence that would justify calling a probability a *ranking* signal, and it must be measured rather than asserted. Clustering by day is required, not stylistic: 50 stocks on one date share the market's move, so a row-level bootstrap would understate every interval and manufacture significance (`test_day_clustering_widens_the_interval_versus_row_resampling`).

**R36. The majority baseline must be measured on the same rows as the accuracy.** *Original:* an earlier note in this file quoted the decision tree as **+2.44 points** over a "majority baseline of 0.5072". *Now:* that comparison was **wrong** and the figure is withdrawn. The tree's validation accuracy is 0.5316, but the validation-year majority baseline is **0.5315**, so its true edge is **+0.0001**. The 0.5072 was a *training-period* baseline (the training-period value is 0.5067); comparing a validation accuracy against a training baseline manufactures 2.5 points of edge out of a difference between two periods. `classifier_metrics.csv` is now written by `train.py` (`04` section 7) and every baseline is computed on exactly the rows whose accuracy it is compared against. *Why:* a baseline is only a baseline relative to the same sample. Corrected result: **all four models are at or below the validation majority baseline** — logreg −0.0335, rf −0.0190, xgb −0.0256, tree +0.0001.

**R37. Baselines must be the benchmarks the spec names.** *Original:* the B1, B2 and B3 providers were placeholders. B1 and B3 both bought the alphabetically first symbol in `DayContext.signals`, and B2 weighted the first `slots` names at `1/slots`. *Now:* B1 buys `data.index_symbol` and **raises** if it has no bar, rather than silently substituting an equity; B2 weights `1/N` over every stock with a bar that day and may hold the whole universe; B3 holds every stock with `sma_20 > sma_50` at `1/N_universe`, rest in cash, recomputed daily. Three supporting fixes were required. (1) `_load_market` filtered prices to `instrument_type = 'EQUITY'`, so `^NSEI` was unreachable and B1 traded nothing; the index is now included explicitly, and only it. (2) `DayContext.signals` is the tradable universe with `prob_up` where the model predicted one, so the index is present with NaN — and NaN never qualifies for entry, so V0 to V2 still cannot select it. (3) `fact_indicator.sma_20`/`sma_50` were not selected, so B3 had no signal and traded nothing. *Why:* a baseline that quietly holds something else is not a benchmark; results measured against it would be wrong in an invisible way.

**R38. The equity curve must be indexed by date.** *Original:* `run_backtest` returned the equity frame with a `RangeIndex`. `summarize` measures elapsed time from the index, so `cagr` returned **NaN for every run** — all 18 grid trials and all baselines reported a blank CAGR while every other metric looked healthy and nothing failed. *Now:* the equity frame is indexed by a `DatetimeIndex` built from `date_key`. *Why:* a metric that is silently undefined across an entire results set is worse than one that raises. `test_cagr_needs_a_datetime_index` guards it.

### Added after the first tuning run (T5.6 run 2)

> **Disclosure, stated once and not repeated in each entry.** The rules in R39 to R42 were introduced **after** the first grid had already run and after its winner had been seen. That winner — rf / 0.55 / 5 slots, Sharpe 2.5679 — was a **one-trade run**. The rules below therefore cannot be described as having been chosen blind. They were fixed from first principles, and the grid is re-run in full, but the honest statement is that the criterion changed because of an observed degenerate winner. Run 1's output is preserved unaltered in `reports/results/tune_grid_validation_run1.csv`, its 18 rows stay in the trial log and stay counted, and both grids are reported.

**R39. Activity filter before selection.** *Now:* a configuration is eligible for selection only if, on the validation period, it completed **at least 30 round trips** and held **mean exposure of at least 10%**. *Justification from first principles, fixed now:* 30 round trips is the smallest sample in which a trade-level hit rate has a usable standard error — at 30 observations the binomial standard error near 50% is about 9 percentage points, so anything smaller cannot distinguish a 55% hit rate from a coin flip. Below 10% mean exposure the strategy is mostly cash and its Sharpe is computed on a handful of active days. *Note, explicitly:* this filter does **not** claim that low exposure inflates Sharpe. Sharpe is a ratio and exposure scales its numerator and denominator together; the reason a one-trade run wins is that a single lucky window is a small sample, not that cash mechanically raises the ratio. *Why:* a Sharpe computed from one trade is a sample statistic, not evidence.

**R40. Null filter against the B4 permutation null.** *Now:* among activity-eligible configurations, take the **top 3 by Sharpe** and run the B4 symbol-permutation null for each, at **100 simulations** with a fixed seed from `project.seed`. A shortlisted configuration is **selectable only if its Sharpe exceeds the 90th percentile of its own null** — that is, only if it beat what meaningless predictions achieve under the same regime rules, exposure and turnover. Nulls are run under the same configuration, not one shared null. *Why:* a Sharpe is meaningless without knowing what an uninformative strategy scores under identical conditions.

**R41. `no_edge_default` when nothing survives.** *Now:* if **no** configuration passes both R39 and R40, the frozen config is recorded as **no demonstrated edge**. The model is then the **highest-validation-ROC-AUC model among the distribution-eligible models (R31)**, with the spec's starting strategy settings: `entry_base` 0.55 with its fallback rule, `slots` 10. `meta.method` is set to **`no_edge_default`**. *Why:* if nothing demonstrably beats an uninformative strategy, the honest freeze names the best classifier and the untouched starting settings rather than promoting a configuration that failed its own null.

**R42. Selection by Sharpe alone is retired.** *Now:* selection by Sharpe on its own is **not used anywhere** — not in `tune`, not in the frozen config, not in `RESULTS.md`. Selection requires passing R39 and R40 first; among survivors, highest Sharpe with the turnover tie-break still orders them, but survival is a precondition rather than a filter applied afterwards. *Why:* the first run demonstrated that an unfiltered Sharpe ranking promotes a degenerate configuration.

**R43. Trial accounting across two grids.** *Now:* run 1's 18 configurations **remain in the trial log and remain counted**. Run 2 adds its own 18, and each B4 null simulation is logged as `kind = "other"`. The reported total is every row with `kind != "calibration"`. A `kind = "other"` row noting that the selection rule was revised after run 1 is logged so the trial log itself carries the history, not just the CSV files. *Why:* a revised protocol that silently replaces the first attempt's results is indistinguishable from having tuned until something looked good.

---

## 3. Open questions (agent: do not block on these; use the stated default)

| ID | Question | Default used |
|---|---|---|
| Q1 | Final value of k (4 or 3) | 4 unless the k-selection review in T3.2 shows meaningless fragments; record the choice in `PROGRESS.md` |
| Q2 | Include association rules, V3 (relative target), V4 (per-regime models), yearly regime refit | Stretch phase only (`BUILD_PLAN.md` Phase 9), after everything else is done |
| Q3 | Risk-free rate | 0.0 |
| Q4 | Email alerts | Console and `logs/alerts.log` only; email only if SMTP variables are set |
