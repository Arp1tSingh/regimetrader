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

---

## 3. Open questions (agent: do not block on these; use the stated default)

| ID | Question | Default used |
|---|---|---|
| Q1 | Final value of k (4 or 3) | 4 unless the k-selection review in T3.2 shows meaningless fragments; record the choice in `PROGRESS.md` |
| Q2 | Include association rules, V3 (relative target), V4 (per-regime models), yearly regime refit | Stretch phase only (`BUILD_PLAN.md` Phase 9), after everything else is done |
| Q3 | Risk-free rate | 0.0 |
| Q4 | Email alerts | Console and `logs/alerts.log` only; email only if SMTP variables are set |
