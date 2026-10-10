# BUILD PLAN

Ordered tasks. Each task has: dependencies, the spec sections to read, and an **acceptance check** you must run before ticking it in `PROGRESS.md`.
Notation: `03§4.2` = `docs/spec/03_warehouse_and_etl.md` section 4.2. `[REAL]` marks tasks that need network access and real market data; if the network is unavailable, mark them BLOCKED in `PROGRESS.md` (not failed), finish everything that runs on synthetic data, and tell the human.

Order of work: Phases 0 to 8 and 10 are required. Phase 9 is stretch (only after everything else passes).
Principle: **every phase leaves the repo in a working state** and extends `make smoke` (the synthetic end-to-end run) by the commands it adds.

---

## Phase 0: Scaffold

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T0.1 | Create the repo skeleton: folders, empty `__init__.py` files, `.gitkeep` files, `git init` if needed. `.gitignore` is already supplied; keep it | none | `07§1` | `tree` matches `07§1`; `git status` clean after first commit |
| T0.2 | `Makefile`, `pyproject.toml` and `requirements.txt` are already supplied: check they match `07§2` and `07§7`, then create the virtualenv and install | T0.1 | `07§2`, `07§7` | `make setup` succeeds; `python -c "import pandas, sklearn, xgboost, yfinance, sqlalchemy, yaml, streamlit"` exits 0 |
| T0.3 | `src/common/config.py`: `load_config()` with `extends`, frozen-config and override precedence; `config_hash()` | T0.2 | `07§3` | Test: loading `config/config.smoke.yaml` returns smoke paths and inherits base values; unknown keys do not crash; hash is stable |
| T0.4 | `src/common/`: `db.py` (engine, `init_db`, `read_sql`, `execute_many`), `log.py`, `trial_log.py`, `manifest.py`, `seeds.py`, `dates.py` (`to_date_key`, `from_date_key`) | T0.3 | `07§4` | Unit tests for each utility pass; `init_db` applies `sql/schema.sql` twice without error and with foreign keys on |
| T0.5 | `tests/test_schema.py`: every table in `03§2` exists with the documented primary key; `dim_regime` has 4 seed rows | T0.4 | `03§2` | `make check` passes |

**Gate P0:** `make check`.

---

## Phase 1: Warehouse and ETL

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T1.1 | `src/etl/universe.py`: `load_universe(cfg)` reads `data.universe_file` (lines starting `#` skipped), validates it (unique symbols; allowed sector labels; `Banking` and `IT` present when the file has at least 40 rows) and appends the index and VIX rows. Tested on a small fixture CSV; needs no network | T0.4 | `03§3` | Fixture tests pass; the index and VIX rows have types `INDEX` and `VOLATILITY` |
| T1.2 | `src/etl/synthetic.py`: seeded regime-switching market generator returning long-format OHLCV for N stocks, `^NSEI`, `^INDIAVIX`, plus a matching universe frame | T0.4 | `TESTS.md§2` | Same seed gives identical output; trading calendar has no weekends; OHLC consistent (`low <= open, close <= high`); VIX correlates with volatility; at least 4 distinct regimes visited in 8 years |
| T1.3 | `src/etl/extract.py`: sources `yfinance`, `csv`, `synthetic`; retries with waits; raw files to `data/raw`; rows to `stg_price_raw`; full and incremental start dates; VIX CSV fallback | T1.1, T1.2 | `03§4.1` | Tests with mocked yfinance failures (retry then continue); synthetic source returns expected columns; raw CSV cache written |
| T1.4 | `src/etl/transform.py`: normalise, deduplicate, trading calendar from `^NSEI`, forward-fill at most 2 days with flag, hard rejects, soft flags, `dim_date` frame | T1.3 | `03§4.2` | All rule tests in `TESTS.md§3` pass (timezone, ffill limit, duplicates, rejects, flags, EQUITY-only volume check) |
| T1.5 | `src/etl/load.py`: one-transaction upsert of dims then `fact_price`, `etl_run_log`, `is_month_end` recompute, rollback on error | T1.4 | `03§4.3` | Loading twice leaves identical row counts and checksums; injected failure rolls back and logs FAILED |
| T1.6 | `src/etl/quality.py`: checks DQ-01 to DQ-07 returning structured results; hard failures raise | T1.5 | `03§5` | Each check has a passing and a failing test (inject a bad row) |
| T1.7 | `src/etl/pipeline.py` CLI (`--full-refresh`, default incremental, `--source`, `--limit-symbols`, `--config`); add ETL step to `make smoke` | T1.6 | `07§5` | `make smoke` ETL step loads synthetic data and all hard checks pass; rerun is a no-op |
| T1.8 | `[REAL]` Acquire the universe file, then run the full refresh on real data. If `config/nifty50_symbols.csv` is missing, download the current Nifty 50 list from NSE (niftyindices.com `ind_nifty50list.csv`), map it to the format in `03§3` and put `# as of <date>` on its first line. If the download fails, **stop and ask the human** to place the file (all earlier tasks run without it) | T1.7 | `03§3`, `03§4` | 50 equity rows in the universe; `python -m src.etl.pipeline --full-refresh` exits 0; `etl_run_log` row is SUCCESS; about 52 instruments; the index has roughly 2,900 trading days since 2015; hard checks pass; record counts, failed symbols and short-history stocks in `PROGRESS.md` |

**Gate P1:** `make check && make smoke` (ETL step). `[REAL]` T1.8 for the real warehouse.

---

## Phase 2: Indicators and OLAP

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T2.1 | `src/features/indicators.py`: `compute_indicators(prices_df) -> DataFrame` with the exact `fact_indicator` columns; `src/features/builder.py --indicators` writes `fact_indicator` for EQUITY and INDEX, full recompute, upsert | T1.7 | `04§2`, `04§2.3` | Tests in `TESTS.md§4` (known values, no-look-ahead truncation test, RSI edge cases); builder rerun is idempotent |
| T2.2 | `src/olap/run.py`: runs every file in `sql/olap/`, reads `-- param:` defaults, allows `--param name=value`, builds the pivot from `olap.pivot_years`, writes CSV and prints a table; add to `make smoke` | T2.1 | `03§6` | `python -m src.olap.run --all` writes 9 result files (08 is empty until Phase 3 and 09 until the final test in Phase 6; empty is not an error); no SQL errors on synthetic data; test checks one result against a hand-computed value |

**Gate P2:** `make check && make smoke`.

---

## Phase 3: Market regimes

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T3.1 | `src/regime/regime_model.py::build_market_features`: from index indicators and VIX, with `use_vix: auto` decision logged; drawdown window per `R11` | T2.1 | `04§5.1` | Test: features for day *t* unchanged when later rows are appended; VIX dropped automatically when coverage is below threshold |
| T3.2 | `choose_k`: inertia and silhouette for k in config range on the training period only; save `reports/results/regime_k_selection.csv` and the elbow and silhouette figure; log each k as a trial | T3.1 | `04§5.2` | Files exist; fit uses only dates `<= train_end` (assert in code) |
| T3.3 | Fit scaler and K-Means on the training period, name clusters, `smooth_trailing_mode`, predict all dates with frozen centres, save `models/regime_model.joblib` with metadata | T3.2 | `04§5.3` | Worked-example test (`TESTS.md§5`); same seed gives identical clusters; smoothing is trailing (test) |
| T3.4 | Write `fact_market_regime`; write `regime_profile.csv` and `regime_counts.csv` per `04§5.4`; run the non-circular sanity checks of `04§5.4`; warn (do not force labels) if they fail | T3.3 | `04§5.4` | No NULL regime for dates with valid features; all sanity checks in `04§5.4` pass on smoke data; counts per regime reported in `PROGRESS.md` |
| T3.5 | Figure: index close with coloured regime bands (`reports/figures/regime_bands.png`); CLI `python -m src.regime.regime_model` with `--fit` (runs the k-selection, then fits), `--k-select`, `--k N`; add to `make smoke` | T3.4 | `06§10` | CLI exits 0 on smoke data |

**Gate P3:** `make check && make smoke`. `[REAL]` rerun T3.x on the real warehouse; record the chosen k (Q1 in `DECISIONS.md`).

---

## Phase 4: Direction classifiers

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T4.1 | `src/features/panel.py`: `FEATURE_COLS_ALL`, `get_feature_cols()`, `build_panel()` (cache rule in `07§6`) (stock features, relative strength, market features, regime one-hot, target), cached to `data/processed/panel.parquet` | T3.4 | `04§2`, `04§4` | Look-ahead test (`TESTS.md§4`); target is the only column using day *t+1*; no NaNs in training rows; warm-up rows dropped; panel has the documented columns |
| T4.2 | `src/models/preprocessing.py::Winsorizer` (fit on train only, sklearn-compatible); `src/models/factory.py::make_model(name, cfg)` returning a Pipeline per `04§6` | T4.1 | `04§3`, `04§6` | Test: winsorizer limits equal training percentiles and do not change when test data changes; all four models fit on a tiny sample |
| T4.3 | `walk_forward_predict` and `src/models/train.py` CLI: yearly expanding window with embargo, one `dim_model` row per model and year, models saved to `models/`, OOS probabilities written to `fact_signal`, `next_day_return` filled; each model's validation-year accuracy logged as a trial | T4.2 | `04§6` | Test: for each test year, `max(train index) < year_start - embargo`; rerun is idempotent; `fact_signal` has `prob_up` for every labelled equity-day with complete features (the panel rows) from `validation_start` onward |
| T4.4 | `src/models/evaluate.py`: per model and year accuracy, majority baseline, precision, recall, F1, ROC-AUC, confusion matrix; `reports/results/classifier_metrics.csv`; confusion matrix, ROC and permutation-importance figures; decision-tree rules text | T4.3 | `04§7` | Worked-example test (`TESTS.md§5`); the CSV always shows the majority baseline next to accuracy |
| T4.5 | Add training and evaluation to `make smoke`; `[REAL]` run on the real warehouse | T4.4 | `07§5` | `make smoke` still passes; real run records accuracy versus baseline per model in `PROGRESS.md` |

**Gate P4:** `make check && make smoke`.

---

## Phase 5: Strategy and backtest

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T5.1 | `src/trading/rules.py`: `build_regime_rules(cfg)`, `target_weights(...)` per `05§4` (with `force_exit` and `no_entry`) | T0.3 | `05` | All target-weight cases in `TESTS.md§6` pass, including the runbook worked example |
| T5.2 | `src/trading/portfolio.py` and `execution.py`: `Portfolio`, `plan_orders` (band, whole shares or `fractional_shares`, sells first, cash cap, a `reason` on every order), `fill` (slippage and cost), `apply_trade`, realized P&L | T0.3 | `06§3` | Cost worked example gives 100,150.05; cash never negative (property test); band cases pass |
| T5.3 | `src/backtest/metrics.py`: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor, turnover, exposure | T0.3 | `06§4` | Fixtures in `TESTS.md§7` pass |
| T5.4 | `src/backtest/engine.py`: `run_backtest(weights_fn, ...)` per the algorithm in `06§2`; writes `fact_trade` and `fact_portfolio_daily` by `run_id`; primary run also updates `fact_signal` signal and weight; CLI `--run-id`, `--period validation` (the CLI refuses the test period), `--model`, `--variant` | T4.3, T5.1, T5.2, T5.3 | `06§2` | No-look-ahead test (fills at next open, not same close); zero-cost constant-weight identity test; a skipped bar does not crash; rerun is idempotent |
| T5.5 | `src/backtest/strategies.py`: `make_weights_fn(name, cfg)` returning the weight providers V0, V1, V2 (`05§7`) and B1, B2, B3 (`06§6`) | T5.1, T5.4 | `05§7`, `06§6` | Each provider runs through the engine on smoke data; V0, V1, V2 equity curves differ; B1 trades once and then holds (no repeated rebalancing); B2 is invested from the first day and rebalances on month-ends |
| T5.6 | `src/backtest/tune.py`: validation-year grid (models x `entry_base` x `slots`), calibration check from `05§6`, logs every trial, writes `tune_grid_validation.csv` and `paths.frozen_config` | T4.3, T5.5 | `05§6`, `DECISIONS R16` | Grid size equals the config product (plus fallback entries when triggered); trial-log rows match and a rerun adds none; the frozen config has the format in `07§3` |
| T5.7 | `src/backtest/baselines.py`: for `--period validation`, run B1 to B3 and V0 to V2 with the frozen model (or `--model`), persist them as `val_*`, run the B4 symbol-permutation null (`06§6`), write `b4_sharpe_distribution_validation.csv` and `summary_validation.csv` | T5.6 | `05§6`, `06§5`, `06§6`, `06§10` | Each baseline runs on smoke data; B4 reproducible from the seed; `summary_validation.csv` has B1 to B3, V0 to V2 and the B4 percentile columns on the V2 row |
| T5.8 | Add tune and baselines to `make smoke`; `[REAL]` run on the real warehouse, **validation period only** | T5.7 | `07§5` | `make smoke` passes; `reports/results/summary_validation.csv` exists on the real warehouse |

**Gate P5:** `make check && make smoke`. The test period must not have been evaluated yet.

---

## Phase 6: Final evaluation and results assets

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T6.1 | `src/backtest/evaluate.py`: block-bootstrap Sharpe interval, per-year metrics, regime breakdown (OLAP 09), with-versus-without-cost ablation (A4) | T5.7 | `06§7` | Bootstrap reproducible from seed; interval contains the point estimate for a long synthetic series (sanity) |
| T6.2 | `src/backtest/final_test.py`: the guard in `06§8`; freezes config (copy to `reports/results/frozen_config_used.yaml` plus commit hash), runs strategy, ablations A1 to A4 and baselines on the test period once, writes `summary_test.csv`, yearly tables, test-year classifier metrics, `b4_sharpe_distribution_test.csv`, run manifest (the `make final-test` target runs `make check` first) | T5.7, T6.1 | `06§8` | Test: second invocation exits non-zero without `--allow-rerun`; lock file written; manifest written |
| T6.3 | `src/backtest/report.py`: all figures in `06§10` and `reports/results/RESULTS.md` (tables, trial count, caveats, disclaimer; numbers only, no claims beyond the data) | T6.2 | `06§10` | All listed files exist; `RESULTS.md` numbers match the CSVs (test) |
| T6.4 | `report.py` writes `reports/results/leakage_checklist.md` (statuses VERIFIED, TEST or STATEMENT per `06§9`); then run `make test` and record in `PROGRESS.md` that the TEST items pass | T6.3 | `06§9` | Every item is VERIFIED, or TEST with a passing run recorded, or a STATEMENT present in `RESULTS.md`; none without evidence |

**Gate P6:** `make check && make smoke` (smoke uses its own guard files). `[REAL]` run `make final-test` once on real data, after all tuning is complete.

---

## Phase 7: Paper trading

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T7.1 | `src/paper/store.py`: read and write `pending_target`, derive positions and cash from `fact_trade` and `fact_portfolio_daily` | T5.4 | `08§3` | Round-trip test; derived positions equal engine state after a backtest |
| T7.2 | `src/paper/daily_job.py`: incremental ETL, holiday check, process each new trading date in order (settle, mark, stop-loss, indicators, regime, model for the year, predict, decide, store, report), `run_id = paper_live` | T7.1, T6.2 | `08§3` | Replay test in `TESTS.md§8`: paper replay over a synthetic period equals the backtest equity within 1e-6; double run changes nothing; holiday exits 0; a new `paper_live` starts at the latest date and is refused without `TEST_RUN_LOCK.json` |
| T7.3 | Failure handling and alerts per `08§5` and `08§6`: retries, FAILED run, stale data banner, fallback model, console and `logs/alerts.log` | T7.2 | `08§5`, `08§6` | One test per row of the failure table |
| T7.4 | `reports/daily/<date>.md` one-page summary; `deploy/cron.example`, `deploy/github_actions_daily.yml` (do not install anything) | T7.2 | `08§3`, `08§4` | Summary lists regime, orders for tomorrow, portfolio value, drawdown; files exist |
| T7.5 | Add paper replay to `make smoke`; `[REAL]` initialise `paper_live` on real data | T7.4 | `07§5` | `make smoke` passes; `python -m src.paper.daily_job` on real data exits 0 |

**Gate P7:** `make check && make smoke`.

---

## Phase 8: Dashboard

| ID | Task | Depends | Spec | Acceptance |
|---|---|---|---|---|
| T8.1 | `src/dashboard/app.py` with the five screens in `02§6`; read-only queries; stale-data banner | T6.3, T7.4 | `02§6` | `streamlit run` starts against the smoke warehouse with no exception on any screen (test with `streamlit.testing.v1.AppTest`) |
| T8.2 | Empty-state handling: each screen shows a clear message when its tables are empty | T8.1 | `02§6` | AppTest on an empty database renders without exception |

**Gate P8:** `make check`.

---

## Phase 9: Stretch (only after Phases 0 to 8 and 10 pass)

| ID | Task | Spec | Acceptance |
|---|---|---|---|
| T9.1 | Association rules (Apriori) on weekly sector up/down, training period only | `04§8` | Worked-example test (support 0.40, confidence 0.667, lift 1.33); rules CSV written |
| T9.2 | V3: relative-return target (`target.kind: relative`, models suffixed `_rel`) as an ablation | `04§4`, `05§7` | Appears in the ablation table; separate `dim_model` rows |
| T9.3 | V4: one classifier per regime, if each regime has enough rows | `05§7` | Row counts per regime printed; ablation row added |
| T9.4 | Yearly regime refit using only past data (`regime.refit: yearly`) | `04§5.3` | Test: regime for day *t* depends only on data before *t*'s refit date |

---

## Phase 10: Hardening and handoff

| ID | Task | Depends | Acceptance |
|---|---|---|---|
| T10.1 | `README.md` (setup, commands, structure, disclaimer, how to reproduce); `requirements.lock` via `pip freeze` | all | A fresh reader can run `make setup && make smoke` from the README alone |
| T10.2 | Clean-clone check: clone to a temp folder, run `make setup && make check && make smoke` (disable the network after `make setup` has installed the dependencies) | T10.1 | All pass |
| T10.3 | Final `PROGRESS.md` summary: trial count, chosen k, selected model, validation and test headline numbers, limitations observed, anything BLOCKED | T10.2 | Complete and honest |
| T10.4 | Confirm the Definition of done in `AGENTS.md` §8, item by item | T10.3 | Every item is true, or listed as not done with the reason |
