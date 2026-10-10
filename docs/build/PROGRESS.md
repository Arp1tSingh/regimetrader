# PROGRESS

Update after **every** task: tick the box, add a log line, record key numbers. This file is the memory between sessions. Do not delete history; append.

Legend: `[ ]` todo, `[x]` done and acceptance passed, `[~]` BLOCKED (reason in Blockers).

## Current position

- Phase: 5
- Next task: T5.6 (tuning grid) — **check with the human first if the grid or the entry-threshold fallback changes**
- Last gate passed: **P4** (`make check && make smoke`). Phases 3 and 4 complete; T5.1-T5.5 done.
- Test count: **352 fast tests** (`make check` runs `-m "not slow"`, so the fast set only), **373 total** with the slow set.

## Pre-registered expectation (written before any backtest result exists)

**2026-10-10.** Recorded now, before T5.6 tuning and before the one-time final test,
so the result cannot be reframed afterwards.

> **Expect the strategy to trail buy-and-hold on return. Drawdown and Sharpe are the
> open questions.**

Basis, stated up front: the best classifier (decision tree) scored **0.5316** against a
**0.5072** majority baseline on the validation year, an edge of **+2.44 points**. Two of
the four models scored *below* the baseline. After 0.30% round-trip costs, a directional
edge that small is very unlikely to survive. The regime exposure rules reduce exposure in
HIGH_VOL and go to cash in BEAR, which may help drawdown and Sharpe, but that is the open
question, not the expectation.

If the test period instead shows the strategy beating buy-and-hold on return, that
result will be reported as anomalous and investigated rather than presented as success.

## Honest scope of results so far

- **No backtest has ever run on the real warehouse.** `fact_trade` and
  `fact_portfolio_daily` are both empty and `fact_signal.signal` / `target_weight` are
  still 0 everywhere, as `DECISIONS` R3 requires.
- The **+2.44** figure is a **validation-year classifier accuracy** (`period =
  "validation"`, `run_id = wf2023`), from `src/models/train.py`. It is not a strategy
  return and not a test-period number.
- It came from the **untuned default settings**: `models.tree` defaults
  (`max_depth=4`, `min_samples_leaf=200`), `strategy.entry_base=0.55`, `slots=10`, and
  `config/frozen_config.yaml` does not exist yet. No tuning has run.
- The **V0/V1/V2 comparison ran on the SYNTHETIC smoke warehouse**, not real data. It
  only confirms the three providers are wired differently; it informed no decision. The
  test is named `test_variants_differ_on_smoke_signals` (it was previously misnamed
  `..._on_real_signals`, which was wrong).
- Verified on 2026-10-10 that the period guard genuinely refuses test-period data:
  `end=2023-12-31` allowed, `end=2024-01-02` and `end=2026-10-09` both raise
  `ValueError` against `test_start=2024-01-01`.

## Stop and check with the human before

1. **T6.2, the one-time final test.** The lock is created once and nobody runs it casually.
2. **Any `paper_live` run on real data.**
3. **T5.6, if the grid size or the entry-threshold fallback changes**, because that
   changes the reported trial count.

## Log

| Date | Task | Result and notes |
|---|---|---|
| 2026-10-10 | T5.4 | `src/backtest/engine.py`: EXECUTE -> MARK -> DECIDE -> STORE ordering, `DayContext`, `run_backtest` with the test-period guard, `fact_trade` and `fact_portfolio_daily` written by run_id (idempotent), `write_signals` reserved for the primary run. 10 tests. **Fill-timing test proves no look-ahead**: with `close_t=100` and `open_(t+1)=200`, the fill is 200. |
| 2026-10-10 | T5.4 | **Bug fixed**: `running_max` was computed from previous rows only, so every new equity high produced a *negative* drawdown (-0.0017 instead of 0), understating drawdown in all results. |
| 2026-10-10 | T5.4 | **Bug fixed**: the stop-loss cooldown was registered when the stop *triggered* rather than when the exit *filled*, so `cooldown_days` counted from the wrong date. Now starts at the exit and counts real trading days. |
| 2026-10-10 | T5.5 | `src/backtest/strategies.py`: V0 (classifier only, BULL rule everywhere), V1 (regime filter only, BULL only), V2 (full strategy), B1 (buy and hold, trades once then None), B2 (equal weight, month-end rebalance), B3 (index). 11 tests. |
| 2026-10-10 | T5.5 | Test fixtures consolidated into `tests/conftest.py` (`engine`, `loaded`, `panel_ready`, `signals_ready`). They had been duplicated across four modules with inconsistent return orders (`(engine, cfg)` vs `(cfg, engine)`), causing `TypeError: 'Engine' object is not a mapping`. |
| 2026-10-10 | T5.5 | **Correction to an earlier claim.** The V0/V1/V2 comparison runs on the **synthetic smoke warehouse**, not real data. The test was misnamed `test_variants_differ_on_real_signals`, now `test_variants_differ_on_smoke_signals`. It only proves the providers are wired differently. The three runs are logged as trials with `period=synthetic_smoke` and a note that they informed no decision. |
| 2026-10-10 | audit | Verified the period guard empirically against the real config: `end=2023-12-31` allowed; `end=2024-01-02` and `end=2026-10-09` both raise `ValueError` against `test_start=2024-01-01`. Confirmed **no backtest has ever run on the real warehouse** (`fact_trade` and `fact_portfolio_daily` empty, `signal` and `target_weight` still 0). |
| 2026-10-10 | audit | **Pre-registered expectation written before any backtest result exists**, so it cannot be reframed afterwards. See the "Pre-registered expectation" section above. |
| 2026-10-10 | T4.1 | `src/features/panel.py`: **135,135 rows**, 50 symbols, 19 active features, **zero NaN**, up share 50.68%. `FEATURE_COLS_ALL` is the documented 19 names in order; `get_feature_cols(False)` gives 18 and still keeps all four regime one-hots. Single `# label-only` forward shift. |
| 2026-10-10 | T4.1 | **Leakage verified on the real warehouse, not only in tests**: deleting every `fact_price` row after 2021-12-31 and rebuilding the panel left all **76,690** earlier rows byte-identical. |
| 2026-10-10 | T4.2 | `src/models/preprocessing.py` and `factory.py`: `Winsorizer` limits equal the fit data's 1st/99th percentiles and never move when other data is transformed; one-hot columns are never clipped. Four pipelines, all seeded from `project.seed`. |
| 2026-10-10 | T4.3 | `src/models/train.py`: expanding-window walk-forward with a 5-day **calendar** embargo, asserted in code (the run aborts if training reaches the cut-off). 16 `dim_model` rows, **184,976 `fact_signal` rows** written, `signal` and `target_weight` verified still 0 everywhere (DECISIONS R3). |
| 2026-10-10 | T4.4 | `src/models/evaluate.py`: metrics always reported next to the majority baseline. |
| 2026-10-10 | **T4.5 real run** | **Validation-year accuracy versus the majority baseline (0.5072, always-up):** tree **0.5316** (+2.44 pts) · rf 0.5125 (+0.53) · xgb 0.5059 (-0.13) · logreg 0.4980 (-0.92). Only the decision tree beats the baseline by a useful margin. This is a weak directional signal and must be reported as such, not dressed up. |
| 2026-10-10 | T4.5 | **Incident during this phase.** A leakage check of mine backed up only rows with `date_key <= 20211231`, deleted everything after it, then "restored" from that truncated backup. The real warehouse silently lost all 2022+ prices; training failed on an empty panel. Detected immediately, then rebuilt with a full ETL refresh (143,481 prices, 140,583 indicators, 2,878 regimes). Recorded because silent data loss is the failure mode most likely to corrupt a reported result. |
| 2026-10-10 | T4.3 | **Bug fixed**: the trainer joined predictions back to the panel on `symbol` alone, a cartesian product across every date that stock trades. Predictions now carry their trade date and join on `(date_key, symbol)`. |
| 2026-10-10 | T5.1-T5.3 | `src/trading/rules.py`, `execution.py`, `src/backtest/metrics.py`. 17 rules tests including the runbook worked example (keeps H1-H7, drops H8 at 0.47, adds N1-N3, 10 names at 0.10) and 26 execution/metrics tests including the cost fixture (fill 1,000.50, cost 100.05, cash out 100,150.05), max drawdown -0.25, Sharpe 0.8819 and CAGR 0.0999. |

## Log

| Date | Task | Result and notes |
|---|---|---|
| 2026-10-10 | T2.1 | `src/features/indicators.py` + `builder.py`: 14 `fact_indicator` columns, pure and trailing-only, always recomputed from full history. Real run wrote **140,583 rows** (137,685 EQUITY + 2,898 INDEX, zero VOLATILITY per DECISIONS R9) with 2,499 NULL `sma_50` warm-up rows. 23 tests including the `test_features_no_lookahead` truncation test at four cut points (rtol 1e-12). |
| 2026-10-10 | T2.2 | `src/olap/run.py`: all 9 queries run, `-- param:` defaults parsed and typed, `--param` overrides honoured, query 06 regenerated from `olap.pivot_years`. Real run: 720 / 180 / 40 / 15 / 12 / 15 / 5 rows; 08 and 09 correctly empty before their phases. Query 02 verified equal to a hand aggregation of the source tables. 20 tests. |
| 2026-10-10 | **Gate P2** | `make check && make smoke` passes. 270 fast tests, 290 total. |
| 2026-10-10 | T3.1-T3.5 | `src/regime/regime_model.py`: market features, VIX `auto` decision, k-selection, fit on the training period only, trailing-mode smoothing, `fact_market_regime`, profile/counts CSV, k-selection and regime-band figures, saved model bundle. 34 tests including the `TESTS` section 5 worked example (distances 0.224 / 1.237 / 3.342) and the smoothing fixtures. |
| 2026-10-10 | T3.2 (Q1 decision) | **k = 3 chosen, not the default 4.** On the real warehouse k=4 produced two clusters under 3% of training days (BEAR 22/1948 = 1.1%, BULL 49/1948 = 2.5%), which is exactly the "meaningless fragments" condition in `DECISIONS` Q1 and `L5`. k=3 gives BEAR 27 days (vol 0.72, VIX 57.2, drawdown -0.29), BULL 1,167 days (vol 0.12, VIX 15.4) and SIDEWAYS 754 days (vol 0.19, VIX 20.2), and **all three sanity checks pass on the real data**. Silhouette: k=2 0.771, k=3 0.362, k=4 0.364. `config.yaml` `regime.k` set to 3 with the reason recorded. |
| 2026-10-10 | T3.4 | **Known limitation, recorded not hidden.** On the *synthetic* market the volatility sanity check cannot pass at k=3 or k=4. `TESTS` section 2 pins the four state volatilities to 0.007 / 0.006 / 0.015 / 0.018, which makes realised volatility bimodal: BULL and SIDEWAYS are nearly identical, so K-Means splits on volatility first and the two high-volatility clusters take the names BULL and BEAR (highest and lowest mean return). Their volatilities are then equal by construction (0.2604 vs 0.2594), so "BEAR volatility above BULL" fails by 0.4%. A cluster-versus-truth crosstab confirms the labels do not track the hidden states: the BULL cluster holds 0% of true BULL days. The same check passes on the real warehouse, where volatility is continuous. Per `04` section 5.4 the failure is warned about and recorded, **not** forced by renaming clusters. The check logic is unit-tested directly against hand-built profiles, both passing and failing. |

## Checklist

**Phase 0 Scaffold:** [x] T0.1 [x] T0.2 [x] T0.3 [x] T0.4 [x] T0.5
**Phase 1 Warehouse and ETL:** [x] T1.1 [x] T1.2 [x] T1.3 [x] T1.4 [x] T1.5 [x] T1.6 [x] T1.7 [x] T1.8

**Gate P1:** `make check && make smoke` plus the `[REAL]` T1.8 refresh. **PASSED.**
**Phase 2 Indicators and OLAP:** [x] T2.1 [x] T2.2

**Gate P2:** `make check && make smoke`. **PASSED.**
**Phase 3 Regimes:** [x] T3.1 [x] T3.2 [x] T3.3 [x] T3.4 [x] T3.5
**Phase 4 Classifiers:** [x] T4.1 [x] T4.2 [x] T4.3 [x] T4.4 [x] T4.5
**Phase 5 Strategy and backtest:** [x] T5.1 [x] T5.2 [x] T5.3 [x] T5.4 [x] T5.5 [ ] T5.6 [ ] T5.7 [ ] T5.8

**Gate P4:** `make check && make smoke`. **PASSED** — 352 fast tests, 373 total.
**Phase 6 Final evaluation:** [ ] T6.1 [ ] T6.2 [ ] T6.3 [ ] T6.4
**Phase 7 Paper trading:** [ ] T7.1 [ ] T7.2 [ ] T7.3 [ ] T7.4 [ ] T7.5
**Phase 8 Dashboard:** [ ] T8.1 [ ] T8.2
**Phase 9 Stretch:** [ ] T9.1 [ ] T9.2 [ ] T9.3 [ ] T9.4
**Phase 10 Handoff:** [ ] T10.1 [ ] T10.2 [ ] T10.3 [ ] T10.4

## Key numbers (fill as they become known)

| Item | Value |
|---|---|
| Universe file source and date | niftyindices.com `ind_nifty50list.csv`, frozen 2026-10-10; 50 rows, all symbols resolve on yfinance |
| Instruments loaded / stocks with under 10 years of history | **52 instruments** (50 EQUITY + `^NSEI` + `^INDIAVIX`); **6 stocks under 10 years**: JIOFIN, ETERNAL, MAXHEALTH, HDFCLIFE, SBILIFE, BSE |
| Trading days loaded, first and last date | 2015-01-02 to 2026-10-09; **2,898 index bars**, 143,481 price rows, 419 rejected (0.29%) |
| VIX used in regimes (yes or no) and coverage | |
| Chosen k and why | |
| Regime day counts (BULL / SIDEWAYS / HIGH_VOL / BEAR) | |
| Selected model, `entry_base`, `slots` (from validation) | not yet tuned (T5.6). Defaults in use: `entry_base` 0.55, `slots` 10, model untuned |
| Classifier accuracy versus majority baseline (per model, validation year 2023) | **tree 0.5316** · rf 0.5125 · xgb 0.5059 · logreg 0.4980; majority baseline **0.5072** |
| Total trials logged | **real warehouse: 9** (5 regime_k + 4 model_year). Smoke warehouse: 12 (adds 3 synthetic V0/V1/V2 wiring checks, not real results). Trial count will change if T5.6's grid or fallback changes |
| Test period evaluated? (date, lock file) | |
| Headline test results (strategy vs B1) | |

## Log

| Date | Task | Result and notes |
|---|---|---|
| 2026-10-10 | T0.1 | Repo skeleton created from `07§1`: 24 folders, 13 `__init__.py`, `.gitkeep` files. `git init` on branch `main` (git 2.55.0 present, contrary to the note in the pack). |
| 2026-10-10 | T0.2 | venv at `.venv`, Python **3.14.8** (only interpreter on this machine; `python3` absent). All 39 packages from `requirements.txt` installed cleanly. Versions resolve newer than spec minimums: pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, streamlit 1.65.0. Acceptance import check exits 0. |
| 2026-10-10 | T0.3 | `src/common/config.py` with `extends`, override precedence, frozen-config whitelist, `config_hash()`, path resolution, output-folder creation, split-order validation. 19 tests pass. `make check` green. |
| 2026-10-10 | T0.4 | `src/common/`: `db.py` (engine with `foreign_keys=ON` + WAL, `init_db` via `executescript`, `read_sql`, `execute_many`), `log.py`, `trial_log.py` (upsert on the R28 key), `manifest.py`, `seeds.py`, `dates.py`. 46 tests pass. `init_db` applies the schema twice without error. |
| 2026-10-10 | T0.5 | `tests/test_schema.py`: all 13 tables from `03§2` exist with the documented primary keys, `dim_regime` has its 4 seed rows, CHECK constraints on `instrument_type` and `fact_trade.reason` reject bad values, no extra tables. |
| 2026-10-10 | **Gate P0** | `make check` passes: ruff clean, **78 tests** green in ~14s. |
| 2026-10-10 | repo | Initialised the GitHub repo `Arp1tSingh/regimetrader` (private) with `gh`; `origin/main` tracks every commit from here on. |
| 2026-10-10 | T1.8 (universe) | **Blocker cleared early.** `config/nifty50_symbols.csv` downloaded from niftyindices.com `ind_nifty50list.csv` (HTTP 200, 50 rows). Built via `scripts/build_universe.py`, which maps NSE industry to project sector and splits `Financial Services` into `Banking` / `Financial Services` by hand (`03§3`). Verified: 50 rows, both `Banking` and `IT` present, and **all 50 symbols resolve on yfinance** (a deliberate control symbol 404s, so the check is meaningful). |
| 2026-10-10 | T1.1 | `src/etl/universe.py`: loads and validates the CSV, skips `#` lines, enforces unique symbols and the mandatory `Banking`/`IT` labels, appends `^NSEI` (INDEX) and `^INDIAVIX` (VOLATILITY). 16 tests. |
| 2026-10-10 | T1.2 | `src/etl/synthetic.py`: seeded 4-state Markov regime chain (BULL/SIDEWAYS/HIGH_VOL/BEAR), index and beta-driven stocks with AR(1) noise, OHLC and VIX construction, weekday calendar with ~9 holidays/year. 26 tests, all four regimes observed. |
| 2026-10-10 | T1.3 | `src/etl/extract.py`: yfinance/csv/synthetic sources, per-config retries with waits, VIX CSV fallback, raw cache to `data/raw/`, staging into `stg_price_raw`. 16 tests including `test_extract_retry`. |
| 2026-10-10 | T1.4 | `src/etl/transform.py`: the seven documented steps in order. Two bugs found and fixed by the tests: the gap-run counter under-filled (grouping bug) and `is_flagged` went NaN on reindex. 21 tests including no-back-fill, bounded forward-fill and rejects. |
| 2026-10-10 | T1.5 | `src/etl/load.py`: single-transaction upsert of `dim_stock`, `dim_date`, `fact_price`; `is_month_end` recomputed; `etl_run_log` written in a separate transaction so `FAILED` survives a rollback. 15 tests including `test_load_idempotent` and `test_load_rollback`. |
| 2026-10-10 | T1.6 | `src/etl/quality.py`: DQ-01..DQ-07 with the documented hard/soft split. Hard failures raise `DataQualityError` (exit 4); soft ones only warn. DQ-06 is skipped for the synthetic source. 26 tests, each check with a passing and a failing case. |
| 2026-10-10 | T1.7 | `src/etl/pipeline.py`: full CLI with `--full-refresh/--source/--limit-symbols/--config`, the documented exit codes (0/1/2/3/4), incremental start from the last index date minus `data.incremental_overlap_days`, and a run manifest. **Gate P1 ETL step passes**: `make smoke` loads 29,890 rows over 14 synthetic instruments, rerun is byte-identical (checksum `db084e49601104d3`), and the incremental run resumes from 2025-06-25 and loads 56 rows with the checksum unchanged. |
| 2026-10-10 | T1.7 | **Two real bugs found by the smoke run and fixed.** (a) With `source: synthetic` the pipeline loaded the *real* Nifty 50 universe against generated SYN01..SYN12 data, so only the 2 shared symbols survived (2 instruments / 4,270 rows). `extract` now carries the generated universe on `raw.attrs` and it wins. Now 14 instruments / 29,890 rows. (b) **Incremental loads corrupted the trading calendar**: `build_dim_date` only saw the incremental slice, so it set `is_trading_day = 0` on all 2,131 earlier trading days, which then failed DQ-03 on 29,834 rows. `load` now recomputes the flags from the index's bars already in the warehouse. Regression test added. |
| 2026-10-10 | T1.8 | **Real-data full refresh completed.** `--source yfinance` over 52 instruments from 2015-01-01. 143,490 rows extracted, 143,481 loaded, 419 rejected (0.29%, under the 1% limit). `etl_run_log` status SUCCESS. All hard checks pass; all seven DQ checks pass. Trading days 2015-01-02 to 2026-10-09 = **2,898 index bars** (T1.8 expects "roughly 2,900"). Incremental rerun from 2026-10-04 loaded 269 rows and left the checksum unchanged (`708ed823a676c394`). |
| 2026-10-10 | T1.8 | The 419 rejections are **all** `NON_TRADING_DAY`, on 9 distinct dates (2015-01-01, 2015-04-15, 2016-01-01, 2016-08-12, 2018-01-01, 2019-01-01, 2019-10-27, 2020-11-14, 2026-01-15). These are genuine NSE holidays where `^NSEI` correctly has no bar but individual stock series still returned one. Expected, not a defect. |
| 2026-10-10 | T1.8 | **6 stocks have under 10 years of history** (DECISIONS R10): JIOFIN.NS from 2023-08-21, ETERNAL.NS from 2021-07-23, MAXHEALTH.NS from 2020-08-21, HDFCLIFE.NS from 2017-11-17, SBILIFE.NS from 2017-10-03, BSE.NS from 2017-02-03. The panel is therefore unbalanced; this is expected and must be reported. |
| 2026-10-10 | T1.8 | **Bug found and fixed while recording T1.8**: `_count_under_10_years` compared two `YYYYMMDD` integers (`last_key - first_key < 1000`). `20221230 - 20220103` is 1127 and `20261009 - 20150102` is 110907, so no real span ever fell under the threshold and the check **always reported 0**. Replaced with a calendar-year comparison; it now correctly reports 6. Regression test added. |
| 2026-10-10 | tooling | GNU Make ships the smoke recipe with `#` comment lines, but `cmd.exe` does not treat `#` as a comment, so the whole target failed with `CreateProcess failed`. Changed to `@REM`, which both shells ignore. |

## Blockers

| Date | Task | Blocker | Needed from human |
|---|---|---|---|
| | | | |

## Deviations from the spec

Anything you changed or could not do as written. Also add a numbered entry to `docs/spec/DECISIONS.md`.

| Date | Task | Deviation and reason |
|---|---|---|
| 2026-10-10 | T0.1 | The pack states git is not installed on this machine. It is (2.55.0). `git init` performed as T0.1 requires. |
| 2026-10-10 | T0.2 | **Environment, needs a human decision.** Three deviations: (a) no `make` was present, so GNU Make 4.4.1 was installed via `winget install ezwinports.make` (user scope, no admin); (b) no POSIX `python3`, so the interpreter is Python 3.14.8 with `python`; (c) the shipped Makefile is POSIX-only, so it was edited in place to branch on `$(OS)` for the venv path and the `rm -rf` smoke clean. Recipes and POSIX behaviour are unchanged. |
| 2026-10-10 | T0.2 | **pandas 3.0.6 is a major version above the `>=2.2` the spec was written against.** Copy-on-write is now default and several 2.x idioms are removed. Code must be written and reviewed for pandas 3 semantics, not backported pandas 2 behaviour. |
| 2026-10-10 | T1.2, T1.4 | Concrete pandas 3 breaks hit while building, recorded so they are not rediscovered: `DatetimeIndex.day_name` and `.month_name` are **methods**, not properties; `.dt.tz` on a tz-naive column raises instead of returning `None` (guarded with `getattr`); string columns are Arrow-backed, so assigning a mixed type into them raises `TypeError`. |
| 2026-10-10 | T0.2 | `make lint` now runs `ruff format` before `ruff check --fix`, and a `format` target was added. The `lint` target's recipe changed; nothing else in the Makefile did. |
| 2026-10-10 | T1.7 | The Makefile `smoke` recipe used `#` to comment out later-phase commands. On POSIX `sh` that is a comment; on `cmd.exe` it is not, and the target aborted with `CreateProcess failed`. Replaced with `@REM`, which both shells ignore. A real portability fix, not a workaround. |
| 2026-10-10 | T1.7 | `scripts/build_universe.py` was added, outside the pack's `07§1` tree, to convert the NSE download into `config/nifty50_symbols.csv`. It is a manually run one-off that makes the sector mapping auditable; nothing in the ETL imports it. |
