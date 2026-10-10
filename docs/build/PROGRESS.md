# PROGRESS

Update after **every** task: tick the box, add a log line, record key numbers. This file is the memory between sessions. Do not delete history; append.

Legend: `[ ]` todo, `[x]` done and acceptance passed, `[~]` BLOCKED (reason in Blockers).

## Current position

- Phase: 1
- Next task: T1.8 (real-data full refresh running)
- Last gate passed: **P0**; P1 ETL step passes (`make smoke` loads 29,890 rows, DQ-01..07 green). 203 fast tests, 212 with slow.

## Checklist

**Phase 0 Scaffold:** [x] T0.1 [x] T0.2 [x] T0.3 [x] T0.4 [x] T0.5
**Phase 1 Warehouse and ETL:** [x] T1.1 [x] T1.2 [x] T1.3 [x] T1.4 [x] T1.5 [x] T1.6 [x] T1.7 [ ] T1.8 (universe acquired; real-data load in progress)
**Phase 2 Indicators and OLAP:** [ ] T2.1 [ ] T2.2
**Phase 3 Regimes:** [ ] T3.1 [ ] T3.2 [ ] T3.3 [ ] T3.4 [ ] T3.5
**Phase 4 Classifiers:** [ ] T4.1 [ ] T4.2 [ ] T4.3 [ ] T4.4 [ ] T4.5
**Phase 5 Strategy and backtest:** [ ] T5.1 [ ] T5.2 [ ] T5.3 [ ] T5.4 [ ] T5.5 [ ] T5.6 [ ] T5.7 [ ] T5.8
**Phase 6 Final evaluation:** [ ] T6.1 [ ] T6.2 [ ] T6.3 [ ] T6.4
**Phase 7 Paper trading:** [ ] T7.1 [ ] T7.2 [ ] T7.3 [ ] T7.4 [ ] T7.5
**Phase 8 Dashboard:** [ ] T8.1 [ ] T8.2
**Phase 9 Stretch:** [ ] T9.1 [ ] T9.2 [ ] T9.3 [ ] T9.4
**Phase 10 Handoff:** [ ] T10.1 [ ] T10.2 [ ] T10.3 [ ] T10.4

## Key numbers (fill as they become known)

| Item | Value |
|---|---|
| Universe file source and date | niftyindices.com `ind_nifty50list.csv`, frozen 2026-10-10; 50 rows, all symbols resolve on yfinance |
| Instruments loaded / stocks with under 10 years of history | pending the real run (T1.8); universe frame = 50 EQUITY + `^NSEI` + `^INDIAVIX` = 52 |
| Trading days loaded, first and last date | |
| VIX used in regimes (yes or no) and coverage | |
| Chosen k and why | |
| Regime day counts (BULL / SIDEWAYS / HIGH_VOL / BEAR) | |
| Selected model, `entry_base`, `slots` (from validation) | |
| Classifier accuracy versus majority baseline (per model, validation) | |
| Total trials logged | |
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
