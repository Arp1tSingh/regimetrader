# AGENTS.md: RegimeTrader

A regime-aware stock direction predictor and paper-trading bot, built on a star-schema warehouse of Nifty 50 daily history.
Academic Data Warehousing and Mining project. **Not investment advice. No real money. No broker integration.**

**Status:** specification complete, **no code exists yet**. You are building it from scratch, following the plan in `docs/build/BUILD_PLAN.md`.

---

## 1. Read order

1. This file.
2. `docs/spec/DECISIONS.md`: locked decisions and every place the spec resolves an ambiguity. **It wins over any other doc on conflict.**
3. `docs/build/BUILD_PLAN.md`: ordered tasks with acceptance checks. Work through it top to bottom.
4. `docs/build/PROGRESS.md`: what is done, blockers, key numbers. Read at session start, update after every task.
5. Spec files, only the sections a task cites:

| File | Covers |
|---|---|
| `docs/spec/02_system_design.md` | Layers, components, principles, dashboard screens |
| `docs/spec/03_warehouse_and_etl.md` | Schema summary, universe, ETL rules, data-quality checks, OLAP |
| `docs/spec/04_mining_models.md` | Features, preprocessing, regimes, classifiers, evaluation |
| `docs/spec/05_trading_strategy.md` | Regime rules, position rules, `target_weights`, calibration |
| `docs/spec/06_backtesting_and_evaluation.md` | Engine, costs, metrics, baselines, honesty protocol |
| `docs/spec/07_project_structure.md` | Folder layout, module contracts, CLI, Makefile, requirements |
| `docs/spec/08_runbook.md` | Research pipeline, daily job, failure handling |
| `docs/spec/TESTS.md` | Required tests with numeric fixtures |

Ready-made files that already exist, to use and not retype: `sql/schema.sql`, `sql/olap/*.sql`, `config/config.yaml`, `config/config.smoke.yaml`, `Makefile`, `pyproject.toml`, `requirements.txt`, `.gitignore`. (`config/nifty50_symbols.FORMAT_EXAMPLE.csv` shows the universe file format only; it is not the real list.)

`docs/human/*` is for the student (report, viva, timeline). **Do not load it to build the system.**

---

## 2. Working protocol

1. Open `PROGRESS.md`; take the first unchecked task in `BUILD_PLAN.md` whose dependencies are done.
2. Read only the spec sections that task cites.
3. Write tests first for pure logic (metrics, rules, costs, leakage). Then implement.
4. Run `make check` (lint plus fast tests). At the end of a phase run that phase's gate command.
5. Tick the task in `PROGRESS.md`, add one log line (date, task, result, key numbers), commit.
6. If a task fails its acceptance check three times, write the blocker in `PROGRESS.md`, then continue with any task that does not depend on it, or stop and ask the human.
7. Never mark a task done on the strength of "it should work". Done means the acceptance command ran and passed.

Commit per task. Message format: `feat(etl): upsert loader (T1.5)`. Keep commits small.

---

## 3. Non-negotiable rules

1. **No look-ahead.** Features for day *t* use data up to and including the close of *t*. The label uses day *t+1* only as the label. Orders decided at the close of *t* fill at the open of *t+1*.
2. **Chronological everything.** Never shuffle. Splits: train to 2022-12-31, validation 2023, test 2024 onwards (all from config).
3. **Fit on training data only:** winsorization limits, scalers, K-Means. Apply them frozen to later dates.
4. **The test period is evaluated once.** Use the guard in `docs/spec/06` section 8. Tune on train and validation only.
5. **Log every experiment** with `log_trial()` (`docs/spec/07` section 4). The report must state the total count.
6. **Config-driven.** Dates, thresholds, costs, seeds, model settings live in `config/config.yaml`. No magic numbers in code.
7. **Idempotent loads.** Re-running any ETL or job for the same dates changes nothing (upserts).
8. **Reproducible.** Seeds from config; `run_id` on every backtest and paper run; write a run manifest (`docs/spec/07` section 4).
9. **One strategy code path.** Backtest and paper trading call the same functions in `src/trading/`.
10. **Warehouse is the source of truth.** Modules read inputs from and write outputs to warehouse tables (plus model files).
11. **Do not invent data.** If a data source fails, use the documented fallback or stop and ask. Never fabricate prices or a constituent list.
12. **Honest reporting.** Never tune to beat buy-and-hold. Report negative results as they are.

---

## 4. Commands

```bash
make setup        # venv + dependencies
make check        # ruff + fast tests (must pass before every commit)
make test         # full test suite
make smoke        # whole pipeline on synthetic data (config/config.smoke.yaml), no network
make etl-full     # python -m src.etl.pipeline --full-refresh
make features     # python -m src.features.builder --indicators   (the panel is built by `make train`)
make regime       # python -m src.regime.regime_model --fit
make train        # python -m src.models.train
make backtest     # tune on validation, then validation baselines and variants (val_* runs)
make final-test   # runs `make check`, then the one-time test-period evaluation (guarded)
make paper        # python -m src.paper.daily_job
make dashboard    # streamlit run src/dashboard/app.py
```

Full CLI contracts: `docs/spec/07_project_structure.md` section 5.

---

## 5. Environment and stop conditions

- Python 3.10+. SQLite warehouse at `data/warehouse/regimetrader.db`. Everything must run on a laptop.
- Real market data needs network access (yfinance). Tests and `make smoke` must never need it.
- **Stop and ask the human** (do not guess) when:
  - `config/nifty50_symbols.csv` is missing and cannot be downloaded from NSE (see `BUILD_PLAN.md` T1.8);
  - the data source is unreachable after the documented retries and fallbacks;
  - a hard data-quality check fails on real data and the cause is not a bug in your code;
  - a task conflicts with `DECISIONS.md`, or two spec files contradict each other;
  - you are about to evaluate the test period for a second time.
- Long jobs (full ETL, training): run with a timeout, write progress to `logs/`, never block on interactive prompts.

---

## 6. Known pitfalls (check against every task)

| Pitfall | Correct behaviour |
|---|---|
| yfinance index is timezone-aware (Asia/Kolkata) | `tz_localize(None)` on the local timestamps. Do **not** convert to UTC first (shifts the date back one day) |
| Index and VIX have volume 0 | Volume checks apply to `EQUITY` only |
| `shift(-1)` in features | Forbidden. Only the label may look forward |
| Back-filling missing values | Forbidden. Forward-fill at most 2 days |
| Fitting scaler, winsorizer or K-Means on all data | Fit on training data only |
| SQLite foreign keys are off by default | Run `PRAGMA foreign_keys = ON` on every connection |
| Date keys | `int(YYYYMMDD)`; never floats or strings |
| Dict or set iteration order | Sort before selecting; break ties by symbol so results are deterministic |
| Random models | Seed everything from `project.seed` |
| Printing results | Use `logging`, and write results to CSV files under `reports/results/` |
| Full-history indicator recompute vs incremental | Always recompute indicators from the full history (small data, avoids EWM drift) |

---

## 7. Code conventions

- Type hints on public functions; short docstrings stating inputs, outputs and units.
- Pure functions for indicators, rules, costs and metrics (no I/O inside them); I/O lives in thin wrappers.
- Parameterised SQL only (named bind parameters). SQL text lives in `sql/`, not in f-strings with user values.
- `pathlib.Path` for paths; all paths come from config.
- Every CLI module exposes `main(argv: list[str] | None = None) -> int` and returns a process exit code.
- Formatting and lint: `ruff` (config in `pyproject.toml`).
- No network calls in unit tests. Use the synthetic generator.

---

## 8. Definition of done (whole project)

- `make setup && make smoke` passes from a clean clone with no network.
- `make test` passes, including the look-ahead, walk-forward and test-lock tests.
- Real-data pipeline run: warehouse loaded, all hard data-quality checks pass.
- `reports/results/` contains the results tables, trial log, run manifests and the leakage checklist; `reports/figures/` contains the figures listed in `docs/spec/06` section 10.
- The daily paper-trading job runs end to end and a replay matches the backtest (`docs/spec/TESTS.md`).
- Dashboard starts and shows all six screens (the five read-only screens plus the Bot demo replay).
- `README.md` documents setup, commands and the disclaimer.
