# 07. Project Structure, Config, Contracts and CLI

This file replaces the original `07_project_structure.md`, which was not supplied (`DECISIONS` R1). It defines where everything lives and the interfaces between modules.

---

## 1. Repository layout

```text
regimetrader/
├── AGENTS.md  CLAUDE.md  README.md  Makefile  pyproject.toml
├── requirements.txt  requirements.lock          # lock created in T10.1
├── .gitignore
├── config/
│   ├── config.yaml                # all tunables
│   ├── config.smoke.yaml          # synthetic end-to-end run
│   ├── nifty50_symbols.csv        # frozen universe (human-supplied or downloaded in T1.8)
│   └── frozen_config*.yaml        # written by tuning
├── sql/
│   ├── schema.sql
│   └── olap/*.sql
├── docs/                          # spec, build, human
├── data/                          # gitignored
│   ├── raw/  external/  processed/  warehouse/
├── models/                        # gitignored: *.joblib
├── reports/                       # results/ figures/ daily/ (results and figures may be committed)
├── logs/                          # gitignored
├── deploy/                        # cron.example, github_actions_daily.yml
├── src/
│   ├── common/    config.py db.py log.py trial_log.py manifest.py seeds.py dates.py
│   ├── etl/       universe.py extract.py transform.py load.py quality.py synthetic.py pipeline.py
│   ├── features/  indicators.py builder.py panel.py
│   ├── olap/      run.py
│   ├── regime/    regime_model.py
│   ├── models/    preprocessing.py factory.py train.py evaluate.py
│   ├── trading/   rules.py portfolio.py execution.py
│   ├── backtest/  strategies.py engine.py metrics.py baselines.py tune.py evaluate.py final_test.py report.py
│   ├── paper/     store.py daily_job.py alerts.py
│   ├── mining/    assoc_rules.py          # stretch
│   └── dashboard/ app.py
└── tests/         conftest.py test_*.py
```

Every package has an `__init__.py`. Run modules from the repo root with `python -m src.<package>.<module>`.

---

## 2. Dependencies

`requirements.txt` (minimum versions; pin with `pip freeze > requirements.lock` in T10.1):

```text
pandas>=2.2
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
yfinance>=0.2.40
sqlalchemy>=2.0
pyyaml>=6.0
pyarrow>=15
joblib>=1.3
matplotlib>=3.8
plotly>=5.20
streamlit>=1.35
mlxtend>=0.23
pytest>=8.0
ruff>=0.4
```

`pyproject.toml`: ruff line length 100, rules `E,F,W,I,B`; pytest `testpaths = ["tests"]`, markers `slow` (integration and smoke tests).

---

## 3. Configuration (`src/common/config.py`)

```python
def load_config(path: str | None = None) -> dict: ...
def config_hash(cfg: dict) -> str: ...        # sha256 of the canonical JSON, first 12 hex chars
```

Precedence (later wins), deep-merged:

1. `config/config.yaml` (the base)
2. the override file given by `path` or `$RT_CONFIG`, if any and if it is not the base file itself (an override may contain `extends: <file>`; the extended file is merged first)
3. the frozen config: the file at `paths.frozen_config` of the result of steps 1 and 2 (so the smoke override's own frozen path is the one used), if it exists. It is applied **last**, and only these keys are taken from it: `strategy.selected_model`, `strategy.entry_base`, `strategy.slots`. Format:

   ```yaml
   strategy: {selected_model: rf, entry_base: 0.55, slots: 10}
   meta: {tuned_at: "2026-10-10T09:00:00", validation_sharpe: 0.62, trials: 16, method: threshold}   # method: threshold | rank_fallback
   ```

`src.backtest.tune` ignores the frozen strategy keys while it runs (it grids them). `final_test` copies the effective frozen file to `reports/results/frozen_config_used.yaml`.

Rules: resolve relative paths against the repo root; create output folders on demand; validate that `splits` dates are ordered (`train_end < validation_start <= validation_end < test_start`) and raise a clear error otherwise; fail fast on missing required keys; ignore unknown keys. Every CLI takes `--config PATH`.

---

## 4. Cross-cutting utilities

**Database (`db.py`):** `get_engine(cfg)` (SQLAlchemy, `sqlite:///` path from config; on connect run `PRAGMA foreign_keys = ON` and `PRAGMA journal_mode = WAL`); `init_db(engine)` applies `sql/schema.sql`; `read_sql(engine, sql, params=None) -> DataFrame`; `execute_many(conn, sql, rows)`. Use `text()` with named parameters.

**Logging (`log.py`):** `get_logger(name)`; INFO to console and `logs/<module>.log`; WARNING for fallbacks.

**Dates (`dates.py`):** `to_date_key(ts) -> int`, `from_date_key(key) -> pd.Timestamp`.

**Trial log (`trial_log.py`):** `log_trial(cfg, *, phase, kind, run_id, model=None, params=None, period=None, metric=None, value=None, extra=None, selected=False, note="")` writes one row to `reports/results/trial_log.csv` (create with header if missing). Columns: `timestamp, phase, kind, run_id, model, params_json, period, metric, value, extra_json, selected, note`. One row per configuration tried: `metric` is the headline number (`sharpe` for strategy trials, `accuracy` for model-year trials, `silhouette` for k trials) and any other numbers go in `extra_json`. `kind` is one of `regime_k`, `model_year`, `strategy_grid`, `calibration`, `other`. **Rows are upserted on the key (`kind`, `run_id`, `model`, `params_json`, `period`)**, so re-running an idempotent step never double-counts. The reported number of experiments is the count of rows with `kind != 'calibration'` (calibration rows are diagnostics, not configurations).

**Manifest (`manifest.py`):** `write_manifest(cfg, run_id, extra=None)` writes `reports/results/manifests/<run_id>.json` with `run_id, timestamp, git_commit ("unknown" if not a repo), config_hash, seed, command_line, package_versions (python, pandas, numpy, scikit-learn, xgboost), data_range (first and last date_key in fact_price), models_used, extra`.

**Seeds (`seeds.py`):** `seed_everything(cfg)` sets `random`, `numpy`; model constructors take `random_state` from config.

---

## 5. CLI contracts

Every module exposes `main(argv=None) -> int` and returns an exit code: **0** success, **1** unexpected error, **2** data or network failure (no trading was done), **3** guard refusal (test lock), **4** hard data-quality failure.

| Command | Flags | Reads | Writes |
|---|---|---|---|
| `python -m src.etl.pipeline` | `--full-refresh`, `--source yfinance\|csv\|synthetic`, `--limit-symbols N`, `--config` | config, source | `stg_price_raw`, `dim_*`, `fact_price`, `etl_run_log`, `data/raw/*` |
| `python -m src.features.builder` | `--indicators`, `--panel`, `--all` | `fact_price`, regimes | `fact_indicator`; `data/processed/panel.parquet` |
| `python -m src.olap.run` | `--all`, `--query STEM`, `--param k=v`, `--pivot-years a,b,c` | warehouse, `sql/olap` | `reports/results/olap/*.csv` |
| `python -m src.regime.regime_model` | `--fit` (runs the k-selection first, then fits with `regime.k` or `--k N`), `--k-select` (selection only), `--k N` | `fact_price`, `fact_indicator` | `fact_market_regime`, `models/regime_model.joblib`, results CSVs, figures |
| `python -m src.models.train` | `--models lr,tree,rf,xgb` (default all), `--first-test-year Y` (default validation year), `--year Y` (single year, used by the daily job), `--no-evaluate` | panel, regimes | `dim_model`, `fact_signal`, `models/*.joblib`, `classifier_metrics.csv`, figures |
| `python -m src.backtest.engine` | `--run-id`, `--variant V0\|V1\|V2`, `--model NAME` (default: frozen selection), `--no-cost`; validation period only, the CLI refuses the test period | warehouse | `fact_trade`, `fact_portfolio_daily` |
| `python -m src.backtest.baselines` | `--all`, `--only B1\|B2\|B3\|B4\|V0\|V1\|V2`, `--period validation`, `--model NAME` (default: frozen selection) | warehouse, frozen config | persisted runs `val_B1..B3`, `val_V0..V2`; B4 simulations; `b4_sharpe_distribution_validation.csv`; `summary_validation.csv` |
| `python -m src.backtest.tune` | none | validation signals | `paths.frozen_config`, trial log, `tune_grid_validation.csv` |
| `python -m src.backtest.final_test` | `--confirm`, `--allow-rerun "reason"` | frozen config | all test-period runs and tables, test-year rows of `classifier_metrics.csv` (`06`§8) |
| `python -m src.backtest.report` | none | results files | figures, `RESULTS.md`, `leakage_checklist.md` |
| `python -m src.paper.daily_job` | `--no-etl`, `--from DATE`, `--to DATE`, `--run-id` | warehouse, models | all paper tables, `reports/daily/` |
| `streamlit run src/dashboard/app.py` | | warehouse, results files | nothing |

---

## 6. Module contracts

Signatures are binding for cross-module calls. Internal helpers are free.

Panel cache: `data/processed/panel.parquet` is accompanied by `panel.meta.json` holding a hash of the config sections `features`, `regime`, `target` and `splits`, the row counts of `fact_indicator` and `fact_market_regime`, and the latest `date_key`. `build_panel` rebuilds when any of them differs or when `use_cache=False` (tests always pass `use_cache=False`). `src.models.train` always calls `build_panel` itself, so `make train` works without a prior `--panel` run.

```python
# src/etl
def load_universe(cfg) -> pd.DataFrame: ...            # symbol, company_name, sector, industry, instrument_type (index and VIX appended)
def make_synthetic_market(cfg) -> tuple[pd.DataFrame, pd.DataFrame]: ...   # (long OHLCV, universe)
def extract(cfg, symbols: list[str], start: str, source: str, engine=None) -> pd.DataFrame: ...   # long format; with an engine it truncates and fills stg_price_raw
def transform(raw: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]: ...     # (clean, rejected)
def build_dim_date(trading_dates, cfg) -> pd.DataFrame: ...                         # one row per calendar day in range
def load(engine, clean, universe, mode: str, stats: dict) -> dict: ...
def run_checks(engine, cfg) -> list[CheckResult]: ...

# src/features
def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame: ...       # one instrument, fact_indicator columns
def build_indicators(engine, cfg) -> int: ...                             # rows upserted
FEATURE_COLS_ALL: tuple[str, ...]                   # the 19 names in order (04 section 2.1)
def get_feature_cols(use_vix: bool) -> list[str]: ...   # 18 names (no india_vix) when VIX is not used
def build_panel(engine, cfg, *, live: bool = False, use_cache: bool = True) -> pd.DataFrame: ... # index = trade date; symbol, active feature columns, target_up, next_day_return; live=True keeps the last unlabeled row

# src/regime
def build_market_features(index_close, vix_close, dd_window=252, dd_min_periods=20) -> pd.DataFrame: ...
def fit_regime_model(train, features, k, seed): ...
def smooth_trailing_mode(labels, window=5) -> pd.Series: ...
def label_all_days(engine, cfg) -> pd.DataFrame: ...                      # writes fact_market_regime
def classify_day(model_bundle, market_features_row, history_labels) -> tuple[int, str]: ...   # (raw cluster, smoothed label) for the daily job

# src/models
class Winsorizer(BaseEstimator, TransformerMixin): ...                    # fit(X): store 1st and 99th percentiles per column
def make_model(name: str, cfg): ...                                       # sklearn Pipeline
def walk_forward_predict(panel, feature_cols, make_model, first_test_year, embargo_days=5, target_col="target_up", progress=None): ...

# src/trading  (pure, no I/O)
def build_regime_rules(cfg) -> dict: ...
def target_weights(day_signals, regime, held, rules, slots=10, exit_prob=0.50, force_exit=frozenset(), no_entry=frozenset()) -> pd.Series: ...
@dataclass
class Position: quantity: int | float; avg_fill: float; avg_cost: float; entry_date: pd.Timestamp
@dataclass
class Portfolio: cash: float; positions: dict[str, Position]
def plan_orders(targets, portfolio, equity_at_decision, opens, prob_up, cfg, band, fractional_shares=False) -> list[Order]: ...   # every Order carries a reason
def fill_order(order, open_price, cfg, costs_enabled=True) -> Trade: ...      # Trade.reason: ENTRY | REBALANCE | EXIT | STOP_LOSS
def apply_trade(portfolio, trade) -> None: ...                             # mutates; computes realized_pnl for sells
def mark_to_market(portfolio, closes) -> tuple[float, float]: ...           # (positions_value, equity); a held symbol missing from closes is valued at its last known close

# src/backtest
def make_weights_fn(name: str, cfg, **kw) -> WeightProvider: ...                  # strategies.py: V0 V1 V2 B1 B2 B3
def run_backtest(cfg, weights_fn, start, end, run_id, **kw) -> BacktestResult: ...    # section 06§2
def summarize(equity_df, trades_df, cfg) -> dict: ...
def block_bootstrap_sharpe(daily_returns, block=10, resamples=1000, rf=0.0, seed=42, ci=(5, 95)) -> dict: ...

# src/paper
def process_day(engine, cfg, portfolio_state, date, run_id) -> DayResult: ...    # one trading day, uses the same trading/ functions
```

---

## 7. Makefile

```makefile
VENV := .venv
PY   := $(VENV)/bin/python
CFG  ?= config/config.yaml
SMOKE := config/config.smoke.yaml

.PHONY: setup lint check test smoke etl-full features regime train backtest final-test paper dashboard

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt

lint:
	$(VENV)/bin/ruff check src tests

check: lint
	$(PY) -m pytest -m "not slow" -q

test:
	$(PY) -m pytest -q

# Whole pipeline on synthetic data. Append one line per phase as it lands. No network.
smoke:
	rm -rf data/warehouse/smoke.db data/raw_smoke data/processed_smoke models_smoke reports_smoke logs_smoke config/frozen_config.smoke.yaml
	$(PY) -m src.etl.pipeline --full-refresh --config $(SMOKE)
	# $(PY) -m src.features.builder --indicators --config $(SMOKE)            # Phase 2
	# $(PY) -m src.olap.run --all --config $(SMOKE)                           # Phase 2
	# $(PY) -m src.regime.regime_model --fit --config $(SMOKE)                # Phase 3
	# $(PY) -m src.models.train --config $(SMOKE)                             # Phase 4
	# $(PY) -m src.backtest.tune --config $(SMOKE)                            # Phase 5
	# $(PY) -m src.backtest.baselines --all --period validation --config $(SMOKE)   # Phase 5
	# $(PY) -m src.backtest.final_test --confirm --config $(SMOKE)           # Phase 6
	# $(PY) -m src.backtest.report --config $(SMOKE)                          # Phase 6
	# $(PY) -m src.paper.daily_job --no-etl --run-id paper_replay --from 2025-01-02 --to 2025-03-31 --config $(SMOKE)   # Phase 7

etl-full:
	$(PY) -m src.etl.pipeline --full-refresh --config $(CFG)

features:
	$(PY) -m src.features.builder --indicators --config $(CFG)

regime:
	$(PY) -m src.regime.regime_model --fit --config $(CFG)

train:
	$(PY) -m src.models.train --config $(CFG)

backtest:
	$(PY) -m src.backtest.tune --config $(CFG)
	$(PY) -m src.backtest.baselines --all --period validation --config $(CFG)

final-test: check
	$(PY) -m src.backtest.final_test --confirm --config $(CFG)
	$(PY) -m src.backtest.report --config $(CFG)

paper:
	$(PY) -m src.paper.daily_job --config $(CFG)

dashboard:
	$(PY) -m streamlit run src/dashboard/app.py
```

The pack ships this as a real `Makefile` at the repo root (recipe lines need tabs, so edit that file in place; do not re-create it from this block). Uncomment each smoke line when its phase's task is done. `requirements.txt` and `pyproject.toml` are shipped too.

---

## 8. Tests overview

Details and numeric fixtures: `TESTS.md`. Rules:

- Unit tests are fast (the whole `make check` under 60 seconds). Anything slower is marked `@pytest.mark.slow` (smoke, replay, dashboard).
- Tests use the synthetic generator and temporary SQLite files (`tmp_path`); never the real warehouse, never the network.
- `tests/conftest.py` provides fixtures: `cfg_smoke` (loaded smoke config pointing at `tmp_path`), `synthetic_market`, `warehouse` (ETL-loaded temporary database), `warehouse_full` (also indicators, regimes, panel, models; session scoped, marked slow).
- A test named in `06`§9 or `AGENTS.md` must exist under that name.
