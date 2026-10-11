VENV := .venv
CFG  ?= config/config.yaml
SMOKE := config/config.smoke.yaml

# POSIX paths on Linux/macOS, Windows Scripts/ paths on native Windows.
# Recipes below are otherwise identical on both.
ifeq ($(OS),Windows_NT)
  PY    := $(VENV)/Scripts/python.exe
  PIP   := $(VENV)/Scripts/python.exe -m pip
  BOOTPY := python
  CLEAN := if exist "$(VENV)" rmdir /s /q "$(VENV)"
  SMOKE_DIRS := data\\raw_smoke data\\processed_smoke models_smoke reports_smoke logs_smoke
  SMOKE_FILES := data\\warehouse\\smoke.db config\\frozen_config.smoke.yaml
  SMOKE_CLEAN := $(foreach d,$(SMOKE_DIRS),if exist "$(d)" rmdir /s /q "$(d)" &) $(foreach f,$(SMOKE_FILES),if exist "$(f)" del /q "$(f)" &)
else
  PY    := $(VENV)/bin/python
  PIP   := $(VENV)/bin/python -m pip
  BOOTPY := python3
  CLEAN := rm -rf $(VENV)
  SMOKE_CLEAN := rm -rf data/warehouse/smoke.db data/raw_smoke data/processed_smoke models_smoke reports_smoke logs_smoke config/frozen_config.smoke.yaml
endif

.PHONY: setup lint format check test smoke etl-full features regime train backtest final-test paper dashboard clean

setup:
	$(BOOTPY) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

lint:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check src tests --fix

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check src tests --fix

check: lint
	$(PY) -m pytest -m "not slow" -q

test:
	$(PY) -m pytest -q

# Whole pipeline on synthetic data. Append one line per phase as it lands. No network.
smoke:
	$(SMOKE_CLEAN)
	$(PY) -m src.etl.pipeline --full-refresh --config $(SMOKE)
	$(PY) -m src.features.builder --indicators --config $(SMOKE)
	$(PY) -m src.olap.run --all --config $(SMOKE)
	$(PY) -m src.regime.regime_model --fit --config $(SMOKE)
	$(PY) -m src.models.train --config $(SMOKE)
	$(PY) -m src.backtest.tune --config $(SMOKE)
	$(PY) -m src.backtest.baselines --all --period validation --config $(SMOKE)
	@REM Phase 6: $(PY) -m src.backtest.final_test --confirm --config $(SMOKE)
	@REM Phase 6: $(PY) -m src.backtest.report --config $(SMOKE)
	@REM Phase 7: $(PY) -m src.paper.daily_job --no-etl --run-id paper_replay --from 2025-01-02 --to 2025-03-31 --config $(SMOKE)

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

clean:
	$(CLEAN)

