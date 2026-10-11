"""Guards on the one-time test evaluation (`spec 06` section 8; `DECISIONS` R13).

These must pass before `final_test.py` is ever run on the real warehouse. They run
entirely on the synthetic smoke warehouse and must never touch the test period, the
real lock file or the real results directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.backtest.final_test import (
    EXIT_ALREADY_RUN,
    EXIT_DATA_FAILURE,
    lock_path,
    preconditions,
    read_lock,
    run,
    write_lock,
)


def _frozen(cfg_smoke: dict[str, Any], tmp_path: Path, model: str = "logreg") -> Path:
    """Write a valid frozen config at the smoke config's frozen path."""
    import yaml

    path = Path(cfg_smoke["paths"]["frozen_config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"strategy": {"selected_model": model, "entry_base": 0.55, "slots": 10}}),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------- lock file


def test_no_lock_means_the_test_period_is_untouched(cfg_smoke: dict[str, Any]) -> None:
    assert read_lock(cfg_smoke) is None
    assert not lock_path(cfg_smoke).exists()


def test_lock_is_written_with_the_required_fields(cfg_smoke: dict[str, Any], loaded: tuple) -> None:
    """R13: timestamp, commit, config hash, frozen path, run count, data end date."""
    engine, cfg = loaded
    _frozen(cfg, Path(cfg["paths"]["frozen_config"]))
    lock = write_lock(cfg, reason=None, count=1)

    for field in (
        "timestamp",
        "git_commit",
        "config_hash",
        "frozen_config_path",
        "run_count",
        "data_end_date",
    ):
        assert field in lock, f"spec 06 section 8 requires {field}"
    assert lock["run_count"] == 1
    assert len(lock["config_hash"]) == 64, "config_hash must be a SHA-256 hex digest"
    assert lock_path(cfg).exists()


def test_second_run_is_refused_without_allow_rerun(
    cfg_smoke: dict[str, Any], loaded: tuple
) -> None:
    """The core of the "evaluated once" rule."""
    _engine, cfg = loaded
    _frozen(cfg, Path(cfg["paths"]["frozen_config"]))
    write_lock(cfg, reason=None, count=1)

    outcome = run(cfg, confirmed=True, allow_rerun=None, with_b4=False)
    assert outcome["refused"], "a completed run must block the next one"
    assert "already been evaluated" in outcome["message"]
    assert "--allow-rerun" in outcome["message"], "the message must say how to override"


def test_allow_rerun_records_the_reason_and_increments_the_count(
    cfg_smoke: dict[str, Any], loaded: tuple
) -> None:
    _engine, cfg = loaded
    _frozen(cfg, Path(cfg["paths"]["frozen_config"]))
    write_lock(cfg, reason=None, count=1)

    lock = write_lock(cfg, reason="crashed before scoring, fixing the traceback", count=2)
    assert lock["run_count"] == 2
    assert "traceback" in lock["rerun_reason"]

    log = Path(cfg["paths"]["results"]) / "test_run_log.jsonl"
    assert log.exists(), "a rerun must be recorded"
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x]
    assert lines[-1]["run_count"] == 2


# --------------------------------------------------------------- preconditions


def test_refuses_without_a_frozen_config(cfg_smoke: dict[str, Any]) -> None:
    message = preconditions(cfg_smoke, confirmed=True)
    assert message is not None
    assert "frozen config" in message


def test_refuses_a_frozen_config_with_no_selected_model(cfg_smoke: dict[str, Any]) -> None:
    import yaml

    path = Path(cfg_smoke["paths"]["frozen_config"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"strategy": {"selected_model": None}}), encoding="utf-8")
    message = preconditions(cfg_smoke, confirmed=True)
    assert message is not None
    assert "selected_model" in message


def test_refuses_without_confirm(cfg_smoke: dict[str, Any]) -> None:
    _frozen(cfg_smoke, Path(cfg_smoke["paths"]["frozen_config"]))
    message = preconditions(cfg_smoke, confirmed=False)
    assert message is not None
    assert "--confirm" in message


def test_accepts_a_valid_frozen_config_with_confirm(cfg_smoke: dict[str, Any]) -> None:
    _frozen(cfg_smoke, Path(cfg_smoke["paths"]["frozen_config"]))
    assert preconditions(cfg_smoke, confirmed=True) is None


# --------------------------------------------------------------- exit codes


def _redirected_config(cfg_smoke: dict[str, Any], tmp_path: Path) -> Path:
    """Write the smoke config with every path redirected under `tmp_path`.

    The CLI-level tests must never load the shipped `config/config.smoke.yaml`, whose
    `paths` point into the real repository: doing so made an earlier version of this
    file write a `TEST_RUN_LOCK.json` into `reports_smoke/`.
    """
    import yaml

    data = yaml.safe_load(Path("config/config.smoke.yaml").read_text(encoding="utf-8"))
    # Keep the fixture's populated database, but redirect every output path so the
    # CLI cannot write into the real repository.
    data["paths"] = dict(cfg_smoke["paths"])
    data["paths"]["results"] = str(tmp_path / "results")
    data["paths"]["frozen_config"] = str(tmp_path / "frozen_cli.yaml")
    config_path = tmp_path / "cli_smoke.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def test_refusal_after_a_completed_run_exits_with_code_3(
    cfg_smoke: dict[str, Any], loaded: tuple, tmp_path: Path
) -> None:
    from src.backtest.final_test import main

    _engine, cfg = loaded
    config_path = _redirected_config(cfg_smoke, tmp_path)

    from src.common.config import load_config

    redirected = load_config(str(config_path))
    _frozen(redirected, Path(redirected["paths"]["frozen_config"]))
    write_lock(redirected, reason=None, count=1)

    code = main(["--config", str(config_path), "--confirm", "--no-b4"])
    assert code == EXIT_ALREADY_RUN


def test_refusal_without_a_frozen_config_is_a_data_failure(
    cfg_smoke: dict[str, Any], loaded: tuple, tmp_path: Path
) -> None:
    from src.backtest.final_test import main

    _engine, cfg = loaded
    config_path = _redirected_config(cfg_smoke, tmp_path)
    code = main(["--config", str(config_path), "--confirm", "--no-b4"])
    assert code == EXIT_DATA_FAILURE


# --------------------------------------------------------------- period guard


def test_final_test_is_the_only_production_caller_of_allow_test_period() -> None:
    """No module outside tests may pass `allow_test_period=True`.

    Parsed with `ast` rather than grepped: a plain text search also matches the prose
    in docstrings that *describes* the rule, which would fail this test for the wrong
    reason.
    """
    import ast

    repo = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (repo / "src").rglob("*.py"):
        if path.name == "final_test.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "allow_test_period"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(f"{path.relative_to(repo)}:{node.lineno}")
    assert not offenders, f"only final_test.py may pass allow_test_period=True: {offenders}"


def test_run_backtest_still_refuses_the_test_period_by_default() -> None:
    """The guard is not weakened by final_test existing."""
    import inspect

    from src.backtest.engine import run_backtest

    source = inspect.getsource(run_backtest)
    assert "allow_test_period: bool = False" in source


@pytest.mark.slow
def test_smoke_final_test_runs_end_to_end_and_writes_its_outputs(
    cfg_smoke: dict[str, Any], loaded: tuple, tmp_path: Path
) -> None:
    """The whole module works on synthetic data, which is the only way to rehearse it."""
    from src.backtest.final_test import main

    _engine, cfg = loaded
    # Point at a config file whose frozen path and results dir are the fixture's.
    import yaml

    base = yaml.safe_load(Path("config/config.smoke.yaml").read_text(encoding="utf-8"))
    base["paths"] = dict(cfg["paths"])
    base["paths"]["results"] = str(tmp_path / "ft_results")
    # The `loaded` fixture's synthetic market ends 2022-12-30, so validation is
    # shortened to leave a real test window inside the data that exists.
    base["splits"]["validation_end"] = "2022-06-30"
    base["splits"]["test_start"] = "2022-07-01"
    base["splits"]["test_end"] = "2022-12-30"
    config_path = tmp_path / "ft_smoke.yaml"
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")

    # A frozen config must exist at the redirected path, or the run correctly refuses.
    frozen = Path(base["paths"]["frozen_config"])
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(
        yaml.safe_dump({"strategy": {"selected_model": "logreg", "entry_base": 0.55, "slots": 5}}),
        encoding="utf-8",
    )

    assert main(["--config", str(config_path), "--confirm", "--no-b4"]) == 0

    results = Path(base["paths"]["results"])
    for name in (
        "summary_test.csv",
        "yearly_test.csv",
        "bootstrap_sharpe.csv",
        "regime_breakdown.csv",
        "TEST_RUN_LOCK.json",
        "frozen_config_used.yaml",
    ):
        assert (results / name).exists(), f"{name} was not written"

    summary = pd.read_csv(results / "summary_test.csv")
    assert set(summary["run_id"]) >= {
        "backtest_main",
        "test_V0",
        "test_V1",
        "test_V2_nocost",
        "test_B1",
        "test_B2",
        "test_B3",
    }
    assert len(summary) == 7
