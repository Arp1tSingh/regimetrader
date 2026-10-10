"""Tests for `src.common.config` (task T0.3, spec 07 section 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.common.config import ConfigError, config_hash, load_config, validate_config

from .conftest import BASE_CONFIG, SMOKE_CONFIG


def test_loads_base_config_and_resolves_paths() -> None:
    cfg = load_config(str(BASE_CONFIG))

    assert cfg["project"]["name"] == "regimetrader"
    assert cfg["project"]["seed"] == 42
    for value in cfg["paths"].values():
        assert Path(value).is_absolute()


def test_smoke_config_inherits_base_values() -> None:
    base = load_config(str(BASE_CONFIG))
    smoke = load_config(str(SMOKE_CONFIG))

    # Inherited unchanged from the base.
    assert smoke["project"]["seed"] == base["project"]["seed"]
    assert smoke["strategy"]["entry_base"] == base["strategy"]["entry_base"]
    assert smoke["strategy"]["slots"] == 5  # overridden by the smoke file
    # Overridden by the smoke file.
    assert smoke["data"]["source"] == "synthetic"
    assert smoke["paths"]["db"].endswith("smoke.db")
    assert smoke["splits"]["train_end"] == "2021-12-31"


def test_smoke_config_does_not_leak_into_base(tmp_path: Path) -> None:
    """Loading an override must not mutate the cached or on-disk base."""
    load_config(str(SMOKE_CONFIG))
    assert load_config(str(BASE_CONFIG))["strategy"]["slots"] == 10


def test_relative_path_resolves_against_repo_root() -> None:
    from src.common.config import REPO_ROOT

    cfg = load_config(str(BASE_CONFIG))
    assert cfg["paths"]["db"] == str(REPO_ROOT / "data" / "warehouse" / "regimetrader.db")
    assert cfg["data"]["universe_file"] == str(REPO_ROOT / "config" / "nifty50_symbols.csv")
    assert Path(cfg["paths"]["db"]).is_absolute()


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {"extends": "config/config.yaml", "totally_unknown": {"a": 1}, "project": {"x": 2}}
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(override))

    assert cfg["totally_unknown"] == {"a": 1}  # preserved, not an error
    assert cfg["project"]["name"] == "regimetrader"  # base still merged


def test_config_hash_is_stable_and_order_independent() -> None:
    a = load_config(str(BASE_CONFIG))
    b = load_config(str(BASE_CONFIG))
    assert config_hash(a) == config_hash(b)

    reordered = {k: a[k] for k in sorted(a, reverse=True)}
    assert config_hash(reordered) == config_hash(a)


def test_config_hash_changes_with_content() -> None:
    cfg = load_config(str(BASE_CONFIG))
    before = config_hash(cfg)
    cfg["project"]["seed"] = 7
    assert config_hash(cfg) != before


def test_frozen_config_applied_last_and_restricted(tmp_path: Path) -> None:
    frozen = tmp_path / "frozen.yaml"
    frozen.write_text(
        yaml.safe_dump(
            {
                "strategy": {"selected_model": "rf", "entry_base": 0.61, "slots": 3},
                "project": {"seed": 999},
                "backtest": {"initial_capital": 1},
            }
        ),
        encoding="utf-8",
    )
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump({"extends": "config/config.yaml", "paths": {"frozen_config": str(frozen)}}),
        encoding="utf-8",
    )

    cfg = load_config(str(override))

    assert cfg["strategy"]["selected_model"] == "rf"
    assert cfg["strategy"]["entry_base"] == 0.61
    assert cfg["strategy"]["slots"] == 3
    # Only the three whitelisted strategy keys may come from the frozen file.
    assert cfg["project"]["seed"] == 42
    assert cfg["backtest"]["initial_capital"] == 1_000_000


def test_frozen_config_absent_is_fine(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump({"extends": "config/config.yaml", "paths": {"frozen_config": str(missing)}}),
        encoding="utf-8",
    )
    assert load_config(str(override))["project"]["seed"] == 42


def test_output_folders_are_created(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    all_keys = (
        "db",
        "raw",
        "external",
        "processed",
        "models",
        "results",
        "figures",
        "daily",
        "logs",
    )
    paths = {key: str(tmp_path / key) for key in all_keys}
    paths["frozen_config"] = str(tmp_path / "frozen.yaml")
    override.write_text(
        yaml.safe_dump({"extends": "config/config.yaml", "paths": paths}), encoding="utf-8"
    )

    cfg = load_config(str(override))

    for key in ("raw", "external", "processed", "models", "results", "figures", "daily", "logs"):
        assert Path(cfg["paths"][key]).is_dir()
    assert Path(cfg["paths"]["db"]).parent.is_dir()


def test_missing_required_section_raises() -> None:
    cfg = load_config(str(BASE_CONFIG))
    del cfg["regime"]
    with pytest.raises(ConfigError, match="regime"):
        validate_config(cfg)


def test_missing_split_date_raises() -> None:
    cfg = load_config(str(BASE_CONFIG))
    del cfg["splits"]["validation_end"]
    with pytest.raises(ConfigError, match="splits"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "splits",
    [
        # train_end == validation_start
        {
            "train_end": "2023-01-01",
            "validation_start": "2023-01-01",
            "validation_end": "2023-12-31",
            "test_start": "2024-01-01",
        },
        # validation_end == test_start
        {
            "train_end": "2022-12-31",
            "validation_start": "2023-01-01",
            "validation_end": "2024-01-01",
            "test_start": "2024-01-01",
        },
        # validation_start after validation_end
        {
            "train_end": "2022-12-31",
            "validation_start": "2023-06-01",
            "validation_end": "2023-01-01",
            "test_start": "2024-01-01",
        },
    ],
)
def test_unordered_splits_raise(splits: dict[str, str]) -> None:
    cfg = load_config(str(BASE_CONFIG))
    cfg["splits"] = splits
    with pytest.raises(ConfigError, match="ordered"):
        validate_config(cfg)


def test_ordered_splits_are_accepted() -> None:
    cfg = load_config(str(BASE_CONFIG))
    validate_config(cfg)  # must not raise


def test_unordered_splits_in_file_raise(tmp_path: Path) -> None:
    """The ordering check must fire through load_config, not only direct validation."""
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump(
            {
                "extends": "config/config.yaml",
                "splits": {
                    "train_end": "2022-12-31",
                    "validation_start": "2023-01-01",
                    "validation_end": "2022-06-30",
                    "test_start": "2024-01-01",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="ordered"):
        load_config(str(override))


def test_env_var_config_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RT_CONFIG", str(SMOKE_CONFIG))
    assert load_config()["data"]["source"] == "synthetic"


def test_base_path_as_override_is_skipped() -> None:
    """Passing the base config explicitly must not merge it onto itself."""
    cfg = load_config(str(BASE_CONFIG))
    assert cfg["data"]["source"] == "yfinance"
