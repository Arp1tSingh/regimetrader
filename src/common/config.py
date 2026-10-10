"""Configuration loading with `extends`, override precedence and the frozen config.

Precedence (later wins), deep-merged:

1. `config/config.yaml` (the base).
2. The override file given by `path` or `$RT_CONFIG`, if any and if it is not the
   base file itself. An override may carry `extends: <file>`; the extended file is
   merged first.
3. The frozen config at `paths.frozen_config` as resolved by steps 1 and 2. It is
   applied **last** and may only set `strategy.selected_model`,
   `strategy.entry_base` and `strategy.slots` (spec `07` section 3, `DECISIONS` R24).

No magic numbers: every tunable in the system is read from here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

#: Repository root: the parent of the `src/` package.
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

#: Keys under `paths:` naming a directory. Created on demand by `load_config`.
DIR_PATH_KEYS = ("raw", "external", "processed", "models", "results", "figures", "daily", "logs")
#: Keys under `paths:` naming a file.
FILE_PATH_KEYS = ("db", "frozen_config")
#: `data:` keys naming a filesystem location.
DATA_PATH_KEYS = ("universe_file", "vix_csv", "csv_dir")

#: The only keys a frozen config may set (`DECISIONS` R24).
FROZEN_KEYS = ("selected_model", "entry_base", "slots")

#: Top-level sections that must be present.
REQUIRED_SECTIONS = (
    "project",
    "paths",
    "data",
    "quality",
    "splits",
    "features",
    "regime",
    "target",
    "models",
    "strategy",
    "backtest",
    "tuning",
    "paper",
    "olap",
    "synthetic",
)

#: Split dates that must be present and ordered.
REQUIRED_SPLIT_KEYS = ("train_end", "validation_start", "validation_end", "test_start")


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed or internally inconsistent."""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file into a dict, raising `ConfigError` on failure."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at the top level")
    return data


def _resolve_ref(ref: str, *, relative_to: Path) -> Path:
    """Resolve an `extends` reference or CLI path against the repo root."""
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    # Prefer the repository root (config files use repo-relative references such as
    # `extends: config/config.yaml`), then fall back to the referring file's folder.
    from_root = REPO_ROOT / candidate
    return from_root if from_root.exists() else (relative_to / candidate)


def _load_with_extends(path: Path, _seen: frozenset[Path] | None = None) -> dict[str, Any]:
    """Load a YAML file, deep-merging its `extends` parent underneath it."""
    path = path.resolve()
    seen = _seen or frozenset()
    if path in seen:
        raise ConfigError(f"circular `extends` chain at {path}")
    seen = seen | {path}

    cfg = _read_yaml(path)
    parent_ref = cfg.pop("extends", None)
    if parent_ref is None:
        return cfg

    parent_path = _resolve_ref(str(parent_ref), relative_to=path.parent)
    if not parent_path.exists():
        raise ConfigError(f"`extends` target not found: {parent_path} (from {path})")
    parent = _load_with_extends(parent_path, seen)
    return deep_merge(parent, cfg)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`. Later wins; lists are replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    """Turn every repo-relative path in the config into an absolute path."""
    path_keys = tuple(DIR_PATH_KEYS) + FILE_PATH_KEYS
    for section, keys in (("paths", path_keys), ("data", DATA_PATH_KEYS)):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if isinstance(value, str) and value:
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = REPO_ROOT / candidate
                cfg[section][key] = str(candidate)
    return cfg


def _ensure_output_dirs(cfg: dict[str, Any]) -> None:
    """Create output folders on demand so downstream writers never fail on a missing path."""
    for key in DIR_PATH_KEYS:
        target = cfg["paths"].get(key)
        if target:
            Path(target).mkdir(parents=True, exist_ok=True)
    for key in FILE_PATH_KEYS:
        target = cfg["paths"].get(key)
        if target:
            Path(target).parent.mkdir(parents=True, exist_ok=True)


def validate_config(cfg: dict[str, Any]) -> None:
    """Fail fast on missing required sections and out-of-order splits.

    Raises:
        ConfigError: If a required section or split date is absent, or the
            split dates are not ordered.
    """
    missing = [section for section in REQUIRED_SECTIONS if section not in cfg]
    if missing:
        raise ConfigError(f"config is missing required section(s): {', '.join(missing)}")

    splits = cfg["splits"]
    missing_dates = [key for key in REQUIRED_SPLIT_KEYS if not splits.get(key)]
    if missing_dates:
        raise ConfigError(
            f"config `splits` is missing required date(s): {', '.join(missing_dates)}"
        )

    train_end = str(splits["train_end"])
    val_start = str(splits["validation_start"])
    val_end = str(splits["validation_end"])
    test_start = str(splits["test_start"])
    if not (train_end < val_start <= val_end < test_start):
        raise ConfigError(
            "config `splits` dates must be ordered as "
            "train_end < validation_start <= validation_end < test_start; got "
            f"train_end={train_end}, validation_start={val_start}, "
            f"validation_end={val_end}, test_start={test_start}"
        )


def _apply_frozen(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen config last, whitelisting only the three strategy keys."""
    frozen_path = cfg.get("paths", {}).get("frozen_config")
    if not frozen_path:
        return cfg
    path = Path(frozen_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        return cfg

    frozen = _load_with_extends(path)
    strategy = frozen.get("strategy") or {}
    for key in FROZEN_KEYS:
        if key in strategy:
            cfg.setdefault("strategy", {})[key] = strategy[key]
    if isinstance(frozen.get("meta"), dict):
        cfg["frozen_meta"] = frozen["meta"]
    return cfg


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load the effective configuration.

    Args:
        path: Optional override config. Also read from `$RT_CONFIG` when unset.
            Ignored when it resolves to the base config itself.

    Returns:
        The merged config with every path absolute and output folders created.

    Raises:
        ConfigError: If a file is missing or malformed, a required key is absent,
            or the split dates are not ordered.
    """
    if not BASE_CONFIG_PATH.exists():
        raise ConfigError(f"base config not found: {BASE_CONFIG_PATH}")
    cfg = _load_with_extends(BASE_CONFIG_PATH)

    override_ref = path if path else os.environ.get("RT_CONFIG")
    if override_ref:
        override_path = _resolve_ref(override_ref, relative_to=Path.cwd())
        if override_path.resolve() != BASE_CONFIG_PATH.resolve():
            if not override_path.exists():
                raise ConfigError(f"override config not found: {override_path}")
            cfg = deep_merge(cfg, _load_with_extends(override_path))

    cfg = _apply_frozen(cfg)
    cfg = _resolve_paths(cfg)
    validate_config(cfg)
    _ensure_output_dirs(cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable 12-character sha256 of the config, for manifests and change detection."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
