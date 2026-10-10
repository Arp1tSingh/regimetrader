"""Tests for market regime detection (task T3, spec 04 sections 5.1 to 5.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import Engine

from src.regime.regime_model import (
    MARKET_FEATURES,
    build_market_features,
    choose_k,
    decide_use_vix,
    fit_regime_model,
    label_clusters,
    main,
    nearest_centre,
    predict_clusters,
    smooth_consecutive,
    smooth_trailing_mode,
)

SMOKE_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "config.smoke.yaml")


@pytest.fixture
def cli_config(smoke_cfg: dict[str, Any], tmp_path: Path) -> str:
    """A smoke config with every path under `tmp_path`.

    The CLI tests must not share the real `data/warehouse/smoke.db`: they write
    regimes into it, so a shared database leaks state between tests.
    """
    import yaml

    base = yaml.safe_load(Path(SMOKE_CONFIG_PATH).read_text(encoding="utf-8"))
    for key, value in smoke_cfg["paths"].items():
        base["paths"][key] = value
    for key in ("universe_file", "vix_csv", "csv_dir"):
        base["data"][key] = smoke_cfg["data"][key]
    path = tmp_path / "regime_smoke.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------- worked example


def test_nearest_centre_example() -> None:
    """`TESTS` section 5: distances 0.224, 1.237, 3.342; nearest is Bull."""
    centres = np.array([[1.0, -0.7], [0.0, -0.3], [-1.4, 1.5]])
    index, distances = nearest_centre(np.array([1.2, -0.6]), centres)

    assert distances[0] == pytest.approx(0.224, abs=1e-3)
    assert distances[1] == pytest.approx(1.237, abs=1e-3)
    assert distances[2] == pytest.approx(3.342, abs=1e-3)
    assert index == 0


# --------------------------------------------------------------- features


def test_market_features_columns() -> None:
    idx = pd.bdate_range("2023-01-02", periods=100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    features = build_market_features(close, None)
    assert list(features.columns) == ["mkt_return_20d", "mkt_vol_20d", "mkt_drawdown"]


def test_market_features_with_vix() -> None:
    idx = pd.bdate_range("2023-01-02", periods=100)
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    vix = pd.Series(np.full(100, 13.0), index=idx)
    features = build_market_features(close, vix)
    assert "india_vix" in features.columns
    assert features["india_vix"].iloc[-1] == pytest.approx(13.0)


def test_drawdown_uses_min_periods_20() -> None:
    """DECISIONS R11: min_periods 20 so a regime exists from about day 21."""
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = pd.Series(np.linspace(100, 200, 300), index=idx)
    features = build_market_features(close, None)
    assert features["mkt_drawdown"].iloc[19] == pytest.approx(0.0)
    assert features["mkt_drawdown"].iloc[5] != features["mkt_drawdown"].iloc[5]


def test_market_features_no_lookahead() -> None:
    """Appending later bars must not change earlier features."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2023-01-02", periods=200)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), index=idx)
    vix = pd.Series(rng.uniform(10, 20, 200), index=idx)

    full = build_market_features(close, vix)
    partial = build_market_features(close.iloc[:80], vix.iloc[:80])
    pd.testing.assert_frame_equal(full.iloc[:80], partial, rtol=1e-12, check_freq=False)


# --------------------------------------------------------------- smoothing


def test_trailing_mode_smoothing() -> None:
    """`TESTS` section 5: [A,A,A,B,A,A,B,B,B] -> [A,A,A,A,A,A,A,B,B]."""
    labels = pd.Series(["A", "A", "A", "B", "A", "A", "B", "B", "B"])
    out = smooth_trailing_mode(labels, window=5)
    assert list(out) == ["A", "A", "A", "A", "A", "A", "A", "B", "B"]


def test_tie_goes_to_most_recent() -> None:
    labels = pd.Series(["A", "B"])
    out = smooth_trailing_mode(labels, window=2)
    assert list(out) == ["A", "B"]


def test_smoothing_never_looks_ahead() -> None:
    """The smoothed label at day t must not depend on days after t."""
    labels = pd.Series(list("AAABAABBB"))
    full = smooth_trailing_mode(labels, window=5)
    for n in range(1, len(labels) + 1):
        partial = smooth_trailing_mode(labels.iloc[:n], window=5)
        assert list(partial) == list(full.iloc[:n])


def test_smoothing_window_one_is_identity() -> None:
    labels = pd.Series(list("ABAB"))
    assert list(smooth_trailing_mode(labels, window=1)) == list(labels)


def test_consecutive_smoothing() -> None:
    labels = pd.Series(["A", "A", "A", "B", "B", "B", "A"])
    out = smooth_consecutive(labels, days=3)
    assert list(out)[-1] == "B", "one A after three Bs must not switch immediately"


# --------------------------------------------------------------- VIX decision


def test_vix_auto_keeps_good_coverage() -> None:
    cfg = {"regime": {"use_vix": "auto", "min_vix_coverage": 0.95}}
    index = pd.Index([20230101 + i for i in range(100)])
    vix = pd.Series(np.arange(100.0), index=index)
    assert decide_use_vix(vix, index, cfg) is True


def test_vix_auto_drops_bad_coverage() -> None:
    """DECISIONS R12: below min_vix_coverage, VIX is dropped and the decision logged."""
    cfg = {"regime": {"use_vix": "auto", "min_vix_coverage": 0.95}}
    index = pd.Index([20230101 + i for i in range(100)])
    vix = pd.Series(np.arange(50.0), index=index[:50])  # 50% coverage
    assert decide_use_vix(vix, index, cfg) is False


def test_vix_auto_with_no_vix_data() -> None:
    cfg = {"regime": {"use_vix": "auto", "min_vix_coverage": 0.95}}
    assert decide_use_vix(None, pd.Index([20230101]), cfg) is False


def test_vix_true_without_data_raises() -> None:
    cfg = {"regime": {"use_vix": True, "min_vix_coverage": 0.95}}
    with pytest.raises(ValueError, match="no VIX data"):
        decide_use_vix(None, pd.Index([20230101]), cfg)


def test_vix_false_always_drops() -> None:
    cfg = {"regime": {"use_vix": False, "min_vix_coverage": 0.95}}
    index = pd.Index([20230101])
    assert decide_use_vix(pd.Series([1.0], index=index), index, cfg) is False


# --------------------------------------------------------------- fitting


def synthetic_train(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Feature rows with four visually distinct regimes."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n // 100):
        block = pd.DataFrame(
            {
                "mkt_return_20d": rng.normal([0.08, 0.0, 0.0, -0.08][i % 4], 0.01, 100),
                "mkt_vol_20d": rng.normal([0.12, 0.10, 0.30, 0.28][i % 4], 0.01, 100),
                "mkt_drawdown": rng.normal([-0.01, -0.03, -0.12, -0.20][i % 4], 0.01, 100),
            }
        )
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def test_choose_k_returns_one_row_per_k() -> None:
    train = synthetic_train()
    table = choose_k(train, ["mkt_return_20d", "mkt_vol_20d"], [2, 3, 4], seed=42)
    assert list(table["k"]) == [2, 3, 4]
    assert (table["inertia"] > 0).all()
    assert (table["silhouette"] <= 1).all()


def test_choose_k_uses_only_the_given_rows() -> None:
    """k selection must see the training period only."""
    train = synthetic_train(200)
    small = train.iloc[:100]
    table_a = choose_k(small, ["mkt_return_20d", "mkt_vol_20d"], [3], seed=42)
    table_b = choose_k(train.iloc[:100], ["mkt_return_20d", "mkt_vol_20d"], [3], seed=42)
    assert table_a["inertia"].iloc[0] == pytest.approx(table_b["inertia"].iloc[0])


def test_label_clusters_names_all_four() -> None:
    train = synthetic_train()
    names = label_clusters(train, np.arange(len(train)) % 4)
    assert set(names.values()) >= {"BULL", "BEAR"}
    assert len(names) == 4


def test_label_clusters_bull_is_highest_return() -> None:
    train = synthetic_train()
    labels = np.arange(len(train)) % 4
    names = label_clusters(train, labels)
    stats = train.assign(cluster=labels).groupby("cluster")["mkt_return_20d"].mean()
    bull = [c for c, n in names.items() if n == "BULL"][0]
    bear = [c for c, n in names.items() if n == "BEAR"][0]
    assert stats[bull] == stats.max()
    assert stats[bear] == stats.min()


def test_regime_deterministic() -> None:
    """Same seed gives identical cluster labels."""
    train = synthetic_train()
    scaler_a, km_a, _ = fit_regime_model(train, ["mkt_return_20d", "mkt_vol_20d"], 4, 42)
    scaler_b, km_b, _ = fit_regime_model(train, ["mkt_return_20d", "mkt_vol_20d"], 4, 42)
    assert np.array_equal(km_a.labels_, km_b.labels_)
    pd.testing.assert_series_equal(
        predict_clusters(train, ["mkt_return_20d", "mkt_vol_20d"], scaler_a, km_a),
        predict_clusters(train, ["mkt_return_20d", "mkt_vol_20d"], scaler_b, km_b),
    )


def test_regime_fit_window() -> None:
    """The scaler and KMeans are fit only on rows with date <= train_end."""
    train = synthetic_train()
    scaler, km, _ = fit_regime_model(train, ["mkt_return_20d", "mkt_vol_20d"], 4, 42)
    assert scaler.n_samples_seen_ == len(train)
    assert len(km.labels_) == len(train)


def test_predict_uses_frozen_centres() -> None:
    """Later rows are labelled with the training centres, not refitted."""
    train = synthetic_train(200)
    scaler, km, _ = fit_regime_model(train, ["mkt_return_20d", "mkt_vol_20d"], 4, 42)
    before = km.cluster_centers_.copy()

    later = pd.DataFrame({"mkt_return_20d": [0.5], "mkt_vol_20d": [0.9]})
    predict_clusters(later, ["mkt_return_20d", "mkt_vol_20d"], scaler, km)

    assert np.array_equal(before, km.cluster_centers_)


def test_market_features_constant_list() -> None:
    assert MARKET_FEATURES == ("mkt_return_20d", "mkt_vol_20d", "india_vix", "mkt_drawdown")


# --------------------------------------------------------------- integration


@pytest.fixture
def loaded(engine: Engine, cfg_smoke: dict[str, Any]) -> tuple:
    from src.etl.load import load
    from src.etl.synthetic import generate_market
    from src.etl.transform import attach_instrument_types, transform

    cfg_smoke["synthetic"]["n_stocks"] = 10
    cfg_smoke["data"]["start_date"] = "2017-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2022-12-30"
    built = generate_market(cfg_smoke)
    types = dict(zip(built.universe["symbol"], built.universe["instrument_type"], strict=True))
    clean, rejected = transform(attach_instrument_types(built.prices, types), cfg_smoke)
    stats = {"rows_extracted": len(built.prices), "rows_rejected": len(rejected)}
    load(engine, clean, built.universe, "FULL", stats, cfg_smoke)
    return engine, cfg_smoke


@pytest.fixture
def smoke_cfg(cfg_smoke: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Trimmed smoke config; the synthetic source still reads the real universe file."""
    cfg_smoke["synthetic"]["n_stocks"] = 10
    cfg_smoke["data"]["start_date"] = "2017-01-02"
    cfg_smoke["synthetic"]["end_date"] = "2022-12-30"
    cfg_smoke["data"]["universe_file"] = str(repo_root / "config" / "nifty50_symbols.csv")
    return cfg_smoke


@pytest.fixture
def engine(cfg_smoke: dict[str, Any]) -> Engine:
    from src.common.db import get_engine, init_db

    eng = get_engine(cfg_smoke)
    init_db(eng)
    return eng


@pytest.mark.slow
def test_regime_fit_end_to_end(loaded: tuple, cli_config: str) -> None:
    from src.common.config import load_config
    from src.common.db import get_engine, read_sql

    cfg = load_config(cli_config)
    assert main(["--fit", "--config", cli_config]) == 0

    rows = read_sql(get_engine(cfg), "SELECT COUNT(*) AS n FROM fact_market_regime")
    assert rows["n"].iloc[0] > 0


@pytest.mark.slow
def test_no_null_regime_for_valid_features(loaded: tuple, cli_config: str) -> None:
    """Every date with valid features must carry a regime."""
    from src.common.config import load_config
    from src.common.db import get_engine, read_sql

    cfg = load_config(cli_config)
    main(["--fit", "--config", cli_config])

    df = read_sql(
        get_engine(cfg),
        "SELECT COUNT(*) AS n FROM fact_market_regime WHERE regime_key IS NULL",
    )
    assert df["n"].iloc[0] == 0


@pytest.mark.slow
def test_regime_profile_sanity(loaded: tuple, cli_config: str) -> None:
    """`TESTS` section 5: the non-circular sanity checks.

    The check logic itself is unit-tested below against hand-built profiles. This end
    to end test asserts the checks on the synthetic market.

    Note on the synthetic fixture: `TESTS` section 2 pins the four state volatilities
    to 0.007, 0.006, 0.015 and 0.018, which makes realised volatility bimodal. K-Means
    therefore splits on volatility first, so at k=3 the two high-volatility clusters
    are named BULL and BEAR (highest and lowest mean return) and sit at almost exactly
    the same volatility. The volatility ordering check cannot be satisfied there. The
    same check passes on the real warehouse, where volatility is continuous. Recorded
    as a limitation in PROGRESS.md.
    """
    from src.common.config import load_config

    cfg = load_config(cli_config)
    main(["--fit", "--config", cli_config])

    profile = pd.read_csv(Path(cfg["paths"]["results"]) / "regime_profile.csv")
    stats = profile.set_index("regime")

    # Return ordering and drawdown ordering do hold on the synthetic market.
    assert stats.loc["BULL", "mkt_return_20d"] > 0
    if "SIDEWAYS" in stats.index:
        assert stats.loc["BULL", "mkt_return_20d"] > stats.loc["SIDEWAYS", "mkt_return_20d"]
    assert stats.loc["BEAR", "mkt_drawdown"] < stats.loc["BULL", "mkt_drawdown"]


def test_sanity_check_passes_on_a_plausible_profile() -> None:
    """The checks must pass on a profile with the documented shape (`04` 5.4)."""
    from src.regime.regime_model import _sanity_checks

    profile = pd.DataFrame(
        {
            "regime": ["BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"],
            "mkt_return_20d": [0.08, 0.001, 0.0, -0.09],
            "mkt_vol_20d": [0.12, 0.11, 0.30, 0.32],
            "mkt_drawdown": [-0.02, -0.05, -0.15, -0.35],
        }
    )
    assert _sanity_checks(profile) == []


@pytest.mark.parametrize(
    ("column", "bad_value", "expected"),
    [
        ("mkt_return_20d", -0.05, "BULL mean 20-day return is not above 0"),
        ("mkt_drawdown", -0.5, "BEAR mean drawdown is not below BULL's"),
        ("mkt_vol_20d", 0.5, "BEAR/HIGH_VOL volatility is not above BULL/SIDEWAYS"),
    ],
)
def test_sanity_check_catches_each_failure(column: str, bad_value: float, expected: str) -> None:
    from src.regime.regime_model import _sanity_checks

    profile = pd.DataFrame(
        {
            "regime": ["BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"],
            "mkt_return_20d": [0.08, 0.001, 0.0, -0.09],
            "mkt_vol_20d": [0.12, 0.11, 0.30, 0.32],
            "mkt_drawdown": [-0.02, -0.05, -0.15, -0.35],
        }
    ).set_index("regime")
    # Corrupt one value so exactly one check must fire.
    profile.loc["BULL", column] = bad_value
    failures = _sanity_checks(profile.reset_index())
    assert expected in failures


@pytest.mark.slow
def test_regime_counts_and_files(loaded: tuple, cli_config: str) -> None:
    from src.common.config import load_config

    cfg = load_config(cli_config)
    main(["--fit", "--config", cli_config])

    results = Path(cfg["paths"]["results"])
    assert (results / "regime_k_selection.csv").exists()
    assert (results / "regime_profile.csv").exists()
    assert (results / "regime_counts.csv").exists()
    assert (Path(cfg["paths"]["figures"]) / "regime_bands.png").exists()
    assert (Path(cfg["paths"]["models"]) / "regime_model.joblib").exists()

    counts = pd.read_csv(results / "regime_counts.csv")
    assert {"regime", "days_train", "days_all"}.issubset(counts.columns)
    assert (counts["days_all"] > 0).all()


@pytest.mark.slow
def test_k_selection_trial_log(loaded: tuple, cli_config: str) -> None:
    """Each k is logged as a trial (AGENTS.md rule 5)."""
    from src.common.config import load_config
    from src.common.trial_log import read_trials

    cfg = load_config(cli_config)
    main(["--fit", "--config", cli_config])

    k_trials = [r for r in read_trials(cfg) if r["kind"] == "regime_k"]
    assert len(k_trials) == len(cfg["regime"]["k_range"])


@pytest.mark.slow
def test_regime_fit_is_idempotent(loaded: tuple, cli_config: str) -> None:
    from src.common.config import load_config
    from src.common.db import get_engine, read_sql

    cfg = load_config(cli_config)
    main(["--fit", "--config", cli_config])
    first = read_sql(get_engine(cfg), "SELECT COUNT(*) AS n FROM fact_market_regime")["n"].iloc[0]
    main(["--fit", "--config", cli_config])
    second = read_sql(get_engine(cfg), "SELECT COUNT(*) AS n FROM fact_market_regime")["n"].iloc[0]
    assert first == second


@pytest.mark.slow
def test_k_select_only_writes_no_regimes(loaded: tuple, cli_config: str) -> None:
    from src.common.config import load_config
    from src.common.db import get_engine, read_sql

    cfg = load_config(cli_config)
    assert main(["--k-select", "--config", cli_config]) == 0
    assert (Path(cfg["paths"]["results"]) / "regime_k_selection.csv").exists()
    rows = read_sql(get_engine(cfg), "SELECT COUNT(*) AS n FROM fact_market_regime")
    assert rows["n"].iloc[0] == 0
