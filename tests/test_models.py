"""Tests for the panel, preprocessing, walk-forward and evaluation (task T4, TESTS section 4/5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.panel import (
    FEATURE_COLS_ALL,
    NEXT_RETURN_COL,
    TARGET_COL,
    get_feature_cols,
)
from src.models.evaluate import classification_report, majority_baseline
from src.models.factory import make_model, resolve_name
from src.models.preprocessing import Winsorizer
from src.models.train import walk_forward_predict

SMOKE_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "config" / "config.smoke.yaml")


# --------------------------------------------------------------- feature names


def test_feature_cols_all_has_nineteen_names() -> None:
    """`04` section 2.1: 19 names in exactly the documented order."""
    assert len(FEATURE_COLS_ALL) == 19
    assert FEATURE_COLS_ALL[0] == "ret_1d"
    assert FEATURE_COLS_ALL[-1] == "regime_bear"
    assert list(FEATURE_COLS_ALL[15:]) == [
        "regime_bull",
        "regime_sideways",
        "regime_high_vol",
        "regime_bear",
    ]


def test_get_feature_cols_drops_vix_when_unused() -> None:
    """DECISIONS R12: without VIX the active features are 18 names."""
    with_vix = get_feature_cols(True)
    without = get_feature_cols(False)
    assert len(with_vix) == 19
    assert len(without) == 18
    assert "india_vix" not in without
    # Four regime columns are always present, even when k = 3.
    assert sum(1 for c in without if c.startswith("regime_")) == 4


# --------------------------------------------------------------- winsorizer


def test_winsorizer_train_only() -> None:
    """`TESTS` section 5: limits equal the fit data's percentiles and never move."""
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(5, 2, 1000)})
    w = Winsorizer(1, 99).fit(train)
    limits = dict(w.limits_)

    assert limits["a"][0] == pytest.approx(np.percentile(train["a"], 1))
    assert limits["a"][1] == pytest.approx(np.percentile(train["a"], 99))

    other = pd.DataFrame({"a": rng.normal(50, 50, 500), "b": rng.normal(-50, 50, 500)})
    w.transform(other)
    assert dict(w.limits_) == limits, "transforming other data must not change the limits"


def test_winsorizer_clips() -> None:
    train = pd.DataFrame({"a": list(range(100))})
    w = Winsorizer(1, 99).fit(train)
    out = w.transform(pd.DataFrame({"a": [-1000.0, 1000.0]}))
    assert out["a"].iloc[0] >= w.limits_["a"][0]
    assert out["a"].iloc[1] <= w.limits_["a"][1]


def test_winsorizer_leaves_one_hot_alone() -> None:
    """Clipping a 0/1 one-hot would turn it into a constant."""
    train = pd.DataFrame({"regime_bull": [0.0] * 50 + [1.0] * 50, "x": np.arange(100.0)})
    w = Winsorizer(1, 99).fit(train)
    assert "regime_bull" not in w.limits_


def test_winsorizer_sklearn_compatible() -> None:
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([("w", Winsorizer()), ("m", make_model("logreg", _cfg()))])
    pipeline.fit(pd.DataFrame({"a": np.arange(50.0)}), pd.Series([0, 1] * 25))
    assert pipeline.predict_proba(pd.DataFrame({"a": [1.0]}))[:, 1].shape == (1,)


# --------------------------------------------------------------- factory


def _cfg() -> dict[str, Any]:
    from src.common.config import load_config

    return load_config(str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"))


@pytest.mark.parametrize("name", ["logreg", "tree", "rf", "xgb"])
def test_models_fit_predict(name: str) -> None:
    """All four models fit on a small sample and return probabilities in [0, 1]."""
    rng = np.random.default_rng(2)
    x = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = (x["a"] + rng.normal(0, 0.3, 300) > 0).astype(int)

    model = make_model(name, _cfg())
    model.fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_alias_resolves() -> None:
    assert resolve_name("lr") == "logreg"
    assert resolve_name("rf") == "rf"


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model("deep_learning", _cfg())


# --------------------------------------------------------------- metrics


def test_classification_metrics_example() -> None:
    """`TESTS` section 5: TP=280 FP=230 FN=210 TN=280."""
    y_true = np.array([1] * 490 + [0] * 510)
    y_pred = np.array([1] * 280 + [0] * 210 + [1] * 230 + [0] * 280)

    m = classification_report(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.560)
    assert m["precision"] == pytest.approx(280 / 510, abs=1e-3)
    assert m["recall"] == pytest.approx(280 / 490, abs=1e-3)
    assert m["f1"] == pytest.approx(0.560, abs=1e-3)
    assert m["tp"] == 280 and m["fp"] == 230 and m["fn"] == 210 and m["tn"] == 280


def test_majority_baseline_example() -> None:
    """Actual up = 490 of 1000, so always-down scores 0.51."""
    y_true = np.array([1] * 490 + [0] * 510)
    assert majority_baseline(y_true) == pytest.approx(0.51)


def test_baseline_always_predicted_alongside_accuracy() -> None:
    """`04` section 7: accuracy alone is meaningless without the baseline."""
    y_true = np.array([1] * 490 + [0] * 510)
    y_pred = np.array([1] * 280 + [0] * 210 + [1] * 230 + [0] * 280)
    m = classification_report(y_true, y_pred)
    assert "majority_baseline" in m
    assert m["edge_over_baseline"] == pytest.approx(m["accuracy"] - m["majority_baseline"])


# --------------------------------------------------------------- walk-forward


def _fake_panel(n_years: int = 4, per_day: int = 40, seed: int = 6) -> pd.DataFrame:
    """A small labelled panel spanning several calendar years."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", f"{2017 + n_years}-12-31")
    frames = []
    for date in dates:
        block = pd.DataFrame(
            {
                "symbol": [f"S{i:02d}.NS" for i in range(per_day)],
                "date_key": int(date.strftime("%Y%m%d")),
                "trade_date": date,
            }
        )
        features = rng.normal(size=(per_day, 3))
        block["a"] = features[:, 0]
        block["b"] = features[:, 1]
        block["c"] = features[:, 2]
        block[TARGET_COL] = (features[:, 0] > 0).astype(float)
        block[NEXT_RETURN_COL] = np.where(block[TARGET_COL] > 0, 0.01, -0.01)
        frames.append(block)
    return pd.concat(frames).set_index("trade_date").sort_index()


def test_walk_forward_embargo() -> None:
    """For every test year, `max(train index) < year_start - embargo_days`."""
    panel = _fake_panel()
    predictions, fitted = walk_forward_predict(
        panel, ["a", "b", "c"], lambda: make_model("tree", _cfg()), 2020, embargo_days=5
    )
    assert not predictions.empty
    for year, (_model, train_max) in fitted.items():
        year_start = pd.Timestamp(year=int(year), month=1, day=1)
        assert train_max < year_start - pd.Timedelta(days=5)
        # No row from the test year leaked into training.
        assert (panel[panel.index < year_start - pd.Timedelta(days=5)].index.year < year).all()


def test_walk_forward_predictions_in_range() -> None:
    panel = _fake_panel()
    predictions, _ = walk_forward_predict(
        panel, ["a", "b", "c"], lambda: make_model("logreg", _cfg()), 2020, embargo_days=5
    )
    assert predictions["prob_up"].between(0, 1).all()
    assert set(predictions["model_year"]) >= {2020, 2021}


def test_walk_forward_is_deterministic() -> None:
    panel = _fake_panel()
    a, _ = walk_forward_predict(panel, ["a", "b", "c"], lambda: make_model("rf", _cfg()), 2020)
    b, _ = walk_forward_predict(panel, ["a", "b", "c"], lambda: make_model("rf", _cfg()), 2020)
    pd.testing.assert_series_equal(a["prob_up"], b["prob_up"])


# --------------------------------------------------------------- panel


def test_forbidden_patterns() -> None:
    """`TESTS` section 4: no forward shift outside the label, no bfill, no shuffle."""
    import src.features.panel as panel_module

    source = Path(panel_module.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        if "shift(-" in line and "# label-only" not in line:
            pytest.fail(f"forward shift outside the label: {line.strip()}")

    for path in (Path(panel_module.__file__),):
        text = path.read_text(encoding="utf-8")
        assert ".bfill(" not in text
        assert "train_test_split(" not in text
        assert "shuffle=True" not in text
