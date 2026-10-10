"""Validation diagnostics (`DECISIONS` R35, R36).

These guard the two errors that would inflate a result without anyone noticing:
a bootstrap that ignores day-level dependence, and a baseline measured on a
different set of rows than the accuracy it is compared against.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.diagnostics import (
    accuracy_edge,
    cluster_bootstrap,
    model_diagnostics,
    quintile_spread,
)
from src.models.evaluate import classifier_metrics_row, majority_baseline


def frame(n_days: int = 40, n_stocks: int = 10, seed: int = 0) -> pd.DataFrame:
    """A frame with a known, deliberately planted cross-sectional signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        for i in range(n_stocks):
            proba = rng.uniform(0.30, 0.70)
            # A planted effect: higher prob_up earns a higher next-day return.
            ret = 0.02 * (proba - 0.50) + rng.normal(0, 0.01)
            rows.append(
                {
                    "date_key": 20230101 + day,
                    "symbol": f"S{i:02d}",
                    "prob_up": proba,
                    "fwd_return": ret,
                }
            )
    out = pd.DataFrame(rows)
    out["target_up"] = (out["fwd_return"] > 0).astype(int)
    return out


# --------------------------------------------------------------- quintile spread


def test_quintile_spread_detects_a_real_planted_signal() -> None:
    """A frame built with a positive prob_up effect must give a positive spread."""
    spread = quintile_spread(frame())
    assert spread > 0


def test_quintile_spread_is_zero_when_probability_carries_no_information() -> None:
    """Shuffling prob_up against returns must destroy any spread."""
    rng = np.random.default_rng(1)
    df = frame()
    shuffled = rng.permutation(df["prob_up"].to_numpy())
    df["prob_up"] = shuffled
    assert abs(quintile_spread(df)) < 0.01


def test_quintile_spread_is_formed_within_each_day() -> None:
    """A day-wide shift in returns cannot create a spread: quintiles are per day."""
    df = frame()
    baseline = quintile_spread(df)
    df["fwd_return"] = df["fwd_return"] + 0.05  # same +5% on every row, every day
    assert quintile_spread(df) == pytest.approx(baseline)


# --------------------------------------------------------------- bootstrap


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    df = frame()
    point = quintile_spread(df)
    low, high = cluster_bootstrap(df, quintile_spread, "date_key", n_boot=300, seed=1)
    assert low < point < high


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    df = frame()
    a = cluster_bootstrap(df, quintile_spread, "date_key", n_boot=200, seed=7)
    b = cluster_bootstrap(df, quintile_spread, "date_key", n_boot=200, seed=7)
    assert a == b


def test_day_clustering_widens_the_interval_versus_row_resampling() -> None:
    """Day-clustered resampling must give a wider interval than row-level resampling.

    Built on a frame where all stocks on a day share one large market shock, so rows
    within a day are strongly correlated. Treating them as independent (a row-level
    bootstrap) understates the interval and manufactures significance.
    """
    rng = np.random.default_rng(11)
    days = 40
    shocks = rng.normal(0, 0.05, days)  # the shared market move
    rows = []
    for day in range(days):
        for i in range(10):
            rows.append(
                {
                    "date_key": 20230101 + day,
                    "symbol": f"S{i:02d}",
                    "fwd_return": shocks[day] + rng.normal(0, 0.001),
                }
            )
    df = pd.DataFrame(rows)

    def mean_return(frame_: pd.DataFrame) -> float:
        return float(frame_["fwd_return"].mean())

    clustered = cluster_bootstrap(df, mean_return, "date_key", n_boot=300, seed=13)
    naive_rng = np.random.default_rng(13)
    naive = [mean_return(df.iloc[naive_rng.integers(0, len(df), len(df))]) for _ in range(300)]
    naive_width = float(np.percentile(naive, 97.5) - np.percentile(naive, 2.5))
    clustered_width = clustered[1] - clustered[0]

    assert clustered_width > naive_width, (
        "row-level resampling must not be trusted when rows share a day"
    )


def test_bootstrap_on_empty_input_is_nan_not_an_error() -> None:
    empty = pd.DataFrame({"date_key": [], "prob_up": [], "fwd_return": []})
    low, high = cluster_bootstrap(empty, quintile_spread, "date_key")
    assert math.isnan(low) and math.isnan(high)


# --------------------------------------------------------------- accuracy edge


def test_accuracy_edge_is_accuracy_minus_the_baseline_of_those_rows() -> None:
    df = frame()
    actual = df["target_up"].to_numpy().astype(int)
    predicted = (df["prob_up"].to_numpy() > 0.5).astype(int)
    expected = float((predicted == actual).mean()) - majority_baseline(actual)
    assert accuracy_edge(df) == pytest.approx(expected)


# --------------------------------------------------------------- baseline period


def test_baseline_is_computed_on_the_same_rows_as_the_accuracy() -> None:
    """R36: the edge must not compare a period's accuracy to another period's baseline.

    The 2023 validation baseline is 0.5315 but the training-period one is 0.5067.
    Comparing 0.5316 against 0.5067 invents a +2.44 point edge that does not exist.
    """
    rng = np.random.default_rng(5)
    validation = pd.DataFrame(
        {
            "target_up": np.ones(1000, dtype=int),  # 100% "up": baseline is 1.0
            "prob_up": rng.uniform(0.4, 0.6, 1000),
        }
    )
    record = classifier_metrics_row(validation, "tree", "validation", 2023)
    assert record["majority_baseline"] == pytest.approx(1.0)
    assert record["edge_over_baseline"] == pytest.approx(record["accuracy"] - 1.0)
    assert record["edge_over_baseline"] <= 0.0, (
        "with a perfect majority baseline no model can have a positive edge"
    )


def test_classifier_metrics_row_reports_the_spec_columns() -> None:
    df = frame()
    record = classifier_metrics_row(df, "rf", "validation", 2023)
    for column in (
        "model",
        "period",
        "year",
        "n",
        "accuracy",
        "majority_baseline",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tp",
        "fp",
        "fn",
        "tn",
    ):
        assert column in record, f"spec 04 section 7 requires column {column}"
    assert record["n"] == len(df)
    assert record["tp"] + record["fp"] + record["fn"] + record["tn"] == len(df)


def test_classifier_metrics_drops_rows_with_a_missing_label() -> None:
    df = frame()
    df.loc[df.index[:5], "target_up"] = np.nan
    record = classifier_metrics_row(df, "rf", "validation", 2023)
    assert record["n"] == len(df) - 5


# --------------------------------------------------------------- end to end


def test_model_diagnostics_returns_all_three_diagnostics() -> None:
    result = model_diagnostics(frame(), n_boot=200, seed=2)
    for key in (
        "roc_auc",
        "edge_over_baseline",
        "edge_ci_low",
        "edge_ci_high",
        "quintile_spread",
        "quintile_ci_low",
        "quintile_ci_high",
    ):
        assert key in result
        assert np.isfinite(result[key]), key


def test_a_degenerate_probability_column_gives_an_auc_of_about_one_half() -> None:
    """The hazard R31 catches: a near-constant model cannot rank anything."""
    df = frame()
    df["prob_up"] = 0.5015
    result = model_diagnostics(df, n_boot=100, seed=4)
    assert result["roc_auc"] == pytest.approx(0.5, abs=0.02)
