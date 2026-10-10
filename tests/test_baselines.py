"""Baselines and the permutation null (`spec 06` sections 6 and 7).

These guard the properties that make a benchmark a benchmark: B1 must actually hold
the index, B2 must be able to hold the whole universe, B3 must follow the SMAs, and
B4 must break the link between a prediction and its own stock while leaving the
distribution of probabilities intact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.backtest.baselines import BASELINE_SETTINGS, permuted_signals
from src.backtest.metrics import block_bootstrap_sharpe
from src.backtest.strategies import make_b1, make_b2, make_b3
from tests.test_strategies import context


def cfg() -> dict[str, Any]:
    """A config carrying the keys the baselines read."""
    return {
        "data": {"index_symbol": "^NSEI"},
        "strategy": {"slots": 10, "entry_base": 0.55, "exit_prob": 0.50},
    }


# --------------------------------------------------------------- B1


def test_b1_buys_the_index_not_the_first_alphabetical_equity() -> None:
    """B1 is buy-and-hold the index. Picking the alphabetically first stock is not it."""
    provider = make_b1(cfg())
    weights = provider(context({"AAA.NS": 0.9, "ZZZ.NS": 0.8, "^NSEI": 0.5}))  # type: ignore[arg-type]
    assert weights is not None
    assert list(weights.index) == ["^NSEI"]
    assert float(weights.iloc[0]) == pytest.approx(1.0)


def test_b1_places_no_orders_after_the_first_decision() -> None:
    provider = make_b1(cfg())
    day = context({"^NSEI": 0.5})  # type: ignore[arg-type]
    assert provider(day) is not None
    assert provider(day) is None


def test_b1_refuses_to_substitute_a_stock_when_the_index_is_missing() -> None:
    """A silently substituted equity would not be the benchmark the report claims."""
    provider = make_b1(cfg())
    with pytest.raises(ValueError, match="NSEI"):
        provider(context({"AAA.NS": 0.9}))  # type: ignore[arg-type]


# --------------------------------------------------------------- B2


def test_b2_weights_one_over_n_for_every_stock_with_a_bar() -> None:
    """The baseline must be able to hold the whole universe, not `slots` names."""
    provider = make_b2(cfg())
    weights = provider(context({f"S{i:02d}": 0.5 for i in range(30)}))  # type: ignore[arg-type]
    assert weights is not None
    assert len(weights) == 30, "B2 may not truncate to the strategy's slot count"
    assert float(weights.iloc[0]) == pytest.approx(1 / 30)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_b2_rebalances_only_on_month_end() -> None:
    provider = make_b2(cfg())
    first = context({f"S{i:02d}": 0.5 for i in range(5)})  # type: ignore[arg-type]
    assert provider(first) is not None
    assert provider(first) is None, "no orders between month ends"
    month_end = context(
        {f"S{i:02d}": 0.5 for i in range(5)},
        is_month_end=True,  # type: ignore[arg-type]
    )
    assert provider(month_end) is not None


def test_b2_excludes_the_index_from_its_equal_weight() -> None:
    provider = make_b2(cfg())
    weights = provider(context({"AAA.NS": 0.5, "BBB.NS": 0.5, "^NSEI": 0.5}))  # type: ignore[arg-type]
    assert weights is not None
    assert "^NSEI" not in weights.index
    assert float(weights.sum()) == pytest.approx(1.0)


# --------------------------------------------------------------- B3


def indicators(values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """A `DayContext`-shaped indicator frame from (sma_20, sma_50) pairs."""
    frame = pd.DataFrame(
        {
            "sma_20": {s: v[0] for s, v in values.items()},
            "sma_50": {s: v[1] for s, v in values.items()},
        },
        dtype="float64",
    )
    return frame


def test_b3_holds_only_stocks_in_an_uptrend() -> None:
    provider = make_b3(cfg())
    day = context(  # type: ignore[arg-type]
        {"UP.NS": 0.5, "DOWN.NS": 0.5},
    )
    day.indicators = indicators({"UP.NS": (110.0, 100.0), "DOWN.NS": (90.0, 100.0)})
    weights = provider(day)
    assert weights is not None
    assert list(weights.index) == ["UP.NS"]
    assert float(weights.iloc[0]) == pytest.approx(0.5), "1 / N_universe, rest in cash"


def test_b3_disqualifies_a_stock_whose_sma_is_still_warming_up() -> None:
    """No signal is not a hold."""
    provider = make_b3(cfg())
    day = context({"WARM.NS": 0.5})  # type: ignore[arg-type]
    day.indicators = indicators({"WARM.NS": (float("nan"), float("nan"))})
    assert provider(day) is None


def test_b3_is_flat_cash_when_nothing_is_in_an_uptrend() -> None:
    provider = make_b3(cfg())
    day = context({"A.NS": 0.5})  # type: ignore[arg-type]
    day.indicators = indicators({"A.NS": (90.0, 100.0)})
    assert provider(day) is None


# --------------------------------------------------------------- B4


def panel() -> pd.DataFrame:
    """Three days, four stocks, known probabilities."""
    rows = []
    for day in range(3):
        for i in range(4):
            rows.append(
                {
                    "date_key": 20230101 + day,
                    "symbol": f"S{i}",
                    "prob_up": 0.5 + 0.1 * i,
                }
            )
    return pd.DataFrame(rows)


def test_b4_permutation_keeps_the_distribution_of_probabilities() -> None:
    """Exposure and turnover must stay comparable, so the values must be preserved."""
    df = panel()
    shuffled = permuted_signals(df, seed=1)
    original = np.sort(df["prob_up"].to_numpy())
    permuted = np.sort(shuffled["prob_up"].to_numpy())
    assert np.allclose(original, permuted)


def test_b4_permutation_is_within_each_day() -> None:
    """Every day keeps the same multiset of probabilities."""
    df = panel()
    shuffled = permuted_signals(df, seed=2)
    for _day, group in df.groupby("date_key"):
        day = shuffled[shuffled["date_key"] == _day]
        assert np.allclose(np.sort(group["prob_up"]), np.sort(day["prob_up"]))


def test_b4_permutation_is_seed_reproducible() -> None:
    df = panel()
    a = permuted_signals(df, seed=7)
    b = permuted_signals(df, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_b4_permutation_is_seed_sensitive() -> None:
    """Different seeds must give different assignments, or the null is not a null."""
    df = panel()
    a = permuted_signals(df, seed=1)["prob_up"].to_numpy()
    b = permuted_signals(df, seed=2)["prob_up"].to_numpy()
    assert not np.allclose(a, b)


def test_b4_actually_decouples_prediction_from_stock() -> None:
    """The permuted prob_up must not equal the stock's own for most rows."""
    df = panel()
    shuffled = permuted_signals(df, seed=3)
    same = int((shuffled["prob_up"].to_numpy() == df["prob_up"].to_numpy()).sum())
    assert same < len(df) / 2


def test_b4_does_not_mutate_the_input_frame() -> None:
    df = panel()
    before = df["prob_up"].copy()
    permuted_signals(df, seed=4)
    pd.testing.assert_series_equal(df["prob_up"], before)


# --------------------------------------------------------------- block bootstrap


def test_block_bootstrap_interval_brackets_a_constant_return_sharpe() -> None:
    returns = pd.Series(np.full(250, 0.001))
    result = block_bootstrap_sharpe(
        returns, block_days=10, resamples=200, ci_low=5, ci_high=95, seed=1
    )
    assert result["sharpe"] > 0
    assert result["ci_low"] <= result["sharpe"] <= result["ci_high"]


def test_block_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))
    a = block_bootstrap_sharpe(returns, block_days=10, resamples=200, ci_low=5, ci_high=95, seed=5)
    b = block_bootstrap_sharpe(returns, block_days=10, resamples=200, ci_low=5, ci_high=95, seed=5)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])


def test_block_bootstrap_on_a_series_shorter_than_one_block_is_nan() -> None:
    result = block_bootstrap_sharpe(
        pd.Series([0.01, -0.02, 0.005]), block_days=10, resamples=100, ci_low=5, ci_high=95
    )
    assert np.isnan(result["ci_low"]) and np.isnan(result["ci_high"])


def test_block_bootstrap_interval_widens_with_a_longer_block_on_correlated_data() -> None:
    """With real autocorrelation, a longer block cannot narrow the interval.

    Daily strategy returns are autocorrelated. Resampling single days treats them as
    independent and understates the long-run variance; resampling blocks long enough to
    span the dependence restores it, so the interval must be wider, not narrower.
    Built on AR(1) data, where the effect is present by construction.
    """
    rng = np.random.default_rng(9)
    phi, n = 0.9, 3000
    noise = rng.normal(0, 0.01, n)
    values = np.empty(n)
    values[0] = noise[0]
    for i in range(1, n):
        values[i] = phi * values[i - 1] + noise[i]
    returns = pd.Series(values)

    short = block_bootstrap_sharpe(
        returns, block_days=1, resamples=300, ci_low=5, ci_high=95, seed=2
    )
    long = block_bootstrap_sharpe(
        returns, block_days=60, resamples=300, ci_low=5, ci_high=95, seed=2
    )
    assert (long["ci_high"] - long["ci_low"]) > (short["ci_high"] - short["ci_low"])


def test_baseline_engine_settings_match_the_spec() -> None:
    """`spec 06` section 6 pins these; a change here changes the benchmark."""
    assert BASELINE_SETTINGS["B1"]["fractional_shares"] is True
    assert BASELINE_SETTINGS["B1"]["band"] == 0.0
    assert BASELINE_SETTINGS["B1"]["stop_loss_enabled"] is False
    assert BASELINE_SETTINGS["B2"]["band"] == 0.0
    assert BASELINE_SETTINGS["B2"]["stop_loss_enabled"] is False
    assert BASELINE_SETTINGS["B3"]["stop_loss_enabled"] is False


# --------------------------------------------------------------- equity index


def test_cagr_needs_a_datetime_index(signals_ready: tuple) -> None:
    """A RangeIndex silently produces NaN for every growth figure.

    `summarize` measures elapsed time from the equity index. An unindexed frame gives
    NaN CAGR for every run, which is how a whole grid of growth numbers went missing
    without anything failing.
    """
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import cagr, summarize
    from src.backtest.strategies import make_weights_fn as build

    cfg_obj, engine = signals_ready
    cfg_obj = {**cfg_obj, "splits": {**cfg_obj["splits"], "test_start": "2030-01-01"}}
    result = run_backtest(
        cfg_obj,
        build("V2", cfg_obj),
        str(cfg_obj["splits"]["validation_start"]),
        str(cfg_obj["splits"]["validation_end"]),
        "cagr_probe",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    equity = result.equity["equity"]
    assert isinstance(equity.index, pd.DatetimeIndex), (
        "the equity curve must be indexed by date for CAGR to be computable"
    )
    assert not pd.isna(cagr(equity))
    assert not pd.isna(summarize(result.equity, result.trades, cfg_obj)["cagr"])
