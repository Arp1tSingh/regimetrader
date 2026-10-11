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


def test_b4_only_touches_the_probability_column() -> None:
    """Prices, returns, regimes and execution must be identical before and after.

    The null's whole claim is that it changes the prediction-to-stock mapping and
    nothing else. If any other column moved, the null would not be a null.
    """
    df = panel()
    df["next_day_return"] = np.linspace(-0.1, 0.1, len(df))
    df["regime_key"] = 3
    shuffled = permuted_signals(df, seed=11)

    assert list(shuffled.columns) == list(df.columns)
    for column in ("date_key", "symbol", "next_day_return", "regime_key"):
        pd.testing.assert_series_equal(shuffled[column], df[column], check_names=False)
    assert not np.allclose(shuffled["prob_up"].to_numpy(), df["prob_up"].to_numpy())


def test_b4_permutation_depends_only_on_row_order_not_on_values() -> None:
    """No look-ahead: the draw must not depend on the probabilities themselves.

    If the permutation were drawn from the values, scaling a frame's probabilities
    could change which rows get swapped. Scaling by an exact factor commutes with a
    fixed index permutation, so `permuted(scaled)` must equal `scale * permuted`.
    That equality can only hold if the same indices were permuted both times.
    """
    df = panel()
    scaled = df.copy()
    scaled["prob_up"] = df["prob_up"] * 10.0

    a = permuted_signals(df, seed=5)["prob_up"].to_numpy()
    b = permuted_signals(scaled, seed=5)["prob_up"].to_numpy()

    assert np.allclose(b, a * 10.0), (
        "the same index permutation must be applied regardless of the values, "
        "so the draw cannot depend on the probabilities"
    )


def test_b4_permutation_pattern_is_independent_of_value_scale() -> None:
    """Recover the permutation explicitly: encoding row identity in the values."""
    df = panel()
    identity = df.copy()
    identity["prob_up"] = np.arange(len(df), dtype=float)

    out = permuted_signals(identity, seed=5)["prob_up"].to_numpy()
    # Every row received some other row's value, so the mapping is a real permutation.
    assert sorted(out.tolist()) == sorted(identity["prob_up"].tolist())
    # And it differs from the identity mapping.
    assert not np.array_equal(out, identity["prob_up"].to_numpy())


@pytest.mark.slow
def test_null_sharpe_centres_on_equal_weight_when_every_stock_is_eligible(
    signals_ready: tuple,
) -> None:
    """With no threshold, the null must behave like an equal-weight portfolio.

    If every stock is always eligible and slots covers the universe, V2 ranks stocks by
    a *permuted* probability — an ordering with no predictive content. The resulting
    portfolios should therefore be indistinguishable from equal weight, so the null's
    median Sharpe must sit near B2's.

    This is the null's most important sanity check: if the null distribution were
    centred somewhere else entirely, a "beats the null" result would mean nothing.

    Tolerance: **0.60 Sharpe**. Twelve simulations of a short synthetic series give a
    median with a standard error near 0.2, so 0.60 is roughly three standard errors
    while still failing loudly if the null is centred on something else entirely.
    """
    from sqlalchemy import text

    from src.backtest.baselines import permuted_signals
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import summarize
    from src.backtest.strategies import make_weights_fn as build
    from src.common.db import execute_many, read_sql

    cfg_obj, engine = signals_ready
    start, end = "2018-01-01", "2022-12-30"

    n_stocks = int(
        read_sql(
            engine,
            "SELECT COUNT(*) AS n FROM dim_stock WHERE instrument_type = 'EQUITY'",
        )["n"].iloc[0]
    )
    assert n_stocks > 1

    # entry_base 0.0: every stock always qualifies. slots covers the universe, so the
    # permuted ranking cannot drop anyone and the book is equal weight by construction.
    variant = {
        **cfg_obj,
        "strategy": {
            **cfg_obj["strategy"],
            "selected_model": "tree",
            "entry_base": 0.0,
            "slots": n_stocks,
        },
    }

    equal_weight = run_backtest(
        variant,
        build("B2", variant),
        start,
        end,
        "null_check_b2",
        engine=engine,
        model_name=None,
        persist=False,
        band=0.0,
        stop_loss_enabled=False,
    )
    b2_sharpe = summarize(equal_weight.equity, equal_weight.trades, variant)["sharpe"]

    stocks = dict(
        read_sql(
            engine, "SELECT symbol, stock_key FROM dim_stock WHERE instrument_type = 'EQUITY'"
        ).itertuples(index=False, name=None)
    )
    base = read_sql(
        engine,
        "SELECT f.date_key AS date_key, s.symbol AS symbol, f.prob_up AS prob_up "
        "FROM fact_signal f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "ORDER BY f.date_key, s.symbol",
    )

    regimes = dict(
        read_sql(engine, "SELECT date_key, regime_key FROM fact_market_regime").itertuples(
            index=False, name=None
        )
    )

    sims = 12
    sharpes = []
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT OR IGNORE INTO dim_model (model_name, version, trained_through, "
                "params_json) VALUES ('null_check', 'b4', 'diagnostic', '{}')"
            )
        )
        key = int(
            conn.execute(
                text("SELECT model_key FROM dim_model WHERE model_name = 'null_check'")
            ).fetchone()[0]
        )
    try:
        for sim in range(sims):
            shuffled = permuted_signals(base, seed=1000 + sim)
            payload = [
                {
                    "d": int(r.date_key),
                    "s": stocks[str(r.symbol)],
                    "r": int(regimes.get(int(r.date_key), 0) or 0),
                    "p": float(r.prob_up),
                }
                for r in shuffled.itertuples(index=False)
                if str(r.symbol) in stocks
            ]
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM fact_signal WHERE model_key = :k"), {"k": key})
                execute_many(
                    conn,
                    "INSERT OR REPLACE INTO fact_signal (date_key, stock_key, model_key, "
                    "regime_key, prob_up, signal, target_weight) "
                    "VALUES (:d, :s, :k, :r, :p, 0, 0.0)",
                    [{**row, "k": key} for row in payload],
                )
            outcome = run_backtest(
                variant,
                build("V2", variant),
                start,
                end,
                f"null_check_{sim}",
                engine=engine,
                model_name="null_check",
                persist=False,
            )
            sharpes.append(summarize(outcome.equity, outcome.trades, variant)["sharpe"])
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fact_signal WHERE model_key = :k"), {"k": key})
            conn.execute(text("DELETE FROM dim_model WHERE model_key = :k"), {"k": key})

    median = float(np.nanmedian(sharpes))
    assert np.isfinite(b2_sharpe), "the equal-weight baseline must produce a Sharpe"
    assert np.isfinite(median), "every null simulation must produce a finite Sharpe"
    assert abs(median - b2_sharpe) <= 0.60, (
        f"null median Sharpe {median:.4f} is not within 0.60 of the equal-weight "
        f"result {b2_sharpe:.4f}; the null is not behaving like an uninformed portfolio"
    )


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


def test_weight_exposure_distinguishes_size_from_time() -> None:
    """`exposure` counts days with any position; `weight_exposure` counts capital.

    A book fully invested half the time and flat the other half has `exposure` 0.5 and
    `weight_exposure` 0.5, but a book running at 30% of capital every day has
    `exposure` 1.0 and `weight_exposure` 0.30. B4 nulls and the real run both hold a
    position nearly every day, so only the size-weighted measure separates them.
    """
    from src.backtest.metrics import exposure, weight_exposure

    half_flat = pd.DataFrame({"equity": [100.0, 100.0], "positions_value": [100.0, 0.0]})
    assert exposure(half_flat) == pytest.approx(0.5)
    assert weight_exposure(half_flat) == pytest.approx(0.5)

    always_full = pd.DataFrame({"equity": [100.0, 100.0], "positions_value": [100.0, 100.0]})
    assert exposure(always_full) == pytest.approx(1.0)
    assert weight_exposure(always_full) == pytest.approx(1.0)

    regime_scaled = pd.DataFrame({"equity": [100.0, 100.0], "positions_value": [30.0, 30.0]})
    assert exposure(regime_scaled) == pytest.approx(1.0)
    assert weight_exposure(regime_scaled) == pytest.approx(0.30)


@pytest.mark.slow
def test_weight_exposure_of_a_real_run_is_finite(signals_ready: tuple) -> None:
    from src.backtest.engine import run_backtest
    from src.backtest.metrics import weight_exposure
    from src.backtest.strategies import make_weights_fn as build

    cfg_obj, engine = signals_ready
    cfg_obj = {**cfg_obj, "splits": {**cfg_obj["splits"], "test_start": "2030-01-01"}}
    result = run_backtest(
        cfg_obj,
        build("V2", cfg_obj),
        str(cfg_obj["splits"]["validation_start"]),
        str(cfg_obj["splits"]["validation_end"]),
        "we_probe",
        engine=engine,
        model_name="tree",
        persist=False,
    )
    assert np.isfinite(weight_exposure(result.equity))
    assert 0.0 <= weight_exposure(result.equity) <= 1.0


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
