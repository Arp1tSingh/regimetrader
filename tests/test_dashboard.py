"""The dashboard (`spec 02` section 6), exercised with `streamlit.testing.v1.AppTest`.

Runs offline against the synthetic smoke warehouse. Two things are checked for every
screen: that it raises no exception, and that it does not quietly drop a baseline or a
caveat. A chart that renders but omits B1 would pass a smoke test that only looks for
exceptions, so the honesty requirements are asserted directly.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.config import load_config
from src.common.db import get_engine, read_sql
from src.dashboard import app as dashboard
from src.dashboard import data as dd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _render(config: str, screen: str) -> Any:
    """Run the app headlessly against `config`, with one screen selected."""
    from streamlit.testing.v1 import AppTest

    previous = os.environ.get("RT_DASH_CONFIG")
    os.environ["RT_DASH_CONFIG"] = config
    try:
        at = AppTest.from_file(str(REPO_ROOT / "src" / "dashboard" / "app.py"), default_timeout=180)
        at.run()
        if not at.exception:
            at.sidebar.radio[0].set_value(screen).run()
        return at
    finally:
        if previous is None:
            os.environ.pop("RT_DASH_CONFIG", None)
        else:
            os.environ["RT_DASH_CONFIG"] = previous


# --------------------------------------------------------------- screens exist


def test_the_five_spec_screens_are_present() -> None:
    """`spec 02` section 6 names exactly five."""
    assert list(dashboard.SCREENS) == [
        "Market overview",
        "OLAP explorer",
        "Today's signals",
        "Backtest results",
        "Model diagnostics",
    ]


@pytest.mark.slow
def test_every_screen_renders_on_a_completely_empty_database(tmp_path: Path) -> None:
    """T8.2 acceptance: AppTest on an empty database renders without exception.

    A real warehouse with the schema applied and zero rows, not a mocked frame.
    Each screen must fall back to `dd.empty` and say what is missing.
    """
    import yaml

    from src.common.db import init_db

    cfg = load_config(str(REPO_ROOT / "config" / "config.smoke.yaml"))
    cfg["paths"]["db"] = str(tmp_path / "empty.db")
    for key in ("results", "models", "figures"):
        cfg["paths"][key] = str(tmp_path / key)
    for key in ("results", "models"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    init_db(get_engine(cfg))
    assert read_sql(get_engine(cfg), "SELECT COUNT(*) AS n FROM fact_price")["n"].iloc[0] == 0

    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    for screen in dashboard.SCREENS:
        at = _render(str(config_file), screen)
        assert not at.exception, f"{screen} raised on an empty database: {at.exception}"
        # A message, not silence: the screen must tell the user what is missing.
        assert list(at.info) or list(at.warning) or list(at.error) or list(at.dataframe), (
            f"{screen} rendered nothing at all on an empty database"
        )


def test_every_screen_handles_empty_tables_with_a_message_not_an_exception() -> None:
    """`spec 02` section 6: a screen MUST show a message when its tables are empty."""
    from src.dashboard import data

    original = data.index_and_regimes
    try:
        data.index_and_regimes = lambda *a, **k: pd.DataFrame()
        # `empty` is the documented path; it must not raise.
        data.empty("no rows")
    finally:
        data.index_and_regimes = original


# --------------------------------------------------------------- honesty


def test_every_result_chart_plots_all_available_baselines() -> None:
    """A chart that omits a baseline would make the strategy look better than it is."""
    labels = dd.result_labels({"data": {"index_symbol": "^NSEI"}})
    frame = pd.DataFrame(
        {
            "run_id": list(dd.RESULT_RUNS),
            "date_key": [20240101 + i for i in range(len(dd.RESULT_RUNS))],
            "equity": [1.0] * len(dd.RESULT_RUNS),
            "drawdown": [0.0] * len(dd.RESULT_RUNS),
        }
    )

    class _Engine:
        pass

    # Patch the data layer so the chart functions see all six runs.
    original = dd.equity_curves
    try:
        dd.equity_curves = lambda engine, runs: frame[frame["run_id"].isin(runs)]
        figure = dashboard._equity_chart(_Engine(), list(dd.RESULT_RUNS), labels)
        plotted = {trace.name for trace in figure.data}
    finally:
        dd.equity_curves = original

    for run_id, label in labels.items():
        assert label in plotted, f"{run_id} is missing from the equity chart"


def test_the_run_labels_take_the_index_symbol_from_config() -> None:
    """A label hardcoding `^NSEI` would go stale if the configured index changed."""
    labels = dd.result_labels({"data": {"index_symbol": "^BSESN"}})
    assert labels["test_B1"] == "B1 buy and hold ^BSESN"
    assert "{" not in labels["test_B1"], "the placeholder must be fully substituted"


def test_the_baselines_are_named_in_the_chart_registry() -> None:
    """B1, B2 and B3 must all be registered, or they cannot be plotted at all."""
    for run_id in ("test_B1", "test_B2", "test_B3"):
        assert run_id in dd.RESULT_RUNS, f"{run_id} must be in the chart registry"
    assert len(dd.RESULT_RUNS) >= 4, "the strategy plus at least three baselines"


def test_the_caveats_cover_every_thing_the_report_must_state() -> None:
    """Intervals crossing zero, V1 beating V2, cost drag, the B4 caveat."""
    text = " ".join(dashboard.CAVEATS).lower()
    assert "confidence interval" in text and "zero" in text
    assert "v1" in text and "beat v2" in text
    assert "costs" in text and "0.895" in text
    assert "b4" in text and "turns" in text or "r44" in text
    assert "classifier" in text


def test_the_paper_disclaimer_is_stated() -> None:
    text = (dd.PAPER_DISCLAIMER + (dashboard.__doc__ or "")).lower()
    assert "operates" in text
    assert "does not demonstrate" in text


@pytest.mark.slow
def test_every_number_quoted_in_the_caveats_matches_summary_test() -> None:
    """The caveats are prose, so nothing stops them drifting from the stored results.

    They quote specific Sharpe and CAGR figures. If a later run changed those figures
    the screen would still assert the old ones, which is exactly the silent-drift
    failure R47 exists to prevent. Each quoted number is checked against the CSV.
    """
    summary = pd.read_csv(REPO_ROOT / "reports" / "results" / "summary_test.csv").set_index(
        "run_id"
    )
    v2 = summary.loc["backtest_main"]
    v1 = summary.loc["test_V1"]
    free = summary.loc["test_V2_nocost"]
    text = " ".join(dashboard.CAVEATS)

    # "Sharpe 0.895 against 0.485"
    assert f"{free.sharpe:.3f}" == "0.895"
    assert f"{v2.sharpe:.3f}" == "0.485"
    assert "0.895" in text and "0.485" in text

    # "[-0.456, 1.544]"
    assert f"{v2.sharpe_ci_low:.3f}" == "-0.456"
    assert f"{v2.sharpe_ci_high:.3f}" == "1.544"
    assert "-0.456" in text and "1.544" in text

    # "322 times instead of 1,004"
    assert int(v1.n_trades) == 322
    assert int(v2.n_trades) == 1004
    assert "322" in text and "1,004" in text

    # "roughly 0.41 Sharpe and 5.7 points of CAGR"
    assert round(free.sharpe - v2.sharpe, 2) == 0.41
    assert round((free.cagr - v2.cagr) * 100, 1) == 5.7
    assert "0.41" in text and "5.7" in text

    # "V1 beat V2 by 0.19 Sharpe and 4.8 points of CAGR"
    assert round(v1.sharpe - v2.sharpe, 2) == 0.19
    assert round((v1.cagr - v2.cagr) * 100, 1) == 4.8
    assert "0.19" in text and "4.8" in text


def test_the_main_disclaimer_denies_investment_advice() -> None:
    assert "not investment advice" in dd.DISCLAIMER.lower()
    assert "no real money" in dd.DISCLAIMER.lower()


# --------------------------------------------------------------- read-only


def test_the_dashboard_data_layer_writes_nothing() -> None:
    """Every SQL string in `data.py` must be a SELECT."""
    source = (REPO_ROOT / "src" / "dashboard" / "data.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(source)
    statements = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    sql = [s.value for s in statements if s.value.strip().upper().startswith(("SELECT", "WITH"))]
    for statement in sql:
        upper = statement.upper()
        for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"):
            assert not any(word in upper for word in (forbidden,)) or upper.strip().startswith(
                "SELECT"
            ), f"non-read-only statement found: {statement[:60]}"
    assert sql, "the data layer must contain the read-only queries"


def test_dashboard_modules_never_import_the_training_or_tuning_writers() -> None:
    """The dashboard must not be able to retrain, retune or rescore anything."""
    for name in ("app.py", "data.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8")
        for forbidden in ("src.models.train", "src.backtest.tune", "src.backtest.final_test"):
            assert forbidden not in source, (
                f"{name} imports {forbidden}; the dashboard must stay read-only"
            )


def test_the_dashboard_holds_no_hardcoded_dates_or_symbols() -> None:
    """Rule 6: dates, symbols and thresholds come from config, never from literals.

    A dashboard that silently hardcoded `20240101` would keep showing the old split
    after someone moved the test period in the config, and nothing would fail.
    """
    import re

    for name in ("app.py", "data.py"):
        source = (REPO_ROOT / "src" / "dashboard" / name).read_text(encoding="utf-8")

        literals = re.findall(r"\b(20[2-3]\d{5})\b", source)
        assert not literals, (
            f"{name} hardcodes the date key(s) {sorted(set(literals))}; "
            "read them from cfg['splits']"
        )

        for symbol in ("^NSEI", "^INDIAVIX"):
            assert symbol not in source, f"{name} hardcodes {symbol}; read it from cfg['data']"


# --------------------------------------------------------------- rendering


@pytest.mark.slow
def test_every_screen_renders_without_exception_on_smoke_data() -> None:
    """T8.1 acceptance: no exception on any screen, offline, on the smoke warehouse."""
    for screen in dashboard.SCREENS:
        at = _render("config/config.smoke.yaml", screen)
        assert not at.exception, f"{screen} raised on smoke data: {at.exception}"


@pytest.mark.slow
def test_every_screen_renders_without_exception_on_real_data() -> None:
    """The same five screens against the real warehouse and its report files."""
    for screen in dashboard.SCREENS:
        at = _render("config/config.yaml", screen)
        assert not at.exception, f"{screen} raised on real data: {at.exception}"


def _screen_text(at: Any) -> str:
    """All rendered text: markdown, captions, warnings and every dataframe cell.

    Run names reach the screen inside dataframes, not as prose, so a text scan that
    ignored them would pass while a baseline was missing.
    """
    parts: list[str] = []
    for element in ("markdown", "caption", "warning", "error", "info", "subheader", "title"):
        parts += [str(getattr(item, "value", "")) for item in getattr(at, element, [])]
    for frame in getattr(at, "dataframe", []) or []:
        value = getattr(frame, "value", None)
        if value is None:
            continue
        try:
            parts += [str(cell) for cell in value.astype(str).to_numpy().ravel()]
        except AttributeError:
            parts.append(str(value))
    return " ".join(parts).lower()


@pytest.mark.slow
def test_the_results_screen_shows_every_baseline_and_the_caveats() -> None:
    """The honesty requirements are visible on the screen, not only in RESULTS.md."""
    at = _render("config/config.yaml", "Backtest results")
    assert not at.exception
    text = _screen_text(at)
    labels = dd.result_labels(load_config(str(REPO_ROOT / "config" / "config.yaml")))
    for run_id, label in labels.items():
        assert label.lower() in text or run_id.lower() in text, f"{run_id} missing from the screen"
    assert "confidence interval" in text
    assert "v1" in text and "v2" in text
    assert "b4" in text
    assert "costs" in text


@pytest.mark.slow
def test_the_app_exposes_the_five_screens_in_the_sidebar() -> None:
    at = AppTest_factory()
    assert not at.exception
    radios = at.sidebar.radio
    assert radios, "the sidebar must offer the screen selector"
    options = list(radios[0].options)
    assert options == list(dashboard.SCREENS)


def AppTest_factory() -> Any:  # noqa: N802 - small helper with a capitalised name
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "src" / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    return at


def test_the_market_screen_uses_regime_bands_not_just_a_price_line() -> None:
    source = inspect.getsource(dashboard._regime_chart)
    for label in ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"):
        assert label in source, f"regime {label} must appear in the index chart"


def test_the_olap_screen_is_a_quarterly_return_heatmap_with_both_filters() -> None:
    """`02` section 6: sector-by-quarter heat map, drill-down, year and regime filters."""
    source = inspect.getsource(dashboard.screen_olap)
    assert "quarter" in source.lower(), "the heat map must be by quarter"
    assert "Regime" in source, "the regime filter is required by the spec"
    assert "Year" in source, "the year filter is required by the spec"
    assert "Drill down" in source, "the sector -> stock drill-down is required"


def test_the_heatmap_shows_returns_not_price_levels() -> None:
    """A sector with a high average close is not thereby a good sector.

    Ranking on `close_price` would make the map a price chart wearing a heat map's
    clothes, so the query must use `fact_indicator.return_1d`.
    """
    source = (REPO_ROOT / "src" / "dashboard" / "data.py").read_text(encoding="utf-8")
    assert "sector_quarter_returns" in source
    body = source[source.index("def sector_quarter_returns") :]
    body = body[: body.index("def quarter_label")]
    assert "return_1d" in body, "the heat map must be built from returns"
    assert "close_price" not in body, "the heat map must not rank on price level"


def test_no_chart_trims_the_date_range() -> None:
    """No favourable window: the equity chart must plot the whole stored series."""
    source = inspect.getsource(dashboard._equity_chart)
    for trimming in ("tail(", "head(", "[-90:", "iloc[-", "last_"):
        assert trimming not in source, (
            f"the equity chart trims its range ({trimming}); that would hide a bad period"
        )
