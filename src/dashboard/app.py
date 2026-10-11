"""RegimeTrader dashboard (`spec 02` section 6).

Five screens, all read-only, all offline: nothing here writes to the warehouse and
nothing recomputes a reported metric. The numbers come from the CSVs the authorised runs
produced, so the screen and `RESULTS.md` cannot disagree.

The dashboard is required to show the negative results honestly. Two rules are baked in
rather than left to the author of any one chart:

* every performance chart plots **all** the baselines beside the strategy, and the date
  range is the full stored period — no chart may quietly show a flattering window;
* the caveats (intervals crossing zero, V1 beating V2, the cost drag, the B4 turnover
  asymmetry) are rendered as text on the results screen, not left to `RESULTS.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard import data as dd  # noqa: E402

#: The caveats that must travel with the results screen (`DECISIONS` R35, R44, R45).
CAVEATS = [
    "**Every confidence interval crosses zero**, including the strategy's own "
    "[-0.456, 1.544]. On this evidence nothing here is distinguishable from luck.",
    "**V1 (the regime filter with the classifier removed) beat V2** by 0.19 Sharpe and "
    "4.8 points of CAGR while trading 322 times instead of 1,004. The classifier was a "
    "net negative and removing it improved the system.",
    "**Costs dominate the result**: V2 without costs scores Sharpe 0.895 against 0.485 "
    "with them. Costs remove roughly 0.41 Sharpe and 5.7 points of CAGR.",
    "**The B4 null is handicapped** (R44): it is matched to the real run on capital "
    "(weight exposure 0.7468 against the real run's 0.7487, measured on validation) but "
    "trades about 2.6x as often on the test period (2,478-2,735 round trips against "
    "1,004), so it pays more costs. 'Beat every permutation' overstates the evidence.",
    "**The classifier's edge over always predicting 'up' was +0.0001** on validation, and "
    "the per-day top-minus-bottom-quintile return intervals all contained zero. No chart "
    "here should be read as evidence that the classifier adds value.",
    "**An alternative explanation was never tested**: the strategy's low-churn holdings "
    "may capture momentum rather than any predictive skill.",
]


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any, places: int = 4) -> str:
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return "n/a"


def _date_str(key: int | None) -> str:
    if not key:
        return "n/a"
    return pd.to_datetime(str(int(key)), format="%Y%m%d").strftime("%Y-%m-%d")


def _header(cfg: dict[str, Any], engine: Any) -> None:
    """Title, disclaimer and the stale-data banner, on every screen."""
    st.title("RegimeTrader")
    st.caption(dd.DISCLAIMER)

    last = dd.latest_data_key(engine)
    today = (
        pd.Timestamp("now").year * 10000 + pd.Timestamp("now").month * 100 + pd.Timestamp("now").day
    )
    stale = dd.is_stale(cfg, last, today)
    if stale:
        st.error(
            f"**STALE DATA** — the latest market bar is {_date_str(last)}, older than "
            f"{cfg['paper']['stale_banner_days']} calendar days. "
            "Do not read anything on this screen as current."
        )
    else:
        st.info(f"Latest market data: {_date_str(last)}")


# ---------------------------------------------------------------- screens


def screen_market(cfg: dict[str, Any], engine: Any) -> None:
    """Index with regime bands, VIX, current regime."""
    st.subheader("Market overview")
    frame = dd.index_and_regimes(cfg, engine)
    if frame.empty:
        dd.empty("No index prices are loaded. Run the ETL first.")
        return

    regime = frame["regime"].dropna()
    if not regime.empty:
        counts = regime.value_counts().to_dict()
        badge = ", ".join(f"**{k}**: {v} days" for k, v in sorted(counts.items()))
        st.markdown(f"Regime distribution over the window — {badge}")

    st.plotly_chart(
        _regime_chart(frame, str(cfg["data"]["index_symbol"])),
        width="stretch",
        key="index_regime",
    )

    vix = dd.vix_series(cfg, engine)
    if not vix.empty:
        st.plotly_chart(_vix_chart(vix, str(cfg["data"]["vix_symbol"])), width="stretch", key="vix")
    else:
        dd.empty("No India VIX data is loaded.")


def _regime_chart(frame: pd.DataFrame, index_symbol: str) -> Any:
    import plotly.graph_objects as go

    x = pd.to_datetime(frame["date_key"].astype(int).astype(str), format="%Y%m%d")
    figure = go.Figure()
    colors = {
        "BULL": "#2e7d32",
        "SIDEWAYS": "#9e9d24",
        "HIGH_VOL": "#ef6c00",
        "BEAR": "#c62828",
    }
    figure.add_trace(
        go.Scatter(x=x, y=frame["close"], mode="lines", name=f"Index ({index_symbol})")
    )
    for label in ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR"):
        mask = (frame["regime"] == label).fillna(False).to_numpy()
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=x[mask],
                y=frame["close"][mask],
                mode="markers",
                marker=dict(color=colors.get(label, "#666666"), size=4),
                name=label,
            )
        )
    figure.update_layout(
        title="Index price, coloured by regime (whole stored history)",
        xaxis_title="date",
        yaxis_title="close",
        height=520,
    )
    return figure


def _vix_chart(frame: pd.DataFrame, vix_symbol: str) -> Any:
    import plotly.graph_objects as go

    x = pd.to_datetime(frame["date_key"].astype(int).astype(str), format="%Y%m%d")
    figure = go.Figure(go.Scatter(x=x, y=frame["close"], mode="lines", name=vix_symbol))
    figure.update_layout(title="India VIX", height=320)
    return figure


def screen_olap(cfg: dict[str, Any], engine: Any) -> None:
    """Sector-by-quarter heat map, drill-down to stock, year and regime filters."""
    st.subheader("OLAP explorer")

    regimes = dd.regime_labels(engine)
    left, middle, right = st.columns(3)
    with left:
        chosen_regime = st.selectbox("Regime", ["All regimes"] + regimes)
    with middle:
        year_choice = st.text_input("Year (blank for all)", value="")
    with right:
        quarter_choice = st.selectbox(
            "Quarter",
            [None, 1, 2, 3, 4],
            format_func=lambda q: "All quarters" if q is None else f"Q{q}",
        )

    year = None
    if year_choice.strip():
        try:
            year = int(year_choice.strip())
        except ValueError:
            st.error(f"`{year_choice}` is not a year.")
            return

    heat = dd.sector_quarter_returns(
        engine,
        year=year,
        quarter=quarter_choice,
        regime=None if chosen_regime == "All regimes" else chosen_regime,
    )
    if heat.empty:
        dd.empty("No indicator rows match those filters. Run the features step first.")
        return

    st.caption(
        "Mean **daily return** per sector per quarter, from `fact_indicator.return_1d`. "
        "This is a return, not a price level."
    )
    st.plotly_chart(_heat_map(heat), width="stretch", key="heat")

    sectors = sorted(heat["sector"].dropna().unique())
    chosen_sector = st.selectbox("Drill down into sector", sectors)
    quarters = sorted({(int(r.year), int(r.quarter)) for r in heat.itertuples()})
    if not quarters:
        dd.empty("That sector has no rows for the selected filters.")
        return

    pick = st.selectbox(
        "Quarter", quarters, format_func=lambda pair: dd.quarter_label(pair[0], pair[1])
    )
    rows = dd.stocks_in_sector_quarter(engine, str(chosen_sector), pick[0], pick[1])
    if rows.empty:
        dd.empty("No stock rows for that sector and quarter.")
        return
    st.dataframe(rows, width="stretch")


def _heat_map(heat: pd.DataFrame) -> Any:
    """Sectors down the side, year-quarters across the top."""
    import plotly.graph_objects as go

    heat = heat.copy()
    heat["quarter_label"] = [
        dd.quarter_label(y, q) for y, q in zip(heat["year"], heat["quarter"], strict=False)
    ]
    sectors = sorted(heat["sector"].dropna().unique())
    columns = sorted(heat["quarter_label"].dropna().unique())

    pivot = heat.pivot_table(
        index="sector", columns="quarter_label", values="avg_daily_return_pct"
    ).reindex(index=sectors, columns=columns)

    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=list(columns),
            y=list(sectors),
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar=dict(title="mean daily %"),
            hovertemplate="%{y} %{x}<br>%{z:.4f}% per day<extra></extra>",
        )
    )
    figure.update_layout(
        title="Sector x quarter: mean daily return (%)",
        xaxis_title="quarter",
        yaxis_title="sector",
        height=max(420, 26 * len(sectors)),
    )
    return figure


def screen_signals(cfg: dict[str, Any], engine: Any) -> None:
    """Today's paper signals."""
    st.subheader("Today's signals")
    st.warning(dd.PAPER_DISCLAIMER)

    date_key = dd.latest_signals_date(engine)
    if date_key is None:
        dd.empty(
            "No paper_live decision has been stored yet. The first run declined to "
            "process the lock's data_end_date, so nothing has been decided."
        )
        return

    st.write(f"Decision date: **{_date_str(date_key)}**")
    signals = dd.latest_signals(engine)
    if signals.empty:
        dd.empty("That decision stored no target weights (the strategy was flat).")
        return

    shown = signals.copy()
    shown["action"] = shown["target_weight"].map(lambda w: "BUY" if float(w) > 0 else "HOLD")
    shown["target_weight"] = shown["target_weight"].map(lambda w: _fmt_pct(w))
    shown["prob_up"] = shown["prob_up"].map(lambda p: _fmt_num(p))
    st.dataframe(shown[["symbol", "action", "target_weight", "prob_up"]], width="stretch")


def screen_results(cfg: dict[str, Any], engine: Any) -> None:
    """Strategy against every baseline, over the whole stored period."""
    st.subheader("Backtest results")
    period = st.radio("Period", ["test", "validation"], horizontal=True, index=0)

    frame = dd.summary(cfg, period)
    if frame.empty:
        dd.empty(f"No `summary_{period}.csv` yet.")
        return

    st.markdown(f"### {period.capitalize()} period — all runs")
    table = frame.copy()
    table["sharpe"] = table["sharpe"].map(lambda v: _fmt_num(v))
    if "sharpe_ci_low" in table.columns:
        table["sharpe_ci_low"] = table["sharpe_ci_low"].map(lambda v: _fmt_num(v))
        table["sharpe_ci_high"] = table["sharpe_ci_high"].map(lambda v: _fmt_num(v))
    table["cagr"] = table["cagr"].map(lambda v: _fmt_pct(v))
    table["max_drawdown"] = table["max_drawdown"].map(lambda v: _fmt_pct(v))
    keep = [
        c
        for c in (
            "run_id",
            "sharpe",
            "sharpe_ci_low",
            "sharpe_ci_high",
            "cagr",
            "max_drawdown",
            "n_trades",
        )
        if c in table.columns
    ]
    st.dataframe(table[keep], width="stretch")

    if "sharpe_ci_low" in frame.columns:
        st.caption(
            "Shaded bands on the chart below are block-bootstrap 95% intervals. "
            "Every one of them contains zero."
        )
        labels = dd.result_labels(cfg)
    st.plotly_chart(_sharpe_ci_chart(frame, labels), width="stretch", key="ci")

    runs = [r for r in dd.RESULT_RUNS if r in set(frame["run_id"])]
    if not runs:
        dd.empty("No equity curves are stored for this period.")
    else:
        st.plotly_chart(_equity_chart(engine, runs, labels), width="stretch", key="equity")
        st.plotly_chart(_drawdown_chart(engine, runs, labels), width="stretch", key="drawdown")
        st.caption(
            "Every baseline is plotted beside the strategy over the **whole** stored "
            "period. No window is trimmed to flatter the strategy."
        )

    yearly = dd.derived_yearly(cfg)
    if not yearly.empty:
        st.markdown("### Per-year return (derived from stored equity curves)")
        pivot = yearly.pivot(index="year", columns="label", values="return")
        st.dataframe(pivot.map(lambda v: _fmt_pct(v)), width="stretch")

    st.markdown("### What these numbers do and do not show")
    for note in CAVEATS:
        st.markdown(f"- {note}")


def _sharpe_ci_chart(frame: pd.DataFrame, labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    names = [labels.get(r, r) for r in frame["run_id"].tolist()]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=names,
            y=frame["sharpe"].to_numpy(),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(frame["sharpe"] - frame["sharpe_ci_low"]).to_numpy(),
                arrayminus=(frame["sharpe"] - frame["sharpe_ci_high"]).to_numpy(),
            ),
            name="Sharpe (95% block bootstrap CI)",
        )
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="red")
    figure.update_layout(
        title="Sharpe with 95% confidence intervals — every interval crosses zero",
        height=420,
        yaxis_title="Sharpe",
    )
    return figure


def _equity_chart(engine: Any, runs: list[str], labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    frame = dd.equity_curves(engine, runs)
    figure = go.Figure()
    for run_id in runs:
        part = frame[frame["run_id"] == run_id]
        x = pd.to_datetime(part["date_key"].astype(int).astype(str), format="%Y%m%d")
        figure.add_trace(
            go.Scatter(
                x=x,
                y=part["equity"],
                mode="lines",
                name=labels.get(run_id, run_id),
            )
        )
    figure.update_layout(
        title="Equity curves — strategy and every baseline, whole stored period",
        xaxis_title="date",
        yaxis_title="equity",
        height=520,
    )
    return figure


def _drawdown_chart(engine: Any, runs: list[str], labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    frame = dd.equity_curves(engine, runs)
    figure = go.Figure()
    for run_id in runs:
        part = frame[frame["run_id"] == run_id]
        x = pd.to_datetime(part["date_key"].astype(int).astype(str), format="%Y%m%d")
        figure.add_trace(
            go.Scatter(
                x=x,
                y=part["drawdown"],
                mode="lines",
                name=labels.get(run_id, run_id),
            )
        )
    figure.update_layout(
        title="Drawdown — strategy and every baseline, whole stored period",
        xaxis_title="date",
        yaxis_title="drawdown",
        height=460,
    )
    return figure


def screen_diagnostics(cfg: dict[str, Any], engine: Any) -> None:
    """Classifier diagnostics."""
    st.subheader("Model diagnostics")
    frame = dd.classifier_metrics(cfg)
    if frame.empty:
        dd.empty("No `classifier_metrics.csv` yet.")
        return

    periods = sorted(frame["period"].unique())
    period = st.selectbox("Period", periods, index=len(periods) - 1)
    window = frame[frame["period"] == period]

    st.dataframe(
        window[
            [
                "model",
                "n",
                "accuracy",
                "majority_baseline",
                "edge_over_baseline",
                "roc_auc",
            ]
        ],
        width="stretch",
    )
    st.warning(
        "`edge_over_baseline` is accuracy minus always predicting 'up'. A positive number "
        "is the only evidence the classifier beats doing nothing, and on this project it "
        "is very small. **The dashboard does not claim the classifier adds value.**"
    )

    models = sorted(window["model"].unique())
    model = st.selectbox("Model for the probability distribution", models)

    # The window comes from the configured splits, never from a literal (rule 6).
    splits = cfg["splits"]
    if period == "test":
        start = int(pd.Timestamp(splits["test_start"]).strftime("%Y%m%d"))
        end = int(pd.Timestamp("now").strftime("%Y%m%d"))
    else:
        start = int(pd.Timestamp(splits["validation_start"]).strftime("%Y%m%d"))
        end = int(pd.Timestamp(splits["validation_end"]).strftime("%Y%m%d"))

    hist = dd.prob_histogram(engine, str(model), start, end)
    if hist.empty:
        dd.empty("No stored probabilities for that model and period.")
    else:
        import plotly.graph_objects as go

        figure = go.Figure(go.Histogram(x=hist["prob_up"], nbinsx=60, name=str(model)))
        figure.update_layout(
            title=f"prob_up distribution — {model} ({period})",
            xaxis_title="prob_up",
            yaxis_title="rows",
            height=380,
        )
        st.plotly_chart(figure, width="stretch", key=f"hist_{model}_{period}")
        if model == "tree":
            st.error(
                "The decision tree has only **3 distinct probability values** with 76.4% of "
                "rows tied at one value. Any threshold at or below that mass selects most "
                "stocks and the tie is broken alphabetically, so the tree was excluded "
                "from strategy tuning (DECISIONS R31)."
            )

    importance = dd.feature_importance(
        str(model), Path(cfg["paths"]["models"]), dd.training_feature_names(cfg, engine)
    )
    if importance.empty:
        dd.empty(f"No saved feature importance for {model}.")
    else:
        st.plotly_chart(_importance_chart(importance), width="stretch", key=f"imp_{model}")


def _importance_chart(frame: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    top = frame.head(15).iloc[::-1]
    figure = go.Figure(
        go.Bar(x=top["importance"], y=top["feature"], orientation="h", name="importance")
    )
    figure.update_layout(title="Feature importance (top 15)", height=520, xaxis_title="importance")
    return figure


# ---------------------------------------------------------------- entry point

SCREENS = {
    "Market overview": screen_market,
    "OLAP explorer": screen_olap,
    "Today's signals": screen_signals,
    "Backtest results": screen_results,
    "Model diagnostics": screen_diagnostics,
}


def config_path() -> str:
    """Which config to read.

    Defaults to `config/config.yaml`, overridable with the `RT_DASH_CONFIG`
    environment variable. That is how the smoke warehouse is exercised offline without
    editing any file, and how the headless tests point at synthetic data.
    """
    import os

    return os.environ.get("RT_DASH_CONFIG") or "config/config.yaml"


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="RegimeTrader", layout="wide")
    cfg, engine = dd.cfg_and_engine(config_path())
    _header(cfg, engine)
    choice = st.sidebar.radio("Screen", list(SCREENS))
    st.sidebar.caption(dd.PAPER_DISCLAIMER)
    SCREENS[choice](cfg, engine)


if __name__ == "__main__":
    main()
