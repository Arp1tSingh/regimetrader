"""RegimeTrader dashboard (`spec 02` section 6).

Six screens (DECISIONS R55): the five read-only screens of `spec 02` §6 in that order,
plus the read-only Bot demo (replay) screen that R54 added. All are read-only and
offline: nothing here writes to the warehouse and nothing recomputes a reported metric.
The numbers come from the CSVs the authorised runs produced, so the screen and
`RESULTS.md` cannot disagree.

The dashboard is required to show the negative results honestly. Two rules are baked in
rather than left to the author of any one chart:

* every performance chart plots **all** the baselines beside the strategy, and the date
  range is the full stored period — no chart may quietly show a flattering window;
* the caveats (intervals crossing zero, V1 beating V2, the cost drag, the B4 turnover
  asymmetry) are rendered as text on the results screen, not left to `RESULTS.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard import data as dd  # noqa: E402
from src.dashboard import ui  # noqa: E402

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

#: The Bot demo banner. The P&L panel is a deliberate, recorded exception (R56): it is
#: fake money on a replay, not a performance claim. The language below is asserted by
#: `tests/test_bot_demo.py`.
BOT_BANNER = (
    "**Simulated live replay — fake money.**\n\n"
    "This is a replay of already-scored data, to demonstrate decisions and simulated "
    "P&L. It makes **no performance claim** of any kind. No real money is involved and "
    "there is no broker connection. The numbers shown are what the frozen strategy "
    "decided on those dates, not a forecast and not a result you could earn. The "
    "out-of-sample test found no demonstrated edge."
)


def _fmt_pct(value: Any, *, signed: bool = True) -> str:
    return ui.fmt_pct(value, signed=signed)


def _fmt_num(value: Any, places: int = 4) -> str:
    return ui.fmt_num(value, places=places)


def _date_str(key: int | None) -> str:
    return ui.fmt_date_iso(key)


def _header(cfg: dict[str, Any], engine: Any) -> None:
    """Title, disclaimer and the stale-data banner, on every screen."""
    ui.page_title(
        "RegimeTrader",
        "A regime-aware Nifty 50 research console: stored evidence, explicit baselines "
        "and no live-money claim.",
        kicker="Academic research console",
    )
    st.caption(dd.DISCLAIMER)

    last = dd.latest_data_key(engine)
    now = pd.Timestamp.now()
    today = int(now.strftime("%Y%m%d"))
    stale = dd.is_stale(cfg, last, today)
    if stale:
        ui.callout(
            f"<strong>STALE DATA</strong> — the latest market bar is {_date_str(last)}, "
            f"older than {cfg['paper']['stale_banner_days']} calendar days. "
            "Do not read anything on this screen as current.",
            "warn",
        )
    else:
        st.caption(f"Latest market data: {_date_str(last)} · stored, not a live quote.")


def _sidebar() -> str:
    """The first sidebar radio is the screen selector, as the tests expect."""
    st.sidebar.markdown(
        '<div class="rt-eyebrow">RegimeTrader</div>'
        '<div style="font-size:1.15rem;font-weight:720;letter-spacing:-0.02em;'
        'margin-bottom:0.8rem;">Research console</div>',
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio("Screen", list(SCREENS))
    st.sidebar.divider()
    st.sidebar.caption("Read-only · offline · no broker connection")
    st.sidebar.caption(dd.PAPER_DISCLAIMER)
    return str(choice)


# ---------------------------------------------------------------- screens


def screen_market(cfg: dict[str, Any], engine: Any) -> None:
    """Index with regime bands, VIX, current regime."""
    frame = dd.index_and_regimes(cfg, engine)
    if frame.empty:
        dd.empty("No index prices are loaded. Run the ETL first.")
        return

    index_label = str(cfg["data"].get("index_name") or cfg["data"]["index_symbol"])
    latest = frame.iloc[-1]
    previous = frame.iloc[-2] if len(frame) > 1 else None
    change = None
    if previous is not None and float(previous["close"]) > 0:
        change = float(latest["close"]) / float(previous["close"]) - 1.0

    regime_series = frame["regime"].dropna()
    current_regime = str(regime_series.iloc[-1]) if not regime_series.empty else "n/a"
    duration = 0
    if not regime_series.empty:
        current = regime_series.iloc[-1]
        for value in reversed(regime_series.tolist()):
            if value != current:
                break
            duration += 1

    vix = dd.vix_series(cfg, engine)
    latest_vix = None if vix.empty else float(vix.iloc[-1]["close"])

    ui.section("Market overview", f"{len(frame):,} stored sessions")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            f"{index_label} close",
            ui.fmt_num(latest["close"], 2),
            None if change is None else _fmt_pct(change),
            help="Latest stored index close; the change is versus the previous stored session.",
        )
    with k2:
        st.metric("Current regime", current_regime, f"{duration} sessions")
    with k3:
        st.metric("India VIX", "n/a" if latest_vix is None else ui.fmt_num(latest_vix, 2))
    with k4:
        changes = (
            int((regime_series != regime_series.shift()).sum() - 1) if len(regime_series) else 0
        )
        st.metric("Regime changes", f"{max(changes, 0):,}")

    if not regime_series.empty:
        counts = regime_series.value_counts().to_dict()
        chips = "".join(
            f'<span class="rt-chip">{ui.regime_badge(label)} '
            f"<strong>{int(counts.get(label, 0)):,}</strong></span>"
            for label in ui.REGIME_ORDER
        )
        st.markdown(f'<div class="rt-chip-row">{chips}</div>', unsafe_allow_html=True)

    st.plotly_chart(
        _regime_chart(frame, str(cfg["data"]["index_symbol"])),
        width="stretch",
        key="index_regime",
    )

    left, right = st.columns([2.15, 1])
    with left:
        st.markdown("### Volatility")
        if vix.empty:
            dd.empty("No India VIX data is loaded.")
        else:
            st.plotly_chart(
                _vix_chart(vix, str(cfg["data"]["vix_symbol"])),
                width="stretch",
                key="vix",
            )
    with right:
        st.markdown("### Regime mix")
        if regime_series.empty:
            dd.empty("No regime labels are stored.")
        else:
            st.plotly_chart(_regime_mix_chart(regime_series), width="stretch", key="regime_mix")
            st.caption(
                "Labels are the stored regime model output. The chart shows the whole "
                "stored history, not a selected window."
            )


def _regime_chart(frame: pd.DataFrame, index_symbol: str) -> Any:
    import plotly.graph_objects as go

    # Explicit four-regime vocabulary: a test reads this function to prove that no
    # regime is silently dropped from the market chart.
    regime_order = ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR")
    x = pd.to_datetime(frame["date_key"].astype(int).astype(str), format="%Y%m%d")
    figure = go.Figure()

    # Light regime bands behind the price line: the colour is a property of the day,
    # not a second price series.
    labels = frame["regime"].fillna("").astype(str).tolist()
    start = 0
    for position in range(1, len(labels) + 1):
        if position == len(labels) or labels[position] != labels[start]:
            label = labels[start]
            if label in ui.REGIME_COLORS:
                figure.add_vrect(
                    x0=x.iloc[start],
                    x1=x.iloc[position - 1],
                    fillcolor=ui.REGIME_COLORS[label],
                    opacity=0.055,
                    line_width=0,
                    layer="below",
                )
            start = position

    figure.add_trace(
        go.Scatter(
            x=x,
            y=frame["close"],
            mode="lines",
            name=f"Index ({index_symbol})",
            line=dict(color=ui.INK, width=1.7),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>",
        )
    )
    for label in regime_order:
        mask = (frame["regime"] == label).fillna(False).to_numpy()
        if not mask.any():
            continue
        figure.add_trace(
            go.Scatter(
                x=x[mask],
                y=frame["close"][mask],
                mode="markers",
                marker=dict(color=ui.REGIME_COLORS.get(label, ui.MUTED), size=3.5),
                name=label,
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Index price with stored regime labels — whole stored history",
        xaxis_title="date",
        yaxis_title="index close",
        height=520,
    )
    return ui.apply_figure(figure)


def _vix_chart(frame: pd.DataFrame, vix_symbol: str) -> Any:
    import plotly.graph_objects as go

    x = pd.to_datetime(frame["date_key"].astype(int).astype(str), format="%Y%m%d")
    figure = go.Figure(
        go.Scatter(
            x=x,
            y=frame["close"],
            mode="lines",
            name=vix_symbol,
            line=dict(color=ui.WARNING, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(180, 83, 9, 0.08)",
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="India VIX — whole stored history",
        xaxis_title="date",
        yaxis_title="VIX",
        height=330,
    )
    return ui.apply_figure(figure)


def _regime_mix_chart(regime_series: pd.Series) -> Any:
    import plotly.graph_objects as go

    counts = regime_series.value_counts().reindex(list(ui.REGIME_ORDER), fill_value=0)
    figure = go.Figure(
        go.Bar(
            x=counts.values,
            y=list(counts.index),
            orientation="h",
            marker_color=[ui.REGIME_COLORS.get(label, ui.MUTED) for label in counts.index],
            text=[f"{int(value):,}" for value in counts.values],
            textposition="outside",
            hovertemplate="%{y}: %{x:,} sessions<extra></extra>",
        )
    )
    figure.update_layout(
        title="Sessions by regime",
        xaxis_title="sessions",
        yaxis_title="",
        height=330,
        showlegend=False,
    )
    return ui.apply_figure(figure)


def screen_olap(cfg: dict[str, Any], engine: Any) -> None:
    """Sector-by-quarter heat map, drill-down to stock, year and regime filters."""
    ui.section("OLAP explorer", "Sector × quarter mean daily return")

    regimes = dd.regime_labels(engine)
    years = dd.available_years(engine)
    with st.container(border=True):
        left, middle, right = st.columns(3)
        with left:
            chosen_regime = st.selectbox("Regime", ["All regimes"] + regimes)
        with middle:
            year_choice = st.selectbox("Year", ["All years"] + [str(year) for year in years])
        with right:
            quarter_choice = st.selectbox(
                "Quarter",
                [None, 1, 2, 3, 4],
                format_func=lambda q: "All quarters" if q is None else f"Q{q}",
            )

    year = None if year_choice == "All years" else int(year_choice)
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
        "This is a return, not a price level. Hover any cell for the exact value; "
        "in-cell labels are shown only when the selected window has 16 or fewer quarters."
    )
    st.plotly_chart(_heat_map(heat), width="stretch", key="heat")

    sectors = sorted(heat["sector"].dropna().unique())
    if not sectors:
        dd.empty("No sectors match those filters.")
        return
    with st.container(border=True):
        st.markdown("### Drill down")
        chosen_sector = st.selectbox("Drill down into sector", sectors)
        quarters = sorted({(int(r.year), int(r.quarter)) for r in heat.itertuples()})
        if not quarters:
            dd.empty("That sector has no rows for the selected filters.")
            return
        pick = st.selectbox(
            "Quarter",
            quarters,
            format_func=lambda pair: dd.quarter_label(pair[0], pair[1]),
            key="drill_quarter",
        )
        rows = dd.stocks_in_sector_quarter(engine, str(chosen_sector), pick[0], pick[1])
        if rows.empty:
            dd.empty("No stock rows for that sector and quarter.")
            return
        shown = rows.copy()
        shown["quarter_return_pct"] = shown["quarter_return_pct"].map(
            lambda value: f"{float(value):+.2f}%"
        )
        shown = shown.rename(
            columns={
                "symbol": "symbol",
                "company": "company",
                "quarter_return_pct": "quarter return",
                "days": "days",
            }
        )
        st.dataframe(shown, width="stretch", hide_index=True)


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
    values = pivot.to_numpy(dtype=float)
    # Forty-eight quarter columns make in-cell labels unreadable: the strings overlap
    # and hide both the colour and the number. Keep hover values always, but print
    # numbers only for a filtered view narrow enough to read.
    show_text = len(columns) <= 16
    text = (
        [["" if pd.isna(value) else f"{value:+.2f}%" for value in row] for row in values]
        if show_text
        else None
    )

    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=list(columns),
            y=list(sectors),
            colorscale=[
                [0.0, ui.NEGATIVE],
                [0.5, "#F8FAFC"],
                [1.0, ui.POSITIVE],
            ],
            zmid=0.0,
            text=text,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(size=10, color=ui.INK),
            colorbar=dict(title="mean daily %", thickness=12),
            hovertemplate="%{y} %{x}<br>%{z:+.2f}% per day<extra></extra>",
        )
    )
    figure.update_layout(
        title="Sector × quarter: mean daily return (%)",
        xaxis_title="quarter",
        yaxis_title="sector",
        height=max(420, 26 * len(sectors)),
    )
    figure.update_xaxes(tickangle=-90, tickfont=dict(size=9))
    figure.update_yaxes(tickfont=dict(size=10))
    return ui.apply_figure(figure)


def screen_signals(cfg: dict[str, Any], engine: Any) -> None:
    """Latest stored paper decision, including an honest flat state."""
    ui.section("Today's signals", "Latest stored paper_live decision")
    st.warning(dd.PAPER_DISCLAIMER)

    date_key = dd.latest_signals_date(engine)
    latest_market = dd.latest_data_key(engine)
    if date_key is None:
        ui.callout(
            "No `paper_live` decision has been stored yet. The daily paper job has not "
            "committed a decision. This screen shows stored decisions, never a live feed.",
            "info",
        )
        return

    model = str(cfg["strategy"].get("selected_model") or cfg["tuning"]["models"][0])
    signals = dd.latest_signals(engine, date_key=date_key, model=model)
    regime = dd.regime_for_date(engine, date_key)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Decision date", _date_str(date_key))
    with k2:
        st.metric("Latest market bar", _date_str(latest_market))
    with k3:
        st.metric("Regime at decision", regime or "n/a")

    if latest_market is not None and int(date_key) < int(latest_market):
        ui.callout(
            f"<strong>NOT TODAY'S DECISION</strong> - the latest stored decision is "
            f"{_date_str(date_key)}, but the latest market bar is {_date_str(latest_market)}. "
            "Run `make paper` to store a decision for the newer session.",
            "warn",
        )
    else:
        st.caption(
            f"Decision stored at the {_date_str(date_key)} close; orders would fill at "
            "the next open. Stored paper decision, not a live quote."
        )
    st.caption(f"Model: `{model}` · source: `{dd.PAPER_RUN_ID}`")

    if signals.empty:
        dd.empty("That decision stored no probabilities or target weights.")
        return

    positive = signals[signals["target_weight"] > 0].copy()
    gross = float(positive["target_weight"].sum()) if not positive.empty else 0.0

    if positive.empty:
        ui.callout(
            "<strong>FLAT DECISION</strong> - the strategy selected no positive target "
            "weights. The target book is cash; this is a real decision, not missing data. "
            "No SELL instruction is shown.",
            "info",
        )
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("BUY targets", "0")
        with k2:
            st.metric("Gross target exposure", _fmt_pct(0.0, signed=False))
        with k3:
            st.metric("Largest target", "n/a")
    else:
        largest = positive.sort_values("target_weight", ascending=False).iloc[0]
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("BUY targets", f"{len(positive):,}")
        with k2:
            st.metric("Gross target exposure", _fmt_pct(gross, signed=False))
        with k3:
            st.metric(
                "Largest target",
                str(largest["symbol"]),
                _fmt_pct(float(largest["target_weight"]), signed=False),
            )

    shown = signals.copy()
    shown["action"] = shown["target_weight"].map(lambda w: "BUY" if float(w) > 0 else "HOLD")
    shown["target_weight"] = shown["target_weight"].map(lambda w: _fmt_pct(w, signed=False))
    shown["prob_up"] = shown["prob_up"].map(lambda p: _fmt_num(p))
    shown = shown.rename(
        columns={
            "symbol": "symbol",
            "action": "action",
            "target_weight": "target weight",
            "prob_up": "prob_up",
        }
    )
    st.dataframe(
        shown[["symbol", "action", "target weight", "prob_up"]],
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "All names scored for this decision. The stored `pending_target` table has no "
        "action field. **BUY** means a positive target weight; **HOLD** includes zero and "
        "is not a sell instruction. No live market quote is involved."
    )
    if not positive.empty:
        st.plotly_chart(_signal_weights_chart(positive), width="stretch", key="signal_weights")
    else:
        st.caption("No positive target weights to chart; the target book is flat.")


def _signal_weights_chart(signals: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    ordered = signals.sort_values("target_weight", ascending=True)
    figure = go.Figure(
        go.Bar(
            x=ordered["target_weight"],
            y=ordered["symbol"],
            orientation="h",
            marker_color=ui.ACCENT,
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Target weights at the latest decision",
        xaxis_title="target weight",
        yaxis_title="",
        height=max(300, 22 * len(ordered)),
        xaxis_tickformat=".0%",
    )
    return ui.apply_figure(figure)


def screen_results(cfg: dict[str, Any], engine: Any) -> None:
    """Strategy against every baseline, over the whole stored period."""
    ui.section("Backtest results", "Stored runs, all baselines, whole stored period")
    period = st.radio("Period", ["test", "validation"], horizontal=True, index=0)

    frame = dd.summary(cfg, period)
    if frame.empty:
        dd.empty(f"No `summary_{period}.csv` yet.")
        return

    labels = dd.result_labels(cfg)
    strategy = frame[frame["run_id"] == "backtest_main"]
    if not strategy.empty:
        row = strategy.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("V2 Sharpe", _fmt_num(row.get("sharpe"), 3))
        with k2:
            st.metric("V2 CAGR", _fmt_pct(row.get("cagr")))
        with k3:
            st.metric("V2 max drawdown", _fmt_pct(row.get("max_drawdown"), signed=False))
        with k4:
            st.metric("V2 final equity", ui.fmt_inr(row.get("final_equity")))
        st.caption(
            "These are the stored test-run numbers, shown with every baseline below. "
            "They are not a claim that the strategy has an edge."
        )

    st.markdown(f"### {period.capitalize()} period — all runs")
    table = frame.copy()
    table["run_id"] = table["run_id"].map(lambda run: labels.get(str(run), str(run)))
    table["sharpe"] = table["sharpe"].map(lambda v: _fmt_num(v))
    if "sharpe_ci_low" in table.columns:
        table["sharpe_ci_low"] = table["sharpe_ci_low"].map(lambda v: _fmt_num(v))
        table["sharpe_ci_high"] = table["sharpe_ci_high"].map(lambda v: _fmt_num(v))
    table["cagr"] = table["cagr"].map(lambda v: _fmt_pct(v))
    table["max_drawdown"] = table["max_drawdown"].map(lambda v: _fmt_pct(v, signed=False))
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
    st.dataframe(table[keep], width="stretch", hide_index=True)

    if "sharpe_ci_low" in frame.columns:
        st.caption(
            "Shaded bands on the chart below are block-bootstrap 95% intervals. "
            "Every one of them contains zero."
        )
        st.plotly_chart(_sharpe_ci_chart(frame, labels), width="stretch", key="ci")

    runs = [r for r in dd.RESULT_RUNS if r in set(frame["run_id"])]
    if not runs:
        dd.empty("No equity curves are stored for this period.")
    else:
        st.markdown("### Equity and drawdown")
        equity_tab, drawdown_tab = st.tabs(["Equity curves", "Drawdown"])
        with equity_tab:
            st.plotly_chart(_equity_chart(engine, runs, labels), width="stretch", key="equity")
        with drawdown_tab:
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

    with st.container(border=True):
        st.markdown("### What these numbers do and do not show")
        for note in CAVEATS:
            st.markdown(f"- {note}")


def _sharpe_ci_chart(frame: pd.DataFrame, labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    names = [labels.get(r, r) for r in frame["run_id"].tolist()]
    colors = [
        ui.ACCENT if run == "backtest_main" else "#94A3B8" for run in frame["run_id"].tolist()
    ]
    figure = go.Figure(
        go.Bar(
            x=names,
            y=frame["sharpe"].to_numpy(),
            marker_color=colors,
            error_y=dict(
                type="data",
                symmetric=False,
                array=(frame["sharpe"] - frame["sharpe_ci_low"]).to_numpy(),
                arrayminus=(frame["sharpe"] - frame["sharpe_ci_high"]).to_numpy(),
                color=ui.MUTED,
            ),
            name="Sharpe (95% block bootstrap CI)",
            hovertemplate="%{x}<br>Sharpe %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color=ui.NEGATIVE, line_width=1)
    figure.update_layout(
        title="Sharpe with 95% confidence intervals — every interval crosses zero",
        height=420,
        yaxis_title="Sharpe",
    )
    return ui.apply_figure(figure)


def _equity_chart(engine: Any, runs: list[str], labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    frame = dd.equity_curves(engine, runs)
    figure = go.Figure()
    for run_id in runs:
        part = frame[frame["run_id"] == run_id]
        x = pd.to_datetime(part["date_key"].astype(int).astype(str), format="%Y%m%d")
        is_strategy = run_id == "backtest_main"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=part["equity"],
                mode="lines",
                name=labels.get(run_id, run_id),
                line=dict(
                    color=ui.ACCENT if is_strategy else None,
                    width=2.6 if is_strategy else 1.3,
                ),
                opacity=1.0 if is_strategy else 0.82,
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Equity curves — strategy and every baseline, whole stored period",
        xaxis_title="date",
        yaxis_title="equity",
        height=520,
        yaxis_tickprefix="₹",
        yaxis_tickformat=",.0f",
    )
    return ui.apply_figure(figure)


def _drawdown_chart(engine: Any, runs: list[str], labels: dict[str, str]) -> Any:
    import plotly.graph_objects as go

    frame = dd.equity_curves(engine, runs)
    figure = go.Figure()
    for run_id in runs:
        part = frame[frame["run_id"] == run_id]
        x = pd.to_datetime(part["date_key"].astype(int).astype(str), format="%Y%m%d")
        is_strategy = run_id == "backtest_main"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=part["drawdown"],
                mode="lines",
                name=labels.get(run_id, run_id),
                line=dict(
                    color=ui.ACCENT if is_strategy else None,
                    width=2.4 if is_strategy else 1.2,
                ),
                opacity=1.0 if is_strategy else 0.78,
                hovertemplate="%{x|%d %b %Y}<br>%{y:.2%}<extra></extra>",
            )
        )
    figure.update_layout(
        title="Drawdown — strategy and every baseline, whole stored period",
        xaxis_title="date",
        yaxis_title="drawdown",
        height=460,
        yaxis_tickformat=".0%",
    )
    return ui.apply_figure(figure)


def screen_diagnostics(cfg: dict[str, Any], engine: Any) -> None:
    """Classifier diagnostics, including the stored confusion matrix."""
    ui.section("Model diagnostics", "Stored validation and test metrics")
    frame = dd.classifier_metrics(cfg)
    if frame.empty:
        dd.empty("No `classifier_metrics.csv` yet.")
        return

    periods = sorted(frame["period"].unique())
    period = st.selectbox("Period", periods, index=len(periods) - 1)
    window = frame[frame["period"] == period]
    models = sorted(window["model"].unique())
    model = st.selectbox("Model", models)
    selected = window[window["model"] == model].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Accuracy", _fmt_pct(selected.get("accuracy"), signed=False))
    with k2:
        st.metric("Edge over majority", f"{float(selected.get('edge_over_baseline')):+.4f}")
    with k3:
        st.metric("ROC AUC", _fmt_num(selected.get("roc_auc"), 3))
    with k4:
        st.metric("Rows", f"{int(selected.get('n')):,}")

    st.markdown("### All models in this period")
    metrics = window.copy()
    for column in ("accuracy", "majority_baseline"):
        if column in metrics:
            metrics[column] = metrics[column].map(lambda value: _fmt_pct(value, signed=False))
    if "edge_over_baseline" in metrics:
        metrics["edge_over_baseline"] = metrics["edge_over_baseline"].map(
            lambda value: f"{float(value):+.4f}"
        )
    for column in ("precision", "recall", "f1"):
        if column in metrics:
            metrics[column] = metrics[column].map(lambda value: _fmt_num(value, 3))
    if "roc_auc" in metrics:
        metrics["roc_auc"] = metrics["roc_auc"].map(lambda value: _fmt_num(value, 3))
    keep = [
        column
        for column in (
            "model",
            "n",
            "accuracy",
            "majority_baseline",
            "edge_over_baseline",
            "roc_auc",
        )
        if column in metrics.columns
    ]
    st.dataframe(metrics[keep], width="stretch", hide_index=True)

    st.warning(
        "`edge_over_baseline` is accuracy minus always predicting 'up'. A positive number "
        "is the only evidence the classifier beats doing nothing, and on this project it "
        "is very small. **The dashboard does not claim the classifier adds value.**"
    )

    st.markdown("### Confusion matrix")
    st.plotly_chart(_confusion_chart(selected), width="stretch", key=f"confusion_{model}_{period}")
    st.caption(
        "Built directly from the stored `tp`, `fp`, `fn` and `tn` counts. Rows are the "
        "actual direction, columns are the predicted direction. This is a diagnostic of "
        "the classifier, not a trading result."
    )

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

        figure = go.Figure(
            go.Histogram(
                x=hist["prob_up"],
                nbinsx=60,
                name=str(model),
                marker_color=ui.ACCENT,
                opacity=0.85,
            )
        )
        figure.update_layout(
            title=f"prob_up distribution — {model} ({period})",
            xaxis_title="prob_up",
            yaxis_title="rows",
            height=380,
        )
        st.plotly_chart(ui.apply_figure(figure), width="stretch", key=f"hist_{model}_{period}")
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


def _confusion_chart(row: Any) -> Any:
    """2×2 confusion matrix from the stored tp/fp/fn/tn counts."""
    import plotly.graph_objects as go

    tp = int(row.get("tp", 0) or 0)
    fp = int(row.get("fp", 0) or 0)
    fn = int(row.get("fn", 0) or 0)
    tn = int(row.get("tn", 0) or 0)
    values = [[tn, fp], [fn, tp]]
    text = [[f"{tn:,}", f"{fp:,}"], [f"{fn:,}", f"{tp:,}"]]
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=["Predicted down (0)", "Predicted up (1)"],
            y=["Actual down (0)", "Actual up (1)"],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=20, color=ui.INK),
            colorscale=[[0.0, "#F1F5F9"], [1.0, ui.ACCENT]],
            showscale=False,
            hovertemplate="%{y} / %{x}: %{z:,}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Confusion matrix",
        height=330,
        xaxis_title="predicted",
        yaxis_title="actual",
    )
    return ui.apply_figure(figure)


def _importance_chart(frame: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    top = frame.head(15).iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=top["importance"],
            y=top["feature"],
            orientation="h",
            marker_color=ui.ACCENT,
            name="importance",
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(title="Feature importance (top 15)", height=520, xaxis_title="importance")
    return ui.apply_figure(figure)


# ---------------------------------------------------------------- bot demo


def screen_bot(cfg: dict[str, Any], engine: Any) -> None:
    """Replay the frozen bot on a throwaway copy. Never touches the real warehouse."""
    ui.section("Bot demo", "Simulated live replay of stored test days")
    st.warning(BOT_BANNER)

    from src.dashboard import bot

    read_engine = dd.readonly_engine(cfg)
    try:
        allowed = bot.available_start_dates(cfg, read_engine)
    finally:
        read_engine.dispose()
    if not allowed:
        dd.empty("This warehouse has no trading days in its configured test period.")
        return

    st.caption(
        "Universe note: the 50 names are today's Nifty 50 constituents held fixed "
        "across all of history, so what the bot could own in 2024 reflects this list, "
        "not the index as it was then (survivorship bias; no point-in-time membership "
        "is stored)."
    )
    st.caption(
        "Price note: prices are split- and dividend-adjusted, so share counts on this "
        "screen are illustrative rather than real tradable quantities."
    )
    st.caption(
        f"Playback window: {bot.WINDOW_DAYS} stored trading days from one of "
        f"{len(allowed)} fixed start dates. The warehouse is opened read-only, every run "
        f"works on a temporary copy that is deleted afterwards, only one run happens at a "
        f"time, and no free text reaches a query."
    )

    labels = {f"Start {index + 1} · {ui.fmt_date(day)}": day for index, day in enumerate(allowed)}
    choice = st.selectbox("Start day", list(labels))
    start_date = labels[choice]

    animate = st.toggle("Animate the replay", value=True)
    speed = st.select_slider(
        "Playback speed",
        options=["Fast", "Normal", "Slow"],
        value="Normal",
        help="Normal is roughly 0.12 seconds per trading day.",
    )

    autoplay = bool(st.session_state.pop("bot_autoplay", False))
    autoplay_speed = str(st.session_state.pop("bot_speed", speed))
    run = bot.cached(start_date)
    if run is not None:
        st.success(
            f"Showing a cached run for {ui.fmt_date(run.start_date)} "
            f"(computed in {run.elapsed_seconds:.1f}s). Press Run to recompute."
        )
        if autoplay:
            _render_bot_live(run, speed=autoplay_speed)
        else:
            _render_bot(run)
            if st.button("Play the replay", type="secondary"):
                # Do not render the full result and start the animation in the same
                # script run: both would create the chart and day-inspector widgets.
                st.session_state["bot_autoplay"] = True
                st.session_state["bot_speed"] = str(speed)
                st.rerun()
        st.divider()

    if st.button("Run the bot", type="primary"):
        _run_bot(cfg, start_date, allowed, animate=animate, speed=speed)


def _run_bot(
    cfg: dict[str, Any],
    start_date: int,
    allowed: tuple[int, ...],
    *,
    animate: bool = True,
    speed: str = "Normal",
) -> None:
    """Run a replay, with every failure mode reduced to a safe, generic message."""
    from src.dashboard import bot

    status = st.empty()
    progress = st.empty()
    try:
        with bot.single_run():
            status.info("Preparing the replay on a temporary copy...")
            progress = st.progress(0.0, text="starting")

            def on_phase(label: str) -> None:
                status.info(label)
                progress.progress(0.0, text=label)

            def on_day(index: int) -> None:
                progress.progress(
                    min(1.0, (index + 1) / bot.WINDOW_DAYS),
                    text=f"day {index + 1} of {bot.WINDOW_DAYS}",
                )

            deadline = bot.timeout_guard(bot.RUN_TIMEOUT_SECONDS)

            def deadline_check() -> None:
                deadline()

            run = bot.run_replay(
                cfg,
                Path(cfg["paths"]["db"]),
                int(start_date),
                allowed=allowed,
                deadline_check=deadline_check,
                per_day_check=on_day,
                phase_check=on_phase,
            )
            progress.empty()
            status.empty()

            run.first_day_check = bot.first_day_check_against_backtest(cfg, run)
            bot.store(run)
            st.session_state["bot_autoplay"] = bool(animate)
            st.session_state["bot_speed"] = str(speed)
            st.rerun()
    except bot.ReplayBusy:
        status.empty()
        progress.empty()
        st.warning("A replay is already running for someone else. Try again in a moment.")
    except TimeoutError:
        status.empty()
        progress.empty()
        st.error("That took too long and was stopped. Try again.")
    except Exception:
        # Nothing internal is shown to an anonymous visitor: no paths, no SQL, no
        # traceback. The detail goes to the server log.
        status.empty()
        progress.empty()
        _log_bot_failure(start_date)
        st.error("The replay could not be completed. Please try again.")


def _log_bot_failure(start_date: int) -> None:
    """Record the real reason server-side, where an anonymous visitor cannot see it."""
    import traceback

    from src.common.log import get_logger

    log = get_logger("src.dashboard.bot", {"paths": {"logs": "logs"}})
    log.error("bot demo failed for start_date=%s\n%s", start_date, traceback.format_exc())


def _render_bot(
    run: Any, *, chart_key: str = "bot_pnl_full", day_key: str = "bot_day_inspector"
) -> None:
    from src.dashboard import bot

    pnl = bot.portfolio_pnl(run)
    st.markdown(
        f"**{bot.WINDOW_DAYS} trading days from {ui.fmt_date(run.start_date)}** — "
        f"completed in {run.elapsed_seconds:.1f}s using {run.lookback} days of lookback "
        f"history."
    )

    if not pnl.empty:
        _live_cards(pnl.iloc[-1], run)
        st.plotly_chart(_pnl_chart(pnl, run), width="stretch", key=chart_key)
    else:
        st.info("The replay returned no daily rows.")

    summary = _bot_summary(run, pnl)
    st.markdown("### Day-by-day summary")
    st.dataframe(summary, width="stretch", hide_index=True)

    if run.days:
        options = list(range(len(run.days)))
        selected = st.selectbox(
            "Inspect a day",
            options,
            format_func=lambda index: f"Day {index + 1} · {ui.fmt_date(run.days[index].date_key)}",
            key=day_key,
        )
        _render_day_detail(run.days[selected])

    check = run.first_day_check
    st.markdown("### No-hindsight check (day 1)")
    if not check:
        st.write("Not run for this result.")
    elif check.get("match"):
        st.success(
            f"The replay's first-day orders **match** the stored `backtest_main` records: "
            f"{check['n_replay']} orders on both sides, same symbols, quantities and "
            f"fill prices. Both filled at the next day's open."
        )
    else:
        st.error(
            "The replay's first-day orders **do not match** the stored "
            "`backtest_main` records. Shown in full rather than hidden."
        )
        st.json(check)

    proof = _read_proof()
    if proof is None:
        st.info("No offline equivalence proof has been recorded yet.")
    else:
        agree = proof.get("all_match")
        safe = proof.get("all_lookahead_safe")
        st.markdown("### Verified offline")
        st.caption(
            f"A full literal recompute was run on {proof.get('generated_at', 'an earlier date')} "
            f"and compared with the fast path this page uses."
        )
        if agree and safe:
            st.success(
                f"Days tested: {proof.get('days_tested')}. Every regime label and target "
                f"weight matched exactly, and every `prob_up` matched within "
                f"{proof.get('float_tolerance')}."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "day": r["day"],
                            "regime": r["regime"],
                            "values compared": r["equivalence"]["prob_values_compared"],
                            "max delta": r["equivalence"]["max_prob_delta"],
                            "literal (s)": r["literal_seconds"],
                            "fast (s)": r["fast_seconds"],
                            "speed-up": f"{r['speedup']}x",
                            "look-ahead safe": r["lookahead_mutation"].get("passed"),
                        }
                        for r in proof.get("results", [])
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.error(
                "The offline proof did **not** pass. The details are below and are not "
                "being hidden."
            )
            st.json(proof)


def _bot_summary(run: Any, pnl: pd.DataFrame) -> pd.DataFrame:
    """Compact day table; the live panel already shows the detailed day cards."""
    rows: list[dict[str, Any]] = []
    for index, day in enumerate(run.days):
        pnl_row = pnl.iloc[index] if index < len(pnl) else {}
        rows.append(
            {
                "day": ui.fmt_date_iso(day.date_key),
                "regime": day.regime or "n/a",
                "entry threshold": (
                    "n/a" if day.entry_threshold is None else f"{day.entry_threshold:.6f}"
                ),
                "candidates": day.candidates,
                "above threshold": day.passed_threshold,
                "targets": len(day.target_weights),
                "fills": len(day.orders),
                "equity": round(day.equity, 2),
                "bot P&L": round(float(pnl_row.get("bot_pnl", 0.0) or 0.0), 2),
                "Nifty P&L": round(float(pnl_row.get("nifty_pnl", 0.0) or 0.0), 2)
                if not pd.isna(pnl_row.get("nifty_pnl", float("nan")))
                else None,
                "difference": round(float(pnl_row.get("difference", 0.0) or 0.0), 2)
                if not pd.isna(pnl_row.get("difference", float("nan")))
                else None,
            }
        )
    return pd.DataFrame(rows)


def _render_day_detail(day: Any) -> None:
    """One day's decisions, fills and cumulative book."""
    st.markdown(f"**{ui.fmt_date(day.date_key)} · {day.regime or 'n/a'}**")
    left, right = st.columns(2)
    with left:
        st.markdown("**Fills at this open**")
        if not day.orders:
            st.caption("No fills today. Decisions at the close fill at the next open.")
        else:
            st.dataframe(pd.DataFrame(day.orders), width="stretch", hide_index=True)
    with right:
        st.markdown("**Holdings after this open**")
        if day.holdings:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"symbol": symbol, "quantity": quantity}
                        for symbol, quantity in sorted(day.holdings.items())
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No positions owned after this open.")

    st.caption(
        f"Entry threshold applied: "
        f"{'n/a' if day.entry_threshold is None else f'{day.entry_threshold:.6f}'} · "
        f"{day.candidates} candidates · {day.passed_threshold} above threshold."
    )

    if day.target_weights:
        st.markdown(
            f"**Target weights at this close** "
            f"({len(day.target_weights)} name(s), exposure "
            f"{sum(day.target_weights.values()):.4f}):"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"symbol": symbol, "target weight": round(weight, 6)}
                    for symbol, weight in sorted(day.target_weights.items())
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("No target weights on this day.")


def _live_header(day: Any, index: int, total: int, run: Any) -> None:
    st.markdown(
        f'<div class="rt-live-head">'
        f'<span class="rt-badge live">Simulated live</span>'
        f'<span class="rt-live-day">Day {index + 1} of {total}</span>'
        f'<span class="rt-live-date">{ui.fmt_date(day.date_key)}</span>'
        f"{ui.regime_badge(day.regime)}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Fake money · {ui.fmt_inr(run.initial_capital)} starting capital · "
        f"benchmark: {run.index_label or run.index_symbol or 'configured index'} "
        f"buy-and-hold · not a live market feed."
    )


def _live_cards(row: Any, run: Any) -> None:
    columns = st.columns(4)
    bot_pnl = row.get("bot_pnl")
    nifty_pnl = row.get("nifty_pnl")
    difference = row.get("difference")
    difference_return = row.get("difference_return")
    with columns[0]:
        st.metric(
            "Fake portfolio",
            ui.fmt_inr(row.get("bot_equity")),
            ui.fmt_pct(row.get("bot_return")),
        )
    with columns[1]:
        st.metric(
            "Bot P&L",
            ui.fmt_inr(bot_pnl),
            ui.fmt_pct(row.get("bot_return")),
            help="Net of the configured simulated costs and slippage.",
        )
    with columns[2]:
        st.metric(
            f"{run.index_label or run.index_symbol or 'Index'} buy-and-hold",
            "n/a" if pd.isna(row.get("nifty_equity")) else ui.fmt_inr(row.get("nifty_equity")),
            "n/a" if pd.isna(nifty_pnl) else ui.fmt_pct(row.get("nifty_return")),
            help="Same starting capital, invested in the configured index from day-1 close.",
        )
    with columns[3]:
        st.metric(
            "Difference vs index",
            "n/a" if pd.isna(difference) else ui.fmt_inr(difference),
            "n/a" if pd.isna(difference_return) else ui.fmt_pp(difference_return),
        )


def _pnl_chart(pnl: pd.DataFrame, run: Any) -> Any:
    import plotly.graph_objects as go

    x = pd.to_datetime(pnl["date_key"].astype(int).astype(str), format="%Y%m%d")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=pnl["bot_equity"],
            mode="lines+markers",
            name="Bot (fake money)",
            line=dict(color=ui.ACCENT, width=2.4),
            marker=dict(size=3),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
        )
    )
    if pnl["nifty_equity"].notna().any():
        figure.add_trace(
            go.Scatter(
                x=x,
                y=pnl["nifty_equity"],
                mode="lines",
                name=f"{run.index_label or 'Index'} buy-and-hold",
                line=dict(color="#94A3B8", width=1.6, dash="dash"),
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>",
            )
        )
    if run.initial_capital:
        figure.add_hline(
            y=float(run.initial_capital),
            line_dash="dot",
            line_color=ui.BORDER,
            line_width=1,
        )
    figure.update_layout(
        title="Fake portfolio vs index buy-and-hold — same starting capital",
        xaxis_title="date",
        yaxis_title="portfolio value",
        height=380,
        yaxis_tickprefix="₹",
        yaxis_tickformat=",.0f",
    )
    return ui.apply_figure(figure)


def _render_bot_live(run: Any, speed: str = "Normal") -> None:
    """Reveal the stored replay day by day; this is presentation, not a live feed."""
    import time

    from src.dashboard import bot

    pnl = bot.portfolio_pnl(run)
    if pnl.empty:
        _render_bot(run)
        return

    delay = {"Fast": 0.03, "Normal": 0.12, "Slow": 0.35}.get(speed, 0.12)
    header = st.empty()
    cards = st.empty()
    chart = st.empty()
    detail = st.empty()
    progress = st.progress(0.0, text="starting the replay")

    for index, day in enumerate(run.days):
        row = pnl.iloc[index]
        with header.container():
            _live_header(day, index, len(run.days), run)
        with cards.container():
            _live_cards(row, run)
        if index == 0 or index % 2 == 0 or index == len(run.days) - 1:
            chart.plotly_chart(_pnl_chart(pnl.iloc[: index + 1], run), width="stretch")
        with detail.container():
            _render_day_detail(day)
        progress.progress(
            (index + 1) / len(run.days),
            text=f"day {index + 1} of {len(run.days)} · {ui.fmt_date(day.date_key)}",
        )
        time.sleep(delay)

    progress.empty()
    for placeholder in (header, cards, chart, detail):
        placeholder.empty()
    _render_bot(
        run,
        chart_key="bot_pnl_live_full",
        day_key="bot_day_inspector_live",
    )


def _read_proof() -> dict[str, Any] | None:
    from pathlib import Path

    path = Path("reports/results/bot_equivalence.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------- entry point

SCREENS = {
    "Market overview": screen_market,
    "OLAP explorer": screen_olap,
    "Today's signals": screen_signals,
    "Backtest results": screen_results,
    "Model diagnostics": screen_diagnostics,
    "Bot demo (replay)": screen_bot,
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
    st.set_page_config(
        page_title="RegimeTrader",
        page_icon="▦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    cfg, engine = dd.cfg_and_engine(config_path())
    ui.inject_css()
    choice = _sidebar()
    _header(cfg, engine)
    SCREENS[choice](cfg, engine)
    ui.footer(dd.DISCLAIMER)


if __name__ == "__main__":
    main()
