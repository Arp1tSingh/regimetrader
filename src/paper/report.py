"""One-page daily report (`spec 08` section 4).

Lists the date, the latest data date, the regime and whether it changed, the portfolio's
equity and drawdown, the held positions with their weights, the fills that happened, the
orders queued for the next open, and any warnings.

**This report is not evidence of performance.** Paper trading shows that the system
runs end to end. The Phase 6 test evaluation found no demonstrated edge, and paper P&L
must never be quoted as if it changed that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DISCLAIMER = (
    "Paper trading only. No real money, no broker connection, no investment advice. "
    "This demonstrates that the system operates; it is not evidence that it makes money."
)


def write_report(cfg: dict[str, Any], engine: Any, run_id: str, date_key: int) -> Path:
    """Write `reports/daily/<date>.md` for one processed date.

    Args:
        cfg: Loaded config.
        engine: Warehouse engine.
        run_id: Paper run identifier.
        date_key: The date just processed.

    Returns:
        The path written.
    """
    from src.common.db import read_sql
    from src.paper.store import load_state, pending_targets

    day = pd_date(date_key)
    rows = read_sql(
        engine,
        "SELECT equity, cash, drawdown, positions_value FROM fact_portfolio_daily "
        "WHERE run_id = :r AND date_key = :d",
        {"r": run_id, "d": int(date_key)},
    )
    state = load_state(engine, run_id)
    equity = float(rows["equity"].iloc[0]) if not rows.empty else float("nan")
    drawdown = float(rows["drawdown"].iloc[0]) if not rows.empty else float("nan")
    cash = float(rows["cash"].iloc[0]) if not rows.empty else float("nan")

    regime_row = read_sql(
        engine,
        "SELECT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key WHERE m.date_key = :d",
        {"d": int(date_key)},
    )
    regime = str(regime_row["regime"].iloc[0]) if not regime_row.empty else "unknown"

    previous_regime = read_sql(
        engine,
        "SELECT r.regime_label AS regime FROM fact_market_regime m "
        "JOIN dim_regime r ON r.regime_key = m.regime_key "
        "WHERE m.date_key < :d ORDER BY m.date_key DESC LIMIT 1",
        {"d": int(date_key)},
    )
    previous = str(previous_regime["regime"].iloc[0]) if not previous_regime.empty else "unknown"
    changed = regime != previous

    latest = read_sql(
        engine,
        "SELECT MAX(date_key) AS d FROM fact_price",
    )
    latest_key = int(latest["d"].iloc[0]) if not latest.empty and latest["d"].iloc[0] else None

    fills = read_sql(
        engine,
        "SELECT s.symbol AS symbol, t.side AS side, t.quantity AS quantity, "
        "       t.fill_price AS fill_price, t.reason AS reason "
        "FROM fact_trade t JOIN dim_stock s ON s.stock_key = t.stock_key "
        "WHERE t.run_id = :r AND t.date_key = :d ORDER BY t.side, s.symbol",
        {"r": run_id, "d": int(date_key)},
    )

    from src.paper.daily_job import _regime_for  # noqa: PLC0415 - local import is fine

    del _regime_for

    targets = pending_targets(engine, run_id, date_key)
    probs = read_sql(
        engine,
        "SELECT s.symbol AS symbol, f.prob_up AS prob_up FROM fact_signal f "
        "JOIN dim_model m ON m.model_key = f.model_key "
        "JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE m.model_name = :m AND f.date_key = :d",
        {
            "m": str(cfg["strategy"].get("selected_model") or cfg["tuning"]["models"][0]),
            "d": int(date_key),
        },
    )
    prob_map = {str(r.symbol): float(r.prob_up) for r in probs.itertuples(index=False)}

    lines = [
        f"# RegimeTrader daily — {day}",
        "",
        f"**Run** `{run_id}`  |  **latest data date** "
        f"{pd_date(latest_key) if latest_key else 'n/a'}",
        "",
        f"**Regime: {regime}**" + ("  (CHANGED)" if changed else "  (unchanged)"),
        "",
        "## Portfolio",
        "",
        f"- Equity: **{equity:,.2f}**",
        f"- Cash: {cash:,.2f}",
        f"- Drawdown: {drawdown:.2%}",
        f"- Positions: {len(state.held)}",
        "",
        "| symbol | quantity | weight | prob_up |",
        "|---|---|---|---|",
    ]
    for symbol in sorted(state.held):
        position = state.held[symbol]
        weight = (
            float(position.quantity) * _last_close(engine, symbol, date_key) / equity
            if equity
            else float("nan")
        )
        lines.append(
            f"| {symbol} | {float(position.quantity):,.0f} | {weight:.2%} | "
            f"{prob_map.get(symbol, float('nan')):.4f} |"
        )
    if not state.held:
        lines.append("| _no positions_ | | | |")

    lines += ["", "## Fills today", ""]
    if fills.empty:
        lines.append("_none_")
    else:
        lines += ["| symbol | side | quantity | fill | reason |", "|---|---|---|---|---|"]
        for row in fills.itertuples(index=False):
            lines.append(
                f"| {row.symbol} | {row.side} | {float(row.quantity):,.0f} | "
                f"{float(row.fill_price):,.2f} | {row.reason} |"
            )

    lines += ["", "## Orders for the next open", ""]
    if not targets:
        lines.append("_flat: the strategy decided to hold no positions_")
    else:
        lines += ["| action | symbol | target weight | prob_up |", "|---|---|---|---|"]
        for symbol in sorted(targets):
            action = "HOLD" if symbol in state.held else "BUY"
            if symbol in state.held and symbol not in targets:
                action = "SELL"
            lines.append(
                f"| {action} | {symbol} | {targets[symbol]:.2%} | "
                f"{prob_map.get(symbol, float('nan')):.4f} |"
            )
        for symbol in sorted(set(state.held) - set(targets)):
            lines.append(f"| SELL | {symbol} | 0.00% | {prob_map.get(symbol, float('nan')):.4f} |")

    lines += ["", "## Warnings", "", "_none_", "", "---", "", f"_{DISCLAIMER}_", ""]

    directory = Path(cfg["paths"]["daily"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{day}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def pd_date(date_key: int | None) -> str:
    """`YYYYMMDD` to `YYYY-MM-DD`."""
    import pandas as pd

    if not date_key:
        return "n/a"
    return pd.to_datetime(str(int(date_key))).strftime("%Y-%m-%d")


def _last_close(engine: Any, symbol: str, date_key: int) -> float:
    from src.common.db import read_sql

    frame = read_sql(
        engine,
        "SELECT p.close_price AS c FROM fact_price p "
        "JOIN dim_stock s ON s.stock_key = p.stock_key "
        "WHERE s.symbol = :sym AND p.date_key <= :d ORDER BY p.date_key DESC LIMIT 1",
        {"sym": symbol, "d": int(date_key)},
    )
    return float(frame["c"].iloc[0]) if not frame.empty else 0.0
