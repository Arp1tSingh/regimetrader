"""Visual language for the read-only dashboard.

The dashboard is a research console, not a trading terminal: off-white surfaces, one
teal accent, semantic regime colours and tabular numbers. This module holds the design
tokens, the global stylesheet, the Plotly theme and a small set of render helpers so the
six screens share one visual grammar.

It deliberately contains no data access and no business logic. Every number still comes
from `src.dashboard.data` or from the replay in `src.dashboard.bot`.
"""

from __future__ import annotations

import html
from typing import Any, Iterable

import pandas as pd

# ------------------------------------------------------------------ tokens

PAGE_BG = "#F7F8FA"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#F1F5F9"
INK = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
ACCENT = "#0F766E"
ACCENT_SOFT = "#CCFBF1"
POSITIVE = "#15803D"
NEGATIVE = "#B91C1C"
WARNING = "#B45309"
INFO = "#0369A1"

REGIME_ORDER = ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR")
REGIME_COLORS = {
    "BULL": "#15803D",
    "SIDEWAYS": "#B45309",
    "HIGH_VOL": "#C2410C",
    "BEAR": "#B91C1C",
}
REGIME_SOFT = {
    "BULL": "#DCFCE7",
    "SIDEWAYS": "#FEF3C7",
    "HIGH_VOL": "#FFEDD5",
    "BEAR": "#FEE2E2",
}

FONT_STACK = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"SFMono-Regular", Consolas, "Liberation Mono", monospace'

_CSS = f"""
<style>
  :root {{
    --rt-page: {PAGE_BG};
    --rt-surface: {SURFACE};
    --rt-surface-muted: {SURFACE_MUTED};
    --rt-ink: {INK};
    --rt-muted: {MUTED};
    --rt-border: {BORDER};
    --rt-accent: {ACCENT};
    --rt-accent-soft: {ACCENT_SOFT};
    --rt-positive: {POSITIVE};
    --rt-negative: {NEGATIVE};
    --rt-warning: {WARNING};
  }}
  html, body, [class*="css"] {{ font-family: {FONT_STACK}; }}
  .stApp {{ background: var(--rt-page); color: var(--rt-ink); }}
  .block-container {{ max-width: 1520px; padding-top: 1.8rem; padding-bottom: 3.2rem; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  [data-testid="stSidebar"] {{
    background: var(--rt-surface);
    border-right: 1px solid var(--rt-border);
  }}
  [data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  [data-testid="stSidebar"] hr {{ margin: 0.9rem 0; }}

  h1, h2, h3, h4 {{ color: var(--rt-ink); letter-spacing: -0.025em; }}
  h1 {{ font-size: 2.15rem; font-weight: 720; }}
  h2 {{ font-size: 1.45rem; font-weight: 680; }}
  h3 {{ font-size: 1.05rem; font-weight: 650; }}
  p, li, .stMarkdown {{ color: var(--rt-ink); }}
  [data-testid="stCaptionContainer"] {{ color: var(--rt-muted); }}
  code {{ font-family: {MONO_STACK}; font-size: 0.86em; }}

  .rt-eyebrow {{
    color: var(--rt-accent);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
  }}
  .rt-title {{
    color: var(--rt-ink);
    font-size: 2.05rem;
    font-weight: 730;
    letter-spacing: -0.035em;
    line-height: 1.12;
    margin: 0;
  }}
  .rt-subtitle {{
    color: var(--rt-muted);
    font-size: 0.96rem;
    line-height: 1.55;
    max-width: 880px;
    margin-top: 0.5rem;
  }}
  .rt-rule {{ height: 1px; background: var(--rt-border); margin: 1.2rem 0 1.4rem; }}
  .rt-section {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    margin: 0.4rem 0 0.2rem;
  }}
  .rt-section h2 {{ margin: 0; }}
  .rt-section-note {{ color: var(--rt-muted); font-size: 0.82rem; }}

  .rt-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    border: 1px solid var(--rt-border);
    border-radius: 999px;
    padding: 0.16rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
  }}
  .rt-badge.live {{ color: #B91C1C; background: #FEF2F2; border-color: #FECACA; }}
  .rt-badge.accent {{
    color: var(--rt-accent);
    background: var(--rt-accent-soft);
    border-color: #99F6E4;
  }}
  .rt-badge.muted {{ color: var(--rt-muted); background: var(--rt-surface-muted); }}
  .rt-regime {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.16rem 0.55rem;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }}
  .rt-regime::before {{
    content: "";
    width: 0.42rem;
    height: 0.42rem;
    border-radius: 999px;
    background: currentColor;
  }}

  .rt-card {{
    background: var(--rt-surface);
    border: 1px solid var(--rt-border);
    border-radius: 16px;
    padding: 1rem 1.1rem;
  }}
  .rt-card-label {{
    color: var(--rt-muted);
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}
  .rt-card-value {{
    color: var(--rt-ink);
    font-size: 1.45rem;
    font-weight: 720;
    font-variant-numeric: tabular-nums;
    margin-top: 0.25rem;
  }}
  .rt-card-note {{ color: var(--rt-muted); font-size: 0.78rem; margin-top: 0.2rem; }}

  .rt-callout {{
    background: var(--rt-surface);
    border: 1px solid var(--rt-border);
    border-left: 4px solid var(--rt-accent);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    color: var(--rt-ink);
    font-size: 0.9rem;
    line-height: 1.55;
    margin: 0.5rem 0 0.8rem;
  }}
  .rt-callout.warn {{ border-left-color: var(--rt-warning); background: #FFFBEB; }}
  .rt-callout.live {{ border-left-color: var(--rt-negative); background: #FFF7F7; }}
  .rt-callout.info {{ border-left-color: {INFO}; background: #F0F9FF; }}

  .rt-chip-row {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.35rem 0 0.8rem; }}
  .rt-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid var(--rt-border);
    border-radius: 999px;
    background: var(--rt-surface);
    padding: 0.28rem 0.6rem;
    color: var(--rt-muted);
    font-size: 0.78rem;
  }}
  .rt-chip strong {{ color: var(--rt-ink); font-variant-numeric: tabular-nums; }}

  .rt-live-head {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.2rem;
  }}
  .rt-live-day {{
    color: var(--rt-ink);
    font-size: 1.3rem;
    font-weight: 720;
    letter-spacing: -0.02em;
  }}
  .rt-live-date {{ color: var(--rt-muted); font-size: 0.86rem; }}

  .rt-footer {{
    border-top: 1px solid var(--rt-border);
    color: var(--rt-muted);
    font-size: 0.76rem;
    line-height: 1.6;
    margin-top: 2rem;
    padding-top: 0.8rem;
  }}

  [data-testid="stMetric"] {{
    background: var(--rt-surface);
    border: 1px solid var(--rt-border);
    border-radius: 14px;
    padding: 0.8rem 0.95rem;
  }}
  [data-testid="stMetricLabel"] {{
    color: var(--rt-muted);
    font-size: 0.7rem;
    font-weight: 750;
    letter-spacing: 0.09em;
    text-transform: uppercase;
  }}
  [data-testid="stMetricValue"] {{
    color: var(--rt-ink);
    font-size: 1.55rem;
    font-weight: 730;
    font-variant-numeric: tabular-nums;
  }}
  [data-testid="stMetricDelta"] {{ font-variant-numeric: tabular-nums; }}

  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--rt-surface);
    border-color: var(--rt-border) !important;
    border-radius: 16px;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 9px 9px 0 0;
    color: var(--rt-muted);
    font-weight: 650;
  }}
  .stTabs [aria-selected="true"] {{ color: var(--rt-accent); }}
  .stButton > button, .stDownloadButton > button {{
    border-radius: 10px;
    font-weight: 650;
  }}
  .stButton > button[kind="primary"] {{
    background: var(--rt-accent);
    border-color: var(--rt-accent);
  }}
  [data-testid="stDataFrame"] {{ border: 1px solid var(--rt-border); border-radius: 12px; }}
  [data-testid="stPlotlyChart"] {{ background: transparent; }}
  [data-testid="stExpander"] {{
    background: var(--rt-surface);
    border: 1px solid var(--rt-border);
    border-radius: 12px;
  }}
  [data-testid="stProgress"] > div > div > div > div {{ background-color: var(--rt-accent); }}
</style>
"""


def inject_css() -> None:
    """Install the global stylesheet. Called once per script run."""
    import streamlit as st

    st.markdown(_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ formatting


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def fmt_num(value: Any, places: int = 2) -> str:
    """Plain number with fixed decimals; `n/a` instead of an exception."""
    try:
        return f"{float(value):,.{places}f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: Any, *, signed: bool = True, places: int = 2) -> str:
    """Decimal fraction to a percentage string (`0.0123` -> `+1.23%`)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:+.{places}%}" if signed else f"{number:.{places}%}"


def fmt_pp(value: Any, *, signed: bool = True, places: int = 2) -> str:
    """Decimal fraction difference to percentage points."""
    try:
        number = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:+.{places}f} pp" if signed else f"{number:.{places}f} pp"


def fmt_inr(value: Any, *, decimals: int = 0, compact: bool = False) -> str:
    """Rupee amount with Indian digit grouping (lakh/crore style).

    The project reports in rupees, so `1000000` should read `₹10,00,000`, not
    `₹1,000,000`. Negative amounts keep the sign in front of the symbol.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    if compact and abs(number) >= 1_00_00_000:
        return f"{number / 1_00_00_000:+.2f} Cr" if number else "₹0"
    if compact and abs(number) >= 1_00_000:
        return f"{number / 1_00_000:+.2f} L" if number else "₹0"

    sign = "-" if number < 0 else ""
    # Format without grouping first: the Indian grouping below inserts the commas.
    # Pre-grouping with `,` leaves them in `rest` and produces output like
    # ``₹1,,04,3,,942`` instead of ``₹10,43,942``.
    text = f"{abs(number):.{decimals}f}"
    integer, dot, fraction = text.partition(".")
    if len(integer) > 3:
        last_three = integer[-3:]
        rest = integer[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        integer = ",".join([*groups, last_three])
    suffix = f".{fraction}" if dot else ""
    return f"{sign}₹{integer}{suffix}"


def fmt_date(key: Any) -> str:
    """`20240101` -> `01 Jan 2024`; tolerant of missing values."""
    if key is None or pd.isna(key):
        return "n/a"
    try:
        return pd.Timestamp(str(int(key))).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return "n/a"


def fmt_date_iso(key: Any) -> str:
    """`20240101` -> `2024-01-01`."""
    if key is None or pd.isna(key):
        return "n/a"
    try:
        return pd.Timestamp(str(int(key))).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return "n/a"


# ------------------------------------------------------------------ markup helpers


def eyebrow(text: str) -> None:
    import streamlit as st

    st.markdown(f'<div class="rt-eyebrow">{_esc(text)}</div>', unsafe_allow_html=True)


def page_title(title: str, subtitle: str | None = None, *, kicker: str | None = None) -> None:
    import streamlit as st

    if kicker:
        eyebrow(kicker)
    st.markdown(f'<h1 class="rt-title">{_esc(title)}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="rt-subtitle">{_esc(subtitle)}</p>', unsafe_allow_html=True)
    st.markdown('<div class="rt-rule"></div>', unsafe_allow_html=True)


def section(title: str, note: str | None = None) -> None:
    import streamlit as st

    right = f'<span class="rt-section-note">{_esc(note)}</span>' if note else ""
    st.markdown(
        f'<div class="rt-section"><h2>{_esc(title)}</h2>{right}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "muted") -> str:
    """Return a badge as HTML, for embedding in a larger markdown block."""
    return f'<span class="rt-badge {kind}">{_esc(text)}</span>'


def regime_badge(label: Any) -> str:
    """A semantic regime pill. Unknown labels stay neutral rather than guessing."""
    text = str(label or "n/a")
    color = REGIME_COLORS.get(text, MUTED)
    background = REGIME_SOFT.get(text, SURFACE_MUTED)
    return (
        f'<span class="rt-regime" style="color:{color};background:{background};">'
        f"{_esc(text)}</span>"
    )


def callout(text: str, kind: str = "info") -> None:
    """A bordered note. `kind` is `info`, `warn` or `live`."""
    import streamlit as st

    st.markdown(
        f'<div class="rt-callout {kind}">{text}</div>',
        unsafe_allow_html=True,
    )


def chip_row(items: Iterable[tuple[str, Any]]) -> None:
    """Compact label/value chips, e.g. regime counts."""
    import streamlit as st

    chips = "".join(
        f'<span class="rt-chip">{_esc(label)} <strong>{_esc(value)}</strong></span>'
        for label, value in items
    )
    st.markdown(f'<div class="rt-chip-row">{chips}</div>', unsafe_allow_html=True)


def footer(text: str) -> None:
    import streamlit as st

    st.markdown(f'<div class="rt-footer">{_esc(text)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ plotly theme


def apply_figure(figure: Any, *, height: int | None = None) -> Any:
    """Apply the console theme to a Plotly figure and return it."""
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_STACK, color=INK, size=12),
        margin=dict(l=12, r=12, t=56, b=92),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=INK)),
        # Keep the legend below the plot. At the top it collides with the chart title
        # whenever the title is long or the legend has several entries.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11),
        ),
        colorway=[ACCENT, "#64748B", "#2563EB", "#D97706", "#7C3AED", "#0891B2"],
    )
    if height is not None:
        figure.update_layout(height=height)
    figure.update_xaxes(
        gridcolor=BORDER,
        linecolor=BORDER,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )
    figure.update_yaxes(
        gridcolor=BORDER,
        linecolor=BORDER,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=MUTED, size=11),
    )
    return figure
