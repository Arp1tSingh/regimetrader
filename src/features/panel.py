"""The modelling panel (`spec 04` section 2.1 and section 4).

One row per (stock, trading day). **The label is the only forward-looking column**: it
is built from day *t+1*, and every feature uses data up to the close of *t*. The single
`shift(-1)` in this file is marked `# label-only` and `test_forbidden_patterns`
greps for any other.

Cached to `data/processed/panel.parquet` with a `panel.meta.json` sidecar; the cache is
rebuilt when the `features`, `regime`, `target` or `splits` config sections change, when
the row counts of `fact_indicator` / `fact_market_regime` move, or when the latest date
changes (`07` section 6). Tests always pass `use_cache=False`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

#: The 19 feature names, in exactly the documented order (`04` section 2.1).
FEATURE_COLS_ALL: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "px_to_sma20",
    "px_to_sma50",
    "sma20_to_sma50",
    "rsi_14",
    "macd_hist_pct",
    "bb_pct_b",
    "atr_pct",
    "vol_20d",
    "volume_ratio_20",
    "rel_strength_20",
    "mkt_ret_20d",
    "india_vix",
    "regime_bull",
    "regime_sideways",
    "regime_high_vol",
    "regime_bear",
)

#: Regime one-hot columns, always present even when k = 3 (`04` section 2.1).
REGIME_ONE_HOT: tuple[str, ...] = (
    "regime_bull",
    "regime_sideways",
    "regime_high_vol",
    "regime_bear",
)

#: Map a `dim_regime.regime_label` to its one-hot column.
REGIME_COLUMN = {
    "BULL": "regime_bull",
    "SIDEWAYS": "regime_sideways",
    "HIGH_VOL": "regime_high_vol",
    "BEAR": "regime_bear",
}

#: Warehouse `fact_indicator` column -> model feature name (DECISIONS R2).
WAREHOUSE_TO_FEATURE = {
    "return_1d": "ret_1d",
    "return_5d": "ret_5d",
    "return_20d": "ret_20d",
}

#: Target column names.
TARGET_COL = "target_up"
NEXT_RETURN_COL = "next_day_return"

#: Config sections hashed into the cache key.
CACHE_SECTIONS = ("features", "regime", "target", "splits")


def get_feature_cols(use_vix: bool) -> list[str]:
    """The active feature columns.

    Args:
        use_vix: Whether VIX is used. When False, `india_vix` is dropped, giving 18
            names (`DECISIONS` R12).

    Returns:
        Feature column names in the documented order.
    """
    if use_vix:
        return list(FEATURE_COLS_ALL)
    return [c for c in FEATURE_COLS_ALL if c != "india_vix"]


def _config_hash(cfg: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {section: cfg.get(section, {}) for section in CACHE_SECTIONS},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_paths(cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    parquet = Path(cfg["paths"]["processed"]) / "panel.parquet"
    return parquet, parquet.with_suffix(".meta.json")


def _read_cache_meta(cfg: Mapping[str, Any]) -> dict[str, Any]:
    _, meta_path = _cache_paths(cfg)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache_meta(cfg: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    _, meta_path = _cache_paths(cfg)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(dict(meta), indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def _current_meta(engine: Any, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Row counts and latest date, used to detect that the cache is stale."""
    from src.common.db import read_sql

    indicators = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_indicator")
    regimes = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_market_regime")
    latest = read_sql(engine, "SELECT MAX(date_key) AS d FROM fact_price")
    return {
        "config_hash": _config_hash(cfg),
        "fact_indicator_rows": int(indicators["n"].iloc[0] or 0),
        "fact_market_regime_rows": int(regimes["n"].iloc[0] or 0),
        "latest_date_key": int(latest["d"].iloc[0]) if latest["d"].iloc[0] is not None else None,
    }


def use_vix_in_regime(cfg: Mapping[str, Any], engine: Any) -> bool:
    """Whether VIX was used for the regime model, read back from the saved bundle."""
    import joblib

    bundle_path = Path(cfg["paths"]["models"]) / "regime_model.joblib"
    if not bundle_path.exists():
        return bool(cfg["regime"].get("use_vix", "auto") is True)
    try:
        return bool(joblib.load(bundle_path).get("use_vix", False))
    except Exception:  # noqa: BLE001 - a missing bundle just means "assume not"
        return False


def _load_base(engine: Any, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Prices, indicators, regimes and the index return, joined into one frame."""
    from src.common.db import read_sql

    equity = read_sql(
        engine,
        """
        SELECT f.date_key AS date_key, s.symbol AS symbol, f.close_price AS close,
               i.return_1d AS return_1d, i.return_5d AS return_5d, i.return_20d AS return_20d,
               i.sma_20 AS sma_20, i.sma_50 AS sma_50, i.macd AS macd,
               i.macd_signal AS macd_signal, i.rsi_14 AS rsi_14, i.bb_pct_b AS bb_pct_b,
               i.atr_14 AS atr_14, i.vol_20d AS vol_20d, i.volume_ratio_20 AS volume_ratio_20
        FROM fact_price f
        JOIN dim_stock s ON s.stock_key = f.stock_key
        LEFT JOIN fact_indicator i ON i.date_key = f.date_key AND i.stock_key = f.stock_key
        WHERE s.instrument_type = 'EQUITY'
        ORDER BY s.symbol, f.date_key
        """,
    )
    if equity.empty:
        return equity

    index = read_sql(
        engine,
        """
        SELECT f.date_key AS date_key, i.return_20d AS index_return_20d,
               f.close_price AS index_close
        FROM fact_price f
        JOIN dim_stock s ON s.stock_key = f.stock_key
        LEFT JOIN fact_indicator i ON i.date_key = f.date_key AND i.stock_key = f.stock_key
        WHERE s.symbol = :index_symbol
        ORDER BY f.date_key
        """,
        {"index_symbol": cfg["data"]["index_symbol"]},
    )

    vix = read_sql(
        engine,
        """
        SELECT f.date_key AS date_key, f.close_price AS vix_close
        FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key
        WHERE s.symbol = :vix_symbol
        """,
        {"vix_symbol": cfg["data"]["vix_symbol"]},
    )

    regimes = read_sql(
        engine,
        "SELECT m.date_key AS date_key, r.regime_label AS regime_label "
        "FROM fact_market_regime m JOIN dim_regime r ON r.regime_key = m.regime_key",
    )

    df = equity.merge(index, on="date_key", how="left")
    if not vix.empty:
        df = df.merge(vix, on="date_key", how="left")
    else:
        df["vix_close"] = pd.NA
    if not regimes.empty:
        df = df.merge(regimes, on="date_key", how="left")
    else:
        df["regime_label"] = pd.NA

    # `04` section 2.1: every derived feature, computed here rather than stored.
    df["px_to_sma20"] = df["close"] / df["sma_20"] - 1
    df["px_to_sma50"] = df["close"] / df["sma_50"] - 1
    df["sma20_to_sma50"] = df["sma_20"] / df["sma_50"] - 1
    df["macd_hist_pct"] = (df["macd"] - df["macd_signal"]) / df["close"]
    df["atr_pct"] = df["atr_14"] / df["close"]
    df["rel_strength_20"] = df["return_20d"] - df["index_return_20d"]
    df["mkt_ret_20d"] = df["index_return_20d"]

    # Rename the warehouse columns to their model-facing feature names (DECISIONS R2).
    df = df.rename(columns=WAREHOUSE_TO_FEATURE)

    if "vix_close" in df.columns:
        df["india_vix"] = df["vix_close"]
    return df


def _add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Attach `next_day_return` and `target_up` per stock (absolute target).

    The next bar is that stock's own next bar, not the index's. This is the only place a
    forward shift is allowed.
    """
    df = df.sort_values(["symbol", "date_key"], kind="stable").copy()
    next_close = df.groupby("symbol", sort=False)["close"].shift(-1)  # label-only
    df[NEXT_RETURN_COL] = next_close / df["close"] - 1
    df[TARGET_COL] = (df[NEXT_RETURN_COL] > 0).astype("float64")
    df.loc[df[NEXT_RETURN_COL].isna(), TARGET_COL] = pd.NA
    return df


def _add_relative_label(df: pd.DataFrame) -> pd.DataFrame:
    """Stretch V3: the target is the stock's return minus the index's, both to t+1.

    The index's next-day return is a market-wide value, so it is computed once per date
    and broadcast to every stock on that date.
    """
    df = df.sort_values(["symbol", "date_key"], kind="stable").copy()
    stock_next = df.groupby("symbol", sort=False)["close"].shift(-1) / df["close"] - 1  # label-only

    by_date = df.drop_duplicates("date_key").sort_values("date_key")
    index_next = by_date["index_close"].shift(-1) / by_date["index_close"] - 1  # label-only
    index_next_by_key = dict(zip(by_date["date_key"], index_next, strict=True))

    relative = stock_next - df["date_key"].map(index_next_by_key)
    df[TARGET_COL] = (relative > 0).astype("float64")
    df.loc[relative.isna(), TARGET_COL] = pd.NA
    df[NEXT_RETURN_COL] = stock_next
    return df


def _one_hot_regimes(df: pd.DataFrame) -> pd.DataFrame:
    for label, column in REGIME_COLUMN.items():
        df[column] = (df["regime_label"] == label).astype("float64")
    return df


def build_panel(
    engine: Any,
    cfg: Mapping[str, Any],
    *,
    live: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build the modelling panel.

    Args:
        engine: Warehouse engine.
        cfg: Loaded config.
        live: Keep the last unlabelled row of each stock, for prediction.
        use_cache: Read `data/processed/panel.parquet` when it is still valid.

    Returns:
        A DataFrame indexed by trade date (a `DatetimeIndex` named `date`) with
        `symbol`, the active feature columns, `target_up` and `next_day_return`.
        Rows with NaN in an active feature are dropped unless `live`, and warm-up rows
        are dropped per `features.warmup_rows`.
    """
    current = _current_meta(engine, cfg)
    cached_meta = _read_cache_meta(cfg)
    parquet, _ = _cache_paths(cfg)

    cache_keys = (
        "config_hash",
        "fact_indicator_rows",
        "fact_market_regime_rows",
        "latest_date_key",
    )
    if use_cache and parquet.exists() and cached_meta:
        if all(cached_meta.get(key) == current[key] for key in cache_keys):
            return pd.read_parquet(parquet)

    df = _load_base(engine, cfg)
    if df.empty:
        return df

    df["trade_date_key"] = df["date_key"]
    if cfg["target"].get("kind") == "relative":
        df = _add_relative_label(df)
    else:
        df = _add_label(df)
    del df["trade_date_key"]
    df = _one_hot_regimes(df)

    use_vix = use_vix_in_regime(cfg, engine)
    columns = get_feature_cols(use_vix)

    warmup = int(cfg["features"]["warmup_rows"])
    if warmup > 0:
        # Drop each stock's first N rows after its own first bar (`04` section 3).
        df["_row_no"] = df.groupby("symbol", sort=False).cumcount()
        df = df[df["_row_no"] >= warmup]
        df = df.drop(columns=["_row_no"])

    complete = df[columns].notna().all(axis=1)
    if live:
        # Keep each stock's final row even though it has no label.
        last_rows = df.groupby("symbol", sort=False).tail(1).index
        keep = complete | df.index.isin(last_rows)
    else:
        keep = complete & df[TARGET_COL].notna()
    df = df[keep]

    # `trade_date` is the panel index, matching the walk-forward contract.
    from src.common.dates import from_date_key

    out = df.reset_index(drop=True)
    out["trade_date"] = [from_date_key(int(k)) for k in out["date_key"]]
    out = out.sort_values("trade_date", kind="stable").reset_index(drop=True)

    keep_cols = ["symbol", "date_key", "trade_date", TARGET_COL, NEXT_RETURN_COL] + columns
    out = out[keep_cols].set_index("trade_date")
    out.index.name = "date"

    if use_cache:
        parquet.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(parquet)
        _write_cache_meta(cfg, {**current, "rows": int(len(out)), "use_vix": bool(use_vix)})

    return out
