"""Market regime detection with K-Means (`spec 04` sections 5.1 to 5.4).

The leakage rule that matters here: K-Means is fitted on the **training period only**
and the centres are then frozen for every later date (`DECISIONS` R11). Fitting on all
years would let future market behaviour shape today's regime definition.

`regime.use_vix: auto` uses VIX only when its coverage on the training period clears
`min_vix_coverage`, and the decision is logged so the report can state it (R12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.common.config import ConfigError, load_config
from src.common.db import get_engine, init_db, read_sql
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.common.trial_log import log_trial
from src.features.indicators import build_indicators

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2

#: Feature columns, excluding the regime one-hots which are derived, not fitted.
MARKET_FEATURES = ("mkt_return_20d", "mkt_vol_20d", "india_vix", "mkt_drawdown")

#: Regime labels, matching `dim_regime`.
REGIME_LABELS = ("BULL", "SIDEWAYS", "HIGH_VOL", "BEAR")


def build_market_features(
    index_close: pd.Series,
    vix_close: pd.Series | None,
    dd_window: int = 252,
    dd_min_periods: int = 20,
) -> pd.DataFrame:
    """Build the market-level features used to detect regimes.

    Args:
        index_close: Index close series, ascending by date.
        vix_close: VIX close series, or None when VIX is not used.
        dd_window: Drawdown lookback window.
        dd_min_periods: Minimum periods for the drawdown rolling max. Set to 20 rather
            than 252 so a regime exists from about day 21 (DECISIONS R11).

    Returns:
        A DataFrame indexed like `index_close` with `mkt_return_20d`, `mkt_vol_20d`,
        `india_vix` (when VIX is used) and `mkt_drawdown`.
    """
    out = pd.DataFrame(index=index_close.index)
    out["mkt_return_20d"] = index_close.pct_change(20)
    out["mkt_vol_20d"] = index_close.pct_change().rolling(20).std() * np.sqrt(252)
    if vix_close is not None:
        out["india_vix"] = vix_close.reindex(out.index).ffill(limit=2)
    out["mkt_drawdown"] = (
        index_close / index_close.rolling(dd_window, min_periods=dd_min_periods).max() - 1
    )
    return out


def decide_use_vix(
    vix_close: pd.Series | None, train_index: pd.Index, cfg: Mapping[str, Any]
) -> bool:
    """Decide whether to use VIX, per `regime.use_vix` (DECISIONS R12).

    `auto` uses VIX only when its coverage on the training period is at least
    `min_vix_coverage`. The decision is returned so it can be logged.

    Args:
        vix_close: VIX series, or None when there is none.
        train_index: Dates making up the training period.
        cfg: Loaded config.

    Returns:
        True when VIX should be used.
    """
    setting = cfg["regime"].get("use_vix", "auto")
    if setting is False:
        return False
    if vix_close is None or vix_close.dropna().empty:
        if setting is True:
            raise ValueError("regime.use_vix is true but no VIX data is available")
        return False
    if setting is True:
        return True

    train = vix_close.reindex(train_index).dropna()
    if len(train_index) == 0:
        return False
    coverage = len(train) / len(train_index)
    threshold = float(cfg["regime"]["min_vix_coverage"])
    return coverage >= threshold


def choose_k(
    train: pd.DataFrame,
    features: list[str],
    k_range: list[int],
    seed: int,
) -> pd.DataFrame:
    """Score each k by inertia and silhouette on the **training period only**.

    Args:
        train: Training rows with the feature columns.
        features: Feature column names.
        k_range: Candidate values of k.
        seed: Random seed from config.

    Returns:
        A DataFrame with columns `k`, `inertia`, `silhouette`, `n_days`.
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(train[features])
    rows: list[dict[str, Any]] = []
    for k in k_range:
        km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(x)
        rows.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x, km.labels_)),
                "n_days": int(len(train)),
            }
        )
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def label_clusters(train: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Name clusters from their average behaviour on the training data.

    Highest mean 20-day return is BULL, lowest is BEAR; of the rest the highest
    volatility is HIGH_VOL and the others are SIDEWAYS.

    Args:
        train: Training feature rows.
        labels: Cluster assignment per row.

    Returns:
        Mapping of cluster id to regime name.
    """
    stats = (
        train.assign(cluster=np.asarray(labels))
        .groupby("cluster")[["mkt_return_20d", "mkt_vol_20d"]]
        .mean()
    )
    names: dict[int, str] = {
        int(stats["mkt_return_20d"].idxmax()): "BULL",
        int(stats["mkt_return_20d"].idxmin()): "BEAR",
    }
    rest = [c for c in stats.index if int(c) not in names]
    if len(rest) == 1:
        names[int(rest[0])] = "SIDEWAYS"
    elif len(rest) > 1:
        high_vol = max(rest, key=lambda c: stats.loc[c, "mkt_vol_20d"])
        for cluster in rest:
            names[int(cluster)] = "HIGH_VOL" if cluster == high_vol else "SIDEWAYS"
    return names


def fit_regime_model(
    train: pd.DataFrame, features: list[str], k: int, seed: int
) -> tuple[StandardScaler, KMeans, dict[int, str]]:
    """Fit the scaler and K-Means on the training period and name the clusters.

    Args:
        train: Training rows with the feature columns.
        features: Feature column names.
        k: Number of clusters.
        seed: Random seed from config.

    Returns:
        `(scaler, kmeans, name_map)`.
    """
    scaler = StandardScaler().fit(train[features])
    km = KMeans(n_clusters=int(k), n_init=10, random_state=seed).fit(
        scaler.transform(train[features])
    )
    return scaler, km, label_clusters(train, km.labels_)


def predict_clusters(
    df: pd.DataFrame, features: list[str], scaler: StandardScaler, km: KMeans
) -> pd.Series:
    """Predict raw cluster ids with the frozen scaler and centres.

    Args:
        df: Rows to label.
        features: Feature column names.
        scaler: Fitted scaler from the training period.
        km: Fitted K-Means from the training period.

    Returns:
        Series of cluster ids indexed like `df`.
    """
    return pd.Series(km.predict(scaler.transform(df[features])), index=df.index)


def smooth_trailing_mode(labels: pd.Series, window: int = 5) -> pd.Series:
    """Trailing mode of the last `window` labels, using the past only.

    Ties go to the most recent tied label, so this never looks ahead.

    Args:
        labels: Raw labels.
        window: Lookback window.

    Returns:
        Smoothed labels on the same index.
    """
    values = np.asarray(labels)
    out: list[Any] = []
    for i in range(len(values)):
        window_values = values[max(0, i - window + 1) : i + 1]
        counts = pd.Series(window_values).value_counts()
        tied = set(counts[counts == counts.max()].index)
        out.append(next(lab for lab in window_values[::-1] if lab in tied))
    return pd.Series(out, index=labels.index)


def smooth_consecutive(labels: pd.Series, days: int = 3) -> pd.Series:
    """Switch only after `days` consecutive days in a new label.

    Args:
        labels: Raw labels.
        days: Consecutive days required before switching.

    Returns:
        Smoothed labels on the same index.
    """
    values = np.asarray(labels)
    out = [values[0]] if len(values) else []
    current = values[0] if len(values) else None
    candidate = current
    streak = 0
    for value in values[1:]:
        if value == current:
            candidate, streak = current, 0
        elif value == candidate:
            streak += 1
            if streak >= days:
                current, candidate, streak = value, value, 0
        else:
            candidate, streak = value, 1
        out.append(current)
    return pd.Series(out, index=labels.index)


def nearest_centre(z: np.ndarray, centres: np.ndarray) -> tuple[int, np.ndarray]:
    """Index of the nearest centre and all Euclidean distances.

    Args:
        z: One observation in scaled space.
        centres: Cluster centres.

    Returns:
        `(argmin, distances)`.
    """
    distances = np.sqrt(((np.asarray(centres) - np.asarray(z)) ** 2).sum(axis=1))
    return int(distances.argmin()), distances


def _load_market_series(engine: Any, cfg: Mapping[str, Any]) -> tuple[pd.Series, pd.Series | None]:
    """Index close and VIX close, indexed by date_key."""
    index_symbol = cfg["data"]["index_symbol"]
    vix_symbol = cfg["data"]["vix_symbol"]

    index_rows = read_sql(
        engine,
        "SELECT f.date_key AS date_key, f.close_price AS close "
        "FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE s.symbol = :symbol ORDER BY f.date_key",
        {"symbol": index_symbol},
    )
    if index_rows.empty:
        raise ValueError("no index rows in fact_price; run the ETL first")
    index_close = pd.Series(
        index_rows["close"].to_numpy(), index=pd.Index(index_rows["date_key"].astype(int))
    )

    vix_rows = read_sql(
        engine,
        "SELECT f.date_key AS date_key, f.close_price AS close "
        "FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key "
        "WHERE s.symbol = :symbol ORDER BY f.date_key",
        {"symbol": vix_symbol},
    )
    vix_close = None
    if not vix_rows.empty:
        vix_close = pd.Series(
            vix_rows["close"].to_numpy(), index=pd.Index(vix_rows["date_key"].astype(int))
        )
    return index_close, vix_close


def _profile_and_counts(
    features: pd.DataFrame, labels: pd.Series, train_mask: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`regime_profile.csv` and `regime_counts.csv` per `04` section 5.4.

    The profile is the training period only (DECISIONS R27), computed in pandas.
    The counts carry both training and all-date day totals.

    Args:
        features: Market features indexed by date_key.
        labels: Smoothed labels on the same index.
        train_mask: Boolean mask selecting training-period rows.

    Returns:
        `(profile, counts)`.
    """
    columns = [c for c in MARKET_FEATURES if c in features.columns]
    train_labels = labels[train_mask]
    profile = (
        features[train_mask]
        .assign(regime=train_labels.to_numpy())
        .groupby("regime")[columns]
        .mean()
        .reset_index()
    )

    regimes = sorted(labels.unique())
    counts = pd.DataFrame(
        {
            "regime": regimes,
            "days_train": [int((train_labels == r).sum()) for r in regimes],
            "days_all": [int((labels == r).sum()) for r in regimes],
        }
    )
    return profile, counts


def _sanity_checks(profile: pd.DataFrame) -> list[str]:
    """Non-circular sanity checks per `04` section 5.4. Returns the failures."""
    failures: list[str] = []
    stats = profile.set_index("regime")
    if "BULL" in stats.index and "mkt_return_20d" in stats.columns:
        if stats.loc["BULL", "mkt_return_20d"] <= 0:
            failures.append("BULL mean 20-day return is not above 0")
        if "SIDEWAYS" in stats.index and (
            stats.loc["BULL", "mkt_return_20d"] <= stats.loc["SIDEWAYS", "mkt_return_20d"]
        ):
            failures.append("BULL mean 20-day return is not above SIDEWAYS")
    if "mkt_vol_20d" in stats.columns:
        volatile = [r for r in ("BEAR", "HIGH_VOL") if r in stats.index]
        calm = [r for r in ("BULL", "SIDEWAYS") if r in stats.index]
        if volatile and calm:
            hot = stats.loc[volatile, "mkt_vol_20d"].min()
            cold = stats.loc[calm, "mkt_vol_20d"].max()
            if hot <= cold:
                failures.append("BEAR/HIGH_VOL volatility is not above BULL/SIDEWAYS")
    if "mkt_drawdown" in stats.columns and "BULL" in stats.index and "BEAR" in stats.index:
        if stats.loc["BEAR", "mkt_drawdown"] >= stats.loc["BULL", "mkt_drawdown"]:
            failures.append("BEAR mean drawdown is not below BULL's")
    return failures


def _write_regimes(
    engine: Any, features: pd.DataFrame, cluster_ids: pd.Series, labels: pd.Series
) -> int:
    """Upsert `fact_market_regime`."""

    from src.common.db import execute_many

    keys = {
        int(r["regime_key"]): r["regime_label"]
        for r in read_sql(engine, "SELECT regime_key, regime_label FROM dim_regime").to_dict(
            "records"
        )
    }
    label_to_key = {label: key for key, label in keys.items()}

    rows = []
    for position, (date_key, row) in enumerate(features.iterrows()):
        rows.append(
            {
                "date_key": int(date_key),
                "regime_key": int(label_to_key[labels.iloc[position]]),
                "cluster_id": int(cluster_ids.iloc[position]),
                "mkt_return_20d": _num(row.get("mkt_return_20d")),
                "mkt_vol_20d": _num(row.get("mkt_vol_20d")),
                "india_vix": _num(row.get("india_vix")),
                "mkt_drawdown": _num(row.get("mkt_drawdown")),
            }
        )

    upsert = """
        INSERT INTO fact_market_regime (
            date_key, regime_key, cluster_id, mkt_return_20d, mkt_vol_20d, india_vix, mkt_drawdown
        )
        VALUES (:date_key, :regime_key, :cluster_id, :mkt_return_20d, :mkt_vol_20d,
                :india_vix, :mkt_drawdown)
        ON CONFLICT (date_key) DO UPDATE SET
            regime_key = excluded.regime_key,
            cluster_id = excluded.cluster_id,
            mkt_return_20d = excluded.mkt_return_20d,
            mkt_vol_20d = excluded.mkt_vol_20d,
            india_vix = excluded.india_vix,
            mkt_drawdown = excluded.mkt_drawdown
    """
    with engine.begin() as conn:
        return execute_many(conn, upsert, rows)


def _num(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def _plot_k_selection(table: pd.DataFrame, path: Path) -> None:
    """Elbow and silhouette figure for the k-selection review."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(table["k"], table["inertia"], marker="o")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("inertia")
    axes[0].set_title("Elbow")
    axes[1].plot(table["k"], table["silhouette"], marker="o")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("silhouette")
    axes[1].set_title("Silhouette")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _plot_regime_bands(engine: Any, path: Path, index_symbol: str) -> None:
    """Index close with coloured regime bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = read_sql(
        engine,
        "SELECT d.full_date AS full_date, p.close_price AS close, r.regime_label AS regime "
        "FROM fact_market_regime m "
        "JOIN dim_date d ON d.date_key = m.date_key "
        "JOIN dim_regime r ON r.regime_key = m.regime_key "
        "JOIN fact_price p ON p.date_key = m.date_key "
        "JOIN dim_stock s ON s.stock_key = p.stock_key AND s.symbol = :symbol "
        "ORDER BY d.full_date",
        {"symbol": index_symbol},
    )
    if rows.empty:
        return

    dates = pd.to_datetime(rows["full_date"])
    colours = {
        "BULL": "#2e7d32",
        "SIDEWAYS": "#8d8d8d",
        "HIGH_VOL": "#ef6c00",
        "BEAR": "#c62828",
    }
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(dates, rows["close"], color="#141011", linewidth=0.8)
    for regime, colour in colours.items():
        mask = rows["regime"] == regime
        axis.fill_between(
            dates, rows["close"].min(), rows["close"].max(), where=mask, color=colour, alpha=0.12
        )
    axis.set_title(f"Index close with market regimes ({index_symbol})")
    axis.set_ylabel("close")
    handles = [Patch(facecolor=c, label=r) for r, c in colours.items()]
    axis.legend(handles=handles, loc="upper left", ncol=4, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def run(
    cfg: dict[str, Any],
    *,
    fit: bool,
    k_select_only: bool,
    k_override: int | None,
) -> int:
    """Select k, fit the model and write the regime table."""
    log = get_logger(__name__)
    engine = get_engine(cfg)
    init_db(engine)

    build_indicators(engine, cfg)

    index_close, vix_close = _load_market_series(engine, cfg)
    train_end = int(pd.Timestamp(cfg["splits"]["train_end"]).strftime("%Y%m%d"))
    train_index = index_close.index[index_close.index <= train_end]

    use_vix = decide_use_vix(vix_close, train_index, cfg)
    log.info("VIX decision (regime.use_vix=%s): %s", cfg["regime"]["use_vix"], use_vix)

    features = build_market_features(
        index_close,
        vix_close if use_vix else None,
        dd_window=int(cfg["regime"]["drawdown_window"]),
        dd_min_periods=int(cfg["regime"]["drawdown_min_periods"]),
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    # Honour `regime.features` from config rather than a hard-coded list.
    columns = [c for c in cfg["regime"]["features"] if c in features.columns]
    if "india_vix" in columns and not use_vix:
        # DECISIONS R12: when VIX is dropped it goes from the regime features too.
        columns.remove("india_vix")
        features = features.drop(columns=["india_vix"])
    features = features.dropna(subset=columns).sort_index()

    train_mask = features.index <= train_end
    train = features[train_mask]
    if len(train) < int(max(cfg["regime"]["k_range"])):
        log.error("only %d training rows, too few to fit", len(train))
        return EXIT_DATA_FAILURE

    # Step 1: k selection on the training period only.
    table = choose_k(train, columns, list(cfg["regime"]["k_range"]), int(cfg["project"]["seed"]))
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "regime_k_selection.csv", index=False)
    _plot_k_selection(table, Path(cfg["paths"]["figures"]) / "regime_k_selection.png")

    for row in table.to_dict("records"):
        log_trial(
            cfg,
            phase="regime",
            kind="regime_k",
            run_id=f"k{int(row['k'])}",
            params={"k": int(row["k"]), "features": columns, "use_vix": use_vix},
            period="train",
            metric="silhouette",
            value=float(row["silhouette"]),
            extra={"inertia": float(row["inertia"]), "n_days": int(row["n_days"])},
        )

    if k_select_only:
        log.info("k selection written to regime_k_selection.csv")
        return EXIT_OK

    k = int(k_override or cfg["regime"]["k"])
    log.info("fitting K-Means with k=%d on %d training rows", k, len(train))

    # Step 2: fit on the training period only, centres frozen afterwards.
    scaler, km, name_map = fit_regime_model(train, columns, k, int(cfg["project"]["seed"]))
    all_clusters = predict_clusters(features, columns, scaler, km)
    all_names = pd.Series([name_map[int(c)] for c in all_clusters], index=features.index)

    smoothing = cfg["regime"]["smoothing"]
    if smoothing["method"] == "consecutive":
        smoothed = smooth_consecutive(all_names, int(cfg["regime"]["smoothing_consecutive_days"]))
    else:
        smoothed = smooth_trailing_mode(all_names, int(smoothing["window"]))

    profile, counts = _profile_and_counts(features, smoothed, train_mask)
    profile.to_csv(results_dir / "regime_profile.csv", index=False)
    counts.to_csv(results_dir / "regime_counts.csv", index=False)

    # Sanity checks run on the training-period profile, per `04` section 5.4.
    failures = _sanity_checks(profile)
    for failure in failures:
        log.warning("regime sanity check failed: %s", failure)

    written = _write_regimes(engine, features, all_clusters, smoothed)
    log.info("wrote %d regime row(s)", written)

    models_dir = Path(cfg["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": km,
            "name_map": name_map,
            "features": columns,
            "train_end": cfg["splits"]["train_end"],
            "k": k,
            "seed": int(cfg["project"]["seed"]),
            "use_vix": bool(use_vix),
        },
        models_dir / "regime_model.joblib",
    )

    _plot_regime_bands(
        engine, Path(cfg["paths"]["figures"]) / "regime_bands.png", cfg["data"]["index_symbol"]
    )

    write_manifest(
        cfg,
        "regime_model",
        engine,
        extra={"k": k, "use_vix": bool(use_vix), "days": written, "sanity_failures": failures},
    )
    log.info("regime model complete")
    return EXIT_OK


def classify_day(
    model_bundle: Mapping[str, Any], row: pd.Series, history_labels: list[str]
) -> tuple[int, str]:
    """Classify one day for the daily paper job.

    Args:
        model_bundle: The saved `regime_model.joblib` contents.
        row: One row of market features for the day.
        history_labels: Recent smoothed labels, oldest first.

    Returns:
        `(raw_cluster, smoothed_label)`.
    """
    features: list[str] = model_bundle["features"]
    scaler = model_bundle["scaler"]
    km = model_bundle["kmeans"]
    name_map = model_bundle["name_map"]

    z = scaler.transform(pd.DataFrame([{f: row[f] for f in features}]))
    cluster = int(km.predict(z)[0])
    name = name_map[cluster]
    window = int(model_bundle.get("smoothing_window", 5))
    combined = pd.Series(list(history_labels)[-(window - 1) :] + [name])
    smoothed = smooth_trailing_mode(combined, window).iloc[-1]
    return cluster, smoothed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.regime.regime_model",
        description="Detect market regimes with K-Means.",
    )
    parser.add_argument("--fit", action="store_true", help="k-selection then fit (default).")
    parser.add_argument("--k-select", action="store_true", help="Run the k-selection only.")
    parser.add_argument("--k", type=int, default=None, help="Override regime.k.")
    parser.add_argument("--config", default=None, help="Override config file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 unexpected, 2 data failure."""
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        return run(cfg, fit=args.fit, k_select_only=args.k_select, k_override=args.k)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the CLI must not traceback at the user
        get_logger(__name__).exception("regime model failed")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
