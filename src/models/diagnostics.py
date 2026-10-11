"""Validation-period classifier diagnostics (`DECISIONS` R35).

Reported, **not** used for eligibility or for strategy selection. Three diagnostics
per model, all on the validation year only:

1. ROC-AUC.
2. The per-day spread between the mean next-day return of the top prob_up quintile
   and the bottom prob_up quintile, with a day-clustered bootstrap interval. This is
   the cross-sectional question: does a higher probability rank a stock above a
   lower one *on the same day*?
3. The accuracy edge over the majority baseline, with a day-clustered bootstrap
   interval.

Resampling is clustered **by day**, not by row. Rows on one day are not independent:
50 stocks on one date share the market's move, so a row-level bootstrap would
understate the interval and manufacture significance.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    clusters: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap interval resampling whole clusters with replacement.

    Args:
        frame: Rows to resample.
        statistic: Maps a resampled frame to a scalar.
        clusters: Column whose distinct values form a cluster.
        n_boot: Number of resamples.
        seed: Seed from config, so the interval is reproducible.

    Returns:
        `(low, high)` at the 2.5% and 97.5% percentiles. `(nan, nan)` when the
        statistic is undefined on the data.
    """
    groups = frame[clusters].to_numpy()
    unique = np.unique(groups)
    if unique.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    index_by_cluster = {value: frame.index[groups == value] for value in unique}
    values = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_by_cluster[value] for value in picked])
        try:
            value = statistic(frame.loc[rows])
        except (ValueError, ZeroDivisionError, IndexError):
            continue
        if value is not None and np.isfinite(value):
            values.append(float(value))

    if len(values) < max(20, n_boot // 20):
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def quintile_spread(rows: pd.DataFrame) -> float:
    """Mean next-day return of the top prob_up quintile minus the bottom quintile.

    Quintiles are formed **within each day**, so the comparison is cross-sectional
    and a common market move cancels out.
    """
    values = []
    for _day, group in rows.groupby("date_key", sort=True):
        if len(group) < 5:
            continue
        ordered = group.sort_values("prob_up")
        cut = max(1, len(ordered) // 5)
        bottom = ordered.head(cut)["fwd_return"].mean()
        top = ordered.tail(cut)["fwd_return"].mean()
        if pd.notna(bottom) and pd.notna(top):
            values.append(float(top) - float(bottom))
    if not values:
        return float("nan")
    return float(np.mean(values))


def accuracy_edge(rows: pd.DataFrame) -> float:
    """Accuracy of `prob_up > 0.5` minus the majority baseline for those rows."""
    actual = rows["target_up"].to_numpy().astype(int)
    if actual.size == 0:
        return float("nan")
    predicted = (rows["prob_up"].to_numpy() > 0.5).astype(int)
    accuracy = float((predicted == actual).mean())
    baseline = max(float(actual.sum()), float(actual.size - actual.sum())) / actual.size
    return accuracy - baseline


def roc_auc_by_model(cfg: Any) -> dict[str, float]:
    """Validation-year ROC-AUC per model, read from `classifier_metrics.csv`.

    Used by the R41 `no_edge_default` path, which must pick the best classifier
    without running a backtest. Reading the stored metrics keeps the choice inside the
    validation period; test-year rows are never read here.

    Args:
        cfg: Loaded config.

    Returns:
        `model -> roc_auc`, missing models omitted.
    """
    from pathlib import Path

    import pandas as pd

    path = Path(cfg["paths"]["results"]) / "classifier_metrics.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    validation = frame[frame["period"] == "validation"]
    if validation.empty:
        return {}
    return {
        str(row["model"]): float(row["roc_auc"])
        for _, row in validation.iterrows()
        if pd.notna(row["roc_auc"])
    }


def model_diagnostics(rows: pd.DataFrame, *, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    """All three diagnostics for one model's validation rows.

    Args:
        rows: Columns `date_key`, `prob_up`, `target_up`, `fwd_return`.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.

    Returns:
        A dict with `n_rows`, `n_days`, `roc_auc`, `accuracy`, `majority_baseline`,
        `edge_over_baseline`, `edge_ci_low`, `edge_ci_high`, `quintile_spread`,
        `quintile_ci_low`, `quintile_ci_high` and `days_in_spread`.
    """
    from sklearn.metrics import roc_auc_score

    actual = rows["target_up"].to_numpy().astype(int)
    proba = rows["prob_up"].to_numpy()

    auc = float(roc_auc_score(actual, proba)) if len(np.unique(actual)) > 1 else float("nan")
    predicted = (proba > 0.5).astype(int)
    accuracy = float((predicted == actual).mean()) if actual.size else float("nan")
    baseline = (
        max(float(actual.sum()), float(actual.size - actual.sum())) / actual.size
        if actual.size
        else float("nan")
    )

    spread = quintile_spread(rows)
    s_low, s_high = cluster_bootstrap(rows, quintile_spread, "date_key", n_boot=n_boot, seed=seed)
    edge = accuracy - baseline
    e_low, e_high = cluster_bootstrap(rows, accuracy_edge, "date_key", n_boot=n_boot, seed=seed)

    return {
        "n_rows": int(len(rows)),
        "n_days": int(rows["date_key"].nunique()),
        "roc_auc": auc,
        "accuracy": accuracy,
        "majority_baseline": baseline,
        "edge_over_baseline": edge,
        "edge_ci_low": e_low,
        "edge_ci_high": e_high,
        "quintile_spread": spread,
        "quintile_ci_low": s_low,
        "quintile_ci_high": s_high,
        "days_in_spread": int(sum(1 for _ in rows.groupby("date_key"))),
    }
