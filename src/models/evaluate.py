"""Classifier evaluation (`spec 04` section 7).

Accuracy is always reported next to the **majority baseline** (always predict the more
common class), because a model at 56% on a 51/49 split has done nothing (`TESTS`
section 5 worked example: accuracy 0.560 against a 0.51 baseline, 5 points).

Test-period rows are scored only by `final_test.py`, after the freeze
(`DECISIONS` R23). Nothing here reads them by itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def majority_baseline(y_true: np.ndarray) -> float:
    """Accuracy of always predicting the more common class.

    Args:
        y_true: Binary labels.

    Returns:
        The share of the majority class, as a fraction.
    """
    values = np.asarray(y_true)
    if values.size == 0:
        return float("nan")
    ones = float((values == 1).sum())
    return max(ones, values.size - ones) / values.size


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, precision, recall, F1, ROC-AUC and the confusion matrix.

    Args:
        y_true: Actual binary labels.
        y_pred: Predicted binary labels (threshold 0.5 applied upstream).

    Returns:
        A dict with `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, the counts
        `tp`/`fp`/`fn`/`tn`, `majority_baseline` and `edge_over_baseline`.
    """
    actual = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if precision and recall and np.isfinite(precision) and np.isfinite(recall):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    baseline = majority_baseline(actual)
    edge = accuracy - baseline if np.isfinite(accuracy) else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float("nan"),  # filled by roc_auc_score when probabilities exist
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "majority_baseline": baseline,
        "edge_over_baseline": edge,
    }


def with_roc_auc(report: dict[str, Any], y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    """Add ROC-AUC to a report, when both classes are present."""
    from sklearn.metrics import roc_auc_score

    actual = np.asarray(y_true).astype(int)
    if len(np.unique(actual)) < 2:
        report["roc_auc"] = float("nan")
    else:
        report["roc_auc"] = float(roc_auc_score(actual, np.asarray(proba)))
    return report


def evaluate_rows(rows: pd.DataFrame) -> dict[str, Any]:
    """Evaluate a frame with `target_up`, `prob_up` and optional `signal` columns.

    Args:
        rows: Panel rows carrying the label and the probability.

    Returns:
        The classification report for the rows supplied.
    """
    actual = rows["target_up"].to_numpy()
    proba = rows["prob_up"].to_numpy()
    report = classification_report(actual, (proba > 0.5).astype(int))
    return with_roc_auc(report, actual, proba)


def classifier_metrics_row(
    rows: pd.DataFrame, model: str, period: str, year: int
) -> dict[str, Any]:
    """One `classifier_metrics.csv` line (`spec 04` section 7).

    The majority baseline is computed on **the same rows** as the accuracy. Comparing a
    validation accuracy against a baseline measured on a different period overstates
    the edge: the 2023 validation baseline is 0.5315 while the training-period one is
    0.5067, and the gap between them is 2.5 points of phantom edge (`DECISIONS` R36).

    Args:
        rows: Columns `target_up` and `prob_up`, already restricted to this period.
        model: Model name.
        period: `validation` or `test`.
        year: Calendar year the rows belong to.

    Returns:
        A dict with the spec's columns.
    """
    scored = rows.dropna(subset=["target_up", "prob_up"])
    report = evaluate_rows(scored)
    return {
        "model": model,
        "period": period,
        "year": int(year),
        "n": int(len(scored)),
        "accuracy": report["accuracy"],
        "majority_baseline": report["majority_baseline"],
        "edge_over_baseline": report["edge_over_baseline"],
        "precision": report["precision"],
        "recall": report["recall"],
        "f1": report["f1"],
        "roc_auc": report["roc_auc"],
        "tp": report["tp"],
        "fp": report["fp"],
        "fn": report["fn"],
        "tn": report["tn"],
    }


def write_classifier_metrics(
    path: str | Path, rows: pd.DataFrame, model: str, period: str, year: int
) -> dict[str, Any]:
    """Upsert one model's line into `classifier_metrics.csv` and return it.

    Args:
        path: CSV path.
        rows: Validation or test rows carrying `target_up` and `prob_up`.
        model: Model name.
        period: `validation` or `test`.
        year: Calendar year.

    Returns:
        The row that was written.
    """
    record = classifier_metrics_row(rows, model, period, year)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record])
    if target.exists():
        existing = pd.read_csv(target)
        # Upsert on the identity of the line, so re-running adds no rows (R28).
        keep = ~(
            (existing["model"] == record["model"])
            & (existing["period"] == record["period"])
            & (existing["year"] == record["year"])
        )
        frame = pd.concat([existing[keep], frame], ignore_index=True)
    frame.to_csv(target, index=False)
    return record
