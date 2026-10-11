"""Preprocessing for the classifiers (`spec 04` section 3).

`Winsorizer` clips each numeric feature at the 1st and 99th percentile **computed on the
training rows only** (`TESTS` section 5: the limits must equal the fit data's
percentiles and must not move when other data is transformed).

One-hot regime columns are passed through untouched: clipping a 0/1 column would turn
it into a constant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#: Columns that are already 0/1 indicators and must not be clipped.
ONE_HOT_PREFIX = "regime_"


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to per-column training percentiles.

    Attributes:
        lower_pct: Lower percentile, 1 by default.
        upper_pct: Upper percentile, 99 by default.
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0) -> None:
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, x: Any, y: Any = None) -> "Winsorizer":
        """Store the 1st and 99th percentile of each clippable column.

        Args:
            x: Feature frame or array.
            y: Ignored, present for sklearn compatibility.

        Returns:
            The fitted estimator.
        """
        frame = _as_frame(x)
        self.columns_ = list(frame.columns)
        self.one_hot_ = [c for c in frame.columns if str(c).startswith(ONE_HOT_PREFIX)]
        self.limits_: dict[str, tuple[float, float]] = {}
        for column in self.columns_:
            if column in self.one_hot_:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                self.limits_[column] = (np.nan, np.nan)
                continue
            self.limits_[column] = (
                float(np.percentile(values, self.lower_pct)),
                float(np.percentile(values, self.upper_pct)),
            )
        return self

    def transform(self, x: Any) -> Any:
        """Clip to the stored limits. Uses the limits from `fit`, never recomputes them."""
        frame = _as_frame(x).copy()
        for column, (low, high) in self.limits_.items():
            if column not in frame.columns:
                continue
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if pd.isna(low) or pd.isna(high):
                continue
            frame[column] = numeric.clip(low, high)
        if hasattr(x, "columns"):
            return frame
        return frame.to_numpy()

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Column names after transform, for sklearn pipeline parity."""
        return np.asarray(self.columns_, dtype=object)


def _as_frame(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(np.asarray(x))
