# 04. Data Mining: Features, Regimes and Classifiers

Reference code blocks are **starting points** that have been checked for the stated behaviour; the tests in `TESTS.md` decide correctness.

---

## 1. Mining tasks

| Task | Technique | Type | Output |
|---|---|---|---|
| Market regime detection | K-Means | Unsupervised | Regime label per trading day |
| Next-day direction | Logistic Regression, Decision Tree, Random Forest, XGBoost | Supervised | `prob_up` per stock per day |
| Sector co-movement (stretch) | Apriori | Unsupervised | Rules such as Banking up implies IT up |

---

## 2. Features

### 2.1 Feature catalogue and warehouse mapping

All features for day *t* use only data up to the close of *t*. Columns come from `fact_indicator` (FI) and `fact_price` (FP). "Index" means the `^NSEI` row for the same date.

| # | Feature (name in `FEATURE_COLS_ALL`) | Formula | Source |
|---|---|---|---|
| 1 | `ret_1d` | FI `return_1d` | FI |
| 2 | `ret_5d` | FI `return_5d` | FI |
| 3 | `ret_20d` | FI `return_20d` | FI |
| 4 | `px_to_sma20` | FP `close_price` / FI `sma_20` - 1 | FP, FI |
| 5 | `px_to_sma50` | FP `close_price` / FI `sma_50` - 1 | FP, FI |
| 6 | `sma20_to_sma50` | FI `sma_20` / FI `sma_50` - 1 | FI |
| 7 | `rsi_14` | FI `rsi_14` | FI |
| 8 | `macd_hist_pct` | (FI `macd` - FI `macd_signal`) / FP `close_price` | FI, FP |
| 9 | `bb_pct_b` | FI `bb_pct_b` | FI |
| 10 | `atr_pct` | FI `atr_14` / FP `close_price` | FI, FP |
| 11 | `vol_20d` | FI `vol_20d` | FI |
| 12 | `volume_ratio_20` | FI `volume_ratio_20` | FI |
| 13 | `rel_strength_20` | stock `return_20d` - index `return_20d` | FI |
| 14 | `mkt_ret_20d` | index `return_20d` | FI |
| 15 | `india_vix` | VIX close for the date (omit if VIX is not used, `DECISIONS` R12) | FP |
| 16-19 | `regime_bull`, `regime_sideways`, `regime_high_vol`, `regime_bear` | One-hot of the **smoothed** regime label; always four columns even if k = 3 | `fact_market_regime` |

`FEATURE_COLS_ALL` is a constant tuple in `src/features/panel.py` holding these 19 names in exactly this order; `get_feature_cols(use_vix)` returns it, minus `india_vix` when VIX is not used (18 names). Everything downstream uses the result of `get_feature_cols`.

### 2.2 `fact_indicator` definitions (`compute_indicators`)

`compute_indicators(prices: DataFrame) -> DataFrame` takes one instrument's daily frame (index = trade date; columns `open, high, low, close, volume`) and returns the `fact_indicator` columns:

| Column | Definition |
|---|---|
| `return_1d`, `return_5d`, `return_20d` | `close.pct_change(n)` |
| `sma_20`, `sma_50` | `close.rolling(n).mean()` |
| `ema_12`, `ema_26` | `close.ewm(span=n, adjust=False).mean()` |
| `macd`, `macd_signal` | `ema_12 - ema_26`; `macd.ewm(span=9, adjust=False).mean()` |
| `rsi_14` | `rsi(close, 14)` below |
| `bb_pct_b` | `bollinger_pct_b(close, 20, 2.0)` below |
| `atr_14` | `atr(high, low, close, 14)` below |
| `vol_20d` | `return_1d.rolling(20).std() * sqrt(252)` (decimal) |
| `volume_ratio_20` | `volume / volume.rolling(20).mean()`; NaN when the average is 0 (index, VIX) |

Warm-up values stay NaN and are stored as NULL. Indicators are computed for `EQUITY` and `INDEX` instruments only (`DECISIONS` R9), always from the full history.

### 2.3 Indicator reference code

```python
import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI. All-gain windows give 100, flat windows give 50."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.mask((loss == 0) & (gain > 0), 100.0)
    out = out.mask((loss == 0) & (gain == 0), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    return line, line.ewm(span=signal, adjust=False).mean()


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    lower, upper = mid - n_std * std, mid + n_std * std
    width = (upper - lower).replace(0, np.nan)
    return (close - lower) / width


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def annualized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)
```

`shift(1)` and past-looking rolling and EWM windows only use the past, so they are leak-free. The truncation test in `TESTS.md` section 4 proves it.

---

## 3. Preprocessing

| Step | Rule |
|---|---|
| Missing values | Drop each stock's first `features.warmup_rows` (50) rows after its first bar; forward-fill at most 2 days (in ETL). **Never back-fill.** Drop any panel row that still has NaN in the active feature columns (`get_feature_cols`) from training and evaluation (keep the last live row for prediction). |
| Outliers | `Winsorizer`: clip each numeric feature at the 1st and 99th percentile **computed on the training rows only**; part of the model pipeline so it is refit per walk-forward year. Do not clip one-hot columns. |
| Scaling | `StandardScaler` for Logistic Regression and for K-Means; fit on training rows only. Trees need none. |
| Encoding | One-hot the regime label (four 0/1 columns). |
| Discretisation | Weekly sector returns to "up" or "down" for Apriori (stretch). |

Pipelines (`src/models/factory.py::make_model`):

- `logreg`: `Pipeline([Winsorizer, StandardScaler, LogisticRegression])`
- `tree`, `rf`, `xgb`: `Pipeline([Winsorizer, estimator])`

All estimators use `random_state = project.seed`.

---

## 4. Target

```text
next_day_return_t = close_(t+1) / close_t - 1          (per stock, next trading bar of that stock)
target_up_t       = 1 if next_day_return_t > 0 else 0   (absolute target, default)
target_up_t       = 1 if next_day_return_t - index_next_day_return_t > 0   (relative target, stretch V3)
```

- This is the **only** place where `shift(-1)` is allowed (mark the line `# label-only`).
- The last row of each stock has no label. Drop it from training and evaluation; keep it for live prediction.
- Roughly 51 to 54% of days are up; no resampling is needed.
- `next_day_return` is also stored in the panel and used to fill `fact_signal.next_day_return`.

---

## 5. Market-regime detection (K-Means)

### 5.1 Market features

From the index close (`fact_price`) and VIX close, per trading day:

| Feature | Definition |
|---|---|
| `mkt_return_20d` | index `close.pct_change(20)` (equals the index's `fact_indicator.return_20d`) |
| `mkt_vol_20d` | index `return_1d.rolling(20).std() * sqrt(252)` |
| `india_vix` | VIX close, forward-filled at most 2 days; omitted when `regime.use_vix` resolves to false |
| `mkt_drawdown` | index close / `close.rolling(252, min_periods=20).max()` - 1 |

`use_vix: auto`: use VIX if its coverage on the training period (non-null share of trading days) is at least `regime.min_vix_coverage`; log the decision and the coverage.

```python
def build_market_features(index_close: pd.Series, vix_close: pd.Series | None,
                          dd_window: int = 252, dd_min_periods: int = 20) -> pd.DataFrame:
    out = pd.DataFrame(index=index_close.index)
    out["mkt_return_20d"] = index_close.pct_change(20)
    out["mkt_vol_20d"] = index_close.pct_change().rolling(20).std() * np.sqrt(252)
    if vix_close is not None:
        out["india_vix"] = vix_close.reindex(out.index).ffill(limit=2)
    out["mkt_drawdown"] = index_close / index_close.rolling(dd_window, min_periods=dd_min_periods).max() - 1
    return out
```

### 5.2 Choosing k

1. Fit K-Means for each k in `regime.k_range` on the **training period only** (dates `<= splits.train_end`).
2. Save inertia and silhouette per k to `reports/results/regime_k_selection.csv`; draw the elbow and silhouette figure; log each k as a trial.
3. Prefer `regime.k` = 4 if the clusters are interpretable; use 3 if k = 4 splits a regime into meaningless fragments (for example a cluster with under 3% of training-period days or two clusters with near-identical profiles). Record the decision and the reason in `PROGRESS.md` (Q1).

### 5.3 Fit, label, smooth, predict

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def choose_k(train: pd.DataFrame, features: list, k_range, seed: int) -> pd.DataFrame:
    X = StandardScaler().fit_transform(train[features])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def label_clusters(train: pd.DataFrame, labels) -> dict:
    """Name clusters from their average behaviour on the training data.
    Highest mean 20-day return = BULL, lowest = BEAR; of the rest, the highest volatility = HIGH_VOL, others SIDEWAYS."""
    stats = train.assign(cluster=labels).groupby("cluster")[["mkt_return_20d", "mkt_vol_20d"]].mean()
    names = {stats["mkt_return_20d"].idxmax(): "BULL", stats["mkt_return_20d"].idxmin(): "BEAR"}
    rest = stats.drop(index=list(names))
    if len(rest) == 1:
        names[rest.index[0]] = "SIDEWAYS"
    elif len(rest) > 1:
        high_vol = rest["mkt_vol_20d"].idxmax()
        for cluster in rest.index:
            names[cluster] = "HIGH_VOL" if cluster == high_vol else "SIDEWAYS"
    return names


def fit_regime_model(train: pd.DataFrame, features: list, k: int, seed: int):
    scaler = StandardScaler().fit(train[features])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaler.transform(train[features]))
    return scaler, km, label_clusters(train, km.labels_)


def predict_clusters(df: pd.DataFrame, features: list, scaler, km) -> pd.Series:
    return pd.Series(km.predict(scaler.transform(df[features])), index=df.index)


def smooth_trailing_mode(labels: pd.Series, window: int = 5) -> pd.Series:
    """Trailing mode of the last `window` labels (past only). Ties go to the most recent tied label."""
    vals = labels.to_numpy()
    out = []
    for i in range(len(vals)):
        w = vals[max(0, i - window + 1): i + 1]
        counts = pd.Series(w).value_counts()
        tied = set(counts[counts == counts.max()].index)
        out.append(next(lab for lab in w[::-1] if lab in tied))
    return pd.Series(out, index=labels.index)


def nearest_centre(z, centres) -> tuple:
    """Index of the nearest centre and all Euclidean distances (used by tests and the worked example)."""
    d = np.sqrt(((np.asarray(centres) - np.asarray(z)) ** 2).sum(axis=1))
    return int(d.argmin()), d
```

Pipeline:

1. Drop rows with NaN market features.
2. Fit scaler and K-Means on rows with date `<= splits.train_end`; name clusters; keep `regime.k` clusters.
3. Predict raw cluster ids for **all** dates with the frozen scaler and centres (`regime.refit: none`).
4. Map ids to names; smooth with `smooth_trailing_mode` (`regime.smoothing.window`) on the **names**; or the `consecutive` method (switch only after `smoothing_consecutive_days` consecutive days in a new label).
5. Write `fact_market_regime`: `cluster_id` = raw id, `regime_key` = smoothed label's key, plus the four feature columns (`india_vix` NULL when unused).
6. Save `models/regime_model.joblib` containing scaler, KMeans, name map, feature list, `train_end`, k, seed, VIX decision.

**Leakage rule:** fitting K-Means on all years lets future market behaviour shape today's regime definition. Fit on the training period only.

### 5.4 Sanity check (do not force labels)

| Regime | 20-day return | Volatility | VIX | Drawdown |
|---|---|---|---|---|
| BULL | Positive | Low to moderate | Low | Near zero |
| SIDEWAYS | Near zero | Low | Low to moderate | Mild |
| HIGH_VOL | Mixed | High | High | Moderate |
| BEAR | Negative | High | High | Deep |

`regime_model.py` writes `regime_profile.csv` (mean of each market feature per regime, **training period**, computed in pandas) and `regime_counts.csv` (columns `regime, days_train, days_all`). `label_clusters` names BULL and BEAR by return, so checking those two would be circular. Instead check, on the training period: BULL mean 20-day return is above 0 and above SIDEWAYS (when present); mean volatility of BEAR and HIGH_VOL (when present) is above that of BULL and SIDEWAYS; BEAR mean drawdown is below BULL's. A failure logs a WARNING and is recorded in `PROGRESS.md`; tests assert all of them on the synthetic market. If the profile is implausible, change k or the features; never rename clusters by hand. OLAP query 08 gives the all-dates version for the report.

---

## 6. Direction classifier

### 6.1 Models

| Name | Settings (from `models.*` in config) | Role |
|---|---|---|
| `logreg` | `C=1.0` | Interpretable baseline |
| `tree` | `max_depth=4`, `min_samples_leaf=200` | Readable rules |
| `rf` | `n_estimators=300`, `max_depth=6`, `min_samples_leaf=50` | Robust ensemble |
| `xgb` | `n_estimators=300`, `learning_rate=0.03`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8` | Strongest candidate |

CLI alias `lr` = `logreg`. These are starting settings; any extra setting you try counts as a trial (`07` section 4).

**One pooled model** per year across all stocks (about 12,500 rows per year; about 100,000 rows for 2015 to 2022).

### 6.2 Splits

| Block | Dates (config) | Use |
|---|---|---|
| Train | start to `splits.train_end` | Fit models, winsorizer, scaler, regime model |
| Validation | `validation_start` to `validation_end` | Choose model and strategy settings |
| Test | from `test_start` | Final results, once (`06` section 8) |

### 6.3 Walk-forward procedure

For each year from `first_test_year` (the validation year) to the latest year: train on all data before `year_start - embargo_days`, predict that year, move on. The embargo (`embargo_days` **calendar** days) exists because a label at the end of training looks one day into the future.

```python
def walk_forward_predict(panel, feature_cols, make_model, first_test_year, embargo_days=5,
                         target_col="target_up", progress=None):
    """Expanding-window yearly retraining; returns out-of-sample probabilities and the fitted models.
    panel: DatetimeIndex (trade date), sorted; columns include feature_cols, target_col (no NaN) and 'symbol'."""
    outputs, fitted = [], {}
    for year in range(first_test_year, panel.index.year.max() + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        train = panel[panel.index < year_start - pd.Timedelta(days=embargo_days)]
        test = panel[panel.index.year == year]
        if train.empty or test.empty:
            continue
        model = make_model()
        model.fit(train[feature_cols], train[target_col])
        proba = model.predict_proba(test[feature_cols])[:, 1]
        outputs.append(test[["symbol"]].assign(prob_up=proba, model_year=year))
        fitted[year] = (model, train.index.max())
    return pd.concat(outputs), fitted
```

Storage (`src/models/train.py`):

- One `dim_model` row per (model name, year): `version = f"wf{year}"`, `trained_through = train.index.max()` (ISO date), `params_json` = model settings.
- Save `models/<name>_wf<year>.joblib`.
- Upsert `fact_signal` rows (`date_key`, `stock_key`, `model_key`, `regime_key` of that date, `prob_up`, `signal = 0`, `target_weight = 0.0`). Never overwrite `signal` / `target_weight` of rows already set by a primary run (use `ON CONFLICT ... DO UPDATE SET prob_up = excluded.prob_up, regime_key = excluded.regime_key` only).
- After writing, fill `fact_signal.next_day_return` for dates whose next trading bar exists.
- Log one trial per model for the **validation year only**, with its out-of-sample accuracy (test-year predictions are stored but not scored, `DECISIONS` R23).

For the first validation year there is no earlier test; predictions for validation and test years both come from this one walk-forward run. The validation year's predictions are used for tuning, the test years' for the final test.

### 6.4 Interpretation

- Permutation importance on the validation year (works for every model; `final_test.py` repeats it on the test years); the tree's built-in importance; `export_text` of the decision tree saved to `reports/results/tree_rules.txt`.
- Optional: `CalibratedClassifierCV` so 0.60 means about 60% (counts as a trial).

---

## 7. Classification evaluation

Per model and year (and overall) out-of-sample:

| Metric | Definition |
|---|---|
| Accuracy | (TP + TN) / total |
| Majority baseline | max(share up, share down) in that sample. **Always reported next to accuracy** |
| Precision, Recall, F1 | Positive class = up |
| ROC-AUC | 0.5 means no skill |
| Confusion matrix | Counts at threshold 0.5 |

**Test-period discipline (`DECISIONS` R23):** `train.py` writes predictions for every year (the daily job and the final test need them) but its evaluation step scores **validation-year rows only** (`--period validation`, the default). Test-year metrics are appended to `classifier_metrics.csv` only by `final_test.py`, after the freeze. Figures made during development (`prob_up_histogram.png`, confusion matrices, ROC) also use validation rows only until `final_test.py` regenerates them.

Write `reports/results/classifier_metrics.csv` (columns: `model, period, year, n, accuracy, majority_baseline, precision, recall, f1, roc_auc, tp, fp, fn, tn`) and the figures `confusion_matrix_<model>.png`, `roc_<model>.png`, `feature_importance_<model>.png`, `prob_up_histogram.png`. Realistic daily-direction accuracy is 51 to 56%; do not tune to inflate it.

---

## 8. Stretch: association rules (Apriori)

Weekly transactions of the sectors whose weekly return was positive; `mlxtend.frequent_patterns.apriori` with `min_support` about 0.3 and `association_rules` with confidence about 0.6; mine on the **training period only**. Output `reports/results/association_rules.csv` (antecedent, consequent, support, confidence, lift). Definitions: support = P(A and B); confidence = P(A and B) / P(A); lift = confidence / P(B).

---

## 9. Forbidden patterns (a test greps for the first four)

| Pattern | Why | Do this |
|---|---|---|
| `shift(-` outside the target line | Uses tomorrow | Past-only features |
| `.bfill(` or `fillna(method="bfill")` | Copies the future backward | Forward-fill at most 2 days |
| `train_test_split(` / `shuffle=True` / `KFold(shuffle` | Mixes future into training | Chronological walk-forward |
| `fit` on scaler, winsorizer or K-Means with non-training rows | Leakage | Fit on training rows only |
| Reporting accuracy alone | Hides the baseline | Report the majority baseline and trading metrics |
| Tuning on the test period | Inflates results | Tune on validation; test once |
| Reporting only the best model | Selection bias | Report every variant tried (trial log) |
