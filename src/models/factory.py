"""Model factory (`spec 04` section 3).

Every model is a `Pipeline` beginning with `Winsorizer`, so the clip limits are refit
per walk-forward year rather than reused. Trees need no scaling; Logistic Regression
does. All estimators take `random_state` from `project.seed`.
"""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import Winsorizer

#: CLI aliases (`04` section 6.1).
ALIASES: dict[str, str] = {"lr": "logreg"}

#: Canonical model names, in the order the CLI reports them.
MODEL_NAMES: tuple[str, ...] = ("logreg", "tree", "rf", "xgb")


def resolve_name(name: str) -> str:
    """Map a CLI alias such as `lr` to its canonical name."""
    return ALIASES.get(name, name)


def _xgb_classifier(params: Mapping[str, Any], seed: int) -> Any:
    """XGBoost classifier, imported lazily because it is the heaviest dependency."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.03)),
        max_depth=int(params.get("max_depth", 3)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=seed,
        eval_metric="logloss",
    )


def make_model(name: str, cfg: Mapping[str, Any]) -> Pipeline:
    """Build one model's pipeline.

    Args:
        name: Model name or alias: `logreg`/`lr`, `tree`, `rf`, `xgb`.
        cfg: Loaded config; `models.<name>` holds the settings and `project.seed` the
            random state.

    Returns:
        A scikit-learn `Pipeline` starting with `Winsorizer`.

    Raises:
        ValueError: If the model name is unknown.
    """
    canonical = resolve_name(name)
    seed = int(cfg["project"]["seed"])
    params = dict(cfg["models"].get(canonical, {}))
    lower = float(cfg["features"].get("winsor_lower_pct", 1))
    upper = float(cfg["features"].get("winsor_upper_pct", 99))
    winsorizer = Winsorizer(lower, upper)

    if canonical == "logreg":
        estimator: Any = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            random_state=seed,
        )
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [("winsorizer", winsorizer), ("scaler", StandardScaler()), ("model", estimator)]
        )

    if canonical == "tree":
        estimator = DecisionTreeClassifier(
            max_depth=params.get("max_depth", 4),
            min_samples_leaf=int(params.get("min_samples_leaf", 200)),
            random_state=seed,
        )
    elif canonical == "rf":
        estimator = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=seed,
        )
    elif canonical == "xgb":
        estimator = _xgb_classifier(params, seed)
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES} or an alias")

    return Pipeline([("winsorizer", winsorizer), ("model", estimator)])
