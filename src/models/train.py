"""Walk-forward training (`spec 04` section 6.3).

Expanding-window yearly retraining: for each year from `first_test_year`, fit on every
row before `year_start - embargo_days` and predict that year. The embargo is in
**calendar** days because a label at the end of training looks one day forward.

Predictions for test years are stored but **not scored**: the freeze must come before
anyone sees test-period numbers (`DECISIONS` R23). Only validation-year accuracy is
logged as a trial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import pandas as pd

from src.common.config import ConfigError, load_config
from src.common.db import get_engine, init_db
from src.common.log import get_logger
from src.common.manifest import write_manifest
from src.common.trial_log import log_trial
from src.features.panel import NEXT_RETURN_COL, TARGET_COL, build_panel, get_feature_cols

from .evaluate import write_classifier_metrics
from .factory import MODEL_NAMES, make_model, resolve_name

#: Exit codes (`07` section 5).
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_DATA_FAILURE = 2


def walk_forward_predict(
    panel: pd.DataFrame,
    feature_cols: list[str],
    make_model: Callable[[], Any],
    first_test_year: int,
    embargo_days: int = 5,
    target_col: str = TARGET_COL,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict[int, Any]]:
    """Expanding-window yearly retraining.

    Args:
        panel: Panel indexed by trade date, containing `symbol`, the feature columns
            and a non-null `target_col`.
        feature_cols: Active feature names.
        make_model: Zero-argument factory returning a fresh unfitted estimator.
        first_test_year: First year to predict, normally the validation year.
        embargo_days: Calendar-day embargo between training and prediction.
        target_col: Label column.
        progress: Optional callback for status lines.

    Returns:
        `(predictions, fitted)` where `predictions` has `symbol`, `prob_up`,
        `model_year` and the training cut-off, and `fitted` maps year to
        `(model, train_max_date)`.
    """
    outputs: list[pd.DataFrame] = []
    fitted: dict[int, Any] = {}
    max_year = int(panel.index.year.max())

    for year in range(int(first_test_year), max_year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        cut_off = year_start - pd.Timedelta(days=embargo_days)
        train = panel[panel.index < cut_off]
        test = panel[panel.index.year == year]
        if train.empty or test.empty:
            if progress:
                progress(f"year {year}: skipped (no train or no test rows)")
            continue

        model = make_model()
        model.fit(train[list(feature_cols)], train[target_col])
        proba = model.predict_proba(test[list(feature_cols)])[:, 1]
        block = test[["symbol", "date_key", target_col]].copy()
        block["prob_up"] = proba
        block["model_year"] = year
        block["train_max"] = train.index.max()
        block["correct"] = block[target_col].to_numpy() == (proba > 0.5)
        # Keep the trade date on the row so predictions join back to the panel
        # unambiguously: joining on `symbol` alone would fan out into a cartesian
        # product across every date that stock appears.
        block["trade_date"] = pd.Series(test.index.to_numpy(), index=test.index).to_numpy()
        outputs.append(block.reset_index(drop=True))
        fitted[year] = (model, train.index.max())
        if progress:
            progress(f"year {year}: trained on {len(train)} rows, predicted {len(test)}")

    if not outputs:
        return pd.DataFrame(
            columns=[
                "symbol",
                "date_key",
                target_col,
                "prob_up",
                "model_year",
                "train_max",
                "correct",
                "trade_date",
            ]
        ), fitted
    return pd.concat(outputs, ignore_index=True), fitted


def _upsert_dim_model(conn: Any, name: str, version: str, trained_through: str, params: Any) -> int:
    """One `dim_model` row per (model name, year)."""
    import json

    from sqlalchemy import text

    result = conn.execute(
        text(
            "INSERT INTO dim_model (model_name, version, trained_through, params_json) "
            "VALUES (:name, :version, :trained_through, :params) "
            "ON CONFLICT (model_name, version) DO UPDATE SET "
            "  trained_through = excluded.trained_through, "
            "  params_json = excluded.params_json"
        ),
        {
            "name": name,
            "version": version,
            "trained_through": trained_through,
            "params": json.dumps(params, sort_keys=True, default=str),
        },
    )
    return int(result.lastrowid or 0)


def _model_key_map(engine: Any) -> dict[tuple[str, str], int]:
    from src.common.db import read_sql

    df = read_sql(engine, "SELECT model_key, model_name, version FROM dim_model")
    return {(r["model_name"], r["version"]): int(r["model_key"]) for _, r in df.iterrows()}


def _stock_key_map(engine: Any) -> dict[str, int]:
    from src.common.db import read_sql

    df = read_sql(engine, "SELECT symbol, stock_key FROM dim_stock")
    return {r["symbol"]: int(r["stock_key"]) for _, r in df.iterrows()}


def _regime_key_by_date(engine: Any) -> dict[int, int]:
    from src.common.db import read_sql

    df = read_sql(engine, "SELECT date_key, regime_key FROM fact_market_regime")
    return {int(r["date_key"]): int(r["regime_key"]) for _, r in df.iterrows()}


def write_signals(
    engine: Any,
    model_name: str,
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    model_keys: Mapping[tuple[str, str], int],
    stock_keys: Mapping[str, int],
    regime_keys: Mapping[int, int],
) -> int:
    """Upsert `fact_signal.prob_up` for every predicted row.

    `signal` and `target_weight` are never touched: only the primary strategy run writes
    them, so a retrain must not reset a live decision (`DECISIONS` R3).
    """

    if predictions.empty:
        return 0

    # Keyed on (date_key, symbol): both are plain columns on the panel, so the
    # lookup does not depend on the index being named anything in particular.
    next_return = {
        (int(r["date_key"]), r["symbol"]): r[NEXT_RETURN_COL]
        for r in panel.reset_index().to_dict("records")
    }

    rows = []
    for record in predictions.to_dict("records"):
        version = f"wf{int(record['model_year'])}"
        model_key = model_keys.get((model_name, version))
        stock_key = stock_keys.get(record["symbol"])
        date_key = int(record["date_key"])
        regime_key = regime_keys.get(date_key)
        if model_key is None or stock_key is None or regime_key is None:
            continue
        rows.append(
            {
                "date_key": date_key,
                "stock_key": stock_key,
                "model_key": model_key,
                "regime_key": regime_key,
                "prob_up": float(record["prob_up"]),
                "next_day_return": _num(next_return.get((date_key, record["symbol"]))),
            }
        )

    upsert = """
        INSERT INTO fact_signal (
            date_key, stock_key, model_key, regime_key, prob_up, signal, target_weight,
            next_day_return
        )
        VALUES (:date_key, :stock_key, :model_key, :regime_key, :prob_up, 0, 0.0,
                :next_day_return)
        ON CONFLICT (date_key, stock_key, model_key) DO UPDATE SET
            prob_up = excluded.prob_up,
            regime_key = excluded.regime_key,
            next_day_return = COALESCE(excluded.next_day_return, fact_signal.next_day_return)
    """
    with engine.begin() as conn:
        from src.common.db import execute_many

        return execute_many(conn, upsert, rows)


def _num(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return float(value)


def run(
    cfg: dict[str, Any],
    *,
    models: list[str] | None,
    first_test_year: int | None,
    single_year: int | None,
    evaluate: bool,
) -> int:
    """Train every requested model and persist signals, metrics and models."""
    log = get_logger("src.models.train", cfg)
    engine = get_engine(cfg)
    init_db(engine)

    panel = build_panel(engine, cfg)
    if panel.empty:
        log.error("panel is empty; run `make features` first")
        return EXIT_DATA_FAILURE

    from src.features.panel import use_vix_in_regime

    feature_cols = get_feature_cols(use_vix_in_regime(cfg, engine))
    panel = panel.dropna(subset=[TARGET_COL])

    if single_year is not None:
        years = [int(single_year)]
    else:
        start = first_test_year or int(str(cfg["splits"]["validation_start"])[:4])
        years = list(range(int(start), int(panel.index.year.max()) + 1))

    requested = [resolve_name(m) for m in (models or MODEL_NAMES)]
    validation_year = int(str(cfg["splits"]["validation_start"])[:4])
    embargo = int(cfg["splits"]["embargo_days"])
    log.info("panel has %d rows over %d features", len(panel), len(feature_cols))

    total_written = 0
    with engine.begin() as conn:
        for name in requested:
            params = dict(cfg["models"].get(name, {}))
            for year in years:
                _upsert_dim_model(conn, name, f"wf{year}", str(cfg["splits"]["train_end"]), params)

    model_keys = _model_key_map(engine)
    stock_keys = _stock_key_map(engine)
    regime_keys = _regime_key_by_date(engine)

    for name in requested:
        log.info("training %s", name)
        predictions, fitted = walk_forward_predict(
            panel,
            feature_cols,
            lambda n=name: make_model(n, cfg),
            years[0],
            embargo_days=embargo,
            progress=log.info,
        )
        if predictions.empty:
            log.warning("%s produced no predictions", name)
            continue

        # The embargo assertion, checked in code and not only in tests.
        for year, (_model, train_max) in fitted.items():
            year_start = pd.Timestamp(year=int(year), month=1, day=1)
            assert train_max < year_start - pd.Timedelta(days=embargo), (
                f"{name} wf{year}: training data reaches {train_max}, "
                f"which is not before {year_start - pd.Timedelta(days=embargo)}"
            )

        joined = predictions
        written = write_signals(engine, name, joined, panel, model_keys, stock_keys, regime_keys)
        total_written += written
        log.info("%s: wrote %d signal row(s)", name, written)

        models_dir = Path(cfg["paths"]["models"])
        models_dir.mkdir(parents=True, exist_ok=True)
        for year, (model, _train_max) in fitted.items():
            joblib.dump(model, models_dir / f"{name}_wf{year}.joblib")

        # R23: only validation-year accuracy is logged.
        if evaluate and validation_year in fitted:
            mask = joined["model_year"] == validation_year
            scored = joined.loc[mask]
            accuracy = float(scored["correct"].mean()) if len(scored) else float("nan")

            # `04` section 7: the classifier comparison, with the majority baseline
            # measured on the same rows as the accuracy (DECISIONS R36).
            record = write_classifier_metrics(
                Path(cfg["paths"]["results"]) / "classifier_metrics.csv",
                scored,
                name,
                "validation",
                validation_year,
            )

            log_trial(
                cfg,
                phase="train",
                kind="model_year",
                run_id=f"wf{validation_year}",
                model=name,
                params={"year": validation_year, "features": feature_cols},
                period="validation",
                metric="accuracy",
                value=accuracy,
                extra={
                    "rows": int(mask.sum()),
                    "majority_baseline": record["majority_baseline"],
                    "edge_over_baseline": record["edge_over_baseline"],
                    "roc_auc": record["roc_auc"],
                },
            )
            log.info(
                "%s validation accuracy %.4f vs majority baseline %.4f "
                "(edge %+.4f, roc_auc %.4f, n=%d)",
                name,
                record["accuracy"],
                record["majority_baseline"],
                record["edge_over_baseline"],
                record["roc_auc"],
                record["n"],
            )

    write_manifest(
        cfg,
        "train",
        engine,
        extra={"models": requested, "years": years, "rows": total_written},
    )
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train",
        description="Walk-forward training of the direction classifiers.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=f"Comma-separated subset of {','.join(MODEL_NAMES)}. Default all.",
    )
    parser.add_argument("--first-test-year", type=int, default=None, help="First year to predict.")
    parser.add_argument("--year", type=int, default=None, help="Train a single year only.")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not score or log trials.")
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

    names = [m.strip() for m in args.models.split(",")] if args.models else None
    try:
        return run(
            cfg,
            models=names,
            first_test_year=args.first_test_year,
            single_year=args.year,
            evaluate=not args.no_evaluate,
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001 - the CLI must not traceback at the user
        get_logger("src.models.train", cfg).exception("training failed")
        print(f"unexpected error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
