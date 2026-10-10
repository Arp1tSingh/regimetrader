"""Data-quality checks DQ-01 to DQ-07 (`spec 03` section 5).

Hard failures raise `DataQualityError`, which the pipeline turns into exit code 4.
Soft failures log a WARNING and are reported in the run summary; they never stop a load.

Each check is a pure query against the warehouse so it can be run on its own after any
load, not only inside the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

import pandas as pd
from sqlalchemy import Engine

from src.common.dates import from_date_key
from src.common.db import read_sql
from src.common.log import get_logger

#: Severity by check id, exactly as `03` section 5 documents.
SEVERITY: dict[str, str] = {
    "DQ-01": "HARD",
    "DQ-02": "HARD",
    "DQ-03": "HARD",
    "DQ-04": "SOFT",
    "DQ-05": "SOFT",
    "DQ-06": "SOFT",
    "DQ-07": "SOFT",
}

#: Relative tolerance for the high/low containment check (`03` section 5).
RANGE_TOL = 1e-6


class DataQualityError(RuntimeError):
    """Raised when a hard data-quality check fails. Pipeline exit code 4."""


@dataclass
class CheckResult:
    """Outcome of one data-quality check."""

    id: str
    severity: str
    passed: bool
    detail: str

    def as_row(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "passed": "PASS" if self.passed else "FAIL",
            "detail": self.detail,
        }


def _ok(check_id: str, detail: str) -> CheckResult:
    return CheckResult(check_id, SEVERITY[check_id], True, detail)


def _fail(check_id: str, detail: str) -> CheckResult:
    return CheckResult(check_id, SEVERITY[check_id], False, detail)


# ---------------------------------------------------------------- hard checks


def check_duplicates(engine: Engine) -> CheckResult:
    """DQ-01 (hard): no duplicate (date_key, stock_key) in fact_price."""
    df = read_sql(
        engine,
        "SELECT date_key, stock_key, COUNT(*) AS n FROM fact_price "
        "GROUP BY date_key, stock_key HAVING COUNT(*) > 1 LIMIT 5",
    )
    if df.empty:
        return _ok("DQ-01", "no duplicate (date_key, stock_key) rows")
    return _fail(
        "DQ-01",
        f"{len(df)} duplicate (date_key, stock_key) group(s), e.g. "
        f"{df.iloc[0]['date_key']}/{df.iloc[0]['stock_key']}",
    )


def check_price_validity(engine: Engine) -> CheckResult:
    """DQ-02 (hard): positive prices, high covers all, low covers all."""
    df = read_sql(
        engine,
        """
        SELECT
            SUM(CASE WHEN close_price IS NULL OR close_price <= 0
                      OR open_price <= 0 OR high_price <= 0 OR low_price <= 0
                     THEN 1 ELSE 0 END) AS bad_price,
            SUM(CASE WHEN high_price < low_price THEN 1 ELSE 0 END) AS inverted,
            SUM(CASE WHEN high_price < MAX(open_price, close_price) * (1 - :tol)
                       OR low_price  > MIN(open_price, close_price) * (1 + :tol)
                     THEN 1 ELSE 0 END) AS out_of_range
        FROM fact_price
        """,
        {"tol": RANGE_TOL},
    )
    bad = int(df["bad_price"].iloc[0] or 0)
    inverted = int(df["inverted"].iloc[0] or 0)
    out_of_range = int(df["out_of_range"].iloc[0] or 0)
    total = bad + inverted + out_of_range
    if total == 0:
        return _ok("DQ-02", "all prices positive and within high/low bounds")
    return _fail(
        "DQ-02",
        f"{total} invalid row(s): {bad} non-positive, {inverted} high<low, "
        f"{out_of_range} outside high/low range",
    )


def check_date_references(engine: Engine) -> CheckResult:
    """DQ-03 (hard): every price date_key is a trading day present in dim_date."""
    df = read_sql(
        engine,
        """
        SELECT COUNT(*) AS n
        FROM fact_price f
        LEFT JOIN dim_date d ON f.date_key = d.date_key
        WHERE d.date_key IS NULL OR d.is_trading_day <> 1
        """,
    )
    bad = int(df["n"].iloc[0] or 0)
    if bad == 0:
        return _ok("DQ-03", "every price row maps to a trading day in dim_date")
    return _fail("DQ-03", f"{bad} row(s) reference a missing or non-trading date")


# ---------------------------------------------------------------- soft checks


def check_row_coverage(engine: Engine, cfg: Mapping[str, Any]) -> CheckResult:
    """DQ-04 (soft): row count per stock within tolerance of the index's, counted
    from that stock's own first bar (DECISIONS R10, short-history stocks are allowed).
    """
    tolerance = float(cfg["quality"]["row_count_tolerance"])
    empty = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")
    if int(empty["n"].iloc[0] or 0) == 0:
        return _fail("DQ-04", "fact_price is empty")

    # Each stock is compared against the index's trading-day count measured from that
    # stock's own first bar, so a recently listed name is not penalised (R10).
    coverage = read_sql(
        engine,
        """
        WITH idx AS (
            SELECT MAX(date_key) AS hi FROM fact_price
            WHERE stock_key = (SELECT stock_key FROM dim_stock WHERE symbol = :index_symbol)
        ),
        per_stock AS (
            SELECT s.symbol, COUNT(DISTINCT f.date_key) AS n, MIN(f.date_key) AS lo
            FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key
            GROUP BY s.symbol
        )
        SELECT p.symbol, p.n AS stock_rows,
               (SELECT COUNT(DISTINCT f2.date_key)
                FROM fact_price f2, idx
                WHERE f2.stock_key = (SELECT stock_key FROM dim_stock
                                      WHERE symbol = :index_symbol)
                  AND f2.date_key BETWEEN p.lo AND idx.hi) AS index_rows
        FROM per_stock p
        """,
        {"index_symbol": cfg["data"]["index_symbol"]},
    )
    coverage["share"] = 1.0 - (coverage["stock_rows"] / coverage["index_rows"].replace(0, pd.NA))
    offenders = coverage[coverage["share"] > tolerance]

    under_10y = _count_under_10_years(engine)
    detail = (
        f"{len(coverage)} instrument(s) compared; {len(offenders)} beyond "
        f"{tolerance:.1%} tolerance; {under_10y} with under 10 years of history"
    )
    if offenders.empty:
        return _ok("DQ-04", detail)
    return _fail("DQ-04", detail + f"; worst: {offenders.iloc[0]['symbol']}")


def _count_under_10_years(engine: Engine) -> int:
    """Stocks whose loaded history is shorter than ten years (DECISIONS R10)."""
    df = read_sql(
        engine,
        """
        WITH idx AS (SELECT MAX(date_key) AS last_key FROM fact_price)
        SELECT COUNT(*) AS n FROM (
            SELECT s.symbol, MIN(f.date_key) AS first_key
            FROM fact_price f JOIN dim_stock s ON s.stock_key = f.stock_key, idx
            WHERE s.instrument_type = 'EQUITY'
            GROUP BY s.symbol
            HAVING idx.last_key - first_key < 1000
        )
        """,
    )
    return int(df["n"].iloc[0] or 0)


def check_flagged_share(engine: Engine, cfg: Mapping[str, Any]) -> CheckResult:
    """DQ-05 (soft): flagged rows below `quality.max_flagged_share`."""
    limit = float(cfg["quality"]["max_flagged_share"])
    df = read_sql(
        engine,
        "SELECT COUNT(*) AS total, SUM(CASE WHEN is_flagged = 1 THEN 1 ELSE 0 END) AS flagged "
        "FROM fact_price",
    )
    total = int(df["total"].iloc[0] or 0)
    flagged = int(df["flagged"].iloc[0] or 0)
    share = flagged / total if total else 0.0
    detail = f"{flagged}/{total} rows flagged ({share:.4%}, limit {limit:.2%})"
    return _ok("DQ-05", detail) if share <= limit else _fail("DQ-05", detail)


def check_staleness(
    engine: Engine, cfg: Mapping[str, Any], *, today: date | None = None
) -> CheckResult:
    """DQ-06 (soft): latest loaded date within `data.stale_after_days` of today.

    Skipped for synthetic and replay runs, where "today" is meaningless.
    """
    source = cfg["data"].get("source", "")
    if source == "synthetic":
        return CheckResult("DQ-06", SEVERITY["DQ-06"], True, "skipped: synthetic source")

    df = read_sql(engine, "SELECT MAX(date_key) AS last_key FROM fact_price")
    last = df["last_key"].iloc[0]
    if last is None or pd.isna(last):
        return _fail("DQ-06", "fact_price is empty")

    max_days = int(cfg["data"]["stale_after_days"])
    last_date = from_date_key(int(last)).date()
    reference = today or date.today()
    age = (reference - last_date).days
    detail = f"latest bar {last_date} is {age} calendar day(s) old (limit {max_days})"
    return _ok("DQ-06", detail) if age <= max_days else _fail("DQ-06", detail)


def check_rejected_share(
    engine: Engine, cfg: Mapping[str, Any], stats: Mapping[str, Any] | None = None
) -> CheckResult:
    """DQ-07 (soft): rejected rows below `quality.max_rejected_share` of extracted."""
    limit = float(cfg["quality"]["max_rejected_share"])
    extracted = int((stats or {}).get("rows_extracted", 0) or 0)
    rejected = int((stats or {}).get("rows_rejected", 0) or 0)
    if extracted <= 0:
        df = read_sql(engine, "SELECT COUNT(*) AS n FROM fact_price")
        extracted = int(df["n"].iloc[0] or 0)
    share = rejected / extracted if extracted else 0.0
    detail = f"{rejected}/{extracted} rows rejected ({share:.4%}, limit {limit:.2%})"
    return _ok("DQ-07", detail) if share <= limit else _fail("DQ-07", detail)


# ---------------------------------------------------------------- runner


def run_checks(
    engine: Engine,
    cfg: Mapping[str, Any],
    stats: Mapping[str, Any] | None = None,
) -> list[CheckResult]:
    """Run DQ-01 to DQ-07 and raise if any hard check fails.

    Args:
        engine: Warehouse engine.
        cfg: Loaded config.
        stats: Extracted and rejected counts, used by DQ-07.

    Returns:
        Every check result, in id order.

    Raises:
        DataQualityError: If a hard check failed. The message lists each failure.
    """
    log = get_logger(__name__)
    results = [
        check_duplicates(engine),
        check_price_validity(engine),
        check_date_references(engine),
        check_row_coverage(engine, cfg),
        check_flagged_share(engine, cfg),
        check_staleness(engine, cfg),
        check_rejected_share(engine, cfg, stats),
    ]

    for result in results:
        if result.passed:
            log.info("%s %s: %s", result.id, result.severity, result.detail)
        elif result.severity == "HARD":
            log.error("%s %s FAILED: %s", result.id, result.severity, result.detail)
        else:
            log.warning("%s %s FAILED: %s", result.id, result.severity, result.detail)

    hard_failures = [r for r in results if r.severity == "HARD" and not r.passed]
    if hard_failures:
        summary = "; ".join(f"{r.id}: {r.detail}" for r in hard_failures)
        raise DataQualityError(f"hard data-quality check(s) failed: {summary}")

    return results
