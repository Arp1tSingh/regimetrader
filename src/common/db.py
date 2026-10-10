"""Warehouse access: SQLAlchemy engine, schema application, reads and batch writes.

`DECISIONS` R19: SQLAlchemy 2.x engine, raw SQL via `text()` with named bind
parameters, foreign keys ON and WAL. SQLite leaves foreign keys off by default, so
the pragma is applied on every new connection (AGENTS.md section 6).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection

#: DDL for the whole warehouse. Canonical copy lives in `sql/schema.sql`.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def get_engine(cfg: Mapping[str, Any]) -> Engine:
    """Create an engine for the configured warehouse with pragmas applied.

    Args:
        cfg: Loaded config; `paths.db` gives the SQLite file path.

    Returns:
        A SQLAlchemy `Engine`. Parent folders are created on demand.
    """
    db_path = Path(cfg["paths"]["db"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path.as_posix()}", future=True)

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
        finally:
            cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Apply `sql/schema.sql`. Idempotent: every statement is IF NOT EXISTS / OR IGNORE.

    Raises:
        RuntimeError: If the schema file is missing.
    """
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"schema file not found: {SCHEMA_PATH}")
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    with engine.begin() as conn:
        # The sqlite driver parses comments and statement boundaries itself;
        # splitting on `;` here would break on semicolons inside `--` comments.
        raw = conn.connection.driver_connection
        raw.executescript(script)


def read_sql(
    engine: Engine,
    sql: str,
    params: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a read query and return the result as a DataFrame.

    Args:
        engine: SQLAlchemy engine.
        sql: Query text. Must use named bind parameters (`:name`), never f-strings.
        params: Optional bind parameters.

    Returns:
        A DataFrame, empty but with columns preserved when the query matches nothing.
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), dict(params or {}))
        rows = result.fetchall()
        columns = list(result.keys())
    return pd.DataFrame(rows, columns=columns)


def execute_many(
    conn: Connection,
    sql: str,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    """Execute one statement once per row of mappings.

    Args:
        conn: An open connection, normally from `engine.begin()` so the batch
            commits or rolls back as a unit.
        sql: Statement with named bind parameters.
        rows: Iterable of parameter mappings.

    Returns:
        The number of rows submitted.
    """
    payload: list[Mapping[str, Any]] = list(rows)
    if not payload:
        return 0
    for row in payload:
        conn.execute(text(sql), dict(row))
    return len(payload)


def scalar(engine: Engine, sql: str, params: Mapping[str, Any] | None = None) -> Any:
    """Return the first column of the first row, or None when there is no row."""
    with engine.connect() as conn:
        row = conn.execute(text(sql), dict(params or {})).first()
    return None if row is None else row[0]


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    """Validate a SQL identifier used to build a statement.

    Parameter values must never be interpolated; table and column names cannot be
    bind parameters, so anything dynamic is validated against this pattern.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return name
