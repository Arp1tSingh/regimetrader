"""Universe loading (`spec 03` section 3).

Reads `config/nifty50_symbols.csv` (columns `symbol, company_name, sector, industry`,
lines starting `#` are comments), validates it, and appends the index and VIX seed rows.

The universe is never fabricated: if the file is missing or empty, raise so the caller
stops and asks the human (`AGENTS.md` rule 11).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

#: Columns the universe file must provide.
REQUIRED_COLUMNS = ("symbol", "company_name", "sector", "industry")

#: Sector labels the OLAP queries default to; both must exist (`03` section 3).
MANDATORY_SECTORS = ("Banking", "IT")

#: Row count above which the mandatory-sector check is applied. Below it the file is
#: a fixture for tests, not the real universe.
FULL_UNIVERSE_ROWS = 40

#: Index and VIX seed rows (`DECISIONS` R9).
INDEX_ROWS = (
    {
        "symbol": "^NSEI",
        "company_name": "Nifty 50 Index",
        "sector": "INDEX",
        "industry": "INDEX",
        "instrument_type": "INDEX",
    },
    {
        "symbol": "^INDIAVIX",
        "company_name": "India VIX",
        "sector": "VOLATILITY",
        "industry": "VOLATILITY",
        "instrument_type": "VOLATILITY",
    },
)


class UniverseError(ValueError):
    """Raised when the universe file is missing, malformed or invalid."""


def _read_universe_file(path: Path) -> pd.DataFrame:
    """Read the universe CSV, skipping `#` comment lines."""
    if not path.exists():
        raise UniverseError(
            f"universe file not found: {path}. It must be downloaded from NSE or placed "
            "by hand; the constituent list is never fabricated (03 section 3)."
        )
    # `csv.DictReader` has no `comment` parameter (only `csv.reader` does), so
    # `#` lines are filtered out before parsing.
    lines = [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if not line.lstrip().startswith("#")
    ]
    rows = list(csv.DictReader(lines))

    if not rows:
        raise UniverseError(f"universe file is empty: {path}")

    missing = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing:
        raise UniverseError(
            f"universe file {path} is missing column(s): {', '.join(missing)}; "
            f"expected {', '.join(REQUIRED_COLUMNS)}"
        )

    df = pd.DataFrame(rows)[list(REQUIRED_COLUMNS)]
    for column in REQUIRED_COLUMNS:
        # Treat whitespace-only cells as missing so validation cannot be bypassed.
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    return df


def _validate(df: pd.DataFrame, path: Path) -> None:
    """Check uniqueness and, for a full universe, the mandatory sector labels."""
    if df["symbol"].isna().any() or (df["symbol"] == "").any():
        raise UniverseError(f"universe file {path} has a blank symbol")

    duplicates = sorted(
        df.loc[df["symbol"].duplicated(), "symbol"].dropna().unique().tolist()
    )
    if duplicates:
        raise UniverseError(f"universe file {path} has duplicate symbol(s): {', '.join(duplicates)}")

    if df[["company_name", "sector", "industry"]].isna().any().any():
        raise UniverseError(f"universe file {path} has blank company_name, sector or industry")

    if len(df) >= FULL_UNIVERSE_ROWS:
        present = set(df["sector"])
        missing_sectors = [s for s in MANDATORY_SECTORS if s not in present]
        if missing_sectors:
            raise UniverseError(
                f"universe file {path} is missing required sector label(s): "
                f"{', '.join(missing_sectors)} (03 section 3)"
            )


def load_universe(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Load and validate the universe, appending the index and VIX rows.

    Args:
        cfg: Loaded config; `data.universe_file`, `data.index_symbol` and
            `data.vix_symbol` are read from it.

    Returns:
        A DataFrame with columns `symbol, company_name, sector, industry,
        instrument_type`, sorted by symbol so downstream iteration is deterministic
        (AGENTS.md section 6). Equities are `EQUITY`, the index `INDEX` and the VIX
        `VOLATILITY`.

    Raises:
        UniverseError: If the file is missing, empty, malformed or invalid.
    """
    path = Path(cfg["data"]["universe_file"])
    df = _read_universe_file(path)
    _validate(df, path)
    df["instrument_type"] = "EQUITY"

    seeds = []
    for template in INDEX_ROWS:
        row = dict(template)
        # Honour the configured symbols rather than hard-coding them.
        key = "index_symbol" if row["instrument_type"] == "INDEX" else "vix_symbol"
        row["symbol"] = cfg["data"][key]
        seeds.append(row)
    seed_df = pd.DataFrame(seeds)[list(REQUIRED_COLUMNS) + ["instrument_type"]]

    combined = pd.concat([df, seed_df], ignore_index=True)
    return combined.sort_values("symbol", kind="stable").reset_index(drop=True)
