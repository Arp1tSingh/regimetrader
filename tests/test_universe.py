"""Tests for the universe loader (task T1.1, spec 03 section 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.etl.universe import UniverseError, load_universe

HEADER = "symbol,company_name,sector,industry\n"


def write_universe(path: Path, rows: list[str], preamble: str = "# as of 2026-10-10\n") -> Path:
    path.write_text(preamble + HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def cfg_for(tmp_path: Path, universe: Path) -> dict[str, Any]:
    return {
        "data": {
            "universe_file": str(universe),
            "index_symbol": "^NSEI",
            "vix_symbol": "^INDIAVIX",
        }
    }


@pytest.fixture
def small_universe(tmp_path: Path) -> Path:
    return write_universe(
        tmp_path / "u.csv",
        [
            "RELIANCE.NS,Reliance Industries Ltd.,Energy,Oil Gas & Consumable Fuels",
            "TCS.NS,Tata Consultancy Services Ltd.,IT,Information Technology",
            "HDFCBANK.NS,HDFC Bank Ltd.,Banking,Financial Services",
            "M&M.NS,Mahindra & Mahindra Ltd.,Auto,Automobile and Auto Components",
            "BAJAJ-AUTO.NS,Bajaj Auto Ltd.,Auto,Automobile and Auto Components",
        ],
    )


# --------------------------------------------------------------- structure


def test_columns_and_types(small_universe: Path, tmp_path: Path) -> None:
    df = load_universe(cfg_for(tmp_path, small_universe))

    assert list(df.columns) == ["symbol", "company_name", "sector", "industry", "instrument_type"]
    assert set(df["instrument_type"]) == {"EQUITY", "INDEX", "VOLATILITY"}


def test_index_and_vix_rows_appended(small_universe: Path, tmp_path: Path) -> None:
    df = load_universe(cfg_for(tmp_path, small_universe))
    indexed = df.set_index("symbol")

    assert indexed.loc["^NSEI", "instrument_type"] == "INDEX"
    assert indexed.loc["^NSEI", "sector"] == "INDEX"
    assert indexed.loc["^NSEI", "company_name"] == "Nifty 50 Index"

    assert indexed.loc["^INDIAVIX", "instrument_type"] == "VOLATILITY"
    assert indexed.loc["^INDIAVIX", "sector"] == "VOLATILITY"
    assert indexed.loc["^INDIAVIX", "company_name"] == "India VIX"


def test_seeds_are_not_duplicated_as_equities(small_universe: Path, tmp_path: Path) -> None:
    """The index and VIX are appended once each, not also counted as equities."""
    df = load_universe(cfg_for(tmp_path, small_universe))
    data_lines = [
        line
        for line in small_universe.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    equity_rows = len(data_lines) - 1  # minus the header
    assert equity_rows == 5
    assert df["symbol"].is_unique
    assert len(df) == equity_rows + 2
    assert (df["instrument_type"] == "EQUITY").sum() == equity_rows


def test_awkward_symbols_survive(small_universe: Path, tmp_path: Path) -> None:
    """`03` section 3 calls out M&M.NS and BAJAJ-AUTO.NS as symbols that often break."""
    df = load_universe(cfg_for(tmp_path, small_universe))
    assert {"M&M.NS", "BAJAJ-AUTO.NS"}.issubset(set(df["symbol"]))


def test_result_is_sorted_for_determinism(small_universe: Path, tmp_path: Path) -> None:
    """AGENTS.md section 6: sort before selecting so results are deterministic."""
    df = load_universe(cfg_for(tmp_path, small_universe))
    assert list(df["symbol"]) == sorted(df["symbol"])


def test_comment_lines_are_skipped(small_universe: Path, tmp_path: Path) -> None:
    df = load_universe(cfg_for(tmp_path, small_universe))
    assert not df["symbol"].astype(str).str.startswith("#").any()


def test_configured_index_symbol_is_used(tmp_path: Path, small_universe: Path) -> None:
    cfg = cfg_for(tmp_path, small_universe)
    cfg["data"]["index_symbol"] = "^CUSTOMIDX"
    df = load_universe(cfg)
    assert "^CUSTOMIDX" in set(df["symbol"])


# --------------------------------------------------------------- validation


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UniverseError, match="not found"):
        load_universe(cfg_for(tmp_path, tmp_path / "nope.csv"))


def test_empty_file_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(UniverseError, match="empty"):
        load_universe(cfg_for(tmp_path, empty))


def test_comment_only_file_raises(tmp_path: Path) -> None:
    only = tmp_path / "c.csv"
    only.write_text("# as of 2026-10-10\n", encoding="utf-8")
    with pytest.raises(UniverseError, match="empty"):
        load_universe(cfg_for(tmp_path, only))


def test_missing_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol,company_name,sector\nA.NS,A,IT\n", encoding="utf-8")
    with pytest.raises(UniverseError, match="industry"):
        load_universe(cfg_for(tmp_path, bad))


def test_duplicate_symbol_raises(tmp_path: Path) -> None:
    dup = write_universe(
        tmp_path / "dup.csv",
        ["A.NS,A,IT,Information Technology", "A.NS,A again,IT,Information Technology"],
    )
    with pytest.raises(UniverseError, match="duplicate"):
        load_universe(cfg_for(tmp_path, dup))


def test_blank_field_raises(tmp_path: Path) -> None:
    blank = write_universe(tmp_path / "blank.csv", ["A.NS,,IT,Information Technology"])
    with pytest.raises(UniverseError, match="blank"):
        load_universe(cfg_for(tmp_path, blank))


def test_mandatory_sectors_enforced_at_full_size(tmp_path: Path) -> None:
    """A 40+ row universe must contain both Banking and IT (`03` section 3)."""

    def rows(*sectors: str) -> list[str]:
        return [f"S{i}.NS,Co {i},{sector},Some Industry" for i, sector in enumerate(sectors)]

    # 40 rows with Banking but no IT must be refused.
    without_it = rows(*(["Banking"] + ["Energy"] * 39))
    path = write_universe(tmp_path / "no_it.csv", without_it)
    with pytest.raises(UniverseError, match="IT"):
        load_universe(cfg_for(tmp_path, path))

    # The same file with both labels present must load.
    with_both = rows(*(["Banking", "IT"] + ["Energy"] * 38))
    path = write_universe(tmp_path / "both.csv", with_both)
    df = load_universe(cfg_for(tmp_path, path))
    assert len(df[df["instrument_type"] == "EQUITY"]) == 40


def test_mandatory_sectors_not_enforced_on_small_fixture(tmp_path: Path) -> None:
    """Below 40 rows the file is a test fixture, so the sector rule must not fire."""
    small = write_universe(tmp_path / "s.csv", ["A.NS,A,Energy,Oil"])
    assert load_universe(cfg_for(tmp_path, small)) is not None


# --------------------------------------------------------------- real file


def test_shipped_universe_file_is_valid(cfg_base: dict[str, Any]) -> None:
    """The frozen universe downloaded from NSE in T1.8 must load and be complete."""
    df = load_universe(cfg_base)
    equities = df[df["instrument_type"] == "EQUITY"]
    assert len(equities) == 50
    assert {"Banking", "IT"}.issubset(set(equities["sector"]))
    assert {"^NSEI", "^INDIAVIX"}.issubset(set(df["symbol"]))
