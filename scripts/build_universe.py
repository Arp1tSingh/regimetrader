"""One-off helper: convert the NSE Nifty 50 CSV into `config/nifty50_symbols.csv`.

Not part of the ETL. Run manually when the universe must be refreshed (T1.8).
Kept in the repo so the mapping from NSE industry to project sector is auditable.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

SOURCE_COLUMNS = ["Company Name", "Industry", "Symbol", "Series", "ISIN Code"]

#: NSE industry -> project sector label (`03` section 3).
SECTOR_BY_INDUSTRY = {
    "Information Technology": "IT",
    "Healthcare": "Pharma and Healthcare",
    "Automobile and Auto Components": "Auto",
    "Metals & Mining": "Metals and Mining",
    "Fast Moving Consumer Goods": "FMCG",
    "Oil Gas & Consumable Fuels": "Energy",
    "Power": "Power",
    "Construction Materials": "Materials",
    "Consumer Durables": "Consumer Durables",
    "Consumer Services": "Consumer Services",
    "Services": "Services",
    "Telecommunication": "Telecom",
    "Construction": "Infrastructure",
    "Capital Goods": "Infrastructure",
}

#: `Financial Services` is split by hand: `Banking` for the banks, the rest stay
#: `Financial Services`. `03` section 3 requires both labels to exist.
BANKS = {"AXISBANK", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN"}


def sector_for(industry: str, symbol: str) -> str:
    """Project sector label for one NSE row."""
    if industry == "Financial Services":
        return "Banking" if symbol in BANKS else "Financial Services"
    if industry not in SECTOR_BY_INDUSTRY:
        raise SystemExit(f"unmapped industry {industry!r} for {symbol}; add it to SECTOR_BY_INDUSTRY")
    return SECTOR_BY_INDUSTRY[industry]


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 2:
        print("usage: python scripts/build_universe.py <nse_csv> <out_csv>")
        return 1
    src, dst = Path(args[0]), Path(args[1])

    with src.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    missing = [c for c in SOURCE_COLUMNS if rows and c not in rows[0]]
    if missing:
        print(f"unexpected NSE CSV, missing columns: {missing}")
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as handle:
        # The first line records when the universe was frozen (`03` section 3).
        handle.write(f"# as of {date.today().isoformat()}\n")
        writer = csv.writer(handle)
        writer.writerow(["symbol", "company_name", "sector", "industry"])
        for row in sorted(rows, key=lambda r: r["Symbol"]):
            symbol = row["Symbol"].strip()
            writer.writerow(
                [f"{symbol}.NS", row["Company Name"].strip(), sector_for(row["Industry"], symbol), row["Industry"]]
            )
    print(f"wrote {len(rows)} rows to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())