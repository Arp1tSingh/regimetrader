#!/usr/bin/env bash
# Consistent snapshot of the RegimeTrader warehouse (for backups).
#
# Uses the SQLite online-backup API, so it is safe while the paper job or dashboard is
# running (WAL included). Run before pulling the database to your laptop:
#
#   bash deploy/db_snapshot.sh            # -> data/warehouse/_snapshot.db
#   bash deploy/db_snapshot.sh /tmp/x.db  # explicit destination
#
# deploy/pull_backup.ps1 (on the Windows laptop) calls this over SSH automatically.

set -euo pipefail

DIR="${REGIMETRADER_DIR:-$HOME/regimetrader}"
cd "$DIR"
OUT="${1:-data/warehouse/_snapshot.db}"

"$DIR/.venv/bin/python" - "$OUT" <<'PYEOF'
import sqlite3
import sys

source = "data/warehouse/regimetrader.db"
target = sys.argv[1]
src = sqlite3.connect(source)
dst = sqlite3.connect(target)
src.backup(dst)  # copies a consistent version, page by page
dst.commit()
dst.close()
src.close()
print(f"snapshot written: {target} (from {source})")
PYEOF
