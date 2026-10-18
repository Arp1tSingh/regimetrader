#!/usr/bin/env bash
# RegimeTrader daily job wrapper (installed by deploy/server_setup.sh).
#
# Runs the daily paper job, appends the exit code to logs/daily.log, and pings
# healthchecks.io when HEALTHCHECK_URL is set in the environment (via
# /etc/regimetrader.env -> systemd EnvironmentFile). Everything else about the job is
# unchanged: see deploy/OPTION_B_FREE.md and docs/spec/08_runbook.md.
#
# Exit codes mirror src/paper/daily_job.py:
#   0 success (including "nothing to do" on a holiday)
#   1 config or other error    2 data/ETL failure (refused to trade on stale data)
#   3 refused by the frozen-config/test-lock guard (should not happen in operation)

set -u

DIR="${REGIMETRADER_DIR:-$HOME/regimetrader}"
cd "$DIR" || exit 1
LOG_DIR="$DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily.log"
HC="${HEALTHCHECK_URL:-}"

"$DIR/.venv/bin/python" -m src.paper.daily_job >>"$LOG" 2>&1
code=$?

stamp="$(date '+%Y-%m-%d %H:%M:%S%z')"
echo "[$stamp] daily_job exit=$code" >>"$LOG"

if [ -n "$HC" ]; then
    if [ "$code" -eq 0 ]; then
        # success ping (healthchecks.io marks the check as up)
        curl -fsS -m 15 --retry 3 -o /dev/null "$HC" || true
    else
        # failure ping: healthchecks.io emails immediately instead of waiting for grace
        curl -fsS -m 15 --retry 3 -o /dev/null "$HC/fail" || true
    fi
fi

exit "$code"
