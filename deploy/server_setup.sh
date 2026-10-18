#!/usr/bin/env bash
# RegimeTrader Option B bootstrap — run on the free-tier VPS (Ubuntu 24.04):
#
#     bash ~/server_setup.sh          # after:  scp deploy\server_setup.sh ... rt-vm:
#
# Idempotent: safe to re-run. Installs system packages, clones the repo, overlays the
# kit files you uploaded to your home directory into ~/regimetrader/deploy/, builds the
# pinned virtualenv, creates swap if missing, installs and enables the systemd timer.
#
# Optional environment overrides:
#   APP_DIR=/opt/regimetrader REPO_URL=... REQUIREMENTS=requirements.txt bash server_setup.sh
#
# Full guide: deploy/OPTION_B_FREE.md (Option B, free tier).

set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/regimetrader}"
REPO_URL="${REPO_URL:-https://github.com/Arp1tSingh/regimetrader.git}"
REQUIREMENTS="${REQUIREMENTS:-requirements.lock}"

KIT_FILES="server_setup.sh daily_wrapper.sh db_snapshot.sh regimetrader.service regimetrader.timer"

echo "== RegimeTrader Option B bootstrap =="
echo "   app dir:      $APP_DIR"
echo "   repo:         $REPO_URL"
echo "   requirements: $REQUIREMENTS"

# --- 1. required system packages (Ubuntu 24.04 ships Python 3.12) -------------------
sudo apt-get update -y
sudo apt-get install -y git make python3 python3-venv python3-pip rsync build-essential

# --- 2. code -------------------------------------------------------------------------
if [ ! -d "$APP_DIR/.git" ]; then
    git clone "$REPO_URL" "$APP_DIR"
else
    git -C "$APP_DIR" fetch origin
    git -C "$APP_DIR" reset --hard origin/main
fi
# The kit files (server_setup.sh, the systemd units, wrappers) are not on GitHub until
# they are committed and pushed; overlay whatever was uploaded to $HOME before this run.
for f in $KIT_FILES; do
    if [ -f "$HOME/$f" ]; then
        cp "$HOME/$f" "$APP_DIR/deploy/$f"
    fi
done
chmod +x "$APP_DIR/deploy/"*.sh 2>/dev/null || true

# --- 3. virtualenv with the exact pinned environment ----------------------------------
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/$REQUIREMENTS"

# --- 4. directories the job and reports write into ------------------------------------
mkdir -p "$APP_DIR/logs" "$APP_DIR/data/warehouse" "$APP_DIR/models" "$APP_DIR/reports/daily"

# --- 5. swap (insurance for pip/ETL memory spikes on small shapes) --------------------
if ! sudo swapon --show 2>/dev/null | grep -q .; then
    echo "== creating 2 GB swap =="
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
    sudo swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || \
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# --- 6. state check (the warehouse and model files are NOT in git) --------------------
ok=1
if [ ! -s "$APP_DIR/data/warehouse/regimetrader.db" ]; then
    ok=0
    echo "WARNING: no warehouse at $APP_DIR/data/warehouse/regimetrader.db"
    echo "         upload it from your laptop next (OPTION_B_FREE.md, step 4), e.g.:"
    echo "           scp data\\warehouse\\regimetrader.db rt-vm:regimetrader/data/warehouse/regimetrader.db"
fi
if ls "$APP_DIR"/models/*.joblib >/dev/null 2>&1; then
    echo "models:   $(ls "$APP_DIR"/models/*.joblib | wc -l) joblib file(s) present"
else
    ok=0
    echo "WARNING: no model files in $APP_DIR/models/ — the job refuses to run without"
    echo "         the year's wf<year>.joblib; upload models/*.joblib from your laptop."
fi

# --- 7. systemd units (timer triggers weekdays 17:00 Asia/Kolkata) --------------------
sudo tee /etc/systemd/system/regimetrader.service >/dev/null \
    < <(sed "s/^User=ubuntu$/User=$USER/" "$APP_DIR/deploy/regimetrader.service")
sudo cp "$APP_DIR/deploy/regimetrader.timer" /etc/systemd/system/regimetrader.timer
sudo systemctl daemon-reload
sudo systemctl enable --now regimetrader.timer

# --- 8. report -------------------------------------------------------------------------
echo
echo "== done =="
systemctl list-timers --no-pager | head -n 5
"$APP_DIR/.venv/bin/python" -c "import sys; sys.path.insert(0, '$APP_DIR')
from src.paper import daily_job
print('daily_job import: ok')"
if [ "$ok" -eq 1 ]; then
    echo
    echo "Next: first manual run   ssh $(hostname):  $APP_DIR/.venv/bin/python -m src.paper.daily_job"
    echo "Then configure healthchecks.io (OPTION_B_FREE.md, step 7)."
else
    echo
    echo "Next: upload the state files (OPTION_B_FREE.md, step 4), then run:"
    echo "      $APP_DIR/.venv/bin/python -m src.paper.daily_job"
fi
