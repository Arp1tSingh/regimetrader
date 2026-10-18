# Option B (free): RegimeTrader daily job on an always-on free VPS

Run Mode B (the daily paper job) on a small always-on Linux box that costs **₹0/month**,
with the dashboard still shown from your laptop. This is the guide `deploy/cron.example`
pointed at ("deployment is the operator's decision") — for the free tier that decision
means Oracle Cloud.

> Scope reminder (README §8): this is a paper-trading academic prototype. "Deploy" means
> keeping the paper run history alive and making it available when you want to look at it.
> There is no broker, no real money, and deliberately none planned.

---

## 0. What runs where

| Component | Where | Why |
|---|---|---|
| `python -m src.paper.daily_job` (scheduled weekdays 17:00 IST) | Free VM | Needs a persistent disk for the SQLite warehouse and network for yfinance |
| Dashboard (`make dashboard`) | Your laptop (default) | You don't need it running 24/7; SSH tunnel shows the *server's* data if you want the freshest warehouse |
| Monitoring | healthchecks.io (free, no card) | Dead-man's switch: email if a scheduled run doesn't happen |
| Backups | Pulled to your laptop weekly | The VM must be treated as disposable (see §7) |

Files this kit adds (all already on disk in `deploy/`):

| File | Purpose |
|---|---|
| `server_setup.sh` | Bootstrap the VM (packages, clone, venv, swap, systemd install) |
| `regimetrader.service` | systemd unit running the job as a one-shot |
| `regimetrader.timer` | Weekday 17:00 Asia/Kolkata trigger with catch-up after downtime |
| `daily_wrapper.sh` | Runs the job, logs the exit code, pings healthchecks.io |
| `db_snapshot.sh` | Server-side consistent SQLite snapshot (for backups) |
| `pull_backup.ps1` | Laptop-side weekly backup pull (Windows) |
| `cron.example`, `github_actions_daily.yml` | Preexisting examples, unchanged |

The systemd timer (not cron) is used on purpose: `Timezone=` keeps the schedule at
17:00 IST regardless of the server's system clock, and `Persistent=true` makes a reboot
or a missed slot re-run the job at the next opportunity — and the job is idempotent
(already-processed dates are skipped), so catch-up is safe by design.

---

## 1. Why Oracle Cloud (and what changed in June 2026)

Oracle's Always Free tier is the only big-provider offer with a **real, always-on VM +
persistent disk at $0 indefinitely**:

- **1× `VM.Standard.A1.Flex` (ARM)** instance — up to **2 OCPUs / 12 GB RAM** since
  12 June 2026 (halved from 4/24 without announcement; verify limits before creating).
- **2× `VM.Standard.E2.1.Micro` (AMD)** small instances on top of that.
- **200 GB total block storage**, free forever.
- 10 TB/month egress — irrelevant for this workload.

Alternatives and why they lose: AWS/GCP/Azure free compute either expires (12 months)
or (GCP e2-micro) is limited to US regions and still weaker; Render/Railway/Heroku/Fly
have no genuine free always-on box with persistent disk; GitHub Actions is ephemeral
(Option C in the repo's YAML, needs the artifact dance).

**Signup requires a real credit or debit card** (identity verification only; no virtual,
prepaid or PIN-dependent debit cards; temporary authorization holds are removed in
3–5 days, nothing is charged for Always Free resources). Signup fraud screening is the
main failure point: use your real banking name and address, no VPN, another browser or
network if rejected. One account per person; a tenancy idle for 30+ days can be
suspended, so the weekly backup pull (§8) doubles as a sign of life.

---

## 2. Create the instance (~15 minutes, console only)

1. Sign up at `signup.oraclecloud.com` (home region matters — Always Free compute is
   locked to it; pick the region you can actually get capacity in).
2. Console → Compute → **Create Instance**:
   - Name: `regimetrader`
   - Image: **Canonical Ubuntu 24.04** (Always Free eligible)
   - Shape: **Ampere A1.Flex**, **2 OCPU, 12 GB** (stay inside the current free limit)
   - Boot volume: default ~50 GB is plenty (state is ~100 MB)
   - Networking: defaults (public IP assigned)
   - SSH keys: paste your laptop's public key (`ssh-keygen -t ed25519` if you don't have one)
3. Security list: keep default — inbound **TCP 22 only**. Nothing else is exposed;
   the dashboard is reached via SSH tunnel (§9), never an open port.
4. In the console, also create a **Budget** of $1 with an alert at threshold — any cost
   you are *not expecting* (i.e. outside Always Free limits) mails you immediately. This
   is the guard against the classic accident of creating one paid resource.
5. On your laptop add an SSH alias (`C:\Users\Admin\.ssh\config`):

   ```
   Host rt-vm
     HostName <instance_public_ip>
     User ubuntu
     IdentityFile ~/.ssh/id_ed25519
   ```

   Test: `ssh rt-vm uname -m` → should print `aarch64`.

If A1 capacity is unavailable ("out of host capacity"): retry on different days/hours
(release is lumpy), or fall back to a 1-OCPU shape, or use one of the two E2.1.Micro
micros (1 GB RAM — small, but this pipeline fits it with the swap this kit adds).

---

## 3. Bootstrap the VM (~5 minutes)

First push the five kit files (the server's fresh clone won't have them yet — they exist
only in your working tree until you commit and push), then bootstrap:

```powershell
# from regimetrader_pack\ on your laptop — upload kit files to the VM's home dir
scp deploy\server_setup.sh deploy\daily_wrapper.sh deploy\db_snapshot.sh deploy\regimetrader.service deploy\regimetrader.timer rt-vm:

# bootstrap (the script clones the repo, overlays your uploaded kit files into
# ~/regimetrader/deploy/, sets up venv and systemd, and enables the timer)
ssh rt-vm "bash server_setup.sh"
```

Then verify the timer:

```bash
ssh rt-vm "systemctl list-timers regimetrader.timer --no-pager"
```

Next trigger should be **the next weekday 17:00 IST** (`Mon..Fri *-*-* 17:00:00`,
`Timezone=Asia/Kolkata`, `Persistent=true`).

---

## 4. Upload the state (the part git cannot give you)

Code, config, `config/frozen_config.yaml`, `config/nifty50_symbols.csv`,
`TEST_RUN_LOCK.json` and `test_run_log.jsonl` are **committed**, so `git clone` on the
server already has them. What is git-ignored and must be uploaded (~80 MB total):

| Local path (repo root) | Remote | Notes |
|---|---|---|
| `data\warehouse\regimetrader.db` | `data/warehouse/` | **The** state — ~65 MB; the 2026-10-18 local test left `paper_live` committed flat at 2026-10-16, so Monday's scheduled run continues from there |
| `models\*.joblib` | `models/` | regime model + every `wf<year>` classifier; the job **refuses to run** without the year's model file — it does not retrain on its own at install time |

(`data/processed/panel.parquet` is **not** needed — the job rebuilds the panel from the
warehouse, `build_panel(..., use_cache=False, live=True)`. `data/raw/` is a cache and
unnecessary. `*.db-wal`/`-shm` were empty at last check.)

From `regimetrader_pack\` on your laptop:

```powershell
scp data\warehouse\regimetrader.db rt-vm:regimetrader/data/warehouse/regimetrader.db
Get-ChildItem models\*.joblib | ForEach-Object { scp $_.FullName "rt-vm:regimetrader/models/" }
```

(One file at a time because Windows `scp` does not expand local `*` globs: the `ForEach`
loop does it. A `tar | ssh` pipe works from WSL if you prefer.)

If you re-upload later, mirror the backup discipline in §8: upload the **snapshot**, and
have the timer stopped while the warehouse is being replaced.

---

## 5. First manual run (do this before Monday)

```bash
ssh rt-vm "cd regimetrader && .venv/bin/python -m src.paper.daily_job"
```

Expected outcomes, all healthy:

| Message | Meaning |
|---|---|
| `<date_key>: n fill(s), equity ...` | The first ETL evening (or catch-up after downtime) worked: it loaded data for the last trading date(s), settled the previous day's targets, and stored tomorrow's |
| `nothing to do` | The latest loaded date is already processed. Expected on a manual re-run: `paper_live` has processed exactly `2026-10-09` so far (the lock's `data_end_date`) |
| `nothing to do (holiday or already processed)` | Market holiday; exit 0. The *stale-data* alert is what distinguishes it from a data problem — see §7 |
| non-zero exit | See the exact codes below; find the traceback in `logs/daily.log` |

Exit-code contract of the job itself (`src/paper/daily_job.py`):

| Code | Meaning | Wrapper/responses |
|---|---|---|
| 0 | success, incl. "nothing to do" | success ping |
| 1 | config/other error | failure ping |
| 2 | ETL/data failure after retries (it refuses to trade on stale data) | failure ping |
| 3 | refused (frozen-config / test-lock guard — must never happen with the shipped state) | failure ping |

Sanity check without touching the ETL (network-free, pure warehouse):

```bash
ssh rt-vm "cd regimetrader && .venv/bin/python -m src.paper.daily_job --no-etl"
```

**Local pre-deploy test, 2026-10-18 (passed, `paper_live` run used):** incremental ETL
loaded 520 rows (52 instruments, data through 2026-10-16), all seven DQ checks passed,
and `paper_live` committed its first flat record at the latest trading date 2026-10-16
(equity 1,000,000, regime SIDEWAYS, 0 targets). Whole run: **~90 s wall time** end to
end, so VM sizing is trivially comfortable.

Note the R26 semantics that test proved out, which is subtler than README §7 suggests:
a *brand-new* live run does **not** back-fill — it starts flat at the *latest* trading
date (in this case Friday 2026-10-16, skipping the Oct 12–15 catch-up entirely; that
record is what `last_processed` reads, store.py:53). Every run after the first follows
the last committed date normally. If the state shipped here (flat at 2026-10-16) lands
before the next trading day, the VM and this laptop continue from the same day. If
deployment waits days, the first server run just flats at whatever the latest date then
is — still correct, no manual intervention.

---

## 6. Scheduling is already enabled

`server_setup.sh` enabled `regimetrader.timer` (`Persistent=true`), which starts
`regimetrader.service` → `deploy/daily_wrapper.sh` → the job, weekdays at 17:00 IST.

```bash
# verify next fire
ssh rt-vm "systemctl list-timers regimetrader.timer --no-pager"
# run manually any time
ssh rt-vm "systemctl start regimetrader.service"
# see the result
ssh rt-vm "journalctl -u regimetrader.service -n 50 --no-pager"
ssh rt-vm "tail -n 20 ~/regimetrader/logs/daily.log"
```

Timing feel: after NSE closes 15:30 IST, EOD data publishes later; 17:00 gives margin
(the runbook's cut-off). A weekend reboot triggers one catch-up run, which answers
"nothing to do" — harmless by idempotency.

---

## 7. Monitoring (free, no card)

1. Create a free **healthchecks.io** account (20 checks, email alerts).
2. One check, **Schedule** mode: schedule `0 17 * * 1-5`, timezone **Asia/Kolkata**,
   grace 2h. Name it `regimetrader daily`.
3. Copy its Ping URL (looks like `https://hc-ping.com/<uuid>`) to the server:

```bash
ssh rt-vm 'echo HEALTHCHECK_URL=https://hc-ping.com/<uuid> | sudo tee /etc/regimetrader.env >/dev/null'
```

`daily_wrapper.sh` pings success on exit 0, failure otherwise; healthchecks emails you
when Monday 17:00 comes and goes without a ping — the "VM vanished, or Oracle reclaimed
it, or data stopped arriving" alarm. That plus the §8 backups is the whole monitoring
plane; nothing else is needed for paper trading.

Optional (job alerts during the day, not just missed-run detection): the project's own
`src/paper/alerts.py` emails when `SMTP_HOST, SMTP_USER, SMTP_PASSWORD, ALERT_TO` are
in the environment — Gmail app-password works. Put them in the same
`/etc/regimetrader.env`; `EnvironmentFile=-` loads it. Without SMTP configured the
alerts still land in `logs/alerts.log`, read via SSH.

Alert-worthy events the project itself raises: failed ETL, no new index bar for
`data.stale_after_days` = 3 days (weekday-without-data vs holiday), regime change,
paper drawdown above `paper.alert_drawdown` = 0.10.

---

## 8. Backup discipline (weekly — this is what makes the VM disposable)

From your laptop, `regimetrader_pack\deploy\pull_backup.ps1`:

```powershell
.\deploy\pull_backup.ps1 -HostAlias rt-vm
```

It asks the server for a consistent SQLite snapshot (`db_snapshot.sh` uses the sqlite3
online-backup API, so it is safe even if the server is mid-run with WAL active), copies
down the warehouse snapshot, `reports/daily/` and `logs/` into `..\backups\<timestamp>\`,
and prunes to the last 8 backups. Schedule it weekly:

```powershell
schtasks /Create /TN RegimeTraderWeeklyBackup /SC WEEKLY /D TUE /ST 09:00 /TR ^
  "powershell -ExecutionPolicy Bypass -File C:\Users\Admin\Projects\DWM\regimetrader_pack\deploy\pull_backup.ps1"
```

Rule for maintenance sessions: **backup first, then update** (`git -C ~/regimetrader
pull --ff-only`), and stop the timer while replacing the warehouse by hand.

---

## 9. Dashboard over an SSH tunnel (no open ports)

The laptop dashboard reads the **local** warehouse; when you want to see the **server's**
fresher warehouse (e.g. right after a weekday evening run):

```powershell
# terminal 1 — tunnel
ssh -N -L 8501:localhost:8501 rt-vm
# terminal 2 — dashboard, headless, localhost-only
ssh rt-vm "cd regimetrader && .venv/bin/python -m streamlit run src/dashboard/app.py --server.headless true"
# browser: http://localhost:8501
```

Everything is read-only by construction (all dashboards queries are SELECTs; tested).
For a viva-period public link, `make demo` on the laptop (cloudflared quick tunnel)
against the local warehouse remains the documented path; do not expose this server's
Streamlit publicly.

---

## 10. Costs and the one real risk: idle reclamation

This deployment's only meaningful risk is **Oracle reclaiming an idle Always Free
instance** against their written policy: over a 7-day window, CPU, network and memory
95th-percentile all under 20% (memory applies only to A1 shapes). A 5-minute daily job
**cannot** keep a 7-day p95 above 20%; assume this policy applies to you. The kit deals
with it honestly rather than fighting it:

- healthchecks.io tells you within the same evening the VM disappeared (§7);
- the weekly backup means the loss is bounded by a few days of paper history;
- recreation is **~20 minutes**: console steps in §2, `server_setup.sh`, upload
  `regimetrader.db` from your most recent `backups\<date>\`, done — no code, config or
  guard to reproduce separately. There is no strategy state anywhere outside the
  warehouse + models dir.

Also legitimate: occasionally use the dashboard over the tunnel (memory+network
activity), keep the tenancy's boot volume under the limit, do not create paid shapes
during the 30-day trial so they don't get reclaimed at trial end (Always Free resources
survive it). Re-check free limits twice a year; Oracle changes them without notice.

Egress/bandwidth: negligible (~daily yfinance calls, weekly ~70 MB backup pull).

---

## 11. Fallback chassis if Oracle is impossible

| Option | Cost | When to pick | Effort |
|---|---|---|---|
| GCP e2-micro (US regions) | $0, needs card | Oracle signup rejected but you have a card | similar; weaker box |
| GitHub Actions (`github_actions_daily.yml`) | $0 (public repo: unlimited minutes; private: 2000 min/mo incl.) | no card at all; OK losing run history unless you add artifact DB restore/save — the YAML documents three fixes; artifact restore is the sane one | lowest |
| Old Android phone + Termux, `cron` inside | $0, no card | you already have the spare phone | quirky; battery/charger discipline |
| (Not free) any $2–4/mo VPS | — | you eventually get tired of reclamation roulette | — |

---

## 12. What you should never have to do

- Nothing in January 2027: the job trains `wf2027` by itself at the first run of the new
  calendar year (its model file gets created in `~/regimetrader/models/`). The regime
  model is never refit (`regime.refit: none`).
- No re-running of `final-test`/`tune`/`train` on the server — ever. Those are research
  commands and running them here would overwrite frozen results. Research happens on
  your laptop repo / a fresh clone only.
- No edits to code: the deploy surface is a systemd timer (`Timezone=` +
  `Persistent=true`) standing in for the `cron.example` line; the cron and Actions files
  stay the uninstalled examples they are (decision R22).

---

*Kit prepared 2026-10-18, verified against `origin/main` of
`github.com/Arp1tSingh/regimetrader` after a passing local trial of the timed command
(see §5, `paper_live` flat at 2026-10-16). The pre-existing `deploy/cron.example` and
`deploy/github_actions_daily.yml` remain the uninstalled examples; nothing here is
installed until you run it on a VM.*
