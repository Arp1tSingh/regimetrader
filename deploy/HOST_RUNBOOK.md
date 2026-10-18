# Runbook: RegimeTrader dashboard — this laptop as the public viva host

Public URL: **https://regimetrader.tail8a7276.ts.net** (fixed domain from the
Tailscale Funnel; node `regimetrader` on the tailnet of `rpit.singh2000@`).

What this machine serves: the Streamlit dashboard on **127.0.0.1:8501** only, reached
through the Tailscale Funnel (outbound relay — no home-router ports are opened and no
other endpoint is funneled). Streamlit is bound to localhost, so even home-LAN devices
can only reach the app through the public HTTPS URL. The daily paper job writes to the
warehouse at 17:00 IST on weekdays; the dashboard is SELECT-only (`mode=ro`) and its
queries are short — SQLite handles one writer plus readers, and the two have already
run side by side without locking problems.

## Daily health check (30 seconds, from anywhere)

```
powershell -Command "(Invoke-WebRequest -UseBasicParsing https://<URL>/_stcore/health).Content"
# expect: ok
```
Then open the URL in a browser: six screens in the sidebar (market overview, OLAP
explorer, today's signals, backtest results, model diagnostics, Bot demo (replay)).
Toolbar/Deploy chrome hidden (`toolbarMode = viewer`).

## If it is DOWN — the checklist, in order

1. **Both feet in:** is the laptop on, plugged in, sitting at the login screen?
   No auto-login was configured: after any reboot someone must type the Windows
   password once, and everything else restarts by itself.
2. **Tunnel:** after login the funnel task re-runs on its own. If `/_stcore/health`
   from the URL fails while `localhost:8501/_stcore/health` works (browse locally),
   re-assert the funnel: `& "C:\Program Files\Tailscale\tailscale.exe" funnel --bg 127.0.0.1:8501`
3. **Dashboard task:** `Get-ScheduledTask -TaskName "*RegimeTrader*" | Format-Table TaskName, State`
   — all three should be `Ready/Running`. The dashboard launcher is a **supervisor
   loop**: it relaunches Streamlit ~8 s after any crash, indefinitely, so the usual
   cause of a dead page is the TASK being stopped — force a start with:
   `schtasks /run /tn RegimeTraderDashboard`
4. **Logs:** `logs\dash_service.out` (app; each crash logs a timestamped
   "streamlit exited ... relaunching" line), `logs\funnel_service.out` (tunnel),
   `logs\paper_daily.out` (paper job), `logs\paper\paper_live.log` (job's own log).
5. **Manual local start (bypasses the task entirely):**
   `.venv\Scripts\python.exe -m streamlit run src/dashboard/app.py --server.headless true --browser.gatherUsageStats false --server.port 8501 --server.address 127.0.0.1`

## Restart everything

```
Stop-Process -Name python -Force          # kills the dashboard (APP ONLY);
                                          # the supervisor relaunches it in ~8 s
schtasks /run /tn RegimeTraderFunnel      # re-asserts the funnel
```

## Reboot-survival (expected sequence — do not reboot without the operator's OK)

Reboot → Windows login screen → you type the password → at-log-on tasks fire:
dashboard (restart-on-failure covers later crashes) + funnel re-assert + jobs stay
scheduled. Verify with the health check URL above within ~1 minute of login.
The one gap (by your choice): an unattended automatic-reboot (Windows Update) leaves
the machine at login until you are back — see Hardening below.

## Hardening applied for the viva window (each undoable)

| Change | Undo |
|---|---|
| Lid close does nothing on AC (`powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` + `/setactive SCHEME_CURRENT`) | same command with `1` (Sleep) instead of `0` |
| Hibernate disabled, Fast Startup off (`powercfg /hibernate off`) | `powercfg /hibernate on` |
| Active hours 6→23 (was 8→17) so update reboots cannot land inside the viva day (`reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v ActiveHoursStart /t REG_DWORD /d 6 /f` and ActiveHoursEnd 23) | same `reg add` with `8` and `17` |
| Update installs don't auto-reboot a logged-on session (`reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /t REG_DWORD /d 1 /f`) | `reg delete ... /v NoAutoRebootWithLoggedOnUsers /f` |

Already true before any change (High Power scheme): sleep = never (AC/DC), hibernate
timeout = never, display-off = never on AC.

## Scheduled tasks (auto-start at log on)

| Task | What | Undo |
|---|---|---|
| `RegimeTraderDashboard` | Streamlit 127.0.0.1:8501 behind a supervisor loop (relaunch ~8 s after any death, indefinitely; tested live) | `schtasks /delete /tn RegimeTraderDashboard /f` |
| `RegimeTraderFunnel` | re-asserts `tailscale funnel --bg 127.0.0.1:8501` | `schtasks /delete /tn RegimeTraderFunnel /f` then `tailscale funnel off` |
| `RegimeTraderPaperJob` | daily job Mon–Fri 17:00 local (= IST), 2 h kill limit, catch-up if missed | `schtasks /delete /tn RegimeTraderPaperJob /f` |

Created by `deploy\host_setup.ps1` (idempotent; re-running replaces, never stacks).
Note on the scheduler restart policy: `RestartCount 999` exists on the task, but a
terminated instance does not count as a "failed" task in Task Scheduler semantics
(test verified: killed process did not trigger a scheduler restart), which is why the
supervisor loop — not the policy — is the actual recovery mechanism.

## What can still fail

- A Windows Update **forced reboot** while unattended → login screen until you return
  (auto-login was declined). Mitigation: pause updates in Settings for the viva week
  (UI — or I can script it with your OK).
- No internet at home (ISP outage) → funnel unreachable; local checks still work.
- Tailscale key/authorization expiry on the node → re-login once interactively.
- Laptop loses power completely → nothing self-heals until mains + a manual login.
- The bot-demo "Run the bot" uses a temp DB copy (~40 s); several concurrent
  visitors each copy the warehouse — fine on fiber, may feel slow on 4G. Same URL
  for everyone; no credential anywhere in the path.
