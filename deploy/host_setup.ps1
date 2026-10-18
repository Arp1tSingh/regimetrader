# Creates the three host tasks for serving the RegimeTrader dashboard from this
# laptop at the viva. Idempotent: re-running replaces the tasks. Run from the repo
# as the logged-on user; the power/update hardening is kept OUT of this script on
# purpose (documented, one-command, undoable changes are applied with your OK).
#
#   Tasks (all trigger at log on, no stored password):
#     RegimeTraderDashboard - Streamlit on 127.0.0.1:8501, restart on failure
#     RegimeTraderFunnel    - re-asserts the Tailscale Funnel public entrypoint
#     RegimeTraderPaperJob  - the daily paper job, Mon-Fri 17:00 local (= IST here)
#
#   Undo (removes everything this script does):
#     schtasks /delete /tn RegimeTraderDashboard /f
#     schtasks /delete /tn RegimeTraderFunnel    /f
#     schtasks /delete /tn RegimeTraderPaperJob  /f

$ErrorActionPreference = "Stop"
$repo = "C:\Users\Admin\Projects\DWM\regimetrader_pack"
$ps = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

# Long-running services: effectively no time limit; crash is handled by restart,
# not by the scheduler's gun. Restart-on-failure: 999 attempts, 1 minute apart
# (-RestartCount > 0 is what switches Task Scheduler's restart policy on).
$settingsService = New-ScheduledTaskSettingsSet `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -AllowStartIfOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew

# The daily job must not be able to hang for days: killed after 2 hours, exit
# code 267011 (task terminated) = it failed = the runbook says to check the log.
$settingsJob = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -AllowStartIfOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew

function New-HostTask {
    param(
        [string] $Name,
        [object[]] $Trigger,
        [Microsoft.Management.Infrastructure.CimInstance] $Settings,
        [Microsoft.Management.Infrastructure.CimInstance] $Action
    )

    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "replaced  : $Name"
    }
    else {
        Write-Host "created   : $Name"
    }
    Register-ScheduledTask -TaskName $Name -Trigger $Trigger `
        -Settings $Settings -Action $Action | Out-Null
}

# 1. Dashboard: hidden, localhost only, survives crashes via restart-on-failure.
New-HostTask -Name "RegimeTraderDashboard" `
    -Trigger @(New-ScheduledTaskTrigger -AtLogOn) `
    -Settings $settingsService `
    -Action (New-ScheduledTaskAction -Execute $ps `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$repo\deploy\host_dashboard_start.ps1`"" `
        -WorkingDirectory $repo)

# 2. Tunnel: re-assert the funnel after boots (the tailscaled service keeps it up).
New-HostTask -Name "RegimeTraderFunnel" `
    -Trigger @(New-ScheduledTaskTrigger -AtLogOn) `
    -Settings $settingsService `
    -Action (New-ScheduledTaskAction -Execute $ps `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$repo\deploy\host_funnel_start.ps1`"" `
        -WorkingDirectory $repo)

# 3. Paper job: weekdays 17:00 local clock (machine timezone = India Standard Time).
New-HostTask -Name "RegimeTraderPaperJob" `
    -Trigger @(New-ScheduledTaskTrigger -Weekly -At 17:00 -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday) `
    -Settings $settingsJob `
    -Action (New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c `"cd /d $repo && .venv\Scripts\python.exe -m src.paper.daily_job --config config/config.yaml >> logs\paper_daily.out 2>&1`"" `
        -WorkingDirectory $repo)

Get-ScheduledTask -TaskName "RegimeTrader*" | Format-Table -AutoSize TaskName, State
Write-Host ""
Write-Host "Done. Start a task deliberately with:  schtasks /run /tn RegimeTraderDashboard"
