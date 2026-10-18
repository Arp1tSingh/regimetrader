# ONE-TIME elevated setup for the laptop-as-host plan. This is the only step that
# needs administrator rights: (1) the three scheduled tasks, (2) the lid/hibernate
# hardening, (3) Windows Update reboot guard. Everything it does is idempotent and
# every change is listed with its undo in deploy\HOST_RUNBOOK.md.
#
#   Run as admin :  powershell -ExecutionPolicy Bypass -File ...host_elevated_setup.ps1
#   Undo         :  the commands in HOST_RUNBOOK.md's table

Start-Transcript -Path "C:\Users\Admin\Projects\DWM\regimetrader_pack\logs\host_elevated_setup.out" -Force

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Refusing: run from an ELEVATED PowerShell." -ForegroundColor Red
    Stop-Transcript
    exit 1
}

Write-Host "`n== 1/3 scheduled tasks =="
& powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Admin\Projects\DWM\regimetrader_pack\deploy\host_setup.ps1"

Write-Host "`n== 2/3 power: lid close = do nothing (AC), hibernate off =="
# Undo: powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1 ; powercfg /setactive SCHEME_CURRENT ; powercfg /hibernate on
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT
powercfg /hibernate off

Write-Host "`n== 3/3 Windows Update: active hours 6-23, no auto-reboot under a logged-on session =="
# Undo: same reg add with 8 and 17; and: reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /f
reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v ActiveHoursStart /t REG_DWORD /d 6 /f
reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v ActiveHoursEnd /t REG_DWORD /d 23 /f
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /t REG_DWORD /d 1 /f

Write-Host "`nElevated setup complete."
Stop-Transcript
exit 0
