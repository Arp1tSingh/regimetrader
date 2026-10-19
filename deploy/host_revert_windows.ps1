# Reverts the viva-window hardening. Run from an ELEVATED PowerShell; prints proof.
#   powershell -ExecutionPolicy Bypass -File deploy\host_revert_windows.ps1

Start-Transcript -Path "C:\Users\Admin\Projects\DWM\regimetrader_pack\logs\host_revert.out" -Force

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Refusing: run from an ELEVATED PowerShell." -ForegroundColor Red
    Stop-Transcript
    exit 1
}

Write-Host "== 1. lid close -> Sleep (value 1) =="
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 1
powercfg /setactive SCHEME_CURRENT
powercfg /q SCHEME_CURRENT SUB_BUTTONS LIDACTION | Select-String "AC Power Setting Index"

Write-Host "`n== 2. hibernate back on =="
powercfg /hibernate on

Write-Host "`n== 3. active hours back to 8-17 =="
reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v ActiveHoursStart /t REG_DWORD /d 8 /f
reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v ActiveHoursEnd /t REG_DWORD /d 17 /f

Write-Host "`n== 4. remove no-auto-reboot policy =="
reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /f 2>&1

Write-Host "`n== 5. auto-login off =="
$reg = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"
Write-Host "   before: AutoAdminLogon = $((Get-ItemProperty $reg).AutoAdminLogon)"
Set-ItemProperty -Path $reg -Name AutoAdminLogon -Value "0" -ErrorAction Continue
Remove-ItemProperty -Path $reg -Name DefaultPassword  -ErrorAction Continue
Remove-ItemProperty -Path $reg -Name DefaultUserName  -ErrorAction Continue
Remove-ItemProperty -Path $reg -Name DefaultDomainName -ErrorAction Continue
Write-Host "   after:  AutoAdminLogon = $((Get-ItemProperty $reg).AutoAdminLogon)"
Write-Host "   residual keys: $((Get-ItemProperty $reg | Select-Object AutoAdminLogon, DefaultUserName, DefaultPassword, DefaultDomainName | Format-List | Out-String).Trim())"

Write-Host "`n== done =="
Stop-Transcript
exit 0
