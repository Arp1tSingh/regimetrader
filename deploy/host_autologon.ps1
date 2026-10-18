# Turn Windows auto-login ON for this machine (or OFF with -Undo).
# Run from an ELEVATED PowerShell. The password you type is stored by Windows in
# the Winlogon registry area (readable only by SYSTEM/administrators) - it never
# travels through any network or assistant.
#
# Requirements: this account must have a real password (a PIN-only account cannot
# auto-log into Windows; create a password first with `net user $env:USERNAME *`).
#
#   Enable :  powershell -ExecutionPolicy Bypass -File ...host_autologon.ps1
#   Undo   :  powershell -ExecutionPolicy Bypass -File ...host_autologon.ps1 -Undo

param(
    [switch] $Undo
)

$reg = "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon"

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Refusing: start an ELEVATED PowerShell (Run as administrator)." -ForegroundColor Red
    exit 1
}

if ($Undo) {
    Set-ItemProperty -Path $reg -Name AutoAdminLogon -Value "0"
    Remove-ItemProperty -Path $reg -Name DefaultPassword -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path $reg -Name DefaultUserName -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path $reg -Name DefaultDomainName -ErrorAction SilentlyContinue
    Write-Host "Auto-login disabled (registry back to empty)."
    exit 0
}

$user = $env:USERNAME
$secure = Read-Host -Prompt "Windows password for '$user'" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if ([string]::IsNullOrWhiteSpace($plain)) {
    Write-Host "Refusing: empty password. A PIN-only account cannot auto-login." -ForegroundColor Red
    exit 1
}

Set-ItemProperty -Path $reg -Name AutoAdminLogon  -Value "1"
Set-ItemProperty -Path $reg -Name DefaultUserName -Value $user
Set-ItemProperty -Path $reg -Name DefaultDomainName -Value $env:COMPUTERNAME
Set-ItemProperty -Path $reg -Name DefaultPassword -Value $plain

Write-Host ""
Write-Host "Auto-login ENABLED for '$user'. Test it on the next reboot:"
Write-Host "  reboot -> the desktop must appear WITHOUT a password prompt."
Write-Host ""
Write-Host "Undo any time:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"(this file)`" -Undo"
exit 0
