# Pulls a consistent backup of the RegimeTrader free-VM state to the laptop.
#
#   .\deploy\pull_backup.ps1                # defaults: -HostAlias rt-vm
#   .\deploy\pull_backup.ps1 -HostAlias my-vm -Keep 12
#
# What it does (see deploy/OPTION_B_FREE.md section 8):
#   1. asks the server for a consistent SQLite snapshot (deploy/db_snapshot.sh),
#   2. copies down the warehouse snapshot, reports/daily/ and logs/,
#   3. prunes old backups, keeping the newest -Keep ones.
#
# Schedule weekly (Tue 09:00):
#   schtasks /Create /TN RegimeTraderWeeklyBackup /SC WEEKLY /D TUE /ST 09:00 /TR ^
#     "powershell -ExecutionPolicy Bypass -File C:\Users\Admin\Projects\DWM\regimetrader_pack\deploy\pull_backup.ps1"

[CmdletBinding()]
param(
    # Alias from ~/.ssh/config created in OPTION_B_FREE.md step 2
    [string]$HostAlias = "rt-vm",

    # Repo directory on the server, relative to the SSH user's home
    [string]$RemoteDir = "regimetrader",

    # Keep this many timestamped backup folders
    [int]$Keep = 8
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path $PSScriptRoot -Parent
$root = Join-Path $repoRoot "backups"
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$dest = Join-Path $root $stamp
$null = New-Item -ItemType Directory -Force -Path $dest

Write-Host "== RegimeTrader backup $stamp (host: $HostAlias) =="

Write-Host "[1/3] server-side snapshot..."
ssh $HostAlias "cd $RemoteDir && bash deploy/db_snapshot.sh data/warehouse/_snapshot.db"
if ($LASTEXITCODE -ne 0) { throw "remote db_snapshot.sh failed (exit $LASTEXITCODE)" }

Write-Host "[2/3] copying down..."
scp "${HostAlias}:$RemoteDir/data/warehouse/_snapshot.db" (Join-Path $dest "regimetrader.db")
if ($LASTEXITCODE -ne 0) { throw "scp of the warehouse snapshot failed" }
scp -r "${HostAlias}:$RemoteDir/reports/daily" (Join-Path $dest "daily")
scp -r "${HostAlias}:$RemoteDir/logs" (Join-Path $dest "logs")
if ($LASTEXITCODE -ne 0) { throw "scp of daily reports/logs failed" }

Write-Host "[3/3] cleaning remote temp snapshot..."
ssh $HostAlias "rm -f $RemoteDir/data/warehouse/_snapshot.db"

# Prune old backups, newest first survive
Get-ChildItem $root -Directory |
    Sort-Object Name -Descending |
    Select-Object -Skip $Keep |
    Remove-Item -Recurse -Force

$db = Get-Item (Join-Path $dest "regimetrader.db")
Write-Host "backup OK: $($db.FullName) ($([math]::Round($db.Length / 1MB, 1)) MB)"
Write-Host "kept the newest $Keep backups under: $root"
