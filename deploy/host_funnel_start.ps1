# Hidden launcher for the stable public tunnel task (RegimeTraderFunnel).
# Re-asserts the Tailscale Funnel after every log on; the config persists in tailscaled
# state, this only restores it after a boot/restart. Outbound-only: opens no ports.

$ts = "C:\Program Files\Tailscale\tailscale.exe"
$log = "C:\Users\Admin\Projects\DWM\regimetrader_pack\logs\funnel_service.out"

if (-not (Test-Path $ts)) {
    Add-Content -Path $log -Value "$(Get-Date -Format s) tailscale.exe not found - run make-host setup again"
    exit 1
}

& "$ts" funnel --bg 127.0.0.1:8501 *>> "$log"
exit $LASTEXITCODE
