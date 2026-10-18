# Hidden launcher + SUPERVISOR for the public dashboard task (RegimeTraderDashboard).
# Runs Streamlit bound to localhost ONLY; the only way in is the Tailscale Funnel.
#
# The outer loop is the restart policy: if the process dies for any reason - crash,
# port conflict, kill - it is relaunched in ~8 seconds, indefinitely. A scheduled
# task never has to restart anything; it simply always has a live child.

Set-Location "C:\Users\Admin\Projects\DWM\regimetrader_pack"
$python = Join-Path (Get-Location) ".venv\Scripts\python.exe"
$log    = Join-Path (Get-Location) "logs\dash_service.out"

while ($true) {
    Add-Content -Path $log -Value "$(Get-Date -Format s) streamlit starting"
    & "$python" -m streamlit run src/dashboard/app.py `
        --server.headless true `
        --browser.gatherUsageStats false `
        --server.port 8501 `
        --server.address 127.0.0.1 `
        *>> "$log"
    Add-Content -Path $log -Value "$(Get-Date -Format s) streamlit exited code=$LASTEXITCODE - relaunching in 8 s"
    Start-Sleep -Seconds 8
}
