# ==============================================================================
# VITChennai_ZeroDrift - Track 3 (Path Finder) verification suite (PowerShell)
#
#   clean EuRoC / synthetic IMU -> official degrade_imu.py -> src/main.py
#   -> rule assertions (rows, heap <= 8 MB, <= 20 ms per 1 s of IMU, unit
#      quaternions, finite) -> evo_ape / evo_rpe -> plots/
#
# Thin wrapper around scripts/benchmark_all.py, run through uv.
#
#   powershell -ExecutionPolicy Bypass -File scripts/evaluate.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/evaluate.ps1 --sequences V1_01_easy
# ==============================================================================

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $RepoRoot

if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv sync --quiet
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    uv run --quiet python scripts/benchmark_all.py --fetch --publish @args
} else {
    Write-Host "[evaluate] uv not found - falling back to the current Python environment" -ForegroundColor Yellow
    $env:ZERODRIFT_NO_UV = "1"
    python scripts/benchmark_all.py --fetch --publish @args
}
exit $LASTEXITCODE
