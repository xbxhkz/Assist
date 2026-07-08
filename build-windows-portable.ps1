#Requires -Version 5.1
<#
  Build a portable Windows distribution for Assist.

  Output layout:
    dist\Assist\Assist.exe
    dist\Assist\static\...
    dist\Assist\scripts\...
    dist\Assist\mcp_servers\...
    dist\Assist\services\hwfit\data\...

  The app then keeps using its normal filesystem layout when frozen.

  Usage:
    powershell -ExecutionPolicy Bypass -File .\build-windows-portable.ps1
    powershell -ExecutionPolicy Bypass -File .\build-windows-portable.ps1 -Fast
      (-Fast: skip pip install + asset fetches when build_assets exist, and
       run PyInstaller incrementally without --clean. Use for iteration;
       release builds stay full/clean.)
#>
param([switch]$Fast)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host ""; Write-Host ("==> " + $msg) -ForegroundColor Cyan }
function Fail($msg) {
    Write-Host ""
    Write-Host ("ERROR: " + $msg) -ForegroundColor Red
    exit 1
}

Write-Step "Checking for Python"
$pyExe = $null
if (Test-Path ".\.venv\Scripts\python.exe") {
    $pyExe = (Resolve-Path ".\.venv\Scripts\python.exe").Path
} else {
    foreach ($c in @("py", "python")) {
        $cmd = Get-Command $c -ErrorAction SilentlyContinue
        if ($cmd) { $pyExe = $cmd.Source; break }
    }
    if ($pyExe -like "*WindowsApps*python.exe") {
        $pyCmd = Get-Command py -ErrorAction SilentlyContinue
        if ($pyCmd) {
            $pyExe = $pyCmd.Source
        }
    }
}
if (-not $pyExe) {
    Fail "Python not found on PATH. Install Python 3.11+ first."
}
Write-Host ("Using Python: " + $pyExe)

if ($Fast -and (Test-Path "build_assets\llama") -and (Test-Path "build_assets\sd")) {
    Write-Step "Fast build: skipping pip install + asset fetches (assets present)"
} else {
    Write-Step "Installing build dependencies"
    & $pyExe -m pip install --upgrade pip --quiet
    & $pyExe -m pip install -r requirements.txt -r requirements-desktop.txt pyinstaller
    if ($LASTEXITCODE -ne 0) { Fail "Dependency install failed." }

    Write-Step "Vendoring offline embedding model"
    & $pyExe scripts/fetch_embedding_model.py
    if ($LASTEXITCODE -ne 0) { Fail "Embedding model fetch failed." }

    Write-Step "Vendoring llama-server (CPU + Vulkan)"
    & $pyExe scripts/fetch_llama_server.py
    if ($LASTEXITCODE -ne 0) { Fail "llama-server fetch failed." }

    Write-Step "Vendoring sd-server (CPU + Vulkan)"
    & $pyExe scripts/fetch_sd_server.py
    if ($LASTEXITCODE -ne 0) { Fail "sd-server fetch failed." }
}

Write-Step "Building portable exe bundle"
if (-not $Fast) { Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue }
$piArgs = @('--noconfirm'); if (-not $Fast) { $piArgs += '--clean' }
& $pyExe -m PyInstaller @piArgs Assist.spec
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller build failed." }

Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "Portable app folder: $PSScriptRoot\dist\Assist" -ForegroundColor Green
Write-Host "Distribute the whole folder (or zip it) so static assets and scripts stay with the exe." -ForegroundColor Green