#Requires -Version 5.1
<#
  Build the Assist Windows installer. Runs the portable build first, then
  compiles installer\Assist.iss into installer\Output\Assist-Setup.exe.
  Requires Inno Setup (ISCC.exe) on PATH or at its default location.
#>
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host ""; Write-Host ("==> " + $msg) -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host ("ERROR: " + $msg) -ForegroundColor Red; exit 1 }

Write-Step "Building portable app folder"
& powershell -ExecutionPolicy Bypass -File .\build-windows-portable.ps1
if ($LASTEXITCODE -ne 0) { Fail "Portable build failed." }

Write-Step "Resolving version from src/constants.py (APP_VERSION)"
$verLine = Select-String -Path "src\constants.py" -Pattern 'APP_VERSION\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $verLine) { Fail "Could not read APP_VERSION from src/constants.py." }
$version = $verLine.Matches[0].Groups[1].Value
Write-Host ("Version: " + $version)

Write-Step "Locating Inno Setup (ISCC.exe)"
$iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $c) { $iscc = $c; break }
    }
}
if (-not $iscc) { Fail "Inno Setup (ISCC.exe) not found. Install Inno Setup 6." }

Write-Step "Compiling installer"
& $iscc "/DMyAppVersion=$version" "installer\Assist.iss"
if ($LASTEXITCODE -ne 0) { Fail "Inno Setup compile failed." }

Write-Host ""
Write-Host "Installer built: $PSScriptRoot\installer\Output\Assist-Setup.exe" -ForegroundColor Green
