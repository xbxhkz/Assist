#Requires -Version 5.1
<#
  Copy the repo's static\ tree into the installed app so UI-only changes are
  testable without a 30-minute rebuild. Restart Assist afterwards to load them.
  Self-elevates (writing under Program Files needs admin).
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$dest = "C:\Program Files\Assist\_internal\static"
if (-not (Test-Path $dest)) {
    Write-Host "ERROR: $dest not found (is Assist installed?)" -ForegroundColor Red
    exit 1
}
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
$admin = (New-Object Security.Principal.WindowsPrincipal($id)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    $shell = if (Get-Command pwsh -ErrorAction SilentlyContinue) { "pwsh" } else { "powershell" }
    try {
        Start-Process -FilePath $shell -Verb RunAs -Wait -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "$PSScriptRoot\dev-sync-static.ps1")
    } catch {
        Write-Host "Elevation declined - static was NOT synced." -ForegroundColor Yellow
        exit 1
    }
    exit $LASTEXITCODE
}
robocopy "$repo\static" $dest /E /NJH /NJS /NDL /NP | Out-Null
# robocopy exit codes < 8 mean success (0 = nothing to copy, 1-7 = copied).
if ($LASTEXITCODE -ge 8) {
    Write-Host "robocopy failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
Write-Host "static synced to $dest - restart Assist to load the changes." -ForegroundColor Green
exit 0
