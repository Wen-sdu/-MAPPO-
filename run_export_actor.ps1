$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $PackageRoot ".venv_mappo\Scripts\python.exe"
$Script = Join-Path $PackageRoot "mappo_training\scripts\export_mappo_actor_to_header.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup_environment.ps1 before exporting an actor."
}

& $Python $Script
exit $LASTEXITCODE

