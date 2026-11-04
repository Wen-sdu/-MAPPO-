$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $PackageRoot ".venv_mappo\Scripts\python.exe"
$Script = Join-Path $PackageRoot "mappo_training\scripts\eval_mappo_deterministic.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup_environment.ps1 before evaluation."
}

& $Python $Script
exit $LASTEXITCODE

