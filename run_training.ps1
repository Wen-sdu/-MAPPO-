$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $PackageRoot ".venv_mappo\Scripts\python.exe"
$Script = Join-Path $PackageRoot "mappo_training\start_mappo_train.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup_environment.ps1 before training."
}

& $Python $Script --no-pause
exit $LASTEXITCODE

