$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $PackageRoot ".venv_mappo\Scripts\python.exe"
$Script = Join-Path $PackageRoot "mappo_training\scripts\verify_mappo_export.py"
$Actor = Join-Path $PackageRoot "mappo_training\results\mappo_mixed_ff32_500000_20261015_145758\run1\models\actor.pt"
$Header = Join-Path $PackageRoot "ccs_deployment\source\mappo_actor_weights.h"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Run setup_environment.ps1 before verifying an export."
}

& $Python $Script $Actor $Header
exit $LASTEXITCODE

