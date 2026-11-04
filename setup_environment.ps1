$ErrorActionPreference = "Stop"
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvRoot = Join-Path $PackageRoot ".venv_mappo"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Requirements = Join-Path $PackageRoot "requirements_dc_microgrid.txt"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.10 -m venv $VenvRoot
    }
    else {
        $SystemPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $SystemPython) {
            throw "Python 3.10 was not found. Install Python 3.10 and retry."
        }
        & $SystemPython.Source -m venv $VenvRoot
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r $Requirements
Write-Host "Environment ready: $VenvPython"

