$ErrorActionPreference = 'Continue'

$Controller = Join-Path $HOME 'Downloads\rayan_n0_kaggle_cpu_lineage_controller_v102.py'

if (-not (Test-Path -LiteralPath $Controller)) {
    Write-Error "Missing controller: $Controller"
    exit 2
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $PythonCommand) {
    Write-Error 'Python was not found on PATH. Run this from the same Anaconda/base PowerShell used for prior A.L.I.C.E. Kaggle jobs.'
    exit 2
}

$Python = $PythonCommand.Source
Write-Host '===== RAYAN N0 KAGGLE CPU LAUNCHER v1.0.2 ====='
Write-Host "PYTHON=$Python"
Write-Host "CONTROLLER=$Controller"
Write-Host 'powershell_json_state=false'
Write-Host 'powershell_native_cli_state_machine=false'
Write-Host 'python_owns_json_state_and_kaggle_cli=true'

& $Python $Controller
$Code = $LASTEXITCODE

Write-Host "RAYAN_N0_KAGGLE_CPU_V102_EXIT=$Code"
exit $Code
