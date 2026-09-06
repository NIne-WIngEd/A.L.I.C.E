$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = "C:\A.L.I.C.E-main"
$Zip = "$HOME\Downloads\ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.4.zip"
$Expected = "C8E60D77F66C33AB73C577C2CB4A1BB707AA810C81DCAED0CA04B6B0443CD0EC"
$Freeze = "$HOME\Downloads\ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip"
$ExpectedFreeze = "22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3"
$Root = "$HOME\Downloads\ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.4_RUN"
$PackageRoot = Join-Path $Root "ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.4"
$Controller = Join-Path $PackageRoot "mc10d_controlled_synthesis_breadth_controller_v104.py"
$Selftest = Join-Path $PackageRoot "selftest_mc10d_controlled_synthesis_breadth_v104.py"
$script:LastPythonExitCode = 0

function Resolve-Python {
    $Candidates = New-Object System.Collections.Generic.List[string]
    $A = Join-Path $HOME "anaconda3\python.exe"
    if (Test-Path -LiteralPath $A -PathType Leaf) { $Candidates.Add($A) }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $C = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $C -PathType Leaf) { $Candidates.Add($C) }
    }
    Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Source } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        ForEach-Object { $Candidates.Add([string]$_) }
    $Unique = @($Candidates | Select-Object -Unique)
    if ($Unique.Count -eq 0) { throw "No usable python.exe found." }
    return [string]$Unique[0]
}

function Invoke-Python([string[]]$Arguments) {
    $Old = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $script:Python @Arguments
        $script:LastPythonExitCode = [int]$LASTEXITCODE
    }
    finally { $ErrorActionPreference = $Old }
}

if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) { throw "Missing package: $Zip" }
if (-not (Test-Path -LiteralPath $Freeze -PathType Leaf)) { throw "Missing freeze: $Freeze" }
$Actual = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToUpperInvariant()
$FreezeHash = (Get-FileHash -LiteralPath $Freeze -Algorithm SHA256).Hash.ToUpperInvariant()
Write-Host "package_sha256=$Actual"
Write-Host "freeze_sha256=$FreezeHash"
if ($Actual -ne $Expected) { throw "Package SHA mismatch." }
if ($FreezeHash -ne $ExpectedFreeze) { throw "Freeze SHA mismatch." }
Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $Zip -DestinationPath $Root -Force
$script:Python = Resolve-Python
Write-Host "PYTHON=$script:Python"
$PythonFiles = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Filter "*.py" | Sort-Object FullName)
foreach ($File in $PythonFiles) { Invoke-Python -Arguments @("-m","py_compile",$File.FullName); if ($script:LastPythonExitCode -ne 0) { throw "Compile failed. No installation performed." } }
Write-Host "python_compile_gate_passed=true files=$($PythonFiles.Count)"
Invoke-Python -Arguments @($Selftest)
if ($script:LastPythonExitCode -ne 0) { throw "Breadth v1.0.4 selftest failed. No installation performed." }
Write-Host "Ratifying only the controlled synthesis-breadth policy after verified v1.8.6 success. No pointwise or candidate execution occurs here."
Invoke-Python -Arguments @($Controller,"--repo-root",$RepoRoot,"--freeze-bundle",$Freeze,"--owner-ratify")
$Code=$script:LastPythonExitCode
if ($Code -eq 0) { Write-Host "MC10D_CONTROLLED_SYNTHESIS_BREADTH_V104_EXIT=0" -ForegroundColor Green; exit 0 }
if ($Code -eq 76) { Write-Host "MC10D_CONTROLLED_SYNTHESIS_BREADTH_V104_EXIT=76" -ForegroundColor Yellow; Write-Host "Deterministic governance stop. Do not bypass." -ForegroundColor Yellow; exit 76 }
Write-Host "MC10D_CONTROLLED_SYNTHESIS_BREADTH_V104_EXIT=$Code" -ForegroundColor Red
exit $Code
