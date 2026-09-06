$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Zip = "$HOME\Downloads\ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.6.zip"
$Expected = "F6EB40965272F9FBD423952345A183CF60D8C981DF7502FA83D07528EAF97F87"
$Freeze = "$HOME\Downloads\ALICE_MC10D_PREEXECUTION_FREEZE_BUNDLE_v1.1.zip"
$ExpectedFreeze = "22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3"
$Root = "$HOME\Downloads\ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.6_RUN"
$PackageRoot = Join-Path $Root "ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.6"
$Controller = Join-Path $PackageRoot "mc10d_public_judge_qualification_rebase_controller_v186.py"
$Selftest = Join-Path $PackageRoot "selftest_mc10d_public_judge_qualification_rebase_v186.py"
$script:LastPythonExitCode = 0
function Resolve-Python {
 $Candidates = New-Object System.Collections.Generic.List[string]
 $A = Join-Path $HOME "anaconda3\python.exe"
 if (Test-Path -LiteralPath $A -PathType Leaf) { $Candidates.Add($A) }
 if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) { $C = Join-Path $env:CONDA_PREFIX "python.exe"; if (Test-Path -LiteralPath $C -PathType Leaf) { $Candidates.Add($C) } }
 Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue | ForEach-Object { $_.Source } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_ -PathType Leaf) } | ForEach-Object { $Candidates.Add([string]$_) }
 $U=@($Candidates|Select-Object -Unique); if($U.Count -eq 0){throw "No usable python.exe found."}; return [string]$U[0]
}
function Invoke-Python([string[]]$Arguments) { $Old=$ErrorActionPreference; try{$ErrorActionPreference="Continue";& $script:Python @Arguments;$script:LastPythonExitCode=[int]$LASTEXITCODE} finally{$ErrorActionPreference=$Old} }
if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) { throw "Missing package: $Zip" }
if (-not (Test-Path -LiteralPath $Freeze -PathType Leaf)) { throw "Missing freeze: $Freeze" }
$Actual=(Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToUpperInvariant();$FreezeHash=(Get-FileHash -LiteralPath $Freeze -Algorithm SHA256).Hash.ToUpperInvariant();Write-Host "package_sha256=$Actual";Write-Host "freeze_sha256=$FreezeHash";if($Actual-ne$Expected){throw "Package SHA mismatch."};if($FreezeHash-ne$ExpectedFreeze){throw "Freeze SHA mismatch."}
Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue;Expand-Archive -LiteralPath $Zip -DestinationPath $Root -Force
$script:Python=Resolve-Python;Write-Host "PYTHON=$script:Python"
$PythonFiles=@(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Filter "*.py"|Sort-Object FullName);foreach($F in $PythonFiles){Invoke-Python -Arguments @("-m","py_compile",$F.FullName);if($script:LastPythonExitCode-ne0){throw "Compile failed before live work."}};Write-Host "python_compile_gate_passed=true files=$($PythonFiles.Count)"
Invoke-Python -Arguments @($Selftest);if($script:LastPythonExitCode-ne0){throw "v1.8.6 selftest failed. No live work started."}
Write-Host "v1.8.6 first ratifies the GLM thinking-off runtime profile from the completed Q04 diagnostic, then runs one fresh full GLM family."
Invoke-Python -Arguments @($Controller)
$Code=$script:LastPythonExitCode
if($Code-eq0){Write-Host "MC10D_V186_EXIT=0" -ForegroundColor Green;exit 0}
if($Code-eq75){Write-Host "MC10D_V186_EXIT=75" -ForegroundColor Yellow;Write-Host "Safe reconciliation pause. Preserve state. Do not blind-rerun." -ForegroundColor Yellow;exit 75}
if($Code-eq76){Write-Host "MC10D_V186_EXIT=76" -ForegroundColor Yellow;Write-Host "Deterministic stop. Preserve state. Do not blind-rerun." -ForegroundColor Yellow;exit 76}
Write-Host "MC10D_V186_EXIT=$Code" -ForegroundColor Red;exit $Code
