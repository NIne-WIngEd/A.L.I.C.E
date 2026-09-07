param(
    [string]$VaultRoot = "C:\ALICE_Vault",
    [string]$RepoRoot = "C:\A.L.I.C.E-main",
    [string]$OutputRoot = (Join-Path $HOME "Downloads")
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Expected = "98C8C5DC625FC04235ED1046DEEC94A016AD890B1DDD920E1E8556F113685A9F"
$ZipBase = "ALICE_MC10D_QWEN_TELEMETRY_SUCCESSOR_v1.0.5"
$PackageName = $ZipBase
$script:LastPythonExitCode = 0
$script:TranscriptStarted = $false

function Resolve-Python {
    $Candidates = New-Object System.Collections.Generic.List[string]
    $Anaconda = Join-Path $HOME "anaconda3\python.exe"
    if (Test-Path -LiteralPath $Anaconda -PathType Leaf) { $Candidates.Add($Anaconda) }
    if (-not [string]::IsNullOrWhiteSpace($env:CONDA_PREFIX)) {
        $CondaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $CondaPython -PathType Leaf) { $Candidates.Add($CondaPython) }
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
    $OldPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $script:Python @Arguments
        $script:LastPythonExitCode = [int]$LASTEXITCODE
    }
    finally { $ErrorActionPreference = $OldPreference }
}

try {
    # Duplicate download filenames are allowed only when their exact bytes match.
    $Candidates = @(Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter "$ZipBase*.zip" | Sort-Object Name)
    $ZipPath = $null
    foreach ($Candidate in $Candidates) {
        $Actual = (Get-FileHash -LiteralPath $Candidate.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($Actual -eq $Expected) { $ZipPath = $Candidate.FullName; break }
    }
    if ([string]::IsNullOrWhiteSpace($ZipPath)) { throw "The exact v1.0.5 ZIP is missing beside this launcher, or its SHA-256 differs." }
    Write-Host "package_sha256=$Expected"
    $script:Python = Resolve-Python
    Write-Host "PYTHON=$script:Python"
    $RunId = [Guid]::NewGuid().ToString("N").Substring(0,12)
    $RunRoot = Join-Path $OutputRoot "ALICE_QwenA3_RUN_$RunId"
    if (Test-Path -LiteralPath $RunRoot) { throw "Unexpected run-folder collision." }
    New-Item -ItemType Directory -Path $RunRoot | Out-Null
    Start-Transcript -Path (Join-Path $RunRoot "terminal-transcript.txt") | Out-Null
    $script:TranscriptStarted = $true
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $RunRoot
    $PackageRoot = Join-Path $RunRoot $PackageName
    $VerifyCode = @'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path=(root/entry['path']).resolve()
    assert path.is_relative_to(root) and path.is_file(), 'manifest path invalid'
    data=path.read_bytes()
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'], 'package content hash drift: '+entry['path']
print('package_manifest_verified=true files='+str(len(manifest['files'])))
'@
    Invoke-Python -Arguments @("-c",$VerifyCode,$PackageRoot)
    if ($script:LastPythonExitCode -ne 0) { throw "Manifest verification failed." }
    $PythonFiles = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Filter "*.py" | Sort-Object FullName)
    foreach ($File in $PythonFiles) {
        Invoke-Python -Arguments @("-m","py_compile",$File.FullName)
        if ($script:LastPythonExitCode -ne 0) { throw "Compile gate failed." }
    }
    Write-Host "python_compile_gate_passed=true files=$($PythonFiles.Count)"
    Invoke-Python -Arguments @("-m","unittest","discover","-s",$PackageRoot,"-p","selftest*.py","-v")
    if ($script:LastPythonExitCode -ne 0) { throw "Offline selftest failed. No source evidence was changed." }
    Write-Host "One explicit a3 execution of the approved public calibration. Preserve closed a1/a2. Verify both parents and the installed publisher before submission; retain raw Git receipts. Magnolia CPU: 20 CPUs, 48 GiB, six-hour limit. No inference retries or automatic successor."
    Invoke-Python -Arguments @((Join-Path $PackageRoot "controller.py"),"--package-zip",$ZipPath,"--package-sha",$Expected.ToLowerInvariant(),"--vault-root",$VaultRoot,"--output-root",$OutputRoot,"--repo-root",$RepoRoot)
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_ASTRA_QWEN_V105_EXIT=$Code"
    if ($Code -eq 74) { Write-Host "Connection or monitoring pending. Rerun this launcher to attach to the same job." -ForegroundColor Yellow }
    if ($Code -eq 75) { Write-Host "Evidence preserved. Rerun this launcher to retry context or terminal telemetry publication." -ForegroundColor Yellow }
    if ($Code -eq 76) { Write-Host "Return the result ZIP if produced and this transcript. Keep the recorded run state; do not switch models or retry old stages." -ForegroundColor Yellow }
    exit $Code
}
catch {
    Write-Host "ALICE_ASTRA_QWEN_V105_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Keep this output for review. A previously submitted job may continue under its six-hour limit; this launcher can reattach."
    exit 76
}

finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
}
