param(
    [string]$VaultRoot = "C:\ALICE_Vault",
    [string]$RepoRoot = "C:\A.L.I.C.E-main",
    [string]$OutputRoot = (Join-Path $HOME "Downloads")
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Expected = "6A20AC96AD24300939478D704FBA6F56C1143CCDF5C6F91CEF5B4A130B8B6722"
$ZipBase = "ALICE_MC10D_V186_OFFLINE_FORENSICS_v1.0.0"
$PackageName = $ZipBase
$script:LastPythonExitCode = 0

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
    if ([string]::IsNullOrWhiteSpace($ZipPath)) { throw "The exact v1.0.0 ZIP is missing beside this launcher, or its SHA-256 differs." }
    Write-Host "package_sha256=$Expected"
    $script:Python = Resolve-Python
    Write-Host "PYTHON=$script:Python"
    $RunId = [Guid]::NewGuid().ToString("N").Substring(0,12)
    $RunRoot = Join-Path $OutputRoot "ALICE_AstraF100_RUN_$RunId"
    if (Test-Path -LiteralPath $RunRoot) { throw "Unexpected run-folder collision." }
    New-Item -ItemType Directory -Path $RunRoot | Out-Null
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
    Invoke-Python -Arguments @((Join-Path $PackageRoot "selftest_forensics.py"))
    if ($script:LastPythonExitCode -ne 0) { throw "Offline selftest failed. No source evidence was changed." }
    Write-Host "Collecting existing public GLM output. Remote compute jobs: zero. Git publication follows collection."
    Invoke-Python -Arguments @((Join-Path $PackageRoot "mc10d_forensics.py"),"--vault-root",$VaultRoot,"--output-root",$OutputRoot,"--repo-root",$RepoRoot,"--publish-context")
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_ASTRA_FORENSICS_V100_EXIT=$Code"
    if ($Code -eq 75) { Write-Host "Evidence ZIP preserved. Return it and this output; only context publication remains pending." -ForegroundColor Yellow }
    if ($Code -eq 76) { Write-Host "Return this output. Do not rerun v186 or breadth v104 to work around this stop." -ForegroundColor Yellow }
    exit $Code
}
catch {
    Write-Host "ALICE_ASTRA_FORENSICS_V100_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "No new compute should be launched. Keep this output for review."
    exit 76
}
