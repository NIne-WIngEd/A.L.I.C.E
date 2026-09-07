param(
    [string]$OutputRoot = (Join-Path $HOME "Downloads")
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Expected = "DC53F84C67F8C286788C856FCFFB1BFAD189EB4541241D19A94BE797A40104A3"
$ZipBase = "ALICE_TELEMETRY_TRANSPORT_RECEIPT_v1.0.0"
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
    if ([string]::IsNullOrWhiteSpace($ZipPath)) { throw "The exact telemetry receipt v1.0.0 ZIP is missing beside this launcher, or its SHA-256 differs." }
    Write-Host "package_sha256=$Expected"
    $script:Python = Resolve-Python
    Write-Host "PYTHON=$script:Python"
    $ReceiptToken = [Guid]::NewGuid().ToString("N").Substring(0,12)
    $RunRoot = Join-Path $OutputRoot "ALICE_TelemetryReceipt_$ReceiptToken"
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
    Write-Host "Collecting three read-only Git transport receipts with the installed SSH wrapper. Return the ZIP even if Git fails."
    Invoke-Python -Arguments @((Join-Path $PackageRoot "collect_transport.py"),"--output-root",$RunRoot)
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_TELEMETRY_TRANSPORT_RECEIPT_EXIT=$Code"
    Write-Host "Return TRACE_ZIP and this transcript. Keep existing a2 state; do not rerun v104."
    exit $Code
}
catch {
    Write-Host "ALICE_TELEMETRY_TRANSPORT_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Return this output and any TRACE_ZIP already printed. No execution recovery is inferred."
    exit 74
}
finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
}
