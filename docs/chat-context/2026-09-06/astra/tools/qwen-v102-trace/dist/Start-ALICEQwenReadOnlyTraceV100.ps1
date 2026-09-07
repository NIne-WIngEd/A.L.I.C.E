param(
    [string]$VaultRoot = "C:\ALICE_Vault",
    [string]$OutputRoot = (Join-Path $HOME "Downloads")
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$Expected = "E8741E855B5BF2EC7A8F0CB782D7C9779654CDB5970327ECCD8FCBA39B60D39A"
$ZipBase = "ALICE_QWEN_READ_ONLY_TRACE_v1.0.0"
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
    $Candidates = @(Get-ChildItem -LiteralPath $PSScriptRoot -File -Filter "$ZipBase*.zip" | Sort-Object Name)
    $ZipPath = $null
    foreach ($Candidate in $Candidates) {
        $Actual = (Get-FileHash -LiteralPath $Candidate.FullName -Algorithm SHA256).Hash.ToUpperInvariant()
        if ($Actual -eq $Expected) { $ZipPath = $Candidate.FullName; break }
    }
    if ([string]::IsNullOrWhiteSpace($ZipPath)) { throw "The exact read-only diagnostic ZIP is missing beside this launcher, or its SHA-256 differs." }
    Write-Host "diagnostic_package_sha256=$Expected"
    $script:Python = Resolve-Python
    Write-Host "PYTHON=$script:Python"
    $TraceId = [Guid]::NewGuid().ToString("N").Substring(0,12)
    $TraceRoot = Join-Path $OutputRoot "ALICE_QwenTrace_$TraceId"
    if (Test-Path -LiteralPath $TraceRoot) { throw "Unexpected diagnostic folder collision." }
    New-Item -ItemType Directory -Path $TraceRoot | Out-Null
    Start-Transcript -Path (Join-Path $TraceRoot "terminal-transcript.txt") | Out-Null
    $script:TranscriptStarted = $true
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $TraceRoot
    $PackageRoot = Join-Path $TraceRoot $ZipBase
    $VerifyCode = @'
import sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
import collect_trace
count=collect_trace.verify_package()
files=sorted(root.glob('*.py'))
for p in files: compile(p.read_bytes(),str(p),'exec')
print('diagnostic_manifest_verified=true files='+str(count))
print('compile_gate_passed=true files='+str(len(files)))
'@
    Invoke-Python -Arguments @("-B","-c",$VerifyCode,$PackageRoot)
    if ($script:LastPythonExitCode -ne 0) { throw "Diagnostic verification failed." }
    Invoke-Python -Arguments @("-B","-m","unittest","discover","-s",$PackageRoot,"-p","selftest_trace.py","-v")
    if ($script:LastPythonExitCode -ne 0) { throw "Diagnostic selftest failed." }
    Write-Host "Reading the existing source and a2 state. No job submission, model download, inference, package staging or state repair."
    Invoke-Python -Arguments @("-B",(Join-Path $PackageRoot "collect_trace.py"),"--vault-root",$VaultRoot,"--output",$TraceRoot)
    $Code = $script:LastPythonExitCode
    Write-Host "ALICE_QWEN_READ_ONLY_TRACE_EXIT=$Code"
    Write-Host "Return the TRACE_ZIP printed above and this transcript. Do not rerun v102 to repair this stop."
    exit $Code
}
catch {
    Write-Host "ALICE_QWEN_READ_ONLY_TRACE_STOP: $($_.Exception.Message)" -ForegroundColor Yellow
    Write-Host "Return this output. This diagnostic does not change ALICE run state."
    exit 76
}
finally {
    if ($script:TranscriptStarted) { Stop-Transcript | Out-Null }
}
