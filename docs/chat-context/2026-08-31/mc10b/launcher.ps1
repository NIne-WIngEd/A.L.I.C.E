# A.L.I.C.E. MC10B Full E-INF Frontier v1.1.0
[CmdletBinding()]
param(
    [string]$RepoRoot = 'C:\A.L.I.C.E-main',
    [string]$VaultRoot = 'C:\ALICE_Vault',
    [ValidateRange(120,600)][int]$MaxGenerationMinutes = 480,
    [ValidateRange(1,14)][int]$TimeoutHours = 12,
    [ValidateRange(10,300)][int]$PollSeconds = 30,
    [int[]]$AcknowledgeTelemetryBlock = @(),
    [switch]$KeepRemote
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Controller = Join-Path $PSScriptRoot 'mc10b_full_controller.py'
if (-not (Test-Path -LiteralPath $Controller -PathType Leaf)) {
    throw "Controller missing: $Controller"
}
$Python = (Get-Command python -ErrorAction Stop).Source
$ArgsList = @(
    $Controller,
    '--repo-root', $RepoRoot,
    '--vault-root', $VaultRoot,
    '--max-generation-minutes', [string]$MaxGenerationMinutes,
    '--timeout-hours', [string]$TimeoutHours,
    '--poll-seconds', [string]$PollSeconds
)
foreach ($Block in @($AcknowledgeTelemetryBlock | Sort-Object -Unique)) {
    $ArgsList += @('--acknowledge-telemetry-block', [string]$Block)
}
if ($KeepRemote) {
    $ArgsList += '--keep-remote'
}
& $Python @ArgsList
if ($LASTEXITCODE -ne 0) {
    throw "MC10B v1.1.0 controller exited with code $LASTEXITCODE"
}
