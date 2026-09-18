param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$CodeOnly
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required."
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it first, then rerun this script."
}

$branch = (git branch --show-current).Trim()
if (-not $branch) {
    throw "Could not determine the current Git branch."
}

Write-Host "repo=$RepoRoot"
Write-Host "branch=$branch"

$installed = Get-Command graphify -ErrorAction SilentlyContinue
if (-not $installed) {
    uv tool install graphifyy
}

if ($CodeOnly) {
    graphify extract . --code-only
}
else {
    graphify extract .
}

if (Test-Path "graphify-out\graph.json") {
    Write-Host "graph=graphify-out\graph.json"
} else {
    throw "Graphify completed without graphify-out\graph.json."
}

Write-Host ""
Write-Host "Suggested validation queries:"
Write-Host '  graphify query "Where is EIPM personality-model N0 now and what sources establish that state?"'
Write-Host '  graphify query "What Magnolia failures and execution lessons are relevant to current N0 work?"'
Write-Host '  graphify query "Which decisions constrain the current personality-model architecture and which older decisions were superseded?"'
Write-Host ""
Write-Host "Do not treat graph output as authority. Verify material claims in original sources."
