$ErrorActionPreference = 'Stop'

$Owner = 'mkrayanyan'
$Slug = 'rayan-n0-cpu-lineage-v01'
$Kernel = "$Owner/$Slug"
$KernelDir = Join-Path $env:TEMP $Slug
$OutDir = Join-Path $HOME 'Downloads\RAYAN_N0_KAGGLE_CPU_OUTPUT'

if (Test-Path $KernelDir) { Remove-Item -Recurse -Force $KernelDir }
New-Item -ItemType Directory -Force -Path $KernelDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Runner = @'
import os
import shutil
import subprocess
from pathlib import Path

repo = Path('/kaggle/working/rayan-eipm-main')
if repo.exists():
    shutil.rmtree(repo)

subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'alice-eipm-v1-build',
    'https://github.com/NIne-WIngEd/A.L.I.C.E.git', str(repo)
], check=True)

subprocess.run([
    'bash', str(repo / 'scripts/eipm/n0/kaggle_cpu_real_lineage.sh')
], check=True)

for path in [
    Path('/kaggle/working/rayan-n0'),
    Path('/kaggle/working/hf-cache'),
    repo,
]:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

print('terminal_dispatch_cleanup=PASS')
'@

$Runner | Set-Content -Path (Join-Path $KernelDir 'runner.py') -Encoding UTF8

$Metadata = @{
    id = $Kernel
    title = 'Rayan N0 CPU Lineage v01'
    code_file = 'runner.py'
    language = 'python'
    kernel_type = 'script'
    is_private = $true
    enable_gpu = $false
    enable_internet = $true
    machine_shape = ''
    dataset_sources = @()
    competition_sources = @()
    kernel_sources = @()
    model_sources = @()
}

$Metadata | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $KernelDir 'kernel-metadata.json') -Encoding UTF8

Write-Host '===== RAYAN N0 KAGGLE CPU DIRECT DISPATCH ====='
Write-Host "kernel=$Kernel"
Write-Host "kernel_dir=$KernelDir"
Write-Host "output_dir=$OutDir"
Write-Host 'dispatch=official_kaggle_cli_direct'
Write-Host 'accelerator=none'
Write-Host 'internet=true'
Write-Host 'private_kernel=true'
Write-Host 'model_training=false'
Write-Host 'private_identity_data=false'

& kaggle --version
if ($LASTEXITCODE -ne 0) { throw 'Kaggle CLI is unavailable or not authenticated.' }

& kaggle kernels push -p $KernelDir
if ($LASTEXITCODE -ne 0) { throw 'kaggle kernels push failed.' }

Write-Host 'kernel_push=PASS'
Write-Host "kernel_ref=$Kernel"

while ($true) {
    Start-Sleep -Seconds 30
    $Status = (& kaggle kernels status $Kernel 2>&1 | Out-String).Trim()
    Write-Host "[$(Get-Date -Format s)] $Status"

    if ($Status -match 'COMPLETE') {
        break
    }
    if ($Status -match 'ERROR|FAILED|CANCELLED|CANCELED') {
        throw "Kaggle run failed: $Status"
    }
}

& kaggle kernels output $Kernel -p $OutDir -o
if ($LASTEXITCODE -ne 0) { throw 'kaggle kernels output failed.' }

Write-Host '===== RAYAN N0 KAGGLE CPU COMPLETE ====='
Write-Host "kernel=$Kernel"
Write-Host "output_dir=$OutDir"
Get-ChildItem -Recurse -File $OutDir | Select-Object FullName, Length
