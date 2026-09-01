# A.L.I.C.E. Kaggle Remote Compute — Operational Handoff

**Status:** VALIDATED / ACTIVE  
**Current version:** `alice-kaggle v0.2`  
**Primary remote GPU backend:** Kaggle  
**Validation date:** 2026-08-23

---

## 1. Purpose

This document is the handoff/source-of-truth for future A.L.I.C.E. chats that need to use Kaggle as the active remote GPU compute backend.

The previous RTX 3050 SSH worker is retired and should not be selected.

Current architecture:

```text
LOCAL WINDOWS
├── C:\A.L.I.C.E-main
├── C:\ALICE_Vault
├── models / databases / datasets
└── alice-kaggle
        │
        │ Kaggle CLI
        ▼
PRIVATE KAGGLE DATASET
        │
        ▼
PRIVATE KAGGLE SCRIPT KERNEL
        │
        ├── GPU
        ├── Kaggle RAM
        ├── /tmp execution workspace
        ├── /kaggle/working/output retained outputs
        └── internet enabled by default
        │
        ▼
RESULTS DOWNLOADED TO WINDOWS
        │
        ├── private kernel deleted
        └── private dataset deleted
```

Kaggle is treated as a trusted remote workspace for A.L.I.C.E.

---

## 2. Trust policy

The old shared-PC hardening policy does **not** apply to Kaggle.

For Kaggle, all of the following are allowed when useful:

```text
full A.L.I.C.E. repo
vault/private data
private corpora
memory data
model weights
databases
checkpoints
intermediate artifacts
temporary Kaggle persistence
internet access
```

The following old RTX-3050 requirements are retired for Kaggle:

```text
MemorySwapMax=0
RAM-only staging
read-only remote disk
generic anonymous paths
zero-residue audit
no A.L.I.C.E. names remotely
no persistent remote state
```

Operational default is still to delete temporary Kaggle resources after each job to avoid clutter.

---

## 3. Kaggle account/config

Current Kaggle username:

```text
mkrayanyan
```

Kaggle CLI authentication has been validated.

Local config:

```text
C:\Users\rayns\.alice-kaggle\config.json
```

Current config validated as:

```text
username:    mkrayanyan
accelerator: NvidiaTeslaT4
internet:    True
project:     C:\A.L.I.C.E-main
```

Kaggle CLI version validated:

```text
Kaggle CLI 2.2.4
```

Doctor check:

```text
auth:        OK
doctor:      PASS
```

---

## 4. Local implementation files

Current active files:

```text
C:\A.L.I.C.E-main\tools\alice_kaggle_v0.py
C:\A.L.I.C.E-main\tools\alice-kaggle.ps1
C:\A.L.I.C.E-main\tools\kaggle-smoke-test.py
```

PowerShell convenience function:

```powershell
function alice-kaggle {
    & "C:\A.L.I.C.E-main\tools\alice-kaggle.ps1" @args
}
```

Primary command:

```powershell
alice-kaggle ...
```

---

## 5. PowerShell execution policy note

If PowerShell blocks:

```text
alice-kaggle.ps1 is not digitally signed
```

for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
```

Recommended permanent user-level setting:

```powershell
Unblock-File C:\A.L.I.C.E-main\tools\alice-kaggle.ps1
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
```

Check with:

```powershell
Get-ExecutionPolicy -List
```

---

## 6. Setup commands

Kaggle CLI is already installed and authenticated.

If revalidation is ever needed:

```powershell
kaggle auth login
```

Then:

```powershell
alice-kaggle -Setup -Username mkrayanyan
```

Then:

```powershell
alice-kaggle -Doctor
```

Expected:

```text
auth:        OK
doctor:      PASS
```

---

## 7. Basic usage

### Run a script

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py
```

### Run a module

```powershell
alice-kaggle `
    -Module alice_memory.some_module
```

### Specify output directory

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -OutputDir C:\A.L.I.C.E-main\remote-results\kaggle\some-job
```

### Pass script arguments

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -Argument @(
        "--batch-size",
        "64",
        "--epochs",
        "5"
    )
```

### Upload external A.L.I.C.E. data

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -InputPath @(
        "C:\ALICE_Vault",
        "C:\Models\some-model"
    )
```

### Install project requirements

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -Requirements C:\A.L.I.C.E-main\requirements-phase1.txt
```

### Install arbitrary packages

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -Pip @(
        "transformers",
        "sentence-transformers"
    )
```

### Require both GPUs

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\scripts\some_job.py `
    -RequireGpus 2
```

---

## 8. Remote environment variables

A.L.I.C.E.-aware code should use:

```text
ALICE_KAGGLE_JOB
ALICE_KAGGLE_REPO
ALICE_KAGGLE_INPUT
ALICE_KAGGLE_OUTPUT
```

Compatibility aliases also exist:

```text
RW_REPO
RW_INPUT
RW_OUTPUT
```

Example:

```python
import os
from pathlib import Path

repo = Path(os.environ["ALICE_KAGGLE_REPO"])
inp = Path(os.environ["ALICE_KAGGLE_INPUT"])
out = Path(os.environ["ALICE_KAGGLE_OUTPUT"])

out.mkdir(parents=True, exist_ok=True)
```

Anything that must return to Windows should be written under:

```text
ALICE_KAGGLE_OUTPUT
```

---

## 9. Kaggle runtime paths

v0.2 uses:

```text
/tmp/alice-kaggle-job
```

for execution scratch space.

That contains:

```text
repo/
inputs/
manifest
temporary job files
```

Retained/downloadable outputs go only under:

```text
/kaggle/working/output
```

This is intentional.

Do not put the staged repo under `/kaggle/working`, because Kaggle treats that tree as saved output and will download it back to Windows.

---

## 10. Why v0.2 exists

v0.1 passed the smoke test but revealed an efficiency bug:

```text
/kaggle/working/alice-kaggle-job/repo/...
```

was downloaded in full as kernel output.

v0.2 fixes that by placing scratch execution state in:

```text
/tmp/alice-kaggle-job
```

Only retained result files should remain in:

```text
/kaggle/working/output
```

Expected download after v0.2:

```text
output/alice-kaggle-status.json
output/alice-kaggle.log
output/<real result files>
<kernel log>
```

---

## 11. Kaggle private Dataset behavior

The launcher creates a **private per-job Kaggle Dataset**.

Example:

```text
mkrayanyan/alice-job-<timestamp>-<token>-data
```

Kaggle may expose uploaded content inside `/kaggle/input` in either form:

### Form A: original ZIP

```text
alice-kaggle-payload.zip
```

### Form B: automatically expanded Dataset

```text
alice-kaggle-manifest.json
repo/
inputs/
```

v0.1+ supports both.

Before GPU submission, the launcher runs a Dataset file listing so mount/data problems are detected earlier.

---

## 12. Private kernel behavior

The launcher creates a **private Kaggle script kernel**.

Example:

```text
mkrayanyan/alice-job-<timestamp>-<token>
```

The normal lifecycle is:

```text
QUEUED
↓
RUNNING
↓
COMPLETE
```

Long `QUEUED` periods can happen due to Kaggle GPU demand.

Do not treat `QUEUED` as a launcher failure.

---

## 13. Confirmed GPU configuration

The final validated smoke test reported:

```text
GPU(s): Tesla T4, Tesla T4
```

Therefore the validated Kaggle runtime exposed:

```text
2 × NVIDIA Tesla T4
```

Important:

```text
2 × 16 GB VRAM ≠ one 32 GB GPU
```

The GPUs have separate VRAM spaces.

A.L.I.C.E. workloads must explicitly use multi-GPU code to benefit from both.

Examples:

```text
torch DistributedDataParallel
Hugging Face Accelerate
explicit device placement
one independent workload shard per GPU
```

For any run, actual availability should still be checked with:

```python
torch.cuda.device_count()
```

---

## 14. Validated smoke test

Command:

```powershell
alice-kaggle `
    -Script C:\A.L.I.C.E-main\tools\kaggle-smoke-test.py `
    -RequireGpus 1 `
    -OutputDir C:\A.L.I.C.E-main\remote-results\kaggle\v0-smoke
```

Validated behavior:

```text
private Dataset created
Dataset reached ready
Dataset files visible
private kernel pushed
kernel RUNNING
kernel COMPLETE
outputs downloaded
2 Tesla T4 GPUs exposed
job passed
kernel deleted
Dataset deleted
```

Final acceptance output:

```text
ALICE KAGGLE JOB PASSED
GPU(s): Tesla T4, Tesla T4
Results: C:\A.L.I.C.E-main\remote-results\kaggle\v0-smoke
Remote cleanup: kernel=OK, dataset=OK
```

---

## 15. Failure handling

The remote worker catches the actual workload exception and writes:

```text
alice-kaggle-status.json
alice-kaggle.log
```

to:

```text
/kaggle/working/output
```

The Kaggle kernel itself is allowed to finish cleanly so outputs remain downloadable.

The local launcher then reads:

```text
alice-kaggle-status.json
```

and returns failure to Windows if the A.L.I.C.E. workload failed.

Therefore a remote Python exception should still give us a usable traceback/log.

---

## 16. Default cleanup

Default lifecycle:

```text
upload private Dataset
run private kernel
download outputs
delete private kernel
delete private Dataset
```

Validated cleanup:

```text
kernel=OK
dataset=OK
```

To intentionally retain Kaggle resources:

```powershell
-KeepRemote
```

Use that only when persistence/debugging is useful.

---

## 17. Local job receipt

Each job writes:

```text
alice-kaggle-job-receipt.json
```

to the local output directory.

It records:

```text
job ID
Dataset ID
kernel ID
accelerator
input paths
arguments
payload size
workload result
timestamps
cleanup result
```

Use this as the local audit/history record for the Kaggle dispatch.

---

## 18. Current backend-selection rule

Until a future machine is deliberately configured:

```text
PRIMARY REMOTE GPU BACKEND = Kaggle
```

Use:

```text
alice-kaggle
```

Do not select the retired:

```text
alice-remote
```

RTX 3050 path.

A future replacement machine can be added later as a separate backend without blocking current work.

---

## 19. Recommended dispatch logic for A.L.I.C.E.

When an A.L.I.C.E. chat needs remote compute:

```text
1. Identify the heavy workload.
2. Decide whether Kaggle GPU/RAM materially helps.
3. Ensure retained artifacts are written to ALICE_KAGGLE_OUTPUT.
4. Include any external vault/data/model paths using -InputPath.
5. Add requirements/pip packages if needed.
6. Use alice-kaggle.
7. Wait through QUEUED if necessary.
8. Confirm RUNNING.
9. Confirm COMPLETE.
10. Download results.
11. Read alice-kaggle-status.json.
12. Accept only if workload status says success.
13. Confirm cleanup or intentionally use -KeepRemote.
```

---

## 20. Multi-GPU rule

Do not assume normal single-GPU code will automatically use both T4s.

A.L.I.C.E. should inspect the workload and choose an explicit strategy.

Examples:

### Independent parallel jobs

```text
GPU 0 -> shard A
GPU 1 -> shard B
```

### Distributed model training

```text
DDP / Accelerate
```

### Model sharding

Only if the specific framework/model supports safe device mapping.

---

## 21. Platform constraints still relevant

Even though Kaggle is trusted, platform limits still matter.

A.L.I.C.E. should account for:

```text
GPU quota
session/runtime limits
queue delays
available disk
available RAM
per-run output limits
internet availability
Kaggle kernel lifecycle
```

Do not assume every job starts immediately.

---

## 22. Known-good acceptance signals

For a healthy job:

```text
Dataset status: ready
KernelWorkerStatus.RUNNING
KernelWorkerStatus.COMPLETE
ALICE KAGGLE JOB PASSED
Remote cleanup: kernel=OK, dataset=OK
```

For the validated GPU smoke:

```text
GPU(s): Tesla T4, Tesla T4
```

---

## 23. Troubleshooting

### A. PowerShell blocks unsigned script

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
```

or:

```powershell
Unblock-File C:\A.L.I.C.E-main\tools\alice-kaggle.ps1
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
```

### B. Doctor says config missing

Run:

```powershell
alice-kaggle -Setup -Username mkrayanyan
```

### C. Authentication fails

Run:

```powershell
kaggle auth login
```

### D. Kernel remains QUEUED

Wait.

This can be normal Kaggle GPU scheduling contention.

### E. Payload not found under `/kaggle/input`

Current v0.2 supports both:

```text
original ZIP
auto-expanded Dataset
```

If it fails again, inspect the diagnostic `/kaggle/input` listing from the worker.

### F. Entire repo starts downloading as output

That means the worker is likely using an older pre-v0.2 layout under:

```text
/kaggle/working/alice-kaggle-job
```

Use v0.2 where scratch state lives under:

```text
/tmp/alice-kaggle-job
```

---

## 24. Current files/version state

Active intended version:

```text
alice-kaggle v0.2
```

Python launcher:

```text
C:\A.L.I.C.E-main\tools\alice_kaggle_v0.py
```

PowerShell wrapper:

```text
C:\A.L.I.C.E-main\tools\alice-kaggle.ps1
```

Smoke test:

```text
C:\A.L.I.C.E-main\tools\kaggle-smoke-test.py
```

The Python filename still contains `v0`; that is intentional for now. The implementation version is v0.2.

---

## 25. Validation record

### Authentication/config

Validated:

```text
Kaggle CLI 2.2.4
username mkrayanyan
auth OK
doctor PASS
```

### Dataset

Validated:

```text
private Dataset created
status ready
manifest visible
repo files visible
```

### Kernel

Validated:

```text
private kernel pushed
RUNNING
COMPLETE
```

### GPU

Validated:

```text
Tesla T4
Tesla T4
```

### Output

Validated:

```text
alice-kaggle-status.json downloaded
alice-kaggle.log downloaded
kaggle-smoke-result.json downloaded
```

### Cleanup

Validated:

```text
kernel=OK
dataset=OK
```

---

## 26. Important note about canonical state

Kaggle is trusted and may store full A.L.I.C.E. data during active work.

However, unless a future architecture explicitly changes this, the local Windows machine remains the normal canonical location for the active repository and retained project state.

Kaggle is the primary compute backend, not necessarily the final source-of-truth for every retained artifact.

Returned outputs should be incorporated into A.L.I.C.E.'s local/canonical state according to the normal project workflow.

---

## 27. Final invariant for future A.L.I.C.E. chats

Use:

```text
alice-kaggle
```

for remote GPU compute.

Assume:

```text
Kaggle trusted
2 × T4 validated
full A.L.I.C.E. data access allowed
private Dataset + private kernel
outputs return to Windows
temporary remote resources deleted by default
```

Do **not** reintroduce the retired shared RTX-3050 hardening policy into the Kaggle backend.

If a future dedicated machine becomes available, add it as a separate backend rather than blocking current Kaggle-based work.

---

## Provenance / validation evidence

Final validated v0.1 run showed:

```text
ALICE KAGGLE JOB PASSED
GPU(s): Tesla T4, Tesla T4
Remote cleanup: kernel=OK, dataset=OK
```

The v0.2 update only changes execution scratch placement so the full staged repo is no longer downloaded as output.

Primary validation log:

```text
Pasted text(20260823-195013).txt
```

Relevant evidence includes:

```text
private Dataset ready
Dataset file listing present
kernel RUNNING
kernel COMPLETE
dual Tesla T4
successful output download
successful kernel deletion
successful Dataset deletion
```

