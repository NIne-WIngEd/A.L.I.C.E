# Actual host trace and the v103 decision — 7 September 2026

The returned trace confirms the source-origin defect. At `2026-09-07T02:28:15Z`,
the old live run had no `package-manifest.json`. Its real package contained
`PACKAGE_MANIFEST.json` with the exact expected hash. All 17 original package
files, all 25 v102 package files and both original ZIPs matched. Eight critical
source-run files also matched. This is a locator error in Astra's v102 verifier
and its fixture. The existing source evidence does not need repair.

[The reviewed observation](QWEN_HOST_TRACE_REVIEW.json) binds the complete
[raw report](tools/qwen-origin-v103/observed/report.json) and the exact uploaded
ZIP. [The earlier reproduction](QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md) documents
how the real collector combines run files and package projections in an export.
The actual original collector is exercised in the new regression tests.

## Why an explicit a2 revision is supported

A2 had only its descriptor and captured preflight failure among the checked
lifecycle files. Its local state was PREPARED and staged, without a job ID.
There was no submission intent, job script, worker, runtime, model lock, probe
or task intent. Every scheduler query succeeded. A2 had no queue/accounting
row; source job 575089 was FAILED with exit `76:0`. The diagnostic left both
local controller states unchanged. Its six native Windows tests passed and
it exited 0. These facts permit a pre-submission revision of the existing a2,
subject to checking them again immediately before that transition.

V103 archives the exact original local state before remote mutation. Under the
same remote submission lock used by v102 it rechecks the source origins, source
job closure, old package and absence of a2 execution. It preserves the old a2
descriptor and failure bytes, writes a deterministic revision intent, atomically
updates the active package reference and records the completion receipt. The
client changes its package reference only after verifying that acknowledgement.
Interrupted transitions and lost acknowledgements can resume from that lineage.
The old launcher rejects the revised descriptor. No state deletion is required.

This is an explicit package revision of `alice-qwen38-a2-c926d9e355dd`. The
logical calibration and approved workload remain unchanged. The source a1 run
and its separate terminal ledger recovery remain historical evidence. There
is no new a3 execution or automatic additional judgment attempt.

## HTTPS and validation limits

Both login-node HTTPS HEAD endpoints returned 200 using the system CA bundle
with certificate and hostname verification enabled. This supports continuing
the already implemented verified-HTTPS preflight. It does not resolve the full
model digest, download the runtime body or establish compute-node throughput.
Those gates remain in the approved execution path.

The final v103 ZIP passed 60 offline tests after fresh extraction. They cover
real physical package/run origins, tampered source bytes, failed scheduler
queries, execution-marker conflicts, five interrupted revision writes, lost
acknowledgements, the actual old launcher refusing revised state and a single
submission across repeated controller invocation. The embedded PowerShell
Python verifier, repeated SSH bootstrap extraction and Bash syntax also passed.
Native v103 PowerShell and actual Magnolia execution remain pending.

The diagnostic labels `job.sbatch` INVALID_JSON because it is a Bash script.
Its file hash matches. That annotation is not evidence of script corruption.
Local state fixtures use explicitly synthetic timestamps because the trace
contains the actual state's hash and selected fields, not all raw bytes. The
production transition requires the exact observed hash and archives real bytes.

## Purpose and next dependency

The next action is the [v103 release](QWEN_PRE_SUBMISSION_REVISION_V103.json).
Qwen calibration supports evaluation of synthetic identity candidates. It does
not establish learned identity, AMFM quality or continuing memory cognition.
Even a passing Qwen result needs four exact family/profile receipts and explicit
successor binding. Breadth prerequisite migration remains separate. E01–E07
retain their unexecuted status; MC8 stays sealed. Preserve the existing candidates,
qualified evidence and the accepted G → G+ → Fable/Friday ordering.
