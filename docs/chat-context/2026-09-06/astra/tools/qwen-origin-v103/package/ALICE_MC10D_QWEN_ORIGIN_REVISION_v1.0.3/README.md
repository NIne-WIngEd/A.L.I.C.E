# Qwen v103: verified origins and an explicit a2 revision

The returned host trace confirmed the v102 defect. Its source verifier looked
for a package projection in the old run directory. The actual manifest is in
the original package directory and matches the frozen hash. All 17 original
package files and all 25 v102 package files match. The eight source run records
match. A2 has only its descriptor and captured preflight failure among the
checked lifecycle records. It has no intent, job, worker, probe or task attempt.
Both HTTPS HEAD endpoints returned 200 with certificate verification enabled.

Observation: `2026-09-07T02:28:15Z`. The exact trace ZIP is under `source/`.
That observation does not replace the live checks described below.

## Scientific scope

The owner already approved Qwen `qwen3.8:27b-q4_K_M` on the frozen 16 public tasks.
The six scientific authority files, prompt, all 16 requests and the 128-token
probe request remain byte-exact. The v102 worker, HTTPS client, telemetry
publisher and context publisher remain unchanged. This release repairs origin
resolution and the declared pre-submission package transition.

Execution stays `alice-qwen38-a2-c926d9e355dd`. Logical calibration stays
`alice-qwen38-a1-8e7a384a745496f4`. Source job 575089 stays closed and unchanged.
There is no a3 identity, automatic requeue, extra judgment attempt or fallback.

Allocation stays Magnolia CPU, 20 CPUs, 48 GiB, exclusive, partition `node`, QOS
`normal`, six hours. Runtime/model identities and CPU throughput must pass
before calibration tasks. A model manifest HEAD success does not resolve its
full digest or prove compute-node throughput. Preflight and the worker still
establish those facts before an eligible result can exist.

## Explicit revision sequence

The Windows launcher acquires the existing a2 controller lock. It requires the
exact observed PREPARED local state and archives its raw bytes. It uploads the
new ZIP under its own content hash and asks the remote agent to revise a2.

The remote agent acquires the same submission lock used by v102. It checks for
any a2 job, intent or execution marker. It rechecks exact a1 files at their real
origins, source job closure and the old v102 package. Failed scheduler queries
stop the operation; empty failed output is never accepted as absence.

It preserves the old descriptor and preflight failure under
`revisions/a2-origin-v103-02ba19cdef70/`. A deterministic intent records both old
and new descriptor/package hashes and the diagnostic lineage. Only then does
it atomically replace the active `run.json` package reference and record
`package-revision.json`. The original descriptor stays byte-exact in history.
The old v102 launcher refuses the revised descriptor instead of submitting it.

After a matching remote acknowledgement the client updates its active package
reference. Its original state and remote receipt remain in its own revision
directory. A lost acknowledgement can resume this transaction. A completed
revision can attach to its existing submitted job. The subsequent submission
path, workload, attempt records and result/publication recovery stay intact.

No manual state deletion or edit is required. The a1 local telemetry-pending
state remains historical; its terminal failure was separately recorded in the
compute ledger. Do not rerun old launchers to clear that state.

## Verification and outputs

The independent result verifier requires revision history and source-origin
proof. The real collector exports those records without copying package
metadata into a source run. The summary identifies the revision and trace hash.

Tests exercise the original collector, real physical package/run fixtures,
changed source packages, stale/active execution markers, failed scheduler
queries, five interruption positions, lost acknowledgements and one-job
submission across reruns. Existing worker/calibration/telemetry tests remain.
Synthetic test responses never become real qualification evidence.

Use `Start-ALICEAstraQwenQualificationV103.ps1` with this versioned ZIP in Downloads.
The launcher verifies the ZIP and manifest, compiles Python, runs offline tests,
revises the exact prepared a2 state and continues the approved calibration.

Exit 74 means transport/monitoring is pending; rerun this same v103 launcher.
Exit 75 means verified evidence is retained and publication is pending; rerun
v103 to attach or recover publication. Exit 76 is a deterministic stop; return
the output and result ZIP if produced instead of rerunning blindly.

Return the printed `RESULT_ZIP` and terminal output. The controller publishes
verified public metadata to `alice-context` and immutable telemetry snapshots
to the compute ledger. It does not publish private candidates or raw rationales.

Passing this public suite remains calibration. Four-family receipt binding,
breadth prerequisite migration, private pointwise execution, acceptance,
promotion and training retain separate gates. MC8 remains sealed. This work
supports evaluation of synthetic identity candidates; learned identity, AMFM
and complete memory still require the experiments in the accepted audit.
