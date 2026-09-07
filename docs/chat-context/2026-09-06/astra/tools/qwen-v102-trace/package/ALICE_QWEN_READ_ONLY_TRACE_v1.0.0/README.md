# Read-only investigation of the v102 stop

The v102 qualification launcher stopped at `package-manifest.json`. Its verifier
confuses the evidence export layout with the live run layout. The original
collector reads that manifest from the package directory and places it in the ZIP.

This diagnostic reports file existence and exact hashes at both origins, the
current a2 descriptor and lifecycle markers, source/a2 scheduler observations,
the selected system CA and two bounded HTTPS HEAD results. All package files
have frozen expected hashes. There are no imports of the qualification workload.

It runs from SSH stdin using the existing Magnolia Python with the known shared
library path. Python `-B` prevents bytecode writes. Scheduler operations are
limited to `squeue` and `sacct`. HTTPS uses chain and hostname verification.
No model or runtime archive body is downloaded. A HEAD result does not prove a
future compute node can run the model, and it does not freeze the model digest.

Only a fresh folder in Windows Downloads is written. The existing a1/a2 local
state is read before and after. This diagnostic never stages a package on
Magnolia, submits or cancels a job, invokes telemetry publication, changes a
run descriptor, deletes files, starts a model, or grants qualification.

Save the versioned ZIP and `Start-ALICEQwenReadOnlyTraceV100.ps1` together in
Downloads. Run the launcher once and return its `TRACE_ZIP` and transcript.
Its ZIP name starts `ALICE_QWEN_TRACE_`. A successful diagnostic collection is
not a qualification pass. Do not rerun v102 to work around this deterministic stop.

The diagnostic package version is independent of qualification versions v100–v102.
It does not create an a3 execution. Source job 575089 and a2 state stay in place.
