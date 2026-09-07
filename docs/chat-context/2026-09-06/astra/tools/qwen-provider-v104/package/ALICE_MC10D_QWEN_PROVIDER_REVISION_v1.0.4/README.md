# Qwen v104: observed Magnolia scheduler repair

The owner-returned provider trace at 2026-09-07T04:40:46Z recorded one failure:
`squeue -h -j 575089 -o %A|%j|%T` exited 1 with
`slurm_load_jobs error: Invalid job id specified`.
The user-wide queue succeeded and was empty. Exact accounting returned source
job 575089 with its original a1 name and FAILED / 76:0. A2 accounting was empty.
Both local and remote a2 inventories were stable and still referenced v102.
The unfinished v103 intent existed locally. No remote revision or checked
submission, runtime, probe or task marker was present.

This is a new observation of the exact failing query. The original v103
invocation discarded its nested error; that historical message is not recovered.

## Changes and scope

One shared Magnolia scheduler module handles source reconciliation, a2 job
lookup and monitoring. Active membership uses the observed user-wide query.
Historical identity uses exact accounting with an explicit start date. Failed
or malformed queries remain unknown. Raw argv, exits, both streams and times
are flushed to scheduler-commands.jsonl before assertions. Collection exports
that journal. Remote stops also return the last command receipt.

The existing v103 local intent is retained byte-for-byte. V104 requires its
exact intent and archived state. Any local acknowledgement or other remote
revision stops the transition. A new revision, a2-provider-v104-3c5ed8da35dc,
explicitly records the superseded uncommitted v103 intent and provider trace.
It keeps the same a2 execution identity. Under the original submission lock,
it rechecks actual jobs, execution markers, source origins and the old package
before changing the active descriptor. Interrupted writes and lost replies
retain the existing recovery mechanism. No state deletion is needed.

Qwen qwen3.8:27b-q4_K_M, its approved profile, scientific authority, prompt,
128-token probe and all 16 public task requests remain byte-exact. Worker,
verified HTTPS, telemetry and context publishers are unchanged. The approved
Magnolia CPU allocation remains 20 CPUs / 48 GiB / six hours. Runtime identity,
model digest and throughput/budget gates still precede eligible calibration.
There is no new execution identity or automatic provider/model fallback.

## Run and outcomes

Place the ZIP and Start-ALICEAstraQwenQualificationV104.ps1 together in Downloads.
The thin launcher uses the established Anaconda-first workflow. It verifies,
extracts and tests the package, reconciles the exact prepared a2 revision, then
continues the already approved public calibration after live gates pass.

Exit 74: transport/monitoring pending; use this same launcher to reattach.
Exit 75: evidence retained and publication pending; use this same launcher.
Exit 76: deterministic stop; return the transcript and result ZIP if produced.
Do not use earlier Qwen launchers. Return RESULT_ZIP and the terminal output.

The release's 66 offline tests include the existing recovery suite and six
focused provider tests. Captured real scheduler responses drive the supported
query shapes; adversarial mutations are labelled synthetic. Tests cover raw
failure/timeout retention, exact prior intent preservation, conflicting revision
stops, interrupted writes and lost acknowledgements without duplicate jobs.
They are not native Windows, live revision, model or scientific pass evidence.

Qwen remains unevaluated at release. A passing public calibration would enable
review of the four exact judge bindings and breadth prerequisite migration.
Private candidate evaluation, A-SYN acceptance, training, Stage G closure and
Phase 2 replacement remain downstream gates.
