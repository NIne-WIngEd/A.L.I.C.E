# ALICE telemetry transport receipt v1.0.0

Read-only follow-up to actual a2 job 575155 and uploaded helper SHA256
ba4482728c09313aede5031e682f68e8007c79c9cfa084d3da4b11aa984d0830.

The helper suppresses fetch/push errors and exits 74 after exhausting retries.
This receipt uses its installed GIT_SSH wrapper and existing ledger clone:

- git --version
- git fetch --dry-run origin main
- git push --dry-run origin HEAD:refs/heads/main

Git commands have 10/20/20-second limits; the outer SSH call has an 85-second
limit. Git stdout/stderr and exit status are retained as bytes. The launcher
returns a trace ZIP even when Git, SSH, parsing or helper verification fails.
It reads wrapper source and selected Git metadata before/after. It reads no
private key contents. It does not execute the telemetry helper, stage packages
on Magnolia, submit jobs, change a2 state, or publish repository content.
Returned raw diagnostics require review before any context publication.

These are current login-node transport observations. Successful dry runs cannot
recover the lost historical stderr or prove a real fetch/commit/push transaction
will succeed. A non-fast-forward dry-run is a distinct observation from an SSH
authentication failure. No scientific execution authority follows any result.

Git 1.8.3.1 documentation supports both dry-run flags:
https://github.com/git/git/blob/v1.8.3.1/Documentation/fetch-options.txt
https://github.com/git/git/blob/v1.8.3.1/Documentation/git-push.txt

Keep the exact ZIP beside Start-ALICETelemetryTransportReceiptV100.ps1.
Run the launcher in the established Windows/Anaconda workflow. Return TRACE_ZIP
and the transcript. Qwen remains unevaluated; do not rerun v104.
