# Next action: read-only Magnolia trace

V102 stopped before submission because its source verifier confused package
projections in an evidence ZIP with files in the live run directory.
Read the [complete diagnosis and project dependency](../../QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md)
and [actual terminal observation](../../QWEN_V102_TERMINAL_OBSERVATION.json).

Use [Start-ALICEQwenReadOnlyTraceV100.ps1](dist/Start-ALICEQwenReadOnlyTraceV100.ps1)
with `ALICE_QWEN_READ_ONLY_TRACE_v1.0.0.zip`. Save both in Downloads and run once:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEQwenReadOnlyTraceV100.ps1"
```

Return the printed `TRACE_ZIP` and transcript. This reads existing source/a2
metadata, package hashes, scheduler status, CA and two HTTPS HEAD endpoints.
It does not stage a package, change either run, invoke a model or submit a job.
The diagnostic's version is independent of qualification v100–v102.

[Release identity](../../QWEN_READ_ONLY_TRACE_RELEASE.json),
[build receipt](dist/BUILD_RECEIPT.json), [test log](dist/BUILD_SELFTEST.log) and
[package scope](package/ALICE_QWEN_READ_ONLY_TRACE_v1.0.0/README.md) give the checks
and limitations. Six diagnostic tests passed from the final ZIP on Linux;
native Windows and actual Magnolia observations are pending. Rebuild with
`python -B build_trace.py` using Python 3.11 or later and bash.

The separate [production-collector reproduction](REPRODUCTION.json) is executable
using `reproduce_collector_boundary.py` with `--original`, `--successor`, `--archive`
and `--output`. Point the first two to the preserved original v100 and v102
package folders in the adjacent tools directories; the archive is the exact a1
source ZIP embedded in v102. The script writes only its output and a temporary
local fixture. Scheduler and telemetry boundaries are mocked; the original
collector and verifier are unmodified. This is not a live scheduler observation.
