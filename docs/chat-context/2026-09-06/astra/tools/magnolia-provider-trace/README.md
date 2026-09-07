# Magnolia read-only provider trace

See [release identity](../../MAGNOLIA_PROVIDER_TRACE_RELEASE.json) and the
[package description](package/ALICE_MAGNOLIA_PROVIDER_TRACE_v1.0.0/README.md).
This source contains no Qwen execution workload or prior-wrapper dependency.

Rebuild from this directory with `python -B build_trace.py`. It emits the ZIP
and thin launcher in `dist`, then verifies fresh extraction, manifest, Python
compilation, nine offline tests and literal remote Bash syntax. The recorded
BUILD_RECEIPT and BUILD_SELFTEST describe the distributed release; a local
rebuild creates its own test receipt. The ZIP is content-hashed and retained
under the distribution artifact ID in the release; it is not recursively
embedded in another package.

The Windows launcher selects Anaconda first, verifies the exact ZIP, extracts
into a fresh Downloads folder, runs the package checks, then collects one
read-only observation through the owner's existing SSH identity.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEMagnoliaProviderTraceV100.ps1"
```

Return TRACE_ZIP and the transcript, including if observations are incomplete.
Nine offline tests are release evidence only; no native Windows or actual
Magnolia result is established at publication. Do not rerun Qwen v103.
