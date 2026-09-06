# Qwen v100 Windows transfer failure and v101 repair

The supplied terminal shows that v100 passed all 35 tests natively on Windows and then stopped during SCP upload with exit 74. The exact workload ZIP hash was `bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e`. This invocation did not reach Slurm submission or Qwen inference. This is a transport failure, with no Qwen calibration outcome.

The exact [terminal bytes](artifacts/S12-terminal-222711.txt) have SHA-256 `78b19e9e4f53a977547b639b839ba9988628eade5d970984f7961cfa72fa24ff`. [Structured observations](QWEN_V100_WINDOWS_TERMINAL_OBSERVATION.json) distinguish reported facts from the reproduced diagnosis. No remote directory listing was collected. The log also confirms the previously reviewed GLM forensic publication; it does not reopen that completed review.

## Root cause

The frozen `contract.py` defines Magnolia's remote root with `Path('/homes/01/mxrayan/rayan-compute')`. The local `controller.Magnolia` converts that value and its children with `str`. On Windows those strings contain backslashes. `stage()` uses `shlex.join` to quote the remote `mkdir` argument. Linux treats the quoted backslashes literally, so `mkdir` can return zero while creating a wrongly named relative directory. It does not create the intended `/homes/.../packages/<run>` location. SCP then reports that its intended destination is absent.

This behavior was reproduced with `PureWindowsPath` and an actual local bash invocation: the old command returned zero while the intended POSIX directory remained absent; the corrected command created the intended directory. This supports the diagnosis from the exact Windows source and observed transfer failure. It is not a claim that Astra inspected Magnolia's directory contents.

The same boundary defect appears later in `download()`: a literal POSIX server receipt is compared with `str(c.REMOTE_RUN / 'result-bundle.zip')`. The Windows-formatted expected value rejects the correct server address before SCP. Both boundaries are corrected together.

The prior tests modeled both ends with the same host path convention. They missed the Windows-client/Linux-server mismatch despite passing natively. Earlier provider infrastructure used explicit POSIX remote strings. The new controller regressed that distinction. The Python [pathlib documentation](https://docs.python.org/3.13/library/pathlib.html#pure-paths) describes the platform-dependent `Path` and platform-independent pure path classes.

## Bounded correction and identity

The v101 ZIP is a client transport wrapper around the exact original v100 ZIP. It verifies the original ZIP, manifest, all source files and approved authority. It then replaces only the current client process's remote root and run metadata with `PurePosixPath`. Local vault, Downloads, repository and SSH-key paths retain their native Windows handling. The original code files on disk, remote bootstrap/worker, prompt, tasks, model/profile, scientific gates and CPU envelope remain unchanged.

The stable run remains `alice-qwen38-a1-8e7a384a745496f4`. The original controller still receives workload hash `bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e`. The wrapper's distinct ZIP hash is recorded separately in [repair identity](QWEN_TRANSPORT_REPAIR_V101.json). There is no state migration or new activation. The original controller reconciles any newer state before submission. No existing state or remote directory is deleted by the repair.

## Evidence and limits

| Check | Evidence |
|---|---|
| Original Windows v100 preflight | 16 manifest files verified, 8 Python files compiled, 35 tests passed with no skips in 69.568 seconds; reported by the supplied terminal. |
| Original live failure | SCP upload failed with exit 74 before `save(staged=True)` or `remote.action('submit')`. |
| Repair release build | Fresh extraction, exact nested workload verification, source compilation and the PowerShell-embedded Python verifier passed. |
| Repair regressions | All 11 passed on Linux; actual upload/download methods tested against literal POSIX addresses, hash preservation, local Windows paths, rejected wrong paths, existing prepared/submitted state reuse, and actual bash reproduction. |
| Next native Windows preflight | Ten portable tests will execute; one Linux bash reproduction is intentionally skipped. |
| Actual v101 transport/inference | Pending. No live Magnolia result is claimed by the source build. |

[Full build receipt](tools/qwen-transport-v101/dist/BUILD_RECEIPT.json) and [test log](tools/qwen-transport-v101/dist/BUILD_SELFTEST.log) are preserved. Run the [v101 launcher](tools/qwen-transport-v101/README.md) and return its terminal output and result ZIP if produced. The same launcher also performs later attachment and collection. The latest scientific result pointer remains pending; this software repair does not establish a successful qualification, four-family successor, pointwise readiness or breadth eligibility.

## Durable lesson

Represent remote filesystem addresses using the remote platform's path flavor. Keep local filesystem paths native. Tests at a platform boundary must give each side an independently specified path representation. A local selftest pass cannot establish that an SSH-side directory or a real model result exists.
