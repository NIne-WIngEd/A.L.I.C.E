# Qwen Windows transport repair v1.0.1

Use `Start-ALICEAstraQwenQualificationV101.ps1` with this repair ZIP. Save both in Downloads and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV101.ps1"
```

The package includes the exact original v100 workload ZIP. No separate download or manual patch is required. The original workload hash remains `bc7c76c29a4f0fcfbd4567a610737ec5bb2052e7165677562959ac4dca7c959e`. The stable run ID remains `alice-qwen38-a1-8e7a384a745496f4`. Use this v101 launcher for later monitoring and collection as well. Preserve the existing controller state.

## What failed

The supplied Windows terminal log shows all 35 v100 tests passed. It then stopped during package upload with exit 74. That invocation did not reach Slurm submission or model inference.

The local controller used platform-native `Path` objects to represent Magnolia's Linux addresses. On Windows, converting those objects to strings produces backslashes. SSH passed the quoted backslashes literally to Linux `mkdir`. A successful exit therefore did not establish the intended `/homes/...` directory. SCP then reported that its destination did not exist. The same native-path conversion would also reject the POSIX result path returned by Magnolia during collection.

The defect was reproduced with `PureWindowsPath` and an actual local bash `mkdir`. The earlier tests mocked both ends using the same host-native representation. They therefore missed the mismatch between a Windows client and a Linux server. Prior infrastructure code used explicit POSIX remote strings; the new controller failed to preserve that distinction.

## What this repair does

The local runner verifies and extracts the unchanged workload. It validates the original authority and compiles the original Python files. It then represents the client's two remote address values with `PurePosixPath`. Local vault, Downloads, repository and SSH-key paths retain their normal Windows representation. Both upload construction and returned-result path validation use the corrected addresses.

The original workload source files, workload manifest, remote worker, frozen public tasks, Qwen profile, scientific gates, six-hour Magnolia CPU envelope, and persistent run identity remain byte-for-byte unchanged. No state migration, new run ID, model substitution or renewed activation is needed. The remote bootstrap and worker still execute the original verified payload. The client records a separate transport-repair receipt in the fresh launcher folder.

The launcher runs the repair's focused offline tests. It reuses the 35 unchanged workload tests already reported passing on the user's Windows machine in terminal SHA-256 `78b19e9e4f53a977547b639b839ba9988628eade5d970984f7961cfa72fa24ff`. The repair tests specifically exercise Windows client paths against literal POSIX server addresses, SCP upload/download boundaries, preservation of local Windows paths, unchanged workload hashes and reuse of an existing controller state.

## Resume and results

The existing controller's submission intent, Slurm reconciliation, per-task checkpoints, telemetry and independent result verification still apply. This failed invocation did not set `staged=true`, so v101 will stage the same workload correctly before submission. If another invocation has progressed since the supplied log, the same original controller identity is reconciled rather than replaced.

Exit 74 means monitoring or transport is pending. Exit 75 means publication is pending. Rerun this v101 launcher for either. Exit 76 means a deterministic stop or unsuccessful calibration; return the result ZIP if produced and the transcript. Exit 0 means public calibration and publication passed. It still does not create a four-family successor binding or establish breadth-v104 eligibility.

Return the resulting `ALICE_QWEN_PUBLIC_RESULT_<hash>.zip` and terminal output. Live Magnolia transfer has not been exercised from Astra's workspace. This repair's validation is recorded separately from the actual Windows execution that generated the input log.

The relevant Python behavior is documented in [pathlib's pure and concrete path classes](https://docs.python.org/3.13/library/pathlib.html#pure-paths). Concrete `Path` follows the client's platform; `PurePosixPath` keeps the remote address in POSIX form across platforms.
