# Active Windows launcher: Qwen transport repair v1.0.1

> Current execution uses the [v102 infrastructure successor](../qwen-runtime-v102/README.md). V101 transport is now verified. Source job 575089 is closed after a zero-inference TLS failure. Statements below about pending v101 execution describe the earlier checkpoint.

Use [Start-ALICEAstraQwenQualificationV101.ps1](dist/Start-ALICEAstraQwenQualificationV101.ps1) with the released `ALICE_MC10D_QWEN_WINDOWS_TRANSPORT_REPAIR_v1.0.1.zip`. Save both in Downloads. The new ZIP includes the exact original v100 workload ZIP. No separate v100 download is needed.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV101.ps1"
```

[Repair identity](../../QWEN_TRANSPORT_REPAIR_V101.json), [failure analysis](../../QWEN_V100_WINDOWS_TRANSFER_FINDINGS.md), [build receipt](dist/BUILD_RECEIPT.json) and [full package instructions](package/ALICE_MC10D_QWEN_WINDOWS_TRANSPORT_REPAIR_v1.0.1/README.md) record the exact scope and validation.

The stable run and original workload hash are preserved. Use v101 for initial recovery, monitoring and collection. Keep `C:\ALICE_Vault\tools\alice-astra\qwen-fallback-a1\controller-state.json`. Existing submission reconciliation and telemetry remain in the original controller.

## Rebuild

This directory contains all rebuild inputs, including the exact frozen workload ZIP under `package/.../workload/`. Run `python build_release.py` on Linux with bash and Python 3.11 or later. The build emits the repair ZIP and launcher under `dist/`, verifies both nested manifests, compiles sources and runs 11 focused regression tests from a fresh extraction. ZIP bytes use fixed metadata; elapsed test time and environment fields in a new build receipt can differ.

The v101 Windows launcher runs the portable tests before transport. One actual POSIX-shell reproduction is intentionally skipped on Windows; it passes in the Linux release build. The 35 original workload tests already passed natively on Rayan's Windows machine in the supplied log. Actual v101 Windows execution and Magnolia transfer still await the next user run. No model-result claim follows from these software checks.
