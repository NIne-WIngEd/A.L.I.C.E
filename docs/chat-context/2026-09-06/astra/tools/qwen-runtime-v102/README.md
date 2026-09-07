# Active Qwen Windows launcher: v1.0.2

> Update, 7 September: the actual host trace is verified and the explicit v103 revision is ready. Follow the [current findings](../../QWEN_HOST_TRACE_FINDINGS.md). Instructions below describe the earlier checkpoint.

> Current checkpoint, 7 September: v102 stopped before submission at the source evidence-origin check. Do not rerun it to repair this stop. Use the read-only trace linked from the [current diagnosis](../../QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md). Execution-pending or ready instructions below describe the earlier checkpoint.

Use [Start-ALICEAstraQwenQualificationV102.ps1](dist/Start-ALICEAstraQwenQualificationV102.ps1) with `ALICE_MC10D_QWEN_RUNTIME_SUCCESSOR_v1.0.2.zip`. Save both in Downloads and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV102.ps1"
```

[Release identity](../../QWEN_RUNTIME_SUCCESSOR_V102.json), [actual source failure review](../../QWEN_V101_RESULT_FINDINGS.md), [package instructions](package/ALICE_MC10D_QWEN_RUNTIME_SUCCESSOR_v1.0.2/README.md), [build receipt](dist/BUILD_RECEIPT.json) and [test log](dist/BUILD_SELFTEST.log) contain the exact scope and validation.

This is one explicitly named infrastructure successor to closed job `575089`. The old execution remains immutable. No model or probe/task attempt occurred in that source job. All scientific authority, prompt and request bytes remain unchanged. A live source check and verified HTTPS preflight run before submission. Distinct immutable telemetry snapshots preserve the existing ledger rules.

All rebuild inputs are included. Run `python build_release.py` from this directory on Linux with Python 3.11 or later and bash. The builder emits the ZIP and launcher into `dist`, runs 48 tests from a fresh extraction and checks literal Windows/SSH verifier code. Fixed ZIP metadata preserves release bytes. Environment and elapsed-time fields in rebuilt receipts can differ. Actual native v102 Windows/Magnolia execution is pending.
