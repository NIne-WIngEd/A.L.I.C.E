# Historical v103 launcher — scheduler stop requires review

The owner ran this exact release. Sixty native Windows tests passed. Remote
`revise` stopped at `Source queue query failed; absence is unknown` before this
invocation's remote revision writes or submission. Do not rerun this launcher
for the captured stop. Follow the [provider review](../../QWEN_V103_PROVIDER_REVIEW.md)
and [actual terminal review](../../QWEN_V103_TERMINAL_REVIEW.json).

The package, launcher and build receipt remain immutable historical evidence.
The instructions below describe the original release checkpoint.


The actual host trace confirmed the source-origin error and unsubmitted a2
state. [Findings](../../QWEN_HOST_TRACE_FINDINGS.md), [reviewed host facts](../../QWEN_HOST_TRACE_REVIEW.json),
[release identity](../../QWEN_PRE_SUBMISSION_REVISION_V103.json) and
[package instructions](package/ALICE_MC10D_QWEN_ORIGIN_REVISION_v1.0.3/README.md) record the decision.

Save `ALICE_MC10D_QWEN_ORIGIN_REVISION_v1.0.3.zip` and
[Start-ALICEAstraQwenQualificationV103.ps1](dist/Start-ALICEAstraQwenQualificationV103.ps1)
in Downloads and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$HOME\Downloads\Start-ALICEAstraQwenQualificationV103.ps1"
```

The launcher verifies package bytes, compiles and runs the offline suite. It
archives the exact observed local state and applies a recoverable remote package
revision before continuing the same approved a2 calibration. It rechecks live
source and scheduler evidence. Existing owner approval covers this unchanged
scientific plan. No manual state cleanup or renewed activation is required.

The release ZIP SHA-256 is `05e1e8aa3fa1233d8acb25da5e87a93b211791f5e568db850f83dca3499630f2`.
It contains 33 files and is 239623 bytes.

[Build receipt](dist/BUILD_RECEIPT.json) and [test log](dist/BUILD_SELFTEST.log)
record 60 passing tests from the final ZIP. All rebuild inputs are included,
including the exact prior v100/v102 packages and uploaded trace. Run
`python build_release.py` with Python 3.11+ and Bash on Linux. Fixed ZIP metadata
preserves package bytes; environment and timing fields in build receipts/logs
can differ. Native v103 Windows and Magnolia execution remain pending.

Return the printed RESULT_ZIP and terminal output. Exit 74 permits reattachment
with this launcher; exit 75 permits publication recovery. For exit 76, return
the captured stop instead of retrying old launchers. No new execution identity,
model/profile substitution, GPU fallback or private pointwise stage is implied.
