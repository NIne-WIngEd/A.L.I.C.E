# Qwen public qualification v1.0.0

This is the owner-approved Qwen fallback package. Read the [execution contract and instructions](package/ALICE_MC10D_QWEN_PUBLIC_QUALIFICATION_v1.0.0/README.md), [build receipt](dist/BUILD_RECEIPT.json), and [offline test log](dist/BUILD_SELFTEST.log). The [checked Windows launcher](dist/Start-ALICEAstraQwenQualificationV100.ps1) is paired with the exact ZIP hash in the build receipt.

All package source and authority files are preserved here. `python build_release.py` reconstructs the deterministic ZIP and launcher in `dist/` and verifies a fresh extraction. The distributed ZIP is also retained as a user artifact; its locator and exact identity are recorded in [QWEN_RELEASE.json](../../QWEN_RELEASE.json). The test runtime and timing fields in a newly generated build receipt may differ from the original receipt.

No actual Qwen model digest, scheduler job ID or calibration result exists in this build checkpoint. The local Windows launcher uses the existing Magnolia SSH key. Follow the [latest run pointer](../../qwen-public/LATEST_QWEN_PUBLIC.json) after execution. A later result must be verified from its actual captured streams and frozen authority; synthetic selftest passes are never qualification evidence.
