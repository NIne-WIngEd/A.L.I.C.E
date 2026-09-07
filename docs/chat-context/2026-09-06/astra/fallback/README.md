# Qwen fallback — owner approved, 6 September 2026

> Current checkpoint, 7 September: v102 stopped before submission at the source evidence-origin check. Do not rerun it to repair this stop. Use the read-only trace linked from the [current diagnosis](../QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md). Execution-pending or ready instructions below describe the earlier checkpoint.

> Current execution uses the [v102 infrastructure successor](../tools/qwen-runtime-v102/README.md). V101 transport is now verified. Source job 575089 is closed after a zero-inference TLS failure. Statements below about pending v101 execution describe the earlier checkpoint.

Rayan explicitly approved the exact proposed amendment with the message “approve”. [Owner approval](OWNER_APPROVAL.json) binds the unchanged [historical draft](QWEN_FALLBACK_AMENDMENT_V1_DRAFT.json) by SHA-256. [Effective approved amendment](QWEN_FALLBACK_AMENDMENT_V1_APPROVED.json) is the authority for the bounded public qualification package implemented in [Qwen public qualification v1.0.0](../tools/qwen-public-v1/README.md).

The approved candidate is `qwen3.8:27b-q4_K_M` with thinking enabled. Magnolia CPU is the selected route: partition `node`, QOS `normal`, 20 CPUs, 48 GiB RAM, exclusive allocation, six-hour maximum job. Resolve and freeze the full model digest before inference. Keep all 16 frozen fictional tasks and the existing decision-centric pass rule. One request per task has a 6144-token ceiling. There is no automatic model, profile, GPU, or provider substitution.

Approval is not a qualification result. No Qwen run or successor binding has completed in this checkpoint. Four independent families remain required. Keep valid Gemma, Mistral and Granite evidence and the failed GLM v186 evidence. The 287-candidate pool, 63 replacements and one deferred item remain unchanged. Private pointwise work, breadth-v104 eligibility, acceptance, promotion and training remain blocked by their existing prerequisites.

The historical `validate_fallback_amendment.py` and its receipt validate the draft as drafted. They are preserved as historical evidence; they do not validate the newly approved runtime package.

The release subsequently passed all 35 tests natively on Windows, then failed during upload before submission. The [active v101 transport repair](../tools/qwen-transport-v101/README.md) corrects the remote path mismatch while reusing the exact original workload and run state. It passed 11 focused regression tests. Actual v101 transfer and the model result remain pending. Follow [QWEN_RELEASE.json](../QWEN_RELEASE.json) and the [latest actual-result pointer](../qwen-public/LATEST_QWEN_PUBLIC.json).
