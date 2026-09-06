# Qwen fallback — owner approved, 6 September 2026

Rayan explicitly approved the exact proposed amendment with the message “approve”. [Owner approval](OWNER_APPROVAL.json) binds the unchanged [historical draft](QWEN_FALLBACK_AMENDMENT_V1_DRAFT.json) by SHA-256. [Effective approved amendment](QWEN_FALLBACK_AMENDMENT_V1_APPROVED.json) is the authority for the bounded public qualification package implemented in [Qwen public qualification v1.0.0](../tools/qwen-public-v1/README.md).

The approved candidate is `qwen3.8:27b-q4_K_M` with thinking enabled. Magnolia CPU is the selected route: partition `node`, QOS `normal`, 20 CPUs, 48 GiB RAM, exclusive allocation, six-hour maximum job. Resolve and freeze the full model digest before inference. Keep all 16 frozen fictional tasks and the existing decision-centric pass rule. One request per task has a 6144-token ceiling. There is no automatic model, profile, GPU, or provider substitution.

Approval is not a qualification result. No Qwen run or successor binding has completed in this checkpoint. Four independent families remain required. Keep valid Gemma, Mistral and Granite evidence and the failed GLM v186 evidence. The 287-candidate pool, 63 replacements and one deferred item remain unchanged. Private pointwise work, breadth-v104 eligibility, acceptance, promotion and training remain blocked by their existing prerequisites.

The historical `validate_fallback_amendment.py` and its receipt validate the draft as drafted. They are preserved as historical evidence; they do not validate the newly approved runtime package.

The release passed 35 offline tests from a fresh ZIP extraction. Native Windows execution and the actual model result remain pending. Follow [QWEN_RELEASE.json](../QWEN_RELEASE.json) and the [latest actual-result pointer](../qwen-public/LATEST_QWEN_PUBLIC.json).
