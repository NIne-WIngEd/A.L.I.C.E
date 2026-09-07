# Current continuation — a1 runtime failure verified; v102 infrastructure successor ready

Rayan explicitly approved the exact Qwen fallback draft with the message **“approve”**. That decision is already recorded in context commit `591dbc0781b1df8e11f0eea7d743952724722bae`. Do not ask for this activation again. Read [owner approval](fallback/OWNER_APPROVAL.json), the [effective approved amendment](fallback/QWEN_FALLBACK_AMENDMENT_V1_APPROVED.json), [current state](STATE.json) and [release identity](QWEN_RELEASE.json).

The actual v101 result is now verified. Windows upload/download worked and Slurm job `575089` ran. It stopped at runtime preparation because Python could not verify the HTTPS issuer chain. No model, throughput probe or task attempt occurred. The independently rebuilt summary exactly matches the user's published summary. Read [actual findings](QWEN_V101_RESULT_FINDINGS.md) and [structured review](QWEN_V101_RESULT_REVIEW.json).

The ledger's initial heartbeat remained immutable. Later updates returned exit 73 because the controller reused the same run ID for different bytes. The terminal failure has now been recorded under a separate immutable snapshot. The original run and uploaded ZIP remain unchanged. Both failures repeat safeguards already documented in the September 4 handoff; those safeguards are now implemented and covered by regression tests.

## The next authorized action

Use the [v102 Windows launcher](tools/qwen-runtime-v102/README.md) with its versioned ZIP. [Release identity](QWEN_RUNTIME_SUCCESSOR_V102.json) records exact package/launcher hashes and the explicit infrastructure successor `alice-qwen38-a2-c926d9e355dd`. The same approved public calibration applies; no renewed model activation is needed. The source job stays closed. The new local state is under `qwen-fallback-a2`.

Before submission the package checks the exact source failure, absence of any prior inference intent and live scheduler termination. It verifies HTTPS using the earlier explicit CA policy and freezes the full public model manifest. A failed check prevents sbatch. The allocation remains Magnolia CPU: 20 CPUs, 48 GiB, exclusive, partition node, QOS normal and six hours. The approved Qwen profile and all probe/task request hashes remain unchanged. No Kaggle/GPU route is enabled.

All 48 tests passed from the final ZIP. Tests model the immutable ledger through the complete simulated worker lifecycle. Actual v102 Windows/Magnolia execution is pending. The controller prints each real telemetry snapshot URL. Later invocations attach or recover publication from the same new execution; they never repeat a task intent or create a further automatic attempt.

Consult the [latest actual-result pointer](qwen-public/LATEST_QWEN_PUBLIC.json) first in future chats. It currently points to the verified a1 infrastructure stop. Preserve that actual result while the a2 run is pending. A prepared v102 package is not a qualification result.

After the real a2 result arrives, recompute the complete ZIP against its exact package authority and requests. A passing Qwen calibration still requires verification of the exact Gemma, Qwen, Mistral and Granite receipts before constructing a new four-family binding. Do not edit v186 into success. Breadth v104 still needs its explicit prerequisite migration with exact eligible receipt hashes. Private pointwise work, acceptance, promotion, training and MC8 remain gated.

## Preserved findings and project direction

[GLM v186 findings](GLM_V186_FINDINGS.md) are complete. Its actual rows produced 9/16 verdict matches, 3/7 critical decisions, and 3/5 mandatory anchors. It approved both fake-history anchors and predicted no HOLD. The actual forensic summary is preserved at context commit `167e1df631a03e3fd51985d4b29e298016821cfb`. No further GLM evidence collection or inference is needed for that completed review.

Keep valid Gemma v182 evidence and the preserved Mistral/Granite bindings. Preserve 63 replacements, one deferred item and the 287-candidate pool. Main remains `0abaed85873c3f8de04765847eb7700b0e20433f`. Private pointwise execution, breadth readiness, acceptance, promotion and training are not granted by Qwen activation. The repeated 16-task suite remains calibration, not independent certification.

The accepted broader audit remains in [audit decisions](AUDIT_DECISIONS.md), [full snapshot](AUDIT_SNAPSHOT.md), [source ledger](AUDIT_SOURCE_LEDGER.json), [working agreement](WORKFLOW.md), [experiment ledger](EXPERIMENT_LEDGER.json), and [prospective evaluation design](PROSPECTIVE_JUDGE_VALIDATION.md). Preserve learned identity/AMFM, complete memory and Mission Graph semantics, source/person/host/relationship boundaries, No Disposable Cognition, and G → G+ → Fable ordering. This public judge repair is a bounded dependency within that plan.

[Previous handoff](START_HERE_BEFORE_QWEN_APPROVAL.md) and [previous state](STATE_BEFORE_QWEN_APPROVAL.json) preserve the exact pre-approval checkpoint. Statements there that Qwen activation is still pending are historical. Earlier project context remains under the dated context directories and the August source index.

[Handoff](START_HERE_BEFORE_QWEN_V101.md), [state](STATE_BEFORE_QWEN_V101.json) and [release](QWEN_RELEASE_BEFORE_V101.json) at initial v100 publication preserve the previous checkpoint. Their statements that native Windows execution is pending are historical. That original source workload remains preserved. The active execution package is v102.

[Pre-v102 handoff](START_HERE_BEFORE_QWEN_V102.md) and [state](STATE_BEFORE_QWEN_V102.json) preserve the preceding checkpoint. Their pending-v101 execution statements are historical.
