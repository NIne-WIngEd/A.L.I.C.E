# Current continuation — v102 source-origin stop traced, 7 September 2026

The supplied terminal passed all 48 v102 tests on native Windows, then stopped
at `Source failure evidence differs: package-manifest.json` with exit 76.
The exact released code places this stop before its new submission, HTTPS
preflight or inference. The original collector and v102 verifier reproduce the
same error locally. [Diagnosis and project dependency](QWEN_V102_ROOT_CAUSE_AND_DIRECTION.md),
[terminal observation](QWEN_V102_TERMINAL_OBSERVATION.json), [current state](STATE.json).

## Next action

Use the [read-only trace](tools/qwen-v102-trace/README.md) and
[its exact release identity](QWEN_READ_ONLY_TRACE_RELEASE.json). Return its
`TRACE_ZIP` and transcript. It observes the real source/package file origins,
a2 state, scheduler and the next HTTPS boundaries, without changing either run
or submitting compute. Do not rerun v102 to repair this deterministic stop.
No replacement qualification workload or automatic a3 execution was released.

The bug is in Astra's recovery code and fixture: an exported result combines
run files and package projections. The tests extracted that combined ZIP into
a simulated live run. The old producer reads `PACKAGE_MANIFEST.json` from its
package directory; it does not place `package-manifest.json` in the live run.
The log cannot distinguish an absent file, symlink and hash mismatch. The
read-only observation resolves that final host fact and checks for any later
a2 activity before a repair is selected.

The a2 descriptor is created or checked before this stop and binds v102's exact
package hash. Replacing its package under the same descriptor would be another
identity error. Preserve both the old descriptor and the evidence when deciding
an explicit revision; never delete state to obtain a fresh start.

## Authority and actual outcomes

Rayan already approved the exact Qwen fallback with **“approve”**, recorded in
commit `591dbc0781b1df8e11f0eea7d743952724722bae`. [Owner approval](fallback/OWNER_APPROVAL.json)
and [approved amendment](fallback/QWEN_FALLBACK_AMENDMENT_V1_APPROVED.json) remain
valid. No renewed activation is needed for that same scientific plan. The next
tool is a diagnostic, not a model execution. Historical identities remain in
[Qwen release](QWEN_RELEASE.json) and [v102 release](QWEN_RUNTIME_SUCCESSOR_V102.json).

The [latest actual-result pointer](qwen-public/LATEST_QWEN_PUBLIC.json) still
points to a1's verified TLS preparation failure. Job `575089` ran through the
v101 path repair but made no model, probe or task attempt. Its exact result ZIP,
independent reconstruction and immutable terminal ledger recovery are preserved
in the [source result review](QWEN_V101_RESULT_REVIEW.json). Do not replace that
actual result with a fabricated a2 result or interpret zero attempts as accuracy.

[GLM v186 findings](GLM_V186_FINDINGS.md) remain 9/16 verdict matches, 3/7 critical
decisions and 3/5 mandatory anchors, with no HOLD predictions. It approved both
fake-history anchors. That completed review does not need more GLM inference.
Preserve Gemma v182 rows, Mistral/Granite bindings, all 287 candidates, 63
replacements and the deferred item. Slot64 is resolved and must not regenerate.

Only an actual passing Qwen calibration can support review of the four exact
family/profile receipts for a successor binding. Breadth still needs explicit
prerequisite migration. Private pointwise work, acceptance, promotion, training
and MC8 remain gated. Keep the approved CPU profile and throughput budget gate;
no silent model, provider, quantization, seed or task substitution is allowed.

## Larger goal and evidence still needed

ALICE remains one continuing entity with learned identity and AMFM, complete
governed memory, source/host/relationship/self boundaries, full Mission Graph
semantics and replaceable executive models. Judge calibration supports synthetic
candidate evaluation; it does not demonstrate learned cognition. E01–E07 remain
unmeasured in this continuation. Preserve the accepted G → G+ → Fable/Friday
ordering and No Disposable Cognition. Evaluation design and deployment planning
can progress without calling this infrastructure work a cognitive result.

Read [accepted audit decisions](AUDIT_DECISIONS.md), [full audit](AUDIT_SNAPSHOT.md),
[source ledger](AUDIT_SOURCE_LEDGER.json), [workflow and failure lessons](WORKFLOW.md),
[experiment ledger](EXPERIMENT_LEDGER.json), and [prospective judge validation](PROSPECTIVE_JUDGE_VALIDATION.md).
Main remains `0abaed85873c3f8de04765847eb7700b0e20433f`.

The diagnostic passed six focused Linux tests from its final ZIP. The literal
PowerShell verifier and SSH scripts were checked. Native diagnostic execution
and current Magnolia observations are pending. Earlier native v102 tests passed
48; the fixture defect explains their practical limit. These are separate facts.

[Pre-trace handoff](START_HERE_BEFORE_QWEN_V102_TRACE.md) and
[state](STATE_BEFORE_QWEN_V102_TRACE.json) preserve the previous v102-ready checkpoint.
Earlier approval, v100 and v101 handoffs and dated source archives remain intact.
Their old pending/ready instructions are historical.
