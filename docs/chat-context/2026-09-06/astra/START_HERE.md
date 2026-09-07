# Current continuation — observed scheduler cause; v104 ready

Read the [master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md), then
[current state](STATE.json) and the [actual provider review](MAGNOLIA_PROVIDER_TRACE_REVIEW.json).
Astra remains primary while available; Sol is the continuity fallback.

The received 04:40:46Z trace verifies that the exact historical-job `squeue -j
575089` query exits 1 with `Invalid job id specified`. The user-wide queue and
historical accounting succeed. A2 remains on v102. Its exact unfinished v103
intent exists locally; no remote revision or checked execution marker exists.
Both inventories are stable. The original invocation's lost stderr remains lost.

The next action is the [v104 provider revision](QWEN_PROVIDER_REVISION_V104.json).
Put its exact ZIP and [launcher](tools/qwen-provider-v104/dist/Start-ALICEAstraQwenQualificationV104.ps1)
in Downloads. Run the launcher and return RESULT_ZIP and the transcript.
It preserves the prior v103 intent, requires the exact observed preparation,
checks live scheduler/source state under the existing lock, records the explicit
revision, then continues the already approved public calibration if eligible.
Do not rerun v103 or delete its intent. No new execution identity is created.

The 66 offline tests passed, including six provider tests driven by captured
responses and labelled adversarial cases. Windows execution of v104, live
revision, model digest, throughput and calibration remain pending. Existing
model/profile, scientific authority, prompt, probe, 16 requests and CPU limits
are unchanged. Qwen remains unevaluated. The actual result pointer still records
a1's TLS failure; its terminal telemetry was separately recovered.

Keep Gemma v182, Mistral/Granite bindings, GLM's failed calibration, the 287
candidates, 63 replacements and one deferred slot unchanged. Judge binding and
breadth prerequisite migration follow only an eligible actual Qwen result.
Private candidate evaluation, acceptance, promotion, training and MC8 remain gated.

Follow the [Phase 2 replacement roadmap](PHASE2_REPLACEMENT_ROADMAP.md).
Stage G remains open; E05 belongs in G+. Final replacement requires Stage J
passing and Rayan's acceptance. Main remains
`0abaed85873c3f8de04765847eb7700b0e20433f`.
