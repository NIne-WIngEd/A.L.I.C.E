# Current continuation — a2 stopped before inference

Read the [master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md),
[state](STATE.json) and [verified v104 result review](QWEN_V104_RESULT_REVIEW.json).
Astra remains primary while available; Sol is the continuity fallback.

V104 completed the explicit revision of the same a2. Magnolia job **575155**
started after source reconciliation, verified login-node HTTPS and registry
manifest resolution passed. The worker stopped at initial telemetry publication:
the installed helper exited **74** before runtime preparation or inference.
All 16 public tasks are **NOT_ATTEMPTED**. Qwen remains **NOT_EVALUATED**.
The 66 native Windows selftests passed in 145.030 seconds; they do not establish
live helper compatibility or scientific qualification. The resolved manifest
digest is not evidence of a loaded model, CPU throughput or an effective contract.

The actual result is already in [LATEST_QWEN_PUBLIC](qwen-public/LATEST_QWEN_PUBLIC.json).
Its original failed publication receipts remain historical evidence. The exact
terminal payloads were subsequently appended through the GitHub connector;
see [separate recovery receipt](QWEN_A2_TERMINAL_RECOVERY.json). This restored
terminal visibility without repairing Magnolia's helper or reconciling the
Windows controller state.

The [installed helper review](TELEMETRY_HELPER_REVIEW.json) now identifies
its explicit exit 74 as retry exhaustion in the Git fetch/push loop. Both Git
streams are suppressed inside the helper. An independent local reproduction
also confirms a separate timestamp-driven retry collision (exit 73); it does
not explain the observed exit 74.

Next: use the [read-only transport receipt](TELEMETRY_TRANSPORT_RECEIPT_RELEASE.json)
ZIP and launcher. It captures Git version, fetch dry-run and push dry-run with
the installed SSH wrapper and preserves raw output. Return TRACE_ZIP even if
commands fail. A new observation cannot recover lost historical stderr.
There is no new scientific execution package. Keep closed a2 state unchanged;
do not rerun v104 or infer a3 authority. Inspect the returned Git error before
selecting the transport repair. The helper itself is not invoked by this check.

Keep Gemma v182, Mistral/Granite bindings, GLM's failed calibration, 287
candidates, 63 replacements and one deferred slot. Judge binding and breadth
prerequisite migration require eligible actual calibration evidence. Private
candidate evaluation, acceptance, promotion, training and MC8 remain gated.

Follow the [Phase 2 replacement roadmap](PHASE2_REPLACEMENT_ROADMAP.md).
Stage G is still open; E05 belongs in G+. Final replacement requires Stage J
passing and Rayan's acceptance. Main was reverified unchanged at
`0abaed85873c3f8de04765847eb7700b0e20433f`.
