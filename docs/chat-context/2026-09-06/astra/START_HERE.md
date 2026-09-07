# Current continuation — old Git publisher cause confirmed

Read the [master handoff](ASTRA_MASTER_CONTINUATION_HANDOFF.md),
[state](STATE.json), and [actual Git trace review](TELEMETRY_GIT183_RESULT_REVIEW.json).
Astra remains primary while available; Sol is the continuity fallback.

The 05:57:25Z receipt confirms Git 1.8.3.1, successful fetch dry-run and
non-fast-forward push rejection. Exact observed hashes identify cached main and
origin/main at ec4d378 while FETCH_HEAD is already at 3f42de3. The installed helper
fetches origin main and then builds on origin/main. Before Git 1.8.4 that fetch
does not refresh the tracking reference. The actual references, installed code
and upstream release notes establish the defect. Lost earlier stderr stays lost.

Next: the [publisher repair](TELEMETRY_PUBLISHER_REPAIR_RELEASE.json).
Save its exact ZIP and launcher in Downloads and run the launcher. It checks
the observed helper/wrapper and closed a2 hashes, runs seven tests on Magnolia's
actual Git, backs up the original helper and installs the exact replacement.
It then verifies the already-failed terminal snapshot and an identical retry.
Return RESULT_ZIP and the transcript, including on failure.

The replacement explicitly refreshes the tracking ref and uses the fetched parent.
It preserves cached metadata, records raw Git receipts and uses a private index
to avoid resetting the shared checkout. A raw push of the cached local HEAD is
therefore not the repaired publisher's verification path. Use its actual receipt.
Nine local release checks passed. Actual installation/publication remain pending.

Qwen is still NOT_EVALUATED. A2 job 575155 is closed with zero probe/task attempts.
V104 revision and login-node TLS/manifest preflight succeeded before telemetry
stopped the worker. The manifest is resolved; runtime/model/throughput and the
16 public requests are still unexecuted. Do not rerun v104, reset a2, or infer a3
authority. A future eligible execution must bind the installed helper and retain
its raw command journal. The repair ZIP hash never replaces the Qwen workload hash.

Keep the actual public result pointer and separate terminal recovery history.
Gemma v182, Mistral/Granite bindings, GLM's failed calibration, 287 candidates,
63 replacements and one deferred slot remain unchanged. Private pointwise work,
breadth, acceptance, promotion, training and MC8 remain gated.

Follow the [Phase 2 replacement roadmap](PHASE2_REPLACEMENT_ROADMAP.md).
Stage G remains open; E05 belongs in G+. Final replacement requires accepted
Stage J. Main remains frozen at 0abaed85873c3f8de04765847eb7700b0e20433f.
