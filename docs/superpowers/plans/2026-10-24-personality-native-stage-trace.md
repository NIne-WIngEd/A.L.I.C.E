# Native conditional-route trace implementation plan

> **For agentic workers:** use superpowers:executing-plans for inline implementation under the owner's existing continuation authorization; use an independent code reviewer before dispatch.

**Goal:** locate where native conditional effects diminish before choosing a learned correction.

**Architecture:** passive hooks capture actual N2 stage tensors. Within one frozen frame session, exact readout caching removes redundant feature projection from counterfactual replays. Preserve complete original custody and prior output controls.

**Tech Stack:** current Python/PyTorch2.7 native model and owner-isolated Magnolia runtime.

**Spec:** `docs/eipm/PERSONALITY_CONDITIONAL_ROUTE_DIAGNOSIS_2026-10-24.md`.

## Constraints and review focus

Frozen Gemma/N1/N2/N3; no optimizer, architecture/loss change, new teacher or checkpoint promotion. Unknown facts and all masks remain intact. Diagnostics contain targets and cannot qualify behavior. Dispatched older files are immutable. Check same-frame/query cache ownership, complete hook removal, unchanged native execution mode, all80 contexts and private-only stage details.

## Task1: native stage observer and bounded readout replay

Files: new `scripts/eipm/gemma_n0/conditional_stage_trace.py` and `tests/personality/test_conditional_stage_trace.py`.

- [x] Test actual native packet equality with passive hooks and cached readout replay.
- [x] Implement `capture_stages(model, frame, call)` returning result, named tensor/mask observations and readout cache; `replay_readout(model, cache)` scoped context; `compare_stages(before, after)` finite masked effect summaries.
- [x] Verify original failure/refusal cases: different query/bank, changed model phase, restored methods/hooks after an exception, mask mismatch, known conditional change versus sham. No private input or training needed.

## Task2: complete owner worker and evidence

Files: new `scripts/eipm/gemma_n0/magnolia_conditional_stage_trace_20261024.py` and matching wrapper; extend public tests with actual worker entry/aggregation checks.

- [x] Preserve source/checkpoint loader, original startup guard and nondestructive live verifier from the successful immutable worker.
- [x] Implement pinning of579470 result/all five private files and each context's prior packet/objective reproduction. Add full available-current N1 facts without inventing unknowns. Protected execution pending.
- [x] Implement named native stages and per-context ranking transitions, with family-level data private and aggregates public. Require closing source/weight custody. Protected execution pending.
- [x] Run affected tests, independent review, three governance validators and exact source/wrapper pins. Dispatch once under mxrayan: job579662. One startup observation is recorded separately.
- [x] Seed FBM v24, preserving all44 prior trace rows and the full replay prefix;45 rows now. Prepare result/plan/code for existing PR100. Exact publication and continuation status are recorded outside this pre-publication plan.

Stop at the external result after publishing and updating the continuation;
do not poll or start speculative training.

Local affected verification:53 tests and9 subtests passed. Independent review found
no blocker at the dispatch pins and separately checked10 focused tests and native
cleanup/bank-refusal probes. These prove diagnostic mechanics, not learned
personality or full protected execution. Observe post-residual normalization
boundaries; do not claim separate residual/normalization attribution.
