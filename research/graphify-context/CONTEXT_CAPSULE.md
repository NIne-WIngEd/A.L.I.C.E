# A.L.I.C.E. Sol Context Capsule

- Request: n0-missing-evidence-localization-ready-20260918-001
- Question: What is the exact current N0 state after final frozen challenge job 575794 and what single fresh non-challenge localization run is now authorized? Identify the maximal experiment frontier, active handoff, failing gate/family, anti-loop constraints, and the ready Magnolia localization runner.
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: True
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-missing-evidence-localization @ 6183ed9be59a75fc7c5118312099a0875b8bab6d — fix(n0): separate graph selection from value geometry in localization

## Source pointers

- [152] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md — N0 Final Frozen Challenge Ready Handoff
  - status:  one final frozen latent challenge authorized and execution-bound; no promotion/training/scale/N0 completion authorized
- [125] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_FAIL_AND_MISSING_EVIDENCE_LOCALIZATION.md — N0 Final Frozen Challenge Valid FAIL + Fresh Missing-Evidence Localization
  - status:  final frozen challenge completed; valid single-gate failure; do not rerun or change thresholds; fresh non-challenge causal-path localization staged
- [88] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_TEACHER_PROBE_JOB_575565.md — Magnolia single-P100 N0 teacher diagnostic — job 575565
- [86] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md — Magnolia 2×P100 N0 Runtime Smoke — Job 575527
- [86] alice-eipm-v1-missing-evidence-localization:.github/workflows/n0-missing-evidence-localization-contract-check.yml — n0-missing-evidence-localization-contract-check.yml
- [85] fable-builder-model:docs/fable-builder/traces/2026-09-18_n0_final_frozen_fail_missing_evidence_localization.jsonl — 2026-09-18_n0_final_frozen_fail_missing_evidence_localization.jsonl
- [83] alice-telemetry:telemetry/n0-final-frozen-challenge-valid-fail-575794.json — n0-final-frozen-challenge-valid-fail-575794.json
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_DDP_JOB_575537.md — Magnolia 2×P100 N0 DDP Mechanics — Job 575537
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_N0_CONTINUATION_JOB_575548.md — Magnolia N0 continuation — job 575548
  - status:  completed successfully
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_REAL_N0_JOB_575546.md — Magnolia 2×P100 Real N0 Training — Job 575546
  - status:  completed successfully; durable weights retained; LR scheduler defect identified before continuation
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_REPAIR_PROBE_JOB_575571.md — Magnolia P100 N0 Failure-Driven Repair Probe — Job 575571
  - status: completed payload
- [75] alice-eipm-v1-missing-evidence-localization:docs/research/eipm-n0-downstream-causal-arbitration-magnolia-execution-v0.1.md — N0 downstream causal arbitration — Magnolia execution contract

## Graphify navigation hints

- skipped: document/branch/private routing was sufficient

## External/private routing

- Recommended: True
- Reason: host-specific operational history may contain newer failure lessons than public receipts

## Trust contract

- This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.
- Open consequential claims in the original branch/path source before acting.
- Keep branch-specific, experimental, superseded and canonical states distinct.
- Graphify nodes are navigation hints only.
- If continuity is stale relative to unmerged experiment heads, inspect those heads before execution.
- If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.
