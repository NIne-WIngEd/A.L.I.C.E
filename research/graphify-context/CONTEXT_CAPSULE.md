# A.L.I.C.E. Sol Context Capsule

- Request: current-n0-calibration-retry-575759-001
- Question: What is the exact current N0 state after Magnolia calibration job 575759 failed before model execution, what was the root cause, which frontier commit fixes it, and what is the next safe execution?
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: False
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-causal-arbitration-binding @ 6c930788da42ad9964c4b5bf6067be3b7d6bfd06 — ci(n0): smoke arbitration import roots without GPU

## Source pointers

- [113] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md — N0 Causal Arbitration Frontier Calibration Handoff
- [101] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-magnolia-execution-v0.1.md — N0 downstream causal arbitration — Magnolia execution contract
- [80] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_READY_FOR_CALIBRATION.md — N0 causal-arbitration frontier ready for Magnolia calibration
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_N0_CONTINUATION_JOB_575548.md — Magnolia N0 continuation — job 575548
  - status:  completed successfully
- [77] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_REAL_N0_JOB_575546.md — Magnolia 2×P100 Real N0 Training — Job 575546
  - status:  completed successfully; durable weights retained; LR scheduler defect identified before continuation
- [73] alice-eipm-v1-build:docs/MEMORY_M2_EXECUTION_PLAN.md — Memory M2 Execution Plan — Contract and Full-Memory Parallelism
  - status:  Owner-directed active execution clarification
- [73] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_DDP_JOB_575537.md — Magnolia 2×P100 N0 DDP Mechanics — Job 575537
- [73] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md — Magnolia 2×P100 N0 Runtime Smoke — Job 575527
- [71] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_TEACHER_PROBE_JOB_575565.md — Magnolia single-P100 N0 teacher diagnostic — job 575565
- [67] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_REPAIR_PROBE_JOB_575571.md — Magnolia P100 N0 Failure-Driven Repair Probe — Job 575571
  - status: completed payload
- [67] feat/memory-m2-closeout-shadow-admission:docs/MEMORY_M2_EXECUTION_PLAN.md — Memory M2 Execution Plan — Contract and Full-Memory Parallelism
  - status:  Owner-directed active execution clarification
- [67] feat/memory-shadow-migration-stage-d:docs/MEMORY_M2_EXECUTION_PLAN.md — Memory M2 Execution Plan — Contract and Full-Memory Parallelism
  - status:  Owner-directed active execution clarification

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
