# A.L.I.C.E. Sol Context Capsule

- Request: n0-qrr-infra-retry-575799-20260918-001
- Question: Resolve current N0 QRR state after Magnolia job 575799 failed in preflight before training. Confirm it is infrastructure-only due to stale 575798 receipt key, identify corrected QRR head and CI result, preserve the partial original output, and identify the single authorized retry directory/action. Confirm architecture unchanged and no model evidence from 575799.
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-18/sol/N0_QUERY_RELATION_ROLE_ROUTER_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: False
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-query-relation-role-router @ a4ba0e7b8629718947d7126f8aa86021aa1cb87c — ci(n0): guard exact 575798 immutability receipt key

## Source pointers

- [92] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_TEACHER_PROBE_JOB_575565.md — Magnolia single-P100 N0 teacher diagnostic — job 575565
- [91] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_N0_CONTINUATION_JOB_575548.md — Magnolia N0 continuation — job 575548
  - status:  completed successfully
- [90] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_REAL_N0_JOB_575546.md — Magnolia 2×P100 Real N0 Training — Job 575546
  - status:  completed successfully; durable weights retained; LR scheduler defect identified before continuation
- [90] alice-telemetry:telemetry/n0-qrr-preflight-infra-failure-575799.json — n0-qrr-preflight-infra-failure-575799.json
- [88] alice-eipm-v1-build:docs/eipm/n0/HOSTED_MODEL_OUTPUT_TRAINING_BOUNDARY_2026-09-13.md — Hosted-Model Output Training Boundary — Owner Authorization Correction — 2026-09-13
  - status:  active owner-directed lineage rule; supersedes the earlier advisory-only restriction in this file
- [85] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md — Magnolia 2×P100 N0 Runtime Smoke — Job 575527
- [83] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_DDP_JOB_575537.md — Magnolia 2×P100 N0 DDP Mechanics — Job 575537
- [81] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_REPAIR_PROBE_JOB_575571.md — Magnolia P100 N0 Failure-Driven Repair Probe — Job 575571
  - status: completed payload
- [79] alice-context:docs/chat-context/2026-09-18/sol/N0_QUERY_RELATION_ROLE_ROUTER_HANDOFF.md — N0 Query–Relation Role Router Handoff
  - status:  role-residual v0.2 failed at dev without heldout exposure; external architecture research completed; query–relation role router v0.3 is CI-ready for one P100 run
- [75] alice-eipm-v1-query-relation-role-router:docs/research/eipm-n0-downstream-causal-arbitration-magnolia-execution-v0.1.md — N0 downstream causal arbitration — Magnolia execution contract
- [73] alice-eipm-v1-query-relation-role-router:configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.12.json — alice_n0_latent_pool_stage_state_v0.12.json
- [71] alice-eipm-v1-query-relation-role-router:docs/research/eipm-n0-query-relation-role-router-external-research-v0.1.md — N0 Query–Relation Role Router — External Architecture Research v0.1

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
