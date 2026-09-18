# A.L.I.C.E. Sol Context Capsule

- Request: n0-post-arbitration-final-challenge-ready-20260918-001
- Question: What is the exact current N0 state after Magnolia job 575792, which experiment frontier is current, what did downstream causal arbitration conclude, and what is the single authorized next model execution?
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: False
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-causal-arbitration-binding @ e535a2a63c2c2bdfe13843171b391096c4c5019e — ci(n0): guard final frozen challenge execution contract

## Source pointers

- [133] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-magnolia-execution-v0.1.md — N0 downstream causal arbitration — Magnolia execution contract
- [105] alice-eipm-v1-causal-arbitration-binding:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [103] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [99] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-causal-arbitration-full-stack-binding-v0.1.md — N0 downstream causal arbitration: full-stack graph binding v0.1
- [83] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md — N0 Final Frozen Challenge Ready Handoff
  - status:  one final frozen latent challenge authorized and execution-bound; no promotion/training/scale/N0 completion authorized
- [80] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_READY_FOR_CALIBRATION.md — N0 causal-arbitration frontier ready for Magnolia calibration
- [80] alice-eipm-v1-downstream-arbitration:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [80] alice-eipm-v1-post-arbitration-gate:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [78] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_TEACHER_PROBE_JOB_575565.md — Magnolia single-P100 N0 teacher diagnostic — job 575565
- [75] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md — N0 Causal Arbitration Frontier Calibration Handoff
- [75] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_N0_CONTINUATION_JOB_575548.md — Magnolia N0 continuation — job 575548
  - status:  completed successfully
- [75] alice-telemetry:telemetry/n0-downstream-causal-arbitration-frontier-20260917.json — n0-downstream-causal-arbitration-frontier-20260917.json

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
