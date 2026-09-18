# A.L.I.C.E. Sol Context Capsule

- Request: current-n0-calibration-frontier-001
- Question: What is the exact current N0 scientific state and next execution now, including the tested causal-arbitration frontier, active handoff, pre-GPU validation, and whether calibration must happen before candidate arbitration?
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md
- Latest observed Magnolia job: 575718
- Continuity stale vs experiment frontier: False
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-causal-arbitration-binding @ d49d6e7f45c15905ca8ec5a33d1ac14c3aef3d59 — ci(n0): verify arbitration env crosses udocker boundary

## Source pointers

- [105] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md — N0 Causal Arbitration Frontier Calibration Handoff
- [99] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_READY_FOR_CALIBRATION.md — N0 causal-arbitration frontier ready for Magnolia calibration
- [99] alice-context:docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md — N0 Endpoint Repair and Downstream Arbitration Handoff
  - status: `FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`.
- [96] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-magnolia-execution-v0.1.md — N0 downstream causal arbitration — Magnolia execution contract
- [83] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-causal-arbitration-full-stack-binding-v0.1.md — N0 downstream causal arbitration: full-stack graph binding v0.1
- [79] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [75] alice-eipm-v1-causal-arbitration-binding:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [64] alice-eipm-v1-causal-arbitration-binding:docs/eipm/n0/N0_V0_2_PRODUCTION_BUILD_PLAN.md — N0 v0.2 Production Build Plan
  - status:  implementation active; GPU training held until CPU/data gates pass
- [58] alice-context:docs/chat-context/2026-09-13/sol/CURRENT_STATE_AND_HANDOFF.md — A.L.I.C.E. Current State and Continuation Handoff — 2026-09-13
- [58] alice-eipm-v1-build:docs/MEMORY_M2_EXECUTION_PLAN.md — Memory M2 Execution Plan — Contract and Full-Memory Parallelism
  - status:  Owner-directed active execution clarification
- [58] alice-eipm-v1-build:docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md — A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13
  - status:  research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.
  - supersession:  this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.
- [56] alice-eipm-v1-build:docs/PHASE_4_LIVE_RESEARCH_EXECUTION.md — Phase 4 P4.10b — Live Governed Research Execution
  - status: additive implementation profile. P4.6a, P4.7a, and P4.7b remain frozen fixture compatibility profiles.

## Graphify navigation hints

- skipped: document/branch/private routing was sufficient

## External/private routing

- Recommended: False
- Reason: None

## Trust contract

- This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.
- Open consequential claims in the original branch/path source before acting.
- Keep branch-specific, experimental, superseded and canonical states distinct.
- Graphify nodes are navigation hints only.
- If continuity is stale relative to unmerged experiment heads, inspect those heads before execution.
- If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.
