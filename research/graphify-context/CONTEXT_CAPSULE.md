# A.L.I.C.E. Sol Context Capsule

- Request: coverage-frontier-capsule-v2-002
- Question: What is the current N0 scientific state now, including the stable build base, the latest continuity handoff, and any newer unmerged experiment frontier that must be inspected before the next execution?
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md
- Latest observed Magnolia job: 575718
- Continuity stale vs experiment frontier: True
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-causal-arbitration-binding @ 8dab1e5debf48e09ebdbab49815e0938fda25a30 — docs: make N0 architecture capability-first without hard ceilings

## Source pointers

- [73] alice-context:docs/chat-context/2026-09-13/sol/CURRENT_STATE_AND_HANDOFF.md — A.L.I.C.E. Current State and Continuation Handoff — 2026-09-13
- [62] alice-context:docs/chat-context/2026-09-07/sol/SOL_CONTINUATION_HANDOFF.md — A.L.I.C.E. — Sol Continuation Handoff (2026-09-07)
- [60] alice-context:docs/chat-context/2026-09-06/astra/ASTRA_MASTER_CONTINUATION_HANDOFF.md — A.L.I.C.E. — Astra Master Continuation Handoff
  - status:  active continuity contract for Astra-primary / Sol-fallback development
- [60] alice-context:docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md — N0 Endpoint Repair and Downstream Arbitration Handoff
  - status: `FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`.
- [60] alice-eipm-v1-build:docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md — A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13
  - status:  research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.
  - supersession:  this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.
- [56] alice-context:docs/chat-context/2026-08-31/src/S02-memory-handoff.md — A.L.I.C.E. Stage G / G+ Memory Architecture Handoff
- [56] alice-context:docs/chat-context/2026-09-13/sol/N0_FRONTIER_RESEARCH_DECISION.md — A.L.I.C.E. N0 Frontier Research Decision — 2026-09-13
  - status:  continuity handoff. No permanent EIPM weights created.
- [54] alice-context:docs/chat-context/2026-09-10/sol/OWNER_OVERRIDE_PERSONALITY_MODEL_MAINSTREAMING.md — Owner override: mainstream the A.L.I.C.E. personality-model build
- [54] alice-eipm-v1-build:docs/RESEARCH_FRONTIERS.md — A.L.I.C.E. Research Frontiers Register
- [52] alice-context:docs/chat-context/2026-08-31/CURRENT_STATE.md — A.L.I.C.E. Current Working State — 2026-08-31
- [52] alice-context:docs/chat-context/2026-09-04/ALICE_MC10D_CONTINUATION_HANDOFF_20260904.md — A.L.I.C.E. MC10D Continuation Handoff — 2026-09-04
- [52] alice-context:docs/chat-context/2026-09-05/MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_V181.md — MC10D Public Judge Qualification Rebase v1.8.1 — 2026-09-05

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
