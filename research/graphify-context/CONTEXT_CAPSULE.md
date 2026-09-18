# A.L.I.C.E. Sol Context Capsule

- Request: coverage-frontier-capsule-v2-003
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

- [90] alice-context:docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md — N0 Endpoint Repair and Downstream Arbitration Handoff
  - status: `FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`.
- [75] alice-eipm-v1-causal-arbitration-binding:docs/eipm/n0/N0_V0_2_PRODUCTION_BUILD_PLAN.md — N0 v0.2 Production Build Plan
  - status:  implementation active; GPU training held until CPU/data gates pass
- [64] alice-eipm-v1-build:docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md — A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13
  - status:  research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.
  - supersession:  this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.
- [58] alice-context:docs/chat-context/2026-09-13/sol/CURRENT_STATE_AND_HANDOFF.md — A.L.I.C.E. Current State and Continuation Handoff — 2026-09-13
- [58] alice-eipm-v1-build:docs/RESEARCH_FRONTIERS.md — A.L.I.C.E. Research Frontiers Register
- [56] alice-eipm-v1-build:docs/eipm/n0/N0_V0_2_PRODUCTION_BUILD_PLAN.md — N0 v0.2 Production Build Plan
  - status:  implementation active; GPU training held until CPU/data gates pass
- [56] alice-eipm-v1-build:docs/eipm/n0/runtime-results/N0_FRONTIER_RESEARCH_AUDIT_20260914.md — N0 Frontier Research Audit — 2026-09-14
  - status: hold the step-1000 -> step-2500 GPU job pending N0 v0.2 redesign**.
- [56] alice-eipm-v1-causal-arbitration-binding:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [50] alice-eipm-v1-build:docs/eipm/EIPM_EXPANSION_GENERATOR_HANDOFF_SPEC_v0.1.md — EIPM Expansion Generator Handoff Spec v0.1
  - status:  implementation handoff; subordinate to owner-ratified EIPM hard rules
- [48] alice-eipm-v1-build:docs/eipm/EIPM_CURATED_FRONTIER_V1_RECEIPT.md — EIPM Curated Frontier v1 Receipt
  - status:  curation complete enough for targeted gap filling; not training-authorized.
- [48] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-causal-arbitration-full-stack-binding-v0.1.md — N0 downstream causal arbitration: full-stack graph binding v0.1
- [47] alice-context:docs/chat-context/2026-09-07/sol/SOL_CONTINUATION_HANDOFF.md — A.L.I.C.E. — Sol Continuation Handoff (2026-09-07)

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
