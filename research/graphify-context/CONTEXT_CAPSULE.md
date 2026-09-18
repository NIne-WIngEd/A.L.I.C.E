# A.L.I.C.E. Sol Context Capsule

- Request: coverage-g4-operational-failure-002
- Question: What is the current working Magnolia execution route, which routes are dead, and what PowerShell or Kaggle failure lesson must not be repeated before I run the next N0 job?
- Source: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132

## Active mission

- Mission schema: alice-context-active-mission-state-v1
- Source commit: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md
- Latest observed Magnolia job: 575718
- Execution rule: Implementation config governs committed model/build state. A newer continuity overlay may supersede its observed runtime status and next-action wording. Open the original handoff before execution.

## Source pointers

- [71] alice-context:docs/chat-context/2026-09-09/sol/MC10D_MAGNOLIA_P100X2_ROUTE_QUALIFIED_NEXT_RUNTIME_GATE.md — MC10D Magnolia 2xP100 route qualified — next runtime gate
- [62] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_SMOKE_JOB_575527.md — Magnolia 2×P100 N0 Runtime Smoke — Job 575527
- [58] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_REPAIR_PROBE_JOB_575571.md — Magnolia P100 N0 Failure-Driven Repair Probe — Job 575571
  - status: completed payload
- [55] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_REAL_N0_JOB_575546.md — Magnolia 2×P100 Real N0 Training — Job 575546
  - status:  completed successfully; durable weights retained; LR scheduler defect identified before continuation
- [54] alice-context:docs/chat-context/2026-09-06/astra/tools/runtime-route-survey/package/ALICE_MAGNOLIA_RUNTIME_ROUTE_v1.0.0/README.md — Magnolia runtime route survey
- [53] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_N0_CONTINUATION_JOB_575548.md — Magnolia N0 continuation — job 575548
  - status:  completed successfully
- [51] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100X2_DDP_JOB_575537.md — Magnolia 2×P100 N0 DDP Mechanics — Job 575537
- [51] alice-eipm-v1-build:docs/eipm/n0/runtime-results/MAGNOLIA_P100_N0_TEACHER_PROBE_JOB_575565.md — Magnolia single-P100 N0 teacher diagnostic — job 575565
- [50] alice-context:docs/chat-context/2026-09-06/astra/MAGNOLIA_RUNTIME_ROUTE_RELEASE.json — MAGNOLIA_RUNTIME_ROUTE_RELEASE.json
- [50] alice-context:docs/chat-context/2026-09-07/sol/RUNTIME_ROUTE_RESULT_SUMMARY.md — Magnolia Runtime Route — Observed Result Summary
- [47] alice-eipm-v1-build:docs/eipm/n0/MAGNOLIA_P100X2_RUNTIME_SMOKE_2026-09-13.md — Magnolia 2×P100 N0 Runtime Smoke — 2026-09-13
  - status:  ready for owner execution
- [45] alice-context:docs/chat-context/2026-09-09/sol/MC10D_MAGNOLIA_CLOSED_KAGGLE_DURABLE_SUCCESSOR.md — MC10D Magnolia closed — Kaggle durable simulation/falsification successor

## Graphify navigation hints

- Regression for Magnolia job 575673 mixed-autocast evaluation failure. -> tests/eipm/test_n0_adaptive_multi_view_latent_pool_objectives_v0_2.py:L65
- magnolia_cpu_n0_v02_preflight.sh -> scripts/eipm/n0/magnolia_cpu_n0_v02_preflight.sh:L1
- magnolia_cpu_n0_v02_tokenizer.sh -> scripts/eipm/n0/magnolia_cpu_n0_v02_tokenizer.sh:L1
- test_candidates_never_enter_derived_indexes_before_promotion() -> tests/phase2/test_memory_candidate_security_gates.py:L312
- CurrentClaimProjection -> src/cognitive_kernel/claim_contracts.py:L755
- DeterministicInformationResearchModeAdapter -> src/alice_information/research_mode.py:L482
- _execution_id() -> src/alice_information/research_execution.py:L277
- _failure() -> src/alice_information/brave_search_live.py:L72
- kaggle_cpu_real_lineage.sh -> scripts/eipm/n0/kaggle_cpu_real_lineage.sh:L1
- n0/__init__.py -> src/alice_personality/n0/__init__.py:L1
- _next_start() -> src/alice_vault/chunking.py:L129
- test_repeated_request_is_idempotent() -> tests/phase2/test_memory_deletion.py:L291
- test_reliability_prior_can_route_between_equivalent_views() -> tests/eipm/test_n0_cross_context_fusion.py:L79
- .test_auto_review_routes_sensitive_and_contradictory_to_manual() -> tests/phase1/test_auto_review.py:L44
- run() -> tests/phase4/test_information_final_evaluation_runtime.py:L47
- cognitive_kernel/__init__.py -> src/cognitive_kernel/__init__.py:L1

## External/private routing

- Recommended: True
- Reason: host-specific operational history may contain newer failure lessons than public receipts

## Trust contract

- This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.
- Open consequential claims in the original branch/path source before acting.
- Keep branch-specific, experimental, superseded and canonical states distinct.
- Graphify nodes are navigation hints only.
- If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.
