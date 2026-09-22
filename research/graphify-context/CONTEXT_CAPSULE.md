# A.L.I.C.E. Sol Context Capsule

- Request: n0-step-role-traversal-control-capability-audit-20260922-036
- Question: Deep-audit exact N0 head alice-eipm-v1-n0-full-envelope-foundation-build-v1@073819f44ebb79c876140119bcc1e8ebd34b6a37 for whether role, traversal, and control are intentionally program-global or accidentally form a capability ceiling. Trace SemanticOperatorState.step_factor_distributions -> SemanticOperatorQSREAdapter -> FullEnvelopeOperatorState -> Binder -> FullEnvelopeQSREExecutorV1 -> behavioral compiler/objectives -> final-v2 contract. The semantic foundation predicts step-conditioned distributions for every factor bank, but the adapter currently preserves step-conditioned direction and modifiers while collapsing role, traversal, and control to program-global distributions. Determine from architecture decisions, historical failures, curricula, and executor mechanics whether valid target workloads require role/traversal/control to change across steps (examples: source read then target read; local select then path/aggregate; relational execution then defer/fallback inside one recurrent program), or whether these factors are deliberately program-level controls whose per-step semantic supervision serves another purpose. Identify any inconsistency where step_factor targets train information that is discarded before execution. Classify as (A) concrete architecture defect/capability ceiling to fix before Magnolia, (B) missing causal static test required first, or (C) deliberate global semantics that should be documented/tested. Preserve open runtime factor semantics, recurrent ordered relations, no hard hop/factor ceilings, and anti-hotfix doctrine. Do not authorize Magnolia/GPU/optimizer/gradient/FINAL/N0 completion. Original exact source is authority; Graphify is navigation only.
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: True

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Current continuity status: active continuity authority for the current Sol handoff
- Current build head: edcfcb447fd9f7a47239827661cf31e678f8b249
- Magnolia authorization: **NO — deep source-level audit remains open**
- N0 complete: false
- Continuity overlay: docs/chat-context/2026-09-22/sol/N0_FULL_ENVELOPE_DEEP_STATIC_AUDIT_AND_PERSONAL_DEVELOPMENT_HANDOFF.md
- Current next action: Continue the deep source-level N0 audit at exact head `edcfcb447fd9f7a47239827661cf31e678f8b249`.

Do not ask the owner for a Magnolia run until that audit has either:

- found and repaired the remaining statically discoverable defects and reached a new exact-head green receipt; or
- produced a reasoned source-level conclusion that the remaining unknowns are genuinely empirical and cannot be resolved cheaply before runtime.

Graphify is navigation only. Original branch-qualified source remains authority.
- Authoritative build state: ```text
source_branch=alice-eipm-v1-n0-full-envelope-foundation-build-v1
current_build_head=edcfcb447fd9f7a47239827661cf31e678f8b249
deep_source_audit_complete=false
exact_head_static_suite=PASS_124
proof_obligations_total=109
proof_obligations_static=95
magnolia_cpu_runtime_authorized=false
gpu_memory_dry_run_authorized=false
optimizer_authorized=false
gradient_training_authorized=false
final_validation_open_authorized=false
n0_complete=false
```

The old stable-build latent-stage pointer is historical routing context, not the current implementation frontier. For execution decisions, this handoff plus the exact unmerged full-envelope branch source takes precedence over stale stable-branch stage-state language.
- Stable-base implementation pointer status (historical/routing): SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Stable-base implementation pointer source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: True
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-n0-full-envelope-foundation-build-v1 @ 073819f44ebb79c876140119bcc1e8ebd34b6a37 — governance(n0): bind runtime view and candidate permutation invariants
- alice-eipm-v1-qsre-production-p2-ordered-evidence-v3 @ 342bdbbe8c5dfbe74ff3ab8210a30b1a456aa482 — state(n0): ratify P2 v3 static pass and authorize CPU qualification
- alice-eipm-v1-qsre-production-p2-failure-localization @ c9b1d35823cb89245625bbc6afb05b0ef5fb69fb — fix(n0): emit strict JSON from P2 localization

## Source pointers

- [163] alice-eipm-v1-n0-full-envelope-foundation-build-v1:docs/research/eipm-n0-semantic-operator-foundation-v1.md — N0 Semantic-Operator Foundation v1 — architecture decision after job 575990
  - status:  architecture decision; no gradient or GPU authorization
- [154] alice-eipm-v1-n0-full-envelope-foundation-build-v1:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [154] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [154] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [153] alice-eipm-v1-n0-full-envelope-foundation-build-v1:.github/workflows/n0-query-semantics-architecture-audit-contract-check.yml — n0-query-semantics-architecture-audit-contract-check.yml
- [153] alice-eipm-v1-qsre-production-p2-failure-localization:.github/workflows/n0-query-semantics-architecture-audit-contract-check.yml — n0-query-semantics-architecture-audit-contract-check.yml
- [153] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:.github/workflows/n0-query-semantics-architecture-audit-contract-check.yml — n0-query-semantics-architecture-audit-contract-check.yml
- [142] alice-context:docs/chat-context/2026-09-22/sol/N0_FULL_ENVELOPE_DEEP_STATIC_AUDIT_AND_PERSONAL_DEVELOPMENT_HANDOFF.md — N0 Full-Envelope Deep Static Audit + Personal-Development Architecture Handoff
  - status:  active continuity authority for the current Sol handoff
- [138] alice-eipm-v1-n0-semantic-operator-foundation-v1:docs/research/eipm-n0-semantic-operator-foundation-v1.md — N0 Semantic-Operator Foundation v1 — architecture decision after job 575990
  - status:  architecture decision; no gradient or GPU authorization
- [138] alice-personal-development-architecture-v1:docs/research/eipm-n0-semantic-operator-foundation-v1.md — N0 Semantic-Operator Foundation v1 — architecture decision after job 575990
  - status:  architecture decision; no gradient or GPU authorization
- [137] alice-eipm-v1-n0-full-envelope-foundation-build-v1:docs/research/eipm-n0-job-575966-frozen-semantic-authority-v3.md — N0 job 575966 — frozen semantic authority v3
- [137] alice-eipm-v1-n0-full-envelope-foundation-build-v1:docs/research/eipm-n0-job575986-p2a-semantic-localization-v1.md — N0 job 575986 — P2A semantic localization v1
  - status:  zero-gradient diagnostic package; no architecture change, optimizer, P2, TEST, or semantic-backbone retraining authorized

## Graphify navigation hints

- test_semantic_operator_adapter_and_full_envelope_binder_executor_integrate() -> tests/eipm/test_n0_full_envelope_successor_mechanics_v1.py:L215
- test_binder_v2_operator_relation_mass_controls_support_semantics() -> tests/eipm/test_n0_qsre_production_binder_v2.py:L129
- test_step_conditioned_factor_distributions_exist_for_every_reasoning_slot() -> tests/eipm/test_n0_semantic_operator_foundation_v1.py:L356
- test_response_wrapper_rejects_split_verified_source_set_across_sentences() -> tests/phase4/test_information_conversation_bridge.py:L628
- adapter() -> tests/phase4/test_information_research_mode.py:L312
- aggregate() -> scripts/eipm/n0/audit_n0_v02_query_edge_binding_identifiability_v0_1.py:L102
- AliceN0V02Model -> src/alice_personality/n0/v02_model.py:L13
- test_event_from_another_host_is_rejected() -> tests/phase5/test_experience_ledger_isolation.py:L18
- semantic_retrieval.py -> src/alice_vault/semantic_retrieval.py:L1
- test_memory_identity_host_learning_architecture.py -> tests/governance/test_memory_identity_host_learning_architecture.py:L1
- audit() -> scripts/audit_capability_barriers.py:L521
- authority() -> tests/eipm/test_n0_qsre_production_operator_v3.py:L87
- authorize_boolean_map() -> src/alice_capability_profiles.py:L88
- bank() -> tests/eipm/test_n0_full_envelope_stack_v1.py:L28
- test_candidates_never_enter_derived_indexes_before_promotion() -> tests/phase2/test_memory_candidate_security_gates.py:L312
- BehavioralBatchCompileConfig -> src/alice_personality/n0/full_envelope_behavioral_batch_v1.py:L30

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
