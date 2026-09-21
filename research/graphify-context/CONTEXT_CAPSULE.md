# A.L.I.C.E. Sol Context Capsule

- Request: n0-job575990-semantic-localization-result-architecture-decision-20260921-022
- Question: Interpret completed Magnolia CPU/no-gradient job 575990 from branch alice-eipm-v1-n0-p2a-semantic-localization-v1@8180ed500fcf2181d3a70f5a01481d00ebbbb24a and route the smallest correct architecture decision. Job 575990 completed 0:0 with empty stderr and reconstructed job575986 exactly. Headline metrics: current P2A auxiliary_holdout=0.2604166567, auxiliary_seen=0.2708333433, heldout_factor_macro=0.3666666731, production_core=0.4068181813. Teacher-native prompt/candidate geometry did not rescue it: holdout=0.2604166567, seen=0.2291666716, factors=0.4166666731, production_core=0.3840909004. Even the posthoc teacher-native + best-hidden-layer token oracle remained weak: holdout=0.3541666567, seen=0.3333333433, factors=0.4166666731, production_core=0.4090909064. Compare this to learned P2S-v2 job575966 best metrics holdout=0.6666666667, seen=0.71875, factors=0.7166666667, production_core=0.9477272727 and later core fit 1.0. Use original source docs and semantic model configuration to determine what is now falsified. In particular inspect the ratified alice-n0-semantic-v0.2 objectives/data scale and whether the 136.6M ModernBERT encoder was ever trained to supply dynamic open-schema relation/factor semantics. Decide whether prompt/readout/fusion remain plausible primary defects or whether the public semantic base/objective must be reopened. Preserve proven structured/evidence/fusion/latent/QSRE executor/binder work unless evidence directly falsifies them. Do not propose another P2S/P2A hotfix, threshold change, blind scale search, or immediate GPU run. Identify a frontier-style architecture family that learns semantic representation and dynamic schema/operator behavior jointly, with runtime-variable schema cardinality, no relation-ID ontology, token-level interaction, explicit role/direction/traversal/modifier/control semantics, uncertainty/plurality, shared recurrent step transition with structural stop, and compatibility with existing QSRE structural execution. This is architecture decision/research only.
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: True

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-18/sol/N0_RELATION_CONDITIONED_MULTILAYER_INTERFACE_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: True
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-n0-p2a-semantic-localization-v1 @ 8180ed500fcf2181d3a70f5a01481d00ebbbb24a — ci(n0): qualify P2A semantic localization
- alice-eipm-v1-qsre-production-p2-ordered-evidence-v3 @ 342bdbbe8c5dfbe74ff3ab8210a30b1a456aa482 — state(n0): ratify P2 v3 static pass and authorize CPU qualification
- alice-eipm-v1-qsre-production-p2-failure-localization @ c9b1d35823cb89245625bbc6afb05b0ef5fb69fb — fix(n0): emit strict JSON from P2 localization

## Source pointers

- [225] alice-eipm-v1-n0-p2a-semantic-localization-v1:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [225] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [225] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [202] alice-eipm-v1-n0-p2a-semantic-localization-v1:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [202] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [202] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [200] alice-eipm-v1-dual-view-late-interaction-binding:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [200] alice-eipm-v1-dual-view-specialist-failure-localization:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [200] alice-eipm-v1-n0-clean-sheet-relational-execution-review:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [200] alice-eipm-v1-n0-closure-pass:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [200] alice-eipm-v1-n0-closure-semantic-v2:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1
- [200] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1

## Graphify navigation hints

- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh script -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- run_n0_v02_qsre_p2a_semantic_localization_v1.sh -> scripts/eipm/n0/run_n0_v02_qsre_p2a_semantic_localization_v1.sh:L1
- AliceN0V02Model -> src/alice_personality/n0/v02_model.py:L13
- train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py -> scripts/eipm/n0/train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py:L1
- test_event_from_another_host_is_rejected() -> tests/phase5/test_experience_ledger_isolation.py:L18
- test_memory_identity_host_learning_architecture.py -> tests/governance/test_memory_identity_host_learning_architecture.py:L1
- auxiliary_relation_schema_text() -> scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v2.py:L36
- base_parts() -> scripts/eipm/n0/build_n0_v02_evidence_graph_curriculum.py:L91
- test_disabled_repair_preserves_original_rejection_behavior() -> tests/phase3/test_conversation_response_repair_orchestration.py:L48
- best_slot_semantic_alignment_loss() -> src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives_v0_2.py:L39
- QSREProductionBinderV2 -> src/alice_personality/n0/qsre_production_binder_v2.py:L16
- Use exactly the evidence windows shown during blind human review. -> src/alice_vault/hhem_calibration.py:L508
- branch_forward() -> scripts/eipm/n0/train_n0_v02_structured_state_pilot.py:L287
- candidate() -> tests/phase5/test_memory_m2_adjudication_contracts.py:L154
- test_dynamic_relation_cardinality_has_no_parameter_axis() -> tests/eipm/test_n0_qsre_production_core_v1.py:L157

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
