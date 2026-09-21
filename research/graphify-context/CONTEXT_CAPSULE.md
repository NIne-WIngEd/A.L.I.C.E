# A.L.I.C.E. Sol Context Capsule

- Request: n0-closure-pass-qualified-topology-20260921-011
- Question: Verify the exact N0 closure-pass implementation after job 575958. Trace P2S QSRESchemaMatcher training and holdout boundary, frozen matcher loading into Production P2, ordered query-evidence coverage, semantic factor schemas with no fixed factor class heads, continuous relation hypotheses through Binder v2 structural sparsity, factor-schema cache lineage through P3/P4/final, the single Magnolia closure runner, and reuse of the original frozen native final validation. Identify any code path that bypasses the frozen matcher, reintroduces relation/factor identity parameters, drops coverage, uses open/final holdout descriptions in gradient, or fails to forward factor-schema lineage.
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

- alice-eipm-v1-n0-closure-pass @ fefc2e4ec54ddde113165b1a3103a1add9be34e8 — fix(n0): repair closure matcher eval indentation
- alice-eipm-v1-qsre-production-p2-ordered-evidence-v3 @ 342bdbbe8c5dfbe74ff3ab8210a30b1a456aa482 — state(n0): ratify P2 v3 static pass and authorize CPU qualification
- alice-eipm-v1-qsre-production-p2-failure-localization @ c9b1d35823cb89245625bbc6afb05b0ef5fb69fb — fix(n0): emit strict JSON from P2 localization

## Source pointers

- [145] alice-eipm-v1-n0-closure-pass:docs/research/eipm-n0-job-575958-closure-pass-v1.md — N0 closure-pass architecture after Magnolia job 575958
  - status:  implementation and static qualification package; GPU remains closed until exact-head qualification passes
- [107] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_FAIL_AND_MISSING_EVIDENCE_LOCALIZATION.md — N0 Final Frozen Challenge Valid FAIL + Fresh Missing-Evidence Localization
  - status:  final frozen challenge completed; valid single-gate failure; do not rerun or change thresholds; fresh non-challenge causal-path localization staged
- [103] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md — N0 Final Frozen Challenge Ready Handoff
  - status:  one final frozen latent challenge authorized and execution-bound; no promotion/training/scale/N0 completion authorized
- [101] alice-eipm-v1-build:docs/PHASE_1_HHEM_HOLDOUT_VALIDATION.md — Phase 1.11 — HHEM Frozen-Threshold Holdout Validation
- [97] alice-eipm-v1-n0-closure-pass:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [97] alice-eipm-v1-qsre-production-core-v1:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [97] alice-eipm-v1-qsre-production-p2-failure-localization:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [97] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [95] alice-context:docs/chat-context/2026-09-15/sol/N0_RELATION_REPAIR_FROZEN_RATIFICATION_PENDING.md — N0 relation repair — frozen ratification pending
- [95] alice-context:docs/chat-context/2026-09-16/sol/FULL_SCALE_FUSION_JOB575619_AND_FROZEN_CHALLENGE.md — Full-Scale Fusion Job 575619 + Frozen Challenge Gate — 2026-09-16
- [93] fable-builder-model:docs/fable-builder/traces/2026-09-13_training_lineage_boundary.jsonl — 2026-09-13_training_lineage_boundary.jsonl
- [87] fable-builder-model:docs/fable-builder/traces/2026-09-18_n0_final_frozen_fail_missing_evidence_localization.jsonl — 2026-09-18_n0_final_frozen_fail_missing_evidence_localization.jsonl

## Graphify navigation hints

- Final N0 closure operator. The filename/class name is retained so P3/P4/frozen-… -> src/alice_personality/n0/qsre_production_operator_v3.py:L25
- test_closure_operator_has_no_fixed_factor_class_heads() -> tests/eipm/test_n0_qsre_production_operator_v3.py:L111
- QSRESchemaMatcher -> src/alice_personality/n0/qsre_schema_matcher.py:L22
- run_final_frozen_challenge_after_graph_arbitration_v0_1.py -> scripts/eipm/n0/run_final_frozen_challenge_after_graph_arbitration_v0_1.py:L1
- QSREProductionBinderV2 -> src/alice_personality/n0/qsre_production_binder_v2.py:L16
- .boundary() -> src/alice_conversation/cli_policy.py:L67
- .test_custom_search_bypasses_production_evidence_expansion() -> tests/phase1/test_grounded_response_injected_retrieval.py:L15
- CachedStructuredDataset -> scripts/eipm/n0/train_n0_v02_structured_state_pilot.py:L259
- classify_owner_relation() -> src/alice_vault/owner_attribution.py:L186
- train_n0_v02_qsre_closure_matcher_v1.py -> scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v1.py:L1
- codes() -> tests/phase3/test_conversation_response_validation_adversarial.py:L18
- test_binder_v2_operator_continuous_state_is_not_support_authority() -> tests/eipm/test_n0_qsre_production_binder_v2.py:L152
- test_n0_curriculum_coverage.py -> tests/eipm/test_n0_curriculum_coverage.py:L1
- evidence() -> tests/phase3/test_conversation_phase1_grounding_bridge.py:L26
- ExactInformationLiveProviderRegistry -> src/alice_information/live_provider_registry.py:L16
- factor_losses() -> scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v1.py:L257

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
