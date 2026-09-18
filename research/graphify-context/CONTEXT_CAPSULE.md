# A.L.I.C.E. Sol Context Capsule

- Request: coverage-frontier-codegraph-reuse-001
- Question: Which functions in code enforce the frozen source revision and artifact hashes before the downstream causal arbitration finalizer can invoke the post-arbitration decision gate?
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: True

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

- [72] alice-eipm-v1-causal-arbitration-binding:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [72] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [72] alice-eipm-v1-downstream-arbitration:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [72] alice-eipm-v1-post-arbitration-gate:docs/eipm/EIPM_N0_DOWNSTREAM_CAUSAL_ARBITRATION_V0_1.md — N0 downstream causal arbitration v0.1
- [66] alice-eipm-v1-causal-arbitration-binding:docs/research/eipm-n0-causal-arbitration-full-stack-binding-v0.1.md — N0 downstream causal arbitration: full-stack graph binding v0.1
- [57] alice-context:docs/chat-context/2026-09-17/sol/N0_ENDPOINT_REPAIR_AND_DOWNSTREAM_ARBITRATION_HANDOFF.md — N0 Endpoint Repair and Downstream Arbitration Handoff
  - status: `FAIL_SELECTOR_REPAIR_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`.
- [45] alice-context:docs/chat-context/2026-09-16/sol/FULL_SCALE_FUSION_JOB575619_AND_FROZEN_CHALLENGE.md — Full-Scale Fusion Job 575619 + Frozen Challenge Gate — 2026-09-16
- [43] alice-context:docs/chat-context/2026-09-16/sol/FUSION_FROZEN_CHALLENGE_FAILURE_AND_SOURCE_ANCHORED_REPAIR.md — Fusion frozen challenge failure + source-anchored repair — 2026-09-16
  - status: `FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION`.
- [43] fable-builder-model:docs/fable-builder/traces/FBM_TRACE_20260916_SOURCE_ANCHORED_FUSION_CONFIRMATORY_GATE.jsonl — FBM_TRACE_20260916_SOURCE_ANCHORED_FUSION_CONFIRMATORY_GATE.jsonl
- [42] alice-context:docs/chat-context/2026-09-15/sol/N0_V02_CONTINUATION_RECOVERY_AND_DEV_GATE.md — N0 v0.2 Continuation Recovery and Teacher-Dev Gate
- [42] alice-eipm-v1-build:docs/eipm/n0/N0A_NATIVE_FOUNDATION_DECISION_v0.1.md — N0A Native Foundation Decision v0.1
  - status:  active build decision; implementation starts here
- [42] alice-eipm-v1-build:docs/eipm/n0/public_source_manifest_v0.1.json — public_source_manifest_v0.1.json

## Graphify navigation hints

- PostArbitrationDecisionGateV01Tests -> tests/eipm/n0/test_post_arbitration_decision_gate_v0_1.py:L12
- post_arbitration_decision_gate_v0_1.py -> scripts/eipm/n0/post_arbitration_decision_gate_v0_1.py:L1
- Finalize one frozen N0 downstream causal arbitration into the decision gate.… -> scripts/eipm/n0/finalize_downstream_causal_arbitration_v0_1.py:L2
- ArbitrationV02Tests -> tests/eipm/n0/test_downstream_causal_arbitration_v0_2.py:L12
- artifact_manifest_digest() -> src/cognitive_kernel/release.py:L118
- test_candidates_never_enter_derived_indexes_before_promotion() -> tests/phase2/test_memory_candidate_security_gates.py:L312
- causal_chain() -> scripts/eipm/n0/build_n0_v02_cross_context_fusion_frozen_challenge_v0_2.py:L235
- codes() -> tests/phase3/test_conversation_response_validation_adversarial.py:L18
- decision() -> tests/phase5/attention_workspace_helpers.py:L74
- downstream_causal_arbitration_v0_1.py -> scripts/eipm/n0/downstream_causal_arbitration_v0_1.py:L1
- test_expected_current_generation_is_enforced() -> tests/phase5/test_projection_prototype.py:L389
- eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py -> scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py:L1
- gate() -> scripts/eipm/n0/eval_n0_v02_adaptive_multi_view_latent_pool_frozen_challenge_v0_1.py:L155
- artifact_hashes -> policies/friday_release_attestation_schema.json:L54
- git_revision() -> scripts/eipm/n0/train_n0_v02_cross_context_fusion_full_scale.py:L51
- _source() -> tests/phase4/test_information_freshness.py:L43

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
