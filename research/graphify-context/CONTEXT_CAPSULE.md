# A.L.I.C.E. Sol Context Capsule

- Request: n0-semantic-v2-post-handoff-refresh-20260921-014
- Question: Post-handoff refresh after recording docs/chat-context/2026-09-21/sol/N0_JOB575962_SEMANTIC_V2_READY.md. Verify the continuity catalog now sees alice-context at that semantic-v2 handoff while the exact code graph remains alice-eipm-v1-n0-closure-semantic-v2@1c94065cd30abb9764260e239b2227b27d098040. Reconfirm the governed next path: preserved job575962 P2S-v1 failure -> semantic-v2 P2S -> frozen matcher Production P2 -> ordered continuous operator -> Binder v2 exact structural sparsity -> P3/P4 -> original frozen final validation, with no holdout/key leakage or downstream gate drift.
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

- alice-eipm-v1-n0-closure-semantic-v2 @ 1c94065cd30abb9764260e239b2227b27d098040 — test(n0): assert semantic-v2 matcher invariants
- alice-eipm-v1-qsre-production-p2-ordered-evidence-v3 @ 342bdbbe8c5dfbe74ff3ab8210a30b1a456aa482 — state(n0): ratify P2 v3 static pass and authorize CPU qualification
- alice-eipm-v1-qsre-production-p2-failure-localization @ c9b1d35823cb89245625bbc6afb05b0ef5fb69fb — fix(n0): emit strict JSON from P2 localization

## Source pointers

- [105] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md — N0 Final Frozen Challenge Ready Handoff
  - status:  one final frozen latent challenge authorized and execution-bound; no promotion/training/scale/N0 completion authorized
- [96] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [96] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [93] alice-eipm-v1-n0-closure-semantic-v2:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [93] alice-eipm-v1-qsre-production-p2-failure-localization:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [93] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:.github/workflows/n0-production-core-v1-preflight.yml — Final-only artifacts may be built/cached/frozen before gradient,
- [89] alice-eipm-v1-n0-closure-semantic-v2:docs/research/eipm-n0-job-575962-semantic-matcher-v2.md — N0 job 575962 — semantic matcher v2 causal repair
- [88] alice-eipm-v1-n0-closure-semantic-v2:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [88] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [88] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md — N0 downstream causal arbitration finalization v0.1
- [87] alice-eipm-v1-n0-closure-semantic-v2:docs/research/eipm-n0-job-575958-closure-pass-v1.md — N0 closure-pass architecture after Magnolia job 575958
  - status:  implementation and static qualification package; GPU remains closed until exact-head qualification passes
- [86] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_FAIL_AND_MISSING_EVIDENCE_LOCALIZATION.md — N0 Final Frozen Challenge Valid FAIL + Fresh Missing-Evidence Localization
  - status:  final frozen challenge completed; valid single-gate failure; do not rerun or change thresholds; fresh non-challenge causal-path localization staged

## Graphify navigation hints

- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh script -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- Final N0 closure operator. The filename/class name is retained so P3/P4/frozen-… -> src/alice_personality/n0/qsre_production_operator_v3.py:L25
- derive_tokenizer_corpus_v021_from_v01.py -> scripts/eipm/n0/derive_tokenizer_corpus_v021_from_v01.py:L1
- run_final_frozen_challenge_after_graph_arbitration_v0_1.py -> scripts/eipm/n0/run_final_frozen_challenge_after_graph_arbitration_v0_1.py:L1
- AliceN0V02Model -> src/alice_personality/n0/v02_model.py:L13
- QSREProductionBinderV2 -> src/alice_personality/n0/qsre_production_binder_v2.py:L16
- chunk_catalog.py -> src/alice_vault/chunk_catalog.py:L1
- train_n0_v02_qsre_closure_matcher_v2.py -> scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v2.py:L1
- codes() -> tests/phase3/test_conversation_response_validation_adversarial.py:L18
- context() -> tests/phase1/test_claim_support_audit.py:L17
- Metadata-only provenance for source truth, inference, and continuity. -> src/cognitive_kernel/contracts.py:L167
- test_binder_v2_operator_continuous_state_is_not_support_authority() -> tests/eipm/test_n0_qsre_production_binder_v2.py:L152
- test_docs_remove_obsolete_opaque_only_boundary() -> tests/governance/test_memory_identity_host_learning_architecture.py:L154
- downstream_causal_arbitration_v0_1.py -> scripts/eipm/n0/downstream_causal_arbitration_v0_1.py:L1
- test_policy_rejects_contract_drift() -> tests/phase3/test_conversation_response_repair_policy.py:L64

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
