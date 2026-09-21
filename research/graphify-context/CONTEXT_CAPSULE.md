# A.L.I.C.E. Sol Context Capsule

- Request: n0-job575986-post-record-continuity-refresh-20260921-020
- Question: Refresh context after recording the valid Magnolia job 575986 P2A failure. Confirm alice-context now includes N0_JOB575986_P2A_FROZEN_AUTHORITY_VALID_FAIL.md at branch head 97fbe41ecce0026054e10d895c198d344969d224 and fable-builder-model includes FBM_TRACE_20260921_N0_JOB575986_P2A_VALID_FAIL.jsonl at branch head 0accefa3139f93a8823bcd19e80fa711af0e94ef. Keep implementation source pinned to alice-eipm-v1-n0-frozen-semantic-authority-v3@0a8ac74aa6fa73259c03dc4d8754a42a6fde2307. Treat job575986 as infrastructure-clean valid model/capability evidence: P2A FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY with Production core 0.406818, auxiliary seen 0.270833, auxiliary holdout 0.260417, heldout factor macro 0.366667; P2 stayed closed. Preserve the anti-loop decision: no identical rerun, no threshold reduction, no P2S-v3, no hyperparameter search, no Production P2, and no semantic-backbone retraining yet. The next authorized investigation is zero-gradient semantic-base localization that separates representation deficiency from prompt/readout/task mismatch and fusion/calibration failure. Reconfirm older maximal heads are causal-history sources rather than current execution authority. This query is continuity refresh only.
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

- alice-eipm-v1-n0-frozen-semantic-authority-v3 @ 0a8ac74aa6fa73259c03dc4d8754a42a6fde2307 — ci(n0): prove repaired graph parent reaches native final
- alice-eipm-v1-qsre-production-p2-ordered-evidence-v3 @ 342bdbbe8c5dfbe74ff3ab8210a30b1a456aa482 — state(n0): ratify P2 v3 static pass and authorize CPU qualification
- alice-eipm-v1-qsre-production-p2-failure-localization @ c9b1d35823cb89245625bbc6afb05b0ef5fb69fb — fix(n0): emit strict JSON from P2 localization

## Source pointers

- [192] alice-context:docs/chat-context/2026-09-21/sol/N0_JOB575986_P2A_FROZEN_AUTHORITY_VALID_FAIL.md — N0 job 575986 — P2A frozen semantic authority valid FAIL
  - status:  valid model/capability failure at P2A; Production P2 remained closed; no rerun/tuning authorized
- [170] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [170] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [147] fable-builder-model:docs/fable-builder/traces/FBM_TRACE_20260921_N0_JOB575986_P2A_VALID_FAIL.jsonl — FBM_TRACE_20260921_N0_JOB575986_P2A_VALID_FAIL.jsonl
- [138] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-query-edge-routing-control-failure-localization-v0.1.md — N0 Query-Edge Routed Specialist Failure Localization v0.1
- [138] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-query-edge-routing-control-failure-localization-v0.1.md — N0 Query-Edge Routed Specialist Failure Localization v0.1
- [138] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-query-edge-routing-control-failure-localization-v0.1.md — N0 Query-Edge Routed Specialist Failure Localization v0.1
- [134] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-qsre-t2-v02-failure-localization-v03-decision.md — N0 QSRE T2 v0.2 failure localization and T2 v0.3 architecture decision
- [134] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-qsre-t2-v02-failure-localization-v03-decision.md — N0 QSRE T2 v0.2 failure localization and T2 v0.3 architecture decision
- [134] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-qsre-t2-v02-failure-localization-v03-decision.md — N0 QSRE T2 v0.2 failure localization and T2 v0.3 architecture decision
- [127] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-qsre-t1-pretraining-validity-correction-v0.1.md — QSRE T1 Pretraining Validity Correction v0.1
- [127] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-qsre-t1-pretraining-validity-correction-v0.1.md — QSRE T1 Pretraining Validity Correction v0.1

## Graphify navigation hints

- run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh -> scripts/eipm/n0/run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh:L1
- qualify_n0_v02_qsre_frozen_semantic_authority_v3.py -> scripts/eipm/n0/qualify_n0_v02_qsre_frozen_semantic_authority_v3.py:L1
- run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh script -> scripts/eipm/n0/run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh:L1
- AliceN0V02Model -> src/alice_personality/n0/v02_model.py:L13
- run_final_frozen_challenge_after_graph_arbitration_v0_1.py -> scripts/eipm/n0/run_final_frozen_challenge_after_graph_arbitration_v0_1.py:L1
- semantic_retrieval.py -> src/alice_vault/semantic_retrieval.py:L1
- authority() -> tests/eipm/test_n0_qsre_production_operator_v3.py:L87
- _authorized() -> tests/phase2/test_memory_service.py:L66
- auxiliary_relation_schema_text() -> scripts/eipm/n0/train_n0_v02_qsre_closure_matcher_v2.py:L36
- .backbone() -> src/alice_personality/n0/v02_model.py:L48
- base_parts() -> scripts/eipm/n0/build_n0_v02_evidence_graph_curriculum.py:L91
- branch_forward() -> scripts/eipm/n0/train_n0_v02_structured_state_pilot.py:L287
- DeterministicInformationGroundingBuilder -> src/alice_information/grounding.py:L667
- CalibrationError -> scripts/eipm/n0/calibrate_downstream_causal_arbitration_metric_policy_v0_1.py:L44
- CapabilityRuntime -> src/alice_evolution/capability_runtime.py:L77
- causal_specialist_probabilities() -> scripts/eipm/n0/audit_n0_v02_dual_view_specialist_failure_localization_v0_2.py:L83

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
