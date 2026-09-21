# A.L.I.C.E. Sol Context Capsule

- Request: n0-job575986-p2a-frozen-authority-failure-causal-audit-20260921-019
- Question: Causal postmortem of owner Magnolia job 575986 on alice-eipm-v1-n0-frozen-semantic-authority-v3 at exact implementation HEAD 0a8ac74aa6fa73259c03dc4d8754a42a6fde2307. Job 575986 is infrastructure-clean and failed at P2A before Production P2 with status FAIL_QSRE_FROZEN_SEMANTIC_AUTHORITY, exit 42, elapsed 00:05:04, no stderr. Preflight/source lineage passed and the selected repaired graph hash matched. P2A metrics were production_core_single_relation_top1=0.4068181812763214 against gate 0.95, auxiliary_seen_relation_top1=0.2708333432674408 against 0.95, auxiliary_holdout_relation_top1=0.2604166567325592 against 0.85, heldout_factor_macro_accuracy=0.36666667064030967 against 0.90. Component diagnostics: Production core joint_preference≈0.17045, principle_alignment≈0.13864, semantic_projection≈0.15909-0.18636, token_evidence≈0.50682-0.52955, combined≈0.40682-0.41591. Auxiliary holdout joint_preference=0, semantic_projection≈0-0.0104, principle_alignment=0.125, token_evidence≈0.219-0.302, combined≈0.260-0.292. Factor combined macro≈0.3667; token evidence was individually much stronger for role=0.875, traversal≈0.667, modifiers≈0.667, direction=0.5, control=0.5 while the learned/frozen semantic surfaces were near chance. Compare this with preserved learned P2S-v2 job575966 best metrics production_core≈0.9477, auxiliary_seen≈0.71875, auxiliary_holdout≈0.6667, heldout_factor_macro≈0.7167. Treat 575986 as valid model/capability evidence, not infrastructure and not a tuning excuse. Reconstruct the semantic-backbone training objectives, targeted-repair history, P2S-v1/v2 failure chain, QSRE operator boundary, N0 role, full EIPM workload envelope, and any earlier decision not to reopen the semantic backbone. Determine what 575986 newly falsifies versus what remains supported. Specifically test these hypotheses against source: (1) the ratified N0 semantic backbone does not contain sufficiently discriminative open-schema relation/factor semantics, (2) the four frozen surfaces are individually miscalibrated for runtime schema discrimination, (3) equal-weight candidate-axis z-score fusion is itself the primary defect, (4) prompt/task mismatch rather than representation deficiency explains the collapse, (5) P2S learned adaptation demonstrates semantic information is latent but inaccessible to zero-gradient readout, (6) the semantic-base architecture must now be reopened rather than adding another head. Do not propose LR/step/width/batch/threshold tuning or another hotfix chain. Compare all maximal unmerged N0 heads and supersession history. Identify the smallest scientifically meaningful next investigation that can distinguish representation deficiency, readout/task mismatch, and fusion/calibration failure without consuming another owner GPU run. Also identify what should be written to alice-context and Fable now. Graphify remains navigation-only; cite original source paths for consequential conclusions.
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

- [231] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-job-575962-semantic-matcher-v2.md — N0 job 575962 — semantic matcher v2 causal repair
- [217] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-job-575966-frozen-semantic-authority-v3.md — N0 job 575966 — frozen semantic authority v3
- [211] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [211] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-production-p2-575956-failure-localization-v1.md — Production QSRE P2 job 575956 failure localization v1
  - status:  zero-gradient diagnosis authority; no rerun or GPU authorization
- [206] alice-eipm-v1-n0-closure-semantic-v2:docs/research/eipm-n0-job-575962-semantic-matcher-v2.md — N0 job 575962 — semantic matcher v2 causal repair
- [203] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [203] alice-eipm-v1-qsre-production-p2-failure-localization:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [203] alice-eipm-v1-qsre-production-p2-ordered-evidence-v3:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [201] alice-context:docs/chat-context/2026-09-21/sol/N0_JOB575962_SEMANTIC_V2_READY.md — N0 job 575962 semantic-v2 closure — qualified for one owner launch
  - status:  valid P2S model failure preserved; semantic-v2 repair implemented; exact-head static/CPU qualification passed; exact-source Graphify verification passed; Fable trace updated; one owner Magnolia launch may proceed
- [199] alice-context:docs/chat-context/2026-09-21/sol/N0_JOB575958_CLOSURE_PASS_READY.md — N0 job 575958 closure-pass — exact-head verified and ready for one Magnolia launch
  - status:  closure package implemented, exact-head static/CPU qualification passed, Graphify exact-source topology verified, Fable failure/success seeds recorded, one owner Magnolia launch may proceed
- [186] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-job-575958-closure-pass-v1.md — N0 closure-pass architecture after Magnolia job 575958
  - status:  implementation and static qualification package; GPU remains closed until exact-head qualification passes
- [184] alice-eipm-v1-n0-frozen-semantic-authority-v3:docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md — N0 Query-Semantics Architecture Audit v0.1

## Graphify navigation hints

- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh script -> scripts/eipm/n0/run_n0_v02_qsre_production_operator_binder_v2_recovery_v1.sh:L1
- run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh -> scripts/eipm/n0/run_n0_v02_qsre_n0_frozen_semantic_authority_v3.sh:L1
- AliceN0V02Model -> src/alice_personality/n0/v02_model.py:L13
- continue_mlm_p100x2_segment.sh -> scripts/eipm/n0/continue_mlm_p100x2_segment.sh:L1
- train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py -> scripts/eipm/n0/train_n0_v02_relation_conditioned_multilayer_interface_v0_1.py:L1
- eval_n0_v02_final_self_validation_v3.py -> scripts/eipm/n0/eval_n0_v02_final_self_validation_v3.py:L1
- eval_n0_v02_qsre_production_p4_v1.py -> scripts/eipm/n0/eval_n0_v02_qsre_production_p4_v1.py:L1
- test_phase5_parity_release.py -> tests/governance/test_phase5_parity_release.py:L1
- canonical_sha256() -> src/cognitive_kernel/canonical.py:L39
- accuracy() -> scripts/eipm/n0/qualify_n0_v02_qsre_frozen_semantic_authority_v3.py:L42
- test_all_padding_is_rejected() -> tests/eipm/test_n0_structured_state.py:L104
- verify_result_against_preflight() -> scripts/eipm/n0/finalize_downstream_causal_arbitration_v0_1.py:L146
- principle_alignment_scores() -> scripts/eipm/n0/qsre_frozen_semantic_authority_runtime.py:L93
- ConstitutionalSourceSnapshot -> src/alice_conversation/constitutional_prompt.py:L50
- test_event_from_another_host_is_rejected() -> tests/phase5/test_experience_ledger_isolation.py:L18

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
