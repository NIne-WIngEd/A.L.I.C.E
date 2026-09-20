# A.L.I.C.E. Sol Context Capsule

- Request: n0-qsre-t2-evaluator-recovery-authorized-20260920-001
- Question: Refresh A.L.I.C.E. N0 continuity to the classified QSRE T2 evaluator-recovery frontier. Governed T2 job 575933 ran from source c43df19e79dc63ed00f15a313d71f7873fa5b71f and stopped after step25 diagnostics with ValueError PATH_FOLLOW requires a non-empty oracle focus frontier. No result.json was produced and the model-gate conclusion is UNDETERMINED. Root cause is evaluator totality: the frozen T1 validator correctly rejects PATH_FOLLOW without focus, but an immature T2 operation head can temporarily predict PATH_FOLLOW on rows with empty oracle focus; the evaluator passed provisional predictions directly into T1 and raised instead of scoring the wrong prediction as failure. Do not change T1. Do not infer model/capacity failure. Bounded evaluator correction revision bff3b2e7c31255e8ca9354add5dafa1bde177b18 canonicalizes impossible path predictions only for safe T1 execution; RELATIONAL rows requiring canonicalization remain explicit downstream failures with zeroed downstream probability, while non-relational operation is semantically inactive. New diagnostics expose invalid_relational_path_without_focus_count/rate. No architecture, optimizer, LR, loss, data, semantic checkpoint, T1 checkpoint, eligibility threshold, or TEST boundary changed. Evaluator-totality qualification revision feee4f2b8cc36dbcf1c3805f401f4347315be945, workflow 35533710899, SUCCESS. Authoritative state v0.54. Recovery scientific core 157153b124d665880de237299d8dd8c81403d45c. Classified recovery authorization workflow 35533910879 SUCCESS and proves exact scientific equivalence to original T2 contract. Preserve failed root training-v0.1, its hidden cache, best-observed checkpoint, stdout/stderr; do not promote step25. One fresh replacement run is authorized at training-v0.2-evaluator-totality-recovery, restarting from original seed with no partial optimizer reuse. Automatic rerun/hotfix chains remain forbidden. T1 rerun, semantic/T1 gradients, T3 support learning, TEST, frozen challenge, private identity gradient, and production promotion remain closed. Durable alice-context handoff commit 9e0f6332336bd9fcdaaf38b608dda3bda5d1e7cc. Full EIPM workload/no-permanent-ceiling doctrine remains active. Graphify is navigation-only.
- Stable build: alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132
- Catalog source: 021c5021a98104b35f9c8e94b19e48d21f25f132
- Graphify used: False

## Active mission

- Mission schema: alice-context-active-mission-state-v2
- Implementation status: SELECTOR_PRIOR_HEAD_REPAIR_FAILED_ENDPOINT_ROLE_DEFECT_LOCALIZED_ENDPOINT_READ_REPAIR_IMPLEMENTED_PENDING_PREP_AND_GPU
- Implementation source: configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.9.json
- Continuity overlay: docs/chat-context/2026-09-18/sol/N0_RELATION_CONDITIONED_MULTILAYER_INTERFACE_HANDOFF.md
- Latest observed Magnolia job: None
- Continuity stale vs experiment frontier: True
- Execution rule: The stable build base and continuity overlay are not sufficient when continuity_freshness is stale. Before issuing an execution command, inspect every maximal unmerged experiment head and its original receipts/docs. Unmerged experiment state remains non-canonical until explicitly promoted.

## Unmerged experiment frontier

- alice-eipm-v1-qsre-t2-operator-learning @ 10eb099a0201257922e5fc245a9b6772af174649 — ci(n0): qualify classified QSRE T2 recovery run

## Source pointers

- [216] alice-context:docs/chat-context/2026-09-20/sol/N0_QSRE_T2_EVALUATOR_TOTALITY_RECOVERY_AUTHORIZED.md — N0 QSRE T2 Evaluator-Totality Recovery Authorized
  - status:  job 575933 classified as evaluator-harness failure, not a model-gate result; one fresh classified replacement run authorized after static qualification
- [144] alice-eipm-v1-qsre-t2-operator-learning:.github/workflows/n0-qsre-t2-evaluator-totality-qualification.yml — n0-qsre-t2-evaluator-totality-qualification.yml
- [141] alice-context:docs/chat-context/2026-09-16/sol/FUSION_FROZEN_CHALLENGE_FAILURE_AND_SOURCE_ANCHORED_REPAIR.md — Fusion frozen challenge failure + source-anchored repair — 2026-09-16
  - status: `FAIL_NO_CHECKPOINT_ELIGIBLE_FOR_RATIFICATION`.
- [135] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_CHALLENGE_READY_HANDOFF.md — N0 Final Frozen Challenge Ready Handoff
  - status:  one final frozen latent challenge authorized and execution-bound; no promotion/training/scale/N0 completion authorized
- [135] alice-eipm-v1-qsre-t2-operator-learning:.github/workflows/n0-qsre-t2-classified-recovery-authorization.yml — n0-qsre-t2-classified-recovery-authorization.yml
- [130] alice-eipm-v1-build:docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md — A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13
  - status:  research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.
  - supersession:  this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.
- [127] alice-context:docs/chat-context/2026-09-10/sol/EIPM_NATIVE_FRONTIER_ARCHITECTURE_RESEARCH.md — A.L.I.C.E.-Native EIPM Frontier Architecture Research — 2026-09-10
  - status:  working frontier recommendation on `alice-context`; not canonical `main`; no A.L.I.C.E. weights created.
  - supersession:  this note supersedes the earlier working recommendation that treated `Qwen3.8-27B + A.L.I.C.E. LoRA` as the permanent EIPM. Qwen may remain an optional disposable teacher/mechanics tool where licensing permits. Its weights are not part of the EIPM.
- [125] alice-context:docs/chat-context/2026-09-20/sol/N0_QSRE_T1_ONE_SHOT_P100_AUTHORIZED.md — N0 QSRE T1 One-Shot P100 Authorized Frontier
- [123] alice-context:docs/chat-context/2026-09-18/sol/N0_FINAL_FROZEN_FAIL_AND_MISSING_EVIDENCE_LOCALIZATION.md — N0 Final Frozen Challenge Valid FAIL + Fresh Missing-Evidence Localization
  - status:  final frozen challenge completed; valid single-gate failure; do not rerun or change thresholds; fresh non-challenge causal-path localization staged
- [122] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md — N0 Causal Arbitration Frontier Calibration Handoff
- [120] alice-eipm-v1-build:docs/eipm/N0_LATENT_POOL_V01_CHALLENGE_FAILURE_AND_SCALE_ADEQUACY_AUDIT_2026-09-16.md — N0 Latent Pool v0.1 Frozen Challenge Failure and Scale Adequacy Audit — 2026-09-16
- [118] alice-context:docs/chat-context/2026-09-16/sol/FULL_SCALE_FUSION_JOB575619_AND_FROZEN_CHALLENGE.md — Full-Scale Fusion Job 575619 + Frozen Challenge Gate — 2026-09-16

## Graphify navigation hints

- skipped: document/branch/private routing was sufficient

## External/private routing

- Recommended: True
- Reason: question references a private/external source class

## Trust contract

- This capsule routes retrieval; it is not canonical A.L.I.C.E. truth.
- Open consequential claims in the original branch/path source before acting.
- Keep branch-specific, experimental, superseded and canonical states distinct.
- Graphify nodes are navigation hints only.
- If continuity is stale relative to unmerged experiment heads, inspect those heads before execution.
- If private external routing is recommended, search the private Library manifest/source corpus before declaring context complete.
