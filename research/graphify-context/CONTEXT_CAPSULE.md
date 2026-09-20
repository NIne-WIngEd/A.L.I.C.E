# A.L.I.C.E. Sol Context Capsule

- Request: n0-qsre-t1-corrected-pretraining-20260920-001
- Question: Refresh A.L.I.C.E. N0 continuity to corrected QSRE T1 pretraining frontier. Branch alice-eipm-v1-qsre-t1-executor-training is at d62e8935811a2fcce41ddc425a7fb9788f7f0aed with authoritative state v0.47. v0.46 GPU eligibility was retracted before any gradient run because the v0.1 curriculum allowed positional shortcuts and the old relation-order test did not prove graph-path traversal. Corrected PATH_FOLLOW uses explicit oracle focus frontier, relation-matched supported edges whose SOURCE is on the current frontier, updates only incident active nodes, advances frontier to reached TARGETs, and restricts final path readout to the reached frontier. T1 rejects direct field-only support and fractional oracle support; edge support is 0/1 membership. Curriculum v0.2 is deterministic SHA 155f62e92cf8c6bff71a3b7d8913927e48c950407985db5a164eadaa056fb063, 504 public TRAIN/DEV rows, 9 families, paired same-graph causal interventions, permuted field/edge order, no TEST/private identity. Real-cache preparation reuses only field_semantic from qualified cache SHA 5bcc4ecf0fb2b776f738e4cca881e6b03a68c81a96923a5b08821930d0ebe823 with split-isolated assignment and bit-identical causal-pair representations. Corrected pretraining CI 35501709024 passed syntax, governance, mechanics, causal curriculum, fake-cache prep, trainer import and CPU backward smoke. GPU training remains closed. Next action is exactly one Magnolia CPU/no-gradient real-cache preparation receipt; only after that may a new state authorize one P100 T1 run. Graphify remains navigation-only.
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

- alice-eipm-v1-qsre-t1-executor-training @ d62e8935811a2fcce41ddc425a7fb9788f7f0aed — fix(n0): correct QSRE T1 path causality and paired curriculum before training

## Source pointers

- [255] alice-context:docs/chat-context/2026-09-20/sol/N0_QSRE_T1_CORRECTED_PRETRAINING_FRONTIER.md — N0 QSRE T1 Corrected Pretraining Frontier
- [165] alice-eipm-v1-qsre-t1-executor-training:docs/research/eipm-n0-relation-conditioned-multilayer-causal-interface-study-v0.1.md — N0 Relation-Conditioned Multi-Layer Causal Interface Study v0.1
  - status:  fresh causal curriculum and preservation contract defined; CPU-only preparation authorized; interface training remains unauthorized
- [163] alice-eipm-v1-qsre-t1-executor-training:docs/research/eipm-n0-qsre-t1-pretraining-validity-correction-v0.1.md — QSRE T1 Pretraining Validity Correction v0.1
- [159] alice-eipm-v1-qsre-t1-executor-training:.github/workflows/n0-qsre-t1-corrected-pretraining-contract.yml — n0-qsre-t1-corrected-pretraining-contract.yml
- [155] alice-context:docs/chat-context/2026-09-18/sol/N0_MULTILAYER_CAUSAL_STUDY_PREPARATION_READY.md — N0 Multi-Layer Causal Study — CPU Preparation Ready
- [155] alice-eipm-v1-qsre-t1-executor-training:docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md — N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1
  - status:  exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action
- [154] alice-eipm-v1-build:docs/eipm/EIPM_N0_FRONTIER_RESEARCH_AND_BOOTSTRAP_PLAN_2026-09-13.md — A.L.I.C.E. EIPM N0 Frontier Research and Bootstrap Plan — 2026-09-13
  - status:  research decision / implementation gate on `alice-eipm-v1-build`. Not canonical `main`. No permanent A.L.I.C.E. EIPM weights are created or authorized by this document.
  - supersession:  this document supersedes any earlier use of `~100M–400M` or `~400M` as an EIPM target, envelope, or ceiling. Those numbers were exploratory estimates for one specialist architecture hypothesis. **There is no ratified parameter count.** The permanent EIPM may be much smaller or much larger if measured capability and identity fidelity require it.
- [152] alice-eipm-v1-build:docs/eipm/n0/runtime-results/N0_FRONTIER_RESEARCH_AUDIT_20260914.md — N0 Frontier Research Audit — 2026-09-14
  - status: hold the step-1000 -> step-2500 GPU job pending N0 v0.2 redesign**.
- [151] alice-context:docs/chat-context/2026-09-18/sol/N0_MULTILAYER_ONE_P100_RUN_AUTHORIZED.md — N0 Multi-Layer Interface — One P100 Run Authorized
- [148] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_READY_FOR_CALIBRATION.md — N0 causal-arbitration frontier ready for Magnolia calibration
- [145] alice-context:docs/chat-context/2026-09-17/sol/N0_CAUSAL_ARBITRATION_FRONTIER_CALIBRATION_HANDOFF.md — N0 Causal Arbitration Frontier Calibration Handoff
- [140] alice-eipm-v1-dual-view-late-interaction-binding:docs/research/eipm-n0-relation-conditioned-multilayer-causal-interface-study-v0.1.md — N0 Relation-Conditioned Multi-Layer Causal Interface Study v0.1
  - status:  fresh causal curriculum and preservation contract defined; CPU-only preparation authorized; interface training remains unauthorized

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
