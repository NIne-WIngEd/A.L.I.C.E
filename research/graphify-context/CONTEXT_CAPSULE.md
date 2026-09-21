# A.L.I.C.E. Sol Context Capsule

- Request: n0-production-single-launch-qualified-20260920-006
- Question: Refresh A.L.I.C.E. N0 continuity to the qualified Production QSRE single-launch frontier. Authoritative experiment branch is alice-eipm-v1-qsre-production-core-v1 @ b32fabe49f206c2d71e17df5197a62b2a37c8c43. Full preflight GitHub Actions 35546280865 SUCCESS. Before launch, static qualification caught that ALICE_N0_EXPECTED_REVISION was not forwarded across Magnolia uDocker; magnolia_udocker_exec.sh now forwards it. Final N0 causal ablation was also corrected so integrated and ablated evidence tensors have identical token geometry; the ablated path carries the same two extra positions as zero+invalid, preventing fp16 sequence-shape effects from masquerading as QSRE contribution. Preflight now bash-parses the full runner/sbatch/uDocker wrapper, checks P100 and revision/workdir forwarding, parses required argparse flags for every P0-P5 script, verifies final validation freezes before P1 and never enters P1/P2/P3/P4, rebuilds the exact 1016-row production curriculum and frozen 320-row native validation, proves P4 has no oracle focus path, reruns mechanics, and audits no accidental capability ceilings. Next authorized action is one owner-submitted Magnolia P100 job using scripts/eipm/n0/magnolia_p100_n0_v02_qsre_production_full_pipeline_v1.sbatch with ALICE_N0_EXPECTED_REVISION bound to exact HEAD. Job executes P0->P1->P2->P3->P4->native P5/final validation; failure stops and preserves evidence. No automatic rerun/hotfix/LR/step/width search. N0 closes only on PASS_N0_FINAL_SELF_VALIDATION_OBJECTIVE with n0_complete=true and n1_authorized=true. Durable alice-context source: docs/chat-context/2026-09-20/sol/N0_PRODUCTION_SINGLE_LAUNCH_QUALIFIED.md @ cca662a1a03556065534b45ebc56552536e67da5. Graphify is navigation only; consequential claims must resolve to original experiment branch/path.
- Stable build: alice-eipm-v1-build (may lag this unmerged experiment frontier)
- Catalog source: unmerged experiment head b32fabe49f206c2d71e17df5197a62b2a37c8c43
- Graphify used: False


- Request: n0-qsre-production-architecture-closure-20260920-005
- Question: Refresh A.L.I.C.E. N0 continuity to the production-QSRE architecture-closure frontier. Owner correctly identified risk of a T2 v0.3 -> v0.4 repair chain before N0 completion. T2 v0.3 was frozen before any GPU execution and remains only an unrun static diagnostic prototype. Authoritative state is configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.59.json. Obsolete v0.3 launcher, train runner, and P100 sbatch now fail closed with exit 90. Deep source review found known production gaps independent of the v0.2 benchmark: fixed relation cardinality, learned fixed hop slots, early argmax destroying plurality, mutually-exclusive operation class despite composable traversal/arbitration, fixed applicability centers, continuous operator state discarded before T1, fixed semantic relation-ID embedding authority inside T1, missing domain/range/type grounding, prematurely averaged schema gloss states, and binder interface not closed before operator redesign. Therefore no additional T2-only GPU run is authorized. Production architecture authority is docs/research/eipm-n0-qsre-production-architecture-closure-v1.md; machine contract configs/eipm/n0/n0_v02_qsre_production_core_v1.json; explicit W1-W15 responsibility map configs/eipm/n0/n0_v02_qsre_production_core_w1_w15_map_v1.json. Production Core v1 requires runtime-variable typed relation schema with no learned relation-class table, schema token/multifacet semantics plus domain/range constraints, UNKNOWN distinct from STOP, one shared iterative relation decoder with runtime stopping and no hop-specific slots, sparse plural operator hypotheses, composable traversal/arbitration factors, continuous applicability/uncertainty, continuous operator state reaching executor, adaptive zero/one/many support, schema-conditioned executor with no relation-ID semantic authority, shared iterative node/edge execution, support-local readout, and exact non-relational pass-through. Build sequence is predesigned once: P0 static/CPU closure -> P1 executor with oracle operator/support -> P2 operator with proven executor/oracle support -> P3 binder with proven operator/executor -> P4 end-to-end QSRE -> P5 N0 fusion. Eventual Magnolia stages may be chained with afterok under one owner launch; failure stops the chain. No LR/step/width tuning chain or permission micro-gates. Architecture contract workflow 35540416348 SUCCESS. Durable alice-context commit 3be86965482e33bef45c472374b7cac021a395ed. Graphify is navigation only; verify consequential claims in original experiment branch files.
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

- alice-eipm-v1-qsre-t2-operator-learning @ b99e28c8408dd863c6fa6e960d99c17995b4e705 — ci(n0): enforce production QSRE architecture closure

## Source pointers

- [364] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-qsre-production-architecture-closure-v1.md — QSRE Production Core v1 — Architecture Closure Before Further GPU Work
  - status:  production-architecture authority; implementation/training remains closed
- [205] alice-context:docs/chat-context/2026-09-20/sol/N0_QSRE_T2_ONE_SHOT_OPERATOR_TRAINING_AUTHORIZED.md — N0 QSRE T2 One-Shot Operator Training Authorized
  - status:  tokenized preparation passed; one governed T2 P100 run is statically qualified and authorized
- [204] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md — N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1
  - status:  exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action
- [199] alice-eipm-v1-qsre-t2-operator-learning:.github/workflows/n0-qsre-production-core-v1-architecture-contract.yml — n0-qsre-production-core-v1-architecture-contract.yml
- [195] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-clean-sheet-relational-execution-audit-v0.1.md — N0 Clean-Sheet Relational Execution Architecture Audit v0.1
  - status:  architecture-review authority; implementation and gradient work frozen
- [191] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-qsre-t2-v02-failure-localization-v03-decision.md — N0 QSRE T2 v0.2 failure localization and T2 v0.3 architecture decision
- [184] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-qsre-t1-pass-t2-operator-decision-v0.1.md — QSRE T1 PASS → T2 Learned-Operator Decision v0.1
  - status:  T1 causal question passed on DEV; T2 static design/CPU mechanics justified; T2 gradient remains closed
- [183] alice-eipm-v1-qsre-t2-operator-learning:docs/research/eipm-n0-relational-execution-family-comparison-v0.1.md — N0 Relational Execution Architecture Family Comparison v0.1
  - status:  research decision; no implementation or gradient authorized
- [181] alice-context:docs/chat-context/2026-09-18/sol/N0_RELATION_CONDITIONED_MULTILAYER_INTERFACE_HANDOFF.md — N0 Relation-Conditioned Multi-Layer Query Interface Handoff
  - status:  architecture designed and CI-qualified from 575804 evidence; exact relation-layer map still must be compiled from the saved audit JSON; training remains unauthorized
- [179] alice-eipm-v1-dual-view-late-interaction-binding:docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md — N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1
  - status:  exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action
- [179] alice-eipm-v1-dual-view-specialist-failure-localization:docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md — N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1
  - status:  exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action
- [179] alice-eipm-v1-n0-clean-sheet-relational-execution-review:docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md — N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1
  - status:  exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action

## Graphify navigation hints

- skipped: document/branch/private routing was sufficient

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
