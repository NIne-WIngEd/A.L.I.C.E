# N0 Full-Envelope Retrospective Architecture Audit v1

**Date:** 2026-09-21  
**Status:** retrospective architecture authority; no gradient/GPU authorization  
**Current architecture parent:** `alice-eipm-v1-n0-semantic-operator-foundation-v1@cef6e5ed48bb69e064596b30a87a73c3e79a0763`  
**Trigger:** Magnolia job `575990` plus owner request to re-audit earlier N0 PASS stages against the complete N0 workload rather than their original local contracts.

## Audit rule

A historical PASS means only that the artifact satisfied the contract that was actually tested at that time.

This audit explicitly rejects the stronger inference:

`local PASS -> globally sufficient N0 architecture`.

Every stage is re-read against:

- the active full EIPM workload envelope;
- dynamic runtime schemas;
- open relation/factor/type vocabularies;
- variable support and candidate cardinality;
- ordered composition;
- uncertainty/plurality;
- long and heterogeneous context;
- downstream interface sufficiency;
- absence of hidden product ceilings;
- independence from a weak upstream target geometry.

A stage may remain historically valid while being reopened for the final N0 architecture.

## Finding summary

| Stage | Retrospective classification | Final disposition |
| --- | --- | --- |
| Semantic v0.2 + targeted repair | **MUST_REOPEN_BEFORE_N0_CLOSURE** | Keep checkpoint as initialization/baseline only. Jointly train semantic + schema/operator capability. |
| Structured state | **HISTORICALLY_VALID_INSUFFICIENT_VALIDATION** | Preserve permutation/type/provenance mechanics. Reopen semantic/type interface and weights. |
| Evidence view adapter | **MUST_REOPEN_INTERFACE** | Preserve residual evidence-view idea. Replace pooled frozen-query dependency; requalify weights. |
| Dual-endpoint evidence graph | **HISTORICALLY_VALID + HIDDEN FIXED ONTOLOGY** | Preserve graph/endpoint mechanics. Remove relation-ID embeddings as semantic authority. |
| Cross-context fusion | **PRESERVE_STRONG_ARCHITECTURE / REQUALIFY WEIGHTS** | Keep multi-stream source-anchored family. Retrain/requalify after new parents. |
| Competitive latent pool v0.2 | **PRESERVE_STRONG_ARCHITECTURE FAMILY / CANDIDATE NOT FULLY RATIFIED** | Keep competitive multi-slot family. Retrain/requalify; do not freeze old step360 weights. |
| Downstream causal arbitration | **PRESERVE_STRONG METHODOLOGY** | Reuse methodology for future component comparisons; never reinterpret as global model proof. |
| Query-edge / role-router repair chain | **SUPERSEDED_DIAGNOSTIC_EVIDENCE** | Preserve failure lessons only. Do not compose these repair heads into the final model. |
| QSRE T1 | **EXPERIMENTAL_CAUSAL_SCAFFOLD_ONLY** | Preserve executor causal evidence/mechanics; fixed relation/role/operation embeddings are not final authority. |
| QSRE T2 v0.x | **SUPERSEDED_CAUSAL_SCAFFOLD** | Do not revive fixed slots/classes. Production Core design supersedes it. |
| Production Core v1 | **PRESERVE_STRONG_ARCHITECTURE WITH SEMANTIC-INTERFACE REOPEN** | Keep dynamic relation schema, recurrent execution, uncertainty and delayed sparsity. Replace failed frozen semantic authority/fixed factor banks. |
| Binder v2 | **PRESERVE_STRONG STRUCTURAL BOUNDARY / REOPEN SEMANTIC INPUT** | Keep adaptive exact-zero structural support. Replace final-layer-only query interface and fixed type-ID semantic assumptions. |
| Final self-validation v1 | **HISTORICALLY_VALID BUT INSUFFICIENT FULL-ENVELOPE GATE** | Preserve as historical frozen test. Supersede with a new broader final contract before N0 closure. |

## 1. Semantic base — must reopen

Sources:

- `configs/eipm/n0/alice_n0_semantic_v0.2.json`
- `configs/eipm/n0/n0_v02_semantic_base_ratification_v0.1.json`
- `src/alice_personality/n0/v02_model.py`
- `src/alice_personality/n0/v02_training.py`

The current checkpoint is a 16-layer, 640-wide ModernBERT-family bidirectional encoder with 136,594,435 parameters.

The ratified public semantic training objective was dominated by span MLM, with smaller candidate-preference, rationale-alignment, and semantic-contrastive objectives. The governed teacher bank contained 1,020 rows over 51 registered public competencies. The selected targeted repair added only 20 repair-train rows and updated the top four backbone layers.

The ratification was valid for the semantic-readiness suites it actually measured.

It did **not** establish native representation of:

- arbitrary runtime relation descriptions;
- held-out relation families;
- explicit source/target argument semantics;
- composable direction/traversal/control/modifier schemas;
- recurrent ordered relation programs;
- runtime-variable schema cardinality;
- out-of-schema defer/unknown;
- semantic type-schema grounding.

Job 575990 now shows that neither teacher-native head geometry nor the best hidden-layer zero-gradient readout recovers those semantics sufficiently.

**Decision:** preserve step80 as an initialization and comparison baseline. Reopen public semantic representation learning jointly with the operator/schema objective. Do not treat the current semantic heads as universal frozen authority.

## 2. Structured state — mechanics useful, validation too coupled to old semantics

Sources:

- `configs/eipm/n0/n0_v02_structured_state_v0.1.json`
- `src/alice_personality/n0/structured_state.py`
- `src/alice_personality/n0/structured_state_objectives.py`

Useful mechanics to preserve:

- field-order permutation equivariance;
- pooled permutation invariance;
- explicit missingness/confidence;
- typed/provenance/temporal channels;
- no enforced field-count ceiling;
- no private identity gradient.

Problems exposed retrospectively:

1. Status is explicitly `PILOT_PASS_STEP080_RATIFIED_CURRENT_BASE`.
2. Selected semantic-alignment cosine was only `0.1591`; the strongest metric was field-preservation cosine against the same frozen semantic parent.
3. Its semantic target and field-preservation target are both derived from the now-insufficient step80 semantic representation.
4. Fixed embedding tables for field type, provenance, relation role and temporal scope are migratable checkpoint vocabularies, but they are still fixed learned semantic IDs unless a successor supplies descriptor semantics.
5. The parent semantic representation was frozen during structured training.

A module can therefore pass by faithfully preserving a representation that later turns out to be insufficient for the full workload.

**Decision:** preserve the structured-set architecture and metadata channels. Reopen its semantic interface. Runtime type/provenance/temporal meanings must be able to enter as shared schema-conditioned semantic descriptors rather than requiring a closed learned ID vocabulary to carry meaning. Old weights require requalification after the semantic foundation changes.

## 3. Evidence view adapter — pooled-query bottleneck and frozen-parent compensation

Source:

- `src/alice_personality/n0/evidence_view_adapter.py`

The evidence adapter is a useful non-destructive residual specialization path. It is permutation-equivariant and has no hard field ceiling.

However:

- query conditioning enters as one pooled `[B,640]` vector;
- its selector combines a learned prior head with frozen parent field weights;
- the parent structured representation was intentionally immutable;
- later parent-value-path diagnostics showed selector starvation;
- prior-head-only repair failed badly and exposed endpoint-role defects.

This is exactly the pattern the full-envelope audit is intended to catch: a downstream adapter was asked to compensate around a frozen upstream semantic/structured representation.

**Decision:** retain the evidence-specialized view concept, but replace the pooled frozen-query interface with the new shared token/schema/operator state. Old adapter weights are comparison evidence, not final frozen parent authority.

## 4. Evidence graph — good structural mechanics, fixed semantic ontology

Sources:

- `src/alice_personality/n0/evidence_graph.py`
- `src/alice_personality/n0/evidence_graph_dual_endpoint.py`
- `configs/eipm/n0/n0_v02_evidence_graph_v0.3.json`
- `configs/eipm/n0/n0_v02_evidence_specialist_ratification_v0.1.json`

The dual-endpoint repair solved a real source/target read defect. The untouched relation-essential challenge produced strong local evidence for the repaired endpoint mechanics.

Preserve:

- directed message passing;
- source/target distinction;
- symmetric conflict treatment;
- query-conditioned endpoint read;
- reliability/provenance metadata;
- permutation-safe graph operation;
- no enforced field/edge count ceiling.

But the historical graph still embeds relation semantics through a fixed `EvidenceRelationType`/embedding vocabulary. Unknown relations require registry/checkpoint migration.

Calling that vocabulary “migratable” avoids a permanent numeric ceiling but does **not** make the architecture runtime-open.

The graph's ratification challenge covered only 80 examples / 40 pairs / 10 relation families. It established those local mechanics, not arbitrary relation semantics.

**Decision:** preserve graph topology/message mechanics and the selected endpoint-repair checkpoint as structural evidence. In the successor path, relation meaning must come from runtime schema states produced by the joint semantic/operator foundation. Relation IDs may remain structural/index metadata but cannot own relation semantics.

## 5. Cross-context fusion — strong family, narrow checkpoint evidence

Sources:

- `src/alice_personality/n0/cross_context_fusion.py`
- `src/alice_personality/n0/cross_context_fusion_anchored.py`
- `configs/eipm/n0/n0_v02_cross_context_fusion_v0.2.json`
- `configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json`

Preserve the architecture family:

- separate heterogeneous streams;
- self-refinement;
- gated bidirectional cross-attention;
- query/reliability conditioning;
- exact source-anchor channels;
- explicit missing-view support;
- contextualized and source channels both retained.

No hard view-count ceiling is asserted.

But the selected checkpoint was trained on only three instantiated public views. Adding another view requires state migration because modules, embeddings, and summary tokens are checkpoint-sized by `num_views`.

Training used 512 rows, and the confirmatory challenge used 160 synthetic rows / 20 families. Objectives and challenge targets are also grounded heavily in representations produced by the frozen old semantic/structured/evidence parents.

**Decision:** preserve the fusion architecture family. Do not assume its old checkpoint remains calibrated after upstream semantics change. Rebuild/requalify its weights against the new foundation and broader full-envelope view combinations. Future runtime-extensible view semantics should avoid treating checkpoint view IDs as the only source of view meaning.

## 6. Adaptive latent pool — strong corrected mechanics, not a fully ratified final parent

Sources:

- `src/alice_personality/n0/adaptive_multi_view_latent_pool_v0_2.py`
- `src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives_v0_2.py`
- `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.1.json` through later stage states
- `docs/eipm/N0_ADAPTIVE_LATENT_POOL_V01_FAILURE_AND_V02_REPAIR_2026-09-16.md`
- `docs/eipm/N0_LATENT_POOL_V01_CHALLENGE_FAILURE_AND_SCALE_ADEQUACY_AUDIT_2026-09-16.md`

v0.1 had a real objective/architecture defect: duplicate target-aligned slots were rewarded and slots did not compete for evidence.

v0.2 correctly introduced competitive evidence allocation, emergent slots, source/contextualized channels, and no fixed trait semantics.

The step360 candidate showed strong DEV metrics, but the untouched v0.2 challenge remained an immutable FAIL. Later diagnostics explicitly concluded that latent failure and capacity bottleneck were **not proven** because the required upstream source representation was itself weak.

Important retrospective consequence:

> step360 is a valuable candidate and architecture proof, not a fully ratified full-envelope N0 representation.

Its objective is also largely measured against the same old semantic geometry and source-view summaries.

**Decision:** preserve the competitive multi-slot architecture family. Do not freeze step360 as final N0 authority. Requalify/retrain it after upstream co-adaptation. Slot count 12, width 640, three layers, and three views remain operating points, not ceilings.

## 7. Downstream causal arbitration — preserve methodology, not overclaim

Sources:

- `docs/research/eipm-n0-causal-arbitration-full-stack-binding-v0.1.md`
- `docs/research/eipm-n0-downstream-causal-arbitration-finalization-v0.1.md`

The candidate-blind calibration, frozen common stack, exact hashes, one-arm change, and independent recomputation are strong causal methodology.

An `IMPROVEMENT` classification proves improvement relative to the frozen comparison stack. It does not prove that the frozen stack itself is full-envelope sufficient.

The governance already respected this: improvement never automatically promoted the graph or marked N0 complete.

**Decision:** preserve and reuse this methodology.

## 8. Query-edge / semantic repair chain — evidence to keep, architecture to retire

The stage-state sequence from the final latent failure through query-semantic, relation-role, multilayer, query-edge, competitive, setwise and dual-view experiments is valuable negative evidence.

It also exposes the process defect:

- semantic parent frozen;
- structured parent frozen;
- graph parent usually frozen;
- one small downstream mechanism allowed to change;
- local failure moved to the next boundary;
- another repair head was introduced.

Several experiments were scientifically disciplined one-change causal studies, but the **composition of their repair mechanisms is not a final architecture**.

The clean-sheet QSRE review correctly terminated this lineage.

**Decision:** retain failure/localization receipts and Fable seeds. Do not revive these heads as a stack.

## 9. QSRE T1/T2 — causal scaffolds, not product architecture

Sources:

- `src/alice_personality/n0/qsre_t1_executor.py`
- `configs/eipm/n0/n0_v02_qsre_t1_executor_design_v0_1.json`
- `configs/eipm/n0/n0_v02_qsre_t1_training_contract_v0_2.json`
- `configs/eipm/n0/n0_v02_qsre_t2_operator_learning_design_v0_2.json`
- `configs/eipm/n0/n0_v02_qsre_t2_training_contract_v0_3.json`
- `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.59.json`

T1 deliberately uses oracle operator/support and learned fixed embeddings for six relations, four roles, and five operations.

That PASS proves an executor can implement the intended relational transformations when the operator/support are already correct.

It does not prove open-schema semantic execution.

T2 retained fixed relation slots/classes and limited relation-step topology. v0.59 explicitly revoked further T2 GPU work because local scaffold repair was being mistaken for production closure.

**Decision:** T1/T2 weights and fixed vocabularies are experimental operating points. Preserve their causal mechanics and tests, not their fixed ontology.

## 10. Production Core v1 — preserve the clean-sheet design

Sources:

- `configs/eipm/n0/n0_v02_qsre_production_core_v1.json`
- `src/alice_personality/n0/qsre_production_core.py`
- `src/alice_personality/n0/qsre_production_operator_v3.py`

Strong properties:

- runtime-variable relation cardinality;
- no relation-count parameter axis;
- shared recurrent relation transition;
- no hop-specific learned slots;
- structural STOP;
- continuous relation hypotheses;
- separate unknown state;
- typed domain/range constraints;
- continuous operator state;
- exact non-relational fallback;
- full semantic width;
- no early exact relation sparsity.

The current v3 operator implementation is nevertheless bound to the now-failed frozen semantic-authority interface and seed factor banks.

**Decision:** preserve Production Core's factorization and execution doctrine. Replace the v3 semantic authority path with the joint semantic-operator foundation. Factor semantics become runtime schema states rather than opaque fixed learned class authority.

## 11. Binder v2 — preserve structural boundary, reopen semantic inputs

Source:

- `src/alice_personality/n0/qsre_production_binder_v2.py`

Strong properties:

- no relation-count-dependent parameters;
- no fixed top-k;
- adaptive exact-zero support;
- exact sparsity begins here rather than during semantic interpretation;
- relation semantics arrive through operator relation mass;
- P1 relation state cannot become hidden matching authority;
- zero/one/many support.

Defects/limitations for the successor:

1. Query matching uses only `query_hidden_states[:, -1]`, despite prior evidence that relation information can be layer-local.
2. Field token states are a single-layer interface.
3. Type compatibility uses integer type IDs and boolean schema masks. This is excellent as an exact structural constraint but does not itself provide open semantic type meaning.
4. The operator object feeding the binder still comes from fixed seed factor topology in current code.

**Decision:** preserve Binder v2's structural role and sparse mechanics. Replace its semantic input adapter with the new shared multi-layer/token foundation state. Keep exact type masks for structural validity while allowing dynamic type descriptors to supply semantic meaning upstream.

## 12. Final self-validation v1 — preserve history, supersede before closure

Sources:

- `configs/eipm/n0/n0_v02_final_self_validation_contract_v1.json`
- `scripts/eipm/n0/build_n0_v02_final_self_validation_v1.py`
- `scripts/eipm/n0/eval_n0_v02_final_self_validation_v3.py`

The contract has useful lineage/freeze discipline and strong relational checks.

It is not sufficient for the successor full-envelope architecture because:

- only 320 total rows;
- 160 relational rows;
- 20 relational families;
- 8 rows per family;
- only two final-only relations (`ENABLES`, `PREVENTS`);
- deterministic synthetic templates and finite entity banks;
- broad general-fabric success is measured largely by cosine back to the old semantic target geometry;
- structured/fusion/latent parent competence can therefore look strong while preserving a semantically insufficient representation.

**Decision:** never rewrite the historical frozen v1 result. Before N0 closure, create a successor final contract with natural public and adversarial synthetic language, relation-family/domain/template/entity/composition isolation, variable schema cardinality, dynamic factor/type schemas, multi-hop branching, unknown/defer, long-context sparse relevance, heterogeneous view stress, and independent behavioral targets rather than semantic-cosine preservation alone.

## 13. Cross-cutting root cause: self-referential validation

A repeated pattern exists across the old stack:

`semantic step80 target -> structured alignment -> evidence query/parent -> fusion semantic target -> latent semantic target -> general-fabric semantic cosine`.

A downstream module can therefore pass by preserving the old semantic geometry very accurately.

Once job 575990 proved that geometry insufficient for the new open-schema semantic/operator requirement, those downstream preservation PASS results can no longer be interpreted as independent evidence that the representation is sufficient.

This does **not** invalidate their local mechanics.

It means the successor validation must include independent behavioral interventions whose truth is not defined by agreement with the parent embedding.

## 14. Cross-cutting root cause: premature freezes

Temporary freezes were valuable for causal localization. The failure was allowing those freezes to harden into architecture assumptions.

Successor rule:

- freeze temporarily to isolate a causal question;
- never infer permanent foundation sufficiency from a local PASS;
- before permanent freeze, test the full downstream interface envelope;
- if later evidence localizes a deficit below a frozen boundary, reopen it;
- preserve historical receipts and ratifications when reopening;
- after joint full-envelope sufficiency is demonstrated, freeze may become durable again.

## 15. Scale audit

No source inspected here proves that the current 136M semantic model, 1.65M structured encoder, 117M fusion model, or 29.6M latent pool is globally too small.

There is also no evidence that those sizes are sufficient forever.

The earlier project correctly removed many numeric runtime ceilings. The remaining danger is subtler:

> a migratable fixed vocabulary/topology is still a closed semantic interface during that checkpoint's lifetime.

Therefore the successor architecture must make runtime-varying semantic categories first-class wherever they are expected to vary at runtime. Checkpoint migration remains acceptable for changes that are genuinely architectural, not merely new semantic labels.

## 16. What is preserved for the successor

Preserve unless new evidence falsifies it:

- semantic tokenizer and step80 initialization;
- token-level/multi-layer semantic states;
- permutation-safe structured set processing;
- explicit confidence/missing/provenance signals;
- query-conditioned evidence specialization;
- dual-endpoint graph message mechanics;
- source-anchored multi-stream fusion;
- competitive multi-slot latent representation;
- causal arbitration methodology;
- Production Core's dynamic relation/recurrent execution factorization;
- ordered query-evidence coverage;
- continuous relation hypotheses;
- exact non-relational fallback;
- Binder v2 delayed adaptive structural sparsity;
- zero-parameter QSRE-to-N0 bridge principle;
- frozen-evaluation lineage discipline.

## 17. Must reopen before N0 closure

- public semantic representation objective;
- semantic/operator co-adaptation;
- structured semantic/type descriptor interface;
- evidence adapter query interface;
- graph relation semantic interface;
- fixed factor-bank semantic authority;
- Binder final-layer-only semantic read;
- fusion/latent weights after upstream representation changes;
- old final validation contract.

## 18. Next build boundary

Before any optimizer or GPU work:

1. define the full-envelope public semantic/operator curriculum;
2. implement runtime-variable schema/factor/type tensor contracts;
3. implement one shared schema-conditioned semantic/operator mechanics module;
4. add static tests for cardinality/permutation/uncertainty/recurrent-step invariants;
5. add curriculum leakage/shortcut audit;
6. define successor structured/evidence/binder interfaces;
7. define a successor full-envelope final-validation contract;
8. only then freeze one training plan.

No P2S-v3, P2A-v4, another router chain, threshold tuning, or blind scale search is authorized.
