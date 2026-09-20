# N0 Clean-Sheet Relational Execution Architecture Audit v0.1

**Date:** 2026-09-20  
**Status:** architecture-review authority; implementation and gradient work frozen  
**Branch:** `alice-eipm-v1-n0-clean-sheet-relational-execution-review`  
**Starting revision:** `d5d2a4f265d5d10a0e9cb6c12c60c864d257c495`

## 1. Why this audit exists

The recent N0 sequence produced multiple genuine model-learning failures despite repeated statements that the project would avoid a local repair loop.

The important correction is to stop treating each failed model as an independent bug.

The failures form a longer lineage around one repeated architectural boundary:

> natural-language query semantics -> relation/argument interpretation -> graph evidence selection -> field-level decision

Across that lineage, the implementation repeatedly changed the mechanism that injects query information while preserving a common downstream assumption: a query-independent or weakly query-conditioned global field scorer remains authoritative, and new relational capability is expected to alter its result through residual biases, gates, mixtures, or routing.

This audit reopens that assumption.

No architecture implementation is authorized by this document.

---

## 2. N0 role remains broader than graph question answering

The governing EIPM role contract remains authoritative.

N0 is the public, identity-neutral semantic/judgment foundation required before private identity learning. It must support:

- language and pragmatics;
- evidence and uncertainty;
- typed structured context;
- relation/evidence reasoning;
- candidate comparison;
- relationship/context operations;
- heterogeneous context fusion;
- multi-view latent representation.

The current relation/evidence failure is one capability inside N0. A replacement must not collapse N0 into a narrow KG-QA system.

The clean-sheet relational component must therefore be a **composable N0 capability**, not a replacement for the semantic backbone, structured state, fusion system, latent representation, or eventual IDP architecture.

---

## 3. What remains supported

### 3.1 Semantic substrate

The native semantic model remains supported.

Evidence:
- token-level representations are first-class outputs;
- later audits recovered strong relation and binding information from token states;
- the binding-identifiability audit showed perfect DEV edge identification from token-level late interaction across every audited hidden layer;
- no current result localizes the failure to semantic width, depth, tokenizer, or general language capacity.

Do not retrain or scale the semantic backbone from the current failure.

### 3.2 Structured state

The typed structured-state encoder remains useful as a representation of field type, provenance, relation role, temporal scope, confidence, missingness, and semantic field content.

The current failures do not establish that its width or depth is insufficient.

Its pooled output must not be treated as the only representation available to relational execution.

### 3.3 Evidence representations

The evidence-view adapter and directed graph parent contain useful learned evidence and endpoint structure.

The dual-endpoint parent demonstrated strong explicit endpoint-role behavior and strong preservation performance.

This does **not** imply that its global field softmax must remain the final executor for every relational query.

### 3.4 Cross-context fusion

The ratified multi-stream fusion architecture remains a supported N0 component.

It preserves semantic, structured, and evidence views separately, supports cross-attention, and retains source anchors.

The relational-execution redesign must produce outputs that can participate in this multi-view fabric rather than replacing the fabric with one graph score.

### 3.5 Adaptive latent representation

The adaptive latent-pool lineage remains separate from the current relational-execution diagnosis.

Its earlier failures and validator corrections remain historical evidence. The present graph-selection failures do not justify deleting the latent architecture or declaring it solved.

No latent training is authorized by this audit.

---

## 4. Full failure lineage

The failure family predates the four most recent architectures.

### A. Relation-semantic grounding

Job 575797 attempted to teach natural relation semantics through the existing mutable relation-read path.

Result:
- natural relation-semantic heldout performance remained weak;
- explicit endpoint-role behavior degraded;
- ordinary replay stayed comparatively healthy.

Interpretation at the time:
- natural relation meaning and explicit endpoint role were entangled inside the same read path.

### B. Semantic role residual

Job 575798 isolated a zero-initialized source/target semantic residual while freezing the parent.

Result:
- DEV failed before heldout;
- additive role residual did not solve relation-semantic composition.

### C. Query-relation-role router

Job 575800 created a separate raw-query + relation-type SOURCE/TARGET/DEFER router and mixed its field distribution with the parent at probability level.

Result:
- preservation became acceptable at later checkpoints;
- role accuracy collapsed toward a global endpoint preference;
- family-min role accuracy remained 0;
- heldout stayed closed.

Important implication:
- even a probability-level side expert, rather than a scalar residual, did not produce robust compositional role binding.

### D. Relation-conditioned multi-layer interface

Job 575825 used intermediate token layers and relation-conditioned token/layer selection.

Result:
- some causal metrics improved;
- endpoint preservation collapsed;
- interface had no endpoint-content-specific binding;
- forced +source/-target antisymmetry was rejected.

### E. Query-edge cross-attention bridge

Job 575888 added actual source/target graph content to the language bridge.

Result:
- parent preservation was much better;
- query-edge representation was demonstrably edge-specific;
- routing supervision targeted a proxy score while runtime contribution used another gate;
- runtime control mismatch remained.

### F. Competitive edge router

Job 575907 made the supervised route distribution the actual runtime contribution distribution and added explicit no-op.

Result:
- preservation passed;
- no-op behavior learned;
- edge identity remained around chance.

The problem was no longer merely parent preservation or specialist activation.

### G. Setwise router

Job 575908 was **not model evidence**. It failed before gradient because of a stale trainer guard.

The later exact preflight work was stopped before becoming another expensive qualification loop.

Do not count 575908 as a learned-model failure.

### H. Binding-identifiability audit

The no-gradient audit showed:
- lexical edge identity DEV = 1.0;
- final mean pooled semantic edge identity remained high but imperfect;
- token late interaction edge identity = 1.0 for every audited semantic layer;
- relation/field information was present before structured compression.

This falsified the hypothesis that another increasingly complex learned edge router was the obvious next move.

### I. Dual-view late-interaction specialist

The dual-view design froze edge identity to the semantic token-level binding path, separated parent-vs-specialist activation, and retained directional residual execution.

Job 575912 still failed its DEV/preservation gate.

Job 575913 then performed counterfactual localization over the trained checkpoints.

At step 200:
- actual activation + soft binding row accuracy: 0.1667;
- forced specialist + soft binding: 0.1875;
- actual activation + model-predicted hard-top1 binding: 0.3681;
- forced specialist + model-predicted hard-top1 binding: 0.3819;
- relevant-edge source/target residual proposal role accuracy: 0.7153.

Relation-level residual role accuracy at step 200 included:
- causes: 1.0;
- supports: 0.5;
- corrects: about 0.542;
- derived_from: 0.75;
- supersedes: 0.75;
- temporal_successor: 0.75.

This is decisive for the clean-sheet review:

1. specialist activation is not the dominant failure;
2. diffuse multi-edge execution materially hurts;
3. even oracle-like hard support does not solve the task;
4. role/argument execution itself is not systematic;
5. therefore another gate/router replacement is not justified.

---

## 5. Common architectural invariant behind the failures

The recent architectures differ substantially upstream, but most of them preserve a common decision abstraction.

The parent graph ultimately computes a global field distribution.

Specialist architectures then do one of the following:

- add source/target scalar biases;
- mix a side field distribution with the parent;
- route residuals to fields;
- add edge-specific source/target residuals to the same global field competition.

The dual-view architecture improved semantic binding but still reduces relational execution to source/target scalar corrections before a global field softmax.

This creates two structural problems.

### 5.1 Binding does not define the execution support

A query can identify the relevant edge correctly, yet unrelated fields continue participating in the final global competition.

The hard-binding counterfactual improvement in job 575913 directly supports this diagnosis.

Correct binding currently **nudges** the global scorer instead of defining the structured support on which the relational computation runs.

### 5.2 Argument role is not a first-class computational object

SOURCE/TARGET meaning has repeatedly been asked to emerge indirectly from:
- relation type;
- query embedding;
- endpoint representations;
- residual MLP heads.

The curriculum contains reversals that require a systematic distinction between relation argument positions.

Yet relation-family results show that the model can memorize or learn some directional families while failing others at chance.

This is evidence against relying on implicit scalar endpoint preference as the sole role-binding mechanism.

---

## 6. Frontier research used for the clean-sheet review

The following work is architectural precedent, not a drop-in implementation specification.

### Neural Bellman-Ford Networks
Zhu et al., 2021  
https://arxiv.org/abs/2106.06935

Relevant principle:
- graph states are explicitly conditioned on the query relation;
- query conditioning is present inside message passing;
- pair/path representations are computed as the actual reasoning process rather than as a post-hoc correction to query-independent node scores.

### LILaC
2025  
https://aclanthology.org/2025.emnlp-main.1037/

Relevant principle:
- fine-grained late interaction can score graph structure;
- selected structure then participates in graph traversal/reasoning;
- binding and graph execution are separate but coupled stages.

### QSRAG / QR-GAT
2026 Findings of ACL  
https://aclanthology.org/2026.findings-acl.398/

Relevant principle:
- query and relation information should affect graph attention and triple scoring;
- query-independent propagation followed by late correction can dilute relation-specific evidence.

### SALR
Gao et al., 2026  
https://arxiv.org/abs/2609.20398

Relevant principle:
- maintain continuous latent reasoning while anchoring it to explicit schema elements;
- avoid committing too early to one discrete intermediate interpretation;
- multiple schema candidates can influence subsequent computation before final commitment.

Transferable lesson:
A.L.I.C.E. should not be forced into a rigid logical-form parser, but a continuous operator state can still be explicitly anchored to relation/role structure.

### Counterfactual MoE routing analysis
Microsoft Research, 2026  
https://www.microsoft.com/en-us/research/publication/when-are-experts-misrouted-counterfactual-routing-analysis-in-mixture-of-experts-language-models/

Relevant principle:
- router confidence is not equivalent to downstream expert utility;
- counterfactual execution is necessary to determine whether routing is actually the bottleneck.

This supports the interpretation of job 575913: the no-op/specialist gate should not receive another redesign simply because routing is imperfect.

### Adaptive sparse attention
DashAttention, 2026  
https://arxiv.org/abs/2605.18753

Relevant principle:
- sparse support can have input-dependent cardinality;
- differentiable sparse support avoids the permanent fixed-k assumption;
- irrelevant regions can receive exact zero mass before fine-grained execution.

This is evidence for investigating **adaptive sparse support**, not for hard-coding top-1.

### Role/filler binding and systematicity
https://doi.org/10.1177/09637214241279504

Relevant principle:
- systematic generalization benefits when relational roles and fillers are represented/bound explicitly rather than relying entirely on implicit memorized conjunctions.

This supports making relation argument position a first-class execution variable.

---

## 7. Clean-sheet relational execution contract

This section defines required computation. It does **not** select implementation details such as entmax versus SparseMAP, transformer versus GNN, or exact layer count.

### Stage A — Query operator state

Input:
- token-level query semantics;
- ACFP typed context;
- known relation vocabulary where available;
- query uncertainty.

Output:
a continuous query-operator state containing recoverable information about:
- requested semantic relation/operator;
- requested argument role or answer role;
- temporal/status semantics;
- uncertainty/plurality;
- whether a relational executor is applicable.

Requirements:
- do not collapse to one raw mean-pooled vector;
- do not require one universal semantic layer;
- do not require a closed permanent relation ontology;
- relation/role anchors may be explicit while the full state remains continuous.

### Stage B — Adaptive structural support

Use fine-grained semantic interaction between the query operator and field/edge representations.

The support must:
- suppress irrelevant fields/edges before relational execution;
- support zero, one, or many active pieces of evidence;
- preserve ambiguity when multiple supports remain plausible;
- be permutation-equivariant where field/edge order has no meaning;
- expose confidence and provenance;
- not use a permanently fixed top-k as a capability rule.

Top-1 may be used only as a diagnostic counterfactual.

### Stage C — Query-conditioned relational execution

The selected support is processed by an executor in which query/operator information conditions the actual relational computation.

At minimum, each active directed relation must expose:
- relation identity/state;
- source argument state;
- target argument state;
- source role;
- target role;
- query-requested role/operator state;
- provenance/confidence/temporal metadata where relevant.

Query information must influence MESSAGE/ATTENTION/UPDATE computation, not merely the final field logits.

Multi-hop composition must be possible without replacing the architecture family.

### Stage D — Structural readout

The relational result must be read from the query-conditioned executor state.

It must **not** be defined as:

`frozen_global_field_score + specialist_scalar_residual`

for all relational queries.

A correct support must be able to determine the relational result without unrelated outside fields defeating it through an unrelated global softmax.

The output may be:
- one field;
- multiple fields;
- a relation/path state;
- a distribution over alternatives;
- defer/not-applicable;
depending on the query.

### Stage E — General parent/fallback composition

The existing parent remains valuable, but its role changes conceptually.

Candidate roles:
- general evidence representation provider;
- fallback for non-relational queries;
- preservation teacher;
- prior/context source;
- auxiliary view for fusion.

The parent must not automatically remain the authoritative global decision surface for a query that the relational executor has validly claimed.

Applicability/fallback is a separate decision from edge identity and argument-role reasoning.

### Stage F — N0 fabric output

Relational execution must emit a rich view for the existing N0 fusion/latent fabric.

It should expose:
- contextualized evidence states;
- selected/support distribution;
- relation/operator state;
- role-binding state;
- uncertainty/plurality;
- provenance/reliability;
- optional path/subgraph state.

It should not reduce the entire capability to one scalar winner.

---

## 8. Proof obligations before architecture implementation

No proposed architecture family may move to a trainer until it satisfies the following on paper and through oracle/no-gradient tests where applicable.

### P0 — N0 breadth preservation
The design remains one composable N0 view/capability and does not replace semantic, structured, fusion, or later identity systems with KG-QA logic.

### P1 — Oracle support closure
With gold relevant support supplied and no learned support selection, the executor must be capable of solving source/target role reversals without using the old global field scorer as the answer mechanism.

If this fails, support routing is not the blocker.

### P2 — Oracle role closure
With relevant edge/support supplied and requested argument role supplied explicitly, the executor must distinguish SOURCE/TARGET behavior across all relation families.

If this fails, the executor's relation/argument representation is insufficient.

### P3 — No global-competition leak
Fields outside an oracle-selected support must not be able to win the relational answer solely through the old parent global softmax.

### P4 — Adaptive support cardinality
The architecture must support:
- no relational support;
- one edge;
- multiple independent supports;
- multi-edge/multi-hop support;
without changing architecture family.

### P5 — Ambiguity preservation
Two genuinely co-valid supports may remain live. The architecture must not force arbitrary singleton selection merely because current training examples contain one answer.

### P6 — Compositional path capability
A relation chain must be executable as ordered relational computation. Noncommutative relation order must remain distinguishable.

### P7 — Explicit role intervention
Changing only the requested argument role on the same query/evidence graph must predictably change the execution/readout.

### P8 — Relation intervention
Changing relation type with endpoints fixed must affect execution where semantics require it.

### P9 — Endpoint intervention
Changing endpoint content with query/relation fixed must affect execution where semantics require it.

### P10 — Permutation invariance/equivariance
Reordering unordered fields/edges must not alter semantic results.

### P11 — Provenance/reliability causality
Reliability/provenance metadata must affect computation when behaviorally required, while irrelevant metadata changes must not cause arbitrary answer flips.

### P12 — Parent fallback isolation
Non-relational examples must be able to defer cleanly without relational side effects.

### P13 — Representation sufficiency before training
No-gradient probes must verify that query/operator, field/edge, and role information required by the executor is available at its inputs.

### P14 — Fresh lexical/entity generalization
TRAIN/DEV/TEST templates, entities, and attributes must remain split enough that success cannot come from memorized query phrasing.

### P15 — No private identity gradient
All work remains public and identity-neutral at N0.

### P16 — No permanent capacity ceilings
Current relation vocabulary, support size, hop count used by a test, field count, layer count, or execution budget is not a product capability ceiling.

---

## 9. Architecture families to evaluate before coding

The next review should compare at least these families against the proof obligations.

### Family 1 — Query-conditioned sparse relational GNN
- adaptive sparse support from token/edge interaction;
- query/operator-conditioned message passing;
- explicit directional argument roles;
- structural readout over support.

Closest precedents:
NBFNet + modern query-conditioned graph attention + adaptive sparse selection.

### Family 2 — Schema-anchored continuous relational executor
- continuous query/operator latent;
- relation/role codebook or migratable schema anchors;
- repeated latent-schema feedback;
- execution over graph support;
- delayed final commitment.

Closest precedent:
SALR, adapted away from fixed KB semantic parsing.

### Family 3 — Relational set/graph transformer
- query/operator tokens and typed edge/argument tokens participate in joint attention;
- adaptive sparse attention/support;
- explicit source/target role embeddings;
- structural output tokens rather than field residuals.

This family may preserve more continuous nuance but must prove that relation direction and graph topology are not lost inside generic attention.

### Hybrid family
A hybrid may be appropriate if:
- token-level late interaction is best for semantic support discovery;
- structured query-conditioned message passing is best for relational execution;
- a continuous schema/operator state is best for relation/role composition.

No hybrid is authorized merely because it contains more mechanisms. Each mechanism must correspond to a proven failure boundary.

---

## 10. Architecture patterns now closed

Do not create another member of these families unless new evidence directly falsifies this audit:

1. query-only relation scalar added to parent field logits;
2. +source/-target antisymmetric residual;
3. raw-query SOURCE/TARGET/DEFER side router whose main role is to mix with the same parent field distribution;
4. learned edge router whose selected specialist still only adds scalar endpoint residuals to the same global field scorer;
5. another setwise/competitive router whose goal is merely to improve edge identity;
6. another no-op/specialist gate redesign before execution is repaired;
7. longer training / smaller LR / wider MLP / larger routing-loss weight as substitutes for execution redesign;
8. hard-top1 as a permanent architecture rule;
9. semantic-backbone retraining without evidence that required information is absent;
10. generic parameter scaling without localized capacity evidence.

---

## 11. Next work authorized

Authorized:
1. architecture comparison against proof obligations P0-P16;
2. oracle/no-gradient **executor-contract** diagnostics that do not train a new model;
3. data-contract design for multi-support, ambiguity, and multi-hop examples;
4. research notes and static interfaces;
5. migration planning for reuse of proven semantic/structured/evidence/fusion checkpoints.

Not authorized:
- new learned relational architecture implementation;
- optimizer creation;
- gradient;
- P100 training;
- TEST opening;
- frozen challenge rerun;
- parameter scaling;
- semantic retraining;
- parent graph retraining;
- private identity gradient;
- promotion;
- N0 completion.

The next architecture implementation requires a separate decision document showing that one family satisfies the proof obligations better than the alternatives and does not recreate the closed residual/global-softmax pattern.

---

## 12. Working hypothesis

The strongest current hypothesis is:

> N0 needs a query-conditioned relational executor whose semantic binding determines an adaptive structural support, whose relation/argument roles are first-class variables inside the executor, and whose structural result is read from that executor rather than expressed as residual corrections to an unrelated global field scorer.

This is a **research hypothesis**, not yet an implementation decision.

The purpose of the next comparison is to determine the simplest production-real architecture that satisfies this contract without narrowing A.L.I.C.E.'s broader N0 role.
