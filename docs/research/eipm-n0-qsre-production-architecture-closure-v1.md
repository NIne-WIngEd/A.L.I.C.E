# QSRE Production Core v1 — Architecture Closure Before Further GPU Work

**Date:** 2026-09-20
**Status:** production-architecture authority; implementation/training remains closed
**Supersedes as production candidates:** T1 executor scaffold, T2 v0.1/v0.2 operator, unrun T2 v0.3 schema-ordered prototype
**Authoritative state:** configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.59.json

## 1. Correction

The project had reached the exact pattern it intended to avoid.

T1 was deliberately a causal scaffold: fixed current relation IDs, fixed current operation IDs, oracle support, and an executor trained only to answer whether execution can work when semantics and binding are correct.

T2 was deliberately another causal scaffold: recover the T1 operator from language while support stays oracle.

Those scaffolds were then being treated too much like incrementally improvable production modules. T2 v0.2 failed relation identity/order, and the immediate response was a relation-only v0.3 prototype. That is useful diagnosis, but another GPU run would still leave known production-architecture gaps outside the tested failure.

Therefore T2 v0.3 is frozen unrun.

No more P100 work occurs until operator, binder, executor, uncertainty, schema migration, and fallback interfaces are closed together against the full N0 workload.

The rule is now: causal stages remain sequential experiments, but the architecture they instantiate is designed once against the full workload before the experiments begin.

## 2. What the v0.2 failure actually proved

Job 575954 was valid model evidence. The frozen semantic token stack supported operation prediction up to 1.0, control prediction up to 1.0, and relational role prediction up to 0.96484375.

It did not support the existing T2 relation interface strongly enough: relation-sequence exact max 0.5381944, full operator exact max 0.4618056, query-view consistency max 0.5416667, and ordered-path family remained weak.

This localizes the observed failure to relation/operator extraction. It does not prove that changing only that extractor yields a production-complete architecture.

## 3. Known weaknesses that make T2 v0.3 insufficient as the next GPU architecture

### 3.1 Relation vocabulary is still a checkpoint-shaped axis

T2 v0.3 still has num_relations=6 in model configuration. Its schema is semantically grounded, but relation cardinality is still part of tensor/model shape. Production must accept relation schema cardinality R at runtime without adding or removing learned relation-class parameters.

### 3.2 Hop count is still a learned slot count

T2 v0.3 has two learned relation-slot queries. Adding a third step still changes learned parameter shape. Production must use a shared iterative relation-step cell with learned STOP and an external runtime compute budget. The same weights must execute one, two, three, or more current reasoning steps.

### 3.3 Early argmax destroys operator plurality

The current bridge converts relation, role, operation, and control logits to argmax before T1 execution. That contradicts the workload requirement to preserve ambiguous/plural operator hypotheses. Production must retain a sparse distribution or adaptive hypothesis set until structural evidence resolves ambiguity.

### 3.4 Operation is incorrectly modeled as one exclusive class

Current T1/T2 chooses exactly one of ROLE_SELECT, PATH_FOLLOW, AGGREGATE, PREFER_RELIABILITY, or PREFER_LATEST. These are not all mutually exclusive. A production query can require path following and latest-state arbitration, or aggregation and reliability preference. Production must factor traversal and arbitration into composable dimensions.

### 3.5 Control/applicability is discretized too early

Current T2 converts a three-way control class into fixed applicability centers 0.10/0.90/0.50. That is a causal-study bridge, not a calibrated production interface. Production must carry continuous applicability, uncertainty, and defer/fallback semantics.

### 3.6 The continuous operator state is currently thrown away at execution

T2 learns a continuous operator state but passes a zero context vector into T1. That was correct for the T2 causal experiment. It is not acceptable for final QSRE: the executor must receive the continuous semantic operator state.

### 3.7 T1 still terminates in a fixed relation ontology

T1 uses nn.Embedding(num_relations, d) and receives integer relation IDs. Therefore even a dynamic natural-language relation schema in T2 terminates at a fixed learned ID embedding in the executor. Production must pass semantic relation state into executor and edge representations directly. IDs may be cache keys, not semantic authority.

### 3.8 Current schema grounding is type-blind

T2 v0.3 relation schema provides glosses and direction but not argument type/domain constraints, range constraints, symmetry/inverse metadata, temporal/provenance compatibility, or expected readout type. Production grounding must use those constraints before support is finalized.

### 3.9 Schema glosses are collapsed too early

The unrun v0.3 prototype averages relation gloss token states to one state per hidden layer before scoring. Production schema memory must retain schema-token states or multiple semantic facets rather than one averaged relation vector.

### 3.10 Binder compatibility was not closed before operator redesign

T3 has not been production-implemented. If operator representation is finalized without defining how relation hypotheses, endpoint roles, type constraints, temporal constraints, and ambiguity enter support selection, T3 can force another operator redesign later.

## 4. Production QSRE Core v1

The architecture family remains QSRE. The correction is to make the original blueprint real rather than upgrade causal scaffolds one failure at a time.

Data flow:

frozen N0 semantic token stack + dynamic typed relation schema -> continuous schema-grounded operator inducer -> token-level type-aware structural binder -> schema-conditioned relational executor -> support-local structural readout -> separate fallback/N0 fusion.

## 5. Dynamic typed schema memory

Each relation entry supplies: relation key/version, one or more natural-language descriptions, source-argument description, target-argument description, symmetric/asymmetric flag, inverse relation when known, domain/source type constraints, range/target type constraints, temporal semantics tags, provenance/evidence semantics tags, and schema provenance/version.

The semantic backbone encodes relation schema entries to token-level states. Schema encoding is cacheable.

Relation cardinality R is runtime data. There is no learned [R,D] class table. Adding a relation changes schema data, not parameter shape.

The operator includes UNKNOWN separate from STOP. STOP means the relational program is complete. UNKNOWN means the query is relational but supplied schema cannot ground it confidently. UNKNOWN must not be coerced into the nearest known relation.

## 6. Continuous schema-grounded operator inducer

Inputs: query hidden stack [B,L,T,D], dynamic relation schema token states [R,L,Ts,D] or equivalent cached multi-facet states, generic structural role schema, and stable operation-factor schema.

Use one shared decoder cell across relation steps. At each step it consumes query token memory, previous continuous decoder state, previous sparse relation distribution/schema expectation, and accumulated operator context. It emits a sparse distribution over the dynamic relation schema, STOP probability, UNKNOWN probability, and updated continuous state.

The decoder iterates until STOP or an external runtime budget. The budget is a serving policy, not a learned slot-count parameter. No per-hop learned query parameter exists.

Use adaptive sparse relation hypotheses rather than early argmax. Preserve the continuous residual state as mandatory executor input.

Endpoint role remains a distribution over SOURCE, TARGET, SYMMETRIC, and NONE until execution/readout.

Replace one exclusive operation class with orthogonal factors. Traversal mode: LOCAL_SELECT, PATH, AGGREGATE. Arbitration modifiers: RELIABILITY, RECENCY, TEMPORAL_CONSTRAINT, PROVENANCE_CONSTRAINT. Modifiers can be simultaneously active.

Expose continuous applicability, fallback/defer state, and uncertainty. Do not map them to fixed centers.

## 7. Token-level type-aware adaptive binder

The binder is designed now even though it is trained later.

Inputs for each field/edge: query token states, field token states, edge relation schema token/state, source/target identity, field types, relation domain/range compatibility, provenance, reliability, temporal metadata, and current sparse operator hypotheses.

The score supervised during training must be the score controlling runtime support. No proxy router.

Support cardinality may be zero, one, many, or ambiguous. Use an exact-zero differentiable sparse family for ordinary support. No permanent top-k.

When multiple operator hypotheses remain plausible, binder support may remain a sparse union/mixture rather than forcing one relation early.

## 8. Schema-conditioned relational executor

Remove relation-ID semantic authority. Replace learned relation ID embeddings with projections of dynamic relation schema states. Edge relations carry semantic schema states. IDs are allowed only as cache/index keys.

Use one shared edge/node update cell for every relation step. Step count comes from the operator program, not hop-specific parameters.

SOURCE and TARGET incidence remain explicit structural roles.

Every edge/node update receives continuous operator state, current relation semantic state, sparse relation confidence, traversal mode, arbitration modifiers, and temporal/provenance/reliability state.

Reliability/latest modify traversal/aggregation/readout and are not mutually exclusive executor modes.

Plural operator/support hypotheses remain represented until calibrated collapse or N0 fusion.

## 9. Structural readout and fallback

Readout competes only inside active support. It exposes supported-node relevance, supported-edge/path relevance, relational summary, operator/path state, uncertainty, and plural alternatives.

No parent global field scorer competes with QSRE inside this readout.

Fallback remains a separate N0 composition decision. If relational applicability/support is absent, relational contribution is structurally inactive and base evidence path remains unchanged.

## 10. Production architecture proof obligations before another GPU run

A. Same source/checkpoint executes with 3, 6, and 11 relation schema entries without parameter creation or checkpoint migration.

B. Same weights execute 1, 2, 3, and 5 relation steps without hop-specific parameter creation.

C. Schema permutation equivariance.

D. Semantic schema intervention: holding cache key fixed while changing supplied schema meaning changes operator/executor state, proving IDs are not semantic authority.

E. Type/domain/range contradiction: a lexically attractive but type-incompatible relation loses to a compatible candidate.

F. UNKNOWN and STOP are distinct.

G. Ambiguous query retains multiple relation hypotheses through operator/binder interface.

H. Mechanics represent PATH+latest, AGGREGATE+reliability, and ROLE_SELECT+temporal constraint without new operation classes.

I. Changing only continuous residual state while discrete factors stay fixed can change executor state.

J. Same interface executes zero, one, two, and many supported edges.

K. Relation sequence reversal changes execution where semantics require it.

L. Field/edge permutation invariance/equivariance.

M. Exact non-relational pass-through.

N. Totality fuzz: every operator state the learned inducer can produce either executes or explicitly defers/fails, never crashes.

O. Full W1-W15 interface mapping: every workload family is assigned to QSRE, another N0 module, or later N1/N2/N3.

## 11. Training/data proof obligations fixed before gradient

Include an open-schema lane where some DEV relation semantics are absent as relation classes from TRAIN and supplied only by descriptions/schema. This proves schema entries are causal rather than decorative wrappers around memorized IDs.

Include relation-order compositional holdout, operation-composition holdout, cue-collision/type-incompatible hard negatives, paraphrase-pair consistency, and ambiguity/plurality rows.

## 12. One precommitted causal build pipeline

All stages use the same Production Core v1 interfaces fixed in advance.

P0 — Static/CPU architecture closure: proof obligations A-O, stupid baselines, real Bash/uDocker behavior, real Magnolia artifact loading, and totality fuzz. No gradient.

P1 — Production schema-conditioned executor/readout with oracle operator + oracle support. If P1 fails, stop. Do not modify operator or binder.

P2 — Freeze exact P1 executor and train production operator inducer with oracle support. Includes open-schema, dynamic-step, compositional-operation, ambiguity, and paraphrase lanes. If P2 fails, stop. No LR/step patching.

P3 — Freeze proven operator/executor and train adaptive binder under zero/one/many/ambiguous support. If P3 fails, stop.

P4 — End-to-end QSRE integration only after independent P1/P2/P3 success.

P5 — N0 fusion consumption only after standalone QSRE works.

This is sequential scientific causality without owner permission micro-gates. The eventual Magnolia launcher may submit dependent jobs in one owner action with afterok fail-closed boundaries.

## 13. Research alignment

SALR (2026) supports schema-mediated latent reasoning and delayed discrete commitment. Layer-Order Inversion (2026) argues against assigning reasoning hops to fixed transformer-depth progression. SAGA (2026) highlights construction-time type/domain/range compatibility rather than lexical-only schema grounding. Recent zero-shot relation-extraction work supports relation descriptions, rejection mechanisms, and semantically hard negatives for dynamic/open-schema relation handling.

These are precedents, not A.L.I.C.E. authority. The design is driven by A.L.I.C.E.'s own failure evidence and W1-W15.

## 14. Decision

Do not run T2 v0.3.

Preserve its code as an isolated diagnostic prototype.

The next engineering work is Production QSRE Core v1 implementation and proof-obligation closure, not T2 v0.4.

There will be no additional GPU architecture version between the current state and a production-core causal pipeline unless the static/CPU architecture proof itself identifies a contradiction before any gradient is spent.
