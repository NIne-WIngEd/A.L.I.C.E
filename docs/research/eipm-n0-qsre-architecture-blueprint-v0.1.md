# QSRE Architecture Blueprint v0.1

**Date:** 2026-09-20  
**Status:** concrete architecture blueprint; no trainable implementation authorized  
**Family:** Query-Conditioned Sparse Relational Executor (QSRE)  
**Depends on:**
- `docs/research/eipm-n0-clean-sheet-relational-execution-audit-v0.1.md`
- `docs/research/eipm-n0-relational-execution-family-comparison-v0.1.md`
- `docs/research/eipm-n0-qsre-preimplementation-contract-result-v0.1.md`

## 1. Architecture objective

QSRE is not a replacement for N0.

It is the relation/evidence execution capability inside the broader N0 fabric.

Its job is:

> Given a semantic query, typed evidence fields, directed relations, provenance/reliability/temporal metadata, and potentially ambiguous evidence, construct a query-conditioned relational state that identifies relevant structural support, executes relation/argument operations on that support, and exposes a structured relational result to the rest of N0.

It must not reproduce the rejected pattern:

`parent global field scores + query-conditioned scalar corrections`.

---

## 2. High-level data flow

```text
frozen semantic multi-layer token states
                    │
                    ▼
        QUERY OPERATOR ENCODER
        ├─ continuous intent state
        ├─ relation-anchor state
        ├─ requested argument-role state
        ├─ temporal/status state
        ├─ applicability state
        └─ uncertainty/plurality
                    │
                    ├──────────────────────┐
                    │                      │
                    ▼                      ▼
 field token states + graph         relation / role anchors
          │
          ▼
 TOKEN-LEVEL STRUCTURAL BINDER
          │
          ▼
 adaptive sparse support
 zero / one / many fields + edges
          │
          ▼
 QUERY-CONDITIONED RELATIONAL EXECUTOR
 node states <-> edge states
 explicit source/target argument positions
 query-conditioned edge/node updates
          │
          ▼
 STRUCTURAL READOUT
 relational candidate distribution
 path/subgraph state
 role-binding state
 uncertainty / plurality
          │
          ▼
 evidence-view contextualized tokens + relational summary
          │
          ▼
 existing N0 fusion / latent fabric
```

General parent evidence behavior remains available in parallel.

QSRE does not mix its result into the parent's global field distribution.

---

## 3. Reused components

### 3.1 AliceN0V02Model — reuse

Use the existing frozen semantic backbone.

Required outputs:
- token states;
- hidden-state stack when operator extraction needs intermediate layers;
- attention masks.

Reason:
- directional relation information was demonstrated at intermediate layers;
- token-level edge binding was demonstrated strongly;
- current failures do not establish semantic-capacity insufficiency.

No new semantic encoder is justified.

### 3.2 StructuredStateEncoder — reuse

Use typed structured features as field metadata/context:
- field type;
- provenance;
- relation role metadata;
- temporal scope;
- confidence;
- missingness.

Do not collapse the executor to its pooled state.

### 3.3 EvidenceViewAdapter — reuse

Use evidence-specialized field representations as one base node representation.

Keep the adapter frozen in the first causal architecture experiment unless a later probe establishes that its representation destroys required information.

### 3.4 DualEndpointEvidenceGraphEncoder — partial reuse

Potentially reuse:
- relation embeddings;
- learned graph-conditioned field representations;
- endpoint-aware representation competence;
- frozen parent behavior for non-relational/general evidence fallback.

Retire as universal relational authority:
- `field_weights` is not the QSRE answer distribution;
- parent global pooling is not the final relational decision surface.

QSRE should consume useful parent/evidence states, not fight the parent's field softmax.

### 3.5 CrossContextFusion — reuse

QSRE should return an evidence-compatible contextualized stream and a relational summary.

The ratified fusion model remains the broader heterogeneous-context mechanism.

QSRE must not become a hidden replacement for semantic/structured/fusion capability.

---

## 4. Module A — Query Operator Encoder

### 4.1 Why it exists

Earlier failures showed two facts at once:

1. relation-direction information exists in the semantic backbone;
2. that information is layer-local and can disappear from final pooled views.

Therefore the query operator must have access to multi-layer token information and must not assume one fixed final layer.

### 4.2 Input

- semantic hidden-state stack `[B, L, T, 640]`;
- query content mask;
- current migratable relation schema;
- current migratable role-anchor schema.

### 4.3 Operator representation

Produce:

```text
q_continuous
q_relation_anchor
q_role_anchor
q_temporal_status
q_applicability
q_uncertainty
```

The operator representation remains continuous.

Relation and role anchors are inspectable basis states, not a complete symbolic program.

### 4.4 Multi-layer access

Do not hard-code layer 12 or final layer 16.

Preferred design principle:

- anchor/query probes attend over the complete hidden-state stack or a checkpoint-migratable layer bank;
- layer contribution is query/operator dependent;
- previous layerwise audit scores may initialize soft priors but are not permanent truth.

This preserves the learned lesson that `causes` and `supports` directionality appeared in different depth windows.

### 4.5 Relation anchor vocabulary

Current checkpoint relation IDs may seed the anchor bank.

The bank is migratable.

It must include a residual/unknown path so unseen relation semantics are not forced into the nearest existing relation.

No finite current relation vocabulary is a permanent ontology.

### 4.6 Argument-role anchors

Current seed roles should distinguish at least:

- SOURCE;
- TARGET;
- SYMMETRIC / relation-invariant;
- NONE / not-applicable.

These are current execution roles, not a claim that all future relational semantics reduce to four roles.

The role bank is migratable.

### 4.7 Output contract

The operator encoder must expose:
- anchor distributions/logits;
- continuous residual state;
- uncertainty/plurality;
- layer-attention diagnostics.

The continuous residual prevents the anchor codebook from becoming a personality or semantic ceiling.

---

## 5. Module B — Token-Level Adaptive Structural Binder

### 5.1 Starting evidence

The binding-identifiability audit already showed that token-level semantic interaction can identify the relevant edge perfectly on the current DEV diagnostic.

Do not relearn this from compressed graph states first.

### 5.2 Inputs

For each field/edge:
- query token states;
- field token states;
- edge relation state;
- source/target field identity;
- operator relation state;
- reliability/provenance if relevant to support.

### 5.3 Base semantic score

Use token-level late interaction as the base semantic compatibility signal.

The previous MaxSim-style diagnostic is a valid initialization because it demonstrated edge identity without gradient.

It is not a permanent exact scoring formula.

### 5.4 Relation compatibility

Support scoring may combine:
- query↔field token compatibility;
- operator↔edge-relation compatibility;
- pair/endpoint compatibility;
- evidence reliability.

The score used to select support must be the score that actually controls support.

No proxy router separate from runtime support is allowed.

### 5.5 Sparse support transform

The support transform must support variable cardinality.

Initial preferred family:
- `alpha-entmax` or another exact-zero differentiable sparse transform.

Reason:
- hard top-k permanently fixes support cardinality;
- dense softmax caused diffuse execution in the current model;
- adaptive exact-zero support allows one query to retain one edge and another to retain several.

The exact sparse transform remains a replaceable implementation choice.

If later evidence requires explicit connected-subgraph constraints, structured sparse selection such as SparseMAP may be evaluated.

Do not add that complexity preemptively.

### 5.6 Applicability is separate

Do not add a parent/no-op item into the same edge-selection simplex.

The earlier lineage repeatedly conflated:
- "is relational execution applicable?"
with
- "which edge is relevant?"

QSRE keeps these decisions separate.

The binder produces support conditional on relational interpretation.

The operator/fusion fabric carries applicability/reliability separately.

---

## 6. Module C — Query-Conditioned Node/Edge Relational Executor

### 6.1 Core representation

Use explicit node states and explicit edge states.

For supported edge `e = (u, r, v)`:

```text
edge_state_e =
    f(
      node_u,
      node_v,
      relation_r,
      q_operator,
      source_role_embedding,
      target_role_embedding,
      reliability/provenance/temporal metadata
    )
```

This makes source and target argument positions first-class.

### 6.2 Why node-edge states rather than endpoint scalar residuals

The executor must represent:

- source identity;
- target identity;
- relation meaning;
- requested role;
- relation order;
- multi-hop path state;
- plural support.

A scalar source residual and scalar target residual cannot carry that full state.

### 6.3 Query-conditioned update

Each executor layer should perform:

1. **edge update** from source node, target node, relation, and query operator;
2. **role-specific incident aggregation** back to nodes;
3. **node update** conditioned on query operator;
4. optionally another edge refresh from updated nodes.

The query operator must participate inside the learned MESSAGE/ATTENTION/UPDATE computation.

It may not be appended only to the final field scorer.

### 6.4 Source/target role channels

Messages arriving at a node through an edge must distinguish whether that node is:
- the edge source;
- the edge target.

Do not force those channels to be negatives of one another.

The architecture may share parameters, but source and target roles remain explicit inputs.

### 6.5 Multi-hop composition

Repeated node-edge updates allow an edge state to influence adjacent edges through shared nodes.

This provides a production path for:
- one-hop endpoint selection;
- multi-edge evidence aggregation;
- ordered multi-hop relation composition.

The architecture family does not change when hop count changes.

A finite training fixture hop count is not a capability ceiling.

### 6.6 Permutation property

No arbitrary edge-order positional embedding.

Graph/field ordering that has no semantic meaning must remain permutation equivariant/invariant.

---

## 7. Module D — Structural Readout

### 7.1 Candidate domain

The relational readout operates on the union of nodes/edges in active QSRE support.

Unrelated fields outside support do not participate in the relational candidate competition.

This closes the global-competition leak.

### 7.2 Readout state

Readout should consume:
- final node state;
- incident final edge states;
- query operator state;
- requested role state;
- support confidence.

### 7.3 Output

Expose:
- per-supported-node relational relevance;
- optional per-supported-edge/path relevance;
- normalized relational candidate distribution;
- relational summary latent;
- role-binding latent;
- support/path state;
- uncertainty/plurality.

A sparse or calibrated distribution may preserve co-valid alternatives.

Do not require one hard winner.

### 7.4 No parent residual composition

Forbidden:

```text
final_relational_score_i =
    parent_global_score_i
    + qsre_delta_i
```

QSRE has its own structural readout.

The parent remains a separate capability source.

---

## 8. Module E — Fallback and broader N0 composition

### 8.1 Do not recreate another router

QSRE should not produce a single scalar gate whose sole purpose is to blend its field distribution with the parent field distribution.

Instead QSRE returns:
- relational applicability/reliability;
- relational view tokens;
- relational summary;
- structural readout.

### 8.2 Exact non-relational pass-through

When relational applicability is absent:
- no relational support is active;
- base evidence tokens remain available;
- QSRE relational summary is unavailable/zero-marked;
- parent/general evidence path remains unchanged.

This gives preservation structurally rather than asking a new specialist to learn not to damage the parent.

### 8.3 Fusion integration

Preferred first integration:

- preserve base evidence tokens;
- contextualize only the supported evidence subset;
- emit one dedicated relational summary token/state;
- feed the result through the existing evidence/fusion pathway;
- preserve exact source anchors already supported by the ratified fusion architecture.

A future explicit fourth relational view is possible through checkpoint migration if causal evidence shows that combining base evidence and relational execution in one view causes interference.

Do not expand the fusion topology preemptively.

---

## 9. Initial parameter/training boundaries for a future implementation

This section describes isolation strategy only. It does not authorize training.

### Frozen initially

- semantic backbone;
- StructuredStateEncoder;
- EvidenceViewAdapter;
- proven parent graph weights used for base representations;
- ratified CrossContextFusion;
- latent pool.

### New QSRE trainable scope when later authorized

Potential new parameters:
- Query Operator Encoder anchor/query projections;
- adaptive support compatibility projections;
- sparse-support calibration parameters;
- node-edge relational executor;
- structural readout.

No existing parent parameter needs to change to test QSRE's core hypothesis.

This makes a future failure causally interpretable.

---

## 10. Future causal training sequence

Do not train the full pipeline at once.

The point of the clean-sheet redesign is to isolate capability boundaries before they become another repair chain.

### T1 — Executor with oracle operator + oracle support

Inputs:
- gold relation/operator;
- gold requested argument role;
- gold structural support.

Train only:
- relational executor;
- structural readout.

Question:

> Can the architecture execute role and relation composition when binding and semantic parsing are perfect?

Required before moving on:
- SOURCE/TARGET reversal;
- every relation family above floor;
- unrelated fields structurally excluded;
- ordered two-hop composition.

If T1 fails, do not touch support selection.

### T2 — Learned operator, oracle support

Freeze/hold a proven executor.

Train/evaluate:
- query operator extraction from frozen semantic states.

Question:

> Can natural language recover the relation/role operator required by a proven executor?

If T2 fails, the issue is operator semantics, not graph support.

### T3 — Learned adaptive support, oracle/proven operator

Train/evaluate:
- token-level structural binder;
- support cardinality.

Question:

> Can the model discover the correct zero/one/many support when execution competence is already proven?

No proxy routing score is allowed.

### T4 — End-to-end QSRE integration

Only after T1-T3 independently pass:
- predicted operator;
- predicted adaptive support;
- relational executor;
- structural readout.

Parent/general evidence path remains frozen.

### T5 — N0 fusion integration

Only after standalone QSRE capability is demonstrated:
- measure whether the existing fusion can consume the QSRE evidence stream without losing causal use;
- modify fusion only if that measurement establishes a fusion bottleneck.

This sequence prevents the project from blaming routing for an executor failure or blaming the executor for missing query semantics.

---

## 11. Future curriculum requirements

### Single-hop factorial cases

Factor:
- relation family;
- edge direction;
- requested SOURCE/TARGET role;
- endpoint lexical content;
- relation query paraphrase;
- distractor fields;
- distractor same-relation edges.

### Multi-support cases

Include:
- one decisive edge;
- two independent valid supports;
- conflicting supports;
- ambiguous co-valid supports;
- no relational support.

### Multi-hop cases

Include:
- ordered relation chains;
- relation-order reversal;
- shared endpoints;
- distractor paths;
- paths that cannot be solved by local endpoint preference.

### Split policy

TRAIN/DEV/TEST must separate:
- entity/subject pools;
- attribute/value pools;
- query templates;
- lexical relation framing.

Causal TEST remains closed until development gates are met.

---

## 12. Research alignment

QSRE is not copied from one paper.

It combines evidence-supported principles:

- **NBFNet:** query-conditioned relational computation and path representations;
- **LILaC:** late interaction for graph/subgraph relevance;
- **QSRAG / QR-GAT:** query + relation conditioning inside graph relevance computation;
- **SALR:** continuous latent reasoning anchored to explicit schema/operation structure without immediate hard textual commitment;
- **adaptive sparse attention work:** variable-cardinality exact-zero support rather than fixed top-k;
- **role/filler systematicity work:** explicit argument-role structure rather than hoping every relation-role conjunction is learned independently.

The combination is justified by A.L.I.C.E.'s own failures, not by novelty for its own sake.

---

## 13. Static implementation requirements

Before a source module may be committed, a separate decision must specify:

1. exact tensor interfaces;
2. exact current relation/role anchor seeds;
3. how hidden-state stacks are obtained/cached;
4. sparse support transform choice and fail-closed behavior;
5. node-edge executor update equations;
6. structural readout domain;
7. exact non-relational pass-through;
8. parent state reuse;
9. permutation tests;
10. D1-D5 mechanics tests;
11. parameter report;
12. no permanent field/edge/support/hop ceilings.

That decision may authorize **CPU/no-gradient mechanics implementation only**.

It must not automatically authorize optimizer, P100, TEST, challenge, or private identity gradient.

---

## 14. Current conclusion

The blueprint is now sufficiently concrete to prevent another local router/residual repair while preserving the useful work already completed.

The next legitimate step is a **static tensor-interface and mechanics decision**, followed by CPU/no-gradient construction only if that decision passes review.

Training remains closed.
