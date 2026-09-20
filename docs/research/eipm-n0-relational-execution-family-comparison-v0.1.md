# N0 Relational Execution Architecture Family Comparison v0.1

**Date:** 2026-09-20  
**Status:** research decision; no implementation or gradient authorized  
**Depends on:** `docs/research/eipm-n0-clean-sheet-relational-execution-audit-v0.1.md`  
**Proof contract:** `configs/eipm/n0/n0_v02_relational_execution_proof_obligations_v0_1.json`

## Decision summary

Advance one architecture family for interface design:

> **F4 — a constrained hybrid: token-level semantic binding + continuous relation/role operator state + adaptive sparse structural support + query-conditioned relational execution + structural readout.**

Working name:

> **QSRE — Query-Conditioned Sparse Relational Executor**

This is an architecture-family decision, not an implementation authorization.

The reason to advance the hybrid is not that it contains the most mechanisms. It is the smallest family in the comparison that directly addresses all three failure boundaries established by the evidence:

1. semantic binding must remain token-level and edge-specific;
2. relation/argument role must be first-class rather than an implicit scalar preference;
3. relational execution must occur on claimed structural support rather than as residual corrections to an unrelated global field softmax.

The hybrid must remain modular and minimal. Any mechanism without a direct evidence-based role is rejected.

---

## 1. Evaluation criteria

The comparison uses proof obligations P0-P16 and four additional engineering criteria:

- **E1 — reuse:** preserve proven N0 components where their competence remains supported;
- **E2 — falsifiability:** failures should localize to binding, operator state, execution, or fallback rather than an opaque monolith;
- **E3 — extensibility:** zero/one/many support and multi-hop behavior must not require a new architecture family;
- **E4 — compute discipline:** the design must not require broad retraining of the 136M semantic backbone to test a relational hypothesis.

---

## 2. F1 — Query-conditioned sparse relational GNN

### Shape

1. token/query state produces support scores;
2. adaptive sparse selection defines a subgraph;
3. query relation/operator conditions graph messages;
4. message passing executes on the subgraph;
5. structural readout produces the relational result.

### Strengths

**P1 / P3:** excellent fit.

The selected subgraph is the execution domain. Unselected unrelated fields need not remain in the answer competition.

**P4 / P6:** strong.

Sparse graph support can naturally contain one edge, several edges, or a path. Query-conditioned message passing can compose relations over multiple steps.

**P8 / P9 / P10:** strong.

Relation type, endpoint content, and topology are explicit inputs. Graph permutation properties can be tested directly.

**E2:** strong.

Binding, message passing, and readout are separable.

### Weaknesses

**P2 / P7:** incomplete by itself.

A conventional query-conditioned GNN still does not guarantee a systematic representation of requested argument role. If SOURCE/TARGET remains implicit inside a query embedding, the project may recreate the same family-specific directional failure.

**P5:** depends on support mechanism.

A hard top-k graph selector can prematurely collapse ambiguity.

**P0:** must be carefully scoped.

A pure KG reasoning design could narrow the role of the N0 evidence view if it is allowed to replace broader semantic and structured context.

### Verdict

**Advance as the execution core, not as the whole solution.**

It needs an explicit continuous operator/argument-role state and an ambiguity-preserving support mechanism.

---

## 3. F2 — Schema-anchored continuous relational executor

### Shape

1. derive a continuous query state;
2. align it to a migratable relation/role schema/code space;
3. feed selected/weighted schema anchors back into continuous reasoning;
4. delay final discrete commitment;
5. emit a relational result.

### Strengths

**P2 / P7:** strongest family for the role-binding problem.

Relation and argument roles can be supervised as explicit recoverable structure without reducing the entire query to a hard symbolic parse.

**P5:** strong.

Several schema hypotheses can remain active during latent computation.

**P13:** strong.

The code/anchor space gives a clear probe target for whether relation and role information are present.

**E2:** strong.

Operator-state errors can be distinguished from graph execution errors.

### Weaknesses

**P3 / P6:** insufficient alone.

Schema anchoring does not itself define how evidence graph structure is executed. Without a dedicated graph executor it can still end in a generic score head or decoder.

**P9:** endpoint-content sensitivity requires a separate mechanism.

**P0:** risk of over-formalization.

A fixed logical-form or closed ontology interpretation would violate the A.L.I.C.E. open-ontology/capability doctrine.

### Verdict

**Use as the query/operator representation layer, not as the whole executor.**

The transferable idea is continuous state anchored to explicit relation/role structure, not semantic parsing as the N0 product.

---

## 4. F3 — Relational set/graph transformer

### Shape

Represent:
- query/operator tokens;
- field tokens;
- relation tokens;
- source-role and target-role tokens;
- metadata tokens;

then perform joint relation-aware attention and read out structured result tokens.

### Strengths

**P0 / P4 / P5:** potentially excellent.

It is flexible enough for heterogeneous evidence and multiple supports.

**P7-P9:** can make relation and endpoint roles explicit.

**E3:** potentially strong.

Multi-edge and multi-hop reasoning can be represented without a separate architecture family.

### Weaknesses

The flexibility is also the main risk.

A generic transformer can silently relearn:
- relation shortcuts;
- global endpoint preference;
- dense attention to irrelevant fields;
- implicit topology.

That would make it difficult to prove that the architecture repaired the current failure rather than hiding it.

**P3:** not guaranteed.

Unless attention support is structurally sparse, unrelated fields can remain active everywhere.

**P6:** not guaranteed.

Graph path composition must be represented explicitly enough that relation order/topology cannot be ignored.

**E2:** weaker than F1/F2.

A single large joint-attention block makes localization harder.

### Verdict

**Do not advance as the first clean-sheet implementation family.**

Its useful mechanisms may be used locally, but the first redesign should be more factorized and causally inspectable.

---

## 5. F4 — Constrained hybrid / QSRE

### Components

QSRE combines exactly four evidence-supported ideas.

#### Component A — Continuous Query Operator State

Source:
- frozen semantic token states;
- typed query context.

It carries:
- semantic intent;
- relation/operator hypotheses;
- requested argument role;
- temporal/status intent;
- uncertainty/plurality;
- applicability signal.

Relation/role anchors are explicit and migratable. The full state remains continuous.

This addresses:
- the QRR global-role collapse;
- family-specific SOURCE/TARGET failures;
- the layer-local relation semantics discovered earlier.

#### Component B — Token-Level Adaptive Structural Support

Fine-grained query-token ↔ field/edge-token interaction computes semantic support.

Requirements:
- use token information rather than only pooled field states;
- retain endpoint-content sensitivity;
- variable support cardinality;
- exact-zero or structurally excluded irrelevant support is allowed;
- no permanent fixed top-k;
- support may remain plural/ambiguous.

This reuses the strongest result from the binding-identifiability audit rather than relearning edge identity with another router.

This addresses:
- pooled/structured binding compression;
- soft diffuse execution observed in the hard-binding counterfactual.

#### Component C — Query-Conditioned Relational Executor

Run only on active structural support.

Each execution layer receives:
- query operator state;
- relation state;
- source state;
- target state;
- explicit argument-position/role state;
- confidence/provenance/temporal metadata as appropriate.

The query/operator state conditions actual message/attention/update computation.

No architecture path may reduce this stage to one scalar source/target residual.

This addresses:
- the repeated role-composition failures;
- the old query-independent/global execution surface;
- multi-hop relation composition.

#### Component D — Structural Readout + Separate Fallback

Read relational results from executor states.

The readout may return:
- one endpoint;
- multiple endpoints;
- path/subgraph state;
- uncertainty/plurality;
- not-applicable/defer.

General-parent fallback is evaluated separately.

The old parent may provide:
- starting evidence representations;
- general non-relational evidence behavior;
- a prior;
- a preservation teacher;
- another fusion view.

But its global field softmax does not automatically override a validly claimed relational execution.

This addresses:
- the global-competition leak demonstrated by job 575913;
- repeated confusion between edge routing and parent/specialist applicability.

---

## 6. Proof-obligation comparison

| Obligation | F1 sparse GNN | F2 schema latent | F3 graph transformer | F4 QSRE |
|---|---|---|---|---|
| P0 N0 breadth | pass if scoped | pass if non-symbolic | pass | pass by modular scope |
| P1 oracle support closure | strong | weak alone | possible | strong |
| P2 oracle role closure | medium | strong | possible | strong |
| P3 no global leak | strong | weak alone | medium | strong |
| P4 adaptive support | strong | medium | strong | strong |
| P5 ambiguity | support-dependent | strong | strong | strong |
| P6 multi-hop | strong | weak alone | possible | strong |
| P7 role intervention | medium | strong | possible | strong |
| P8 relation intervention | strong | strong | possible | strong |
| P9 endpoint intervention | strong | weak alone | strong | strong |
| P10 permutation | strong | n/a/medium | design-dependent | strong |
| P11 provenance causality | strong | medium | strong | strong |
| P12 fallback isolation | medium | medium | medium | explicit |
| P13 input sufficiency probes | strong | strong | harder | strong |
| P14 fresh generalization | data property | data property | data property | data property |
| P15 identity neutral | pass | pass | pass | pass |
| P16 no capability ceiling | pass | pass | pass | pass |

The table is an architecture-contract assessment, not empirical model performance.

---

## 7. Why QSRE is not “architecture #5” in the old sense

The old sequence generally preserved this shape:

```text
parent field scoring
      +
new query-conditioned correction
      ↓
global field decision
```

QSRE changes the computation boundary:

```text
query semantics
      ↓
continuous relation/role operator state
      ↓
adaptive semantic binding
      ↓
claimed sparse support
      ↓
query-conditioned relational execution ON SUPPORT
      ↓
structural relational readout
      ↓
separate applicability/fallback composition
      ↓
N0 fusion/latent fabric
```

The critical change is not a new router.

It is that binding defines the relational execution domain and query/relation/role information participates inside the executor.

---

## 8. Proven components and their proposed roles

### AliceN0V02 semantic backbone

**Keep.**

Role:
- query token states;
- field text token states;
- semantic operator substrate.

No current evidence justifies semantic retraining.

### StructuredStateEncoder

**Keep.**

Role:
- typed field metadata;
- provenance;
- temporal scope;
- confidence;
- missingness;
- structured field representation.

Its pooled readout is not the only executor input.

### EvidenceViewAdapter

**Keep unless later oracle diagnostics show its field representation destroys necessary endpoint information.**

Current evidence does not show that.

### DualEndpointEvidenceGraphEncoder

**Split its roles.**

Potentially keep:
- learned endpoint/evidence node representations;
- relation embeddings;
- useful message-passing parameters;
- general non-relational/explicit-role fallback.

Do not assume continued authority for:
- `field_weights` as the universal relational answer distribution;
- its global pooling surface as final relational execution.

### Ratified CrossContextFusion

**Keep.**

QSRE becomes an evidence/relational view producer compatible with fusion.

### Adaptive latent pool

**Do not modify from this audit.**

Its status remains governed by its own lineage.

---

## 9. Minimality constraints on QSRE

The family is rejected if implementation planning introduces unnecessary mechanisms.

Specifically:

- no generic MoE expert bank;
- no LLM-style decoder inside N0;
- no textual chain-of-thought requirement;
- no fixed symbolic logical-form language;
- no permanent top-k;
- no giant new semantic encoder;
- no duplicate graph encoder unless reuse proves impossible;
- no dense all-field relational softmax after sparse support has already been claimed;
- no scalar residual as the primary relational execution output.

A mechanism must exist because one of P0-P16 requires it.

---

## 10. Pre-implementation diagnostics required

Before code for QSRE is authorized, define and run **deterministic/oracle contracts** for the following.

### D1 — Oracle support / role factorization

Create examples where:
- gold support is supplied;
- query role alternates SOURCE/TARGET;
- unrelated fields are present.

Required finding:
once the support is fixed, unrelated fields must be irrelevant by construction, and the intended execution contract must have a direct role-conditioned output.

This validates the proposed factorization itself.

### D2 — One vs many support

Create:
- one-edge decisive queries;
- two independent evidence-support queries;
- ambiguous two-support queries;
- no-relational-support queries.

The architecture interface must represent all four without changing output schema.

### D3 — Ordered two-hop composition

Create relation chains where:
- A→B→C and A→D→C differ;
- reversing relation order changes the answer;
- local endpoint heuristics cannot solve the item.

The executor interface must preserve ordered path state.

### D4 — Parent isolation

For relational examples:
- parent global field preference may deliberately favor a distractor;
- oracle relational support remains correct.

The contract must still return the relational result.

For non-relational examples:
- relational support is empty;
- parent fallback remains unchanged.

### D5 — Continuous ambiguity

Create examples with:
- two plausible relation/operator interpretations;
- insufficient evidence to collapse one.

The query operator/support interface must preserve plurality/uncertainty rather than manufacture confidence.

No gradient is required for D1-D5.

---

## 11. Decision

**Advance F4/QSRE to interface and deterministic diagnostic design.**

Do **not** yet:
- implement trainable QSRE modules;
- create a trainer;
- create a GPU launcher;
- train;
- open TEST/challenge;
- alter the semantic backbone;
- retrain the parent graph.

Implementation becomes eligible only after D1-D5 have concrete interfaces and expected invariants, and a separate decision confirms that the design has not reintroduced the global-softmax residual pattern.
