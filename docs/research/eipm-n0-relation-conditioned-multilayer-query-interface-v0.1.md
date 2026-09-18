# N0 Relation-Conditioned Multi-Layer Query Interface v0.1

**Date:** 2026-09-18  
**Status:** architecture designed from jobs 575801 and 575804; training remains unauthorized

## Evidence basis

### 575801 representation audit

Token-level late interaction preserved more directional relation signal than either pooled representation:

- raw mean pool role accuracy: 0.5625
- trained semantic projection role accuracy: 0.5625
- token late-interaction role accuracy: 0.6875

Token-level opposite-role pair distance was also materially larger.

### 575804 layerwise audit

Directional relation semantics were layer-local.

Observed examples:

- causes rose above chance through a broad middle-layer region and returned to chance in the final two layers;
- supports rose above chance only in the narrow layer-12/13 region;
- final-layer aggregate token accuracy remained competitive while specific directional capabilities disappeared.

Therefore:

1. a single final pooled query is insufficient;
2. a single universal token layer is not justified;
3. the semantic backbone already contains useful directional information;
4. the correct next architecture should expose relation-conditioned access to intermediate token layers before evidence-graph field selection.

## Design goals

The interface must:

- keep the semantic backbone frozen for the first causal study;
- consume token-level hidden states, not only a pooled vector;
- let relation type condition both token selection and layer selection;
- derive relation-specific candidate layers from the frozen 575804 audit;
- treat audit scores as priors, not permanent routing truth;
- avoid hardcoding layer 12 as a universal answer;
- safely defer for symmetric or unmapped relations;
- remain independent from private identity data;
- preserve the no-ceiling doctrine.

## Derived layer map

Compiler:

`scripts/eipm/n0/compile_n0_v02_relation_conditioned_layer_map_v0_1.py`

Source:

`query-semantics-layerwise-audit-v0.1/query_semantics_layerwise_audit.json`

The compiler emits, per relation:

- best observed token-role accuracy;
- all exact-best layers;
- all layers within one audit observation quantum of best;
- a compact candidate layer set chosen to retain both high score and depth diversity;
- full ranked layer evidence.

The candidate set is an operating prior for the first causal study, not a permanent architecture ceiling.

## Interface

Source:

`src/alice_personality/n0/relation_conditioned_multilayer_query.py`

Class:

`RelationConditionedMultiLayerQueryInterface`

For every directed graph edge:

1. relation type selects the audit-derived candidate layer mask;
2. a relation embedding attends over query tokens inside each candidate layer;
3. every candidate layer produces a relation-conditioned token summary;
4. a second relation-conditioned mixer scores those layer summaries;
5. audit accuracy contributes only an initialization prior;
6. the weighted result becomes an edge-specific semantic query state in graph space.

Conceptually:

```text
frozen semantic hidden states
  layer 0 ... layer 16
          |
          +---- relation-specific candidate mask
          |
          v
relation-conditioned token attention
   inside each candidate layer
          |
          v
relation-conditioned multi-layer mixing
          |
          v
edge-specific semantic query state
          |
          v
future evidence-graph field selection
```

## Why this differs from the failed QRR path

The failed QRR architecture still reduced the query to one semantic stream before role routing.

This interface instead moves relation-conditioned language interaction *before* query reduction.

It therefore tests a different causal hypothesis:

> the required semantic role is present in intermediate token representations but is lost or blurred when the query is collapsed too early.

## Symmetric / unsupported relations

`conflicts_with` receives no directional interface because its endpoint role is symmetric for this capability.

PAD and unmapped relations return no active interface state.

The architecture therefore fails closed rather than inventing source/target semantics.

## Training status

No training is authorized by this design document.

Before any gradient:

1. compile the exact relation-layer map from the saved 575804 result;
2. inspect the map for pathological one-layer hardcoding or unsupported relations;
3. run static shape/contract validation of the interface;
4. define one fresh causal curriculum and one preservation contract;
5. only then decide whether a bounded interface-training experiment is justified.

## Anti-loop boundary

This design is not QRR v0.4.

It is a new query→graph interface derived from architecture-level evidence.

If the eventual interface experiment fails, do not patch it locally. Reopen the larger semantic/graph boundary and compare against the possibility that some relation semantics belong in a higher reasoning layer rather than evidence selection.
