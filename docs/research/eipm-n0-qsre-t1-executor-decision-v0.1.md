# QSRE T1 Executor Decision v0.1

**Date:** 2026-09-20  
**Status:** authorizes trainable source definition and CPU/no-gradient forward qualification only  
**Causal stage:** T1 — executor/readout competence with oracle operator + oracle support

## 1. Why T1 exists

The previous architecture family repeatedly mixed several unanswered questions:

- did the query encode the right relation/role?
- did the model bind the right evidence?
- did the executor know how to operate on that evidence?
- did fallback/no-op routing interfere?

QSRE explicitly separates those questions.

T1 isolates the executor.

The upstream semantic interpretation and support selection are treated as perfect oracle inputs.

If T1 cannot learn the required structural operations, no router, binder, or semantic-query change is allowed to compensate for that failure.

## 2. T1 is not a SOURCE/TARGET toy

A trivial endpoint selector would not validate the full EIPM workload.

T1 must be architecturally capable of:

- SOURCE/TARGET endpoint role;
- relation-filtered execution;
- ordered multi-hop paths;
- one/many supported edges;
- plural/co-valid target distributions;
- reliability/provenance-sensitive choice;
- temporal-latest choice;
- fallback/defer control;
- fields outside support being unable to win.

These are public N0 cognitive operations. They do not encode private Elaina identity.

## 3. Oracle operator input

T1 receives typed oracle operator tensors.

Current seed interface:

```text
relation_sequence_id   [B,S]
relation_sequence_mask [B,S]
role_id                [B]
operation_id           [B]
focus_field_weight     [B,F]
operator_context       [B,Dc]
applicability          [B]
```

Current operation seed vocabulary:

1. ROLE_SELECT
2. PATH_FOLLOW
3. AGGREGATE
4. PREFER_RELIABILITY
5. PREFER_LATEST

This vocabulary is a first public training seed, not a permanent ontology.

The operator interface retains `operator_context` so future operations and nuance do not require every semantic distinction to become a discrete ID.

T2 will later learn this operator state from natural language. T1 must not attempt that.

## 4. Oracle support input

T1 receives:

```text
field_support_weight [B,F]
edge_support_weight  [B,E]
```

These weights are supplied by the curriculum/fixture.

No learned support selector exists in T1.

The executor consumes these exact weights.

T3 will later learn support discovery.

## 5. Evidence representation input

T1 accepts:

```text
field_state     [B,F,Din]
field_metadata  [B,F,Dfm]
field_valid     [B,F]

edge_index       [B,E,2]
edge_relation_id [B,E]
edge_metadata    [B,E,Dem]
edge_valid       [B,E]
```

During eventual real T1 training, `field_state` should come from frozen, already-qualified evidence representations.

The T1 module does not receive parent global field logits.

This is a hard architectural boundary.

## 6. Trainable architecture

### Node initialization

```text
n_i^0 =
  NodeProjection(field_state_i)
  + MetadataProjection(field_metadata_i)
  + focus_weight_i * FocusEmbedding
```

The focus term marks an oracle query anchor/subject when the operation needs one.

### Operator state

At each relation step `s`:

```text
q_s =
  RelationEmbedding(relation_sequence_id_s)
  + RoleEmbedding(role_id)
  + OperationEmbedding(operation_id)
  + ContextProjection(operator_context)
```

The relation sequence preserves order.

### Supported edge update

For edge `e=(u,r,v)`:

```text
z_e^s =
  EdgeMLP([
    n_u^s,
    n_v^s,
    RelationEmbedding(r),
    EdgeMetadataProjection(meta_e),
    q_s,
    SourcePositionEmbedding,
    TargetPositionEmbedding
  ])
```

The current oracle relation step supplies a structural relation gate:

```text
g_e^s =
  edge_valid_e
  * edge_support_weight_e
  * 1[r_e == requested_relation_s]
  * step_active_s
```

Exact relation identity is permissible here because T1 receives an oracle operator. Natural-language relation inference is T2.

### Role-specific messages

```text
m_source = g_e^s * SourceMessage([z_e^s, q_s])
m_target = g_e^s * TargetMessage([z_e^s, q_s])
```

Source and target use explicit channels.

They are not forced antisymmetric.

### Node update

Messages are scatter-added to the corresponding source/target nodes.

```text
a_i^s =
  sum source messages incident as SOURCE
  + sum target messages incident as TARGET

n_i^(s+1) = NodeGRU(
  [a_i^s, q_s],
  n_i^s
)
```

The graph update is permutation equivariant.

### Structural readout

Readout consumes the final supported node states plus the full operator summary.

The relational softmax domain is only the support-local field mask.

Outside-support fields receive zero relational probability by construction.

## 7. Multi-hop mechanics

The executor iterates over `relation_sequence_id`.

Therefore the same architecture supports:

- one relation step;
- two-hop paths;
- longer future paths.

Current curriculum lengths are optimization shapes, not product hop ceilings.

## 8. Reliability and temporal metadata

T1 edge metadata reserves explicit channels for:

- reliability/confidence;
- normalized temporal order or recency;
- future typed metadata through dimension migration.

The executor must learn whether those channels matter based on the oracle operation state.

This avoids hard-coding "always newest" or "always most reliable."

## 9. Control state

T1 reuses the separate mechanics:

- RELATIONAL
- FALLBACK
- DEFER

When a batch item is not RELATIONAL, its QSRE relational probability is zero.

The module does not synthesize the parent fallback answer.

That remains outside QSRE.

## 10. Forward qualification before any gradient

Randomly initialized T1 must pass mechanics invariants:

- no parent-global-logit input exists;
- output probability is exactly zero outside support;
- probability sums to one for RELATIONAL rows with support;
- FALLBACK/DEFER rows produce no asserted relational distribution;
- edge order permutation does not change results;
- field permutation gives the corresponding permuted result;
- relation sequence order can change the execution state;
- variable F/E/S shapes work;
- parameter scope belongs only to T1;
- no semantic/evidence parent parameters are present.

This qualification tests mechanics, not accuracy.

## 11. T1 curriculum contract

The public curriculum will contain factorial families for:

- endpoint role;
- relation filtering;
- multi-support aggregation;
- ordered path following;
- reliability arbitration;
- temporal arbitration;
- ambiguity/plurality;
- fallback/defer;
- high outside-support distractors.

TRAIN/DEV splitting must vary graph topology, node IDs, relation sequences, metadata values, and operation combinations.

No private data.

## 12. Boundary

Authorized by this decision:

- source implementation containing T1 parameters;
- curriculum builder/evaluator implementation;
- CPU/no-gradient random-forward qualification;
- static CI.

Not authorized:

- optimizer;
- backward;
- P100;
- T1 training;
- learned query operator;
- learned support selection;
- TEST/challenge;
- private identity gradient.

A separate decision after forward and curriculum qualification is required before one T1 training run.
