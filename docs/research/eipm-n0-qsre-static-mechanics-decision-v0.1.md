# QSRE Static Tensor Interface and Mechanics Decision v0.1

**Date:** 2026-09-20  
**Status:** authorizes CPU/no-gradient reference mechanics only  
**Architecture family:** QSRE  
**Workload authority:** `docs/research/eipm-full-eipm-workload-envelope-v0.1.md`

## 1. Decision

The deterministic D1-D5 contract established that the clean-sheet factorization is expressive enough to avoid the old global-softmax-residual trap.

This decision now pins the exact **interfaces and mechanics** that a future learned implementation must respect.

It authorizes only:
- pure reference mechanics;
- shape/interface validation;
- exact-zero sparse-support reference behavior;
- source/target incidence mechanics;
- support-local readout mechanics;
- fallback/defer mechanics;
- ordered path mechanics;
- CPU/no-gradient CI.

It does **not** authorize a learned QSRE module.

---

## 2. Dynamic tensor axes

The contract uses symbolic dynamic axes.

- `B`: batch
- `L`: available semantic hidden-state layers/views
- `Tq`: query token count
- `F`: evidence field count
- `Tf`: field token count
- `E`: directed edge count
- `R`: currently instantiated relation-anchor vocabulary
- `K`: currently instantiated argument-role anchor vocabulary
- `C`: candidate count for later judgment consumers

None is a permanent product ceiling.

Current checkpoint widths such as semantic 640 or graph 512 are reusable checkpoint shapes, not architecture limits.

---

## 3. Query semantic input

```text
query_hidden_states : [B, L, Tq, Ds]
query_token_mask    : [B, Tq]
```

The interface must preserve access to multiple semantic layers.

A future Query Operator Encoder may produce query-dependent layer weights:

```text
operator_layer_weight : [B, L]
```

No universal fixed semantic layer is part of the contract.

---

## 4. Evidence inputs

### Field-level inputs

```text
field_token_states : [B, F, Tf, Ds]
field_token_mask   : [B, F, Tf]
field_base_state   : [B, F, Dg]
field_struct_state : [B, F, Dst]
field_metadata     : [B, F, Dm]
field_valid_mask   : [B, F]
```

Metadata may represent provenance, temporal scope, confidence, missingness, actor/relationship role, or later migratable typed context.

### Edge-level inputs

```text
edge_index          : [B, E, 2]
edge_relation_state : [B, E, Dr]
edge_metadata       : [B, E, Dem]
edge_valid_mask     : [B, E]
```

The two endpoint positions are semantically ordered:

```text
edge_index[..., 0] = SOURCE position
edge_index[..., 1] = TARGET position
```

This ordering is an explicit relational variable. It is not inferred from arbitrary field order.

---

## 5. Query Operator State

Future learned implementation must expose:

```text
q_continuous       : [B, Dq]
q_relation_anchor  : [B, R]
q_role_anchor      : [B, K]
q_temporal_status  : [B, Dt]
q_applicability    : [B]
q_uncertainty      : [B, Du]
q_layer_weight     : [B, L]
```

Requirements:

- relation anchors remain migratable;
- role anchors remain migratable;
- unknown/new relation meaning has a residual continuous path;
- plurality is permitted;
- `q_applicability` is separate from edge support;
- no anchor distribution is required to be one-hot.

The operator state is a structured continuous latent, not a fixed symbolic logical form.

---

## 6. Adaptive Structural Support

QSRE has two related support surfaces:

```text
field_support_logit : [B, F]
edge_support_logit  : [B, E]
```

and corresponding support weights:

```text
field_support_weight : [B, F]
edge_support_weight  : [B, E]
```

### Reference sparse projection

The CPU mechanics reference uses **masked sparsemax** because it has:

- exact zeros;
- adaptive support size;
- no fixed top-k.

The future learned implementation may use another exact-zero or structured sparse projection if evidence justifies it.

Permanent hard top-k remains forbidden.

### Runtime relation

Edge support is not merely an explanation score.

The actual relational executor must consume `edge_support_weight`.

Any learned score supervised for support must be the score that controls runtime support.

This closes the previous proxy-router/runtime mismatch.

### Field support

Field support may arise from:

- direct query↔field evidence;
- incident supported edges;
- later unary operators.

The mechanics contract therefore keeps `field_support_weight` first-class rather than deriving all support exclusively from edges.

---

## 7. Explicit source/target incidence

For directed edge `e=(u,v)`, construct role-specific incidence:

```text
I_source[b, f, e] = 1 if f == u else 0
I_target[b, f, e] = 1 if f == v else 0
```

masked by valid edge/field status.

A future executor may share parameters across the two channels, but it must not erase which argument position a node occupies.

Do not encode TARGET as the negative of SOURCE.

---

## 8. Future learned executor equations

This section fixes the architecture mechanics, not parameter values.

### Initial node state

A future learned implementation may construct:

```text
n_i^0 = NodeInit(
    field_base_state_i,
    field_struct_state_i,
    pooled_or_attended_field_token_state_i,
    field_metadata_i
)
```

The exact projection can migrate.

### Edge update

For supported edge `e=(u,r,v)` at executor layer `l`:

```text
z_e^l = EdgeUpdate_l(
    n_u^l,
    n_v^l,
    relation_state_e,
    edge_metadata_e,
    q_continuous,
    q_relation_anchor,
    q_role_anchor,
    SOURCE_position,
    TARGET_position
)
```

Query/operator state is inside the update.

### Role-specific messages

```text
m_e_to_source^l =
    edge_support_weight_e
    * SourceMessage_l(z_e^l, q_operator)

m_e_to_target^l =
    edge_support_weight_e
    * TargetMessage_l(z_e^l, q_operator)
```

Source and target are explicit channels.

### Node aggregation

```text
a_i^l =
    Sum_{e: source(e)=i} m_e_to_source^l
    +
    Sum_{e: target(e)=i} m_e_to_target^l
```

with reliability/provenance modulation when behaviorally relevant.

### Node update

```text
n_i^(l+1) = NodeUpdate_l(
    n_i^l,
    a_i^l,
    q_operator,
    field_metadata_i
)
```

Optional repeated edge refresh from updated node states is allowed.

### Multi-hop

Repeated node↔edge updates carry information along connected supported structure.

No separate architecture family is introduced when hop count grows.

---

## 9. Structural Readout

The relational candidate domain is the supported field set.

Define a support-local mask:

```text
M_rel[b,f] =
    field_valid_mask[b,f]
    AND (
        field_support_weight[b,f] > 0
        OR f incident to an edge with edge_support_weight > 0
    )
```

Future learned readout:

```text
relational_logit_i =
    Readout(
      n_i^K,
      incident_edge_states_i,
      q_operator,
      field_metadata_i
    )
```

then normalize **only on `M_rel`**.

Forbidden:

```text
relational_logit_i = parent_global_logit_i + qsre_delta_i
```

for the universal relational path.

Fields outside `M_rel` receive zero relational probability by construction.

---

## 10. Applicability, fallback, and defer

Applicability is independent of support identity.

Reference mechanics defines three control states:

- `RELATIONAL`: applicability high and valid support exists;
- `FALLBACK`: relational applicability absent or support empty;
- `DEFER`: applicability genuinely unresolved.

A future learned system may calibrate these boundaries, but it must preserve the state distinction.

### Exact non-relational pass-through

When `FALLBACK`:

- QSRE does not modify the parent/global evidence distribution;
- no relational result is asserted;
- base evidence tokens remain intact;
- QSRE emits not-applicable state.

This makes preservation structural rather than an auxiliary "do no harm" loss.

---

## 11. Output contract

QSRE returns a **relational view**, not only a winner:

```text
supported_field_state
supported_edge_state
field_support_weight
edge_support_weight
relational_candidate_distribution
relational_summary_state
role_binding_state
path_or_subgraph_state
applicability
uncertainty_plurality
provenance_reliability_state
```

The eventual N0 fusion/latent fabric consumes this view.

This supports the full EIPM workload envelope:
- one or many relevant evidence items;
- conflicting evidence;
- relationship state;
- temporal change;
- later identity concepts;
- multi-candidate judgment;
- voice/emotional context as additional views.

---

## 12. CPU/no-gradient mechanics implementation

The reference implementation may now test:

1. masked sparsemax;
2. variable support cardinality;
3. support-local field masks;
4. explicit source/target incidence;
5. role reversal;
6. path order;
7. parent-global distractor isolation;
8. fallback/defer;
9. dynamic large F/E shapes;
10. permutation invariance/equivariance of mechanics.

Reference mechanics may not contain:
- `nn.Parameter`;
- optimizer;
- backward;
- learned QSRE projections;
- TEST/challenge access;
- private identity data.

---

## 13. Speed rule

Engineering work may be batched:

- workload contract;
- interface decision;
- reference mechanics;
- mechanics fixtures;
- static CI;
- result interpretation

may be completed in one development batch.

Scientific causality remains ordered.

A future learned build must still isolate:
- T1 executor competence;
- T2 operator learning;
- T3 support learning;
before T4 end-to-end integration.

The workflow is faster by batching engineering, not by merging causal questions.
