# N0 Query–Relation Role Router Handoff

**Date:** 2026-09-18  
**Status:** role-residual v0.2 failed at dev without heldout exposure; external architecture research completed; query–relation role router v0.3 is CI-ready for one P100 run

## Current frontier

`alice-eipm-v1-query-relation-role-router @ 2790c1e3cd4183377edae8805752f9cc0b7a1915`

Stable build:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

Frontier is 70 ahead / 0 behind stable.

## Trigger failure

Magnolia job `575798` completed with exit `0:0`.

Result:

`FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_DEV_STOP_NO_HELDOUT_EXPOSURE`

Result SHA-256:

`d9abbd6d3da3030361201a8a32f84aa8bf458fb5bebcc07db241a217b3557d8e`

Key facts:

- preservation candidates: `step-00000040` only;
- eligible candidates: none;
- semantic heldout test was not evaluated;
- heldout rows remained unopened;
- parent endpoint graph parameters stayed exactly unchanged;
- scale remained unauthorized.

This is a valid dev failure, not an infrastructure failure.

## Why v0.2 architecture was insufficient

The v0.2 residual isolated parameters but still consumed the endpoint parent's already-specialized projected query:

`pool_query + query_projection(query_semantic) -> normalize`

The raw frozen semantic query was available but did not directly feed the residual.

Therefore the side module was isolated in weights but not in information path.

It also composed by additive endpoint bias, forcing a new semantic skill to numerically fight the parent's learned logits.

## External architecture research

Research checkpoint:

`docs/research/eipm-n0-query-relation-role-router-external-research-v0.1.md`

Reviewed and applied lessons from:

- Progressive Neural Networks;
- Side-Tuning;
- Ladder Side-Tuning;
- AdapterFusion;
- Neural Bellman-Ford Networks;
- Heterogeneous Graph Transformer;
- Hypernetwork Knowledge Graph Embeddings;
- gradient-interference / orthogonal continual-learning work.

Resulting architecture decision:

1. keep the proven endpoint graph immutable;
2. tap raw frozen semantic query in a side path;
3. condition role routing jointly on raw query + relation type;
4. predict explicit `SOURCE / TARGET / DEFER_TO_PARENT`;
5. compose at probability level rather than additive-logit fighting;
6. supervise the semantic role abstraction directly;
7. if this design fails dev, require architecture-level audit before any further gradient run.

## New architecture

Source:

`src/alice_personality/n0/evidence_graph_query_relation_role_router.py`

Class:

`QueryRelationRoleRouterEvidenceGraphEncoder`

The frozen parent remains `DualEndpointEvidenceGraphEncoder`.

New side modules:

- `semantic_query_side`
- `semantic_relation_side`
- `semantic_relation_norm`
- `semantic_role_router`

Router inputs:

- raw `query_semantic`;
- edge relation type.

Router classes per active edge:

- `SOURCE`
- `TARGET`
- `DEFER_TO_PARENT`

Composition:

`final_weights = (1 - route_probability) * parent_weights + route_probability * semantic_router_weights`

The initial router is strongly biased to DEFER, preserving parent behavior before learning.

The router does not use the parent's `query_projection` as its semantic input.

## Training objective

Natural semantic rows:

- direct router CE for SOURCE/TARGET;
- field-selection consistency CE;
- relation-flip consistency.

Preservation replay:

- DEFER supervision;
- final-distribution KL to frozen parent.

Parent graph tensors remain frozen and are verified unchanged after optimization steps/checkpoints.

## Fresh v0.3 data

Builder:

`scripts/eipm/n0/build_n0_v02_query_relation_role_router_curriculum_v0_3.py`

12 relation-semantic families.

Per family:

- 30 train pairs;
- 8 dev pairs;
- 8 unopened heldout test pairs.

Train/dev/test use disjoint:

- query paraphrase banks;
- subject pools;
- attribute/value pools.

No reuse of:

- frozen challenge rows;
- job-575795 localization rows;
- relation-semantic v0.1 rows;
- role-residual v0.2 rows;
- endpoint-repair v0.2 heldout rows.

## Heldout / anti-loop policy

Heldout opens only if:

1. endpoint parent behavior is preserved;
2. ordinary replay is preserved;
3. direct semantic-role dev readiness passes;
4. semantic field-selection dev readiness passes.

If no checkpoint passes:

`FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE`

and next action is:

`architecture_level_audit_required_before_any_further_gradient_run`

No automatic v0.4 repair is authorized from that failure.

## State

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.13.json`

## CI

Workflow:

`N0 Query Relation Role Router Contract Check`

Run:

`35387454354`

Result:

`SUCCESS`

Passed:

- syntax;
- external-research gate;
- raw-query side path contract;
- SOURCE/TARGET/DEFER routing;
- probability-level expert composition;
- fresh lexical split;
- direct role supervision;
- no old-row reuse;
- parent freeze;
- heldout discipline;
- anti-loop stop condition;
- no hard parameter ceiling;
- no scale authorization;
- udocker output forwarding.

## Exact next action

Run one P100 query–relation role-router experiment from:

`2790c1e3cd4183377edae8805752f9cc0b7a1915`

Do not make another model-changing patch before its result.

If it passes dev and heldout:
- run one fresh downstream causal check.

If it fails dev or heldout:
- stop gradient work;
- perform a broader architecture-level audit;
- no automatic repair chain.


## Infrastructure failure 575799 and exact repair

Magnolia job `575799` did **not** execute QRR training.

Observed Slurm state:

- job: `575799`
- state: `FAILED`
- exit: `1:0`
- node: `gpu001`
- elapsed: `00:00:20`

The failure occurred in the runtime provenance preflight before the trainer started.

Error:

`575798 parent immutability drift`

Root cause:

The QRR runner checked the nonexistent/old field:

`parent_graph_parameters_exactly_unchanged`

but the preserved 575798 result schema correctly records:

`parent_endpoint_graph_parameters_exactly_unchanged`

The preserved 575798 receipt itself is valid; its parent graph was unchanged. The preflight key spelling was wrong.

This is infrastructure/provenance-contract failure, **not model evidence**.

No QRR gradient ran.

No QRR checkpoint exists from 575799.

No heldout data was opened.

The architecture remains unchanged.

Single causal repair commits on the QRR frontier:

- `bbdfcf9b61e6f0b19b51eb115d14273f519bb120` — bind preflight to the actual 575798 immutability field.
- `a4ba0e7b8629718947d7126f8aa86021aa1cb87c` — make CI reject the stale field name in future.

CI run:

`35389242107`

Result:

`SUCCESS`

Current QRR head for retry:

`a4ba0e7b8629718947d7126f8aa86021aa1cb87c`

The partial original output directory from 575799 must be preserved. Retry in a fresh directory:

`$HOME/rayan-compute/rayan-n0/n0-v02/query-relation-role-router-v0.3-retry-575799`

This retry is the same research-grounded QRR experiment. It is **not** a new repair, does not change the architecture, and does not consume the anti-loop model-failure budget.
