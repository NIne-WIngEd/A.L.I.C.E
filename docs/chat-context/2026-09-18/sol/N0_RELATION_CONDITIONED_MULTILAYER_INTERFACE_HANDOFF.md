# N0 Relation-Conditioned Multi-Layer Query Interface Handoff

**Date:** 2026-09-18  
**Status:** architecture designed and CI-qualified from 575804 evidence; exact relation-layer map still must be compiled from the saved audit JSON; training remains unauthorized

## Current frontier

`alice-eipm-v1-relation-conditioned-multilayer-interface @ 5c6801577695b315b4a7443c28f5c7d42c1c3446`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

Frontier is 95 commits ahead and 0 behind stable.

## Evidence basis

Magnolia job `575804` completed the no-gradient layerwise semantic audit.

Result SHA-256:

`ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`

Key findings:

- best token late-interaction overall: layer 12, accuracy 0.6875;
- best mean-pool overall: layer 4, accuracy 0.6666666666666666;
- causal-direction signal appears through middle layers and is lost again at final layers;
- support-direction signal is above chance only in a narrow layer-12/13 window;
- final-layer aggregate accuracy can remain competitive while specific relation-direction capability disappears.

Therefore:

- the relation-semantic capability is not absent from the frozen backbone;
- final-layer-only query interfaces are insufficient;
- layer 12 must not be hardcoded as a permanent universal layer;
- relation-conditioned multi-layer token access is the appropriate next architecture hypothesis.

## Derived map

Compiler:

`scripts/eipm/n0/compile_n0_v02_relation_conditioned_layer_map_v0_1.py`

Guarded runner:

`scripts/eipm/n0/run_n0_v02_compile_relation_conditioned_layer_map_v0_1.sh`

The map is compiled directly from the exact 575804 JSON.

For each relation it records:

- exact best layers;
- near-best layers within one audit observation quantum;
- full ranked layer evidence;
- a compact depth-diverse candidate set.

The candidate set and its size are operating priors only. They are not permanent architecture limits.

The compiler emits:

- `training_authorized=false`
- `scale_authorized=false`
- `hard_parameter_ceiling=null`

## Interface design

Source:

`src/alice_personality/n0/relation_conditioned_multilayer_query.py`

Class:

`RelationConditionedMultiLayerQueryInterface`

For each active directed relation edge:

1. use relation type to obtain the audit-derived candidate-layer mask;
2. attend over query tokens separately inside each candidate semantic layer;
3. produce one relation-conditioned token summary per candidate layer;
4. mix those summaries with relation-conditioned layer scoring;
5. add the frozen audit scores only as soft initialization priors;
6. project the resulting edge-specific query state into graph space.

The architecture therefore moves language↔relation interaction before graph field selection.

It does not:

- consume only the final pooled query;
- hardcode layer 12;
- train the frozen semantic backbone;
- train the frozen graph parent;
- invent directional semantics for `conflicts_with`;
- impose a hard capacity ceiling.

PAD, symmetric conflict, and unmapped relations fail closed with no directional interface state.

## Design document

`docs/research/eipm-n0-relation-conditioned-multilayer-query-interface-v0.1.md`

## Authoritative state

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.17.json`

Training is explicitly unauthorized.

Next action is map compilation and inspection only.

## CI

Workflow:

`N0 Relation Conditioned MultiLayer Interface Contract Check`

Final run:

`35392846604`

Result:

`SUCCESS`

The earlier failed CI attempts were contract-test issues only:

- first caught missing explicit no-ceiling metadata in the compiled map;
- second caught a brittle wording assertion in CI.

The compiler now carries `hard_parameter_ceiling=null`, and the CI assertion was changed to a structural check. The interface design itself was not modified to satisfy wording.

## FBM

New FBM trace:

`docs/fable-builder/traces/FBM_TRACE_20260918_N0_MULTILAYER_QUERY_INTERFACE_DESIGN.jsonl`

It captures the reusable lesson:

> a diagnostic best layer should become a soft relation-specific prior, not a permanent universal architecture choice.

The FBM parallel-capture contract already requires future architecture audits and negative evidence to be logged without owner reminders.

## Exact next action

On Magnolia login node only:

1. update to frontier `5c6801577695b315b4a7443c28f5c7d42c1c3446`;
2. run the guarded map compiler;
3. inspect the per-relation map.

No `sbatch`.

No GPU.

No gradient.

No model training.

After the map is inspected, decide whether the interface topology is well-supported enough to stage one bounded fresh causal experiment.
