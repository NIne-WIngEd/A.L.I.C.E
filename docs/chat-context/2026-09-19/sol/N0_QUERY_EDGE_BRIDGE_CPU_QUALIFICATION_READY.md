# N0 Query–Edge Cross-Attention Bridge — CPU Qualification Ready

**Date:** 2026-09-19

## Why this branch exists

Magnolia job `575825` validly failed the one-shot relation-conditioned multi-layer interface experiment.

Failed result:

`6f2f3fe6ddab764d947a7c63fe94ef24d5cd77e3cc68821c8678fdb8f2ea36bd`

Status:

`FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`

That failure closed the architecture family:

`query-only relation-conditioned scalar -> +source / -target`

The failure was not infrastructure. Parent tensors stayed exact. The new path learned some query-role signal but destroyed endpoint preservation.

## New branch

`alice-eipm-v1-query-edge-cross-attention-bridge`

Current head:

`c6170597e2b55cc378c5d8cf1560ca87802ea085`

Static contract workflow:

`35463216917`

Conclusion:

`SUCCESS`

Authoritative current state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.23.json`

Status:

`QUERY_EDGE_CROSS_ATTENTION_BRIDGE_IMPLEMENTED;FRESH_MULTI_EDGE_CAUSAL_STUDY_DEFINED;STATIC_CONTRACT_PASS;CPU_NO_GRADIENT_RUNTIME_QUALIFICATION_AUTHORIZED`

## Architectural change

New specialist:

`src/alice_personality/n0/query_edge_cross_attention_bridge.py`

Integrated graph:

`src/alice_personality/n0/evidence_graph_query_edge_bridge.py`

Unlike the failed residual, the new bridge receives the actual source and target graph states before reading query tokens.

For each directed edge it combines:

- relation embedding;
- source graph state;
- target graph state;
- directional source-target difference.

That edge-specific state becomes the query used to attend over relation-conditioned intermediate semantic token layers.

Therefore two edges with the same relation type can produce different query-token attention.

The bridge emits separate:

- source residual proposal;
- target residual proposal.

They are not constrained to be negatives of one another.

A separate edge-specific gate controls specialist contribution.

Only the final gate projection is zero-initialized. Residual heads remain live. This preserves the exact parent at initialization without creating a doubly-zero learning path.

The proven `DualEndpointEvidenceGraphEncoder` remains the parent/shared expert.

## Fresh causal study

Builder:

`scripts/eipm/n0/build_n0_v02_query_edge_binding_curriculum_v0_1.py`

The study has:

- 6 directed relation families;
- 180 quads / 720 rows;
- 8 fields per row;
- 4 edges per row;
- 3 edges with the same target relation;
- 1 different-relation distractor;
- rotating relevant same-relation edge position;
- source/target query-role counterfactual;
- A/B edge-direction counterfactual;
- isolated train/dev/test subjects, attributes, and query framing.

The relevant edge is explicitly labeled for routing diagnostics.

A relation-global endpoint polarity does not identify the relevant edge. Without content binding, uniform guessing among same-relation edges has an upper bound of 1/3.

## Preservation anchors

Builder:

`scripts/eipm/n0/build_n0_v02_query_edge_preservation_train_anchors_v0_1.py`

It only enumerates existing TRAIN rows from:

- ordinary graph curriculum;
- endpoint-role curriculum.

DEV and other non-train rows are explicitly excluded.

These anchors do not authorize gradient use. They merely make a future preservation-training design possible without contaminating frozen DEV.

The future mechanism—distillation, routing constraint, gradient constraint, or another justified method—has not been selected.

## Runtime qualification

Qualifier:

`scripts/eipm/n0/qualify_n0_v02_query_edge_cross_attention_bridge_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_qualify_query_edge_cross_attention_bridge_v0_1.sh`

The CPU/no-gradient qualification must prove:

- exact parent field weights;
- exact parent pooled state;
- all parent tensors unchanged;
- zero specialist gate at initialization;
- same-relation edges with different endpoints produce different edge queries;
- endpoint content changes query-token attention;
- conflicts receive no directional specialist;
- PAD receives no specialist;
- forced antisymmetry is absent;
- future trainable scope is bridge-only.

It is bound to:

- failed result SHA `6f2f3fe6...`;
- parent graph SHA `3ae08aa2...`;
- existing audited layer map SHA `ac0e5e28...`.

## Current authorization boundary

Authorized now:

- one CPU/no-gradient runtime qualification.

Not authorized:

- optimizer;
- gradient;
- GPU training;
- heldout/test;
- frozen challenge;
- scale;
- parent retraining;
- semantic retraining;
- private identity gradient;
- promotion;
- N0 completion.

A qualification PASS does **not** automatically authorize training.

After qualification, review the receipt and make a separate training-objective decision.

## Magnolia rules

Use whole-stage udocker.

Do not use host Python.

Set `RAYAN_UDOCKER_NVIDIA=0`.

Do not submit a GPU job.

If the qualification output directory is created and the stage fails, preserve it and classify the failure before any corrective change.
