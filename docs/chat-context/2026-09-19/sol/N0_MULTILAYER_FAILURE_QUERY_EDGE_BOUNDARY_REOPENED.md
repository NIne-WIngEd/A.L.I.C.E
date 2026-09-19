# N0 Multilayer Failure — Query–Edge Boundary Reopened

**Date:** 2026-09-19

## Valid experiment result

Magnolia job `575825` completed normally on a Tesla P100.

- scheduler: `COMPLETED`
- exit: `0:0`
- experiment source: `8a8531884bd9d59dfff5023579b0789dc7146ea7`
- result SHA-256: `6f2f3fe6ddab764d947a7c63fe94ef24d5cd77e3cc68821c8678fdb8f2ea36bd`
- result status: `FAIL_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`
- eligible checkpoints: none
- selected candidate: none
- test opened: false
- frozen challenge evaluated: false
- parent graph parameters changed: false

The single authorized P100 experiment is consumed.

Do not submit it again.

Do not delete:

`$HOME/rayan-compute/rayan-n0/n0-v02/relation-conditioned-multilayer-interface-v0.1/causal-interface-study-v0.1/training-v0.1`

That directory is failure evidence.

## What failed

The interface learned some query-role signal but failed preservation.

Baseline causal quad accuracy was `0.08333333333333333`.

At step 40 it became `0.2222222222222222`, with causes and supports both improving.

At the same checkpoint, the proven endpoint pair accuracy fell:

`0.9583333333333334 -> 0.6041666666666666`

and endpoint mean margin fell:

`0.8209420529504617 -> 0.24894556403160095`

No parent tensor moved.

This is a language↔graph boundary failure, not parent corruption and not an infrastructure failure.

## Architectural localization

The failed specialist receives query hidden states and relation type, but it never receives the actual source or target graph states before choosing its endpoint delta.

For a fixed query and relation type, same-type edges therefore receive the same specialist signal.

The final path is also constrained to one anti-symmetric scalar:

`+delta(source), -delta(target)`

The proven parent is different. It conditions its relation read on query + relation + source state + target state and has separate source/target heads.

The one-edge/two-field causal curriculum could reward the wrong abstraction because it required choosing a role but not identifying which edge content the query referred to.

## Architecture family closed

Do not revive:

`query-only relation-conditioned scalar -> +source / -target`

under another version number or name.

That includes:

- QRR v0.4
- another signed endpoint router
- smaller learning rate
- longer training
- wider copy
- relaxed preservation tolerances
- replay loss attached to the same scalar abstraction

## Next hypothesis

The next design is a **Query–Edge Cross-Attention Bridge**.

Its role is to bind query semantics to a specific graph edge.

For each active directed edge, the specialist must jointly use:

- relation-conditioned intermediate query tokens
- source graph state
- target graph state
- relation embedding

The specific edge must condition token attention.

The specialist emits independent source and target residuals rather than a forced zero-sum delta.

A learned edge-specific gate controls whether the specialist contributes.

Parent dual-endpoint behavior is the shared frozen expert and remains exact at initialization.

## New causal study requirement

Do not reuse the old one-edge/two-field causal study as the decisive gate.

The next fresh causal curriculum must include:

- multiple fields
- multiple directed edges
- multiple same-type edges
- distractor relations
- content-matched and mismatched endpoints
- source-role queries
- target-role queries
- edge reversals
- isolated train/dev/test templates and entities

The task must require query-to-edge binding. A relation-global SOURCE/TARGET policy must not solve it.

## Preservation learning

Frozen preservation DEV remains evaluation-only.

A future training decision may use TRAIN-only anchors from the old ordinary and endpoint curricula so the new specialist learns stability against the frozen parent without contaminating DEV.

The exact mechanism—distillation, routing constraint, gradient constraint, or combination—is not yet selected.

No training objective is authorized yet.

## Current authoritative experiment state

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.22.json`

Status:

`ONE_SHOT_MULTILAYER_EXPERIMENT_FAILED_VALIDLY;QUERY_ONLY_ANTISYMMETRIC_RESIDUAL_REJECTED;LANGUAGE_GRAPH_BOUNDARY_REOPENED;QUERY_EDGE_BRIDGE_DESIGN_AUTHORIZED_NO_GRADIENT`

## Next scope

Authorized:

- implement the new edge-specific bridge
- build fresh multi-edge causal curriculum
- build preservation TRAIN-anchor compiler
- static CI
- CPU/no-gradient exact-parent qualification

Not authorized:

- optimizer
- gradient
- GPU training
- heldout/test
- frozen challenge
- scale
- parent retraining
- semantic retraining
- private identity gradient
- promotion
- N0 completion

If this next architecture later fails a genuine model gate, stop again and re-open the larger architecture question before another change.
