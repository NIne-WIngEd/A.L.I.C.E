# N0 Relation-Conditioned Multi-Layer Interface — Runtime Qualification Ready

**Date:** 2026-09-18  
**Continuity status:** active N0 handoff after exact 575804 relation-layer map compilation

## Live experiment frontier

Branch:

`alice-eipm-v1-relation-conditioned-multilayer-interface`

Prepared head:

`99411e2d638d9a98a09b5fb1ff1709d0da89a760`

GitHub contract workflow:

`N0 Relation Conditioned MultiLayer Interface Contract Check`

Result at this head:

`SUCCESS`

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The experiment branch is not canonical production promotion.

## What completed on Magnolia

The exact relation-conditioned layer map was compiled from frozen no-gradient layerwise audit job 575804.

Source audit SHA-256:

`ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`

Compiled map SHA-256:

`ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304`

Map path:

`$HOME/rayan-compute/rayan-n0/n0-v02/relation-conditioned-multilayer-interface-v0.1/design/relation_conditioned_layer_map.json`

Candidate sets:

- causes: `[5, 6, 7, 8]`
- corrects: `[10, 11, 16]`
- derived_from: `[3, 4, 5, 6]`
- supersedes: `[11, 12, 13, 14]`
- supports: `[0, 6, 12, 13]`
- temporal_successor: `[3, 4, 5, 6]`

Global candidate layer bank:

`[0, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16]`

The map is relation-conditioned and does not collapse to one universal layer.

The map itself records:

- `training_authorized=false`
- `scale_authorized=false`
- `n0_complete=false`
- `hard_parameter_ceiling=null`

## Magnolia operational correction preserved

Two failed attempts before the successful compile were infrastructure execution mistakes, not model evidence.

1. Running the Python 3.11-era compiler with Magnolia host Python produced a syntax error.
2. Entering udocker without the explicit mounted repository-root binding made the inner runner fall back to container `$HOME=/root`.

The branch now hardens the wrapper and the correct execution boundary remains:

- whole-stage udocker;
- explicit `ALICE_N0_REPO_ROOT`;
- persistent host namespace under `$HOME/rayan-compute`;
- explicit N0 workdir/output variables;
- no host-Python execution for N0 stages;
- no assumption that container `$HOME` equals host `$HOME`;
- CPU-only stages set `RAYAN_UDOCKER_NVIDIA=0`.

These failures do not alter the N0 architecture conclusion.

## Scientific reason for the current architecture

Job 575800 stopped the local repair path. The trained query-relation router preserved parent capability but collapsed directional role accuracy to 0.5. Heldout stayed closed. That was treated as an architecture-level failure, not an invitation to create QRR v0.4.

Job 575801 then showed that token-level late interaction preserved more relation-direction signal than the pooled views:

- raw mean pool role accuracy: 0.5625
- trained semantic projection role accuracy: 0.5625
- token late interaction role accuracy: 0.6875

However causes/supports were still at chance in the final layer.

Job 575804 performed a no-gradient layerwise audit and found that the missing direction signal is layer-local:

- causes is above chance through a middle-layer window and disappears at layers 15-16;
- supports is above chance only around layers 12-13;
- final aggregate token accuracy can remain competitive while specific directional capabilities disappear.

Therefore the active architecture exposes relation-conditioned access to intermediate token layers before graph field selection. It does not hardcode layer 12 and it does not retrain the semantic backbone.

## New prepared stage

New files at the live experiment head:

- `scripts/eipm/n0/qualify_n0_v02_relation_conditioned_multilayer_interface_v0_1.py`
- `scripts/eipm/n0/run_n0_v02_qualify_relation_conditioned_multilayer_interface_v0_1.sh`
- `docs/research/eipm-n0-relation-conditioned-multilayer-interface-runtime-qualification-v0.1.md`
- `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.18.json`

The qualifier is CPU-only and no-gradient.

It binds to the exact compiled map hash and source audit hash. It instantiates the real `RelationConditionedMultiLayerQueryInterface` and verifies:

- every directed relation uses only its candidate layers;
- layer weights normalize inside the active candidate mask;
- PAD fails closed;
- symmetric `conflicts_with` fails closed;
- masked query tokens receive no attention mass;
- output tensors have valid shapes and finite values;
- full interface state hash is exactly unchanged before/after execution;
- no gradients are created;
- no optimizer exists;
- no GPU is required;
- no heldout/frozen/private identity data is used.

## Single next authorized action

Run the exact-map CPU-only runtime qualification **once**.

Do not train before it.

Do not submit a GPU job before it.

Do not open heldout.

Do not rerun the frozen challenge.

Do not change thresholds.

Do not scale.

Do not modify the semantic backbone or graph parent.

## Magnolia command contract

First update the existing working tree to the prepared head. Magnolia uses the established older-Git-compatible checkout workflow rather than `git switch`.

Then export:

`ALICE_N0_REPO_ROOT="$PWD"`

`ALICE_N0_WORKDIR="$HOME/rayan-compute/rayan-n0/n0-v02"`

`ALICE_N0_MULTILAYER_INTERFACE_DIR="$ALICE_N0_WORKDIR/relation-conditioned-multilayer-interface-v0.1"`

`RAYAN_UDOCKER_NVIDIA=0`

Run the complete qualification through:

`scripts/eipm/n0/magnolia_udocker_exec.sh`

with:

`scripts/eipm/n0/run_n0_v02_qualify_relation_conditioned_multilayer_interface_v0_1.sh`

Do not delete the existing interface output directory. The qualifier deliberately writes into a new `qualification/` child while preserving the already-compiled map.

## Decision after the runtime qualification

If it passes, **still do not train immediately**.

The next work is to define:

1. one fresh causal interface curriculum;
2. one preservation contract for already-valid semantic/graph behavior;
3. the evidence rule for deciding whether a bounded interface-training study is justified.

If it fails, classify the failure before any patch:

- infrastructure/runtime binding;
- exact-map/schema contract;
- interface implementation defect;
- architecture contradiction.

A real architecture contradiction reopens the language↔graph boundary. It does not trigger a chain of local hotfixes.

## Durable doctrine

N0 remains the full-production foundation for the eventual EIPM, not a disposable pilot.

The permanent personality model is A.L.I.C.E.'s durable identity/judgment mechanism. N0 is still identity-neutral and receives no private Elaina gradient.

No current parameter count, layer count, field count, graph size, latent width/depth/slot count, candidate-layer count, context length, training-step count, or compute budget is a permanent capability ceiling.

Efficiency matters, but efficiency must remove waste rather than remove required capability.

The MC10/MC10D lesson remains active: build A.L.I.C.E. directly, use validation only when it changes a real causal decision, and do not let validator/infrastructure work become the project itself.

FBM should receive the runtime-qualification result only after the real qualification receipt exists.
