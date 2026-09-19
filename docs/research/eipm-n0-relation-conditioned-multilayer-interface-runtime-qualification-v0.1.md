# N0 Relation-Conditioned Multi-Layer Interface Runtime Qualification v0.1

**Date:** 2026-09-18  
**Status:** exact 575804-derived layer map compiled on Magnolia; one CPU-only no-gradient runtime contract qualification is the next authorized action

## Preserved evidence

The relation-conditioned map was compiled from the frozen job 575804 layerwise audit with the exact source audit hash:

`ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`

Observed compiled map SHA-256:

`ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304`

Observed candidate sets:

- `causes`: 5, 6, 7, 8
- `corrects`: 10, 11, 16
- `derived_from`: 3, 4, 5, 6
- `supersedes`: 11, 12, 13, 14
- `supports`: 0, 6, 12, 13
- `temporal_successor`: 3, 4, 5, 6

Global candidate layer bank:

`[0, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16]`

The map is relation-conditioned and depth-diverse. It does not collapse to one universal layer. The candidate sets remain operating priors for the first causal study rather than permanent architecture limits.

## Why this qualification exists

The architecture design required four pre-gradient actions:

1. compile the exact relation-layer map from the preserved 575804 audit;
2. inspect the map for pathological universal-layer hardcoding or unsupported relation behavior;
3. run static/runtime shape and contract qualification of the interface;
4. define one fresh causal curriculum and one preservation contract before deciding whether any bounded interface training is justified.

Steps 1 and 2 are now complete.

This document authorizes only step 3.

It does not create another validation program. It is the single runtime qualification already required by the architecture design.

## Qualification contract

Source:

`scripts/eipm/n0/qualify_n0_v02_relation_conditioned_multilayer_interface_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_qualify_relation_conditioned_multilayer_interface_v0_1.sh`

The qualifier must:

- bind to the exact compiled map SHA-256 above;
- bind through that map to the exact 575804 audit SHA-256;
- instantiate the real `RelationConditionedMultiLayerQueryInterface`;
- exercise PAD, all current directed relation types, and symmetric `conflicts_with`;
- verify directed relations receive only their map-selected candidate layers;
- verify layer weights normalize only across active candidate layers;
- verify PAD and symmetric `conflicts_with` fail closed with no interface state;
- verify token masks are respected;
- verify all returned interface tensors are finite and have the expected shapes;
- hash the full interface state before and after qualification and require exact equality;
- require no gradients;
- require no optimizer;
- require no GPU;
- use no heldout rows;
- use no frozen-challenge rows;
- use no private identity data.

## Magnolia execution boundary

Run the complete stage inside the validated whole-stage udocker environment.

Do not run the Python qualifier on Magnolia host Python.

The caller must bind the live mounted repository root through `ALICE_N0_REPO_ROOT` and preserve the existing N0 workdir/output path.

This qualification is CPU-only. Set `RAYAN_UDOCKER_NVIDIA=0`.

The udocker wrapper must remain responsible for forwarding `ALICE_REPO_ROOT`, `ALICE_N0_WORKDIR`, and `ALICE_N0_MULTILAYER_INTERFACE_DIR` into the container.

## Decision after qualification

### If the exact-map runtime qualification passes

Do not train immediately.

The next architecture work is:

1. define one fresh causal interface curriculum;
2. define one preservation contract for already-valid graph/semantic behavior;
3. define the minimum evidence that would justify or reject a bounded interface-training study;
4. make one explicit training decision from those contracts.

### If the qualification fails

Do not hotfix forward through version churn.

Classify the failure first:

- infrastructure/runtime binding;
- map/schema contract;
- interface implementation defect;
- or architecture contradiction.

Infrastructure failure is not model evidence.

A real interface implementation defect may be corrected once at its cause and the same qualification repeated. A broader architecture contradiction reopens the language↔graph boundary rather than creating a chain of local routers.

## Explicit non-authorizations

This stage does **not** authorize:

- optimizer creation;
- gradient training;
- semantic-backbone retraining;
- graph-parent retraining;
- new GPU execution;
- scaling;
- heldout opening;
- frozen-challenge rerun;
- threshold changes;
- private identity gradients;
- production promotion;
- N0 completion.

## Capability doctrine

The current candidate-layer budget is not a permanent capability ceiling.

No parameter count, layer count, width, depth, field count, graph size, latent-slot count, or runtime budget is frozen as the final A.L.I.C.E. personality-model limit by this qualification.

Efficiency remains a design objective. Capability and identity fidelity remain the governing constraints.
