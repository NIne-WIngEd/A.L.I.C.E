# N0 Multi-Layer Causal Study — CPU Preparation Ready

**Date:** 2026-09-18  
**Continuity status:** exact-map runtime qualification passed; fresh factorial causal curriculum and preservation contract are defined; CPU preparation is the single next authorized action

## Live N0 experiment frontier

Branch:

`alice-eipm-v1-relation-conditioned-multilayer-interface`

Head:

`02a05bf19545b5b281c3cfe7dc2f674b35534793`

GitHub contract workflow result:

`SUCCESS`

Stable build base remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

## Magnolia runtime qualification result

Source revision executed:

`99411e2d638d9a98a09b5fb1ff1709d0da89a760`

Qualification status:

`PASS_EXACT_MAP_NO_GRADIENT_RUNTIME_CONTRACT`

Qualification receipt SHA-256:

`779b19459fa8ab67bb1c3ed56b3e9cdb1b89b8382f145e1b3d7c4d98312b2b57`

Compiled relation-layer map SHA-256:

`ac0e5e28749cbb1e8dda62262c4c2761db887b965cbc3a74493ff6c9b5355304`

Source 575804 audit SHA-256:

`ee88e80513fb000e2ecde8ca9e4f7e714cb06e43a144bd28c8930d26c38965ab`

The qualification confirmed:

- parameter state exactly unchanged;
- no gradient;
- no optimizer;
- no GPU requirement;
- no heldout use;
- no frozen challenge use;
- no training authorization;
- no scaling authorization;
- no heldout opening authorization;
- no frozen challenge rerun authorization;
- N0 remains incomplete.

This completes the designed interface's runtime-contract gate.

## New authoritative experiment state

State:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.19.json`

Status:

`EXACT_MAP_RUNTIME_QUALIFICATION_PASS;FRESH_CAUSAL_CURRICULUM_AND_PRESERVATION_CONTRACT_DEFINED;CPU_PREPARATION_AUTHORIZED;TRAINING_DECISION_PENDING`

## Fresh causal curriculum

Builder:

`scripts/eipm/n0/build_n0_v02_relation_conditioned_multilayer_interface_curriculum_v0_1.py`

The curriculum is a 2x2 factorial design.

For every scenario:

- query role independently varies between SOURCE-seeking and TARGET-seeking;
- stored edge direction independently varies between A and reversed B;
- field text stays fixed;
- relation type stays fixed;
- target field must flip when query role flips;
- target field must also flip when edge direction flips.

Relations:

- corrects;
- supersedes;
- derived_from;
- causes;
- supports;
- temporal_successor.

Operating size:

- 18 train quads per relation;
- 6 dev quads per relation;
- 6 unopened test quads per relation;
- 180 quads total;
- 720 rows total.

No prior failed QRR, semantic-grounding, role-residual, endpoint-heldout, localization, or frozen-challenge row is reused.

All curriculum rows still say:

`training_authorized=false`

The curriculum is an experiment design artifact, not an optimizer authorization.

## Preservation contract

Path:

`configs/eipm/n0/n0_v02_relation_conditioned_multilayer_preservation_contract_v0_1.json`

Frozen parent graph:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

Frozen parent components:

- semantic backbone;
- structured-state encoder;
- evidence-view adapter;
- selected evidence graph.

Preservation lanes:

1. ordinary graph replay;
2. endpoint-role replay;
3. symmetric `conflicts_with`;
4. PAD/unmapped fail-closed behavior.

Candidate results are forbidden from defining preservation tolerances.

Any numerical tolerance must be frozen from repeated canonical-parent behavior before candidate training.

## CPU preparation

Runner:

`scripts/eipm/n0/run_n0_v02_prepare_relation_conditioned_multilayer_causal_study_v0_1.sh`

Output root:

`$ALICE_N0_MULTILAYER_INTERFACE_DIR/causal-interface-study-v0.1`

Preparation will:

- re-bind the map, qualification receipt, source audit, and selected parent graph;
- materialize the 720-row curriculum;
- validate all factorial quads;
- bind semantic/structured/adapter/graph/replay artifacts by SHA-256;
- bind the preservation contract;
- create one preparation receipt.

It will not:

- create an optimizer;
- perform a gradient;
- submit a GPU job;
- open test;
- open frozen challenge;
- authorize scaling;
- authorize private identity learning.

## Single next authorized action

Run the CPU-only causal-study preparation once through whole-stage udocker.

Stop after the preparation receipt.

Do not submit a GPU job afterward.

After the receipt exists, the next decision is whether one bounded interface-only training experiment is scientifically justified.

## Training decision boundary

If later authorized, a candidate must use the real:

`RelationConditionedMultiLayerQueryInterface`

and its per-edge:

`edge_relation_query_state`.

Forbidden substitutes:

- raw mean-pooled router;
- final-layer-only router;
- hardcoded layer 12;
- QRR v0.4;
- previous semantic-role residual;
- generic endpoint classifier.

The selected parent graph remains an immutable expert.

A future candidate must initialize from exact parent behavior and expose the new interface contribution explicitly enough to determine whether any gain is actually caused by relation-conditioned multi-layer semantic access.

## Anti-loop

If causal dev fails while preservation passes, reopen the language-graph boundary.

If causal dev passes while preservation fails, reject the candidate.

Do not create an automatic hotfix chain in either case.

Infrastructure failure remains non-model evidence.

## Capability doctrine

N0 remains the full-production foundation, not a disposable pilot.

The 720-row study size, current candidate-layer sets, current semantic depth, graph width, and future interface width are operating points, not permanent capability ceilings.

Efficiency is achieved through selective layer access and better computation, not by shrinking required capability.
