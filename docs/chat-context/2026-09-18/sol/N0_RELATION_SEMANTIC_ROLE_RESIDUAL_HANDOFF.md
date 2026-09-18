# N0 Relation-Semantic Role Residual Repair Handoff

**Date:** 2026-09-18  
**Status:** relation-semantic grounding v0.1 failed validly; failure localized to shared read-path entanglement; isolated zero-init semantic-role residual is CI-ready for one P100 run

## Current experiment frontier

`alice-eipm-v1-relation-semantic-role-residual @ 23a84b808a0f16c63678d0875b9a7b2d388f07e7`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

The current frontier is 60 commits ahead and 0 behind stable. It is experimental and noncanonical.

## Preserved evidence chain

### Endpoint parent still valid

Selected endpoint-role parent:

`relation-endpoint-repair-v0.2/training/step-00000200/evidence_graph_dual_endpoint.safetensors`

SHA-256:

`3ae08aa2fc2c46ed74a47792310c6c2bf202fad800549cecc36c5365523dc47f`

Earlier downstream causal arbitration classified this parent as:

`IMPROVEMENT`

Do not discard or overwrite this parent.

### Final frozen challenge remains immutable

Job `575794` remains a valid frozen FAIL.

Do not rerun it. Do not change its thresholds.

### Missing-evidence localization

Job `575795` localized the first unresolved failure to:

`EVIDENCE_GRAPH_RELATION_SELECTION_REMAINS_DEFECTIVE`

Result SHA-256:

`1733bf1a5fdc4f979beab6d5198bfb6a77a0e17b4fc96adc134dafd03dd5ebbe`

The graph reacted to relation direction, fusion preserved that difference, but the graph usually selected the wrong endpoint.

### Relation-semantic grounding v0.1

Job `575797` completed successfully as an execution with exit `0:0`, but the model-side repair failed.

Result SHA-256:

`cfd2a6aee3ba2e441363700cf0d57917cf7cf16c3c81b2a789b5c1ecf3614980`

Status:

`FAIL_RELATION_SEMANTIC_GROUNDING_HELDOUT_STOP_AND_LOCALIZE_NO_AUTOMATIC_HOTFIX`

Selected checkpoint:

`step-00000080`

Selected graph SHA-256:

`af71588237028354cc249bdbac3bc2a285a793b62fef72c8910dcbe6f1cf3f20`

Semantic heldout:

- pair accuracy: `0.4444444444444444`
- family-min pair accuracy: `0.0`
- row accuracy: `0.6111111111111112`
- mean graph target margin: `-0.15873396520813307`
- source semantic pair accuracy: `0.5`
- target semantic pair accuracy: `0.3888888888888889`

Parent explicit endpoint-role dev pair accuracy:

`0.9583333333333334`

Selected v0.1 checkpoint endpoint-role dev pair accuracy:

`0.7083333333333334`

Across all saved v0.1 checkpoints:

- semantic worst-family dev pair accuracy remained `0.0`;
- endpoint-role behavior stayed below the parent;
- ordinary replay remained comparatively healthy.

Therefore the problem is not merely undertraining or one unlucky checkpoint.

## Causal localization after 575797

The v0.1 repair mutated the same parent parameters that already encoded explicit source-vs-target endpoint skill:

- `pool_query`
- `query_projection`
- `pool_relation_embedding`
- `source_relation_pool_mlp`
- `directed_relation_pool_mlp`

Natural relation-semantic training improved some semantic families but interfered with the previously learned endpoint-role behavior.

Current localization:

`shared mutable relation-read path entangles explicit endpoint-role skill with natural relation-semantic role grounding`

This is not evidence of:

- insufficient graph width;
- insufficient graph depth;
- fusion failure;
- latent capacity failure;
- total parameter-count ceiling.

Do not scale from this result.

## One causal change: isolated semantic-role residual

New source:

`src/alice_personality/n0/evidence_graph_semantic_role_residual.py`

Class:

`SemanticRoleResidualEvidenceGraphEncoder`

It subclasses the proven `DualEndpointEvidenceGraphEncoder`.

The parent endpoint graph is loaded unchanged.

New trainable modules only:

- `semantic_role_query_adapter`
- `semantic_role_delta_mlp`

All pre-existing graph parameters are frozen.

The new residual produces one signed scalar per directed relation:

- positive delta => prefer relation source;
- negative delta => prefer relation target.

The delta is added to the source and subtracted from the target.

Inputs:

- existing parent query representation;
- relation-type embedding.

It deliberately does not need endpoint content to decide the semantic endpoint role.

Conflicts remain symmetric and receive no role delta.

### Exact parent preservation

The residual output is exactly zero at initialization because its final layer is zero-initialized.

Before training, the trainer requires exact metric parity with the step-200 parent on:

- endpoint-role dev;
- ordinary replay.

During training, every saved checkpoint verifies that every pre-existing parent graph tensor is byte-for-byte unchanged.

Any parent parameter movement aborts the run.

## Fresh v0.2 semantic curriculum

Builder:

`scripts/eipm/n0/build_n0_v02_relation_semantic_role_residual_curriculum_v0_2.py`

Fresh corpus:

- 12 relation-semantic families;
- 36 relation-flip pairs per family;
- 24 train;
- 6 dev;
- 6 heldout test;
- 864 total rows.

Families:

- corrects_current
- corrects_previous
- supersedes_current
- supersedes_previous
- temporal_current
- temporal_previous
- causes_cause
- causes_effect
- supports_supported
- supports_supporter
- derived_item
- derived_basis

Within every pair:

- same fields;
- same query;
- same confidence;
- same temporal metadata;
- same provenance;
- only relation direction flips;
- correct answer flips according to relation semantics.

No v0.1 semantic row is reused.

No frozen-challenge row is reused.

No job-575795 diagnostic row is reused.

No endpoint-repair heldout row is reused.

## Heldout discipline

Trainer:

`scripts/eipm/n0/train_n0_v02_relation_semantic_role_residual_v0_2.py`

The new v0.2 heldout test is **not evaluated automatically after training**.

A checkpoint must first satisfy:

1. ordinary replay preservation;
2. explicit endpoint-role dev preservation;
3. semantic dev readiness.

Only then may the heldout test be opened once.

If no checkpoint is eligible, the run ends with:

`FAIL_RELATION_SEMANTIC_ROLE_RESIDUAL_DEV_STOP_NO_HELDOUT_EXPOSURE`

and the heldout remains untouched.

## Runtime

Runner:

`scripts/eipm/n0/run_n0_v02_relation_semantic_role_residual_v0_2.sh`

Magnolia wrapper:

`scripts/eipm/n0/magnolia_n0_v02_relation_semantic_role_residual_v0_2.sbatch`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.12.json`

## CI

Workflow:

`N0 Relation Semantic Role Residual Contract Check`

Run:

`35385643620`

Result:

`SUCCESS`

Passed:

- Python/Bash syntax;
- zero-init residual architecture contract;
- source/target complementary delta contract;
- conflict symmetry;
- fresh v0.2 data isolation;
- residual-only trainable scope;
- parent exact-preservation enforcement;
- heldout access discipline;
- exact evidence/hash binding;
- no Git mutation in runtime runner;
- no frozen/localization/v0.1 row reuse;
- no hard parameter ceiling;
- no premature scale authorization;
- udocker output forwarding.

## Exact next action

Run exactly one P100 role-residual repair from:

`23a84b808a0f16c63678d0875b9a7b2d388f07e7`

Do not modify the model again before that evidence.

If no dev-eligible checkpoint exists:
- heldout stays closed;
- localize from dev without automatic hotfix.

If dev + preservation pass and heldout passes:
- run one fresh downstream causal check on independent semantic analogues;
- do not rerun the frozen challenge.

If heldout fails:
- preserve the failure;
- inspect the single residual failure;
- no automatic rerun or scale.

## Standing doctrine

- N0 remains the full production personality-model foundation, not a pilot.
- current shapes and parameter counts are operating points, never product ceilings.
- efficiency may not silently remove required capability.
- scale only after a localized capacity/expressivity bottleneck.
- one causal change per valid model failure.
- infrastructure failures are not model evidence.
- failed receipts remain immutable.
- private identity gradient remains closed at N0.
- no production promotion while the current repair is unresolved.
