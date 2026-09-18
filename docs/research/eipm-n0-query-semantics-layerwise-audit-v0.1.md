# N0 Query-Semantics Layerwise Audit v0.1

**Date:** 2026-09-18  
**Trigger:** Magnolia job 575801, successful no-gradient representation audit  
**Policy:** all gradient work remains frozen

## 575801 result

Job 575801 completed with exit 0:0.

Result SHA-256:

`8bb4aa0f249b763342d78f5bf0f2f960a0bd73fafa67d18f8e0e7342f62f24b6`

The representation audit compared:

- raw final-layer mean pool;
- trained semantic projection;
- final-layer token late interaction.

Aggregate role accuracy:

- raw mean pool: `0.5625`;
- trained semantic projection: `0.5625`;
- token late interaction: `0.6875`.

Mean opposite-role pair distance:

- raw mean pool: `0.05102736999591192`;
- trained semantic projection: `0.013605944812297821`;
- token late interaction: `0.11760250230630238`.

Per-relation token-late-interaction accuracy:

- corrects: `0.875`;
- supersedes: `0.75`;
- temporal_successor: `0.75`;
- derived_from: `0.75`;
- causes: `0.5`;
- supports: `0.5`.

Therefore token-level contextual states preserve materially more directional relation signal than either pooled representation, but the final token layer is still insufficient for causes/supports.

The trained 256-D semantic projection is not a remedy for the graph interface; on this audit it has the same role accuracy as the raw pool and substantially smaller source-target pair distance.

## Architectural interpretation

The result rules out a simple next step of "replace raw pool with semantic projection."

It also does not justify immediately training a token-level graph adapter, because two relation families remain at chance.

The correct next question is whether causes/supports relation direction exists in an earlier frozen transformer layer and is lost by later representation shaping.

## External research connection

Layerwise probing literature reports that relation signals can be distributed unevenly across transformer depth rather than being maximized at the final layer.

Recent semantic-relation probing reports directional asymmetries and mid-layer peaks for some relation classes.

Linear relational concept work also reports stronger relational concept recovery from selected earlier/intermediate activations than from a generic final representation.

This motivates a no-gradient layerwise audit before any new graph or semantic training.

## Layerwise audit

Script:

`scripts/eipm/n0/audit_n0_v02_query_semantic_layers_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_query_semantics_layerwise_audit_v0_1.sh`

Magnolia job:

`scripts/eipm/n0/magnolia_n0_v02_query_semantics_layerwise_audit_v0_1.sbatch`

The audit evaluates every hidden-state output from the frozen 16-layer semantic backbone, including embedding output and final layer.

For each layer it measures:

- token-level late-interaction SOURCE/TARGET role accuracy;
- mean-pool SOURCE/TARGET role accuracy;
- opposite-role query-pair distance;
- relation-specific token role accuracy, with causes/supports surfaced explicitly.

No optimizer is created.

No gradient is computed.

No model tensor is mutated.

No heldout/frozen/QRR/prior-repair row is used.

## Decision rule

### If an earlier layer materially restores causes/supports

Directional relation semantics exist in the semantic backbone but are transformed/lost by later layers.

Next architecture should expose selected token layers to a relation-aware language↔graph interface rather than retraining the evidence graph against the final pooled state.

### If no layer restores causes/supports

The semantic model's current public training objective/data does not reliably encode those directional relation families.

Stop graph work.

Reopen semantic representation curriculum/objectives before any graph gradient.

## Anti-loop

This is still architecture diagnosis, not repair.

A layerwise result does not automatically authorize training. One architecture decision must be made from the result first.
