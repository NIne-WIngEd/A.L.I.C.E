# N0 Query-Semantics Architecture Audit Handoff

**Date:** 2026-09-18  
**Status:** QRR v0.3 failed validly at dev; heldout remained unopened; all gradient work is frozen pending one no-gradient semantic-representation audit

## Current frontier

`alice-eipm-v1-query-semantics-architecture-audit @ c169f98f439705a517191b2d93dc6388bc891a1a`

Stable build remains:

`alice-eipm-v1-build @ 021c5021a98104b35f9c8e94b19e48d21f25f132`

Frontier is 79 commits ahead and 0 behind stable.

## Magnolia job 575800

Job `575800` completed successfully as an execution:

- state: `COMPLETED`
- exit: `0:0`
- elapsed: `00:01:45`
- node: `gpu001`

Result SHA-256:

`0f7917446d9f23c5d58e1eb1e1e5641a8df38da1a4730926a90d73ec044305a0`

Result:

`FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE`

Preservation candidates:

- step 120
- step 160
- step 200
- step 240
- step 260

Eligible candidates:

`[]`

Best dev checkpoint:

`step-00000260`

Heldout:

- semantic heldout evaluated: false
- heldout rows opened after training: false

Parent graph remained exactly unchanged.

Scale remained unauthorized.

N0 remains incomplete.

## Failure geometry

The router solved preservation but not semantic direction.

At late checkpoints, endpoint-role preservation returned to the step-200 parent level and ordinary replay was strong.

But semantic role accuracy stayed exactly `0.5` with family-min `0.0`.

The families collapsed by endpoint role:

- source-seeking semantic families -> SOURCE
- target-seeking semantic families -> also SOURCE

Earlier checkpoints sometimes flipped to the opposite global role.

Therefore the remaining problem is not destructive interference. It is failure to recover the directional semantic distinction from the query representation supplied to the router.

## Architecture-level finding

The graph repair/cache path constructs query semantics using:

`AliceN0V02Model.encode(...)`

which returns a raw mean pool over contextual token states.

The semantic model itself exposes:

- token states through `encode_tokens/encode_views`;
- a dedicated 256-D `semantic_projection`;
- a parameter report explicitly stating pooled readout is not a capability ceiling.

The public semantic multitask objective applies semantic contrastive learning through:

`project_semantic_pooled(...)`

The evidence graph currently consumes neither the trained semantic projection nor token-level query states.

Therefore the next question is representation-level, not router-level.

## External research

Research note:

`docs/research/eipm-n0-query-semantics-architecture-audit-v0.1.md`

Relevant lessons were drawn from:

- AdapterFusion;
- ColBERT / token-level late interaction;
- relation-extraction representation probing;
- predicate-argument enriched sentence encoding;
- relation-aware language–graph transformers;
- question-adaptive relation-aware graph attention.

No claim is made that mean pooling is generally invalid. The audit measures this exact N0 checkpoint.

## No-gradient audit

Script:

`scripts/eipm/n0/audit_n0_v02_query_semantic_representations_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_query_semantics_architecture_audit_v0_1.sh`

Magnolia:

`scripts/eipm/n0/magnolia_n0_v02_query_semantics_architecture_audit_v0_1.sbatch`

Representations compared:

1. raw 640-D mean pool currently consumed by graph;
2. trained 256-D semantic projection;
3. token-level symmetric late interaction.

Measurements:

- relation-conditioned source/target role-gloss accuracy;
- opposite-role query-pair distance;
- source-minus-target direction consistency across paraphrases.

Fresh audit-only text is used.

No frozen, heldout, QRR, prior repair, or localization rows are reused.

No gradient or model mutation is permitted.

## Decision policy

If trained semantic projection is clearly more informative:
- redesign graph query interface around trained semantic geometry;
- do not make another role router.

If token-level representation is clearly more informative:
- move language↔relation interaction before graph reduction using token-level cross-attention / late interaction.

If all three fail:
- stop evidence-graph repair;
- reopen semantic representation learning/objectives and directional semantic coverage.

No new gradient run is authorized until this audit result is interpreted.

## CI

Workflow:

`N0 Query Semantics Architecture Audit Contract Check`

Run:

`35390844449`

Result:

`SUCCESS`

## Standing anti-loop rules

- no QRR v0.4;
- no loss-weight patch;
- no longer-training patch;
- no router-width patch;
- no frozen-challenge rerun;
- no threshold change;
- no heldout opening;
- no scaling from current evidence;
- private identity gradient remains closed.
