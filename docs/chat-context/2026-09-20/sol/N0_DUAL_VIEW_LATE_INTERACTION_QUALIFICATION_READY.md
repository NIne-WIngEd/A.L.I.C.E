# N0 Dual-View Late-Interaction Binding — Qualification Ready

Date: 2026-09-20

## Source evidence

Binding identifiability audit:
- receipt SHA256: 6d166af91240cb67cae270c95bb963cafca0597d65da77de3b0ada53bb33b721
- status: PASS_BINDING_IDENTIFIABILITY_AUDIT_NO_GRADIENT
- localization: FIELD_POOLING_BOTTLENECK_SUPPORTED_TOKEN_LATE_INTERACTION_RECOVERS_BINDING
- TEST unopened
- no optimizer, gradient, or GPU.

DEV binding accuracy:
- lexical token overlap: 1.000 all-edge / 1.000 same-relation
- final pooled field semantic: 0.8056 / 0.8889
- structured parent field state: 0.2083 / 0.2361
- adapter field weight: 0.3194 / 0.3750
- parent graph field weight: 0.1111 / 0.3542
- token-level late interaction: 1.000 / 1.000 at every audited semantic layer 0..16
- final semantic token layer (16): 1.000 / 1.000
- DEV best-margin layer was 2, but it is not hard-coded.

Interpretation:
- semantic token states retain the edge identity signal perfectly on DEV;
- pooled semantics retain much of it;
- the query-agnostic structured path destroys most query-to-field binding geometry;
- another router over structured graph states is therefore not the justified next move.

## Architecture decision

Branch:
alice-eipm-v1-dual-view-late-interaction-binding

Current static-contract head:
b736f55363254eb887664f33c35ac1a144a0b08e

CI:
35490590850 — SUCCESS

Authoritative state:
configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.33.json

Candidate architecture:
DualViewLateInteractionBridge
DualViewLateInteractionEvidenceGraphEncoder

Single causal change:
separate fine-grained edge identity binding from structured graph reasoning.

Binding channel:
- final semantic query token states;
- final semantic field token states;
- query-to-field MaxSim late interaction;
- edge score = mean(source field score, target field score);
- source/target reversal cannot change pair identity;
- edge rank itself is not learned;
- only the binding confidence scale may be learned later.

Reasoning channel:
- existing frozen structured-state/evidence-view/dual-endpoint graph parent;
- existing relation-conditioned multi-layer query specialist;
- independent source/target residual proposals.

Routing factorization:
P(no-op) = 1 - P(specialist)
P(edge=e) = P(specialist) * P(e | specialist)

P(e | specialist) is the semantic late-interaction edge distribution.
The learned route problem is only parent-vs-specialist activation.

Fail-closed behavior:
- padded fields may have no content tokens;
- valid fields must have content tokens;
- if no eligible directional edge exists, specialist probability is forced to zero and no-op is one.

Exact parent initialization:
- source/target residual readouts are zero-initialized;
- qualification must prove exact parent field weights and pooled state.

## Qualification only

Script:
scripts/eipm/n0/qualify_n0_v02_dual_view_late_interaction_v0_1.py

Runner:
scripts/eipm/n0/run_n0_v02_qualify_dual_view_late_interaction_v0_1.sh

Required runtime qualification checks:
- 1.0 DEV all-four edge conditional top1
- 1.0 DEV same-relation conditional top1
- 1.0 source-role and target-role edge top1
- 1.0 every relation-family edge top1
- edge reversal invariance
- edge permutation equivariance
- route probabilities sum to one
- specialist activation initializes exactly at 0.5 when an eligible edge exists
- exact parent field weights
- exact parent pooled state
- parent parameters unchanged

Closed:
- optimizer
- gradient
- GPU training
- setwise replacement run
- TEST
- frozen challenge
- scale
- semantic retraining
- graph-parent retraining
- private identity gradient
- production promotion

If qualification passes, make a separate scientific decision about one bounded residual + specialist-activation training experiment. Do not auto-authorize training.
