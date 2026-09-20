# N0 Query–Edge Binding Identifiability Audit — Router Repair Loop Halted

**Date:** 2026-09-20

## Why the current route was stopped

The latest competitive-router experiment is valid model evidence:

- result SHA-256: `08bef659779c2e6fcfb0d60ef209d9f6c7f03b0198bcf370e097e278f5436328`
- status: `FAIL_COMPETITIVE_EDGE_ROUTER_CAUSAL_DEV_OR_PRESERVATION_STOP_NO_HELDOUT_NO_AUTOMATIC_HOTFIX`
- no eligible checkpoint;
- TEST unopened;
- frozen challenge unopened;
- parent graph unchanged.

By step 200 the model had learned the old-task no-op decision almost perfectly:

- ordinary no-op top-1: `1.0`
- endpoint no-op top-1: `1.0`
- ordinary mean no-op probability: `0.9983423650`
- endpoint mean no-op probability: `0.9996101552`
- all frozen parent preservation checks passed.

But causal edge selection remained weak:

- routing accuracy: `0.2361111111`
- family-min routing accuracy: `0.0833333333`
- mean target-edge route probability: `0.2368120873`
- causal no-op probability: `0.1245869685`.

Once no-op is mostly suppressed, the remaining probability mass over four active edges is approximately `0.8754`; uniform edge mass is approximately `0.2189`. The target receives only `0.2368` on average. The learned router is therefore only slightly above uniform probability allocation and below the 0.25 top-1 chance rate among four edges.

This is materially different from failing to distinguish parent/no-op from specialist work.

## Why setwise routing is not yet justified

A setwise router was implemented and CPU-qualified:

- branch: `alice-eipm-v1-query-edge-setwise-router`
- qualified revision: `7a8bc6b63a9f32095ba3fe4ed27db5f62cf5e061`
- qualification receipt: `4b1591e1eec5c450a872264a1bdf1130def5a29e443c8c35e974ae05f2861666`
- all edges jointly contextualized;
- permutation equivariance passed;
- nonlocal competitor influence passed.

That qualification proves routing mechanics and cross-edge context. It does not prove that the correct edge is identifiable from the representation supplied to the router.

Setwise P100 job `575908` failed before optimization because the exact training executable retained a stale competitive-qualification status guard.

This is not model evidence.

The subsequent exact CPU trainer preflight is infrastructure/boundary validation. It should not become the next model program.

## Deeper representation asymmetry

The query-edge curriculum asks for one specific subject + attribute pair among multiple graph edges.

The query side receives:

- 17 layers of token-level semantic hidden states;
- token attention masks;
- relation-conditioned late interaction.

The field side receives:

- each evidence sentence encoded once;
- final-layer mean pooling via `AliceN0V02Model.encode()`;
- one 640-dimensional pooled field vector;
- structured-state transformation;
- evidence adapter;
- graph transformation.

In this causal curriculum the structured metadata channels do not identify the pair:

- field type IDs are constant;
- provenance IDs are constant;
- relation-role IDs are constant;
- temporal-scope IDs are constant;
- confidence is effectively constant.

Therefore the only signal telling the model that query `Alto + worker mode` belongs to one pair rather than another must survive the pooled field-semantic vector.

That assumption was never directly validated.

It is especially important because the semantic model itself documents token representations as first-class and its pooled output as a compatibility readout rather than a fusion capability ceiling.

Earlier N0 audits already demonstrated that token-level late interaction retained directional information that mean pooling lost.

## Tokenizer is not the suspected bottleneck

N0 v0.2.1 uses a 48K byte-complete BPE tokenizer.

Its governed audit reported zero unknown tokens and zero round-trip failures.

Novel synthetic names remain representable.

The open question is semantic/compositional binding after field-side compression, not whether the tokenizer can spell the names.

## Current localization question

Before another router, optimizer, or P100 run:

**Can the relevant graph edge be identified from the frozen representations at all?**

The new audit compares:

1. discriminative raw token-ID overlap;
2. final mean-pooled field semantics;
3. structured parent field states;
4. query-conditioned adapter field weights;
5. frozen parent graph field weights;
6. token-level query↔field MaxSim late interaction across all 17 semantic layers on DEV.

It reports separately:

- top-1 among all four edges;
- top-1 among the three same-relation edges;
- relevant-edge margins;
- relation-family breakdown;
- 0.25 four-edge chance;
- 1/3 same-relation chance;
- best token layer;
- pooled-to-token identifiability gap.

No model probe is trained.

No optimizer.

No gradient.

No GPU.

TEST remains closed.

## Current diagnostic branch

`alice-eipm-v1-query-edge-binding-identifiability-audit`

Exact head:

`0610c10ebb74b1518f160ffba90f9acd57e27672`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.32.json`

CI:

`35490048679 — SUCCESS`

## Audit input reuse

The interrupted exact setwise trainer preflight already created:

`query-edge-setwise-router-v0.1/training-preflight-v0.2/setwise_query_edge_train_dev_hidden_cache.pt`

Observed size:

`827461729` bytes.

The identifiability audit reuses this cache. It does not rerun the expensive causal query/field compilation.

Only DEV field texts are re-encoded token-wise for the late-interaction comparison.

## Decision logic after the audit

### Case A — pooled bottleneck

If raw text is strongly identifiable, pooled field representations are weak, and token-level field late interaction recovers the correct edge:

> Replace the pooled field-side fusion bottleneck with a symmetric token-level / late-interaction field binding interface.

Do not add router complexity first.

### Case B — router/optimization

If pooled field representations already identify the relevant edge reliably but the trained competitive router remains near chance:

> Reopen router/optimization. Setwise arbitration may then be justified.

### Case C — semantic substrate

If raw text is identifiable but even token-level semantic late interaction remains weak:

> Reopen the semantic objective / entity-binding capability. Do not add routers.

### Case D — curriculum

If raw token-ID overlap itself cannot identify the relevant pair cleanly:

> The curriculum semantics are the problem. Repair the study, not the model.

## Closed until audit result

- setwise replacement GPU;
- router variants;
- loss-weight tuning;
- learning-rate tuning;
- more training steps;
- TEST;
- frozen challenge;
- scale;
- semantic retraining;
- parent graph retraining;
- private identity gradient;
- production promotion.

This is the anti-MC10 boundary: validation may guard the next model action, but it may not become the work itself.
