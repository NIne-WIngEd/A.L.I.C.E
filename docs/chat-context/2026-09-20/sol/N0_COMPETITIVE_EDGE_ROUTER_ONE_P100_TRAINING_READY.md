# N0 Competitive Edge Router — One P100 Training Ready

**Date:** 2026-09-20

## CPU qualification

Competitive-router CPU/no-gradient qualification passed from:

`93901b4ea0e3a9773e59047181414e507d3ed904`

Qualification receipt SHA-256:

`ba2dcf640ffd2572ba0a1177c571861aff1838784ab19c0f8166728f0dc59f37`

Status:

`PASS_COMPETITIVE_EDGE_ROUTER_NO_GRADIENT_RUNTIME_CONTRACT`

Validated:

- parent parameters exactly unchanged;
- exact parent field weights;
- exact parent pooled state;
- route probabilities sum to one;
- explicit parent/no-op route;
- all active directed edges compete;
- same-relation edges remain distinct;
- edge content still changes query-token attention;
- route probability is exact runtime source contribution control;
- route probability is exact runtime target contribution control;
- no proxy dot-product router;
- no independent per-edge tanh gate;
- CPU only, no optimizer, no gradient, no GPU.

Observed edge-specific deltas:

- same-relation query max delta: `0.4772298336029053`
- same-relation token-attention max delta: `0.01974017173051834`

Future trainable scope:

`competitive_query_edge_bridge_only`

Future trainable parameter count:

`8,832,773`

## Training objective decision

Decision config:

`configs/eipm/n0/n0_v02_competitive_edge_router_training_decision_v0_1.json`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.28.json`

Exactly one P100 experiment is authorized.

### Causal TRAIN

- evidence-field selection CE;
- target-margin loss weight `0.25`;
- direct actual-route CE weight `1.0`;
- actual route target is `relevant_edge_index + 1`;
- route class `0` is parent/no-op.

The route CE uses `route_logits_with_noop`, which is the exact runtime distribution whose edge probabilities scale source/target residuals.

### Preservation TRAIN anchors

Ordinary and endpoint TRAIN anchors each provide:

- frozen-parent field-weight KL;
- direct no-op route CE with target class `0`.

The two preservation lanes share total preservation and no-op weights equally.

### Deliberately absent

- proxy routing score;
- same-relation-only route objective;
- irrelevant-edge gate penalty;
- load-balancing loss;
- PCGrad or gradient surgery;
- route-probability detachment.

Reason: test the corrected runtime control plane directly before adding optimization mechanisms.

## Current training frontier

Branch:

`alice-eipm-v1-query-edge-competitive-router`

Exact head:

`a6b263be31d822f8781cefadf450b63dcae00acd`

Training contract CI:

`35486665533`

Conclusion:

`SUCCESS`

CI verified:

- exact qualification receipt and source revision;
- exact runtime route supervision;
- explicit parent/no-op preservation supervision;
- no rejected proxy/gate mechanisms;
- frozen parent trainable scope;
- five-route causal readiness floor;
- exact artifact lineage;
- output directory claimed before GPU work;
- one bounded P100 wrapper.

## Operating budget

- one P100;
- 200 steps;
- checkpoints every 40;
- learning rate `1.5e-4`;
- weight decay `0.02`;
- warmup 12;
- causal quad batch 4;
- preservation batch 16.

These are operating values, not product ceilings.

## Checkpoint eligibility

A checkpoint is eligible only if:

- frozen parent preservation policy passes;
- no relation-family row accuracy regresses vs zero-init parent baseline;
- overall row accuracy strictly improves;
- overall quad accuracy strictly improves;
- worst-family quad accuracy does not regress;
- mean target margin strictly improves;
- actual route accuracy exceeds `0.5`;
- worst-family actual route accuracy is strictly above uniform five-route chance `0.2`;
- actual route accuracy strictly improves over initialization;
- mean target-route probability strictly improves over initialization.

Selection prioritizes:

1. preservation;
2. worst-family actual-route accuracy;
3. worst-family quad accuracy;
4. overall actual-route accuracy;
5. quad accuracy;
6. row accuracy;
7. target margin;
8. earlier checkpoint.

## Still closed

Even a passing DEV checkpoint does not automatically open anything.

Still unauthorized:

- causal TEST;
- frozen challenge;
- scale;
- semantic retraining;
- parent graph retraining;
- private identity gradient;
- production promotion;
- N0 completion.

## Failure doctrine

If this one P100 run genuinely fails, stop.

Do not tune routing weight, no-op weight, learning rate, steps, thresholds, width, or batch size automatically.

Reassess whether the issue is route representation, route/expert optimization interference, residual expressivity, or curriculum semantics from the larger N0 purpose before one next causal change.
