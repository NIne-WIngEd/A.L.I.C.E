# N0 Query–Edge Routed Specialist — One P100 Run Ready

**Date:** 2026-09-20

## Runtime qualification evidence

Magnolia CPU/no-gradient qualification passed from:

`c6170597e2b55cc378c5d8cf1560ca87802ea085`

Qualification receipt SHA-256:

`b1e073fc6226ac3f3aa88ed10d4ed31059b6e27b0338aa7738c686239f33624f`

Status:

`PASS_QUERY_EDGE_BRIDGE_NO_GRADIENT_RUNTIME_CONTRACT`

Validated:

- parent parameters exactly unchanged;
- parent field weights exact;
- parent pooled state exact;
- specialist gate exactly zero at initialization;
- same-relation edges produce distinct edge queries;
- actual edge content changes query-token attention;
- no forced source/target antisymmetry;
- future trainable scope is query-edge bridge only;
- no optimizer or gradient was used.

Observed edge-specific separation:

- edge 0 vs 1 query-state max delta: `0.4681481122970581`
- edge 1 vs 2 query-state max delta: `0.5135583877563477`
- edge 0 vs 1 token-attention max delta: `0.021744057536125183`

Frozen study artifacts:

- causal curriculum: `c3618a6022519af2a6a140f8b571281c68735fd1412c3ffa904979e9bbea2dcb`
- curriculum manifest: `0d05f1dcd7d11cc1defdd0a4112c10e3b1d7febfaabf97fe611a854d6905b99a`
- preservation TRAIN anchors: `f5b3f438f776216f6dbd557a8c785aac70af38d256e3f7cf94f0f71a43f53e94`

## Training decision

Current experiment branch:

`alice-eipm-v1-query-edge-cross-attention-bridge`

Exact training frontier:

`cf15e2b13a5ca4d22a7070e54c95460959206c33`

Training contract workflow:

`35484969093`

Conclusion:

`SUCCESS`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.24.json`

Training decision:

`configs/eipm/n0/n0_v02_query_edge_training_decision_v0_1.json`

Exactly one P100 experiment is authorized.

## Training objective

The qualified architecture is unchanged.

Trainable:

`query_edge_bridge.*`

Frozen:

- semantic backbone
- structured-state encoder
- evidence-view adapter
- proven DualEndpoint graph parent

Gradient-bearing new-task data:

- fresh query-edge causal TRAIN only

Gradient-bearing preservation data:

- ordinary graph TRAIN anchors only
- endpoint-role TRAIN anchors only

Never gradient-bearing:

- ordinary DEV
- endpoint DEV
- query-edge causal DEV
- query-edge causal TEST
- frozen challenge
- private identity data

Per training step:

1. causal evidence-field selection CE;
2. causal target-margin loss, weight `0.25`;
3. explicit same-relation relevant-edge routing loss, weight `0.50`;
4. parent-output KL distillation on ordinary TRAIN anchors;
5. parent-output KL distillation on endpoint TRAIN anchors;
6. irrelevant-edge specialist-gate no-op penalty, weight `0.05`.

The two parent-distillation lanes share total preservation weight `1.0` equally.

No PCGrad or other gradient surgery is present in this experiment.

Reason: test the simpler causal hypothesis first. The architecture now has the right query-edge binding substrate; routing supervision addresses route allocation directly, while teacher preservation addresses old-skill interference directly.

## Frontier rationale

Recent MoE analyses show that downstream task loss alone does not ensure useful routing. Router-only corrections can recover performance while experts remain frozen, which supports treating routing as its own learned capability.

Recent expert-router coupling work likewise adds explicit auxiliary supervision to align routing decisions with expert behavior.

The proven graph parent remains frozen, matching the broader pattern of protecting a robust shared expert/reasoner while training a specialist interaction path.

This does not turn N0 into a generic MoE. These are transferable design principles only.

## Bounded operating budget

- one P100;
- 200 steps;
- checkpoint every 40;
- learning rate `1.5e-4`;
- weight decay `0.02`;
- warmup 12;
- causal quad batch 4;
- preservation batch 16 per lane.

These are experiment operating values, not product ceilings.

## Checkpoint eligibility

A checkpoint is eligible for a separate heldout decision only if:

- frozen parent preservation policy passes;
- no relation family row accuracy regresses against initial parent;
- overall row accuracy strictly improves;
- overall quad accuracy strictly improves;
- worst-family quad accuracy does not regress;
- mean target margin strictly improves;
- routing accuracy exceeds 0.5;
- worst-family routing accuracy is at least 1/3;
- routing accuracy strictly improves from initialization.

Selection prioritizes:

1. preservation;
2. worst-family routing;
3. worst-family quad accuracy;
4. overall routing;
5. overall quad accuracy;
6. row accuracy;
7. target margin;
8. earlier checkpoint.

## Still closed

A successful DEV checkpoint does not automatically open anything.

Still unauthorized:

- causal test;
- frozen challenge;
- scale;
- semantic retraining;
- parent graph retraining;
- private identity gradient;
- production promotion;
- N0 completion.

## Failure doctrine

If the one P100 experiment genuinely fails, stop.

Do not modify learning rate, loss weights, width, routing threshold, or training length as an automatic repair.

Reassess the training objective and query-edge boundary from the larger N0 purpose, consult frontier evidence, then make one causal change only if justified.

Preserve the failed training directory.
