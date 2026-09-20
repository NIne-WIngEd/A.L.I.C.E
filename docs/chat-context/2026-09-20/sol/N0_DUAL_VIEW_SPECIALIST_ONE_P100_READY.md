# N0 Dual-View Specialist — One P100 Training Ready

**Date:** 2026-09-20

## Qualification result

Corrected CPU/no-gradient qualification:

- status: `PASS_DUAL_VIEW_LATE_INTERACTION_NO_GRADIENT_RUNTIME_CONTRACT`
- source revision: `edcee7dd7a53bac158793b805a29cfaa0c1b11b0`
- receipt SHA-256: `714f06d4939abad2b816a7e3ffb5c5fa10112129ef0b520fd838f2d182286278`
- DEV conditional edge all-four top1: `1.0`
- DEV same-relation top1: `1.0`
- source-role top1: `1.0`
- target-role top1: `1.0`
- every relation family top1: `1.0`
- edge reversal max delta: `1.7881393432617188e-07`
- permutation max delta: `5.960464477539063e-08`
- route-sum max error: `1.1920928955078125e-07`
- specialist probability initialization delta from 0.5: `0.0`
- exact parent field weights: true
- exact parent pooled state: true
- parent parameters unchanged: true
- no optimizer / gradient / GPU / TEST.

The edge-identification problem is therefore considered solved for this diagnostic boundary before training.

## Training question

The one authorized P100 experiment tests only:

1. can the model learn when to invoke the specialist rather than the frozen parent?
2. can the relation-conditioned specialist learn the correct independent source/target residual needed to select the right evidence endpoint?

It does **not** test or train edge identity.

## Frozen binding channel

The dual-view semantic late-interaction channel remains exactly as qualified.

The parameter:

`dual_view_query_edge_bridge.binding_logit_scale`

is explicitly excluded from the optimizer.

There is no conditional-edge CE or routing loss.

This keeps edge ordering fixed so a failure cannot be blamed on a moving binding channel.

## Trainable scope

Trainable:
- dual-view bridge relation embeddings/projections;
- relation-conditioned token/layer interaction;
- fusion;
- parent-vs-specialist activation;
- independent source and target residual readouts.

Frozen:
- binding logit scale;
- semantic backbone;
- structured state;
- evidence-view adapter;
- dual-endpoint parent graph.

## Objective

Causal TRAIN:
- field-selection CE;
- target margin weight `0.25`;
- BCE on the actual runtime `specialist_logit`, target `1`.

Preservation TRAIN anchors:
- frozen-parent field-weight KL;
- BCE on that same runtime `specialist_logit`, target `0`;
- ordinary and endpoint lanes share preservation weight equally.

Absent:
- conditional-edge binding loss;
- flat route CE;
- proxy router;
- load balancing;
- irrelevant-edge penalty;
- PCGrad / gradient surgery;
- binding-scale tuning.

## Eligibility

A checkpoint must simultaneously keep:

- all-four conditional edge top1 = `1.0`;
- same-relation conditional edge top1 = `1.0`;
- frozen parent preservation policy pass;
- no relation-family row regression vs zero-init parent;
- row accuracy improve;
- quad accuracy improve;
- worst-family quad not regress;
- target margin improve;
- causal specialist top1 = `1.0`;
- every relation-family specialist top1 = `1.0`;
- ordinary no-op top1 = `1.0`;
- endpoint no-op top1 = `1.0`;
- causal specialist probability improve over initial `0.5`;
- preservation no-op probability improve over initial `0.5`.

## Operating budget

- one P100 run;
- 200 steps;
- checkpoints 40 / 80 / 120 / 160 / 200;
- LR `1.5e-4`;
- weight decay `0.02`;
- warmup 12;
- causal quad batch 4;
- preservation batch 16.

Operating values are not product ceilings.

## Current training frontier

Branch:

`alice-eipm-v1-dual-view-late-interaction-binding`

Exact head:

`57b085fd815cc784ae4090b8db7b619adec57040`

Training contract CI:

`35492036440 — SUCCESS`

Authoritative state:

`configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.35.json`

Training decision:

`configs/eipm/n0/n0_v02_dual_view_training_decision_v0_1.json`

Trainer:

`scripts/eipm/n0/train_n0_v02_dual_view_specialist_v0_1.py`

Runner:

`scripts/eipm/n0/run_n0_v02_dual_view_specialist_training_v0_1.sh`

P100 wrapper:

`scripts/eipm/n0/magnolia_p100_n0_v02_dual_view_specialist_training_v0_1.sbatch`

## Still closed

Even a passing DEV checkpoint does not automatically open:
- causal TEST;
- frozen challenge;
- scale;
- semantic retraining;
- graph-parent retraining;
- private identity gradient;
- production promotion;
- N0 completion.

If this experiment genuinely fails, do not tune binding scale, LR, steps, weights, or thresholds. Localize the failure specifically between specialist activation and directional residual reasoning.
