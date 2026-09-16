# N0 Adaptive Multi-View Latent Pool v0.1 Failure and v0.2 Repair — 2026-09-16

## Result that must remain historical

Magnolia job 575670 completed successfully at the infrastructure level but the model-development result was:

`FAIL_NO_LATENT_POOL_CHECKPOINT_CLEARED_TRAINING_DEV_GATE`

All three v0.1 checkpoints (120/240/360) failed the development gate because the learned slot set collapsed. Semantic readout became strong, but `mean_max_offdiag_slot_cosine` reached approximately 1.0. No checkpoint is ratified. No private identity gradient occurred.

## Root cause audit

Two interacting causes were found.

### 1. Objective bug: duplicate slots were explicitly rewarded

The v0.1 semantic set loss used:

`temperature * logsumexp(slot_target_cosine / temperature)`

without subtracting a slot-count normalization. With 12 slots and temperature 0.12, making all slots perfect copies adds `0.12 * log(12) ~= 0.298` to the smooth maximum. At the configured 0.45 semantic weight this can improve the weighted objective by about 0.134, while the old near-duplicate diversity penalty at cosine 1 contributes only about 0.0039 at its configured weight. The training objective therefore strongly favored collapse.

This is an objective defect, not evidence that 12 slots are insufficient.

### 2. Architecture symmetry: slots did not compete for evidence

The v0.1 cross-attention normalized independently over input tokens for every slot. Every slot could therefore attend to the same evidence. Slot self-attention then provided another path for representations to homogenize.

The successor uses competitive Slot-Attention-style evidence allocation: each input token first allocates ownership across slots, then each slot normalizes its assigned evidence for aggregation. Slot-to-slot exchange remains available through a learned gate but starts gated down.

## v0.2 correction

The successor is a fresh latent initialization over the same ratified upstream stack. Failed v0.1 latent weights are not resumed.

Architecture:
- `src/alice_personality/n0/adaptive_multi_view_latent_pool_v0_2.py`
- competitive cross-attention across slots;
- exact source + contextualized fusion channels retained;
- 12 slots / 640 width / 3 layers / 10 heads remains the current checkpoint shape only, not a permanent limit;
- no fixed slot meanings or trait taxonomy;
- no hard parameter/slot/view ceiling.

Objective:
- `src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives_v0_2.py`
- hard best-slot semantic set requirement, so duplicate target slots receive no extra semantic reward;
- pooled convenience-readout alignment;
- near-duplicate penalty without requiring orthogonal personality traits;
- permutation-free view-specialization objective;
- view and source/context channel coverage;
- decisive-source counterfactual sensitivity only on genuinely decisive rows;
- no exact routing-percentage supervision.

## Development gate philosophy

The v0.2 training/dev gate is an anti-collapse capability gate, not final ratification and not a personality ontology. It checks semantic usefulness, minimum evidence/channel reachability, zero missing-view attention, nontrivial effective slot rank, and nontrivial view specialization.

A passing development checkpoint still requires a new untouched latent-pool challenge before ratification.

## Capacity policy

Do not increase model size merely to compensate for the v0.1 objective bug. If v0.2 still shows a real capability bottleneck after the corrected architecture/objective, scale slots, width, depth, heads, views, or upstream adaptation as required. Capability and eventual personality fidelity remain primary; efficiency is secondary.

Private identity gradient remains CLOSED.
