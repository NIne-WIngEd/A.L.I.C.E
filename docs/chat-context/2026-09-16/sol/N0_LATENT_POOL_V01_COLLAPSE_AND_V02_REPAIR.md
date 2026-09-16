# N0 Adaptive Latent Pool v0.1 Collapse + v0.2 Repair — 2026-09-16

## Owner warning

Owner explicitly said this is an important step and asked for care. Do not accept superficially high semantic metrics if the multi-slot representation itself collapses.

## Observed v0.1 result

Magnolia job 575670 completed infrastructure-clean (`COMPLETED`, exit `0:0`, ~1m54s) but model development FAILED:

`FAIL_NO_LATENT_POOL_CHECKPOINT_CLEARED_TRAINING_DEV_GATE`

All saved checkpoints 120 / 240 / 360 had `training_gate_pass=false`; winner=null.

Semantic readout became very strong, but every trained checkpoint collapsed the multi-slot state:
- step120 pooled semantic ~0.9552
- step240 pooled semantic ~0.9725
- step360 pooled semantic ~0.9755
- trained `mean_max_offdiag_slot_cosine` ~= 1.0
- slot-weight entropy ~= 1.0
- missing-view attention remained 0

No checkpoint is ratified. Private identity gradient remained closed.

## Root cause

### Objective bug

v0.1 used an unnormalized smooth max:

`temperature * logsumexp(cosine / temperature)`

with 12 slots and temperature 0.12. If all 12 slots become perfect target copies, the smooth maximum becomes approximately:

`1 + 0.12 * ln(12) ~= 1.29819`

Thus the semantic loss becomes negative and rewards duplicate target slots. At the configured semantic weight 0.45, the duplicate-slot advantage is about 0.134. The old diversity penalty at cosine 1 and margin 0.82 contributes only about 0.0039 after its 0.12 weight. Collapse was therefore roughly 34x more rewarding than its penalty.

### Architecture symmetry

v0.1 standard cross-attention normalized independently over bank tokens for every slot. Every slot could therefore consume the same evidence independently. Unrestricted slot self-attention then provided another homogenization path.

The failure does NOT prove 12 slots, 640 width, or 3 layers are too small. Do not scale merely to compensate for this objective/interaction bug.

## Failure-driven successor

### Model
`src/alice_personality/n0/adaptive_multi_view_latent_pool_v0_2.py`

- fresh latent initialization; failed v0.1 latent weights are NOT reused;
- competitive Slot-Attention-style cross-attention: bank token ownership is normalized across slots first, then each slot normalizes its assigned evidence over tokens;
- slot-to-slot exchange remains learnable but begins gated down (`logit=-2`) so evidence specialization can form before unrestricted mixing;
- exact source + contextualized fusion channels retained;
- view identity / channel identity / reliability / ratified fusion routing retained as features;
- current shape remains 12 slots / width640 / 3 layers / 10 heads / FFN2560 only as checkpoint topology, with no hard slot/view/parameter ceiling;
- slots remain unnamed and permutation-free, never fixed personality traits.

### Corrected objective
`src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives_v0_2.py`

- hard best-slot semantic set requirement; duplicate target slots give no extra semantic reward;
- pooled convenience-readout alignment;
- near-duplicate penalty without requiring orthogonal personality traits;
- permutation-free mutual-information-style view specialization;
- available-view coverage;
- exact-source + contextualized channel coverage;
- decisive-source counterfactual sensitivity;
- centered effective-rank metric after removing common slot mode;
- no exact routing percentage supervision.

### Counterfactual scope correction

The first v0.2 trainer draft would have accidentally applied a counterfactual margin to non-decisive rows by replacing only decisive rows and leaving copies elsewhere. This was caught before execution.

Authoritative execution wrapper:
`scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_v0_2_1_full_scale.py`

It applies the counterfactual margin only on genuinely decisive rows.

## Reuse policy

The upstream parent cache and ratified fusion cache produced by job 575670 are immutable products of already-ratified upstream models. The v0.2 prep gate verifies their hashes against the failed-run checkpoint lineage and the ratified fusion SHA before reuse.

Do NOT recompute upstream caches unless the hash/lineage gate fails.
Do NOT reuse any v0.1 latent weights.

## v0.2 training/dev gate

This is a development anti-collapse/capability gate, NOT final ratification and NOT a trait taxonomy.

Current frozen development requirements:
- best slot semantic cosine >= 0.90
- pooled semantic cosine >= 0.90
- family-min best slot >= 0.86
- family-min pooled >= 0.84
- mean pairwise off-diagonal slot cosine <= 0.97
- mean centered slot effective rank >= 0.10
- mean view specialization >= 0.05
- available-view best-slot coverage >= 0.12
- source/context channel coverage >= 0.10
- missing-view attention <= 1e-6

If a checkpoint clears this, it is only eligible for a NEW untouched latent-pool challenge.

## Capacity doctrine

Do not scale capacity simply because v0.1 failed; v0.1 had a mathematical objective defect. If corrected v0.2 still shows a measured capacity/fidelity bottleneck, then scale slots, width, depth, heads, views, or upstream adaptation as needed. There is no rigid ceiling.

Private identity/N1 gradient remains CLOSED.

## Next execution

1. Pull latest `alice-eipm-v1-build` using `git checkout` (Magnolia git does not support `git switch`).
2. Run `scripts/eipm/n0/run_n0_v02_adaptive_multi_view_latent_pool_v0_2_prepare.sh` CPU-only.
3. Inspect actual test/receipt output. Do not proceed on failure.
4. Do not change revision after prep PASS.
5. Submit `scripts/eipm/n0/magnolia_p100_n0_v02_adaptive_multi_view_latent_pool_v0_2_train.sbatch`.
6. Evaluate 120/240/360 development checkpoints. No automatic ratification.
