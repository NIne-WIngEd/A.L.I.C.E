# N0 fusion ratified; adaptive multi-view latent pooling opened — 2026-09-16

## Fusion confirmatory v0.3 result

The unchanged source-anchored fusion repair candidate passed the corrected untouched v0.3 behavioral-constraint challenge.

Selected fusion checkpoint:
- `cross-context-fusion-repair-training-v0.2/repair-step-00000240`
- SHA-256 `4d51494beb788f74ddc03590da05ea00f0e36574294649c2cd5d438f9472e577`
- 116,981,764 fusion parameters (checkpoint fact, not capacity ceiling)
- total fusion update count 480

Challenge v0.3:
- 160 rows / 20 families
- candidate was preselected before challenge
- candidate weights unchanged after v0.2
- no checkpoint selection on challenge
- challenge rows not used for training
- no gradient during evaluation
- routing validation uses behavioral constraints, not exact hand-authored soft percentages

Pass metrics:
- routing constraint pass rate: 1.0
- worst-family routing constraint pass rate: 1.0
- singleton required-top accuracy: 1.0
- fused semantic cosine: 0.949729336053133
- worst-family fused semantic cosine: 0.9197454825043678
- disagreement geometry MAE: 0.03407624921528622
- mean routing constraint violation: 0.0
- missing-view max weight: 0.0
- source-summary anchor max error: 0.0
- source-token anchor max error: 0.0

The v0.2 result remains an immutable historical failure under its frozen exact-distribution validator. It was not retroactively changed to PASS. Post-result audit established that its exact hand-authored routing percentages overconstrained an internal latent mechanism. The model remained unchanged until v0.3.

## Ratification

Fusion is now ratified as the public N0 fusion base, not full EIPM or production promotion.

Manifest:
`configs/eipm/n0/n0_v02_cross_context_fusion_ratification_v0.1.json`

Ratified public stack through fusion:
- semantic: 136,594,435
- structured: 1,656,064
- evidence adapter: 16,002,561
- evidence graph: 6,664,963
- fusion: 116,981,764
- descriptive total through fusion: 277,899,787

Every count is a checkpoint-lineage measurement, not a successor ceiling.

## Current stage: adaptive multi-view latent pooling

The next actual model component is implemented as a full-scale adaptive latent-slot readout, not a reduced pilot and not a single `[CLS]` bottleneck.

Files:
- `src/alice_personality/n0/adaptive_multi_view_latent_pool.py`
- `src/alice_personality/n0/adaptive_multi_view_latent_pool_objectives.py`
- `configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_v0.1.json`
- `configs/eipm/n0/n0_v02_adaptive_multi_view_latent_pool_training_v0.1.json`
- `scripts/eipm/n0/train_n0_v02_adaptive_multi_view_latent_pool_full_scale.py`
- `scripts/eipm/n0/run_n0_v02_adaptive_multi_view_latent_pool_prepare.sh`
- `scripts/eipm/n0/run_n0_v02_adaptive_multi_view_latent_pool_train.sh`
- `scripts/eipm/n0/magnolia_p100_n0_v02_adaptive_multi_view_latent_pool_train.sbatch`

Current instantiated topology:
- semantic/latent width 640
- 12 learned latent slots
- 3 latent layers
- 10 heads
- FFN 2560
- 3 current public views
- two channels per view: exact source tokens + contextualized fusion tokens

Important: 12 slots is a current checkpoint shape only. There is no slot-count ceiling, view-count ceiling, or hard parameter ceiling. Slots are not pre-labeled traits. Their semantics must emerge and be measured.

Fusion routing weights enter as metadata features only. They are not treated as calibrated probabilities or exact training targets.

Initial latent training freezes the ratified fusion and upstream parents only to isolate the new component. This is a stage control, not a permanent freeze. Progressive upstream unfreezing remains allowed if measured capability/fidelity requires it.

## Training source and objectives

The first full-scale latent run uses the existing 512-row public fusion-repair training tranche only as source material:
- 384 train
- 128 dev
- old `target_view_distribution` values are ignored as supervision
- frozen challenge v0.1/v0.2/v0.3 rows remain excluded

Objectives:
- multi-slot semantic target alignment
- pooled convenience-readout alignment
- anti-collapse slot diversity
- available-view representation coverage
- source/contextualized channel reachability
- target-grounded decisive-view counterfactual sensitivity

Coverage is not authority: stale or contradictory views may still deserve latent representation so historical reasoning and uncertainty can use them.

## Immediate execution

1. Pull the current `alice-eipm-v1-build` head.
2. Run CPU-only `run_n0_v02_adaptive_multi_view_latent_pool_prepare.sh` through the udocker wrapper.
3. Do not change Git after prep.
4. Submit `magnolia_p100_n0_v02_adaptive_multi_view_latent_pool_train.sbatch` on one P100.
5. Inspect `adaptive_multi_view_latent_pool_comparison.json`.
6. Only if a training/dev checkpoint clears the full-scale gate do we create a new untouched latent-pool frozen challenge.

No private identity gradient is authorized. N0 remains incomplete.
