# N0 Latent Pool v0.1 Frozen Challenge Failure and Scale Adequacy Audit — 2026-09-16

## Decision

Do **not** scale or retrain the latent pool from the v0.1 frozen-challenge failure.

The selected competitive latent pool remains step 360 with SHA-256 `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`.

The v0.1 frozen challenge remains an immutable historical FAIL. It is not retroactively reclassified as PASS.

## Why the failure is not evidence of undercapacity

The v0.1 frozen challenge passed every ratification gate except `counterfactual_family_min_drop`.

Observed challenge metrics included:

- best-slot semantic cosine: 0.9670817155390978
- worst-family best-slot semantic cosine: 0.9376877695322037
- pooled semantic cosine: 0.9579608980566263
- worst-family pooled semantic cosine: 0.938840813934803
- mean pairwise off-diagonal slot cosine: 0.43672637939453124
- centered slot effective rank: 0.20203436594456434
- disagreement-weighted specialization: 0.9678216249418219
- source-view semantic recoverability: 0.8958158196881414
- missing-view attention max: 0.0

The only failed family was the post-fusion counterfactual for `missing_structured_evidence`, whose mean target drop was -0.0034492164850234985.

## Counterfactual intervention bug

The v0.1 evaluator first built all parent representations, ran the ratified cross-context fusion stage, and cached contextualized view streams. Only **after fusion** did the counterfactual remove one view from the latent-pool masks.

That is not a valid causal intervention for a source that may already have influenced other contextualized streams through fusion. The ratified fusion architecture is intentionally cross-contextual: evidence can be transferred into semantic or structured contextualized representations before latent pooling.

Therefore:

> Removing one view only at the latent-pool boundary does not mean removing that source's information from the full stack.

A near-zero latent-boundary target drop can indicate successful redundant transfer by fusion, not failure to use the source.

## Corrected causal test

The successor confirmatory challenge must intervene **before fusion**:

1. start from an untouched public synthetic challenge row;
2. remove the uniquely necessary source view from `view_available` and set its reliability to zero;
3. rebuild the parent cache from that counterfactual row;
4. rerun the unchanged ratified fusion model;
5. run the unchanged selected latent-pool checkpoint;
6. compare the full-stack target support before and after the pre-fusion intervention.

The v0.1 challenge rows are retired from future latent-pool ratification because their observed result informed this validator correction. A new untouched v0.2 challenge is required.

## Scale adequacy

Current public N0 parameterization through the latent pool:

- semantic: 136,594,435
- structured: 1,656,064
- evidence adapter: 16,002,561
- evidence graph: 6,664,963
- ratified fusion: 116,981,764
- competitive latent pool: 29,557,764
- total through latent pool: **307,457,551**

These counts are descriptive checkpoint facts, not ceilings.

No current measurement demonstrates a latent-capacity bottleneck. The challenge shows strong semantic generalization, source-view recoverability, specialization, and multi-slot diversity. Scaling now would confound capacity with validator correctness and could waste compute.

## Future scale triggers

Increase slots, width, depth, fusion/latent capacity, or other representation capacity when one or more of the following is observed under a valid untouched evaluation:

- semantic or historical-query fidelity plateaus below required capability;
- source/evidence recoverability degrades as evidence complexity increases;
- effective slot rank or specialization saturates while unresolved information remains;
- added views/concepts cannot be represented without measurable interference;
- downstream Identity Decision Packet heads expose a representation bottleneck;
- N1 private identity fidelity requires additional capacity after authorization;
- a larger architecture materially improves hard capability/fidelity tests rather than merely fitting arbitrary validator targets.

No parameter, slot, view, concept, context, node, field, or edge count in the current checkpoint is a permanent product ceiling.

## Governance

- v0.1 frozen challenge result: immutable FAIL
- v0.1 challenge rows used for training: false
- gradient from v0.1 result: forbidden until a valid capability failure is established
- candidate weights changed after v0.1 challenge: false
- private identity data: false
- private identity gradient: false
- production promotion: false
