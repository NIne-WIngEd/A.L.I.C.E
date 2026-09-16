# N0 Parent Value-Path Localization v0.1

Current latent candidate remains step 360, SHA `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`, 29,557,764 parameters. It is not ratified and its weights have not changed.

The frozen latent challenge v0.2 remains an immutable FAIL. The subsequent value-contrast diagnostic v0.2 is diagnostic-only and cannot reclassify that result.

## Why the next audit moved upstream

The value-contrast diagnostic showed that the designated required source representation contains the target value on all 32 rows but prefers the target over the matched foil only 59.375% overall. In `missing_structured_evidence` it prefers the target only 25% of rows with mean margin -0.0075212642550468445.

A source-view summary is not necessarily a raw source embedding:

- semantic source summary = masked mean of semantic token states from raw text;
- structured source summary = structured-state pooled output;
- evidence source summary = evidence adapter + directed evidence graph pooled output.

Therefore the negative evidence-source cosine can be caused by query-conditioned graph selection failure, transformed-space cosine miscalibration, or an earlier semantic/value-geometry issue. It cannot yet be blamed on the latent pool.

## New authoritative work

Build branch state: `configs/eipm/n0/alice_n0_latent_pool_stage_state_v0.6.json`.

Parent-path diagnostic v0.1 uses the canonical `build_parent_cache` path with forward hooks only. The hooks capture structured, evidence-adapter, and evidence-graph intermediate outputs without mutating parents. The evaluator requires exact parity between captured pooled outputs and canonical parent-cache source summaries.

It measures:

1. target-vs-foil semantic separation;
2. raw-text masked-mean vs pooled semantic encoder margins;
3. raw field semantic margins;
4. structured field states and weights;
5. evidence-adapter field states and weights;
6. evidence-graph field states and weights;
7. evidence graph relation-status bias for target vs foil fields;
8. structured/evidence pooled margins.

For `missing_structured_evidence`, the decisive diagnostic is whether the graph gives the corrected current field more query-conditioned weight/bias than the stale field.

## Decision discipline

- If graph selection is wrong, repair the evidence graph before latent work.
- If graph selection is right but pooled sentence cosine is wrong, replace the metric rather than retrain.
- If semantic pooled encoding is right but masked-mean source summary is wrong, repair semantic source pooling.
- If direct semantic value geometry is weak, audit the semantic objective/readout.
- Reassess latent repair or scaling only after a valid upstream value signal is demonstrated.

No challenge-row training, no private identity gradient, no latent weight change, and no scaling is authorized at this stage.
