# N0 Parent Value-Path Diagnostic Contract — 2026-09-16

## Why this diagnostic exists

The latent-pool value-contrast diagnostic v0.2 cannot cleanly attribute the frozen latent challenge failure to the latent pool. The designated required source representation contains the target value in every contrast row, but its pooled representation prefers the target over the foil only 59.375% overall and 25% for `missing_structured_evidence`.

The term `required_source` was also too coarse. In the parent cache it is not always a raw source-text embedding:

- semantic view: masked mean of semantic token states from `raw_text`;
- structured view: pooled output of the structured-state encoder;
- evidence view: pooled output of the evidence adapter + directed relation graph.

Therefore a negative target-vs-foil cosine at the source-view summary does not by itself prove that the raw source text or field encoding lost the value.

## Diagnostic question

Trace the same frozen 32 value-contrast cases through the immutable public parents and identify the first stage where target-vs-foil discrimination or query-conditioned field selection fails.

This diagnostic is inference-only. It has no ratification effect and may not train on challenge rows.

## Measurements

For every contrast case:

1. semantic target-vs-foil separation;
2. canonical required source-view target-vs-foil margin;
3. raw semantic `raw_text` masked-mean margin;
4. raw semantic `raw_text` pooled-encoder margin;
5. raw field-semantic target/foil discrimination when matching fields exist;
6. structured field-state target/foil discrimination;
7. evidence-adapter field-state target/foil discrimination;
8. evidence-graph field-state target/foil discrimination;
9. structured, adapter, and graph field-weight mass on target-containing vs foil-containing fields;
10. evidence-graph relation-status-bias mass on target-containing vs foil-containing fields;
11. structured pooled and evidence pooled target-vs-foil margins.

The diagnostic must verify that its reconstructed semantic, structured, and evidence source-view summaries numerically match the canonical `build_parent_cache` outputs before interpreting any result.

## Interpretation

### Evidence graph selection defect

For rows with both a target-containing and foil-containing field, especially `missing_structured_evidence`, the graph gives no consistent target-field advantage in field weight and/or relation-status bias for a current-value query.

This is an upstream evidence-graph capability defect. Repair the graph/query-conditioned pooling path before changing the latent pool.

### Representation-space / metric mismatch

The graph consistently selects the target field, but direct cosine between graph pooled state and target/foil sentence embeddings is weak or inverted.

This means sentence-embedding cosine is not a valid probe of the transformed graph representation. Replace the diagnostic metric; do not retrain or scale merely to satisfy it.

### Semantic pooling defect

The raw text clearly contains the authoritative target and not the foil, the pooled semantic encoder distinguishes target from foil, but the masked-mean source summary does not.

This indicates the semantic-view summary construction is losing value identity. Repair source-view pooling/fusion inputs rather than scaling the latent pool.

### Semantic encoder/value-geometry defect

Even direct pooled semantic representations of target-bearing raw text/fields cannot distinguish the target from the matched foil reliably.

Investigate semantic training/objectives or value-aware readouts before downstream repair.

## Governance

- Historical latent frozen challenge v0.2 remains FAIL.
- Value-contrast diagnostic v0.2 remains diagnostic-only.
- No private identity data or gradients.
- No latent weight changes.
- No scaling decision until the first failing stage is localized.
- No arbitrary fixed slot identities or exact routing percentages.
- Existing ratified parent checkpoints remain immutable during this audit.
