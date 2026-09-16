# N0 latent value-contrast v0.2 diagnostic result

Date: 2026-09-16

## Scope

This durable handoff records the result of the query-conditioned latent-pool value-contrast diagnostic v0.2. It is diagnostic-only. It does not ratify the latent pool, reclassify the frozen v0.2 challenge, authorize training, or authorize scaling.

## Candidate and immutable challenge state

- Candidate: adaptive multi-view latent pool step 360.
- Candidate SHA-256: `503d4064df6258d3a1bc0edeae17888cbeb4fb67e07ad7e27042f785ed092ba4`.
- Candidate weights were unchanged.
- Frozen challenge v0.2 remains `FAIL_PRESELECTED_CANDIDATE_NOT_ELIGIBLE_FOR_LATENT_POOL_RATIFICATION`.
- Frozen challenge rows and diagnostic rows remain prohibited from training.
- No private identity gradient.

## Diagnostic v0.2 contract

- 32 rows.
- Four families, 8 rows each:
  - `missing_structured_evidence`
  - `novel_semantic_authority`
  - `novel_structured_authority`
  - `semantic_reliability_reversal`
- Query-conditioned target/foil semantics were materialized before GPU.
- No global current-value assumption.
- No historical-query rows.
- Pre-fusion source ablation.
- Diagnostic-only, no ratification effect, no gradient.

## Key result

The diagnostic does not cleanly isolate a latent-pool causal failure because the required source representation itself is not reliably value-discriminative under the current full-sentence embedding contrast.

Overall across 32 rows:

- target/foil semantic separation: `0.009924205020070076`
- required source contains target value: `1.0`
- required source prefers target: `0.59375`
- required-source target-vs-foil margin: `0.0022809499059803784`
- normal latent set prefers target: `0.9375`
- normal latent-set target-vs-foil margin: `0.016335993190295994`
- counterfactual latent-set margin: `0.0143133889650926`
- latent-set margin drop: `0.002022604225203395`
- normal pooled readout prefers target: `0.28125`
- pooled margin drop: `-0.00010925531387329102`
- absolute target cosine drop: `0.08747471310198307`

For the historically failing `missing_structured_evidence` family:

- target/foil semantic separation: `0.013950437307357788`
- required source prefers target: only `0.25`
- required-source target-vs-foil margin: `-0.0075212642550468445`
- normal latent set prefers target: `0.75`
- normal latent-set margin: `0.016449838876724243`
- counterfactual latent-set margin: `0.015193291008472443`
- latent-set margin drop: `0.0012565478682518005`
- absolute target cosine drop: `-0.00983363389968872`

This is the decisive interpretation: the required source contains the target value on every diagnostic row, but the pooled required-source representation itself often ranks the matched foil as equally or more similar. In the failing family, that happens on 6/8 rows. Therefore a small latent target-vs-foil margin drop cannot be interpreted as proof that the latent pool ignored uniquely necessary evidence. The causal chain is confounded upstream of the latent module.

## Current decision

Do not:

- ratify step 360,
- reclassify the historical frozen FAIL,
- train on challenge or diagnostic rows,
- repair the latent pool yet,
- scale the model merely to satisfy this contrast metric.

Scaling remains open and uncapped, but no valid capacity bottleneck has been isolated yet.

## Next required work

Perform an upstream query-conditioned value-discrimination audit before latent repair or scaling:

1. Separate raw token/span value information from pooled source-view embeddings.
2. Measure target-vs-foil discrimination at raw semantic tokens, structured/evidence source outputs, source summaries, fusion inputs/outputs, latent slots, and pooled convenience readout.
3. Determine exactly where value identity becomes weak or inverted.
4. Use a value-aware/query-conditioned probe when full-sentence cosine cannot distinguish small value changes.
5. Do not use frozen challenge/diagnostic rows as gradient data.
6. Only after the upstream chain is valid may a persistent downstream failure justify latent repair or scaling.

## Governance

- Validation must not become an artificial capability ceiling.
- Exact internal allocation remains non-authoritative unless grounded.
- Challenge failure remains immutable history.
- Diagnostic results cannot retroactively alter ratification state.
- Public N0 only; private N1 identity gradient remains closed.
