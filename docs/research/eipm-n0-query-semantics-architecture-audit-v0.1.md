# N0 Query-Semantics Architecture Audit v0.1

**Date:** 2026-09-18  
**Trigger:** Magnolia job 575800, `FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE`  
**Policy:** no further gradient run until this architecture audit resolves the semantic information path.

## Job 575800 evidence

The research-grounded query–relation role router reached real training and completed normally.

Final result:

`FAIL_QUERY_RELATION_ROLE_ROUTER_DEV_STOP_NO_HELDOUT_EXPOSURE`

Key properties:

- no eligible dev checkpoint;
- heldout remained unopened;
- parent graph remained exactly unchanged;
- multiple later checkpoints preserved endpoint and ordinary behavior;
- direct role accuracy converged to exactly 0.5;
- families collapsed by role direction rather than becoming noisy.

At the final checkpoint, source-seeking families were classified as SOURCE and target-seeking families as SOURCE as well, producing 0.5 aggregate role accuracy. Earlier checkpoints sometimes flipped to the opposite global role. This is consistent with the router learning a relation-independent or weakly-query-dependent endpoint preference rather than decoding directional query meaning.

## Internal architecture finding

The graph repair cache builds `query_semantic` with:

`AliceN0V02Model.encode(...)`

which returns:

`mean_pool(token_states)`

The graph therefore receives one raw 640-D mean-pooled backbone state per query.

However the semantic model itself explicitly exposes:

- token-level contextual states through `encode_tokens` / `encode_views`;
- a dedicated `semantic_projection` head;
- a parameter report stating that pooled readout is not a capability ceiling.

The public N0 semantic multitask objective trains semantic contrastive geometry through:

`project_semantic_pooled(candidate_pooled)`

not by directly constraining the raw 640-D pooled vector alone.

Therefore the evidence graph currently bypasses both:

1. the trained semantic projection; and
2. the token-level representation that the semantic model explicitly keeps first-class.

This compatibility shortcut may have accidentally become a graph-architecture bottleneck.

## External research implications

Relevant external work supports auditing fine-grained representations before another router:

- AdapterFusion: separate knowledge extraction from non-destructive composition.
- ColBERT / late-interaction work: fine-grained token representations can retain matching information lost by one dense vector.
- Relation extraction probing: encoder architecture and linguistic feature access directly determine what relation information is recoverable.
- Predicate-argument enriched sentence encoders: explicit directional/predicate structure can materially improve relation-sensitive semantics.
- Relation-aware language–graph transformers / question-adaptive graph attention: language and relation structure can interact before graph reduction.

Mean pooling is not assumed to be inherently defective. Recent work shows modern contrastively trained encoders can often make mean pooling robust. The question is empirical for this exact N0 checkpoint and this exact directional relation capability.

## Audit question

Before any model change, determine where the relation-role information actually exists:

1. raw 640-D mean pool currently consumed by the graph;
2. trained 256-D semantic projection;
3. token-level contextual states using parameter-free late interaction.

## Diagnostic

Script:

`scripts/eipm/n0/audit_n0_v02_query_semantic_representations_v0_1.py`

Fresh audit-only queries cover:

- corrects: current vs previous;
- supersedes: replacement vs superseded;
- temporal_successor: later vs earlier;
- causes: cause vs effect;
- supports: supporter vs supported claim;
- derived_from: derived item vs derivation basis.

No previous train/dev/test/frozen/diagnostic row is reused.

No optimizer is created.

No gradient is performed.

No parameter is mutated.

### Measurements

For each relation-role query, compare matching vs opposite relation-role gloss using:

- raw mean-pooled cosine;
- trained semantic-projection cosine;
- token-level symmetric late interaction.

Also measure:

- source-vs-target opposite-query pair distance in each representation;
- consistency of the source-minus-target direction across paraphrases.

No capability threshold is frozen in this audit. The evidence is comparative.

## Decision after audit

### If trained projection is materially stronger than raw pool

The graph has been bypassing the semantic space trained for reusable meaning.

Next architecture should project/condition the graph through the trained semantic representation or a compatible full-width adaptation of it, without another local role classifier.

### If token-level late interaction is materially stronger

Directional semantics exist in the semantic backbone but are lost during pooling.

Next architecture should replace the pooled query→graph interface with token-level language↔relation interaction / cross-attention before field selection.

### If both projection and token views fail

The needed relation-role semantic distinction is not sufficiently represented by the frozen semantic model.

Stop graph repairs. Reopen N0 semantic representation learning/objectives and data coverage before graph work.

## Anti-loop rule

This audit itself does not authorize a new gradient run.

After the audit result, perform one architecture decision. Do not chain another local repair automatically.
