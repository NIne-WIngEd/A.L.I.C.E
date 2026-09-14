# N0 v0.2 tokenizer corpus fill failure and rebalance

The first 120M-character v0.2 tokenizer-corpus materialization failed intentionally at the source-fill gate. `common-pile/oercommons_filtered` supplied only 1,156,312 accepted Public Domain characters against a 3,600,000-character target (fill ratio 0.3212) while 5,036 rows were rejected by the explicit license allowlist.

Decision: do not widen the license gate just to hit a quota. Remove OER Commons from active production mixture and redistribute its 3% educational share to already-verified permissive educational sources: StackExchange 10%, LibreTexts 6%, PressBooks 6%.

The run also exposed coarse sampling from giant whole-book/report rows: Project Gutenberg overshot its share by ~2.04x, DOAB ~1.20x, and USGPO ~1.28x. The v0.2 materializer now chunks long records deterministically at <=16k chars by default, preserves document-level split/partition lineage, deduplicates documents and chunks, and trims the last chunk to the remaining source budget.

Preserve failed `tokenizer-corpus-v0.1` as evidence. Do not train tokenizer from it. Next clean target is `tokenizer-corpus-v0.2` using `public_corpus_v0.2.1.activated.json`. No GPU and no private identity gradient.
