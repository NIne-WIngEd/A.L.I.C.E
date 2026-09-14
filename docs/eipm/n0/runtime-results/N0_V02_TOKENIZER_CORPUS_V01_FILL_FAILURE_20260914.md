# N0 v0.2 tokenizer-corpus v0.1 fill failure and correction

**Date:** 2026-09-14  
**Result:** expected fail-closed behavior; production manifest corrected before tokenizer training  
**GPU used:** no

## Observed run

The 120M-character tokenizer corpus materialization used the exact-revision 22-source v0.2 activated manifest and produced 124,936,801 accepted characters across 29,008 output rows. The run correctly exited nonzero because `common-pile/oercommons_filtered` filled only 32.12% of its 3.6M-character budget under the explicit Public Domain allowlist.

Observed OER Commons counters:

- seen: 5,249
- accepted: 208
- accepted characters: 1,156,312
- license rejected: 5,036
- fill ratio: 0.3212

No license gate is widened merely to satisfy the mixture quota. OER Commons is removed from the active v0.2.1 manifest. Its 3% share is redistributed inside the same educational/explanatory category: StackExchange 9% -> 10%, LibreTexts 5% -> 6%, PressBooks 5% -> 6%. The educational category therefore stays at 22% and the complete source-share sum remains 1.0.

## Second issue discovered

The same run showed that very large source records made character-budget control too coarse. Examples included Project Gutenberg at ~2.04x its budget, DOAB at ~1.20x, and USGPO at ~1.28x because one accepted row could contain an entire book/report.

This is a sampling-granularity problem, not a reason to lower the fill gate. `materialize_public_corpus_v02.py` now deterministically chunks long documents (default 16,000 characters), keeps all chunks of one document in the same train/dev/test split and deterministic partition, exact-deduplicates both documents and output chunks, and trims the final chunk to the remaining source budget when practical.

## Lineage decision

- Preserve `tokenizer-corpus-v0.1` and its failed receipt as evidence.
- Do not train a tokenizer from the failed corpus.
- New active manifest: `configs/eipm/n0/public_corpus_v0.2.1.activated.json`.
- New clean output target: `tokenizer-corpus-v0.2`.
- Private identity data/gradient remain false.
- No GPU training is authorized by this correction.
