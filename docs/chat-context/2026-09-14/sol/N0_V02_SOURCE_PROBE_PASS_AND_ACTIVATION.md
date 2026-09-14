# N0 v0.2 source probe pass and activation

The 2026-09-14 Magnolia login-node source probe inspected 64 rows from each of 22 planned public sources. All 22 passed the required `text`, `id`, row-license, and provenance schema checks; zero sources failed; the probe exited 0. The probe itself authorized no training and contained no private identity gradient.

Production source state is now frozen in `configs/eipm/n0/public_corpus_v0.2.activated.json` on `alice-eipm-v1-build`. Every source uses its exact resolved Hugging Face revision SHA and a fail-closed source-specific license allowlist. Unseen license strings must be rejected, not inferred safe.

A new `materialize_public_corpus_v02.py` builds deterministic source-balanced tranches from the target shares, exact-deduplicates across sources, supports disjoint hash partitions, and fails if a source cannot fill at least 95% of its assigned share. The first next action is a 120M-character public tokenizer corpus on the login node. No GPU or private data is involved. Inspect that receipt before training the tokenizer.

Do not return to N0 v0.1 semantic growth. N0 v0.2 remains the production candidate; v0.1 step-1000 remains the pathfinder baseline only.
