# Phase 1.11 — Frozen-Threshold Query–Claim Relevance Holdout

The V2 relevance calibration found that the local
`cross-encoder/ms-marco-MiniLM-L6-v2` is promising as a conservative
**irrelevance rejection filter**, not as a strict direct-answer classifier.

The calibrated non-irrelevance objective treats:

- `relevant` as positive;
- `partially_relevant` as positive;
- `irrelevant` as negative.

The calibration selected a candidate threshold of `-9.76161`, with observed
precision `1.0` and recall `0.666667` on the nine-item V2 calibration sample.

This holdout workflow freezes `-9.76161` before human holdout review and never
sweeps or retunes the threshold on holdout labels.

Holdout preparation:

1. regenerates candidates using the current post-fix P1.11 pipeline;
2. excludes the three known regression query IDs;
3. excludes exact calibration `item_id`s;
4. scores candidates with the existing local MS MARCO cross-encoder;
5. caps selection at two claims per query;
6. reuses the separate query-claim relevance review UI.

Because the pilot benchmark is small and generation can produce new claims for
previously seen questions, this is a **fresh-generation claim-level holdout**.
Query IDs may overlap with calibration and that overlap is reported explicitly.

Passing the holdout only makes the relevance filter a candidate for production
integration review. The holdout itself cannot change the production gate.
