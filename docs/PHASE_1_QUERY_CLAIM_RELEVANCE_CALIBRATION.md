# Phase 1.11 — Query–Claim Relevance Calibration

This stage calibrates a separate question from evidence support: does a candidate claim directly answer, or materially help answer, the user's question?

It reuses the local `cross-encoder/ms-marco-MiniLM-L6-v2` reranker to score `(question, claim)` pairs. Scores are treated as raw ranking logits; no universal zero/probability boundary is assumed.

Human labels are `relevant`, `partially_relevant`, and `irrelevant`. Only `relevant` is positive for a conservative gate.

The known regression queries `personal-004`, `personal-006`, and `personal-018` are excluded from calibration. Sampling spans the observed score distribution with rank buckets and query diversity.

This is private, local, read-only, and diagnostic. Any selected threshold must be frozen and validated on a fresh independent holdout before production use.
