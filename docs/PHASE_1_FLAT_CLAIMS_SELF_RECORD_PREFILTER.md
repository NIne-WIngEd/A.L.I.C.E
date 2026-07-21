# Phase 1.11 — Flat Claims + Personal Self-Record Prefilter

## Diagnosis

The response-context reranker and saved context packages were correct, but the
experimental source-partitioned nested output contract produced claims whose
declared citation group did not reliably correspond to the source text that
inspired the claim.

The nested contract also increased malformed structured-JSON failures.

## Change

This patch replaces only `ollama_generate_evidence_claims()` using AST function
boundaries, so local helper functions and the Personal Claim Validity Contract
are not overwritten.

The generator returns to the original flat structured claim schema.

Malformed JSON remains retryable.

For first-person personal queries, the generation prompt is deterministically
restricted to `owner_self_record` evidence whenever at least one such source is
available. This reduces contamination from unrelated chat logs, textbooks,
third-party records, and unknown-owner files while keeping original citation
IDs.

If no `owner_self_record` evidence exists, the full original evidence package
is preserved and the existing Personal Claim Validity Contract remains the
safety boundary.

## Unchanged

- response passage reranker;
- Personal Claim Validity Contract;
- query-claim relevance threshold;
- FEVER/ANLI support thresholds;
- read-only/private guardrails;
- model and temperature.
