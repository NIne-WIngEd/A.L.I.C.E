# P1.11 Evidence Expansion Test-Boundary Fix

The response-evidence expansion layer depends on the real local semantic model
and semantic index.

Two existing unit tests intentionally inject a fake `context_search_fn` and use
an empty `TemporaryDirectory` vault. Those tests are designed to exercise
grounded-response evaluation and checkpoint behavior without provisioning the
production embedding model.

The initial evidence-expansion integration ran the production semantic passage
expander even when a custom retrieval function was injected. That caused the
tests to fail with:

`FileNotFoundError: Local embedding model is missing.`

This patch defines a clean dependency-injection boundary:

- `context_search_fn is None`: production retrieval path; response-time evidence
  passage expansion runs normally and requires the local semantic model/index.
- custom `context_search_fn`: caller owns evidence construction; production
  evidence expansion is skipped.

No production retrieval, grounding, citation, privacy, memory, or action
guardrail is weakened.
