# P1.11 FEVER-NLI stale test fix

The FEVER-NLI verifier changed the active label order to:

1. entailment
2. neutral
3. contradiction

Two older fake-model tests still emitted logits in the previous order:

1. contradiction
2. entailment
3. neutral

The production policy and new FEVER verifier test were correct; only the two
legacy fake-model fixtures were stale.

This patch updates those fixtures. It changes no runtime NLI thresholds,
retrieval behavior, claim filtering, model selection, or privacy settings.
