# P1.11 Resilience Policy Compatibility Fix

The evidence-passage expansion patch accidentally rebuilt
`grounded_response_policy.json` from the original P1.11 policy instead of the
later resilient-evaluation policy.

As a result, the Python loader still expected:

- `request_retry_count`
- `request_retry_backoff_seconds`
- `keep_alive`
- `maximum_output_tokens`

but the policy file no longer contained those keys. Direct dictionary lookup
therefore raised `KeyError` before any grounded-response test could run.

This patch:

- restores the resilient values to the repository policy;
- keeps evidence-passage expansion enabled;
- changes the loader to use safe backward-compatible defaults for resilience
  fields if an older policy is loaded;
- adds regression tests for both the current merged policy and a legacy policy.

No retrieval, citation, memory, action, or privacy guardrail is weakened.
