# P1.11 Output Budget Test Expectation Fix

The structured-output budget was intentionally raised from 512 to 2048 tokens
to prevent truncated JSON responses from Ollama.

One older compatibility test still asserted that the repository policy must use
512 tokens. That assertion is now stale.

This patch updates only the current repository-policy expectation to 2048.
The legacy-policy compatibility test still verifies the loader's backward-
compatible default of 512 when `maximum_output_tokens` is absent.
