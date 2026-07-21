# Phase 1.11 — Ollama Structured JSON Retry

The grounded-response generator already retries local Ollama transport timeouts, but malformed structured JSON was previously parsed only after the transport retry loop had ended. A single truncated or malformed JSON response therefore failed the entire query immediately.

This patch keeps transport and structured-output validation in the same retry loop. Retryable conditions include local timeouts, a missing text response, and JSON decoding failure in the structured response. The model, schema, prompt, temperature, and safety boundaries are unchanged.
