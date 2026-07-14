# Phase 1 — Auto-Review Resilience Patch

**Subphase:** P1.5 automated pilot review  
**Status:** Runtime correction after first production run

## Observed production failures

The first optimized run processed 59 semantic items but produced 59 unresolved
model results:

- 35 timeouts;
- 18 invalid structured-JSON responses;
- 6 out-of-range relevance scores.

The prior implementation treated an entire failed batch as failed, checkpointed
those transient failures as completed results, and used a response schema that
was too verbose for the available local inference speed.

## Corrections

- Default local reviewer changed to `qwen3:4b-instruct`.
- Initial batch size reduced from 6 to 3.
- Preview size reduced from 3,000 to 1,800 characters.
- Maximum generated tokens reduced from 1,200 to 600.
- Structured output uses compact field names and omits redundant summaries.
- Confidence is emitted as an integer percentage and safely clamped to 0–100.
- Timeout, connection, length-limit, invalid-JSON, omitted-item, and validation
  failures are tracked separately.
- A failed batch is recursively split until individual records can be retried.
- Individual records receive a final retry with a 900-character preview.
- Transient semantic failures are never checkpointed as completed results.
- Re-running the same command retries only unresolved records.
- Short request IDs replace long SHA-256 content keys in model output.

## Recommended command

```powershell
ollama pull qwen3:4b-instruct

py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:4b-instruct" `
  --use-presidio
```

The revised defaults are:

```text
batch_size = 3
max_chars = 1800
num_predict = 600
timeout_seconds = 240
single_item_retries = 1
thinking = false
resume = true
```

The previous schema-v2 checkpoint is not reused because schema version 3 has a
different configuration hash. It may remain in the private vault for audit
purposes.

## Success criteria

A valid run should have:

- `model_error_count` materially below the previous value of 59;
- nonzero `auto_approved` and/or `auto_rejected` counts;
- `semantic_failure_reason_counts` explaining any unresolved records;
- `ollama_metrics.done_reason_counts` dominated by `stop`, not `length`;
- a run-specific CSV and updated canonical review CSV.
