# Phase 1 — Fast, Resumable Local Auto-Review

**Status:** Replacement for the original one-document-per-request reviewer

## Why the original run was slow

The first reviewer sent one Ollama request per unique document, used previews of
up to 24,000 characters, did not explicitly disable Qwen3 thinking, and created
a new Presidio analyzer during every privacy scan. It also wrote the canonical
CSV only after all model calls, so a Windows file lock could fail the run at the
end.

## New behavior

- Qwen3 thinking is explicitly disabled.
- Six documents are reviewed in each model request by default.
- Each document preview is capped at 3,000 characters by default.
- Presidio is initialized once and reused.
- Every resolved content result is checkpointed immediately.
- A stopped or failed run resumes from its compatible checkpoint.
- The complete result is first written to a unique run-specific CSV.
- Failure to overwrite an Excel-locked canonical CSV no longer loses results.
- Ollama timing and token metrics are recorded in the aggregate summary.

The default changes reduce approximately 115 individual model requests to about
20 batched requests before deterministic exclusions and checkpoint reuse. Actual
runtime still depends on CPU/GPU performance and document contents.

## Run

Close any open `pilot-review-*.csv` file before starting.

```powershell
py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:8b" `
  --use-presidio
```

Default fast settings:

```text
batch_size = 6
max_chars = 3000
think = false
num_ctx = 8192
num_predict = 1200
resume = true
```

To use a smaller local reviewer:

```powershell
ollama pull qwen3:4b-instruct

py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:4b-instruct" `
  --use-presidio
```

A different model or setting creates a separate checkpoint.

## Resume

Run the same command again. Completed content objects are loaded from:

```text
pilot-auto-review-checkpoint-<PROPOSAL>-<CONFIG>.jsonl
```

Use `--no-resume` only when you intentionally want to discard the compatible
checkpoint and recompute everything.

## Locked CSV recovery

Every successful computation creates:

```text
pilot-review-auto-<RUN>.csv
```

If the canonical `pilot-review-<PROPOSAL>.csv` was open in Excel, close it and
run:

```powershell
py scripts\promote_auto_review.py `
  --vault "C:\ALICE_Vault"
```

## Hybrid architecture decision

Raw personal data, extraction, privacy scanning, review, provenance, and memory
governance remain local.

Later model fine-tuning/training may run in the cloud, but only from a separately
approved, minimized, redacted training export. The raw 27 GiB archive, vault
database, credentials, identity documents, and highly sensitive records are not
cloud-training inputs by default.
