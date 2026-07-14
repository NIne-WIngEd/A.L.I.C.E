# Phase 1 — Local Automated Pilot Review

**Subphase:** P1.5c  
**Safety model:** deterministic extraction/privacy rules + local Ollama + human review of exceptions

## Guarantees

- Original source files are never modified.
- No file is uploaded to a cloud model.
- Ollama endpoints are restricted to `localhost` or `127.0.0.1`.
- Model names containing `cloud` are rejected.
- Document content is explicitly treated as untrusted data.
- The model cannot approve identity documents, credentials, highly sensitive material, third-party private data, contradictions, parser failures, or truncated previews.
- Only the remaining `pending` rows require human review.

## Supported pilot formats

JSON, HTML, CSV, plain text, PDF, DOCX, XLSX, PPTX, ICS, VCF, SRT, and XML.

## Install dependencies

```powershell
py -m pip install `
  -r requirements-dev.txt `
  -r requirements-phase1.txt `
  -r requirements-auto-review.txt
```

Optional Presidio layer:

```powershell
py -m pip install -r requirements-presidio.txt
py -m spacy download en_core_web_lg
```

## Local Ollama

Install Ollama for Windows, disable cloud features, restart it, and pull a local model.

```powershell
[Environment]::SetEnvironmentVariable("OLLAMA_NO_CLOUD", "1", "User")
ollama pull qwen3:8b
```

Use `gemma3:4b` instead on a lower-memory machine.

Verify:

```powershell
Invoke-RestMethod "http://127.0.0.1:11434/api/tags"
```

## Run

```powershell
py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:8b"
```

With Presidio:

```powershell
py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:8b" `
  --use-presidio
```

Optional private owner profile:

```powershell
py scripts\auto_review_pilot.py `
  --vault "C:\ALICE_Vault" `
  --model "qwen3:8b" `
  --profile "C:\ALICE_Vault\config\review_profile.txt"
```

## Private outputs

- `pilot-auto-review-summary-<RUN>.json` — aggregate, suitable to share.
- `pilot-review-<PROPOSAL>.csv` — complete prefilled review record.
- `pilot-manual-review-<RUN>.csv` — only exceptions requiring human review.
- `pilot-auto-review-details-<RUN>.json` — private per-content evidence.

Do not upload any CSV or details JSON.

## Finish the exceptions

Edit only these columns in the small manual CSV:

- `decision`
- `review_notes`
- `known_contradiction_group`
- `contains_identity_document`
- `contains_credentials_or_secrets`

Then apply it:

```powershell
py scripts\apply_manual_review.py `
  --vault "C:\ALICE_Vault" `
  --manual-csv "C:\ALICE_Vault\manifests\exports\pilot-manual-review-<RUN>.csv"
```
