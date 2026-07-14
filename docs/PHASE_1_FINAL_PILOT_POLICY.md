# Phase 1 — Final Conservative Pilot Policy

**Subphase:** P1.5c  
**Model calls:** None  
**Purpose:** Resolve the remaining pilot decisions without inspecting every file

## Policy

The parser pilot is deliberately conservative:

- high-confidence, positive-category records with explicit semantic approval
  are included;
- previously approved/rejected calibrated decisions are preserved;
- identity documents, credentials, substantial third-party data, financial,
  medical, legal/immigration, relationship, no-text, parser-error, and other
  sensitive/unsupported records are excluded from `pilot-v1`;
- low-confidence or ambiguous records are excluded rather than requiring the
  owner to review them;
- exclusion does not delete or modify source data;
- excluded files may be reconsidered in later, separately approved pilots.

The policy treats the model's `relevant_to_alice` boolean as advisory when it
conflicts with an explicit high-confidence `approve` recommendation and a
positive document category. This inconsistency was observed in the real review
results.

## Run

```powershell
py scripts\apply_final_pilot_policy.py `
  --vault "C:\ALICE_Vault"
```

The command uses existing calibration results and makes no Ollama requests.

## Audit

A small private audit CSV is generated with ten approved and five rejected
records. Reviewing the audit is recommended but is not equivalent to manually
reviewing all 120 source files.

## Validate

```powershell
py scripts\validate_pilot_review.py `
  --vault "C:\ALICE_Vault" `
  --minimum-approved 50
```

If the summary reports fewer than two valid contradiction groups, use the
initial parser-pilot setting:

```powershell
py scripts\validate_pilot_review.py `
  --vault "C:\ALICE_Vault" `
  --minimum-approved 50 `
  --minimum-contradiction-groups 0
```

Contradiction evaluation will be implemented in the structured memory phase.

## Finalize

After validation succeeds:

```powershell
py scripts\finalize_pilot.py `
  --vault "C:\ALICE_Vault" `
  --pilot-name "pilot-v1" `
  --minimum-approved 50 `
  --minimum-contradiction-groups 0
```
