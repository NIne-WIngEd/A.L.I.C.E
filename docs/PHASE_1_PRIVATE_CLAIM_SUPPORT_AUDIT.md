# P1.11 — Private Claim-Support Auditor

P1.11 now has a deterministic response verifier, but exact benchmark-source
agreement is not the same as citation faithfulness.

This auditor evaluates the narrower question:

> Does the evidence cited by each generated claim actually support that claim?

## Audit unit

The response layer already emits structured atomic claims:

```json
{
  "text": "The AFM project used a U-Net.",
  "claim_type": "fact",
  "citations": ["[S4]"]
}
```

For each claim, the auditor receives only:

- the claim;
- its claim type;
- the source(s) explicitly cited by that claim;
- the selected evidence text for those citations;
- trusted owner-relation metadata.

It does not receive uncited evidence.

## Verdicts

- `supported`: every material assertion is directly stated, clearly entailed,
  or a trivial paraphrase of the cited evidence;
- `partially_supported`: some material assertions are supported but at least
  one is not established;
- `unsupported`: a critical assertion is not established or conflicts with the
  cited evidence.

Topic similarity alone is not support.

## Privacy and authority

The audit model is local Ollama by default.

The auditor has:

- no memory-write authority;
- no tool access;
- no web access;
- no external-action authority.

Private claim text, evidence, rationales, citations, and case details are stored
under:

```text
C:\ALICE_Vault\manifests\audits\pilot-v1\
```

Only aggregate summaries are written to:

```text
C:\ALICE_Vault\manifests\exports\
```

## First audit: exact-source mismatches

Use the v7 response-evaluation details to select only cases where A.L.I.C.E.
cited a different source from the original benchmark label:

```powershell
py scripts\audit_grounded_claim_support.py `
  --vault "C:\ALICE_Vault" `
  --benchmark $BenchmarkPath `
  --scope expected-source-misses `
  --response-evaluation-details $V7Summary.private_details_path `
  --pilot-name "pilot-v1"
```

This is a diagnostic audit, not the final P1.11 gate.

## Final audit

After the mismatch-only audit is understood, audit all approved cases:

```powershell
py scripts\audit_grounded_claim_support.py `
  --vault "C:\ALICE_Vault" `
  --benchmark $BenchmarkPath `
  --scope all `
  --pilot-name "pilot-v1"
```

Only the `all` scope is marked `eligible_as_final_p1_11_gate=true`.

## Metrics

- `citation_support_rate`
- `high_confidence_support_rate`
- `partially_supported_claim_count`
- `unsupported_claim_count`
- `manual_review_required_claim_count`
- `fully_supported_case_count`

Current policy targets:

```text
citation_support_rate >= 0.90
high_confidence_support_rate >= 0.80
```

## Important limitation

The default auditor uses the same local model family as the response generator.
That is useful for automated diagnosis but is not independent ground truth.
Low-confidence, partial, or unsupported judgments remain candidates for manual
review or a future second-model/NLI audit.
