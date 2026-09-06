# MC10D Public Judge Source→Target Clarification Boundary — 2026-09-05

## Observed v1.8.1 Gemma result

The ratified budget amendment worked: Gemma produced structured output for all 16 public fictional qualification tasks under the 2048/4096/6144 ladder.

Preserved evidence:

- result SHA256: `F584C777C3B2A096BFCA490B372E3F1C8987696C5244862FB8FAA536EEBBDC0C`
- rows SHA256: `059AE371DC8CB93E54D697041A5E247C44CDB8406CEF627BCBBD77C35D11BFBA`
- failure SHA256: `E86FA767632EE4BCEA6892A5A84F07F00E726C3A3F9E98261C22B220E5F95A06`

Summary:

```text
tasks=16
verdict_matches=10
full_gold_field_matches=6
critical_all_correct=false
Q01_anchor=false
Q03_anchor=true
qualification_passed=false
```

## Specification defect

The task set intentionally presents a fictional source-person/core named **Mira** and a fictional target persona named **Ava**.

The unchanged gold labels treat source→target personality/behavioral completion as valid. The original system prompt, however, never states that source and target names may differ intentionally.

Gemma repeatedly treated the name difference itself as actor mismatch on Q01, Q12, Q14 and Q16. This makes the v1.8.1 Gemma failure inadmissible as a clean judge disqualification because the benchmark is under-specified at the exact concept it scores.

## Intended semantics

- `fixed_core` = fictional source-person personality/behavioral core.
- `target.actor` = fictional target persona being completed.
- Different source/target names alone are **not** actor mismatch, contradiction, or history transfer.
- Actor/role/state/context/direction is judged against `target` and `candidate_behavior`.
- Personality/behavioral traits may bridge source→target.
- Source-person events/history/lived memories do **not** transfer without explicit evidence.

This matches the A-SYN provenance doctrine: synthetic behavior may be derived from source-person personality/values/behavioral envelope, while source history cannot be laundered into target autobiography.

## Governance consequence

A second narrow owner-ratified amendment is required before another Gemma/GLM qualification run.

Only the public qualification system prompt may change. The following remain frozen:

- 16 tasks
- gold labels
- thresholds
- model tags/digests
- qualified profiles
- seeds
- temperature
- num_ctx
- schema/parser
- ratified budget ladder 2048/4096/6144
- preexisting Mistral/Granite bindings
- 287-candidate pool
- MC8 seal
- no A-SYN acceptance/promotion/training
- pointwise not started

Mistral and Granite remain preexisting bound judges. The clarified public-role qualification applies only to the still-missing Gemma/GLM families.

## Amendment candidate

Package:

`ALICE_MC10D_PUBLIC_JUDGE_SOURCE_TARGET_CLARIFICATION_AMENDMENT_v1.0.0.zip`

SHA256:

`007D2E202CB1D9C4C633515B686331893F60033CA768C33A603FF0E4EC923A19`

Clarified worker SHA256:

`729151D711507CDE7694FDF35F699B447F161A29EED412850F7D32FA27F96BFC`

This package is governance-only and grants no qualification or pointwise authority.
