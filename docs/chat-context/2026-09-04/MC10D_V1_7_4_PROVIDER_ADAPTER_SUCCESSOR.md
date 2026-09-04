# MC10D v1.7.4 Provider-Adapter Successor — 2026-09-04

## Boundary

v1.7.3 successfully completed all local scientific binding and selftest gates, including:

- canonical main `0abaed85873c3f8de04765847eb7700b0e20433f`;
- frozen v1.1 bundle;
- 63 valid replacements;
- 1 deferred slot;
- 287 effective candidates;
- exact Mistral and Granite bindings;
- exact Gemma Q12 empty-content predecessor failure;
- full-family requalification requirement;
- neutral task-set SHA `7D91C291A65A61D16D5CC0623D58B49D856944F8466FEAD8DBB866EC26B867AA`.

v1.7.3 then stopped in Magnolia provider preflight before any live compute.

## v1.7.3 failure classification

Observed failing preflight used:

`test -d '$HOME/rayan-compute'`

The remote shell therefore treated `$HOME` literally because it was inside single quotes.

Classification:

`PROVIDER_PREFLIGHT_REMOTE_HOME_LITERAL_QUOTING_ADAPTER_BUG`

At the stop:

- Magnolia runtime job submitted: false
- Gemma job submitted: false
- GLM job submitted: false
- runtime download started: false
- model pull started: false
- judge inference started: false
- Drive checkpoint write started: false
- MC10D science changed: false
- canonical main changed: false

## v1.7.4 release

Package:

`ALICE_MC10D_PROVIDER_NEUTRAL_JUDGE_CONTINUATION_v1.7.4.zip`

SHA256:

`E025528FFE4637D0E665174BA342B8155F0AC40A4ECB6A27AD6972051929795F`

Launcher:

`Start-ALICEMC10DProviderNeutralJudgeContinuationV174.ps1`

SHA256:

`A7B8EF3FF86AD502E88662B232DCCAA1E172F06FE9F13D26E7EE7DB175F4F174`

Parent v1.7.3 package SHA256:

`F7E815B759FDCD3F1C3D32664589A147C90486D9B074C8A20E9B7EB730D5C483`

Frozen scientific workload SHA256 remains:

`F1F79C15EA3052E420AF769E806F9DDBA148C9057671749482A03F353741F6CC`

## v1.7.4 provider-only corrections

1. Reuse exact PREPARED-only v1.7.3 neutral run/task bytes rather than inventing a new scientific run.
2. Preserve v1.7.3 source state unchanged.
3. Reconcile absence of v1.7.3 remote runtime/Gemma/GLM jobs and remote run directory before mutation.
4. Use absolute remote root `/homes/01/mxrayan/rayan-compute` in provider preflight.
5. Select CPU and RAM from the same actual Magnolia node class.
6. Prefer >=48 GiB RAM node classes when available.
7. Submit complete Gemma and GLM family qualifications concurrently.
8. If one submitted family fails, continue reconciling the already-submitted peer. Preserve a valid peer result durably before exiting for the failure.
9. Provider telemetry/job identities advance to v1.7.4. Scientific task identity does not change.
10. Family Slurm time allowance is 12 hours. This is scheduling only.

## Frozen science unchanged

- 16 qualification tasks per family
- no Q12-only resume
- output budgets 768 / 1536 / 2048
- judge models/digests/profiles unchanged
- prompts unchanged
- parser semantics unchanged
- qualification thresholds unchanged
- technical failure is not abstention
- slot 63 not regenerated
- slot 64 not regenerated
- MC8 sealed
- A-SYN accepted = 0
- A-SYN promoted = 0
- training = false
- pointwise screen not started

## Execution route

v1.7.4 uses Magnolia CPU only for Gemma/GLM:

- Kaggle GPU submission: false
- Magnolia P100 selected: false
- Magnolia A100 selected: false
- model substitution: false

If both Gemma and GLM validate, control returns to the unchanged v1.7.2 -> v1.7.0 chain for four-judge binding and 287-candidate refreeze. The package stops at the pointwise-ready boundary.
