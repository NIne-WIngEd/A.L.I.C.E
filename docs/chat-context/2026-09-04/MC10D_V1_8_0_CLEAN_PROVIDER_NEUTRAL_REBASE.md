# MC10D v1.8.0 Clean Provider-Neutral Rebase — 2026-09-04

## Status

Released for live execution.

This is the clean replacement for the invalidated pre-release v1.8.0 candidate and for the recursive v1.7.3-v1.7.6 provider-wrapper ancestry.

## Scientific authority

- canonical main: `0abaed85873c3f8de04765847eb7700b0e20433f`
- MC10D freeze SHA256: `22B0ADBCCF442B0B3654F964E35AB77A044AF7623891E2A60C72B06A94ECE9A3`
- sole MC10D scientific parent: `ALICE_MC10D_JUDGE_SCHEMA_RECOVERY_v1.7.2.zip`
- v1.7.2 SHA256: `4D9EE7FE7AACB510A1170B5C53B9E9F07D1182FFDC04EE55FA2A533097324533`
- v1.7.3-v1.7.6 scientific authority: false
- v1.7.3-v1.7.6 code parents embedded: false

## Current MC10D state

- replacements: 63
- deferred slots: 1
- effective pool: 287
- slot 63 deferred
- slot 64 resolved; regeneration forbidden
- A-SYN accepted: 0
- A-SYN promoted: 0
- model training: false
- MC8 sealed-pending
- pointwise screen not started
- Stage G not closed
- Phase 2 not replaced

## Corrected provider routing

Gemma and GLM large-model public qualification remain on Kaggle GPU.

- Gemma: `gemma4:31b-it-q4_K_M`
- GLM: `glm-4.7-flash:q4_K_M`
- complete 16-task qualification per family
- no Q12-only resume
- exact frozen v1.7.2 worker/controller executes the judge science
- prompts, tasks, seeds, budgets, parser, thresholds, models, digests and refreeze logic unchanged

Magnolia remains qualified infrastructure and may execute later work only when the frozen capability contract says Magnolia can do it at full fidelity. No large Gemma/GLM judge is routed to Magnolia in v1.8.0.

## Provider-neutral infrastructure

- R13-R17 package SHA256: `BCA2FD9BBDB55A7F74F71538D33B0F5FF7568F6B54D05CA0130E5A8BBF914E23`
- R13-R17 durable receipt SHA256: `490DC2983A5D2D1F605B0FA8799F75FCD2E2E93A6DD8A0C9A6A7824332FE56E1`
- R18-R25 package SHA256: `77E788CA33F443ED984E91B1EB4D641793D8A346C4D49D1904C77AD2B7664AA0`
- R18-R25 durable receipt SHA256: `400113E93B855C4BA65E2D2B04B5798B55652CEC1D15817C45E98B5214760ABD`

Successful pointwise-ready output is made content-addressed on Drive and must pass round-trip SHA verification before `CHECKPOINT_DURABLE=true`.

## Historical Magnolia attempt

Known provider-only failure evidence remains immutable:

- job ID: `574947`
- job name: `rayan-mc10d-v175-runtime-860f751f`
- state: FAILED
- exit: `1:0`
- judge science executed: false

v1.8.0 only reconciles this as historical evidence.

## Release artifacts

Package:
`ALICE_MC10D_PROVIDER_NEUTRAL_REBASE_v1.8.0.zip`

SHA256:
`E354CB2D6122567DD74C725314AB27EB070F8DB5ACAED7B5DC74F9E8A91C5B68`

Launcher:
`Start-ALICEMC10DProviderNeutralRebaseV180.ps1`

SHA256:
`BF1BA7BBAE76CF2DD43156B4FA5BA39137327F933313D443F687E1D7823E41F6`

Build audit SHA256:
`77C02907EFD7E713C5F36E2FB5FEFE19569C37CC5B6B8342206A55A1F564E476`

## Release validation

- deterministic byte-identical rebuild PASS
- ZIP CRC PASS
- clean extraction PASS
- py_compile PASS
- exact v1.7.2 selftest PASS
- R13-R17 offline selftest PASS
- R18-R25 offline selftest PASS
- long historical Slurm JobName width corrected
- exact infrastructure receipt field semantics bound
- recursive v1.7.3-v1.7.6 code ancestry absent

## Success boundary

`MC10D_POINTWISE_SCOPE_JUDGE_SCREEN_CURRENT_EFFECTIVE_POOL_WITH_OPEN_COMPLETION_FRONTIER`

A successful v1.8.0 run stops at pointwise-ready. It does not run the pointwise screen itself.
