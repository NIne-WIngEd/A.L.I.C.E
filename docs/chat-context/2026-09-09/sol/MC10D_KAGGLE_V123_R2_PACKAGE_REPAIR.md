# MC10D Kaggle durable v1.2.3 R2 package repair

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.2.3 R1 observed stop

The repaired R1 launcher passed:
- package SHA verification;
- external compile gate;
- 20-test offline regression;
- live scientific-input verification;
- exact simulation payload verification;
- live MC8 re-verification.

It then failed locally inside `migrate_v122_checkpoint()` before Kaggle discovery/preflight/push:

`NameError: name 'V121_SOURCE_JOB_ID' is not defined`

Therefore:
- no Kaggle kernel was submitted by the R1 attempt;
- no new GPU time was consumed;
- no private MC10D candidate was attempted;
- no scientific state changed.

Root cause: policy already contained the exact v1.2.1 source job ID
`4953D4C9DC03F29F71270786985FD29E583922CF9FF1423FF17D48BE5AFED880`,
but controller.py failed to bind
`V121_SOURCE_JOB_ID=POLICY["v121_source_job_id"]`
at module scope.

## R2 release

Package:
`ALICE_MC10D_KAGGLE_DURABLE_FULL_MC10D_v1.2.3-R2.zip`

Package SHA-256:
`4ED7E8490D7E6C7A6B98266F1DA304BD2F7387228566793F38F56DE0576F68F9`

Launcher:
`Start-ALICEMC10DKaggleDurableFullMC10DV123R2.ps1`

Launcher SHA-256:
`1EAA693C2E77C9DF1BFDE42B22A28595BBB77CAAD2B85616D3B6B1F1688DAB90`

Build audit SHA-256:
`59AD088C58BE8C08EE173592D9D7FC0864ABDCC4E8DE1F16085BE8E3FE4B4104`

R2 uses a clean workroot:
`alice-mc10d-kaggle-durable-full-v1-2-3-r2.work`

Scientific and live-worker design are unchanged from v1.2.3.

## New regressions added

R2 increases offline gates from 20 to 22.

New gate 1:
`test_controller_global_name_resolution`

This parses the real controller with Python symtable and requires every referenced global name to exist in module scope. It directly catches the missing `V121_SOURCE_JOB_ID` class of error.

New gate 2:
`test_executable_v122_to_v123_migration_regression`

This constructs a synthetic but structurally exact v1.2.2 durable checkpoint, v1.2.2 dispatch authority, and v1.2.1 remote job-result authority. It then executes the real `migrate_v122_checkpoint()` function with network/Drive calls mocked locally. It verifies:
- migration actually executes;
- corrected spend is 5520 seconds;
- exact v1.2.1 / v1.2.2 job order is preserved;
- superseded evidence is preserved;
- migration receipt is produced.

Clean extraction, external compilation, manifest verification, SHA256SUMS verification, ZIP CRC, and all 22 regressions pass. Deterministic rebuild is byte-identical.

## Current execution contract

- provider: Kaggle 2xT4
- corrected local safety spend: 5520 seconds
- local working budget total: 21600 seconds
- protected reserve: 3600 seconds
- local working-budget remainder: 16080 seconds
- first family job: public context probe -> public exact-32 schema probe -> one private pilot
- first job soft limit: 2700 seconds
- later job soft limit max: 3600 seconds
- later private task limit: 12
- one active GPU job at a time
- full four-family attempt coverage
- no early-fail shortcut
- A-SYN acceptance/promotion false
- training false
- MC10E auto-execution false

Do not use `-NewQuotaWindow` in the current quota window.
