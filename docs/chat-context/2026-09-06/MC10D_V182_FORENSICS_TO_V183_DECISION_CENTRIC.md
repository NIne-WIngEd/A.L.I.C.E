# MC10D v1.8.2 Forensics → v1.8.3 Decision-Centric Qualification — 2026-09-06

## Canonical boundary

Canonical `main` remains frozen at:

`0abaed85873c3f8de04765847eb7700b0e20433f`

No pointwise screening, A-SYN acceptance, promotion, training, Stage G closure, or Phase 2 replacement has occurred.

## v1.8.2 forensic result

Gemma v1.8.2 executed the ratified source→target clarification and budget ladder.

Preserved result:

- result SHA256: `5499102590BCDCE21E9CC94BDAF2568F33E9E1C3E14992EF15C0FBE6DA759B56`
- rows SHA256: `67B6651E15C17EEFE3F5D0D2BC95B439850C0D45155051E76B3AAA2836CFC7C1`
- rendered worker SHA256: `1E1F16D6669D8482CB3A9C471D4C9FC9DD7176EDFE0353F9BB9147E504C024C8`
- verdict matches: 14 / 16
- full six-field gold matches: 10 / 16
- Q01 anchor: PASS
- Q03 anchor: HOLD
- residual source→target-name mismatch signature: 0

The source→target clarification therefore fixed the dominant v1.8.1 ambiguity.

The remaining qualification failure exposed a scoring-contract defect: legacy `critical_all_correct` required every critical task to reproduce all six internal diagnostic fields exactly. Correct REJECT decisions on Q02, Q04, and Q06 were therefore treated as critical failures because secondary explanation fields differed.

The genuine verdict misses were Q07 and Q09. Only Q07 is a critical task.

## v1.8.3 scoring amendment

v1.8.3 uses a decision-centric public qualification contract:

- structured outputs: 16
- minimum verdict matches: 14 / 16
- critical decision match = verdict + critical_veto match
- minimum critical decision matches: 6 / 7
- mandatory hard anchors: Q02, Q04, Q05, Q11, Q13
- Q01 must PASS
- Q03 must HOLD
- secondary decomposition fields remain recorded but are diagnostic-only
- technical failure is still not abstention
- tasks, gold labels, prompts, model identities/digests, seeds, temperature, num_ctx, and budget ladder remain unchanged

Gemma is re-adjudicated from the exact preserved v1.8.2 rows. Gemma is not rerun remotely.

GLM must run a fresh complete 16-task family under the already-frozen decision-centric contract.

If GLM passes, v1.8.3 performs the four-judge bind and current 287-candidate refreeze. It stops at pointwise-ready.

## v1.8.3 release

Package:

`ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.3.zip`

SHA256:

`99911541026A76C26C4FE9E068C07CBF8B49B82939AE67BCF1DDE5A4CC42B62B`

Decision worker SHA256:

`931607AA75AC8FE16D7792ACFE86BE37AA0BA1E17D269170C677D0F8B132551D`

Expected decision-scoring ratification receipt SHA256:

`60BDBA8C405767075F386420761802FDF710702218857F0FFC137F3041A47661`

The v1.7.0 refreeze controller is verified at SHA256:

`936DD235CC2BF486A7CD830446594677FDDA6ADE687B700D5B5144F9E4BFE74E`

Only Stage-B qualification validation is transformed. The old v1.7 public-GPU fallback is disabled in the disposable copy. Stage C dynamic refreeze logic is unchanged.

## Controlled synthesis breadth v1.0.1

Prepared but not executable before v1.8.3 success.

Package:

`ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.1.zip`

SHA256:

`8CB3C9D30238AEBC7D4D39E44F0647D1C810DE50A083C0A1CB2E614213E4E2CE`

Policy direction:

- future high-impact A-SYN floor: 3 E0 families
- 4+ remains strong-anchor tier
- 41 held three-family gaps reopen only for a future expansion frontier
- current 287 pool stays unchanged
- 97.5th-percentile outlier becomes challenge trigger rather than automatic rejection
- noncritical uncertainty may narrow/lower-tier/shadow/defer
- repairable candidate gets new ID + full retest
- all 18 falsification tests remain
- all 8 zero-tolerance gates remain
- E-INF historical policy remains strict
- no acceptance/promotion/training authority is granted

v1.0.1 requires the successful v1.8.3 receipt and exact v1.8.3 pointwise-ready bundle hash before ratification.
