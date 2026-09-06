# MC10D v1.8.4 Worker-Lineage Stop → v1.8.5 — 2026-09-06

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.8.4 deterministic stop

v1.8.4 successfully:
- verified canonical main and both prior amendments;
- preserved the failed v1.8.3 double-serialized staging receipt;
- installed the correctly serialized decision-scoring ratification receipt;
- re-adjudicated exact v1.8.2 Gemma rows at 14 verdict matches / 6 critical decision matches;
- kept Gemma remote rerun disabled.

It then stopped before GLM execution.

Root cause:
`patch_template()` was passed the decision-centric worker but compared its bytes against `EXPECTED_CLARIFIED_WORKER`.

Decision worker SHA256:
`931607AA75AC8FE16D7792ACFE86BE37AA0BA1E17D269170C677D0F8B132551D`

Older clarified-worker SHA256:
`729151D711507CDE7694FDF35F699B447F161A29EED412850F7D32FA27F96BFC`

No GLM remote run occurred in v1.8.4.

## v1.8.5

Package:
`ALICE_MC10D_PUBLIC_JUDGE_QUALIFICATION_REBASE_v1.8.5.zip`

SHA256:
`AA64B7E093DDE4A707F283774911F2FCAFDB85B1292B41D1F9D8A2B6ECB0C6F8`

Launcher SHA256:
`C479AD912766319ED56385A398A097867D44C30EE08322A8E7711EA535D46FC2`

Engine SHA256:
`B32F0B6ED4C29C0E2A6C106A4A7DB4D19E3857079B09B628467AE8C4F980A27F`

Controller SHA256:
`7CC6C97E2BEC3FA2ABA200D8C6245A969D72EE8443E8AEE8FFDBA04BFBA9E960`

Single execution fix:
- scratch-patch input invariant now compares against the decision worker SHA.

No scoring science changes:
- verdict >=14/16
- critical decision matches >=6/7
- hard anchors Q02/Q04/Q05/Q11/Q13
- Q01 PASS
- Q03 HOLD

The scoring receipt remains the already owner-ratified v1.8.4 receipt:
`FD1228D88F7F7B7498F70F9DB7595A042E94D8293F19D156C5A970C06B775BF5`

Gemma is reused from exact v1.8.2 rows. Gemma remote rerun remains false.
GLM remains the only fresh 16-task GPU family.

The v1.8.5 selftest executes the exact previously failing patch/render path against the decision worker, renders a GLM worker, and compiles it before live execution.

Current authority ceiling remains:
- effective pool 287
- MC8 sealed
- pointwise false
- A-SYN acceptance false
- A-SYN promotion false
- training false
- Stage G open
- Phase 2 not replaced

## Controlled synthesis breadth v1.0.3

Package:
`ALICE_MC10D_CONTROLLED_SYNTHESIS_BREADTH_AMENDMENT_v1.0.3.zip`

SHA256:
`09E6D336C552055759CA335CAD9AA5823ACF8E41D3C8F6571BC59FF73C6CD604`

Launcher SHA256:
`ED4478FDB7A0FA8314BA9468C86ADB865C582685D5E7082B02DFF7254E581B9D`

v1.0.3 corrects stale prerequisite metadata in v1.0.2 and consistently requires:
- successful v1.8.5 qualification receipt;
- exact v1.8.5 pointwise-ready bundle hash.

Breadth science remains unchanged:
- high-impact A-SYN generation floor 3 E0 families;
- 4+ strong-anchor tier;
- 41 held three-family gaps reopen only for a later expansion frontier;
- current 287 pool unchanged;
- outlier becomes challenge trigger rather than auto-reject;
- 18 falsification tests unchanged;
- 8 zero-tolerance gates unchanged;
- E-INF historical policy unchanged.
