# MC10D v1.9.2 post-refreeze stop and v1.9.3 recovery

Date: 2026-09-07

## What v1.9.2 actually completed

v1.9.2 completed the substantive deterministic pre-pointwise science:

- effective pool = 287
- deferred = 1
- bound families = Gemma + Qwen + Mistral + Granite
- current-pool scenario freeze = complete
- scenario probe upper bound = 55,104
- successorized internal v1.7 pointwise-ready validator = PASS
- canonical atomic publish to the Vault = complete
- exact pointwise-ready ZIP created
- exact ZIP SHA-256 = `5CC3C64EF4E64D75BD41E860F34CCD8FE8C882EEDBB6F6C6D0F68CB13218EF30`

No private pointwise inference, MC8 access, full simulation, A-SYN acceptance/promotion, or training occurred.

## Why v1.9.2 still exited 76

The remaining failure was in the **outer successor validator only**.

After the historical refreeze had already succeeded, the v1.9.2 outer controller extracted the returned pointwise-ready ZIP and called a helper that required exactly one directory wrapper. The historical v1.7 output is a flat archive with multiple top-level entries. Therefore the outer validator raised:

`pointwise-ready expected one root, found 7`

This is an archive-shape assumption defect. It is not a scientific failure and does not justify rerunning the historical refreeze.

## v1.9.3 recovery rule

v1.9.3 consumes the exact v1.9.2 pointwise-ready ZIP by SHA-256 and **does not rerun v1.7**.

It resolves the bundle root by unique semantic markers:

- `POINTWISE_READY_RECEIPT.json`
- `judges/mc10d_judge_binding_receipt_v2.json`
- `EVALUATOR_SEAL_PENDING.json`

It supports the historical flat layout and a single wrapped layout. Multiple semantic roots fail closed.

It also requires the extracted ZIP tree to equal the atomically published canonical Vault tree before successor governance is completed.

## v1.9.3 release

- package: `ALICE_MC10D_QWEN_SUCCESSOR_POSTREFREEZE_RECOVERY_v1.9.3.zip`
- package SHA-256: `8A8AC8C9FAA1C999E6F1236F80BC04FBA625F10CA9B3DF4D9F9FF94D2EBC8D9D`
- launcher: `Start-ALICEMC10DPostRefreezeRecoveryV193.ps1`
- launcher SHA-256: `28C90285AAAFA1EE8C3E57A38EA67A18ED74C5D8EC29FD3B59A1E03C70567D53`
- build receipt SHA-256: `3F811258D983B8AA4415F5E5AB0A3030C6DBB4627F0D5B27853BAFC407AE8D2B`

Selftest covers:
- the actual flat/multi-top-level failure class;
- wrapped layout;
- ambiguous-layout fail-closed behavior;
- E0 never generated;
- breadth floor 3 / strong anchor 4;
- 18 falsification families;
- 8 zero-tolerance vetoes;
- no historical refreeze rerun;
- no remote inference.

The PowerShell launcher also preserves native Python exit codes without allowing Windows PowerShell native-stderr handling to mask deterministic exit 76.

## v1.9.3 outputs on success

- `ALICE_MC10D_POINTWISE_DISPATCH_AUTHORITY_v1.9.3.json`
- `ALICE_MC10D_POINTWISE_BUNDLE_LAYOUT_v1.9.3.json`
- `ALICE_MC10D_PRIVATE_POINTWISE_INPUT_FREEZE_v1.9.3.zip`
- `ALICE_MC10D_PRIVATE_POINTWISE_INPUT_FREEZE_RECEIPT_v1.9.3.json`

The private-pointwise input freeze becomes the exact next-stage content-addressed input. Private pointwise then runs against those bytes. Full simulation/falsification remains blocked until pointwise freezes its survivor set.
