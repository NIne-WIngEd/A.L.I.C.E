# MC10D v1.9.1 second validator stop and v1.9.2 whole-validator fix

Date: 2026-09-07

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.9.1 observed state

The owner executed v1.9.1. It reached the same scientifically correct successor state as v1.9.0:

- effective pool: 287
- deferred: 1
- four-family binding: Gemma + Qwen + Mistral + Granite
- full current-pool scenario freeze: complete
- scenario-probe upper bound: 55,104
- private pointwise: not started
- MC8: sealed
- full simulation: not started
- A-SYN acceptance/promotion: 0/0
- training: false

Terminal SHA-256:

`AF16543469016BC80A7DF6AA88CDCE63F033273DF28ADB9506B85C4AA2548620`

## Root cause

v1.9.1 correctly patched the first legacy family-set assertion in the copied 29-line v1.7 pointwise-ready validator:

`POINTWISE_READY_RECEIPT.json -> judge_families`

The same validator contains a second independent family-set assertion over:

`judges/mc10d_judge_binding_receipt_v2.json -> bound_judges`

That second assertion still required `gemma,glm,mistral,granite`. The generated successor binding correctly contained `gemma,qwen,mistral,granite`, so validation stopped at `RuntimeError: binding`.

This is a deterministic migration/controller defect, not a scientific failure.

## v1.9.2 correction

Release:

- package: `ALICE_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_v1.9.2.zip`
- package SHA-256: `5BA15F6807A66FEA8601333F809F3C798E0621F95B4C9A9383BFEBD940873A39`
- launcher: `Start-ALICEMC10DQwenSuccessorRefreezeBreadthV192.ps1`
- launcher SHA-256: `C87BF4ABCAA59B7970C6CDB7CF9FA562929DADD71CFF215EA9AA0F04726D017E`
- build receipt SHA-256: `1382347D43B2182136DDFB15C64083FA9E06B4369B23BFBE7C28CAD56A226FBA`

v1.9.2 no longer patches one observed line at a time. It audits the entire copied historical validator family-binding surface.

The exact historical family literal:

`['gemma','glm','mistral','granite']`

must occur exactly **two** times. Both occurrences are replaced with:

`['gemma','qwen','mistral','granite']`

The patch then requires:

1. the successor receipt-family assertion exactly once;
2. the successor binding-family assertion exactly once;
3. zero residual historical family literals;
4. successful Python compilation before execution.

Any count or anchor drift fails closed.

The original v1.7 package remains byte-for-byte untouched.

## Regression audit

Fresh-extraction selftest covers both v1.9.0 and v1.9.1 failure classes.

It verifies:
- package CRC;
- deterministic byte-identical rebuild;
- Qwen 84-file evidence audit;
- independent Qwen score recomputation;
- no-inference Qwen adapter;
- all eight v1.7 successor patch anchors;
- exactly two historical validator family literals;
- both successor assertions after migration;
- zero residual historical literal;
- 18 falsification families;
- 8 zero-tolerance vetoes;
- E0 never generated.

No remote inference is started by v1.9.2. Existing exact Qwen adapter evidence is reused. Slot 63 attempt 4 remains forbidden. Slot 64 remains resolved and is not regenerated.

## Next action

Run v1.9.2 once.

On success it must emit:

- `ALICE_MC10D_POINTWISE_READY_QWEN_SUCCESSOR_v1.9.2.zip`
- `ALICE_MC10D_POINTWISE_DISPATCH_AUTHORITY_v1.9.2.json`

The next package must bind private blinded pointwise to those exact hashes. Full simulation/falsification remains blocked until that pointwise result freezes.
