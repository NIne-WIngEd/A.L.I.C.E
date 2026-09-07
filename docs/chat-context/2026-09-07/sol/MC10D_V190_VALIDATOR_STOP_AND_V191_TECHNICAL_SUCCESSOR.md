# MC10D v1.9.0 deterministic validator stop and v1.9.1 technical successor

Date: 2026-09-07

## Observed v1.9.0 state before stop

The owner executed v1.9.0. The run independently revalidated Qwen, recovered preserved Gemma, verified frozen Mistral/Granite, reused/installed the no-inference Qwen adapter, and reached the deterministic refreeze boundary.

Observed:
- effective pool: 287
- deferred slots: 1
- four-family binding complete: gemma, qwen, mistral, granite
- full scenario freeze current pool complete: true
- scenario-probe upper bound: 55,104
- private pointwise: not started
- MC8: sealed
- full simulation: not started
- A-SYN accepted/promoted: 0/0
- training: false

Terminal transcript SHA-256:
`75281E6B74306D5F4C449339840663B6915DDF7F2CB52385AADAF67C021A5263`

v1.9.0 exited 76.

## Root cause

The successor controller correctly changed the live judge family set from GLM to Qwen. However, the historical v1.7 refreeze copied `validate_mc10d_pointwise_ready_v170.py` unchanged into the disposable ready directory.

That validator still required:

`gemma, glm, mistral, granite`

The generated successor receipt correctly contained:

`gemma, qwen, mistral, granite`

The validator therefore failed with `RuntimeError: judges`.

This is a controller/validator migration defect. It is not a semantic or scientific failure of Qwen, Gemma, Mistral, Granite, the 287-candidate pool, or the scenario freeze.

## v1.9.1 correction

Release:
- package: `ALICE_MC10D_QWEN_SUCCESSOR_REFREEZE_BREADTH_v1.9.1.zip`
- SHA-256: `397EE77874D9D2AF7AE197AF5EC3FEB478D222930D46F22AE52C953B31023D72`
- launcher: `Start-ALICEMC10DQwenSuccessorRefreezeBreadthV191.ps1`
- SHA-256: `E03304DED31E0A9492DE111428B27A0044AEACD91BD7D245CC9D37C708154E02`
- build receipt SHA-256: `EE2227AF4F256087608AD8F441D48E66A0BE18A913C27FD2F6592F73CD7E5B3E`
- controller SHA-256: `7FEF25C8D9C9F623E4F5B88B6AABA95BB342AA12608473B6AC1E2419D6359ED9`
- selftest SHA-256: `66B9D628CF84DA64E9BFC7AFDC93F594C32E473F5FAC935652C151DA8D11A40A`

v1.9.1 changes only the copied validator assertion after exact v1.7 package/controller verification. It requires the legacy GLM family assertion to occur exactly once, replaces it with Qwen, recompiles the copied validator, and fails closed on drift. The original v1.7 package is not modified.

Existing exact Qwen adapter evidence is reused. There is no Qwen/Gemma/Mistral/Granite inference rerun. Slot 63 attempt 4 remains forbidden. Slot 64 remains resolved and is not regenerated.

## Audit

- package CRC: pass
- deterministic byte-identical rebuild: pass
- fresh extraction compile: pass
- fresh extraction selftest: pass
- Qwen 84-file evidence audit: pass
- Qwen score recompute: pass
- v1.7 patch patterns: 8/8
- legacy pointwise-validator GLM→Qwen regression test: pass
- 18 falsification families preserved
- 8 zero-tolerance vetoes preserved
- E0 never generated

## Next action

Run v1.9.1 once. On success it must emit:
- `ALICE_MC10D_POINTWISE_READY_QWEN_SUCCESSOR_v1.9.1.zip`
- `ALICE_MC10D_POINTWISE_DISPATCH_AUTHORITY_v1.9.1.json`

Then build/run the blinded private pointwise screen bound to those exact hashes. Full simulation/falsification remains blocked until pointwise freezes.
