# MC10D v1.9.3 Gemma schema stop and v1.9.4 recovery

Date: 2026-09-07

## v1.9.3 observed state

v1.9.3 successfully verified the exact v1.9.2 pointwise-ready bundle SHA-256:

`5CC3C64EF4E64D75BD41E860F34CCD8FE8C882EEDBB6F6C6D0F68CB13218EF30`

It then proved:
- archive shape = flat historical v1.7 layout;
- effective pool = 287;
- successor families = Gemma + Qwen + Mistral + Granite;
- scenario probe upper bound = 55,104;
- extracted ready tree equals the atomically published canonical Vault tree, 83 files.

No private pointwise, MC8 access, full simulation, A-SYN acceptance/promotion, or training occurred.

The run stopped at `Gemma identity`.

Terminal transcript SHA-256:
`3252AE39303CB1D175149FDA2DDC6976DAAB30CE89C00E279741CAD2F640A98B`

## Root cause

The v1.9.3 ready-bundle validator had already required the exact Gemma identity from the content-addressed successor binding:

- tag: `gemma4:31b-it-q4_K_M`
- digest: `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`
- profile: thinking on

After that proof, v1.9.3 separately required the older preserved Gemma qualification-result JSON to contain top-level model and model-digest fields. The preserved result schema may omit those fields. v1.9.0-v1.9.2 intentionally treated those legacy identity fields as optional while requiring any present identity field to match.

Therefore v1.9.3 confused schema absence with contradictory identity evidence.

This was a deterministic compatibility-validator defect, not a Gemma scientific failure.

## v1.9.4 correction

v1.9.4 does not rerun the historical refreeze.

The exact v1.9.2 successor binding remains the required identity authority.

The preserved Gemma 16-row public-calibration evidence is rescored and must remain:
- verdict matches = 14/16;
- critical decision matches = 6/7;
- mandatory hard anchors = 5/5;
- Q01 = PASS;
- Q03 = HOLD.

For the legacy Gemma result JSON:
- qualification_passed must be true;
- family, if present, must be gemma;
- each present model/model_tag/tag must match the frozen tag;
- each present model_digest/digest must match the frozen digest;
- qualified_profile, if present, must match thinking-on;
- missing identity fields are schema absence only;
- conflicting identity fields fail closed.

## v1.9.4 release

- package: `ALICE_MC10D_QWEN_SUCCESSOR_POSTREFREEZE_RECOVERY_v1.9.4.zip`
- package SHA-256: `215C671664B560B8DD13EAD150B7D8BEDFE6A9A8D38DD8E4CCB9BD98F7D3906D`
- launcher: `Start-ALICEMC10DPostRefreezeRecoveryV194.ps1`
- launcher SHA-256: `DCD9AD4BD2267BB1141927097E485B1B66AD3A19DBF6E86D21130980B0B6ADEA`
- build receipt SHA-256: `00879C351227740512DE208DA1F3E272E8EDAC0429D4D58C9E54D12AC6189C8D`

Audit:
- package manifest verified;
- ZIP CRC pass;
- deterministic byte-identical package rebuild;
- fresh-extraction compile pass;
- fresh-extraction selftest pass;
- sparse legacy Gemma-result regression pass;
- conflicting result identity fail-closed pass;
- conflicting bound identity fail-closed pass;
- flat/wrapped archive support;
- ambiguous archive fail-closed;
- deterministic output ZIP rebuild pass;
- historical refreeze invocation absent.

## Frozen science

- effective pool = 287
- deferred = 1
- bound judges = Gemma, Qwen, Mistral, Granite
- scenario probe upper bound = 55,104
- E0 generated = false
- high-impact future A-SYN minimum independent E0 families = 3
- strong-anchor tier = 4+
- falsification families = 18
- zero-tolerance vetoes = 8
- private pointwise = not started
- MC8 = sealed
- full simulation = not started
- A-SYN accepted/promoted = 0/0
- training = false

## Next action

Run v1.9.4 against the exact existing v1.9.2 ready ZIP. On success it emits the private-pointwise dispatch authority and a single exact private-pointwise input freeze. The next stage is then the actual blinded private pointwise screen.
