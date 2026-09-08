# MC10D v1.9.4 success and v2.0.0 execution boundary

Date: 2026-09-07

## v1.9.4 completed successfully

The exact post-refreeze recovery exited 0 and closed the pre-pointwise gate.

Frozen outputs:

- pointwise dispatch authority SHA-256: `277C41AE07A96EBFFC714D55D8DAB9CB2E2E5BD922814C7D2EFE8BE04519800D`
- pointwise layout SHA-256: `63FD0B3464BE0F41296D8B14C0A141C286EF5DB1B345311A5DEA74BE20AC3A2A`
- private pointwise input freeze SHA-256: `F232721A412D24304BE05DA100BFE27D6CA6B0124E7D1AC504AC14D85197EF01`
- input-freeze receipt SHA-256: `6254D296CDD19463E9F277AAC48BC7096E5BD3967243A886242C3C03925745DD`

Scientific state remains:

- effective pool = 287
- deferred slot = 1
- four-family successor binding = Gemma + Qwen + Mistral + Granite
- A-SYN accepted = 0
- A-SYN promoted = 0
- model training = false
- private pointwise not yet started
- MC8 sealed

## Final v2.0.0 release

The first pre-release v2.0.0 archive was withheld after a resume defect was found: after validated output had been retrieved and its Kaggle Kernel deleted, a controller restart could attempt remote reconciliation instead of consuming the durable local evidence.

The final source fixes that class. A validated terminal output now resumes directly from the Vault without contacting Kaggle. A preserved `INFRASTRUCTURE_OR_WORKER_STOP` result remains a deterministic failure on resume.

Release artifacts:

- package: `ALICE_MC10D_PRIVATE_POINTWISE_SIMFALSIFY_v2.0.0.zip`
- package SHA-256: `9DB60EE723F4804F2443AF0A4290D40CC14865D28DF4867634B76688AA1812AA`
- launcher: `Start-ALICEMC10DPrivatePointwiseSimFalsifyV200.ps1`
- launcher SHA-256: `1093E18BDC9486B8EB0026E435FB366049B8A9AF2F7C88C9B033396F6091472C`
- build receipt SHA-256: `EF7ACAB117E7E3F9DA6F4686E5A8D4B145A86588ED69CB7886A788A1F9C62158`

The stale pre-release package hash `0B555ACDF014C4D2D14DD92C6B17F1D7C43DD0DF01FF88DE84C054AD1EB3A12D` is superseded and must not be run.

## Final release audits

Passed:

- deterministic byte-identical second ZIP build
- ZIP CRC
- clean extraction
- Python compile gate
- full package selftest
- exact v1.9.4 parent revalidation
- 287 candidates / 24 packets / 24 UNKNOWN competitors
- 55,104 frozen probe upper bound
- 18 falsification families
- 8 zero-tolerance vetoes
- blinded pointwise payload with zero real candidate IDs, zero real packet IDs, and zero generator identity
- push-once / no blind repush
- durable validated-local-output resume without Kaggle
- infrastructure-stop preservation on resume
- Windows native process exit-code preservation

MC8 was not opened during the build audit.

## Execution order

`auto` performs:

1. exact v1.9.4 input verification;
2. blinded four-family private pointwise;
3. immutable pointwise freeze;
4. only then exact MC8 evaluator verification;
5. 64 × 3 simulation for pointwise survivors;
6. canonical 18-family falsification and eight-veto gate;
7. separate Synthetic Simulation Ledger;
8. MC10D decision freeze;
9. MC10E SELECT / ABSTAIN proposal.

Pointwise PASS, sim/falsification PASS, and MC10E SELECT_PROPOSAL do not themselves accept/promote A-SYN or authorize model training.

Next action: `RUN_MC10D_PRIVATE_POINTWISE_SIMFALSIFY_V200_AUTO`.
