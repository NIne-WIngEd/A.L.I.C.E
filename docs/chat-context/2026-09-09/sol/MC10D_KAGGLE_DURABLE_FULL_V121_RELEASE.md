# MC10D Kaggle durable full-coverage v1.2.1 release

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.2.0 observed stop

The v1.2.0 launcher passed all offline gates, then stopped locally before any Kaggle kernel push:

- observed payload SHA-256: `FEB8412F0932EE8B5AF076FAE5D9F69A229205EEA1253E91D56DA0C04BEB73FC`
- frozen v2.0.6.1 authority: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- stop: `exact simulation payload hash mismatch`
- exit: 76

The stop occurred before `locate_kaggle()`, Kaggle preflight, or any `kernels push`. Therefore v1.2.0 consumed no new Kaggle GPU execution time.

Root cause: v1.2.0 reused the parent line-oriented canonical JSON serializer, which appends a trailing newline, for the v2.0.6.1 payload-lineage hash. The post-pointwise v2.0.6.1 freeze defined the exact payload SHA over compact canonical JSON with no trailing newline. Exact simtask content was not changed.

## v1.2.1 release

- package: `ALICE_MC10D_KAGGLE_DURABLE_FULL_MC10D_v1.2.1.zip`
- package SHA-256: `CCAD30102A88FC414E625F57606F7482B01BCBB778E3811F3EF2B4DC0045AFF9`
- launcher: `Start-ALICEMC10DKaggleDurableFullMC10DV121.ps1`
- launcher SHA-256: `9214EE2C1B233A3EBB6B9D4B8C6E83EFF9EE7628FD9907470287A00FE5302FFB`
- build audit SHA-256: `16F24C90D666E60554BACF46CF1FB7859B9BFA63A471DDFD85A24C38041BC3F6`
- deterministic rebuild: byte-identical
- clean extraction + py_compile + 16-test offline regression: PASS

v1.2.1 separates:
- internal line-terminated JSON/JSONL serialization;
- frozen scientific canonical serialization used for v2.0.6.1 payload lineage.

The live payload gate now uses the exact no-trailing-newline v2.0.6.1 byte contract.

## Checkpoint hardening for the reported ~7h quota

- one active Kaggle 2xT4 kernel;
- local GPU-session budget 6h;
- 1h reported account reserve;
- max remote soft window 2h;
- no new candidate starts with <30m left;
- content-addressed Drive checkpoint after every terminal kernel;
- upload + readback + exact SHA required before `CHECKPOINT_DURABLE=true`;
- terminal provider/worker failures first preserve any independently verified candidate-boundary evidence and quota spend to Drive;
- a complete remote `tasks_attempted` receipt is required before unattempted obligations may continue automatically;
- missing attempt-count authority fails closed to avoid retrying an unknown already-attempted candidate.

## Scientific authority unchanged

- parent v2.0.0: `9DB60EE723F4804F2443AF0A4290D40CC14865D28DF4867634B76688AA1812AA`
- input freeze: `F232721A412D24304BE05DA100BFE27D6CA6B0124E7D1AC504AC14D85197EF01`
- hybrid pointwise freeze: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- simulation payload: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- 216 survivors; 71 pointwise rejects; 24/24 packets
- full four-family attempt coverage required
- early-fail shortcut disabled
- technical invalidity is not semantic PASS/FAIL
- A-SYN acceptance/promotion false
- model training false
- MC10E execution false

The same controller composes all remaining safe MC10D steps: live MC8 re-verification, full simulation/falsification, unaccepted synthetic ledger, exact v2.0.0 gates/falsification/vetoes, MC10D sim/falsification freeze, MC10D decision freeze, MC10E SELECT/ABSTAIN proposal, and final Drive round-trip durability. It stops before MC10E execution.
