# MC10D Kaggle durable full-coverage v1.2.0 release

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Release

- package: `ALICE_MC10D_KAGGLE_DURABLE_FULL_MC10D_v1.2.0.zip`
- package SHA-256: `CA0E924B662397B31DFF48613EACF329E90EDED22EA7BED53AC91DA07ECDF90A`
- launcher: `Start-ALICEMC10DKaggleDurableFullMC10DV120.ps1`
- launcher SHA-256: `19524629DC5CAF66CFAA47FE3891EEE742F398D7F92CD3078442640220D8EC17`
- build audit SHA-256: `E2791102256F4AC35F66537B5FE7929610A3406C3B4B1A990E8ACD65B45D9B19`
- deterministic rebuild: byte-identical
- clean extraction + py_compile + 14-test offline regression: PASS

## Scientific authority

- parent v2.0.0 SHA-256: `9DB60EE723F4804F2443AF0A4290D40CC14865D28DF4867634B76688AA1812AA`
- input freeze SHA-256: `F232721A412D24304BE05DA100BFE27D6CA6B0124E7D1AC504AC14D85197EF01`
- hybrid pointwise freeze SHA-256: `8495B3C4B3207511D615BD0B7788EB064114B3F763E7E4299EDCC1EE869617B7`
- simulation payload SHA-256: `7D217570C167575E24630E3BE150786B260C16518D0349EF2EF94E1A088A0925`
- simulation eligible: 216
- pointwise reject: 71
- packets covered: 24/24

v1.2.0 intentionally removes the provisional early-definitive-fail optimization from the unreleased v1.1.0 design. Every one of the 216 survivors must receive one attempted simulation/falsification obligation from all four frozen families. Family execution order (Mistral -> Granite -> Gemma -> Qwen) is scheduling only. Exact parent v2.0.0 prompts, schemas, profiles, models, request construction, gates and aggregation remain bound.

Technical-invalid evidence consumes the candidate-family obligation once, matching the parent no-semantic-retry behavior. It remains technical and aggregates as `SIMFALSIFY_DEFER_TECHNICAL`; it is never converted into PASS or FAIL.

## Current 7-hour Kaggle quota policy

Owner reported approximately seven GPU hours remaining.

- local budget: 6h
- account reserve: 1h
- one active Kaggle GPU kernel
- max soft window: 2h
- no new candidate starts with <30m left
- exact simulation/falsification request timeout: 1800s
- at most 40 candidate obligations supplied per kernel
- durable Drive checkpoint after every terminal kernel
- `CHECKPOINT_DURABLE=true` only after upload, readback and exact SHA-256 match
- completed verified candidate-family obligations are never recomputed

Exit 78 protects the local 1h reserve. Exit 77 records explicit Kaggle quota rejection. After an actual quota refresh, the same launcher is resumed with `-NewQuotaWindow`, which resets provider-budget metadata only.

## Composed remaining MC10D pipeline

The same controller runs all remaining steps that can safely be composed after the frozen pointwise boundary:

1. live local MC8 hash re-verification against the exact frozen receipt;
2. full four-family simulation/falsification;
3. unaccepted synthetic simulation ledger;
4. exact parent L2 gates, 18 falsification families and 8 vetoes;
5. MC10D simulation/falsification freeze;
6. MC10D scientific decision freeze;
7. MC10E SELECT/ABSTAIN proposal;
8. final content-addressed Drive round-trip durability.

The controller stops before MC10E execution. It does not accept/promote A-SYN, train a model, close Stage G or replace Phase 2.
