# MC10D Kaggle durable full-coverage v1.2.2 release

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## v1.2.1 durable source boundary

The first real v1.2.1 Mistral simulation kernel completed a bounded two-hour window and produced recoverable candidate-boundary evidence.

Authoritative forensic result:
- source job ID: `4953D4C9DC03F29F71270786985FD29E583922CF9FF1423FF17D48BE5AFED880`
- source family: Mistral
- tasks supplied: 40
- tasks attempted: 23
- valid semantic results: 0
- technical-invalid results: 23
- exact common stage: `scenario_batch_1`
- exact common error: `Stop:non-stop finish`
- source quota spend: 7200 seconds
- source durable checkpoint SHA-256: `98B98D8475C56F32C9328AA174DD618DDD895E14C49B35C1F5DFBA91A65F5B15`

The previously reported `MISSING_JOB_RESULT` status was a controller lookup defect. The actual nested Kaggle `job-result.json` exists and independently proves `tasks_attempted=23`. The 23 local candidate records match the recovered remote records.

No v1.2.1 technical-invalid result has semantic PASS/FAIL authority.

## Technical capacity amendment

v1.2.2 preregisters one global technical execution amendment:

`MC10D_TECHNICAL_CONTEXT_CAPACITY_REPAIR_V1`

Changes:
- Ollama service context: 8192 -> 32768 for all four families.
- Qwen explicit `options.num_ctx`: 8192 -> 32768.

Unchanged:
- exact 216-survivor simulation payload;
- model tags and manifest digests;
- system/scenario/falsification prompts;
- 32+32 scenario batching;
- output schemas;
- 18 falsification families;
- 8 zero-tolerance vetoes;
- Mistral/Granite native profiles;
- Gemma thinking profile;
- all Qwen sampling/seed options other than `num_ctx`;
- exact parent v2.0.0 aggregation gates;
- full four-family attempt coverage;
- no early-fail shortcut;
- technical invalidity never becomes semantic failure.

The exact 23 source Mistral blind IDs are preserved append-only under `superseded/v121-context-8192/mistral/` and receive exactly one replay authorization under the repaired envelope. After a v1.2.2 candidate-family attempt exists, normal no-semantic-retry behavior applies.

## Public capacity canary

Before the first private candidate for each family, that family must pass a public-only canary:
- no private candidate content;
- no owner E0;
- no MC8 material;
- same 32-result scenario schema;
- request >= 44,000 bytes;
- model-advertised context >= 32768;
- `prompt_eval_count > 8192`;
- `done_reason=stop`;
- `prompt_eval_count + eval_count <= 32768`.

A failed canary runs no private candidate for that family. If a repaired private kernel yields attempted candidate evidence but zero valid results, the controller makes the evidence durable and stops further automatic GPU spend.

## Quota / checkpoint policy

The existing quota window is migrated instead of reset:
- local budget total: 21600 seconds (6 hours);
- preserved account reserve: 3600 seconds;
- already spent/charged: 7200 seconds;
- maximum new kernel: 5400 seconds (90 minutes);
- minimum time before starting another candidate: 1800 seconds;
- one active 2xT4 Kaggle kernel at a time.

Every terminal kernel is made content-addressed Drive-durable only after upload, readback and exact SHA-256 verification.

`-NewQuotaWindow` is authorized only after Kaggle quota actually refreshes.

## v1.2.2 release identity

- package: `ALICE_MC10D_KAGGLE_DURABLE_FULL_MC10D_v1.2.2.zip`
- package SHA-256: `7C384AA619E0260C62F35E10CEFDD6DB6BA2DDD724E124C7F8D5D0AACE510B80`
- launcher: `Start-ALICEMC10DKaggleDurableFullMC10DV122.ps1`
- launcher SHA-256: `97950287D52D99650515190A5E750062253B0BFE77473289A043F864E9CAC1DA`
- build/audit SHA-256: `058FB0E71A5258CA2A36EAD0A37990AA514316489E8ED7FADF7850CE5605E00B`
- deterministic rebuild: byte-identical
- parent controller/worker: byte-exact to v2.0.0
- clean extraction: Python compile PASS
- ZIP CRC: PASS
- offline regression: 20/20 PASS

No live Kaggle work was performed while producing this release.

## Remaining composed MC10D pipeline

The same controller composes:
1. live MC8 re-verification;
2. full four-family simulation/falsification;
3. unaccepted synthetic simulation ledger;
4. exact parent v2.0.0 gates, 18 falsification families and 8 vetoes;
5. MC10D simulation/falsification freeze v2.0.8;
6. MC10D scientific decision freeze v2.0.8;
7. MC10E SELECT/ABSTAIN proposal v2.0.8;
8. final content-addressed Drive round-trip durability.

The controller stops before MC10E execution. A-SYN acceptance/promotion remain false. Model training remains false. Stage G remains open.
