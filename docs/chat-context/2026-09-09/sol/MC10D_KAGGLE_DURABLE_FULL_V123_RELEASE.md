# MC10D Kaggle durable full-coverage v1.2.3 release

Date: 2026-09-09

Canonical main remains frozen at `0abaed85873c3f8de04765847eb7700b0e20433f`.

## Exact v1.2.2 provider failure

The existing Kaggle kernel `mkrayanyan/alice-mc10d-sf-mis-abc67430d799` failed before the v1.2.2 worker evidence boundary.

Provider logs show:
- failure at about 1.125 seconds;
- exact traceback at worker main binding;
- exact stop: `Stop: context amendment binding`;
- controller embedded key: `technical_context_capacity_amendment_id`;
- v1.2.2 worker required key: `context_capacity_amendment_id`.

Therefore:
- v1.2.2 public canary never started;
- v1.2.2 private candidate attempts = 0;
- the previous terminal classification "public capacity canary failed" was not authoritative;
- the v1.2.2 5400-second ledger charge was a missing-receipt fallback, not measured provider runtime.

The Kaggle log's last event is about 6.921 seconds. v1.2.3 conservatively records 60 seconds for v1.2.2.

## Corrected current safety accounting

- v1.2.1 authoritative worker elapsed: 5436.12148728 seconds -> corrected local charge 5460 seconds.
- v1.2.2 provider log terminal evidence: <=6.921 seconds -> corrected local charge 60 seconds.
- corrected migrated local spend: 5520 seconds.
- local safety budget total: 21600 seconds.
- protected account reserve: 3600 seconds.
- local budget remaining in current window: 16080 seconds.

These are local safety-accounting values, not claims about Kaggle UI billing.

## v1.2.3 release identity

- package: `ALICE_MC10D_KAGGLE_DURABLE_FULL_MC10D_v1.2.3.zip`
- package SHA-256: `4BA1F35B77ADDA667EEEC67430C123A8D4545EA09D221154AA28430BFCDD51CD`
- launcher: `Start-ALICEMC10DKaggleDurableFullMC10DV123.ps1`
- launcher SHA-256: `98501C5D67290BA59382D01F520DF6F6679E51E8B914BB83EF263B5468E8E939`
- build/audit SHA-256: `6BEC94F0CC8D173916C3083C903386FAB5E3CF94DE74F9554584A1F418B1E605`
- deterministic package rebuild: byte-identical.
- clean extraction, package-manifest verification, external compile gate, ZIP CRC, and 20-test offline regression: PASS.
- no live Kaggle work was performed while building or auditing v1.2.3.

## v1.2.3 binding and evidence repair

- canonical embedded amendment key is `technical_context_capacity_amendment_id` everywhere.
- worker writes `worker-entered.json` before embedded-job decoding/binding.
- broad startup failure path preserves job failure/result evidence after output initialization.
- provider logs are collected after terminal status before quota accounting.
- Ollama service log is written into durable output.
- quota accounting uses job-result elapsed, then provider-log max event time, with soft-limit fallback only as a last resort.

The v1.2.3 regression suite constructs a real controller job, renders the real Kaggle script, decodes the real embedded job, crosses the worker binding boundary, and requires both `worker-entered.json` and `worker-started.json`. This exact test would have caught v1.2.2.

## Technical context-capacity amendment

`MC10D_TECHNICAL_CONTEXT_CAPACITY_REPAIR_V1` remains preregistered:
- service context 8192 -> 32768 for all families;
- Qwen explicit num_ctx 8192 -> 32768.

No change to:
- exact 216 survivor payload;
- model tags/digests;
- scientific system/scenario/falsification prompts;
- 32+32 scenario geometry;
- output schemas;
- 18 falsification families;
- 8 vetoes;
- parent aggregation gates;
- full four-family coverage.

Private request cross-check:
- Mistral exact parent v2.0.0 body;
- Granite exact parent v2.0.0 body;
- Gemma exact parent v2.0.0 body;
- Qwen exact parent body except preregistered num_ctx 8192 -> 32768.

## Bounded live qualification architecture

For each family, before scale-out:
1. public context probe: truly token-count gated, small output, no private content;
2. public exact-32 schema probe, no private content;
3. exactly one private candidate pilot.

The first family job carries at most one private task and has a 2700-second soft limit.
After a valid pilot, later jobs carry at most 12 candidate-family obligations with at most a 3600-second soft window.
No new candidate starts with fewer than 900 seconds remaining.

A technical-invalid private pilot blocks further automatic spend for that family.

## Remaining MC10D pipeline

1. live MC8 re-verification
2. full four-family simulation/falsification
3. unaccepted synthetic simulation ledger
4. exact v2.0.0 gates / 18 falsification families / 8 vetoes
5. MC10D simulation/falsification freeze v2.0.9
6. MC10D scientific decision freeze v2.0.9
7. MC10E SELECT/ABSTAIN proposal v2.0.9
8. final content-addressed Drive round-trip durability

Stop boundary remains before MC10E execution.
A-SYN acceptance/promotion remain false. Training remains false. Stage G remains open.
