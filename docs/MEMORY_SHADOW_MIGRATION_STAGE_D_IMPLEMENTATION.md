# Memory Shadow Migration Stage D Implementation

**Version:** 1.1.0
**Status:** deterministic historical-backfill prototype operational; Stage F+G successor research admitted
**Baseline:** PR #84 merge `e5edd541dc06f486904d94ffaca1d0bb95be36ea`; merged by PR #85 at `401d828a3d03ae42553c2719151fc9a4e0837917`
**Scope:** backend-neutral historical backfill contracts, replay safety, lineage preservation, reconciliation, and synthetic evaluation

## 1. Material state

PR #83 made Stage A inventory/registration and Stage B deterministic read-only adapters prototype-operational.

PR #84 made Stage C destination-candidate profiling and Stage E nonproduction shadow-read evaluation prototype-operational. The Stage C+E evaluation recommended eligibility for the next research gate while Phase 2 remained the released authority and test oracle.

This tranche implements Stage D as a deterministic historical-backfill prototype. The public repository provides the contracts, batch executor, checkpoint model, idempotency semantics, reconciliation receipts, and synthetic evaluation needed to perform historical migration safely.

The synthetic evaluator does not execute private historical payload backfill. A real private batch requires an explicit owner-authorized manifest that identifies the source registration, destination candidate, source snapshot digest, mapping generation, custody context, and authorization reference.

## 2. Stage D invariants

Every backfill stream records:

- source registration;
- destination candidate;
- source snapshot digest;
- mapping version;
- workload class and authorization reference when applicable;
- adaptive preferred batch size rather than a universal maximum;
- authority and serving effects for the active profile.

Every record records:

- source checkpoint;
- source-record digest;
- mapped-record digest;
- mapping version;
- deterministic idempotency key;
- provenance state;
- evidence lineage;
- deletion lineage;
- accepted, rejected, quarantined, or ambiguous disposition;
- a digest-bound mapping receipt.

Every batch records:

- first and last source checkpoints;
- accepted, rejected, quarantined, and ambiguous counts;
- applied and replay-duplicate counts;
- destination-side rejection/quarantine/ambiguity counts;
- record and destination receipt digests;
- evidence-lineage digest;
- deletion-lineage digest;
- reconciliation digest;
- final batch receipt digest.

## 3. Provenance and ambiguity

Stage D never invents missing provenance.

A source record whose provenance state is missing cannot be accepted into a destination candidate. It must remain explicitly rejected, quarantined, or ambiguous with a reason.

Partial provenance is represented as partial. It is not silently upgraded to complete.

Deletion lineage is carried independently from ordinary evidence lineage so later graph, vector, object, cache, dataset, replay, model, replica, and backup projection work can preserve deletion obligations.

## 4. Idempotency and replay

An idempotency key is derived from the manifest, source identity, source checkpoint, source digest, mapped digest, and mapping version.

Replaying the same accepted batch against an idempotent destination returns duplicate receipts rather than creating a second logical migration.

Checkpoint state binds completed batch receipt hashes and cumulative disposition counts. Resume logic can therefore prove exactly which deterministic work has already been reconciled.

## 5. Backend neutrality

The Stage D executor accepts a destination-writer function behind the historical-backfill contract. The repository does not select one database, topology, host, graph engine, workflow engine, vector engine, object store, or deployment profile as the permanent migration destination.

The current reversible M2 polyglot candidate may be used by synthetic evaluation. Other registered candidate compositions remain valid under the same logical contracts.

## 6. Private-data execution boundary

The Stage D code is capable of operating under an `owner_authorized` manifest, but this repository tranche does not automatically discover, open, copy, upload, or persist private historical payloads.

A later execution may use owner-authorized private batches when:

1. the exact source registration and snapshot are selected;
2. the destination candidate is registered;
3. an authorization reference is bound to the manifest;
4. the mapping version is fixed for that run;
5. custody, deletion, rollback, and report locations are known;
6. results remain auditable through batch and checkpoint receipts.

This is an execution gate for private owner state, not a prohibition on Stage D capability or research.

## 7. Authority boundary

Stage D populates a shadow destination candidate only.

Phase 2 remains the released canonical authority for its current profile. The Stage D prototype does not activate controlled write mirroring, canary authority, production serving, canonical transfer, cutover, Phase 2 retirement, Friday state transfer, or P5.1e storage admission.

Those successor capabilities remain independently researchable and may activate through their own evidence profiles.

## 8. Synthetic evaluation

`src/cognitive_kernel/shadow_migration_stage_d_evaluation.py` performs a deterministic synthetic run that verifies:

- accepted, rejected, quarantined, and ambiguous source outcomes;
- deletion-lineage preservation;
- missing-provenance quarantine;
- destination application receipts;
- idempotent replay;
- deterministic reconciliation;
- digest-bound checkpoint continuation;
- external report custody;
- unchanged Phase 2 authority and serving state.

The synthetic report is written outside public Git.

## 9. Exit and rollback

Rollback is file-level and evidence-preserving:

1. remove Stage D source, policy, tests, and documentation;
2. restore Stage C+E successor-state text, migration-plan state, catalog, README, admission policy, and package exports;
3. retain external reports when needed for audit or remove them under the owner-selected report-retention policy;
4. retain Phase 2, Stage A+B, and Stage C+E evidence unchanged.

## 10. Successor gates

After Stage D prototype evidence is accepted, independent next gates may include:

- owner-authorized execution of real historical backfill batches;
- Stage F controlled write-mirroring research with one explicit canonical writer;
- Stage G graph/vector/workflow generation research;
- later Stage H canary authority under owner-ratified scope.

No successful Stage D result itself grants production authority.

## 11. Successor state

PR #85 established Stage D deterministic backfill contracts and synthetic evaluation as an accepted migration oracle. Stage F+G may consume explicit canonical outbox changes and registered source generations in nonproduction research profiles, but they do not reinterpret Stage D results as production authority or bypass private-data authorization, deletion, reconciliation, rollback, or external-report evidence.
