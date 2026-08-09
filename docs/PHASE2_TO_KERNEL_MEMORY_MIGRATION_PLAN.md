# Phase 2 to Cognitive Fabric Memory Migration Plan

**Version:** 1.7.0
**Status:** Stage A+B, Stage C+E, deterministic Stage D, Stage F controlled-mirroring, and Stage G projection-generation prototypes operational; persistent F+G reference integration operational; private execution, live candidate integrations, and production authority stages independently gated
**Source baseline:** Released Phase 2 Memory Core
**Destination:** Backend-neutral Memory Architecture v4.1

## 1. Objective

Migrate Phase 2 memory into a claim-centered, evidence-linked, polyglot cognitive fabric without creating an ambiguous source of truth, losing provenance, leaking private state, or removing a tested fallback before successor evidence exists.

The target may combine embedded, relational, distributed, event, graph, vector, object, workflow, and model systems. The plan does not assume one database or one host.

## 2. Migration principles

1. Released Phase 2 source and tests remain a compatibility baseline and test oracle.
2. Every target authority is registered before accepting production writes.
3. One component is canonical for each authority type at a given generation.
4. Secondary writes are projections, replicas, outbox deliveries, or shadow authorities with explicit status.
5. Dual-write periods require idempotency, reconciliation, divergence metrics, and rollback.
6. Private payloads and owner state remain within authorized custody.
7. Corrections and deletion lineage migrate before production cutover.
8. Graph, vector, summary, cache, model, and episode systems remain provenance-linked derivatives unless explicitly ratified otherwise.
9. Research and prototypes may proceed in parallel. Cutover proceeds by evidence.

## 3. Target planes

The migration may populate:

- Experience/Event Fabric;
- Claim Authority and current-state projection;
- Cognitive Graph;
- lexical, vector, and multimodal indexes;
- object and archive storage;
- durable workflow runtime;
- dataset and model registries;
- owner-authorized replicas and multi-device synchronization;
- inspection, deletion, and rollback services.

A single-node edge profile and a distributed profile implement the same logical contracts.

## 4. Stages

### Stage A — Inventory and registration

Register every source store, schema, generation, authority role, encryption domain, data classification, record count, integrity state, and deletion capability.

No source file becomes authoritative merely because it is discovered.

### Stage B — Contract adapters

Implement read adapters that translate Phase 2 records into neutral Evidence Event, Claim Identity, Claim Version, Evidence Relation, Conflict, Correction, and Deletion records.

Adapters record loss, ambiguity, and unsupported semantics.

### Stage C — Destination candidates

Build one or more destination backends behind `EvidenceLog`, `ClaimAuthority`, `CurrentClaimProjection`, `GraphProjection`, `VectorProjection`, `PayloadStore`, and `DeletionCoordinator` contracts.

Candidates may include embedded and distributed systems. Selection follows benchmark and reliability evidence.

### Stage D — Historical backfill

Backfill in deterministic batches with:

- source checkpoint;
- record digest;
- mapping version;
- idempotency key;
- accepted, rejected, quarantined, and ambiguous counts;
- evidence and deletion lineage;
- reconciliation receipt.

Backfill never invents missing provenance.

### Stage E — Shadow reads

Run current Phase 2 reads and successor reads against the same synthetic and authorized workloads.

Compare result quality, authority correctness, conflict handling, latency, staleness, deletion, privacy, and explanation.

### Stage F — Controlled write mirroring

Keep one canonical writer. Publish canonical changes through outbox records to successor projections or a shadow authority.

If a successor is selected as canonical during canary, reverse projection to Phase 2 may continue for rollback. The authority transition is explicit.

### Stage G — Graph, vector, and workflow build

Construct graph and vector generations from registered claims and evidence. Launch durable projection, deletion, and repair workflows. Record build manifests and deletion watermarks.

Backend durability alone is not sufficient Stage G exit evidence. Before Stage G can close, the successor cognitive-memory fabric must also pass owner-authorized and synthetic qualification covering the learned Memory Formation Model, the Elaina Identity / Personality Model, Rayan host learning without Elaina-identity drift, deterministic authority/routing, per-layer behavior, cross-layer projection consistency, retrieval/context fusion, conflict and temporal correction, deletion/revocation propagation, A.L.I.C.E./Friday isolation, failure/recovery, concurrency, rebuild, rollback, scale, and README/governance promise alignment. Stage H is not eligible until this integrated Stage G qualification is accepted.

### Stage H — Canary authority

Enable a bounded owner-authorized profile for selected namespaces, claim classes, or missions.

Canary evidence includes divergence, rollback rehearsal, recovery, failure injection, privacy, and deletion verification.

### Stage I — Cutover

Freeze the old canonical position, drain outboxes, reconcile, verify integrity, record cutover manifest, activate the new authority generation, and preserve a tested rollback window.

### Stage J — Compatibility operation

Phase 2 becomes a read-only compatibility projection, fallback, export source, or retired archive according to profile.

Stage J must itself be evaluated and owner-accepted. For this migration, final replacement or retirement is complete only after Stage J acceptance; neither Stage H canary authority nor Stage I cutover alone completes final Phase 2 replacement.

It is not deleted until retention, rollback, Stage J acceptance, and owner authority permit.

## 5. Distributed and multi-device semantics

Migration records authority namespace, owner partition, shard or stream position, logical clock, causal order, device/cluster identity, replication conflict, and reconciliation.

Partition strategy cannot leak owner or product state. Cross-owner federation requires an explicit export rather than implicit shared storage.

## 6. Deletion migration

Before cutover:

- import all active deletion requests and tombstones;
- verify ordinary retrieval exclusion;
- propagate to graph, vector, object, cache, dataset, replay, model, replica, and backup manifests;
- test archive restore with deletion replay;
- disclose model-influence limitations;
- rehearse rebuild or retirement of noncompliant derivatives.

## 7. Rollback

Rollback restores the prior authority generation or a known-good successor snapshot, replays accepted canonical events, reapplies deletion lineage, verifies current state, and records divergence.

Rollback never silently discards writes accepted after cutover. Compensation or forward repair is used when reversal would lose valid authority history.

## 8. Exit evidence

A production cutover requires:

- exact source and destination hashes;
- registered backends and profiles;
- mapping and reconciliation reports;
- synthetic and authorized benchmark results;
- failure and recovery tests;
- product-isolation and privacy review;
- deletion and restore verification;
- public-claim update;
- owner or mission authority appropriate to consequence.

## 9. Current state

The M2 contract sequence is closed at the implemented-contract and reversible-prototype level.

A deterministic synthetic cross-prototype evaluation has passed across Claim Authority, shadow adjudication, projection, bounded serving, durable workflow, and deletion propagation.

The admission review authorized preparatory read-only work for Stage A, Stage B, Stage C, and Stage E. PR #83 made Stage A inventory/registration and Stage B deterministic read-only adapters prototype-operational. PR #84 made Stage C destination-candidate profiling and Stage E deterministic synthetic or separately owner-authorized shadow-read evaluation prototype-operational.

The Stage D tranche adds deterministic historical-backfill manifests, source checkpoints, record digests, mapping versions, idempotency keys, accepted/rejected/quarantined/ambiguous accounting, evidence and deletion lineage, reconciliation receipts, and checkpoint continuation. Its repository evaluator uses synthetic records only. Real private historical batch execution requires an explicit owner-authorized manifest and remains separately auditable.

The Stage F+G successor tranche adds a nonproduction controlled-mirroring profile over an explicit canonical writer/outbox stream and generation-bound graph, vector, and workflow build receipts with deletion watermarks. Its repository evaluator is synthetic. Phase 2 remains the canonical writer and current released authority.

Owner-authorized private backfill, expanded mirroring, canary authority, canonical authority transfer, production influence, cutover, Phase 2 retirement, and P5.1e unblock remain available under their own evidence and authority gates.

## Persistent Stage F+G integration evidence after PR #86

The persistent integration profile proves restart/replay durability and persistent projection-receipt semantics through a SQLite compatibility/reference oracle. SQLite is not selected as the migration destination. KurrentDB, Neo4j, Qdrant, Temporal, and later alternatives remain independently evaluable candidates. Phase 2 remains the canonical writer/current released authority. Live candidate integrations, owner-authorized private Stage D execution, full Stage G cognitive-memory qualification, Stage H bounded canary review, canonical transfer, production serving, cutover, Stage J compatibility acceptance, Phase 2 final replacement/retirement, and P5.1e remain separate evidence gates.

The owner-ratified identity and host-learning boundary is recorded in `docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md`: A.L.I.C.E. is an Elaina-derived clone, Rayan is its owner/host, ordinary Rayan learning may update host and relationship models but not the core Elaina-derived identity anchor, and the Memory Formation Model remains a separate learned non-authoritative component.
