# Phase 2 to Cognitive Kernel Memory Migration Plan

**Draft:** 0.2<br>
**Canonical authority decision:** Accepted 2026-08-03<br>
**Goal:** Unify Phase 2 authoritative memory and Phase 5 experience storage
without creating two competing sources of truth.

## 1. Current state

Phase 2 provides:

- durable memory rows;
- provenance;
- temporal state;
- correction/conflict relations;
- sensitive storage;
- candidate staging;
- lexical and semantic retrieval;
- deletion foundations.

Phase 5 provides:

- immutable Experience Ledger metadata;
- raw payload references;
- content-addressed storage;
- lifecycle decisions;
- tier movement;
- product/host/encryption-domain isolation.

The systems are intentionally separate today.

## 2. Accepted source-of-truth topology

The accepted migration target is:

- Experience Ledger: immutable evidence and action lineage;
- Claim Store: canonical append-only, bitemporal adjudicated knowledge;
- Phase 2 Memory Core: compatibility projection during shadow migration;
- episodes and cognitive models: versioned derivatives;
- search, graph, summary, cache, and current-state indexes: rebuildable
  derivatives.

Claim Store v1 uses new kernel contracts and tables in the existing host-local
SQLite architecture. It does not require a new database service.

## 3. Migration rule

Do not bulk-copy all Phase 2 rows into the Experience Ledger.

Use explicit roles:

- Experience Ledger: event and action lineage;
- payload store: opaque evidence bytes;
- claim store: versioned adjudicated knowledge;
- projections: current state and cognitive models;
- indexes: derived search structures.

## 4. Target flow

```text
Phase 3/4/7 event
    -> Experience Ledger event
    -> optional payload reference
    -> curation outbox
    -> memory candidate
    -> claim/episode/mission version
    -> current-state projection
    -> derived indexes
    -> Memory Context Packet
    -> model/action
    -> outcome event
```

## 5. Compatibility strategy

### Stage 0 — Freeze and inventory

- record exact Phase 2 schemas and callers;
- profile row counts, indexes, latency, and sensitive-store use;
- inventory every API import;
- define compatibility tests.

### Stage 1 — Introduce canonical kernel contracts

Add host-neutral contracts for:

- memory unit envelope;
- evidence binding;
- claim version;
- episode;
- context packet;
- retrieval trace;
- curation task and receipt.

No behavior changes.

### Stage 2 — Add event-to-candidate bridge

New Phase 5 evidence events create outbox tasks. Each task receives a durable
workflow ID and replayable event history. Tasks may stage candidates in a new
kernel candidate API. Existing Phase 2 authoritative memory remains the serving
source.

### Stage 3 — Add append-only claim versions

Introduce new Claim Identity, Claim Version, Claim Evidence, Claim Adjudication,
and Current Claim Projection tables through Cognitive Kernel contracts in the
existing host-local SQLite architecture.

Dual authoritative write is prohibited. The Claim Store transaction becomes the
canonical write. A durable outbox updates the Phase 2 compatibility projection
and all derived indexes.

### Stage 4 — Rebuild Phase 2 compatibility projection

Generate Phase 2-compatible rows from claim versions. Compare every field and
retrieval result against the original store.

### Stage 5 — Shadow serving

Run old and new retrieval in parallel. Log disagreements without changing model
context. Evaluate:

- result overlap;
- correct current state;
- temporal behavior;
- conflict expansion;
- classification;
- latency;
- token efficiency.

### Stage 6 — Controlled cutover

Enable the v4 serving plane for a limited capability profile. Retain the Phase 2
compatibility projection and rollback path.

### Stage 7 — Deprecate direct Phase 2 writes

All new writes pass through kernel contracts. Phase 2 write APIs become
compatibility adapters.

### Stage 8 — Archive legacy authority

After repeated rebuild and rollback tests, the original Phase 2 database becomes
a signed migration artifact rather than the active authority.

## 6. Sensitive memory

Sensitive payload migration requires:

- key and encryption-domain continuity;
- no plaintext export;
- row-by-row integrity verification;
- owner authorization;
- deletion-state preservation;
- access-event continuity;
- rollback.

## 7. Identifier mapping

Maintain permanent mappings between:

- Experience Event ID;
- payload reference;
- Phase 2 memory ID;
- claim version ID;
- episode ID;
- mission ID;
- projection version ID;
- deletion request ID.

Mappings are append-only and inspection-safe.

## 8. Rollback

Every migration stage records:

- source commit;
- source database digest;
- schema version;
- migration code digest;
- output digest;
- record counts;
- mismatch report;
- rollback procedure.

A cutover is invalid without a tested rollback.

## 9. Exit conditions

Migration is complete only when:

- all production writes use kernel contracts;
- new stores rebuild all compatibility projections;
- old/new shadow retrieval disagreements meet the ratified threshold;
- sensitive memory is verified;
- correction/deletion propagation works;
- performance targets pass;
- rollback passes after later writes;
- owner approval is recorded.
