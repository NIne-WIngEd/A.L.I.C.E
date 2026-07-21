# Phase 2 — Memory Core Architecture

**Status:** P2.0 foundation contract
**Phase 1 dependency:** Frozen, read-only evidence layer
**Owner:** MK Rayan

## 1. Purpose

Phase 2 turns trusted Phase 1 evidence and explicitly approved personal knowledge into a durable,
structured, inspectable, correctable, and deletable memory system.

Phase 2 does not replace or mutate Phase 1. Phase 1 remains the validated evidence layer.

## 2. Core boundary

The architectural boundary is:

```text
Phase 1 — Evidence
    |
    | read-only references
    v
Phase 2 — Memory
```

Phase 1 owns:
- verified extraction;
- deterministic chunks;
- source and content hashes;
- provenance catalogs;
- lexical, semantic, and hybrid evidence retrieval;
- grounded-response verification.

Phase 2 owns:
- durable memory records;
- memory lifecycle;
- memory provenance links;
- correction and supersession;
- temporal state;
- conflict state;
- memory inspection;
- deletion;
- memory-specific indexes;
- memory access control.

Phase 2 MUST NOT rewrite Phase 1 evidence records.

## 3. Authoritative store

The authoritative Phase 2 store is a private SQLite-compatible relational database.

The live database belongs outside the public repository, under the private A.L.I.C.E. vault.

Derived lexical or vector indexes are not authoritative. They must be completely rebuildable from
the authoritative memory store.

No production/private memory database is created by P2.0. P2.0 defines schema contracts only.

## 4. Package boundary

Phase 2 code lives in a new package:

```text
src/alice_memory/
```

Phase 1 remains:

```text
src/alice_vault/
```

Phase 2 tests live under:

```text
tests/phase2/
```

## 5. Memory record requirements

A durable memory record must preserve, where technically applicable:

- unique memory ID;
- schema version;
- content;
- content digest;
- optional normalized memory key;
- memory category;
- knowledge status;
- confidence;
- data classification;
- applicable time range;
- recording time;
- verification time;
- Rayan confirmation state;
- validity state;
- retention state;
- deletion state;
- creation and update timestamps.

The initial controlled vocabularies are defined in `alice_memory.schema`.

## 6. Memory categories

The Phase 0 Memory Policy categories are authoritative:

- working;
- profile;
- episodic;
- project;
- goal;
- procedural;
- relationship;
- reflective.

Durable working memory is permitted only after explicit promotion into the durable store.

## 7. Knowledge status

The Phase 0 Memory Policy vocabulary is authoritative:

- verified_fact;
- rayan_statement;
- external_claim;
- alice_inference;
- estimate;
- uncertain;
- disputed;
- historical;
- superseded.

Reflective memory must remain labeled as inference unless separately verified.

## 8. Data classification

Every memory and related object must carry one of:

- PUBLIC;
- INTERNAL;
- PRIVATE;
- HIGHLY_SENSITIVE;
- SECRETS.

`SECRETS` are prohibited from ordinary memory storage.

The schema retains the classification field, but runtime rejection of `SECRETS` belongs to the
memory service/access-control layer and must be implemented before live memory writes are enabled.

## 9. Provenance model

Memory content and source evidence remain separate.

A memory may have zero or more provenance links during draft/proposal stages, but a durable
production memory must satisfy the applicable provenance policy before activation.

Phase 1 provenance links may preserve:

- Phase 1 source reference;
- source content SHA-256;
- source text SHA-256;
- chunk ID;
- file ID;
- source date;
- support relationship.

The Phase 2 store must reference Phase 1 evidence; it must not copy private source material into the
public repository.

## 10. Temporal model

The architecture distinguishes:

- `recorded_at`: when A.L.I.C.E. learned or stored the memory;
- `valid_from`: when the fact/state began to apply;
- `valid_to`: when the fact/state stopped applying;
- `time_precision`: precision of the applicable time range.

A historical record is not automatically false.

Later information may supersede current-state interpretation without deleting history.

## 11. Conflict and supersession

Memories are not silently overwritten.

Memory-to-memory relations support:

- supersedes;
- conflicts_with;
- supports;
- duplicates;
- derived_from;
- corrects.

Corrections and supersessions create explicit relation chains.

Material unresolved conflicts must remain inspectable and must not be silently presented as
confirmed facts.

## 12. Deletion guarantee

Deletion is defined against active memory state, not merely an index entry.

A successful deletion must remove the targeted active memory from:

- the authoritative memory table;
- provenance joins whose lifetime is tied only to that memory;
- full-text indexes;
- vector indexes;
- caches;
- active derived summaries.

A sanitized tombstone may preserve:

- deleted memory ID;
- content digest;
- deletion time;
- deletion scope;
- associated audit event ID.

A tombstone must not preserve deleted sensitive plaintext unnecessarily.

Critical invariant:

```text
create
-> index
-> retrieve
-> delete
-> cannot retrieve
-> destroy indexes
-> rebuild from authoritative store
-> still cannot retrieve
```

Backups may retain encrypted copies until expiry, but deleted records must never be silently restored
to active memory.

## 13. Sensitive-memory access

Memory retrieval is default-deny with deterministic enforcement.

Runtime retrieval must consider:

- caller;
- purpose;
- requested operation;
- data classification;
- maximum allowed classification;
- sensitivity;
- authorization context.

`HIGHLY_SENSITIVE` memories require purpose-limited retrieval and stronger controls.

Semantic similarity alone must never be sufficient reason to surface intimate or painful memories.

`SECRETS` are never eligible for ordinary memory retrieval because they are never eligible for
ordinary memory storage.

## 14. Model and training boundary

Phase 2 does not require training a new model.

Personal facts that may change belong in memory/RAG rather than model weights.

Any future A.L.I.C.E. training or fine-tuning is a separately approved workflow. Heavy model training
will be performed using cloud GPU infrastructure rather than relying on the local laptop.

Personal memory must not be included in a training dataset by default.

## 15. Initial authoritative tables

P2.0 defines these schema-level tables:

- `schema_migrations`;
- `memories`;
- `memory_sources`;
- `memory_relations`;
- `memory_derivations`;
- `memory_entities`;
- `memory_events`;
- `memory_tombstones`.

Additional tables require a versioned schema migration.

## 16. P2.0 exit criteria

P2.0 is complete when:

1. the Phase 1/Phase 2 boundary is documented;
2. Phase 2 has an independent package and test namespace;
3. the initial memory schema is versioned;
4. SQLite foreign-key enforcement is tested;
5. required tables and controlled vocabularies are tested;
6. the schema can initialize transactionally in an isolated test database;
7. no live private memory data is written;
8. the existing Phase 1 regression suite remains unchanged.

## 17. Next milestone

After P2.0 passes targeted and full regression tests, P2.1 will implement the authoritative store
service and migration runner around this schema contract.
