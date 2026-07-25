
# Phase 2 — Memory Core Architecture

**Status:** P2.0–P2.9 implemented; Phase 2 complete
**Phase 1 dependency:** Frozen, read-only evidence layer
**Owner:** MK Rayan

## 1. Purpose

Phase 2 turns trusted Phase 1 evidence and explicitly approved personal knowledge into a durable, structured, inspectable, correctable, and deletable memory system.

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
- memory access control;
- non-authoritative memory candidates;
- deterministic candidate assessment;
- explicitly authorized candidate promotion.

Phase 2 MUST NOT rewrite Phase 1 evidence records.

### 2.1 Implemented milestone status

The current implementation includes:

- P2.0 — memory architecture and schema foundation;
- P2.1 — authoritative private SQLite store and migration runner;
- P2.2 — read-only Phase 1 provenance bridge;
- P2.3 — permission-gated lifecycle operations and metadata-safe inspection;
- P2.4 — correction, supersession, conflict, and valid-time resolution;
- P2.5 — authorization-aware lexical, semantic, and hybrid memory retrieval;
- P2.6 — encrypted `HIGHLY_SENSITIVE` storage and purpose-bound local access;
- P2.7 — candidate formation, deterministic assessment, ordinary promotion, transition-aware promotion, and adversarial promotion gates;
- P2.8 — ordinary and protected deletion lifecycles, candidate-lineage cleanup, derived-index purge and rebuild guarantees, and adversarial deletion gates;
- P2.9 — versioned evaluation policy and benchmark, deterministic source-cited answers, adversarial end-to-end Memory Core gates, and private final release audit.

P2.9 completes the Phase 2 roadmap scope. Conversational orchestration remains Phase 3 work.

## 3. Authoritative store

The authoritative Phase 2 store is a private SQLite-compatible relational database.

The live database belongs outside the public repository, under the private A.L.I.C.E. vault.

Derived lexical or vector indexes are not authoritative. They must be completely rebuildable from the authoritative memory store.

The public repository contains schema, migration, service, retrieval, and test code only. It must never contain a live private memory database.

## 4. Package boundary

Phase 2 code lives in:

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

The controlled vocabularies are defined in `alice_memory.schema`.

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

Durable working memory is permitted only after explicit promotion into the authoritative store.

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

Ordinary candidate staging accepts only `PUBLIC`, `INTERNAL`, and `PRIVATE`. `HIGHLY_SENSITIVE` content requires the dedicated encrypted sensitive-memory path. `SECRETS` remain prohibited.

`HIGHLY_SENSITIVE` authoritative memory uses encrypted payload storage and purpose-bound local access. It must not enter ordinary lexical, semantic, or hybrid retrieval.

## 9. Provenance model

Memory content and source evidence remain separate.

Every authoritative memory created through the ordinary service or candidate-promotion path must preserve validated provenance.

Phase 1 provenance links may preserve:

- Phase 1 source reference;
- source content SHA-256;
- source text SHA-256;
- chunk ID;
- file ID;
- source date;
- support relationship.

The Phase 2 store references Phase 1 evidence. It must not copy private source material into the public repository.

Candidate provenance is stored separately from authoritative provenance. Promotion copies validated candidate provenance into the new authoritative memory within the same transaction.

## 10. Candidate formation and promotion

A proposed memory is not an authoritative memory.

The candidate pipeline is:

```text
Phase 1 evidence or explicit user input
        |
        v
non-authoritative candidate
        |
        v
deterministic assessment
        |
        v
risk, provenance, duplicate, and conflict checks
        |
        v
explicit candidate-bound authorization
        |
        v
authoritative memory or deterministic no-op
```

Candidate records live in separate candidate tables and are excluded from authoritative lexical, semantic, and hybrid retrieval.

Candidate origins are:

- `explicit_user`;
- `deterministic_import`;
- `model_proposed`.

A model-proposed candidate:

- cannot claim user confirmation at formation time;
- must preserve policy, model, prompt, and run metadata;
- always requires explicit user review before promotion;
- cannot authorize its own promotion;
- cannot choose or authorize its own correction, supersession, or conflict transition.

Deterministic assessment produces one of:

- `rejected`;
- `review_required`;
- `promotion_eligible`.

Ordinary promotion is candidate-bound, atomic, and re-runs deterministic assessment while holding the write transaction.

When a current authoritative memory already occupies the same logical key, transition-aware promotion requires authorization bound to:

- the candidate;
- the target authoritative memory;
- the transition type;
- an audit-safe authorization ID.

Supported transition resolutions are:

- duplicate no-op;
- correction;
- supersession;
- conflict.

Promotion preserves provenance and derivation metadata. Candidate state, authoritative memory creation, relations, target changes, and audit events commit or roll back together.

## 11. Temporal model

The architecture distinguishes:

- `recorded_at`: when A.L.I.C.E. learned or stored the memory;
- `valid_from`: when the fact or state began to apply;
- `valid_to`: when the fact or state stopped applying;
- `time_precision`: precision of the applicable time range.

A historical record is not automatically false.

Later information may supersede current-state interpretation without deleting history.

Valid-time resolution uses half-open intervals:

```text
valid_from <= at < valid_to
```

## 12. Conflict and supersession

Memories are not silently overwritten.

Memory-to-memory relations support:

- supersedes;
- conflicts_with;
- supports;
- duplicates;
- derived_from;
- corrects.

Corrections and supersessions create explicit relation chains.

Material unresolved conflicts remain inspectable and must not be silently presented as confirmed facts.

Transition-aware candidate promotion reuses these same authoritative relation semantics.

## 13. Deletion guarantee

Deletion is defined against active memory state, not merely an index entry.

A successful deletion must remove the targeted active memory from:

- the authoritative memory table;
- provenance joins whose lifetime is tied only to that memory;
- memory relations and derivations tied to that memory;
- memory entities tied to that memory;
- promoted candidate plaintext and candidate lineage tied to that memory;
- full-text indexes;
- vector indexes;
- caches;
- active derived summaries.

### 13.1 Authorized deletion lifecycle

Ordinary memory deletion uses an explicit, exact-resource lifecycle:

```text
active
  |
  | authorized deletion request
  v
pending_deletion
  |              |
  | cancel       | strong confirmation
  v              v
active        deleted
```

Deletion authorization is bound to the exact memory and deletion scope. Strong confirmation expires after at most two minutes.

A pending-deletion memory cannot expose plaintext through the ordinary content-access path.

Completed deletion is atomic with authoritative dependent cleanup and sanitized audit creation. A failed tombstone write rolls back the authoritative deletion.

Completed retries are idempotent. Reusing a tombstoned memory identifier is prohibited.

### 13.2 Protected sensitive deletion

`HIGHLY_SENSITIVE` memory uses a separate protected deletion path.

The protected path requires:

- direct-user authorization;
- authorization bound to the exact memory and deletion scope;
- strong confirmation within the allowed validity window;
- encrypted-payload and sentinel integrity checks;
- sanitized deletion audit evidence.

Successful protected deletion removes the encrypted payload, ciphertext, nonce, and key reference with the authoritative memory.

Deletion evidence must not contain deleted sensitive plaintext, ciphertext, nonce, encryption key identifier, source reference, source digest, or free-form reason text.

### 13.3 Candidate lineage and derived state

When an authoritative memory was promoted from a candidate, deletion also removes the linked candidate plaintext, candidate provenance, and candidate audit rows.

Only SHA-256 digests of deleted candidate identifiers may remain in sanitized deletion evidence.

Mismatched content digests, non-promoted links, duplicated lineage proofs, or tampered linkage fail closed.

Unlinked candidates remain unchanged.

The deletion service also fails closed when unmanaged active cache or derived-summary tables are present.

### 13.4 Tombstones and audit integrity

A sanitized tombstone may preserve only:

- deleted memory ID;
- content digest;
- deletion time;
- deletion scope;
- associated deletion-event ID.

The tombstone identifier and content digest are verified deterministically.

Deletion-request, cancellation, and completion events use exact sanitized structures. Unexpected fields, invalid transitions, false confirmation evidence, malformed dependent counts, duplicate sensitive decisions, or mismatched tombstone links fail closed.

A tombstone must not preserve deleted sensitive plaintext or encrypted payload material.

### 13.5 Derived-index purge and rebuild

Derived lexical and semantic indexes are non-authoritative.

Pending deletion changes authoritative state and invalidates stale indexes. Index verification checks authoritative digests and exact memory-ID sets before retrieval.

After completed deletion, the system can:

1. remove the full lexical-index root;
2. remove every immutable semantic-index generation;
3. rebuild both indexes from current authoritative memory;
4. verify that the deleted memory is absent;
5. verify that retained memories remain present.

Index purge is idempotent. If filesystem cleanup fails after authoritative deletion commits, stale indexes continue to fail closed and purge may be retried.

Critical invariant:

```text
create
-> index
-> retrieve
-> request deletion
-> confirm deletion
-> cannot retrieve
-> destroy indexes
-> rebuild from authoritative store
-> deleted memory still cannot retrieve
-> retained memory still retrieves
```

Backups may retain encrypted copies until expiry, but deleted records must never be silently restored to active memory.

## 14. Sensitive-memory access

Memory retrieval is default-deny with deterministic enforcement.

Runtime retrieval considers:

- caller;
- purpose;
- requested operation;
- exact resource;
- data classification;
- maximum allowed classification;
- authorization expiry;
- authorization context.

`HIGHLY_SENSITIVE` memories require encrypted storage, purpose-limited local access, exact operation and resource scope, expiry enforcement, and sanitized audit records.

Semantic similarity alone is never sufficient reason to surface intimate or painful memories.

`SECRETS` are never eligible for ordinary memory retrieval because they are never eligible for ordinary memory storage.

## 15. Model and training boundary

Phase 2 does not require training a new model.

Personal facts that may change belong in memory and retrieval rather than model weights.

Any future A.L.I.C.E. training or fine-tuning is a separately approved workflow. Heavy model training will be performed using cloud GPU infrastructure rather than relying on the local laptop.

Personal memory must not be included in a training dataset by default.

## 16. Current authoritative tables

Schema version 3 defines:

- `schema_migrations`;
- `memories`;
- `memory_sources`;
- `memory_relations`;
- `memory_derivations`;
- `memory_entities`;
- `memory_events`;
- `memory_tombstones`;
- `memory_sensitive_payloads`;
- `sensitive_memory_access_events`;
- `memory_candidates`;
- `memory_candidate_sources`;
- `memory_candidate_events`.

Additional tables require a versioned schema migration.

## 17. P2.7 completion criteria

P2.7 is complete when:

1. candidate records remain separate from authoritative memory;
2. candidates cannot enter authoritative lexical, semantic, or hybrid indexes;
3. candidate formation preserves origin and provenance;
4. deterministic assessment handles weak provenance, duplicates, conflicts, confidence, knowledge status, and confirmation state;
5. rejected candidates cannot use ordinary promotion;
6. model-origin candidates require explicit human confirmation;
7. ordinary promotion is explicitly authorized, candidate-bound, atomic, and idempotent;
8. correction, supersession, conflict, and duplicate resolution are target-bound and transition-bound;
9. stale assessments are rechecked under the write transaction;
10. promotion preserves authoritative provenance and derivation metadata;
11. tampered promotion links, derivations, audit events, and relations fail closed;
12. ordinary candidate storage continues to reject `HIGHLY_SENSITIVE` and `SECRETS`;
13. existing Phase 1 and Phase 2 regression tests remain passing.

Final P2.7 verification on the feature branch:

- 15 adversarial P2.7e tests passed;
- 94 combined Phase 2.7 and temporal-transition tests passed;
- 232 Phase 2 tests passed;
- 356 full-suite tests passed;
- 14 subtests passed.

## 18. P2.8 completion criteria

P2.8 is complete when:

1. ordinary deletion requires exact-resource authorization and strong confirmation;
2. a pending deletion can be cancelled before irreversible deletion;
3. pending-deletion plaintext access fails closed;
4. `HIGHLY_SENSITIVE` deletion uses a separate protected path;
5. authoritative deletion and dependent cleanup are atomic;
6. promoted candidate plaintext and tied lineage are removed;
7. unrelated candidates remain unchanged;
8. sanitized tombstones preserve no deleted plaintext or encrypted payload material;
9. malformed, replayed, duplicated, or tampered deletion audit state fails closed;
10. stale lexical, semantic, and hybrid indexes fail closed after deletion-state changes;
11. every derived retrieval index can be destroyed and rebuilt from authoritative memory;
12. deleted memories remain absent after rebuild;
13. retained memories remain available after rebuild;
14. unmanaged active cache or derived-summary state blocks deletion;
15. completed deletion retries are idempotent;
16. existing Phase 1 and Phase 2 regression tests remain passing.

Final P2.8 verification on the feature branch:

- 20 P2.8a ordinary-deletion tests passed;
- 10 P2.8b index-purge and rebuild tests passed;
- 21 P2.8c protected sensitive-deletion tests passed;
- 14 P2.8d lineage-integrity tests passed;
- 15 P2.8e adversarial and benchmark tests passed;
- 80 combined P2.8 deletion tests passed;
- 312 Phase 2 tests passed;
- 436 full-suite tests passed;
- 14 subtests passed.

## 19. P2.9 final evaluation and release audit

P2.9 closes the remaining Memory Core exit criteria without introducing a conversational model.

The final evaluation is:

- versioned and synthetic-only;
- deterministic and reproducible;
- read-only after fixture construction;
- offline;
- unable to call tools or perform external actions;
- restricted to private output;
- bound to the approved policy, benchmark, fixture snapshot, and repository commit.

Deterministic answer packets cite exact authoritative memory and `memory_sources` rows. The evaluator verifies claim text, content digests, source records, temporal eligibility, correction state, conflicts, uncertainty labels, deletion absence, candidate boundaries, permission denial, sensitive-data denial, and prompt-injection resistance.

The final release audit produces a tamper-evident JSON record under the private vault. The public repository contains only the evaluation implementation, synthetic fixtures, policy, benchmark, tests, and this sanitized release summary.

## 20. P2.9 completion criteria

P2.9 is complete when:

1. the evaluation policy and benchmark are versioned and validated;
2. all required Memory Core suites have deterministic synthetic cases;
3. the fixture snapshot matches the approved benchmark digest;
4. personal answer claims match authoritative memory plaintext and content digests;
5. each supported claim cites exact authoritative source rows;
6. unsupported questions produce no personal claims;
7. temporal current and historical states are distinguished correctly;
8. material conflicts remain surfaced;
9. corrected records cannot support current answers;
10. uncertainty remains explicitly labeled;
11. permission-denied cases expose no protected memory;
12. `HIGHLY_SENSITIVE` memory cannot enter ordinary answer generation;
13. deleted memory remains absent;
14. unpromoted candidates remain non-authoritative;
15. prompt-injection text is treated as untrusted data;
16. every governing metric passes its threshold;
17. every critical zero-tolerance gate passes;
18. the final report is digest-verified;
19. the release decision records the repository commit, UTC time, policies, limitations, and rollback target;
20. the private release record cannot be written into the repository;
21. existing Phase 1 and Phase 2 regression tests remain passing.

Final P2.9 verification on the feature branch:

- 32 P2.9a evaluation-contract and fixture tests passed;
- 34 P2.9b authoritative citation tests passed;
- 30 P2.9c/P2.9d final-gate and release-audit tests passed;
- 96 combined P2.9 tests passed;
- 408 Phase 2 tests passed;
- 532 full-suite tests passed;
- 14 subtests passed.

## 21. Phase 2 release decision

Phase 2 is complete when the private release-audit runner returns `approved=true` for the final feature-branch commit and all required repository checks pass.

The private release record is not committed. It preserves the exact evaluated commit, policy and benchmark digests, approved fixture snapshot, metric results, known limitations, rollback commit, and a tamper-evident record digest.

Phase 2 completion establishes the Memory Core only. A working conversational assistant, orchestration loop, constitutional dialogue behavior, and conversation state remain Phase 3 responsibilities.
