# Memory Shadow Migration Stage F+G Persistent Integration Foundation

**Version:** 1.0.0
**Status:** persistent reference integration operational; external backend candidates registered for research
**Baseline:** PR #86 merge `1a3c668121d69f467740f518bb8d549f1ea26791`
**Scope:** persistent Stage F controlled-mirroring evidence plus Stage G projection-generation durability without production authority transfer

## 1. Why this tranche exists

PR #86 made Stage F controlled mirroring and Stage G graph/vector/workflow generation operational as nonproduction synthetic prototypes. Those contracts proved deterministic outbox, idempotency, reconciliation, generation digests, deletion watermarks, and repair accounting, but the public evaluation used an in-memory mirror sink.

This tranche adds persistence evidence before any Stage H authority review.

Phase 2 remains the released canonical writer and current authority. Real owner-private Stage D execution, live external backend integrations, bounded Stage H canary authority, canonical transfer, production serving, cutover, Phase 2 retirement, and P5.1e remain independently evidence-gated.

## 2. Persistent compatibility/reference adapter

The repository now includes `SQLitePersistentStageFGReferenceAdapter`.

SQLite is a compatibility/reference durability oracle in this tranche. It is **not** selected as A.L.I.C.E.'s permanent migration destination.

The adapter demonstrates:

- WAL-backed durable writes;
- ordered outbox sequence collision detection;
- idempotency across process close/reopen;
- persistent evidence and deletion-lineage metadata;
- persistent graph/vector/workflow generation receipts;
- integrity checking;
- deterministic persistent-state digests;
- explicit WAL checkpoint evidence.

Synthetic database artifacts are written outside public Git.

## 3. Backend candidate registry

The current research registry includes:

- **KurrentDB** for event streams, outbox subscriptions, replay, and server-side checkpoints;
- **Neo4j** for graph projection and graph-native reasoning;
- **Qdrant** for vector and multimodal retrieval generations;
- **Temporal** for durable workflow and recovery coordination;
- **SQLite** only as the current compatibility/reference durability oracle.

This registry is not exhaustive and is not a destination selection. Additional or replacement engines may be admitted whenever evidence supports them.

## 4. Research basis

The persistent-integration design reflects current vendor capabilities:

- KurrentDB persistent subscriptions preserve checkpoints server-side and support reconnecting consumers and at-least-once delivery;
- Neo4j managed transactions provide ACID transaction boundaries, retries, and require idempotent transaction functions under retry;
- Qdrant persists mutations through a write-ahead log and supports persistent collection storage and snapshots;
- Temporal remains a first-class durable-workflow candidate under the ratified polyglot memory program.

Candidate-specific live integration still requires its own dependency, configuration, custody, recovery, deletion, benchmark, and rollback evidence.

## 5. What the synthetic evaluation proves

The public evaluator:

1. creates a synthetic Stage F outbox workload;
2. writes it to the persistent reference adapter;
3. closes and reopens the database;
4. replays the same changes and verifies deterministic duplicates;
5. persists a Stage G graph/vector/workflow generation receipt;
6. reopens and replays the generation receipt;
7. verifies SQLite integrity, WAL mode, record counts, and persistent-state digests;
8. emits the report and reference database outside public Git.

It does not connect to external candidate services, execute real private historical backfill, or activate production serving.

## 6. Authority and privacy

This tranche preserves:

- owner sovereignty;
- truthful material state;
- private-state custody;
- product/host isolation;
- evidence and deletion lineage;
- correction and rollback;
- Phase 2 canonical writer status;
- separate Friday private state;
- explicit production activation gates.

The reference database contains synthetic evaluation data only during the public repository validation flow.

## 7. Successor evidence

Independent next gates may include:

- candidate-specific live KurrentDB event/outbox integration;
- candidate-specific live Neo4j graph generation;
- candidate-specific live Qdrant vector generation;
- candidate-specific live Temporal workflow generation;
- owner-authorized real Stage D private backfill;
- bounded Stage H canary review after persistent candidate evidence and rollback evidence;
- later canonical transfer, production serving, cutover, compatibility operation, retirement, and P5.1e storage admission.

Research may proceed in parallel. Production activation remains evidence- and authority-governed.
