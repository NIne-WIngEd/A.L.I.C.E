# Memory Shadow Migration Stage A+B Implementation

**Version:** 1.1.0
**Status:** prototype operational
**Baseline:** PR #82 merge `8d27de4f7050ea8fbc420806bb837f9e5362e89b`; merged by PR #83 at `2ad8ffcafa25ffc8bc5129ae32dbb201236950ca`
**Scope:** Phase 2 inventory/registration and deterministic read-only adapters

## 1. Material state

M2.0 through M2.6 are closed at the implemented-contract and reversible-prototype level.

PR #83 completed the first admitted Phase 2 shadow-migration prototype tranche:

- Stage A — inventory and registration;
- Stage B — deterministic read-only contract adapters.

The released Phase 2 Memory Core remains the compatibility baseline, test oracle, fallback, and current authority for its released profile.

## 2. Implemented artifacts

### Stage A

`src/cognitive_kernel/phase2_shadow_adapter.py`:

- opens a Phase 2 SQLite source through SQLite URI `mode=ro`;
- enables `PRAGMA query_only`;
- rejects a live source path inside the public repository;
- inventories expected tables, schema columns, schema generation, record counts, integrity state, data classifications, and deletion capabilities;
- emits a digest-bound `Phase2SourceInventory`;
- emits a compatibility-scoped `StoreRegistration`;
- does not enumerate memory content, ciphertext, source text, event details, or private payload columns;
- does not make a discovered source authoritative.

### Stage B

The same module provides deterministic adapters for caller-supplied synthetic records:

- memory to evidence/claim/current-projection candidate receipts;
- relation to evidence/conflict/correction candidate receipts;
- tombstone to deletion-propagation candidate receipts;
- batch reconciliation with exact accounting.

Each receipt records:

- source record identity and digest;
- mapping version;
- destination record kinds and candidate IDs;
- authority class;
- mapping outcome and adjudication hint;
- ambiguity and information-loss codes;
- provenance, correction, and deletion lineage;
- explicit read-only/no-production-write state;
- canonical SHA-256 binding.

## 3. Activation boundary

This implementation is reversible and preparatory.

It creates no destination authority. It performs no historical private backfill. It performs no write mirroring. It changes no canonical writer. It provides no production-serving influence. It does not activate canary authority, cutover, Phase 2 retirement, Friday state transfer, or P5.1e.

Those are successor activation states. They remain available when their exact evidence and authority gates are satisfied. This boundary is not a technology, topology, scale, context, graph, workflow, model, training, deployment, or research ceiling.

## 4. Privacy and custody

The inventory path is metadata-only by design for this tranche. That artifact boundary does not limit A.L.I.C.E.'s full-memory destination.

Real private-store inspection requires an owner-authorized path outside Git. The implementation never stores a private path in a committed artifact. Tests use synthetic fixtures only.

## 5. Exit and rollback

Rollback is file-level:

1. remove the Stage A+B source, policy, tests, and documentation;
2. restore the README, catalog, migration-plan, and package-export edits;
3. retain the released Phase 2 source and tests unchanged;
4. retain M2 closeout evidence unchanged.

No private data or destination store is created by this tranche.

## 6. Validation

Required before PR:

- Stage A+B targeted tests;
- M2 foundation and closeout tests;
- relevant Phase 2 store tests;
- Python compile;
- repository phase-boundary audit;
- changed-line capability-barrier audit;
- full capability-barrier audit;
- `git diff --check`;
- clean secret/private-artifact scan;
- exact candidate commit and evidence report.

## 8. Successor state

Stage A+B remains the source inventory, registration, adapter, and reconciliation oracle for Stage C+E. The Stage C+E profile may compare destination candidates and shadow observations, but it may not reinterpret Stage A+B discovery as authority or bypass its ambiguity, correction, deletion-lineage, custody, and reconciliation receipts.
