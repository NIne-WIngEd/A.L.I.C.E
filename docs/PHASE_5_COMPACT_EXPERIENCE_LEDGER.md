# Phase 5 Compact Experience Ledger Foundation

## Milestone

P5.1a establishes persistent storage for the existing metadata-only
`ExperienceEvent` contract. The implementation is host-neutral and remains
separate from the Phase 2 A.L.I.C.E. Memory Core database.

## Implemented

- SQLite WAL persistence with `synchronous=FULL`;
- explicit atomic write transactions;
- deterministic, contiguous sequence numbers;
- immutable event rows protected by update and delete triggers;
- SHA-256 entry chaining from a deterministic ledger genesis digest;
- duplicate event identity and digest rejection;
- product, host-instance, and encryption-domain binding;
- refusal to place a live ledger inside the public repository;
- complete event-envelope and hash-chain integrity verification;
- deterministic transaction receipts;
- sanitized inspection records without raw payloads or provenance details;
- reconstruction of validated `ExperienceEvent` metadata envelopes.

## Boundaries

P5.1a stores canonical metadata envelopes only. An opaque payload reference may
remain in the event envelope, but payload bytes are never accepted or written by
the ledger API.

This milestone does not implement:

- raw-buffer capture or expiration;
- content-addressed blob storage;
- payload deduplication;
- hot, warm, cold, quarantine, or deleted-tier movement;
- automated retention or storage-pressure decisions;
- backup manifests or restore execution;
- Learning Curator behavior;
- migration of the Phase 2 Memory Core database;
- Friday product source or Friday capability activation.

## Isolation

One database is bound to exactly one product, host instance, and encryption
domain. A record with a different binding is rejected, and an existing database
cannot be reopened under a different binding. Record schema versions may evolve
without collapsing the stable storage boundary.

## Append-only meaning

The ledger is logically append-only. Public APIs provide no mutation or deletion
operation, and SQLite triggers reject ordinary row updates and deletes. Integrity
verification still reconstructs every event and recomputes the full chain so
out-of-band file tampering remains detectable.
