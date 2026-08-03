# Phase 5.1d — Governed Non-Destructive Tier Transition Executor

Status: implemented runtime foundation

## Purpose

P5.1d executes an already-approved P5.1c lifecycle decision by copying host-sealed opaque payload bytes from `raw_buffer`, `hot`, `warm`, `cold`, or `quarantine` into an approved durable target tier. It does not create retention decisions. It does not delete or overwrite the source payload.

## Authorization and blocker gates

Every execution binds the exact lifecycle decision identifier and SHA-256 digest. The executor requires an approved `transition`, `quarantine`, or owner-authorized `override` decision. It rejects stale decisions that have a superseding child decision. Before any physical publication, it verifies the lifecycle journal and rejects execution while a matching retention blocker remains open.

## Publication protocol

The executor appends an immutable prepared intent before copying bytes. It writes to a temporary object in the target tier, fsyncs the file, verifies length and SHA-256, atomically publishes the final object, fsyncs the directory, and appends an immutable publication receipt. A published source object remains available. Re-execution is idempotent, and a prepared intent can recover after interruption without deleting the source.

## Store guarantees

The tier-transition store is bound to one product, host instance, and encryption domain. It uses SQLite WAL and synchronous FULL, repository path rejection, deterministic identities, append-only intents and receipts, reopen durability, crash recovery, host-scoped physical deduplication within a target tier, full metadata and object integrity verification, and sanitized inspection.

## Security boundary

The kernel handles only host-sealed opaque bytes and metadata. It does not hold encryption keys, require plaintext, access a network or cloud service, import the private Phase 2 memory store, or add Friday product source.

## Deferred behavior

Automatic retention, the Learning Curator, automatic expiry, payload deletion, deletion propagation, storage-pressure eviction, backup and restore, cross-host deduplication, and production activation remain deferred.
