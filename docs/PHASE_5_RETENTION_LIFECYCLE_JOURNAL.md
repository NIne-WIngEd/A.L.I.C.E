# Phase 5.1c — Retention Lifecycle Decision Journal

Status: implemented runtime foundation

## Purpose

P5.1c adds the deterministic, product-neutral journal that records why a retained artifact remains in place, changes logical tier, is quarantined, is blocked from a later action, or becomes eligible for governed deletion. It records decisions and lineage only. It does not move or delete payload bytes.

## Contracts

Lifecycle decisions bind an opaque subject reference and SHA-256 content digest to the current and proposed logical tier, retention class, reason codes, policy bindings, provenance reference, actor, authority level, authority-decision lineage, timestamp, outcome, and optional prior-decision relationship. Retention blockers are independent append-only records. Resolution creates a new record linked to the original open blocker. Authorized overrides preserve the prior decision rather than mutating it.

The transition matrix is explicit rather than all-to-all. The permanent compact ledger and deleted state have no outgoing transitions. Quarantine and deletion eligibility use dedicated decision types. Recorded deletion eligibility is not physical deletion. Owner-verified authority is required for overrides and owner holds.

## Journal guarantees

The SQLite journal is bound to one product, host instance, and encryption domain. It uses WAL, synchronous FULL, deterministic sequence numbers, a SHA-256 hash chain, atomic multi-record transactions, duplicate rejection, immutable rows, reopen durability, full integrity verification, path rejection inside the public repository, and sanitized inspection.

## Metadata-only boundary

The journal stores identifiers, digests, classifications, policy bindings, and lineage. It stores no payload bytes, source-person content, credentials, encryption keys, private-memory records, or Friday product state.

## Deferred behavior

Automatic retention scoring, the Learning Curator, automatic expiry, physical tier movement, payload deletion, deletion propagation, storage-pressure eviction, backup and restore, cloud storage, Phase 2 migration, and production activation remain deferred.
