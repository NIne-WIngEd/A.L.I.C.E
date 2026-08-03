# Phase 5.1b — Raw Buffer and Content-Addressed Payload Foundation

Status: implemented runtime foundation

## Purpose

P5.1b adds a product-neutral raw-buffer store for full recent payloads while preserving the permanent compact event ledger introduced in P5.1a. The kernel stores only opaque host-sealed bytes and metadata contracts. It does not hold encryption keys, require plaintext, or place private payloads in source code, fixtures, policy files, or distributable kernel artifacts.

## Scope

The store is bound to one product, host instance, and encryption domain. SHA-256 identifies payload bytes. Identical bytes are physically stored once only inside that scope. Each capture still creates a distinct logical reference with independent sensitivity, retention, provenance linkage, and lifecycle meaning.

The runtime provides atomic temporary-file publication, transactional reference metadata, path safety outside the public repository, reopen durability, digest verification, orphan-object detection, sanitized metadata inspection, and logical-versus-physical storage accounting.

## Deferred behavior

P5.1b does not implement automatic expiry, Learning Curator decisions, hot/warm/cold migration, deletion propagation, storage-pressure eviction, backup or restoration, encryption-key custody, Phase 2 memory migration, production activation, or Friday product source.

The original P5.0b foundation policy remains a historical contract snapshot. Current runtime implementation state is recorded by the P5.1a ledger policy, this P5.1b policy, and the active capability profile rather than rewriting the historical foundation milestone.
