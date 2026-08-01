# Phase 5 P5.0b — Cognitive Kernel Scope and Provenance Foundation

## Status

This milestone creates the first independently versioned, host-neutral Personal Cognitive Kernel package boundary under `src/cognitive_kernel`.

It implements contracts only. It does not implement persistent ledger storage, the raw experience buffer, the complete Cognitive Workspace UI, autonomous learning, Friday product source, or production promotion.

## Delivered contracts

- deterministic canonical JSON and SHA-256 helpers;
- explicit product, host-instance, schema, and encryption scope;
- exact provenance vocabulary shared with private-companion custody policy;
- exact clone-aware identity-layer vocabulary;
- A.L.I.C.E.-only opaque private-companion references without payload or ciphertext;
- tamper-evident metadata-only Experience Event envelopes;
- storage-tier, retention-class, and deletion-lineage fields;
- policy loading that cross-validates product, custody, identity, and storage authorities;
- synthetic A.L.I.C.E./Friday multi-host isolation tests.

## Security and privacy boundary

The package contains neutral contracts and synthetic fixtures only.

It does not contain:

- private source-person material;
- plaintext directive meanings or codebooks;
- private model state;
- encrypted persona payloads;
- Friday product implementation;
- cross-host caches or deduplication;
- raw event payload content.

`OpaquePrivateCompanionReference` is restricted to an explicit A.L.I.C.E. host scope and records only opaque identifiers, approved directive codes, identity layers, metadata-only provenance, and a tamper-evident digest.

## Experience Event boundary

`ExperienceEvent` is an append-only envelope contract. It binds:

- product and host scope;
- encryption domain;
- event type and time;
- content digest rather than content;
- provenance;
- retention class;
- storage tier;
- deletion lineage;
- parent and outcome references;
- policy bindings;
- an optional opaque payload reference.

Persistence, transactionality, blob storage, backup, restore, replay, and deletion execution are later Phase 5 milestones.

## Successor milestones

- **P5.0c:** Mission Graph, node/edge, semantic-routing, Result Capsule, and traceback contracts.
- **P5.0d:** attention decision, workspace projection, speaker context, guest session, guest grant, and authority-request contracts.
- **P5.0e:** candidate-learning, evaluation, release evidence, migration, package inspection, and final foundation release audit.

This ordering is an implementation sequence, not a permanent capability ceiling.
