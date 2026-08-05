# Memory Claim Identity and Version — Ratified M1-D1 Standard

**Version:** 1.0.0
**Status:** Owner-ratified under Memory M1 on 2026-08-05
**Architecture:** backend-neutral Claim Authority

## 1. Goals

Claim identity must remain stable across backend replacement, device migration, replication, sharding, export, graph/vector projection, and local or distributed deployment.

It must support append-only correction, bitemporal history, idempotency, conflict, federation, deletion, and rollback without embedding physical topology into semantic identity.

## 2. Claim Identity

```text
claim_id
product_id
authority_namespace_id
canonical_subject
canonical_predicate
canonical_value
qualifiers
semantic_scope
canonicalization_version
semantic_digest
created_at
retired_at
retirement_reason
```

`claim_id` is an opaque, collision-resistant identifier. UUIDv7 is the initial candidate because it supports global uniqueness and useful locality without making wall-clock order authoritative.

`authority_namespace_id` separates owner, product, import, test, and federation authority.

## 3. Semantic digest

The digest covers canonical type-tagged semantic fields:

```text
product_id
authority_namespace_id
canonical_subject
canonical_predicate
canonical_value
qualifiers
semantic_scope
canonicalization_version
```

It excludes:

```text
physical backend locator
host or cluster locator
shard and replica locator
encryption-routing metadata
storage tier
index generation
graph node ID
vector point ID
```

A digest match is an optimization. Full canonical equality is required before identity reuse.

## 4. Claim Version

```text
claim_version_id
claim_id
version_sequence
store_sequence
valid_from
valid_to
transaction_time
event_stream_position
logical_clock
causal_parents[]
value
qualifiers
authority_class
confidence
adjudication_state
evidence_relations[]
conflict_set_id
supersedes[]
superseded_by[]
correction_of[]
deletion_state
writer
workflow_id
idempotency_namespace
idempotency_key
request_digest
schema_version
```

Claim versions are immutable after commit except for authorized cryptographic erasure or privacy deletion procedures that preserve the minimum deletion proof.

## 5. Ordering

The canonical order inside a Claim Authority is `store_sequence` or an equivalent authoritative position assigned at commit.

`transaction_time` records when the authority accepted the version. `valid_from` and `valid_to` describe the represented world interval. `event_stream_position`, `logical_clock`, and `causal_parents` support distributed and replay semantics.

Wall-clock timestamps do not independently decide commit order.

## 6. Idempotency

The tuple:

```text
idempotency_namespace
idempotency_key
request_digest
```

identifies one semantic write attempt.

Repeating the tuple returns the prior result. Reusing namespace and key with a different request digest is an explicit conflict.

## 7. Current state

`CurrentClaimProjection` records:

```text
claim_id
current_claim_version_id
authority_generation
projection_generation
adjudication_state
validity_state
conflict_state
deletion_state
updated_at
source_position
```

It is derived and rebuildable. Ordinary reads do not replay complete history.

## 8. Backend and replication metadata

A separate registration or receipt records:

```text
component_id
backend_type
backend_locator
partition_key
shard_id
replica_id
region_or_device_scope
consistency_model
replication_generation
source_position
target_position
deletion_watermark
health
```

Moving a claim between backends does not change semantic identity.

## 9. Federation and export

An exported claim keeps its origin identity and receives an import/federation record naming source authority namespace, target namespace, export authority, allowed fields, purpose, digest, deletion contract, and conflict policy.

An import does not silently merge owner namespaces.

## 10. Graph and vector receipts

Graph and vector projections record claim/version IDs, projection generation, mapping, algorithm/model, deletion watermark, and rebuild recipe.

A graph node or vector point cannot become Claim Authority through identifier reuse.

## 11. Conflict and partition semantics

Concurrent incompatible versions join a conflict set. Resolution appends an adjudication record and successor version. Losing alternatives remain inspectable unless privacy deletion applies.

Network partitions and late replicas do not use last-writer-wins by default. The registered authority policy selects expected-version rejection, causal merge, domain adjudication, or quarantine.

## 12. Deletion and retirement

Identity retirement marks an assertion family inactive while preserving nonprivate historical authority.

Authorized privacy deletion may remove payloads and sensitive semantic content, propagate tombstones and deletion watermarks, rebuild derivatives, retire models, or destroy keys. The system records completed and technically limited surfaces.

## 13. Examples

### Same claim on a new backend

A Claim Authority moves from an embedded backend to distributed SQL. `claim_id` and semantic digest remain stable. Component, shard, replica, and source-position metadata change through registration and migration receipts.

### Concurrent correction

Two devices append incompatible corrections while disconnected. Both versions retain their causal metadata. Reconciliation creates a conflict set and later adjudication rather than silently choosing the later wall-clock timestamp.

### Graph projection

A relationship edge is generated from a claim version. The edge records the source version and graph generation. A graph algorithm proposes a new relation. That proposal becomes a candidate and requires adjudication before entering Claim Authority.

## 14. Nonclaims

This ratified decision does not select a permanent database, graph engine, event store, vector system, or deployment topology. It does not assert that a Claim Authority runtime is implemented.
