# Memory Record and Provenance Standard — Polyglot Cognitive Fabric

**Version:** 1.0.0
**Status:** Owner-ratified under Memory M1 on 2026-08-05
**Applies to:** Memory Architecture v4.1 and successor profiles

## 1. Purpose

This standard prevents evidence, claims, graph relations, vector representations, workflows, datasets, models, and operational state from collapsing into one ambiguous notion of memory.

Every material record declares its truth role, authority role, provenance, scope, generation, deletion behavior, and successor path.

## 2. Common envelope

Every authoritative or registered derivative record carries, directly or through a bound manifest:

```text
record_id
record_type
schema_version
product_id
authority_namespace_id
host_or_cluster_id
authority_role
deployment_profile
created_at
valid_time
transaction_time
logical_clock
causal_parents[]
source_records[]
generation
state
data_classification
encryption_domain
retention_class
deletion_state
provenance_digest
content_digest
writer
workflow_or_request_id
idempotency_namespace
idempotency_key
supersedes[]
superseded_by[]
rollback_reference
```

Physical backend locators, shard keys, replica locations, and encryption routing do not redefine semantic identity.

## 3. Required record families

### 3.1 Store and Capability Fabric Registration

Registers databases, streams, graph engines, vector systems, object stores, workflow runtimes, model servers, training clusters, datasets, replicas, and archives.

Required fields include capability descriptor, authority role, backend type/version, consistency, availability, encryption, region/device scope, health, performance, cost, deletion endpoint, rollback endpoint, backup profile, derives-from, replicates, synchronizes-with, and successor.

### 3.2 Evidence Event

Records an observation, supplied artifact, action, invocation, outcome, correction, measurement, or system event.

An Evidence Event does not become an adjudicated claim merely because an extractor assigns confidence.

### 3.3 Event Stream Registration and Position

Records stream identity, expected revision semantics, global or partition position, subscription checkpoints, retention, replay, compaction, and authority namespace.

### 3.4 Claim Identity

A backend-neutral semantic identity for an assertion family. It includes product and authority namespace plus canonicalized subject, predicate, object or value, qualifiers, and scope.

### 3.5 Claim Version

An append-only bitemporal version with validity, transaction time, authority, evidence relations, confidence, adjudication, conflict state, supersession, and deletion state.

### 3.6 Current Claim Projection

A materialized projection naming the current adjudicated version and generation. It is rebuildable from Claim Versions and adjudication records.

### 3.7 Evidence Relation

Binds evidence to a claim or model as support, contradiction, correction, context, derivation, evaluation, or deletion cause.

### 3.8 Adjudication and Conflict Record

Records the authority, rule, evidence considered, alternatives, confidence, outcome, and rollback path for add, revise, supersede, dispute, quarantine, merge, split, or reject decisions.

### 3.9 Episode and Cognitive Model Version

Records versioned episodes, owner models, source-person models, relationship models, self models, world/social/causal models, beliefs, predictions, skills, and mission projections.

Each remains derived unless a separately ratified authority role says otherwise.

### 3.10 Graph Projection Registration

Records ontology, source generations, node and edge mappings, temporal semantics, algorithms, embeddings, write-back policy, graph-to-claim reconciliation, deletion watermark, and rebuild recipe.

### 3.11 Vector Generation Registration

Records embedding model and digest, modality, preprocessing, dimensionality, metric, normalization, source generation, sharding/replication, calibration, deletion behavior, and rebuild recipe.

### 3.12 Object Manifest

Records payload digest, encryption, custody, physical copies, erasure coding, retention, integrity, references, deletion state, restoration behavior, and export lineage.

### 3.13 Durable Workflow and Activity Receipt

Records workflow identity, run, code version, input digest, history position, activity attempts, side-effect receipts, retries, signals, updates, cancellation, result, and repair state.

Workflow history is operational authority for orchestration, not knowledge authority.

### 3.14 Cross-Backend Transaction, Outbox, Inbox, and Saga Receipt

Records canonical write, projected writes, expected versions, delivery attempts, idempotency, reconciliation, compensation, and completion.

### 3.15 Synchronization and Replication Receipt

Records source and target component, positions, logical clock, causal parents, conflict, resolution, deletion watermark, encryption domain, and verification.

### 3.16 Dataset and Replay Manifest

Records selected evidence/claim/model versions, exclusions, sampling, weighting, purpose, consent/authority, deletion lineage, contamination checks, splits, and digest.

### 3.17 Model or Adapter Artifact Registration

Records base model, dataset manifest, code, environment, hyperparameters, compute, checkpoints, evaluation, failure cases, serving state, deletion limits, rollback artifact, and successor.

### 3.18 Inference and Serving Trace

Records selected models, tools, sources, claims, graph paths, vectors, context plan, rejected evidence, uncertainty, output digest, and downstream influence.

### 3.19 Deletion Propagation Receipt

Records authority request, exact target, completed and pending surfaces, technical limitations, rebuilds, retirement, restored-copy handling, and verification.

### 3.20 Cutover and Rollback Manifest

Records old and new authorities, generations, migration checks, shadow comparison, freeze point, cutover, rollback conditions, irreversible steps, and final disposition.

## 4. Identity and ordering

Opaque global IDs use a collision-resistant scheme such as UUIDv7. Store-assigned authoritative sequence or stream position determines committed order inside an authority.

Wall-clock timestamps support temporal semantics but do not independently establish transaction order.

Canonical values are type-tagged. Canonicalization version is recorded. Digest collisions require full semantic equality verification before reuse.

Idempotency combines namespace, key, and request digest. Reusing a key with a different request is an explicit conflict.

## 5. Truth and authority roles

At minimum:

```text
evidence_authority
claim_authority
operational_workflow_state
registered_projection
cache
replica
archive
candidate
model_artifact
evaluation_artifact
```

A record may not claim a stronger role than its registered component and profile permit.

## 6. Provenance graph

Provenance is a directed graph from observations and supplied sources through extraction, adjudication, projection, context, inference, action, outcome, training, model, and later decision influence.

Material edges are append-only except for authorized privacy deletion and cryptographic-key destruction. Corrections add new records and supersession links.

## 7. Privacy, product isolation, and export

Public repositories use neutral schemas and opaque identifiers. Private payloads, owner-specific paths, credentials, cryptographic keys, source-person material, and owner models remain outside Git.

Exports declare product, authority namespace, permitted fields, recipient, purpose, expiry, deletion contract, and digest. An export does not silently merge authority namespaces.

## 8. Deletion

Deletion can remove or cryptographically destroy payloads while retaining the minimum tombstone or proof needed to prevent restoration and relearning, subject to owner authority and legal constraints.

All affected derivatives, datasets, models, replicas, archives, and restores are represented in the deletion graph. Technical limits are truthful state, not permission to hide residual influence.

## 9. Registration rule

An unregistered database file, graph, vector collection, model, dataset, stream, workflow, replica, or archive cannot act as production authority.

Registration may be automated under an authorized profile, but it must produce inspectable records and a successor or retirement path.
