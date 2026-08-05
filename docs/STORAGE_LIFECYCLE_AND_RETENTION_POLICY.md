# A.L.I.C.E. Storage Lifecycle and Retention Policy

**Version:** 1.0.0
**Authority:** A.L.I.C.E. Constitution 1.1.0, Memory and Knowledge Policy 3.1.0, and ADR-008
**Applies to:** A.L.I.C.E., the Personal Cognitive Kernel, and the consumer product internally codenamed Friday

## 1. Decision

A.L.I.C.E. uses **aggressive temporary capture with selective durable retention**.

The system should not discard a potentially important experience before it can be evaluated. It also should not assume that permanently retaining every raw byte increases intelligence. The durable design is:

```text
broad eligible observation
        ↓
permanent compact event ledger
        +
policy-bounded raw-experience buffer
        ↓
Learning Curator
        ↓
retain | compress | promote | replay | archive | quarantine | delete
```

## 2. Objectives

The storage system must maximize:

- future learning opportunity;
- provenance and reproducibility;
- retrieval quality;
- rare-case and failure retention;
- correction and deletion propagation;
- host privacy and product isolation;
- resilience and recoverability;
- intelligence gained per unit of storage and compute.

It must minimize:

- duplicate payloads;
- stale or low-value retrieval noise;
- poisoned persistent content;
- unnecessary checkpoint and cache growth;
- cross-host equality leakage;
- silent data loss under storage pressure.

## 3. Permanent compact event ledger

Every material event receives an append-only logical ledger record containing, where applicable:

- event identifier and timestamp;
- product and host instance;
- event type and task/mission context;
- source/provenance identifiers;
- content digest and payload location, when retained;
- sensitivity and encryption domain;
- outcome and correction links;
- importance, novelty, and utility scores;
- retention class and active tier;
- derived memory, belief, skill, dataset, model, or code identifiers;
- deletion lineage and lifecycle decisions.

The compact ledger may remain after the full payload is compressed, archived, or deleted. Owner-authorized deletion may reduce the ledger to a non-content tombstone when retaining descriptive metadata would conflict with the deletion request.

## 4. Storage tiers

| Tier | Purpose | Typical media | Learning availability |
|---|---|---|---|
| `ledger` | Compact permanent history, provenance, outcomes, and lifecycle state | transactional database | always searchable |
| `raw_buffer` | Recently captured full payload awaiting curation | profile-selected low-latency storage | available, not automatically trusted |
| `hot` | Active memories, evidence, indexes, current projects, current replay | NVMe/SSD | immediate |
| `warm` | Infrequently accessed but operationally useful artifacts | SSD/HDD/NAS | online with moderate latency |
| `cold` | Large historical sources, experiments, media, and retired checkpoints | encrypted HDD/NAS/offline or E2EE archive | restored on demand |
| `quarantine` | Untrusted, poisoned, malformed, or policy-disputed payloads | isolated encrypted store | excluded from learning and ordinary retrieval |
| `deleted` | Payload removed; minimal deletion lineage or tombstone remains | ledger only | unavailable |

Tiers describe access and lifecycle state. They do not weaken encryption, provenance, host isolation, or deletion requirements.

## 5. Initial retention classes

The policy engine provides configurable defaults rather than constitutional time limits.

| Class | Initial default | Required decision before expiry |
|---|---:|---|
| `authoritative_source` | no time-only expiry | owner deletion, supersession, or verified migration |
| `active_project` | project lifetime + review | retain, compress, or archive |
| `high_value_experience` | 365 days before review | promote, replay, archive, or delete |
| `ordinary_experience` | 90 days in raw buffer | curate before payload expiry |
| `transient_web_or_tool_cache` | 30 days | retain only when cited, reused, or promoted |
| `failed_experiment` | 180 days | preserve lesson, counterexample, evaluator, and reproducibility needs |
| `training_replay` | while referenced by active/challenger models | rebalance, archive, or retire |
| `quarantine` | policy-defined review window | clear, extract safe facts, archive for security research, or delete |
| `owner_hold` | no automated expiry | explicit owner release |

A profile may lengthen or shorten these periods after evaluation. Time alone may not delete protected artifacts.

## 6. Retention blockers

An artifact cannot be automatically deleted while it is required by:

- authoritative memory or cited-answer provenance;
- an unresolved correction, conflict, or deletion investigation;
- an active project or owner hold;
- a current training manifest or representative replay set;
- evaluation reproducibility;
- champion/challenger comparison;
- rollback or disaster recovery;
- a legally or contractually required hold explicitly accepted by the owner.

The system must expose why an artifact is retained and which dependencies block deletion.

## 7. Content addressing and deduplication

Full payloads are identified by SHA-256 or a later ratified cryptographic digest. Duplicate payload bytes may be stored once inside an owner-authorized authority namespace and key domain while logical records, provenance, retention, and deletion references remain distinct.

Cross-owner physical deduplication is outside ordinary profiles because shared digests or object identity can leak possession. Privacy-preserving deduplication may be evaluated inside an explicitly authorized owner namespace and key domain when leakage controls, deletion isolation, and accounting are demonstrated. Friday may not trade host privacy for vendor storage efficiency.

Deduplication never merges:

- provenance;
- sensitivity;
- retention class;
- access authorization;
- deletion lineage;
- learning influence.

Two records may share payload bytes while retaining distinct logical meaning and lifecycle state.

## 8. Learning Curator decisions

The Curator evaluates:

- future utility;
- novelty and redundancy;
- source quality and provenance;
- rarity and surprise;
- correction or contradiction value;
- causal and counterfactual value;
- task outcome;
- active-goal relevance;
- sensitivity and contamination risk;
- reproducibility needs;
- storage and retrieval cost;
- expected intelligence gain per byte.

Allowed decisions are:

```text
retain_raw
promote_hot
compress_to_summary
extract_fact_or_belief
extract_skill_or_failure_case
add_to_replay_manifest
move_warm
archive_cold
quarantine
retain_metadata_only
delete_payload
```

Important decisions record the evaluator version and evidence used.

## 9. Representative replay

Continual training does not use every stored event equally. Replay sets are explicitly versioned manifests selected for:

- distribution coverage;
- rare and surprising cases;
- corrections and owner feedback;
- successful and failed procedures;
- catastrophic-forgetting risk;
- causal counterexamples;
- identity and preference continuity;
- historical benchmark coverage;
- diversity across time, tasks, modalities, and sources.

Replay selection is budgeted and measured. A larger replay buffer is accepted only when it improves retention, calibration, robustness, or downstream task performance enough to justify its cost.

## 10. Storage pressure and capacity safety

Every ingestion, indexing, training, checkpointing, export, and rebuild job estimates peak temporary storage before execution.

Initial profile defaults reserve:

- 15% free capacity for ordinary operation;
- an emergency reserve profile at which low-priority work is deferred according to mission value, recoverability, and owner policy.

Before deleting useful durable information, the system attempts, in order:

1. remove exact duplicates and orphaned temporary files;
2. compress text, logs, and media derivatives losslessly where possible;
3. evict reproducible caches and build artifacts;
4. retire inferior model challengers and intermediate checkpoints after extracting their evaluation history;
5. move inactive payloads from hot to warm or cold storage;
6. reduce low-value raw capture and pause nonessential jobs;
7. request owner action when protected data still exceeds available capacity.

Storage pressure may never silently erase authoritative evidence, active rollback state, owner-held records, or deletion/audit lineage.

## 11. Backup and restoration

The resilience target is three copies across at least two media classes with one offline or off-site encrypted copy when resources permit.

Backups must be:

- encrypted with owner/host-controlled keys;
- content- and manifest-integrity checked;
- scoped by product, host, and encryption domain;
- restorable without vendor-held secrets;
- tested periodically through sampled and full restoration drills;
- versioned so corruption or accidental deletion is not immediately replicated into every recovery point.

Replication or mirroring alone is not considered a complete backup.

## 12. Deletion and relearning

Deletion propagates through:

- raw and tiered payload storage;
- indexes, caches, and derivatives;
- memory and beliefs;
- replay and training manifests;
- skills and evaluations that expose the payload;
- adapters or models where supported;
- future retraining queues;
- new backups and Identity Capsules.

Cold archives and immutable backups follow a disclosed purge schedule. A restored archive must replay active deletion lineage before becoming available.

Deleted payloads may not be deliberately reacquired or regenerated from retained traces merely to defeat the deletion request.

## 13. Multi-host and product isolation

Every storage object, ledger record, lifecycle decision, replay manifest, backup set, and restoration job is scoped by:

- `product_id`;
- `host_instance_id`;
- `encryption_domain`;
- `content_digest`;
- `retention_class`;
- `storage_tier`;
- `deletion_lineage`.

A.L.I.C.E. personal data never enters Friday distribution artifacts. Friday host data never enters another host or the shared-kernel distribution.

## 14. Phase mapping

### Phase 5

Implement the compact ledger, raw buffer, content-addressed blob store, host-scoped deduplication, retention classes, tier movement, accounting, storage-pressure controls, backup manifests, restore verification, and kernel-neutral interfaces.

### Phase 6

Expose storage, retention, backup, replay, archive, deletion, and capacity controls in the Cognitive Inspector.

### Phase 8

Train and evaluate the Learning Curator's retention, compression, replay-selection, archival, and forgetting decisions.

### Phase 13

Use representative replay and lifecycle lineage in continual model training, challenger evaluation, machine-unlearning experiments, and adapter retirement.

## 15. Evaluation

Measure:

- useful-signal recall before raw expiry;
- false deletion and false retention;
- duplicate bytes avoided;
- hot-tier hit rate and retrieval latency;
- storage growth by artifact class;
- archive restore success and time;
- backup integrity and recovery-point age;
- replay diversity and distribution coverage;
- forgetting reduction per replay byte;
- downstream task gain per retained byte;
- poisoning quarantine accuracy;
- deletion propagation completeness;
- cross-host isolation;
- storage-pressure behavior;
- owner ability to inspect and override lifecycle decisions.

## 16. Non-goals

The policy does not require:

- permanent retention of all raw interactions;
- universal cloud storage;
- vendor-visible backups;
- cross-host deduplication;
- deleting useful evidence merely to minimize data;
- retaining useless payloads merely because capacity exists;
- implementing Phase 5 runtime storage during Phase 4.

## Capability-first distributed storage amendment

Storage policy is backend-neutral. Registered classes may include:

- encrypted local filesystems and content-addressed stores;
- NAS and owner-controlled edge replicas;
- S3-compatible object storage and encrypted cloud archives;
- distributed and erasure-coded object stores;
- event-stream persistence and relational claim stores;
- graph and vector projections;
- model, dataset, checkpoint, and evaluation registries;
- cold, offline, offsite, and multi-region owner-authorized replicas.

Retention values, free-space reserves, copy counts, and tier placements are profile defaults. They may adapt to hardware, mission value, cost, risk, legal requirements, restore objectives, and measured failure modes. They do not define destination capacity.

Every backend must declare custody, encryption, authority role, consistency, durability, deletion mode, backup and restore behavior, exportability, health, and successor path. Restored data must apply active deletion lineage before serving or learning. A derivative that cannot support direct excision must be retired, rebuilt, or truthfully quarantined according to the deletion profile.

A.L.I.C.E. may retain high-value information across authorized storage classes. Local capability is preserved through owner-controlled copies, export, migration, and recovery rather than by limiting storage to one machine.
