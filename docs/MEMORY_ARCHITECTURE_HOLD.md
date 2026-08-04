# A.L.I.C.E. Memory Architecture Hold

**Status:** Owner-approved; canonical Claim Store direction accepted<br>
**Effective:** 2026-08-03<br>
**Scope:** A.L.I.C.E. repository and Personal Cognitive Kernel memory/storage work<br>
**Supersedes:** Ordinary Phase 5 progression where it conflicts with this hold<br>
**Owner:** MK Rayan

## 1. Decision

A.L.I.C.E. is under a Memory Architecture Hold.

P5.1e and all new memory, retention, storage-pressure, curation, consolidation,
belief, personality-model, relationship-model, procedural-memory, replay,
adapter-learning, and model-training runtime work are paused until Memory
Architecture v4 is ratified.

The hold exists because A.L.I.C.E.'s public and technical claims depend on a
coherent long-term memory system. Continuing to add individually correct stores
without a unified authority, serving, curation, performance, and migration model
would increase architectural risk.

## 1.1 Accepted architecture decision

On 2026-08-03, the owner accepted the following Memory Architecture v4
direction:

- the Phase 5 Experience Ledger remains the immutable evidence and event layer;
- a new logical append-only, bitemporal Claim Store becomes the canonical
  adjudicated-knowledge layer;
- the Phase 2 Memory Core becomes a compatibility projection during shadow
  migration rather than the permanent universal source of truth;
- episodes, missions, beliefs, preferences, traits, relationship models,
  identity models, graphs, summaries, indexes, caches, and context blocks remain
  versioned and rebuildable derivatives;
- Claim Store v1 uses new Cognitive Kernel contracts and tables in the existing
  host-local SQLite architecture. It does not require a new microservice or
  database product;
- ordinary correction and supersession append new versions. Authorized privacy
  deletion remains a separate propagation and excision workflow.

This decision is accepted as the canonical authority topology. The broader hold
remains active until the complete v4 architecture, performance standard,
migration plan, deletion model, and evaluation gates are ratified.

## 2. What the hold blocks

The following work must not enter production code while the hold is active:

- P5.1e Storage Capacity Safety and Admission Control implementation;
- automatic memory formation or promotion;
- Learning Curator runtime behavior;
- automatic consolidation, reflection, or belief revision;
- autonomous retention, forgetting, or payload deletion;
- new persistent memory types without a ratified record standard;
- graph-memory production activation;
- owner, source-person, relationship, self, world, or social-model runtime;
- personal adapters, LoRA, weight editing, or parametric personal memory;
- Phase 2 Memory Core migration;
- replacement of the Experience Ledger, raw buffer, or existing Memory Core;
- Friday memory capability activation derived from unratified A.L.I.C.E. work.

## 3. What remains allowed

The following work may continue:

- documentation branches and pull requests required to review, ratify, and record Memory Architecture v4;
- documentation, architecture, ADR, and policy drafting;
- read-only repository and private-store audits;
- benchmark harnesses and synthetic evaluation fixtures;
- performance profiling that does not change production behavior;
- isolated prototypes that cannot write to authoritative or private stores;
- critical bug fixes required to preserve already merged behavior;
- security, privacy, deletion, or integrity repairs;
- exact-state repository cleanup and release-record preservation.

## 4. Existing foundations remain valid

The hold does not roll back the following released foundations:

- Phase 2 authoritative memory schemas, authorization, provenance, correction,
  conflict, candidate staging, sensitive storage, and deletion foundations;
- Phase 5 Experience Ledger, raw buffer, content-addressed payload store,
  lifecycle journal, and governed source-preserving tier transitions;
- Mission Graph and Cognitive Workspace contracts;
- owner sovereignty, product isolation, and Friday release governance.

These components are inputs to Memory Architecture v4. They are not automatically
the final architecture.

## 5. Required hold deliverables

The hold cannot be lifted until the following are ratified:

1. `MEMORY_CLAIM_COVERAGE_MATRIX.md`
2. `MEMORY_EXTERNAL_SYSTEMS_REVIEW.md`
3. `MEMORY_ARCHITECTURE_V4.md`
4. `MEMORY_RECORD_AND_PROVENANCE_STANDARD.md`
5. `MEMORY_PERFORMANCE_AND_RELIABILITY_STANDARD.md`
6. `PHASE2_TO_KERNEL_MEMORY_MIGRATION_PLAN.md`
7. `MEMORY_RENOVATION_PLAN.md`
8. a benchmark and scale-test plan covering correctness, staleness, mutation,
   deletion, latency, storage growth, poisoning, and rollback;
9. explicit public-claim release language for implemented, experimental, and
   destination capabilities.

## 6. Exit criteria

The hold may be lifted only when:

- every material public README memory claim maps to an implementation and test;
- evidence, authoritative claims, episodes, beliefs, predictions, projections,
  skills, and model-derived memories have non-overlapping authority semantics;
- the synchronous memory path has hard latency and token budgets;
- background curation has bounded queues, quotas, retries, heartbeats,
  idempotency, backpressure, and degraded modes;
- Phase 2 and Phase 5 have a migration path with no dual source of truth;
- correction, deletion, and rollback propagate through every derivative;
- graph, vector, lexical, and summary indexes are explicitly derived and
  rebuildable;
- personal parametric learning remains blocked unless deletion, rollback,
  replay lineage, and unlearning limits are ratified;
- synthetic scale tests pass at the ratified record counts;
- no critical or major architecture finding remains unresolved;
- MK Rayan records explicit approval to resume implementation.

## 7. Exception rule

An exception requires an exact-scope owner directive recording:

- the files and behavior allowed;
- the reason the work cannot wait;
- the tests and rollback plan;
- whether the exception changes a public claim;
- the date the exception expires.

Absence of an exception means the hold remains controlling.
