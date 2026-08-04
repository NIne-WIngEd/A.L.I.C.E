# A.L.I.C.E. Memory Renovation Plan

**Draft:** 0.2<br>
**Claim Store authority decision:** Accepted 2026-08-03<br>
**Controlling condition:** Memory Architecture Hold

## M0 — Hold and evidence baseline

Deliver:

- architecture hold;
- public-claim matrix;
- external systems review;
- repository memory inventory;
- private-store inventory;
- current latency and scale baseline.

No runtime behavior changes.

## M1 — Ratify Memory Architecture v4

### Accepted M1 decision

- Experience Ledger is evidence authority.
- A new logical append-only, bitemporal Claim Store is canonical adjudicated
  knowledge.
- Phase 2 is a compatibility projection during shadow migration.
- Derived memory and indexes remain rebuildable.
- Claim Store v1 remains inside the host-local SQLite architecture.

### Remaining M1 decisions

Ratify:

- exact Claim Identity and Claim Version schemas;
- authority lattice;
- adjudication and owner-review rules;
- temporal semantics;
- context packet;
- online/offline boundary;
- SLOs;
- deletion and rollback;
- shadow migration thresholds.

## M2 — Kernel memory contracts

Implement metadata-only contracts for:

- Memory Unit envelope;
- Evidence Binding;
- Claim Identity;
- Claim Version;
- Claim Evidence Relation;
- Claim Adjudication;
- Current Claim Projection;
- Episode;
- Projection Version;
- Context Packet;
- Retrieval Trace;
- Curation Task/Receipt;
- Durable Memory Workflow/Activity Event;
- Deletion Propagation Receipt.

No automatic memory formation.

## M3 — Retrieval and serving renovation

First production code improvements:

1. materialized current-state projection;
2. batch memory hydration;
3. batch conflict/provenance loading;
4. generation-bound index manifests;
5. bounded retrieval plan;
6. Memory Context Packet builder;
7. retrieval trace;
8. no-memory session mode;
9. stale-index degraded fallback.

This stage addresses the current N+1 and global-scan bottlenecks before adding
more memory.

## M4 — Evidence-to-candidate bridge

- durable outbox from Experience Ledger;
- deterministic workflow history and replay;
- session quiescence and debounce;
- cheap deterministic eligibility;
- candidate extraction;
- exact/hash deduplication;
- bounded existing-memory shortlist;
- model/prompt/policy lineage;
- no automatic promotion.

## M5 — Governed promotion and episodic consolidation

- claim matching;
- authority review;
- explicit add/update/supersede/dispute operations;
- episode construction;
- raw evidence preservation;
- summary regeneration;
- owner-review queue;
- rollback.

## M6 — Cognitive projections

Implement one projection at a time:

1. active owner constraints;
2. mission/commitment projection;
3. stable preference model;
4. goal/value model;
5. relationship model;
6. source-person model;
7. belief/prediction model;
8. self/world/social/causal projections.

Each projection requires a separate benchmark and release profile.

## M7 — Scale, staleness, and adversarial evaluation

Evaluate:

- LongMemEval;
- LoCoMo and LoCoMo-Plus style tasks;
- MemoryAgentBench;
- STALE implicit invalidation;
- Memora/FAMA stale-memory penalties;
- internal mission/outcome benchmarks;
- memory poisoning;
- deletion;
- rollback;
- model replacement;
- 10M-event scale.

## M8 — Learning Curator activation

Activate only low-risk operations first:

- duplicate rejection;
- topic/mission segmentation;
- episode candidate creation;
- explicit owner-memory candidate staging.

Belief revision, skill formation, learned retention, and forgetting remain
separate later gates.

## M9 — Parametric learning gate

Remain blocked until:

- replay lineage;
- held-out evaluation;
- champion/challenger promotion;
- rollback;
- contamination tests;
- deletion impact;
- unlearning limitations;
- owner approval

are complete.

## Current planning branch

The active documentation-review branch is:

```text
planning/memory-architecture-v4
```

The current repository patch is documentation only. It must not modify `src/`,
active capability flags, or Phase 5 runtime policies.

## First implementation branch after ratification

Proposed:

```text
feat/memory-v4-serving-foundation
```

Proposed initial scope:

- batch retrieval;
- current-state projection contract;
- context packet and retrieval trace;
- performance benchmark harness;
- no automatic curation;
- comparative prototype of a local SQLite durable workflow runner versus
  self-hosted Temporal, with no production activation.
