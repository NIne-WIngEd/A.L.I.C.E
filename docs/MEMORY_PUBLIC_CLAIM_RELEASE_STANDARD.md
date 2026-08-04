# Memory Public Claim Release Standard

**Baseline commit:** `d5d311ec49f1e4a3e5a7cf688062c7dc2f46d4ec`<br>
**Generated:** `2026-08-04T05:07:30Z`<br>
**Status:** M0 controlling documentation draft

## 1. Purpose

This standard prevents architecture direction, storage foundations, experiments,
and production memory behavior from being described as the same thing.

## 2. Approved release labels

### Implemented

Use only when the exact runtime behavior exists, is enabled by the declared
capability profile, passes correctness/privacy/performance/recovery evaluation,
and is bound to a signed release artifact.

### Foundation implemented

Use when durable lower-level contracts or stores exist, but the complete
user-facing memory behavior is not available.

### Contract implemented

Use when schemas and interfaces exist without production activation.

### Experimental

Use only for isolated prototypes or evaluations that cannot mutate authoritative
or private production stores.

### Under active development

Use for scoped work with an approved implementation plan that has not completed
release gates.

### Destination capability

Use for long-term intended behavior that is not yet implemented.

### Blocked by Memory Architecture Hold

Use when the architecture explicitly prohibits implementation or activation.

## 3. Evidence required for “implemented”

1. exact commit and signed artifact;
2. active capability-profile entry;
3. implementation and migration status;
4. correctness and temporal-update tests;
5. privacy, authorization, and deletion tests;
6. latency, scale, and storage evidence;
7. crash recovery and rollback;
8. owner-visible inspection behavior;
9. public-claim coverage-matrix update;
10. explicit known limitations.

## 4. Prohibited language

Do not say that A.L.I.C.E. currently:

- learns autonomously from every interaction;
- maintains a complete owner, relationship, or source-person model;
- connects all missions, decisions, and outcomes;
- forgets safely across every derivative;
- remains behaviorally identical across arbitrary model replacement;
- performs long-term memory at proven million-record scale;
- uses a production Claim Store;

unless the exact capability has passed its release gate.

## 5. Current M0 language

Approved descriptions are:

- Phase 2 provides an authoritative-memory foundation with provenance,
  authorization, correction, temporal, retrieval, and deletion contracts.
- Phase 5 provides an Experience Ledger and governed storage substrate.
- Memory Architecture v4 ratifies the future Claim Store authority topology.
- Claim Store runtime implementation and Phase 2 migration have not started.
- P5.1e and autonomous memory work remain paused under the Hold.

## 6. Update rule

Every memory-related PR must state whether it changes:

- evidence capture;
- authoritative claims;
- episodes;
- missions;
- cognitive projections;
- context serving;
- curation;
- deletion/rollback;
- indexes;
- public capability language.

Silence means no capability advancement.
