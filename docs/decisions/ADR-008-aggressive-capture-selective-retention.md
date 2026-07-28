# ADR-008 — Aggressive Capture with Selective Durable Retention

**Status:** Accepted
**Date:** July 28, 2026
**Decision owner:** MK Rayan

## Context

A.L.I.C.E. is intended to learn throughout years of conversation, research, tool use, coding, multimodal ingestion, experimentation, and model adaptation. A purely minimal-retention system would discard evidence that future models could use. A permanent store-everything system would accumulate duplicates, stale content, poisoned payloads, checkpoints, and retrieval noise faster than useful intelligence.

Continual-learning research supports replay of prior experience, but the useful object is a representative and evaluated replay set rather than indiscriminate equal retention of every event. Mature storage systems likewise separate frequently accessed data from colder archives and use lifecycle transitions and expiration rules.

## Decision

Adopt the doctrine:

> Aggressive temporary capture + permanent compact event ledger + utility-weighted durable retention + representative replay + encrypted tiered archive + verified deletion.

Phase 5 owns the storage substrate. Phase 8 owns learned curation and replay selection. Phase 13 consumes representative replay for model adaptation.

Full payloads use content addressing and host/encryption-domain-scoped deduplication. Cross-host deduplication is prohibited by default. Protected provenance, evaluation, training, rollback, owner-hold, and deletion-investigation dependencies block automatic deletion.

## Consequences

Positive:

- preserves opportunities for future reprocessing and model improvement;
- limits active retrieval noise;
- reduces duplicate storage;
- supports catastrophic-forgetting mitigation with representative replay;
- makes capacity pressure predictable;
- supports local-first Friday instances without vendor access;
- improves backup, restore, deletion, and audit correctness.

Costs:

- additional metadata and lifecycle complexity;
- need for background curation, archive movement, and restore tooling;
- need to evaluate retention decisions rather than using a simple time-to-live;
- cold storage and backup hardware may grow over time;
- model influence and deletion lineage remain difficult research problems.

## Rejected alternatives

### Store almost nothing

Rejected because it prevents future reprocessing, weakens causal learning, and provides insufficient replay for continual adaptation.

### Keep every payload permanently active

Rejected because it increases noise, cost, attack surface, and training redundancy without guaranteeing better intelligence.

### Cross-user global deduplication

Rejected because content equality can leak information between hosts and conflicts with Friday's vendor-non-access position.

### Immediate implementation in Phase 4

Rejected because Phase 4 is still building public information intelligence. The decision is ratified now, but runtime implementation starts in Phase 5.
