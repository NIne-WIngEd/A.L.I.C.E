# ADR-005 — Project-Wide Capability Unblocking and Profile Runtime

**Status:** Ratification candidate
**Date:** July 27, 2026

## Context

The public repository contained release-local constraints in source contracts, policy loaders, tests, workflows, schemas, and historical documents. Several were worded or implemented as permanent project ceilings: all capabilities false, empty tool registries, exact phase/version binding, PUBLIC-only information flow, fixed providers, fixed budgets, no retry/fallback, no background learning, and mandatory manual production self-change.

Changing only governance documents would leave these implementation barriers active.

## Decision

1. Constitution 1.1 establishes that no legacy file can veto ratified direction.
2. A machine-readable capability-profile registry becomes the successor activation mechanism.
3. Existing Phase 1–4 restrictions are reclassified as compatibility profiles or historical records.
4. A mission-scoped evolvable runtime is added under `src/alice_evolution`.
5. A one-time migration scans and registers every tracked local barrier, including unpublished work.
6. CI refuses new unscoped permanent capability ceilings.
7. Current release tests remain available as compatibility evidence but do not control successor profiles.

## Consequences

- Current behavior can remain reproducible without defining A.L.I.C.E.'s maximum capability.
- New models, tools, data classes, providers, budgets, learning methods, and autonomy mechanisms can be introduced through profiles and missions.
- Some historical regression tests must be moved to an explicit compatibility matrix as their runtimes are refactored.
- The migration may expose previously hidden assumptions; those are implementation defects to resolve, not reasons to retain the old ceiling.
