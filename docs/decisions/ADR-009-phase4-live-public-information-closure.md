# ADR-009 — Phase 4 Operational Live-Public-Information Closure

**Status:** Accepted
**Date:** July 30, 2026
**Decision owner:** MK Rayan

## Context

P4.0–P4.9 created and released a strong deterministic information architecture:

- provider-neutral contracts;
- controlled HTTPS transport;
- injection resistance;
- freshness and conflict handling;
- citation grounding;
- Phase 3 projection;
- bounded orchestration;
- explicit research mode;
- adversarial evaluation;
- exact-commit release audit.

The released end-to-end provider path nevertheless remains deterministic-fixture-only. The release audit deliberately blocks live network access and therefore cannot establish that A.L.I.C.E. can perform current public-web research with a real provider.

The capability catalog correctly continues to mark fresh web research and read-only public information tools as in development.

## Decision

Add P4.10 as an additive post-release operational closure milestone.

P4.9 remains a valid fixture-governed compatibility release. P4.10 does not weaken or rewrite its guarantees.

P4.10 must add a narrow live PUBLIC research profile and real-provider acceptance. It must not activate Phase 5 storage, private-data transmission, authenticated browsing, external actions, recursive browsing, or background operation.

Phase 5 is blocked until P4.10 is approved and merged.

## Consequences

- Phase 4 status becomes “fixture-governed compatibility release complete; operational live closure active.”
- README and Roadmap no longer imply that fixture execution alone proves current-web access.
- A permanent every-file phase-boundary audit is added.
- Later information profiles may remain broader, but P4.10 uses a narrow acceptance profile.
- A second exact-commit private record will bind live acceptance and rollback evidence.
