# ADR-006 — Separate Friday Product and Personal Cognitive Kernel

**Status:** Accepted; consumer-release consequence amended 2026-09-27
**Date:** 2026-07-27

## Context

A.L.I.C.E. is an owner-specific flagship with personal data, research autonomy, and rapidly evolving capabilities. The intended commercial system must initialize a separate local intelligence for every host without inheriting Rayan-specific state or forcing vendor access to personal data.

Waiting until the final platform phase to generalize would embed owner-specific assumptions throughout the Experience Ledger, control plane, integrations, memory, and learning systems.

## Decision

1. Begin host-neutral kernel extraction at A.L.I.C.E. Phase 5.0.
2. Create separate Friday and shared-kernel repositories/packages at the Phase 6.5 Product Separation Gate.
3. Keep A.L.I.C.E. as the research flagship and first kernel consumer.
4. Build Friday as a separate product and host identity, not a rename or sanitized data copy.
5. Make every persistent kernel artifact product- and host-scoped.
6. Permit migration of Phase 1–4 files whenever required by the shared-kernel and Friday architecture.
7. Treat `Friday` as a working codename until public-name clearance is complete.

## Consequences

- New Phase 5+ contracts must be host-neutral from inception.
- Some completed Phase 1–4 modules and tests will be refactored.
- Product and host isolation become testable architecture requirements.
- The historical interpretation that Phase 8 creates an earliest credible Friday/Fable closed-alpha consumer release is superseded. Phase 8/F5 is an internal selective-memory and learning qualification milestone.
- F4 through F11 are internal engineering/qualification stages. The first consumer Fable v1 release requires the complete transferable personal cognitive foundation to pass one full release gate.
- Historical `closed_alpha`, `alpha`, and `beta` channel names may remain for controlled testing/attestation but do not define reduced consumer capability tiers.
- General feature models may remain replaceable API services in v1; the local personal cognitive foundation may not be omitted or outsourced.
- A.L.I.C.E. may remain more experimental than the commercial product after the full v1 boundary is met.
