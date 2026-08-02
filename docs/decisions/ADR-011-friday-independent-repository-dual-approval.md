# ADR-011 — Friday Independent Repository and Dual Production Approval

> [!IMPORTANT]\n> **OWNER-RATIFIED FLAGSHIP CAPABILITY RULE:** A.L.I.C.E. is the flagship and mandatory default capability upstream. Through at least completion of A.L.I.C.E. Phase 15, Friday must receive every transferable A.L.I.C.E. capability. Friday may gain a new capability only after A.L.I.C.E. has implemented, evaluated, approved, and gained it, unless MK Rayan records an explicit exact-scope owner override.\n>\n> This owner-ratified rule supersedes conflicting capability-order, team-independence, or Phase 6.5 repository-creation language in this document.


**Status:** Accepted
**Date:** 2026-07-31
**Decision owner:** MK Rayan

## Context

The earlier plan deferred Friday repository creation to Phase 6.5 and allowed an interim product implementation inside A.L.I.C.E. The owner has now established an independent Friday repository from the first product commit, same-feature Mission Graph/Cognitive Workspace development through host-neutral contracts, and permanent two-key production governance.

## Decision

Friday product source never enters A.L.I.C.E. Phase 6.5 becomes an Independent Product Readiness Gate. Phase 5 adds shared semantic contracts and release-attestation schemas; Phase 6 builds the UI. Friday production promotion requires both an exact-artifact A.L.I.C.E. audit attestation and Rayan approval. Emergency rollback/disablement cannot add new behavior.

## Consequences

- product repositories and private state remain independent;
- generalizable Mission Graph/Cognitive Workspace behavior has an immediate parity lane;
- a Friday team may develop independently but cannot unilaterally ship;
- production signing is eventually enforced through exact-artifact manifests;
- the old repository-creation-at-6.5 language is superseded.

**Baseline superseded:** `07e95a85d27b0c08b08ab857c6d9b75cdf8a6446` where conflicting.
