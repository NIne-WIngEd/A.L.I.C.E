# Shared Kernel Extraction Standard

> [!IMPORTANT]
> **OWNER-RATIFIED FLAGSHIP CAPABILITY RULE:** A.L.I.C.E. is the flagship and mandatory default capability upstream. Through at least completion of A.L.I.C.E. Phase 15, Friday must receive every transferable A.L.I.C.E. capability. Friday may gain a new capability only after A.L.I.C.E. has implemented, evaluated, approved, and gained it, unless MK Rayan records an explicit exact-scope owner override.
>
> This owner-ratified rule supersedes conflicting capability-order, team-independence, or Phase 6.5 repository-creation language in this document.


## 1. Purpose

This standard prevents A.L.I.C.E.'s owner-specific implementation from becoming inseparable from Friday's general product.

## 2. Product-neutrality requirements

Shared code must:

- accept a `ProductIdentity` and `HostInstance` explicitly;
- use instance-scoped paths and encryption domains;
- avoid literal user names and machine paths;
- load authority, memory, learning, and privacy behavior from manifests;
- support synthetic host tests;
- expose stable versioned contracts;
- record product and host scope on every persistent event;
- carry content digest, retention class, storage tier, provenance, and deletion lineage across storage interfaces;
- implement deduplication inside a host and encryption domain, never across unrelated hosts by default;
- avoid vendor services as mandatory dependencies;
- preserve data export and deletion lineage.

## 3. No phase immunity

Phase 1–4 code, schemas, policies, tests, and documents may be refactored or replaced when they violate this standard. Released behavior is retained through compatibility profiles and migrations where useful, not through architectural veto.

## 4. Data-isolation tests

Every shared storage or learning component must prove:

- host A cannot retrieve host B data;
- product A cannot silently read product B state;
- shared caches include host scope or contain no personal content;
- logs and exceptions do not leak raw content across scopes;
- training jobs consume only declared host datasets;
- exports contain only the selected host identity;
- storage pressure cannot delete another host's records or protected lineage;
- backup, archive, restore, and replay manifests remain host-scoped.

## 5. Extraction workflow

1. identify reusable module;
2. document current owner-specific dependencies;
3. introduce neutral interface and manifest configuration;
4. add synthetic multi-host tests;
5. migrate A.L.I.C.E. through an adapter;
6. verify unchanged released behavior where required;
7. move module into shared namespace/package;
8. publish version and migration note;
9. consume from A.L.I.C.E. and Friday independently.

## 6. Storage lifecycle contract

The Phase 5 kernel must expose product-neutral interfaces for the compact event ledger, raw buffer, content-addressed blobs, lifecycle decisions, storage accounting, replay manifests, archive/restore, and deletion propagation. A.L.I.C.E. and Friday may select different hardware or retention profiles, but they share the same contract and evaluation vocabulary.


## 7. Independent repository consumers

Friday product source never enters the A.L.I.C.E. repository. A.L.I.C.E. and Friday pin explicit kernel contract versions and may not import each other's private source. Phase 6.5 certifies independent readiness rather than initiating repository creation.

## 8. Mission and workspace contracts

The kernel may contain Mission Graph, node/edge, semantic-routing, Result Capsule, traceback, attention-decision, workspace-projection, speaker-context, guest-session, guest-grant, and authority-request contracts. Shared UI components remain projections over canonical state and contain no product-private data.

## 9. Private companion exclusion

Private companion sources, plaintext directives, owner model, relationship state, keys, encrypted payloads, derived private examples, and host-specific adapters are A.L.I.C.E. product state. The kernel may define neutral provenance and confidentiality schemas, but no real private payload or correlating ciphertext enters its distribution or fixtures.
