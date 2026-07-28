# Shared Kernel Extraction Standard

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
