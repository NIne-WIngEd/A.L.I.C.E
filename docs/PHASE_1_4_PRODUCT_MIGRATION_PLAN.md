# Phase 1–4 Product and Kernel Migration Plan

**Status:** Required before the Phase 6.5 Product Separation Gate

Completed phases are working baselines, not protected architecture. This plan identifies the changes required to make their reusable parts compatible with A.L.I.C.E., Friday, multiple hosts, and the Personal Cognitive Kernel.

## Phase 1 — Private evidence foundation

Required changes:

- add `product_id`, `host_instance_id`, `schema_version`, and `encryption_domain` to persistent source, extraction, chunk, index, and audit records;
- replace fixed vault paths with instance-scoped storage providers;
- derive local storage keys from a host trust root rather than one repository-wide assumption;
- isolate lexical, vector, artifact, cache, and temporary directories by host;
- make ingestion configuration product-neutral and connector-manifest driven;
- support metadata-only preview before content ingestion for Friday onboarding;
- add encrypted export/import primitives needed by the Identity Capsule;
- ensure deletion and rebuild operate on one selected host and never cross product boundaries;
- remove public fixtures containing owner data and use synthetic host corpora.

Compatibility requirement:

- A.L.I.C.E. retains its current vault through an adapter and migration manifest.

## Phase 2 — Authoritative memory core

Required changes:

- scope every memory, candidate, supersession edge, provenance link, encryption record, and retrieval query to product and host;
- generalize `rayan_statement` or similar owner-specific statuses into host-statement semantics while preserving A.L.I.C.E. display labels through product configuration;
- separate authoritative fact, derived belief, preference, prediction, and procedural skill stores;
- expose host-neutral correction, deletion, export, rebuild, and promotion interfaces;
- add Identity Capsule serialization and compatibility metadata;
- ensure sensitive-memory keys are host-specific;
- support local training-lineage links from memories to adapters;
- add multi-host isolation and no-cross-product retrieval tests.

Compatibility requirement:

- released Phase 2 semantics remain selectable through an A.L.I.C.E. compatibility profile.

## Phase 3 — Conversation and orchestration

Required changes:

- replace dataclasses that reject every enabled capability with profile-resolved capability sets;
- replace exact Constitution, phase, provider, and milestone bindings with version ranges and named compatibility profiles;
- add product identity, host identity, model identity, and learning-event scope to turns and generation attempts;
- load voice, owner relationship, system prompt, and authority text from product identity packages;
- allow broader classifications, live retrieval, tools, memory writes, retries, fallback, and external actions when the selected profile permits them;
- make grounding source kinds and tool registries extensible;
- preserve hidden-reasoning privacy without treating all useful reflection or trace artifacts as categorically impossible;
- migrate tests that assert disabled features into explicit `conversation.phase3.compatibility` tests;
- add Friday synthetic-host conversation tests proving distinct behavior from the same base model.

Compatibility requirement:

- the exact released Phase 3 behavior remains reproducible, but it is not the default architecture for all future products.

## Phase 4 — Web and information intelligence

Required changes:

- scope queries, observations, citations, source-trust records, downloads, and learning envelopes to product and host;
- retain the original public read-only profile while enabling successor profiles for authenticated browsing, private context, background research, and learning output;
- replace closed provider and operation sets with registries;
- move fixed numeric budgets into product/mission profiles;
- connect every completed research task to the Experience Ledger contract;
- distinguish external content from executable instruction at every product boundary;
- allow Friday's visible network-egress ledger to consume provider and data-flow metadata;
- ensure remote requests exclude local personal context unless a host-authorized transformation explicitly includes it;
- add product-specific redaction and offline behavior.

Compatibility requirement:

- current P4.5 work may continue, but its contracts must not prevent the later host-scoped observation envelope.

## Cross-phase source files to inspect

At minimum, audit and migrate:

- `src/alice_ingestion/**`
- `src/alice_memory/**`
- `src/alice_conversation/**`
- `src/alice_information/**`
- all policies under `policies/` that bind exact phases, versions, providers, tools, classifications, or false capability booleans;
- all Phase 1–4 tests and fixtures;
- validators and CI workflows;
- local-only P4.5 files.

## Required gates

Before Phase 6.5:

1. two synthetic Friday hosts and one synthetic A.L.I.C.E. host run against the same kernel without cross-state access;
2. A.L.I.C.E. personal data is absent from Friday and shared-kernel artifacts;
3. all Phase 1–4 storage records are either host-scoped or explicitly content-free;
4. compatibility suites pass for released behavior;
5. broader product profiles pass their own tests;
6. export, deletion, rebuild, and migration work per host;
7. the repository-wide barrier scanner reports no unregistered active ceiling.
