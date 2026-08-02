# A.L.I.C.E.–Friday Separation Plan

> [!IMPORTANT]
> **OWNER-RATIFIED FLAGSHIP CAPABILITY RULE:** A.L.I.C.E. is the flagship and mandatory default capability upstream. Through at least completion of A.L.I.C.E. Phase 15, Friday must receive every transferable A.L.I.C.E. capability. Friday may gain a new capability only after A.L.I.C.E. has implemented, evaluated, approved, and gained it, unless MK Rayan records an explicit exact-scope owner override.
>
> This owner-ratified rule supersedes conflicting capability-order, team-independence, or Phase 6.5 repository-creation language in this document.


**Version:** 2.0.0
**Status:** Owner-ratified repository and product boundary
**Decision:** A.L.I.C.E., Friday, and the Personal Cognitive Kernel are independent identities. Friday product source never lives in the A.L.I.C.E. repository.

## 1. Effective topology

```text
personal-cognitive-kernel/
    Host-neutral libraries, schemas, state machines, evaluators, and migrations

A.L.I.C.E/
    Rayan-specific flagship, private adapters, private policies, and research

Friday/
    Consumer shell, onboarding, UI, packaging, updates, and product integrations

friday-model-packs/
    Optional signed generic model, adapter, and evaluation manifests
```

The kernel may begin as an independently versioned package boundary while extraction is underway, but no product may import another product's private implementation or state. Friday's repository exists before its first Friday-specific source commit.

## 2. Phase timing

- **Phase 5.0:** establish host-neutral kernel identity, Experience Ledger, storage/evaluation contracts, Mission Graph contracts, Result Capsules, traceback, attention/workspace projections, speaker context, guest grants, and release-attestation schemas.
- **Phase 6:** implement the Cognitive Workspace and control-plane UI against those contracts.
- **Phase 6.5:** Independent Product Readiness Gate. Prove independent builds, versioned kernel consumption, host isolation, release attestations, production-signing gates, migration, rollback, and parity tracking.
- **Phase 7 onward:** release and evolve products independently within the A.L.I.C.E.-first capability-precedent and dual-approval rules while sharing versioned contracts and evaluation suites.

## 3. Kernel ownership

The Personal Cognitive Kernel may own:

- product and host identity contracts;
- Experience Ledger and evaluation contracts;
- memory, belief, learning, and storage interfaces;
- Mission Graph and task-node state machines;
- Result Capsule and traceback contracts;
- attention decisions and workspace projections;
- speaker-context and guest-grant primitives;
- model/runtime abstraction;
- permission primitives;
- capability manifests and product-neutral migrations.

It must not contain:

- Rayan's memories, private companion source, private directives, identity, credentials, goals, models, or adapters;
- Friday customer data, host state, voice profiles, branding, account logic, or commercial configuration;
- hard-coded host names or machine paths;
- cross-product training data.

## 4. A.L.I.C.E. ownership

A.L.I.C.E. owns Rayan's Constitution and authority relationship, private vault, private companion materials, owner-specific Mission Graph, memories, beliefs, models, adapters, goals, experimental autonomy profiles, research integrations, and frontier experiments.

## 5. Friday ownership

Friday owns host enrollment, host-selected identity, product shell, Cognitive Workspace integration, installer/updater, consumer privacy defaults, hardware/model selection, local onboarding, accessibility, support, diagnostics, and signed downstream releases.

## 6. Data and repository isolation

Every persistent kernel record carries `product_id`, `host_instance_id`, `encryption_domain`, schema version, digest, provenance, retention class, storage tier, and deletion lineage. Two synthetic hosts must remain isolated across primary storage, caches, logs, backups, restore, evaluations, and deletion.

A.L.I.C.E. personal data must never seed Friday defaults, fixtures, training, model packs, or telemetry. Friday data must never enter A.L.I.C.E. private state.

## 7. Capability parity

Friday shares A.L.I.C.E.'s generalizable destination capabilities. A.L.I.C.E. may lead experiments; temporary productization lag is allowed. Permanent capability omission is not. Mission Graph and Cognitive Workspace behavior have a dedicated working-parity lane.

## 8. Production governance

Before formal handover, Friday work is limited to repository foundation, maintenance, and downstream productization of capabilities already eligible through A.L.I.C.E. After formal handover, a qualified Friday team may independently manage maintenance, product experience, packaging, support, and downstream implementation or testing of capabilities already eligible through A.L.I.C.E. A capability that A.L.I.C.E. has not yet gained remains proposal-only and must be routed upstream; Friday may not implement, activate, or test it as a Friday capability unless Rayan records an explicit exact-scope override. Production promotion requires both:

1. an A.L.I.C.E. audit attestation bound to the exact commit, artifacts, policies, migrations, models, evaluations, and rollback manifest; and
2. MK Rayan's explicit production approval bound to the same candidate.

Either may veto or return the candidate for revision. Emergency rollback, disablement, containment, and reversion to previously approved behavior are allowed. Emergency status may not introduce new capability or replacement production behavior. Only Rayan may amend this dual-approval rule.

## 9. Migration rule

Released Phase 1–4 artifacts remain compatibility baselines, not permanent authority. They may be migrated, generalized, replaced, or retired through versioned evidence-backed migrations.
