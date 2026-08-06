# Memory M1 Decision Register

**Version:** 1.0.0
**Status:** M1-DX0 through M1-D9 owner-ratified on 2026-08-05
**Last updated:** 2026-08-05

| ID | Decision | Ratified disposition | Ratification evidence |
|---|---|---|---|
| M1-DX0 | Capability Supremacy | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-DX1 | Polyglot Cognitive Fabric | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-DX2 | Deployment and federation profiles | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-DX3 | Learning and training profiles | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D0 | Store and Capability Fabric Registry | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D1 | Claim Identity and Version | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D2 | Authority and adjudication | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D3 | Stability and cognitive models | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D4 | Temporal and distributed semantics | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D5 | Adaptive context and serving | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D6 | Durable curation and mission workflows | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D7 | Deletion, rollback, and model influence | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D8 | Cognitive graph | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |
| M1-D9 | Migration, federation, and scale | `ratified` | [owner ratification](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md) |

## M1-DX0 — Capability Supremacy

**Ratified decision**

A.L.I.C.E. shall remain owner-sovereign, local-capable, and deployment-unbounded. No present technology, topology, model, scale, context strategy, graph system, workflow engine, training method, research sequence, or deployment form establishes a permanent ceiling.

A temporary limit is valid only inside an exact profile and must record reason, evidence, review point, removal criterion, successor path, research state, shadow state, production activation condition, and rollback or exit.

**Preserved invariants**

Owner sovereignty, privacy, truthful state, provenance, product isolation, deletion, rollback, authority custody, and shutdown control remain binding.

## M1-DX1 — Polyglot Cognitive Fabric

**Ratified decision**

Logical evidence, claim, graph, vector, object, workflow, model, dataset, synchronization, inspection, deletion, and rollback contracts are independent of physical engines.

Each authority type has one registered canonical owner per generation. Secondary systems are registered projections, replicas, indexes, caches, archives, or challengers.

Cross-backend work uses expected versions, idempotency, outbox/inbox, event sourcing, sagas, durable workflows, reconciliation, projection generations, causal metadata, deletion watermarks, and repair.

## M1-DX2 — Deployment and federation profiles

**Ratified decision**

The same logical contracts may be implemented across edge, mobile, workstation, multi-GPU, home-cluster, private-cluster, hybrid-cloud, distributed, and frontier-research profiles.

Profiles declare custody, topology, consistency, availability, resource budgets, privacy, deletion, rollback, and authority. A profile does not define destination scale.

## M1-DX3 — Learning and training profiles

**Ratified decision**

Candidate generation, offline research, challenger training, shadow inference, canary influence, production influence, and A5 evolution are distinct activation states.

Research and challenger construction may proceed under lineage, containment, privacy, and evaluation controls. Production influence requires the selected profile and authority.

## M1-D0 — Store and Capability Fabric Registry

**Ratified decision**

Every database, stream, graph, vector collection, object store, workflow runtime, model server, training cluster, dataset, replica, archive, or synchronization endpoint that materially participates in cognition must be registered.

Required fields:

```text
component_id
component_kind
authority_role
product_id
authority_namespace_id
host_or_cluster_id
deployment_profile
backend_type
backend_version
schema_or_contract_version
capability_descriptor
consistency_model
availability_model
encryption_domain
region_or_device_scope
generation
state
derives_from[]
replicates[]
synchronizes_with[]
deletion_endpoint
rollback_endpoint
backup_profile
health
performance_profile
cost_profile
owner
created_at
superseded_by
```

## M1-D1 — Claim Identity and Version

**Ratified decision**

Claim semantic identity includes product, authority namespace, canonical subject, predicate, value/object, qualifiers, and scope. Host, shard, replica, encryption routing, and physical backend locator are deployment bindings rather than semantic identity.

Claim IDs are opaque collision-resistant identifiers such as UUIDv7. Committed order comes from a store-assigned sequence or authoritative stream position. Claim versions are append-only and bitemporal.

Canonical values are type-tagged and versioned. Idempotency includes namespace, key, and request digest. Digest reuse requires full semantic equality verification.

Identity retirement appends a terminal lifecycle record. Authorized privacy deletion follows the deletion standard and does not rewrite unrelated authority history.

Distributed metadata records partition, stream position, logical clock, causal parents, replication state, federation identity, and conflict resolution.

## M1-D2 — Authority and adjudication

**Ratified decision**

Authority is explicit and profile-bound. Automated adjudication may accept, revise, supersede, dispute, quarantine, merge, split, or reject candidates according to authority class, evidence, confidence, consequence, and evaluation.

Constitutional and identity-kernel activation remains owner-ratified.

## M1-D3 — Stability and cognitive models

**Ratified decision**

Preferences, values, traits, goals, temporary state, source-person hypotheses, relationship models, self/world/social/causal models, beliefs, and predictions are versioned projections with evidence thresholds, uncertainty, drift, conflict, and rollback.

Generated reconstruction remains distinct from source history.

## M1-D4 — Temporal and distributed semantics

**Ratified decision**

Records distinguish valid time, transaction time, event-stream time, logical time, and causal order. Late events and backfills preserve original occurrence time and new transaction time. Replication conflicts produce receipts and adjudication rather than silent overwrite.

## M1-D5 — Adaptive context and serving

**Ratified decision**

Context plans optimize quality, mission value, privacy, latency, cost, freshness, uncertainty, and available compute. Compact packets, very large contexts, iterative retrieval, graph traversal, sources, tools, agents, and simulations are selectable strategies.

## M1-D6 — Durable curation and mission workflows

**Ratified decision**

Long-running work uses durable identity, replayable orchestration decisions, idempotent activities, side-effect receipts, signals, updates, retries, checkpoints, cancellation, repair, and code-version migration.

Workflow state remains separate from claim authority.

## M1-D7 — Deletion, rollback, and model influence

**Ratified decision**

Deletion tracks every authoritative and derived influence surface. Residual model influence and unlearning limits are disclosed. Restores apply active deletion lineage before use.

Rollback covers code, schema, configuration, model, dataset, workflow, projection, deployment, and authority cutover.

## M1-D8 — Cognitive graph

**Ratified decision**

The Cognitive Graph is a first-class reasoning plane with registered ontology, temporal semantics, provenance, projection generations, algorithms, embeddings, and reconciliation.

Graph outputs are candidates or projections unless explicitly adjudicated into Claim Authority.

## M1-D9 — Migration, federation, and scale

**Ratified decision**

Migration may target any registered backend composition. Research may evaluate alternatives in parallel. Production cutover uses shadow comparison, reconciliation, deletion verification, failure recovery, owner-namespace isolation, and rollback.

Scale certification progresses through measured checkpoints and can extend without an architectural maximum.

## Owner resolutions

The owner resolved all five open M1 questions on 2026-08-05.

1. The deployment profile is named `distributed_multi_region`. This does not establish a permanent topology, provider, geographic, federation, or scale ceiling.
2. Identity-adjacent research, dataset construction, training, simulation, shadow evaluation, and canary testing are authorized under applicable profiles. Final constitutional identity authority remains profile-governed and evolvable rather than permanently prohibited.
3. The first canary profile begins with low-consequence operational claims. Sensitive claim classes remain available for research and may progress into later specialized canary or production profiles when their evidence and authority requirements are satisfied.
4. Deletion governs the required outcome rather than mandating one permanent technical mechanism. Verified unlearning, retraining, model editing, adapter removal, rollback, key destruction, isolation, withdrawal, replacement, or retirement may be selected according to evidence.
5. The Cognitive Graph begins as a provenance-linked projection and reasoning plane but may later hold canonical authority for one or more cognitive domains under a registered, evidence-backed authority profile.

The complete binding language is recorded in [`MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md`](MEMORY_M1_OWNER_RATIFICATION_2026-08-05.md).
