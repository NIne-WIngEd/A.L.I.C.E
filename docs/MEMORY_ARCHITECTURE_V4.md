# Memory Architecture v4.1 — Capability-First Polyglot Cognitive Fabric

**Status:** Owner-ratified Memory M1 architecture
**Prepared:** 2026-08-05
**Supersedes:** Memory Architecture v4 physical-topology assumptions while preserving its evidence-first and claim-centered authority model
**Authority:** A.L.I.C.E. Constitution, owner directives, Capability Unblocking Policy, and ratified product-family isolation rules

## 1. Decision

A.L.I.C.E. memory is an owner-sovereign, local-capable, deployment-unbounded cognitive fabric.

The architecture separates logical authority from physical implementation. Embedded stores, relational systems, distributed SQL, event stores, graph engines, vector systems, object stores, durable workflow engines, model registries, training systems, edge devices, private clusters, hybrid deployments, and later technologies may participate when they improve capability, reliability, continuity, scale, or intelligence.

No current engine is the permanent topology. SQLite remains a valid reference, embedded, edge, compatibility, migration, and fallback backend. It does not define the destination topology.

Temporary operational limits are profile-scoped. They must be measurable, reviewable, removable, and connected to a successor path.

No permanent ceiling may be encoded for technology, topology, scale, model, context, deployment, graph, workflow, training, or research.

## 2. Enabling invariants

Every deployment and backend preserves:

1. **Owner sovereignty.** The owner controls constitutional authority, production activation, inspection, correction, export, deletion, rollback, pause, and shutdown.
2. **Truthful material state.** Implemented, research-active, prototype, shadow, canary, production, degraded, superseded, and retired states remain distinct.
3. **Epistemic integrity.** Evidence, claims, inferences, predictions, reconstructions, generated content, and beliefs are not silently collapsed.
4. **Provenance.** Material records, projections, datasets, models, actions, and decisions are traceable to evidence and processing lineage.
5. **Privacy and custody.** Private payloads, keys, credentials, owner models, and source-person material remain within owner-authorized custody domains.
6. **Product and host isolation.** A.L.I.C.E. state does not seed Friday, another product, or another host except through an explicit permitted export.
7. **Deletion.** Authorized deletion propagates through authority stores, projections, indexes, caches, replicas, datasets, replay, model influence, backups, and restored archives to the technically supported extent, with limitations disclosed.
8. **Rollback and recovery.** Material changes have a known-good recovery path or explicitly recorded irreversibility.
9. **Constitutional honesty.** Source history, source-person models, reconstruction inferences, and A.L.I.C.E. continuity remain distinguishable.

These invariants govern how capability is used. They are not research or technology ceilings.

## 3. Authority topology

### 3.1 Experience and evidence authority

The Experience Fabric records observations, supplied material, actions, model/tool invocations, outcomes, corrections, measurements, and processing events.

Candidate implementations include embedded append-only ledgers, KurrentDB, Kafka, Pulsar, distributed logs, immutable manifests, and content-addressed archives.

Requirements:

- append identity and authoritative ordering;
- expected-version or equivalent concurrency control;
- idempotent append;
- replay and subscription positions;
- payload and evidence bindings;
- correction and deletion lineage;
- tamper-evidence or integrity verification;
- export and replacement paths.

### 3.2 Claim authority

The Claim Fabric is the canonical adjudicated-knowledge authority.

It stores append-only bitemporal claim versions, current adjudicated projections, authority, evidence relations, conflicts, confidence, validity, transaction time, supersession, correction, deletion state, and federation identity.

Candidate implementations include SQLite, PostgreSQL, distributed SQL, a custom bitemporal store, FoundationDB-backed services, Datomic-like fact systems, and later evidence-selected engines.

Ordinary serving uses materialized current state. Historical reconstruction remains explicit and bounded.

### 3.3 Cognitive graph plane

The Cognitive Graph represents mission, identity, temporal, causal, social, relationship, project, skill, tool, concept, evidence, belief, prediction, outcome, and model-lineage relationships.

Candidate implementations include Neo4j, FalkorDB, Memgraph, Neptune, relational graph projections, and custom graph engines.

Graph algorithms may support retrieval, pathfinding, dependency analysis, community detection, similarity, centrality, anomaly discovery, embeddings, link prediction, planning, and explanation.

The graph is provenance-linked to evidence and claims. Graph convenience does not replace claim authority or erase history.

### 3.4 Vector and multimodal retrieval plane

Vector systems support semantic and multimodal retrieval over text, code, image, audio, video, sensor, scientific, and learned representations.

Candidate implementations include Qdrant, Milvus, Vespa, Weaviate, pgvector, FAISS, DiskANN, HNSW, and later systems.

Every vector generation records model, dimensionality, preprocessing, source generation, scope, build manifest, calibration, quality evidence, and deletion behavior. Multiple generations may coexist when the serving plan can calibrate and trace them.

### 3.5 Object and archive plane

Object storage holds raw evidence, payloads, datasets, checkpoints, models, replay manifests, exports, and backups.

Candidate implementations include local content-addressed storage, NAS, MinIO, S3-compatible stores, encrypted cloud archives, distributed object stores, erasure-coded systems, and offline media.

Physical placement is independent of logical authority. Every object has custody, retention, integrity, deletion, restoration, and successor metadata.

### 3.6 Durable workflow plane

Durable workflows coordinate Curator jobs, migration, deletion propagation, projection rebuilds, training, evaluation, long missions, repair, federation, and recovery.

Candidate implementations include Temporal, Dagster, Prefect, Ray, durable event consumers, and local durable runners.

Workflow history is operational state. It does not become claim authority. Activities are idempotent, side effects are receipt-bound, and retries cannot duplicate semantic writes.

### 3.7 Model, dataset, and training plane

This plane supports distributed inference, learned retrieval, routing, rankers, preference models, procedural models, adapters, LoRA, world models, source-person reconstruction components, continual-learning challengers, and later weight updates.

Candidate implementations include PyTorch Distributed, FSDP, DeepSpeed, Ray, Slurm, Kubernetes, vLLM, MLflow, model registries, and owner-authorized local or remote accelerators.

Every neural artifact records:

- base model and artifact digest;
- dataset and replay manifests;
- evidence and deletion lineage;
- exclusions and contamination checks;
- code and environment;
- hyperparameters and compute receipts;
- checkpoints and metrics;
- challenger comparisons;
- failure cases;
- serving profile;
- rollback artifact;
- known deletion and unlearning limitations.

Research, dataset construction, challenger training, and shadow serving may proceed when lineage and evaluation controls exist. Production influence follows the selected activation profile and owner authority.

### 3.8 Serving and adaptive context plane

The Context Planner selects the best combination of current claims, evidence, graph paths, vector results, episodes, sources, tools, models, simulations, and agents.

A compact packet may be optimal for one invocation. Another mission may justify very large context, iterative retrieval, graph traversal, full-source inspection, multiple agents, external memory tools, or simulations.

The planner optimizes measured answer quality, mission value, privacy, latency, cost, freshness, uncertainty, and resource availability. Context settings are profile defaults, not a universal ceiling.

### 3.9 Federation and synchronization plane

The federation plane supports owner-authorized multi-device, edge, cluster, regional, and cross-environment operation.

It records:

- authority namespace;
- device or cluster identity;
- replication generation;
- logical and causal clocks;
- source and target positions;
- consistency model;
- conflict receipts;
- reconciliation outcomes;
- encryption and custody domain;
- deletion watermarks;
- export and import lineage.

Federation does not create cross-owner access. Owner namespaces and product boundaries remain explicit.

## 4. Canonical cognitive flow

```text
multimodal experience and action streams
        ↓
durable evidence/event fabric
        ↓
candidate extraction, entity linking, temporal matching, and evaluation
        ↓
append-only bitemporal claim authority
        ↓
current adjudicated knowledge
        ↓
episodes, missions, commitments, world models, owner models,
source-person models, self models, causal models, skills, and predictions
        ↓
graph, vector, symbolic, neural, relational, and summary projections
        ↓
adaptive context and tool plans
        ↓
reasoning, action, learning, simulation, and training
        ↓
outcome observation and continuous revision
```

## 5. Canonical ownership and polyglot consistency

A.L.I.C.E. does not pretend that all backends share one global transaction.

For each authority type, one registered component is canonical at a given generation. Secondary systems are projections, replicas, indexes, caches, archives, or challengers.

Cross-backend consistency uses:

- outbox and inbox records;
- event sourcing;
- expected versions;
- idempotency keys;
- sagas and durable workflows;
- projection generations;
- causal metadata;
- conflict receipts;
- reconciliation and repair jobs;
- deletion watermarks;
- cutover and rollback manifests.

A projection never silently becomes authority. An authority change requires an explicit cutover record, validation evidence, and rollback or irreversible-transition disclosure.

## 6. Deployment profiles

The same logical contracts may be implemented through:

```text
embedded_edge
mobile
single_workstation
multi_gpu_workstation
home_cluster
private_cluster
hybrid_cloud
distributed_multi_region
frontier_research
```

Profiles select backends, budgets, consistency, availability, privacy, and activation state. They do not define the maximum future architecture.

## 7. Capability state model

Every material capability uses one of:

```text
destination
research_active
prototype_operational
shadow_evaluated
canary_enabled
production_profile_enabled
degraded
superseded
retired
compatibility_only
```

Documentation and public claims must identify the exact state and profile. A destination statement is not an implementation claim.

## 8. Deletion and rollback

Deletion is a cross-backend influence graph, not a single-row operation.

It tracks evidence, claims, current state, episodes, missions, graphs, vectors, objects, caches, summaries, replicas, datasets, replay, models, adapters, backups, exports, and restored archives.

Deletion receipts record completed, pending, technically limited, retired, rebuilt, quarantined, and verified surfaces. Restores apply active deletion lineage before serving or learning.

Rollback records configuration, code, schema, model, data generation, workflow, deployment, and authority transitions. A change lacking a reliable rollback path must record why it is irreversible and which containment and recovery controls remain.

## 9. Product isolation

The Personal Cognitive Kernel exposes host-neutral contracts. A.L.I.C.E. and Friday use separate product identities, authority namespaces, state, encryption domains, credentials, model artifacts, and release governance.

Transferable code and neutral schemas may flow through the approved product-family process. Personal evidence, owner models, source-person material, private adapters, and learned identity do not transfer.

## 10. Migration rule

Released Phase 2 and Phase 5 source and tests remain compatibility baselines, test oracles, migration sources, and fallback implementations.

Successor backends are introduced through registered authority contracts, shadow projections, reconciliation, dual-read or controlled dual-write periods, cutover evidence, and rollback manifests.

Delivery order governs production cutover. It does not restrict parallel research or prototypes.

## 11. M1 ratification boundary

This document proposes destination architecture and contracts. It does not assert that the Claim Fabric, graph plane, distributed vector plane, workflow engine, federation, or parametric learning system is implemented.

M1 ratification remains an explicit owner decision. Runtime activation begins only under a later authorized implementation profile.
