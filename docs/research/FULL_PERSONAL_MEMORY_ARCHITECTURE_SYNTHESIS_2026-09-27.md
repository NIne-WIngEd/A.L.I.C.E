# Full personal-memory architecture synthesis — 2026-09-27

**Status:** current architecture research basis  
**Purpose:** select the strongest current memory architecture for A.L.I.C.E./Fable without reducing capability for deployment convenience

## Governing product constraints

The architecture must support the *The Second Mind* destination and Fable/Friday parity doctrine:

- lifelong whole-life context without one giant prompt;
- selective but reconstructible memory;
- temporal, causal, social, mission, relationship and identity structure;
- native personal judgment and independent disagreement;
- host, relationship and assistant-self development;
- multimodal memory;
- model replacement without identity loss;
- multi-device continuity;
- correction/deletion/revocation;
- governed self-improvement with rollback;
- no permanent capability, model-size, context, graph, data, device or topology ceiling.

A technically elegant memory paper is rejected as the complete architecture if it cannot satisfy these requirements.

## Research synthesis

### MAGMA — orthogonal relational views

ACL 2026 MAGMA represents each memory across semantic, temporal, causal and entity graphs and uses intent-aware policy traversal.

**Adopt:** one logical Cognitive Multi-Graph with independently retrievable relation views and query-conditioned traversal.

**Do not copy as a ceiling:** A.L.I.C.E. also needs evidence/provenance, social/relationship, mission/project, skill/procedure, source-trust, identity, model-lineage and outcome views.

Primary source: https://aclanthology.org/2026.acl-long.1709/

### SYNAPSE / HippoRAG 2 — associative activation

SYNAPSE uses episodic-semantic graph structure, spreading activation, lateral inhibition and temporal decay. HippoRAG 2 uses graph propagation/Personalized PageRank with passage/concept structure.

**Adopt:** associative activation/graph propagation as a rebuildable retrieval computation over authoritative-linked graph projections.

**Boundary:** activation score is accessibility, never truth or Claim authority.

Primary sources:
- https://arxiv.org/abs/2601.02744
- https://arxiv.org/abs/2502.14802

### A-MEM — dynamic links and evolving notes

A-MEM shows the utility of atomic memory notes with contextual metadata and dynamic linking/evolution.

**Adopt:** dynamic semantic/association projections and linked summaries.

**Boundary:** historical source evidence is immutable; evolving note context cannot silently rewrite the underlying event/Claim authority.

Primary source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/19909c36f51abc4856b4560aff3d36d6-Abstract-Conference.html

### Nemori / GAM — event boundaries and fast/slow consolidation

Nemori learns semantic episode boundaries and uses prediction gaps for adaptive learning. GAM separates rapid event progression from slower topic/associative consolidation.

**Adopt:** learned episode formation and two-timescale memory processing: immediate safe indexing plus asynchronous consolidation/reflection.

**Boundary:** consolidation remains a proposal/derived process behind deterministic authority.

Primary sources:
- https://arxiv.org/abs/2508.03341
- https://aclanthology.org/2026.acl-long.1600/

### MemOS — memory as a managed resource

MemOS unifies plaintext, activation and parameter-level memory under lifecycle/scheduling abstractions.

**Adopt:** explicit Memory Resource Manager/Scheduler spanning external, episodic, graph, vector, procedural, parametric and activation forms.

**A.L.I.C.E. addition:** keep Claim/Evidence authority separate from representation scheduling. Moving a memory between forms never changes what is true.

Primary source: https://arxiv.org/abs/2507.03724

### CreaMem — life-scene interference and dual coding

CreaMem partitions personalized memory by life scenes and stores both episodic and trait perspectives.

**Adopt:** overlapping dynamic scene/domain membership and dual episodic + behavioral/trait projections.

**Reject as fixed ontology:** Life/Work/Interest or any finite authored scene taxonomy cannot define a person's lifetime structure.

Primary source: https://arxiv.org/abs/2609.08550

### LongMemEval-V2 — procedural/environment experience and source-native search

LongMemEval-V2 tests static state, dynamic state, workflow knowledge, environment gotchas and premise awareness over very large history. Its coding-agent source-file baseline is a strong reminder that direct source search can outperform fixed retrieval pipelines.

**Adopt:** procedural/environment memory and agentic source-native retrieval as first-class memory paths.

**Boundary:** source-native retrieval complements, rather than replaces, vector/graph/episode retrieval.

Primary source: https://arxiv.org/abs/2605.12493

### Multimodal/perceptual personal memory

Recent parametric multimodal user-memory work shows that captions discard identity-critical perceptual information such as faces and voices.

**Adopt:** grounded referent extraction plus specialist perceptual embeddings/keys and protected personal perceptual memory.

**Boundary:** a learned perceptual key is a recognition projection, not historical authority.

Primary source: https://arxiv.org/abs/2608.28609

### Execution-state unlearning and agentic backflow

Execution-State Unlearning shows that deleting plaintext memory can leave summaries, plans and KV/cache state contaminated; provenance-guided selective replay can reconstruct the counterfactual runtime. Agentic-unlearning work separately highlights parameter-memory recontamination.

**Adopt:** one deletion influence graph spanning durable memory, derived projections, active execution state, training data and parametric personal models.

**Required consequence:** Fable's deletion contract must include active contexts/plans/cache generations and model influence, not only rows/files/vector points.

Primary sources:
- https://arxiv.org/abs/2609.04875
- https://arxiv.org/abs/2602.17692

## Physical infrastructure research

### Experience/Event Fabric — KurrentDB + NATS JetStream federation

Selected canonical Experience/Event store: **KurrentDB**, behind A.L.I.C.E./Fable-owned event/evidence contracts.

Why it wins the memory role:

- fine-grained immutable streams;
- exact expected-revision optimistic concurrency;
- idempotent event append semantics;
- persistent subscriptions with durable checkpoints;
- multi-stream atomic appends;
- event-native replay/history rather than adapting a generic message log into an event store.

Selected federation/edge-ingress transport: **NATS JetStream**.

NATS supplies:

- durable edge/device streams;
- replay and consumer state;
- replication;
- leaf/domain/federation patterns;
- transport between devices/services and the canonical Experience fabric.

The project layer still owns event identity, evidence semantics, product/host/device scope, causal clocks, correction/deletion lineage, custody and merge/reconciliation rules.

KurrentDB's KLv1 is not OSI-approved and restricts providing KurrentDB itself as a hosted/managed service. That is recorded as a commercial/deployment constraint. It does not make NATS a more capable Experience Ledger. If a future managed Fable service conflicts with KLv1, the event contract allows a licensed Kurrent deployment or a successor event store without changing cognitive semantics.

Sources:
- https://docs.kurrent.io/server/v26.1/
- https://docs.kurrent.io/clients/node/v1.3/appending-events
- https://docs.kurrent.io/server/v22.10/persistent-subscriptions
- https://docs.nats.io/reference/2.12/jetstream

### Claim Fabric — XTDB v2

Selected current Claim Authority: **XTDB v2**.

The automatic valid-time + system-time model directly matches the bitemporal Claim contract: "what was true/effective when" and "what did the system know when."

Its single-writer-per-database indexing architecture is acceptable for serialized Claim authority and can be partitioned by host/authority database when scale demands it. This does not make XTDB the raw high-volume event fabric.

Sources:
- https://docs.xtdb.com/intro/what-is-xtdb.html
- https://docs.xtdb.com/about/time-in-xtdb.html
- https://docs.xtdb.com/ops/troubleshooting.html

### Cognitive Graph — hardware-adaptive graph engine, same contract

A single engine is not allowed to become a graph-capability ceiling.

**Host-local/embedded default:** LadybugDB:
- MIT;
- embedded/serverless;
- property graph + Cypher;
- columnar disk storage;
- multi-core analytical queries;
- serializable ACID;
- graph-algorithm extension including PageRank.

**Scale-out private-cluster path:** NebulaGraph:

- symmetrically distributed;
- separated storage/compute services;
- horizontal scalability;
- Raft-backed strong consistency;
- openCypher-compatible query language;
- native distributed graph deployment.

This is a cleaner scale-out continuation of the Cypher-oriented graph contract than introducing a Gremlin-only abstraction plus a separate Cassandra/HBase storage choice.
- Apache-2.0;
- distributed property graph;
- pluggable Cassandra/HBase/Scylla-class storage;
- no single-master graph bottleneck under suitable backends.

**Existing A.L.I.C.E. reference:** Neo4j remains valid for current projections and comparison evidence. It is not the universal Fable physical dependency because Community GDS limits algorithm concurrency to four cores while clustering/sharding/expanded GDS require commercial editions.

The logical graph and associative-compute contracts are identical across placements.

Sources:
- https://ladybugdb.com/
- https://docs.ladybugdb.com/get-started/graph-algorithms/
- https://janusgraph.org/
- https://neo4j.com/docs/graph-data-science/current/introduction/

### Vector/Multimodal Plane — Qdrant

Selected engine: **Qdrant**.

Reasons:

- dense + sparse vectors;
- named vectors;
- multivectors / late-interaction representations;
- local/embedded Edge path;
- server/cluster path;
- sharding/replication and Raft topology;
- Apache-2.0 server engine.

This allows the same logical semantic/multimodal plane to move from one host to a private cluster without removing the capability.

Sources:
- https://qdrant.tech/documentation/concepts/points/
- https://qdrant.tech/documentation/edge/
- https://qdrant.tech/documentation/scaling/distributed_deployment/

### Durable Workflow — Temporal

Selected workflow engine: **Temporal**.

Projection, migration, repair, deletion, training/evaluation, long-mission and self-improvement workflows require durable state/replay/retries. Temporal is MIT-licensed and can be self-hosted.

Source: https://docs.temporal.io/

### Workspace/cache — Valkey + process-local L1

Valkey is the shared/distributed ephemeral state/cache plane. Process-local L1 remains an optimization.

No cache contains unique authority.

Source: https://valkey.io/

## Final architecture decision

Do not replace Memory v4.1 with a paper architecture.

**Expand Memory v4.1 into the full personal cognitive fabric described in the execution map:**

1. Raw Evidence/Object;
2. Experience/Event;
3. bitemporal Claim Authority;
4. Episodes/autobiographical memory;
5. Cognitive Multi-Graph;
6. associative graph compute;
7. vector/multimodal retrieval;
8. source-native/live retrieval;
9. perceptual personal memory;
10. personal cognitive-model projections;
11. parametric personal memory;
12. working/activation memory;
13. MFM fast/slow formation;
14. deterministic Memory Gate;
15. Memory Resource Manager/Scheduler;
16. Retrieval Orchestrator/Cognitive Recollection;
17. Lifecycle Curator;
18. cross-layer deletion/unlearning;
19. durable workflows;
20. workspace/cache;
21. model/dataset lineage;
22. multi-device/federation.

This architecture is intentionally larger than any single cited paper because A.L.I.C.E./Fable has a larger objective than the benchmarks those papers target.

## Validation boundary

The decision above does **not** authorize a new MC10-style technology tournament.

Validate:

- that the selected implementation actually satisfies its required contract;
- integration boundaries that can corrupt authority or identity;
- real scale/latency/resource behavior;
- failures that would change an implementation decision.

Do not validate every alternative merely because an alternative exists.
