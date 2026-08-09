# Stage G Memory Fabric Candidate Qualification Matrix

**Version:** 1.0.0
**Status:** Owner-ratified Stage G acceptance requirement
**Applies to:** Stage G candidate evaluation, complex synthetic Rayan-life stress testing, cross-plane interoperability, README/governance promise coverage, and Stage G closure

## 1. Purpose

Stage G does not close merely because one backend works, because the primary stack works, or because graph/vector persistence survives restart.

Before Stage G can close, A.L.I.C.E. must evaluate every concrete named candidate in this matrix against the same governed memory contracts and progressively scaled complex synthetic Rayan-life workload.

Every concrete candidate is qualified:

1. individually;
2. against every other concrete candidate in the same logical role;
3. across every registered architecturally meaningful cross-role interaction using candidate substitution;
4. in multi-plane combinations;
5. in complete end-to-end cognitive-memory runs;
6. under correction, deletion, revocation, restore, failure, recovery, concurrency, replay, rebuild, rollback, and scale stress;
7. against the promises made in `README.md`, memory governance, identity/host-learning rules, and the Phase 2 migration plan.

This matrix is a qualification obligation rather than a permanent technology selection. Passing Stage G does not make a backend permanent.

## 2. Canonical candidate inventory

| Role | Primary/current candidate | Other candidates/challengers |
|---|---|---|
| Raw Buffer / Workspace | **Redis-class** | in-memory reference; future distributed-state class |
| Experience/Event Fabric | **KurrentDB** | Kafka; Pulsar; embedded append-only ledger; distributed-log class |
| Claim Fabric | **distributed/bitemporal SQL-class** | PostgreSQL; distributed-SQL class; Datomic-like fact architecture; FoundationDB-backed custom store; SQLite edge/reference |
| Cognitive Graph | **Neo4j** | FalkorDB; Memgraph; Amazon Neptune; custom graph engine |
| Vector/Multimodal | **Qdrant** | Milvus; Vespa; Weaviate; pgvector; FAISS; DiskANN; HNSW |
| Object/Archive | **S3-compatible / MinIO / NAS** | encrypted cloud/object/archive class |
| Durable Workflow | **Temporal** | Dagster; Prefect; Ray; durable-consumer implementation; custom workflow service |
| Existing compatibility authority | **SQLite** | retained as Phase 2/reference oracle; not selected successor destination |
| Model serving | **vLLM** | other model-serving-stack class |
| Training | **PyTorch Distributed / Ray** | DeepSpeed; FSDP; Slurm; Kubernetes |
| Model/dataset registry | **MLflow-class / object-backed registry** | future registry-challenger class |

The matrix currently contains 32 required named role assignments representing 31 unique concrete product/library candidates because Ray is evaluated in both workflow and training roles.

The required concrete named candidates are:

```text
Redis
KurrentDB
Kafka
Pulsar
PostgreSQL
SQLite edge/reference
Neo4j
FalkorDB
Memgraph
Amazon Neptune
Qdrant
Milvus
Vespa
Weaviate
pgvector
FAISS
DiskANN
HNSW
MinIO
Temporal
Dagster
Prefect
Ray
SQLite Phase 2 reference oracle
vLLM
PyTorch Distributed
DeepSpeed
FSDP
Slurm
Kubernetes
MLflow
```

All concrete named candidates above are mandatory Stage G evaluation targets unless Rayan later ratifies an amendment to this matrix.

The table also contains implementation-family candidates that cannot be evaluated as names alone:

```text
in-memory reference
future distributed-state class
embedded append-only ledger
distributed-log class
distributed/bitemporal SQL class
distributed-SQL class
Datomic-like fact architecture
FoundationDB-backed custom store
custom graph engine
S3-compatible implementation
NAS implementation
encrypted cloud/object/archive class
durable-consumer implementation
custom workflow service
other model-serving-stack class
object-backed registry
future registry-challenger class
```

Before Stage G acceptance, each implementation-family candidate must either:

- be instantiated by at least one concrete runnable implementation and then receive the same applicable qualification as a named candidate; or
- be reclassified through a separate owner-ratified amendment as a descriptive architecture family rather than an active Stage G candidate.

A family placeholder never counts as passed without a concrete runnable implementation.

## 3. Qualification levels

Every required named candidate and every concretized implementation-family candidate produces evidence for all applicable levels.

### Q0 - Registration and reproducibility

Record:

- exact engine/product/library name;
- role being evaluated;
- version;
- image, package, build, or source digest;
- platform and architecture;
- configuration;
- deployment profile;
- schema or contract generation;
- dataset generation;
- benchmark generation;
- resource profile;
- encryption/custody domain where applicable;
- benchmark code and report hashes.

### Q1 - Individual candidate stress qualification

Test each candidate in isolation behind its logical contract.

Required dimensions include:

- correctness;
- ordering or transaction semantics;
- idempotency;
- concurrency;
- correction;
- deletion and revocation;
- replay and rebuild;
- restart durability;
- malformed input;
- partial failure;
- recovery;
- export and replacement;
- observability;
- latency, throughput, storage, memory, CPU/GPU, and network behavior where applicable.

Starting a service or persisting a sample record is not sufficient qualification evidence.

### Q2 - Same-role all-pairs comparison

Within every logical role, compare every concrete candidate against every other concrete candidate in that role using the same gold workload and logical contract.

Examples include:

- KurrentDB vs Kafka vs Pulsar;
- Neo4j vs FalkorDB vs Memgraph vs Amazon Neptune;
- Qdrant vs Milvus vs Vespa vs Weaviate vs pgvector vs FAISS vs DiskANN vs HNSW;
- Temporal vs Dagster vs Prefect vs Ray;
- PyTorch Distributed vs Ray vs DeepSpeed vs FSDP vs Slurm vs Kubernetes.

Record semantic loss, unsupported contract behavior, correctness, reliability, latency, throughput, operational complexity, recovery behavior, and resource profile.

### Q3 - Cross-role candidate-substitution interoperability

Every registered interaction edge is tested with every active candidate substitution on both sides of the edge.

Registered Stage G interaction edges are:

- Raw Buffer / Workspace <-> Experience/Event Fabric;
- Raw Buffer / Workspace <-> retrieval/model-serving context;
- Experience/Event Fabric <-> Claim Fabric;
- Experience/Event Fabric <-> Object/Archive;
- Experience/Event Fabric <-> Durable Workflow;
- Claim Fabric <-> Cognitive Graph;
- Claim Fabric <-> Vector/Multimodal;
- Claim Fabric <-> Durable Workflow;
- Claim Fabric <-> SQLite compatibility/reference oracle;
- Cognitive Graph <-> Vector/Multimodal;
- Cognitive Graph <-> retrieval/model serving;
- Vector/Multimodal <-> retrieval/model serving;
- Object/Archive <-> Training;
- Object/Archive <-> Model/Dataset Registry;
- Durable Workflow <-> Event, Claim, Graph, Vector, Object, Training, and Registry planes;
- Model serving <-> Claim, Graph, Vector, Workspace, Retrieval, and Context planes;
- Training <-> Object/Archive, Model/Dataset Registry, and Model serving;
- Model/Dataset Registry <-> Training and Model serving;
- SQLite compatibility/reference oracle <-> successor authority/projection comparisons.

When two candidates are claimed not to interact on a registered edge, the report records an explicit non-interaction rationale and tests the isolation boundary. An unexecuted case does not silently become non-applicable.

### Q4 - Multi-plane combination testing

Run combinations capable of exposing higher-order failures that individual or pairwise tests cannot reveal.

Required minimum combination classes:

- all-current-primary stack after every family-primary has a concrete implementation;
- one-challenger-at-a-time substitution;
- same-role comparison winners substituted into the full fabric;
- challenger substitutions across adjacent planes;
- multiple simultaneous challenger substitutions;
- mixed workstation, networked, distributed, remote, edge, cloud, and multi-region profiles where supported;
- degraded-mode combinations;
- backup/restore combinations;
- replay/rebuild combinations;
- deletion/revocation propagation across every active derivative plane.

When a full Cartesian product is impractical, use documented combinatorial coverage. The report records covered combinations, uncovered combinations, the selection algorithm, and the justification for any residual gap.

### Q5 - Full cognitive-memory end-to-end qualification

Exercise the complete flow:

```text
synthetic/authorized life event
        |
        v
Raw Buffer / capture
        |
        v
Experience/Event Fabric + Object/Archive
        |
        v
Formation Context Planner
        |
        v
Memory Formation Model
        |
        v
MemoryProposalBundle
        |
        v
deterministic Memory Gate / Authority Manager
        |
        v
Claim Fabric
        |
        v
Projection Manager
        |
        v
Graph + Vector + Episodes + Host/Relationship/Self/Mission projections
        |
        v
Retrieval Orchestrator
        |
        v
Context Manager / Memory Fusion
        |
        v
reasoning / action / outcome
        |
        v
Experience feedback and revision
```

Learned components interpret and propose. Deterministic policy still controls authority.

## 4. Complex synthetic Rayan-life workload

The candidate matrix uses the complex synthetic continuation of Rayan's life required by the Stage G architecture.

The workload includes:

- changing preferences;
- projects with dependencies and competing priorities;
- school and work transitions;
- people, relationships, and same-name collisions;
- long-running goals and missions;
- plans that succeed;
- plans that fail;
- decisions whose rationale changes later;
- direct owner statements;
- outside-source claims;
- observations;
- inferences;
- predictions;
- uncertain possibilities;
- contradictory documents;
- malicious or misleading documents;
- stored prompt-injection-like content;
- temporary/session-only facts;
- TTL and ephemeral state;
- corrections;
- supersessions;
- deletions;
- revocations;
- restore from backup after deletion;
- stale summaries;
- stale graph edges;
- stale vectors;
- duplicated evidence;
- reordered events;
- concurrent writes;
- retries;
- partial outages;
- malformed or partial events;
- false claims that "Rayan said X";
- Elaina/source-person vs Rayan/host vs A.L.I.C.E.-continuity separation;
- A.L.I.C.E./Friday product-isolation attacks.

The synthetic continuation spans enough simulated time to test historical vs current truth, recurring patterns, changed goals, relationship evolution, episode formation, correction lineage, deletion lineage, and long-horizon retrieval.

## 5. Scale and stress ladder

Stage G certification uses progressively larger workload points:

```text
1K
10K
100K
1M
```

These values are certification points rather than architectural ceilings. Larger tiers such as `10M`, `100M`, `1B+`, and other scales remain available when resources and research value justify them.

Stress testing continues until the relevant saturation, failure, or resource boundary is characterized well enough to compare candidates. Reports prioritize correctness, then record p50/p95/p99 latency where applicable, throughput, projection lag, replay/rebuild time, storage amplification, memory, CPU/GPU, network, and operational failure behavior.

## 6. Zero-tolerance failure classes

A Stage G candidate or combination cannot pass while any of these critical cases remains unresolved:

- deleted information served as current;
- revoked information influencing active output without disclosure;
- inference promoted to owner-stated fact;
- another person's memory assigned to Rayan;
- Rayan data rewritten as Elaina canon;
- A.L.I.C.E. continuity relabeled as Elaina history;
- external text treated as an owner statement;
- generated reconstruction presented as historical truth;
- graph, vector, workflow, summary, cache, or model state overriding Claim authority;
- correction lost in a projection;
- stale projection revived after rebuild or restore;
- invented provenance;
- cross-host or A.L.I.C.E./Friday private-data leakage;
- unauthorized model/dataset contamination;
- an aggregate score hiding a critical failure;
- backend-specific behavior silently changing the logical authority contract.

## 7. README and governance promise coverage

The Stage G acceptance report maps evidence to the promises made by the repository.

At minimum verify that A.L.I.C.E. can:

- preserve continuity across long-running work;
- connect projects, missions, dependencies, decisions, and outcomes;
- explain why a past decision was made;
- distinguish current truth from historical truth;
- distinguish direct statements, verified facts, outside claims, observations, inferences, predictions, reconstructions, disputes, outdated beliefs, corrections, and uncertainty;
- preserve provenance, time, confidence, supporting evidence, corrections, influence, active/deleted state;
- learn from outcomes without rewriting evidence;
- detect conflicts and support evidence-based disagreement;
- reconstruct the Experience Ledger chain from request through result, correction, and lesson;
- retrieve exact source evidence when exact wording matters;
- retain selective durable memory without treating all raw capture as permanent memory;
- propagate correction, deletion, and revocation through every derivative layer;
- restore safely without resurrecting deleted information;
- replace models or backends without losing A.L.I.C.E. continuity;
- preserve the Rayan-host / Elaina-source-person / A.L.I.C.E.-continuity boundary;
- preserve A.L.I.C.E./Friday isolation;
- remain inspectable, reversible, and honest about uncertainty and failure.

A promise with no executable or inspectable evidence remains unverified.

## 8. Stage G closure gate

Stage G cannot close until all of the following are true:

1. G2 formation and identity foundations pass their acceptance gates.
2. Every required concrete named candidate completes every applicable Q0-Q5 level.
3. Every implementation-family candidate is concretized and qualified or separately owner-ratified as a descriptive non-candidate.
4. Same-role all-pairs comparison coverage is complete.
5. Every registered cross-role interaction edge has complete candidate-substitution coverage.
6. Multi-plane combination coverage is documented and passes.
7. Full-fabric end-to-end runs pass on the complex synthetic Rayan-life workload.
8. Correction, deletion, revocation, restore, concurrency, failure, replay, rebuild, rollback, and product-isolation tests pass.
9. Scale and stress evidence is recorded for every applicable candidate.
10. README/governance promise coverage is complete.
11. No zero-tolerance failure remains unresolved.
12. The evidence package records exact versions, hashes, configurations, benchmark generations, failures, limitations, and unresolved research questions.
13. Rayan explicitly accepts the integrated Stage G result.

Stage G acceptance selects evidence-backed roles for later migration stages. It does not create a permanent technology ceiling.

## 9. Relationship to Phase 2 and later stages

Phase 2 remains the released authority, compatibility baseline, oracle, and fallback while Stage G is open.

Stage H remains bounded canary authority.

Stage I remains cutover.

Stage J remains compatibility/fallback transition.

Final Phase 2 replacement or retirement still requires Stage J acceptance and separate owner acceptance.
