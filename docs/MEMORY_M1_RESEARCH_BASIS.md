# Memory M1 Research Basis — Capability-First Polyglot Cognitive Fabric

**Version:** 1.0.0
**Prepared:** 2026-08-05
**Status:** Owner-ratified M1 research basis; not runtime implementation evidence
**Method:** Primary documentation and primary research are preferred. Product examples inform experiments; they do not dictate architecture.

## 1. Research questions

M1 must answer:

1. How should evidence, claims, graph relations, vector representations, workflow state, datasets, and model artifacts divide authority?
2. Which embedded and distributed backends best satisfy each role?
3. How can one logical identity survive backend replacement, replication, federation, and sharding?
4. How should graph algorithms influence claims without bypassing adjudication?
5. How should durable workflows coordinate curation, deletion, migration, training, and long missions?
6. How should adaptive context select among claims, sources, graphs, vectors, models, tools, agents, and simulations?
7. How can parametric learning remain lineage-bound, deletion-aware, measurable, and reversible?
8. How should A.L.I.C.E. preserve useful local continuity while using authorized clusters and remote compute?
9. How should product and owner namespaces prevent A.L.I.C.E. state from leaking into Friday or another host?
10. How should certification scale without turning checkpoints into a maximum?

## 2. Evidence and event fabrics

### KurrentDB

Primary documentation:

- https://docs.kurrent.io/getting-started/introduction
- https://docs.kurrent.io/server/v26.0/http-api/persistent
- https://docs.kurrent.io/server/v26.0/release-schedule/release-notes

Research focus:

- append-only event logs and streams;
- expected revisions and optimistic concurrency;
- stream and global positions;
- catch-up and persistent subscriptions;
- server-side checkpoints;
- replay, retention, tombstones, and deletion;
- clustering, failover, and operational recovery;
- outbox and projection patterns.

### Kafka and Pulsar

Primary documentation:

- https://kafka.apache.org/documentation/
- https://pulsar.apache.org/docs/

Research focus:

- partition and ordering semantics;
- consumer groups and offsets;
- retention, compaction, transactions, and replay;
- schema evolution;
- multi-region and tiered storage;
- event-store authority versus transport role.

## 3. Claim authority and temporal data

Primary sources and documentation:

- PostgreSQL documentation: https://www.postgresql.org/docs/
- FoundationDB documentation: https://apple.github.io/foundationdb/
- CockroachDB architecture: https://www.cockroachlabs.com/docs/stable/architecture/overview
- SQL:2011 temporal-model literature and bitemporal database research
- Datomic information model documentation: https://docs.datomic.com/

Research focus:

- append-only bitemporal versions;
- materialized current state;
- expected-version writes;
- distributed transactions and partition tolerance;
- serializable and causal consistency options;
- audit history and correction;
- deletion, encryption, export, and restore;
- embedded-to-distributed migration;
- owner-namespace partitioning.

## 4. Cognitive graph

### Neo4j and graph algorithms

Primary documentation:

- https://neo4j.com/docs/graph-data-science/current/
- https://neo4j.com/docs/graph-data-science/current/algorithms/
- https://neo4j.com/docs/graph-data-science/current/machine-learning/machine-learning/

Research focus:

- temporal and provenance-linked property graphs;
- centrality, community detection, similarity, pathfinding, DAG algorithms, node embeddings, and link prediction;
- graph projections and algorithm write-back;
- graph-native mission, causal, social, identity, and model-lineage reasoning;
- graph-to-claim reconciliation;
- graph failure and rebuild.

Additional candidates:

- FalkorDB: https://docs.falkordb.com/
- Memgraph: https://memgraph.com/docs
- Amazon Neptune: https://docs.aws.amazon.com/neptune/

## 5. Vector and multimodal retrieval

### Qdrant

Primary documentation:

- https://qdrant.tech/documentation/scaling/distributed_deployment/
- https://qdrant.tech/documentation/scaling/consistency-guarantees/
- https://qdrant.tech/documentation/scaling/resilience/

Research focus:

- distributed deployment;
- sharding and user-defined shard placement;
- replication and consistency;
- failure recovery;
- tenant and owner isolation;
- vector generation and deletion.

### Milvus

Primary documentation:

- https://milvus.io/docs/architecture_overview.md
- https://milvus.io/docs/scaleout.md

Research focus:

- separated storage and compute;
- distributed indexing and query;
- consistency and high availability;
- multimodal scale;
- deletion and compaction behavior.

Additional candidates:

- Vespa: https://docs.vespa.ai/
- Weaviate: https://docs.weaviate.io/
- pgvector: https://github.com/pgvector/pgvector
- FAISS: https://faiss.ai/
- DiskANN primary paper and implementation.

## 6. Durable workflows

### Temporal

Primary documentation:

- https://docs.temporal.io/
- https://docs.temporal.io/workflows
- https://docs.temporal.io/activities
- https://docs.temporal.io/workflow-execution

Research focus:

- durable workflow history;
- deterministic replay;
- idempotent activities;
- retries, timeouts, heartbeats, signals, updates, and cancellation;
- worker crash recovery;
- code-version and migration semantics;
- long-running Curator, deletion, migration, training, and mission workflows.

Additional candidates:

- Dagster: https://docs.dagster.io/
- Prefect: https://docs.prefect.io/
- Ray Workflows and task systems: https://docs.ray.io/

## 7. Distributed training and inference

### Ray

Primary documentation:

- https://docs.ray.io/en/latest/train/train.html
- https://docs.ray.io/en/latest/ray-core/walkthrough.html

Research focus:

- distributed data and training;
- accelerator scheduling;
- failure recovery and checkpoints;
- hyperparameter and evaluation workloads;
- cluster autoscaling;
- owner-authorized local and remote compute.

### vLLM

Primary documentation:

- https://docs.vllm.ai/en/latest/serving/distributed_serving.html
- https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

Research focus:

- tensor and pipeline parallel serving;
- multi-node inference;
- model routing and throughput;
- context and KV-cache behavior;
- traceable model serving and rollback.

Additional candidates:

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- FSDP: https://pytorch.org/docs/stable/fsdp.html
- DeepSpeed: https://www.deepspeed.ai/
- Slurm: https://slurm.schedmd.com/documentation.html
- Kubernetes: https://kubernetes.io/docs/

## 8. Synchronization, federation, and CRDTs

Primary research:

- Shapiro et al., “A Comprehensive Study of Convergent and Commutative Replicated Data Types.”
- Kleppmann and Beresford, “A Conflict-Free Replicated JSON Datatype.”
- Lamport, “Time, Clocks, and the Ordering of Events in a Distributed System.”
- vector-clock, dotted-version-vector, and causal-consistency literature.

Research focus:

- owner-namespace federation;
- multi-device offline operation;
- causal order and late events;
- claim conflict rather than silent last-write-wins;
- deletion watermarks;
- import/export identity;
- partition and reconciliation receipts.

## 9. Machine unlearning and model editing

Primary research program:

- exact and approximate machine-unlearning literature;
- SISA training;
- influence functions;
- retraining and shard retirement;
- model editing methods;
- deletion auditing and residual-influence measurement;
- privacy attacks against memorized training data.

Research focus:

- when deletion requires rebuild, retraining, model retirement, or disclosure;
- model and dataset lineage;
- unlearning evaluation;
- rollback and contamination;
- source-person and owner-model sensitivity.

No method is treated as universally sufficient. Deletion limits remain truthful.

## 10. Multimodal and embodied memory

Primary research program:

- multimodal representation and retrieval;
- temporal event segmentation;
- video and audio memory;
- sensor fusion;
- embodied world models;
- robotics experience replay;
- continual learning under distribution shift.

Research focus:

- provenance across modalities;
- temporal alignment;
- raw versus summarized evidence;
- graph and vector fusion;
- large object storage and lifecycle;
- safety and rollback for embodied skills.

## 11. Competing personal-memory systems

Systems to evaluate through source code, documentation, and reproducible tests:

- Graphiti;
- Letta;
- Mem0;
- LangGraph;
- Hindsight;
- LightMem;
- MemOS;
- Cognee;
- Zep.

Evaluation dimensions:

- truth and authority semantics;
- provenance;
- temporal validity;
- correction and deletion;
- graph and vector architecture;
- context construction;
- operational durability;
- scale and latency;
- product and host isolation;
- model and dataset lineage;
- backend replacement;
- inspectability and rollback.

## 12. Experiment program

### Authority experiments

- append and expected-version conflicts;
- bitemporal correction;
- current-state projection;
- event-to-claim lineage;
- cross-backend outbox and reconciliation;
- authority cutover and reverse cutover.

### Graph experiments

- mission dependency and causal path retrieval;
- social and relationship temporal edges;
- community and similarity algorithms;
- graph embedding quality;
- graph proposal to claim adjudication.

### Vector experiments

- hybrid lexical, vector, graph, and source retrieval;
- multi-generation calibration;
- distributed sharding and replication;
- deletion and rebuild;
- multimodal retrieval.

### Workflow experiments

- worker crash and replay;
- duplicate activity delivery;
- long-running deletion and migration;
- signals, cancellation, and version migration;
- cluster and network failure.

### Training experiments

- dataset lineage;
- challenger training;
- shadow inference;
- canary influence;
- contamination and deletion tests;
- model retirement and rollback.

### Scale experiments

Certification begins at 1K, 10K, 100K, and 1M, then extends through 10M, 100M, 1B, and later profile-defined workloads. These are progressive evidence points.

## 13. Selection principle

A technology is selected when measured evidence shows that it improves the relevant authority, capability, reliability, scale, continuity, cost, privacy, or intelligence objective.

Operational complexity is an engineering cost. It is not an automatic reason to cap A.L.I.C.E.

Vendor lock-in, opaque authority, weak export, poor deletion, uninspectable state, and missing rollback count against a candidate. They do not establish a categorical technology ban.

## 14. Current truth

This research basis identifies candidates and experiments. It does not assert that any listed system is selected, integrated, or production-enabled.
