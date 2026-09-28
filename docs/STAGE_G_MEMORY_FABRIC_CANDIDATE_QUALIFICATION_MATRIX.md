# Stage G Full Cognitive-Memory Fabric Qualification

**Version:** 3.0.1  
**Status:** owner-directed execution requirement  
**Applies to:** the full A.L.I.C.E./Fable successor cognitive-memory fabric and Phase 2 replacement readiness

## 1. What Stage G qualifies

Stage G qualifies the **complete personal cognitive-memory architecture**.

It does not qualify a lightweight deployment subset and it does not run an infrastructure tournament.

The architecture is fixed by capability requirements first. Physical engines are selected from current evidence. A challenger is opened only when an observed capability, scale, correctness, reliability, privacy, deletion, distribution, licensing, or cost problem can change the decision.

## 2. No-capability-reduction rule

Stage G cannot pass by removing or narrowing:

- Experience/Event history;
- bitemporal Claim authority;
- raw evidence/object retention;
- episodic/autobiographical memory;
- graph/relational memory;
- associative graph retrieval;
- vector/multimodal retrieval;
- source-native retrieval;
- host/source-person/relationship/self/mission state;
- procedural memory;
- parametric personal memory;
- working/activation memory;
- deletion/unlearning;
- durable workflows;
- multi-device/federation semantics.

A physical backend can change. A logical capability cannot silently disappear because another backend is easier to package.

## 3. Selected current implementation architecture

| Logical plane | Current selected implementation | Role |
| --- | --- | --- |
| Raw Evidence/Object | encrypted content-addressed object store behind S3-compatible abstraction | originals, large payloads, datasets, artifacts, backups |
| Experience/Event | KurrentDB canonical event store behind project-owned EvidenceLog/Event contracts; NATS JetStream federation/edge ingress | canonical append/replay/subscriptions plus device/federation transport lineage |
| Claim Authority | XTDB v2 | immutable/bitemporal adjudicated claims and historical/current views |
| Episodic/Autobiographical | governed episode store/projection rooted in Event + Claim identities | learned event boundaries, narratives, outcomes, scenes |
| Cognitive Multi-Graph — local | LadybugDB | embedded host-local graph projection/traversal/analytics |
| Cognitive Multi-Graph — scale-out | NebulaGraph backend | same graph contract when host/private-cluster scale requires distribution |
| Existing graph reference | Neo4j | retained A.L.I.C.E. reference/projection and migration evidence |
| Associative graph compute | engine-independent graph-compute service | PPR/spreading activation/temporal decay/inhibition/path relevance |
| Vector/Multimodal | Qdrant / Qdrant Edge or server/cluster placement | dense/sparse/named/multivector retrieval |
| Source-native/live | filesystem, SQL/FTS, structured APIs and live systems | exact/current/source-owned evidence |
| Working/ephemeral | process-local L1 + Valkey shared state | workspace/cache only |
| Durable workflow | Temporal | migration/projection/deletion/training/mission/recovery workflows |
| Model/data registry | content-addressed signed/hashed lineage registry | datasets/models/checkpoints/evaluations/deletion influence |
| Training | PyTorch + Accelerate with DDP/FSDP/offload as measured need dictates | personal-model construction and evolution |

The table is a current implementation choice, not a capability ceiling. One host may run embedded forms; a private cluster may run distributed forms. Both implement the same logical architecture.

## 4. Architectural memory behavior required by Stage G

### 4.1 Fast path

An incoming authorized experience can immediately:

- enter the event fabric;
- bind exact source/provenance;
- receive subject/time/device/custody metadata;
- receive safe exact/lexical/vector/multimodal keys;
- participate in temporal ordering;
- remain available for exact reconstruction.

The fast path must not invent authority.

### 4.2 Slow path

Asynchronous formation/consolidation may produce proposals for:

- claims;
- episode boundaries/narratives;
- graph relations;
- semantic/trait/scene projections;
- host/relationship/self/world/mission state;
- procedural lessons;
- retention/importance;
- training candidates.

Slow processing still flows through deterministic authority where authority is required.

### 4.3 Dynamic scene/domain structure

Memories may belong to overlapping dynamically learned contexts:

- project;
- school/work;
- relationship;
- mission;
- role;
- place;
- environment;
- interest;
- time period;
- recurring situation.

No fixed Life/Work/Interest or other authored taxonomy defines the user's lifetime.

### 4.4 Cognitive multi-graph views

Graph retrieval supports independent but linked views:

- semantic;
- temporal;
- causal;
- entity;
- evidence/provenance;
- social/relationship;
- mission/project;
- goal/dependency;
- procedure/skill;
- source trust;
- identity/person;
- world model;
- model/data lineage;
- decision/outcome.

Query intent selects/fuses views. Graph scores never grant factual authority.

### 4.5 Associative recollection

The graph-compute layer may use:

- Personalized PageRank;
- spreading activation;
- lateral inhibition;
- temporal decay;
- path/bridge discovery;
- usage-derived accessibility;
- community/locality signals.

These are retrieval accessibility mechanisms only.

### 4.6 Multimodal personal memory

The fabric preserves native evidence and learned representations for text, image, audio, video, code, documents, sensor/scientific data, people/voice/object/place identity and future modalities.

Captions are not sufficient substitutes for perceptual identity evidence.

### 4.7 Memory Resource Manager

The system explicitly schedules and tracks:

- plaintext/external memory;
- episodes;
- graph/vector projections;
- procedural memory;
- personal-model/parametric memory;
- activation/working memory;
- hot/warm/cold/archive state;
- regeneration vs persistence;
- model/context budgets.

Scheduling never changes authority.

### 4.8 Cognitive recollection

The Retrieval Orchestrator can plan parallel/iterative retrieval over all applicable planes, reconcile candidates against authority/provenance, record what evidence was actually consumed, calibrate influence, assess sufficiency, and recollect when necessary.

There is no universal vector-first, graph-first or semantic-fallback order.

## 5. Qualification levels

### Q0 — exact registration

Record every selected component's:

- exact version/digest;
- config;
- schema generation;
- data/projection generation;
- custody/encryption domain;
- runtime environment;
- rebuild/restore path;
- rollback path.

### Q1 — per-plane contract tests

Test the selected implementation against the logical contract it owns.

This includes:

- Event append/order/idempotency/replay/causal/device lineage;
- Claim bitemporality/conflict/correction/deletion;
- object integrity/source reconstruction;
- episode lineage/rebuild;
- graph view generation/rebuild;
- associative retrieval determinism/bounds;
- vector generation/rebuild/deletion;
- source-native exact/live retrieval;
- workflow durability/idempotency/recovery;
- cache loss without authority loss;
- model/data lineage.

Q1 is not a same-role backend comparison.

### Q2 — cross-plane invariants

Test failure-prone boundaries:

- Event -> MFM;
- MFM -> Memory Gate;
- Gate -> Claim;
- Claim/Event -> Episodes;
- Claim/Event -> graph/vector/personal state;
- correction/deletion -> every derivative;
- source/native evidence -> citation/read receipts;
- retrieval -> Context Planner;
- host/source-person/relationship/self -> EIPM;
- outcome -> Experience/procedural/personal revision;
- model/data lineage -> promotion/rollback;
- device reconciliation -> authority;
- workflow -> idempotent external actions.

### Q3 — full personal loop

Exercise:

```text
experience
 -> Event Fabric
 -> Formation Context Planner
 -> MFM fast/slow formation
 -> deterministic Memory Gate
 -> Claim Authority
 -> Episodes / Multi-Graph / Vector / personal-state / procedural projections
 -> Memory Resource Manager
 -> Retrieval Orchestrator / Cognitive Recollection
 -> EIPM native judgment
 -> response/action
 -> outcome
 -> Experience + governed personal/model revision
```

Stage G closes on coherent end-to-end behavior.

## 6. Required frontier-derived invariants

### 6.1 Evidence-consumption lock

Final factual support must resolve to authoritative/source evidence actually consumed by the producing invocation. Retrieval scores, summaries, graph reachability and unopened references are not themselves source evidence.

### 6.2 Consolidation path dependence

Reordering/grouping the same evidence must not silently create different authoritative truth. Test bounded schedules to detect consolidation instability while permitting raw retention/no-consolidation.

### 6.3 Memory-use calibration

Retrieved memory must have the right influence:

- decisive evidence controls when it should;
- bounded evidence remains bounded;
- irrelevant memory is ignored;
- uncertainty/conflict remains visible.

### 6.4 Retrieval strategy adequacy

Representative workloads cover exact, source-native, semantic, graph, temporal, causal, episodic, procedural, multimodal, relationship, mission and live-source cases.

The test asks whether the **planner chooses/combines a capable route**, not whether every route wins every benchmark.

### 6.5 Procedural memory boundary

Past workflow/tool lessons may guide retrieval/action but cannot become factual evidence for the current world.

### 6.6 Execution-state deletion

A deletion/revocation test covers:

- durable stores;
- summaries;
- active context;
- pending plans;
- caches/KV generations where applicable;
- projections;
- replay/training data;
- models/adapters influenced by the target.

When exact counterfactual execution-state removal is required, restore/replay from the clean provenance boundary.

### 6.7 Parameter-memory backflow

After deletion, surviving external memory must not re-teach a scrubbed model, and surviving parametric influence must not regenerate deleted external memory.

### 6.8 Identity separation

No workload may collapse:

- Elaina source-person history;
- Rayan host state;
- A.L.I.C.E. self;
- A.L.I.C.E.-Rayan relationship;
- mission/world state.

Fable uses the host/Fable-self separation without inventing an A.L.I.C.E.-style source-person axis when none exists.

### 6.9 Personal-development causality

Isolated changes to host, relationship and self state must change only relevant later judgments.

## 7. Scale program

Stage G does not have a maximum scale.

Convenient checkpoints may include:

- 1K;
- 10K;
- 100K;
- 1M;
- 10M;
- 100M;
- 1B records/elements where the relevant plane and hardware make the tier meaningful.

Continue beyond any checkpoint when required to expose the actual scaling boundary.

Performance evidence includes:

- p50/p95/p99 latency;
- throughput;
- memory/storage;
- rebuild/restore time;
- projection lag;
- write amplification;
- retrieval quality;
- cross-device sync;
- cost;
- failure recovery.

## 8. Multi-device/federation qualification

Test:

- causal ordering;
- offline device continuation;
- reconnect/reconciliation;
- conflict receipts;
- device loss/revocation;
- key/custody boundaries;
- projection generation mismatch;
- duplicate/reordered events;
- private-host isolation.

Multi-device continuity is part of the architecture, not a future capability omission.

## 9. Challenger rule

A challenger is admitted only when it can resolve a concrete open decision, such as:

- current implementation cannot satisfy its logical contract;
- measured quality/fidelity gap;
- scale/latency/resource blocker;
- recovery/deletion/privacy/custody blocker;
- licensing/distribution blocker;
- frontier evidence exposes a missing mechanism.

Compare only the necessary surfaces and stop when the decision is resolved.

No all-pairs backend tournament. No Cartesian substitutions. No technology benchmark whose result cannot change implementation.

## 10. Zero-tolerance failures

Stage G cannot pass with unresolved:

- deleted/revoked information served or acted on as current;
- inference promoted to direct fact;
- subject/person misassignment;
- source-person/host/self history collapse;
- projection overriding Claim authority;
- correction lost in a derivative;
- stale projection revived after rebuild/restore;
- invented provenance;
- unconsumed evidence cited as support;
- retrieval score treated as truth;
- popularity/usage feedback becoming factual authority;
- procedural lesson presented as current fact;
- external memory/model recontamination after deletion;
- cross-host/product leakage;
- unauthorized model/dataset contamination;
- critical identity/provenance failures hidden in aggregate scores;
- backend-specific semantic drift.

## 11. Stage G closure

Stage G is ready for Stage H only when:

1. every logical plane required by the full architecture is operational;
2. selected implementations satisfy their contracts;
3. MFM/Gate/Claim/projection/retrieval/EIPM/outcome loop passes;
4. fast/slow formation passes;
5. dynamic scene/domain and episode behavior passes;
6. multi-graph/associative/vector/source-native retrieval passes;
7. multimodal/perceptual memory passes at the qualified modalities;
8. correction/deletion/unlearning passes across durable, execution and parametric influence;
9. personal-development causal tests pass;
10. multi-device/federation semantics pass;
11. realistic scale/resource evidence exists with no artificial certification ceiling;
12. rebuild/restore/rollback pass;
13. no zero-tolerance failure remains;
14. exact versions/configs/hashes/receipts are recorded;
15. Rayan accepts the result.

Stage H is canary authority. Stage I is cutover. Stage J completes final Phase 2 replacement/retirement.
