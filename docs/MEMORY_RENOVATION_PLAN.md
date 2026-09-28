# Memory Renovation Plan — Parallel Capability Tracks

**Version:** 1.3.0
**Status:** Owner-ratified M1 capability-track plan; implementation activation remains separate
**Architecture:** Memory v4.1 Capability-First Polyglot Cognitive Fabric

## 1. Objective

Renovate A.L.I.C.E. memory into a claim-centered, evidence-linked, graph-capable, distributed, multimodal, continually learning cognitive fabric without sacrificing truth, custody, deletion, or recovery.

The plan uses parallel research and independently activated production profiles. A small first implementation may be useful, but it does not define the destination.

## 2. Execution selection

The destination remains **Memory v4.x Capability-First Polyglot Cognitive Fabric**. The current implementation is selected from the full cognitive requirements rather than from packaging simplicity.

Current path:

- KurrentDB Experience/Event Fabric with NATS JetStream federation/ingress behind first-party event/evidence contracts;
- XTDB v2 bitemporal Claim Authority;
- encrypted content-addressed object storage;
- dynamic Episodes/Autobiographical plane;
- hardware-adaptive Cognitive Multi-Graph: LadybugDB host-local, NebulaGraph distributed-scale placement, existing Neo4j reference retained;
- engine-independent associative graph compute;
- Qdrant dense/sparse/multivector/multimodal retrieval;
- exact/source-native/live retrieval;
- protected perceptual personal memory;
- governed host/source-person/relationship/self/world/mission/skill projections;
- parametric personal memory with influence lineage;
- Cognitive Workspace/activation-state provenance;
- Memory Resource Manager and Retrieval Orchestrator;
- Lifecycle Curator and cross-layer deletion/unlearning coordinator;
- Temporal durable workflows;
- process-local L1 plus Valkey shared ephemeral state;
- content-addressed model/dataset registry;
- PyTorch + Accelerate hardware-adaptive training.

This selection does not authorize a reduced desktop architecture. One machine and a private cluster instantiate the same logical planes with different placement.

Broad infrastructure tournaments remain prohibited. Challengers are opened only when they can change a concrete implementation decision.

## 2.1 Common entry requirements

Each track defines:

- logical contracts and authority role;
- registered candidate backends;
- synthetic and owner-authorized evaluations;
- provenance and truth-state semantics;
- privacy and product-isolation controls;
- deletion behavior;
- rollback or retirement;
- performance profiles;
- public-claim state;
- production activation evidence.

## 3. Track A — Authority and Claim Fabric

Deliver:

- Store and Capability Fabric Registry;
- Evidence Event and stream position;
- Claim Identity and Claim Version;
- current adjudicated projection;
- Evidence Relation;
- authority, conflict, and adjudication;
- bitemporal and causal semantics;
- owner namespace and federation identity;
- deletion and rollback receipts.

Use XTDB v2 as the selected bitemporal Claim implementation. Treat its storage/log deployment as physical topology, not a reduction in Claim semantics.

## 4. Track B — Event and Experience Fabric

Build the selected NATS JetStream-backed Experience/Event implementation with project-owned semantics for ordered identity, expected stream version, idempotent append, replay/checkpoints, durable consumption, device/causal metadata, correction/deletion lineage, integrity, archive metadata, and projection interfaces.

KurrentDB remains a strong event-native reference/challenger. It is not the universal Fable dependency because the current KLv1 product/license boundary is less suitable for a user-owned distributable platform than NATS's Apache-2.0 base.

## 5. Track C — Cognitive Multi-Graph and associative compute

Build a single logical multi-relational graph projection with orthogonal views for:

- semantic/concept;
- entity;
- temporal;
- causal;
- evidence/provenance;
- social/relationship;
- mission/project;
- goals/dependencies;
- skills/procedures;
- source trust;
- identity/person;
- world/social/causal model;
- model/dataset lineage;
- decisions/outcomes.

Use LadybugDB for the current host-local embedded placement. Use NebulaGraph with a qualified distributed backend when the graph must scale across a private cluster. Preserve Neo4j as an existing A.L.I.C.E. reference/migration source rather than deleting useful evidence.

Build an engine-independent associative compute layer for Personalized PageRank, spreading activation, temporal decay, lateral inhibition, path/bridge discovery and query-conditioned view traversal.

Graph/activation outputs remain rebuildable retrieval projections and never Claim authority.

## 6. Track D — Vector, multimodal, perceptual, and source-native retrieval

Use Qdrant as the full semantic/multimodal plane, with Edge/server/cluster placement chosen by hardware.

Support dense, sparse, named and multivector/late-interaction representations across text, code, image, audio, video, sensor and scientific data.

Maintain grounded perceptual personal memory for authorized people/voice/object/place identity so captions do not become a lossy substitute for native perception.

Retain exact/source-native and live-source retrieval as a parallel first-class plane. The model may progressively search files, records, structured fields, metadata, APIs and source systems with iterative query reformulation/direct reads.

The vector plane is part of the full architecture even when a particular query bypasses it. Source-native retrieval complements rather than replaces semantic/multimodal retrieval.

## 7. Track E — Durable Curation and Mission Workflows

Use Temporal as the selected durable workflow plane across both host-local and distributed deployments.

Run candidate extraction, consolidation, adjudication, projection refresh, deletion/unlearning, execution-state replay, migration, repair, training, evaluation, model promotion/rollback, federation, and long missions with idempotency, retries, signals, cancellation and recovery.

A single host may self-host the workflow service. That placement does not remove the durable-workflow capability.

## 8. Track F — Memory Resource Management, Cognitive Recollection and Serving

Build:

- Memory Resource Manager/Scheduler across external, episodic, graph, vector, procedural, parametric and activation memory;
- fast/slow memory-formation scheduling;
- dynamic episode/scene/domain routing;

- Context Planner;
- Retrieval Trace;
- invocation-scoped evidence-consumption/read receipts and citation-lock validation;
- source and evidence expansion;
- graph/vector/claim fusion;
- uncertainty and contradiction preservation;
- evidence-aware retention that can preserve old task-critical context instead of relying on recency alone;
- versioned context-construction policies/programs as sandboxed champion/challenger artifacts;
- usage-aware retrieval/accessibility projections learned from governed retrieval and outcome receipts;
- compact through very-large-context plans;
- multi-agent and simulation plans;
- local and distributed inference;
- low-VRAM streamed/offloaded inference challengers, including layer-wise and sparse-expert streaming, evaluated on exact model/hardware/storage profiles for quality, first-token latency, throughput, disk/network I/O, host RAM, storage footprint, energy, concurrency, and failure recovery;
- stale-index, no-index, and no-memory fallbacks.

## 9. Track G — Owner, Source, Relationship, Self, and World Models

Create versioned, evidence-linked projections for preferences, values, traits, goals, temporary state, source-person hypotheses, relationship state, self-model, world/social/causal models, and predictions.

Unknowns remain uncertain. Generated reconstruction is not source history.

## 10. Track H — Procedural and Parametric Learning

Build:

- reusable skills and failure cases;
- evidence-grounded failure localization across skill, context/harness, retrieval, model, tool/executor, environment, evaluator, and unknown causes before mutation;
- separately versioned procedural-skill and context/harness challengers, including skill-only, harness-only, joint, and parametric comparisons;
- dataset and replay manifests;
- learned retrieval, routing, ranking, accessibility, and preference models;
- retrieval-experience learning that may amortize reranking or learn usage-aware associations without becoming claim authority;
- adapters and LoRA;
- challenger models and continual-learning experiments;
- shadow and canary serving;
- machine-unlearning and model-editing research;
- later deeper weight updates.

External artifact evolution and parametric learning are complementary substrates. No frozen-model assumption is a permanent architecture rule. Optimizer summaries, textual gradients, and failure diagnoses remain derived proposals linked to the underlying rollout evidence and cannot become evidence or Claim authority by themselves.

Production influence follows evaluation and authority profiles. Research begins when lineage and containment are sufficient.

## 11. Track I — Distributed Deployment and Synchronization

Build profiles for edge, mobile, workstation, multi-GPU, home cluster, private cluster, hybrid cloud, distributed global, and frontier research.

Implement replication, sharding, synchronization, causal metadata, conflict resolution, owner-namespace federation, failover, export, and replacement.

## 12. Track J — Inspection, Evaluation, Deletion, Unlearning, and Rollback

Provide one inspection and control surface across claims, events, episodes, graphs, vectors, objects, source-native evidence, active contexts/plans, workflows, personal models, datasets, replicas and archives.

Deletion propagation spans durable stores **and execution/parametric influence**. Track summaries, context, pending plans, caches/KV generations where controllable, replay/training examples, adapters/weights and backup/export generations.

Test counterfactual execution-state replay where required, parameter-memory backflow prevention, restore filtering, cutover, rollback, model retirement, projection rebuild and public-claim accuracy.

## 13. Integration waves

### Wave 1 — contracts and registry

Ratify M1 and implement the first neutral authority, lineage, and capability-discovery contracts. Public contract artifacts may carry metadata and content references rather than private payload bytes. That artifact-local representation does not define the memory-data, runtime, research, prototype, or destination boundary.

### Wave 2 — serving and shadow planes

Build current-state serving, batch hydration, traces, and shadow graph/vector/event/workflow candidates.

### Wave 3 — evidence-to-candidate and adjudication

Connect Experience Fabric to candidate generation, rejection, conflict, and Claim Authority.

### Wave 4 — cognitive projections and learning

Activate episodes, graph reasoning, owner/source/self models, skills, datasets, and challenger learning by profile.

### Wave 5 — federation and scale

Expand multi-device, cluster, remote, and distributed profiles through measured certification.

Waves describe integration maturity. Track research and reversible prototypes may proceed in parallel. No later wave is a blanket prerequisite for another track's research or prototype work.

## 13.1 Contract-artifact boundary and parallel execution

Wave 1 defines neutral authority, lineage, and capability-discovery interfaces. A public contract artifact may carry metadata and content references instead of private payload bytes. That representation is local to the artifact. It is not a restriction on memory content, persistence, runtime implementation, research, prototypes, candidate formation, adjudication, cognitive projections, learning, training, or the destination architecture.

The implementation program therefore has two concurrent lanes:

- **contract lane:** complete the neutral contracts needed for authority, interoperability, deletion, rollback, and later cutover;
- **full-memory lane:** research and build reversible prototypes for Claim Authority persistence, event-to-candidate flow, adjudication, current-state serving, graph/vector projections, automated selective memory, cognitive models, learning, deletion propagation, and rollback.

A later integration wave is not a prerequisite for beginning another track's research or reversible prototype. Dependencies govern production influence, canonical authority, private-data use, and irreversible cutover.

## 14. Exit evidence for each production profile

- exact contracts and backend registrations;
- correctness, authority, performance, and adversarial evaluation;
- custody and product-isolation review;
- deletion and rollback rehearsal;
- degraded mode and recovery;
- public-claim language;
- exact commit and artifact hashes;
- owner or mission authority appropriate to consequence.

## 15. Current state

This owner-ratified plan does not assert runtime implementation, production activation, Claim Authority deployment, or Phase 2 migration start.
