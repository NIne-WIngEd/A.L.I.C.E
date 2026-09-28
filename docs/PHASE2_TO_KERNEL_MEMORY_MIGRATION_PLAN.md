# Phase 2 to Cognitive Fabric Memory Migration Plan

**Version:** 2.0.1
**Status:** Stage A+B, Stage C+E, deterministic Stage D, Stage F controlled-mirroring, and Stage G projection-generation prototypes operational; persistent F+G reference integration operational; private execution, live candidate integrations, and production authority stages independently gated
**Source baseline:** Released Phase 2 Memory Core
**Destination:** Backend-neutral Memory Architecture v4.1

## 1. Objective

Migrate Phase 2 memory into a claim-centered, evidence-linked, polyglot cognitive fabric without creating an ambiguous source of truth, losing provenance, leaking private state, or removing a tested fallback before successor evidence exists.

The target may combine embedded, relational, distributed, event, graph, vector, object, workflow, and model systems. The plan does not assume one database or one host.

## 2. Migration principles

1. Released Phase 2 source and tests remain a compatibility baseline and test oracle.
2. Every target authority is registered before accepting production writes.
3. One component is canonical for each authority type at a given generation.
4. Secondary writes are projections, replicas, outbox deliveries, or shadow authorities with explicit status.
5. Dual-write periods require idempotency, reconciliation, divergence metrics, and rollback.
6. Private payloads and owner state remain within authorized custody.
7. Corrections and deletion lineage migrate before production cutover.
8. Graph, vector, summary, cache, model, and episode systems remain provenance-linked derivatives unless explicitly ratified otherwise.
9. Research and prototypes may proceed in parallel. Cutover proceeds by evidence.

## 3. Target planes

The migration may populate:

- Experience/Event Fabric;
- Claim Authority and current-state projection;
- Cognitive Graph;
- lexical, vector, and multimodal indexes;
- object and archive storage;
- durable workflow runtime;
- dataset and model registries;
- owner-authorized replicas and multi-device synchronization;
- inspection, deletion, and rollback services.

A single-node edge profile and a distributed profile implement the same logical contracts.

## 4. Stages

### Stage A — Inventory and registration

Register every source store, schema, generation, authority role, encryption domain, data classification, record count, integrity state, and deletion capability.

No source file becomes authoritative merely because it is discovered.

### Stage B — Contract adapters

Implement read adapters that translate Phase 2 records into neutral Evidence Event, Claim Identity, Claim Version, Evidence Relation, Conflict, Correction, and Deletion records.

Adapters record loss, ambiguity, and unsupported semantics.

### Stage C — Full successor implementation

Build the full successor behind the neutral event, claim, episode, graph, associative-retrieval, vector/multimodal, source-native, object, workflow, personal-model, registry, federation, and deletion/unlearning contracts.

The current physical architecture is:

- KurrentDB as the canonical replayable Experience/Event store behind project-owned EvidenceLog/event semantics, with NATS JetStream for durable federation/edge ingress;
- XTDB v2 for bitemporal Claim Authority;
- encrypted content-addressed object storage for raw evidence and artifacts;
- LadybugDB for host-local Cognitive Multi-Graph placement and NebulaGraph on qualified distributed storage for scale-out private-cluster placement;
- Neo4j retained as an existing A.L.I.C.E. graph reference/projection during migration;
- engine-independent associative graph compute;
- Qdrant Edge/server/cluster for vector and multimodal projections;
- direct source-native/live retrieval;
- Temporal for durable workflows;
- process-local L1 plus Valkey for shared ephemeral workspace;
- content-addressed model/dataset lineage and hardware-adaptive PyTorch/Accelerate training.

Embedded and distributed placements implement the same logical planes. A single-host installation is not a reduced-capability class.

This selection is replaceable but not tentative by default. A challenger is opened only when a concrete capability, scale, correctness, recovery, privacy, deletion, distribution, licensing, or cost problem can change the implementation decision.

### Stage D — Historical backfill

Backfill in deterministic batches with:

- source checkpoint;
- record digest;
- mapping version;
- idempotency key;
- accepted, rejected, quarantined, and ambiguous counts;
- evidence and deletion lineage;
- reconciliation receipt.

Backfill never invents missing provenance.

### Stage E — Shadow reads

Run current Phase 2 reads and successor reads against the same synthetic and authorized workloads.

Compare result quality, authority correctness, conflict handling, latency, staleness, deletion, privacy, and explanation.

### Stage F — Controlled write mirroring

Keep one canonical writer. Publish canonical changes through outbox records to successor projections or a shadow authority.

If a successor is selected as canonical during canary, reverse projection to Phase 2 may continue for rollback. The authority transition is explicit.

### Stage G — Full cognitive-memory fabric build

Construct and integrate the complete successor fabric: Event, Claim, episodes, Cognitive Multi-Graph, associative activation, vector/multimodal, source-native retrieval, personal cognitive models, parametric/activation memory governance, Memory Resource Manager, Retrieval Orchestrator, lifecycle curation, cross-layer deletion/unlearning, durable workflows, and multi-device/federation state.

Backend durability alone is not sufficient Stage G exit evidence. Before Stage G can close, the successor must pass owner-authorized and synthetic qualification covering learned Memory Formation, the Elaina Identity / Personality Model, Rayan host learning without identity drift, relationship/self development, deterministic authority, fast/slow formation, per-layer behavior, cross-layer consistency, cognitive recollection, conflict/temporal correction, multimodal memory, execution-state and parametric deletion influence, A.L.I.C.E./Fable isolation, failure/recovery, concurrency, rebuild, rollback, multi-device continuity, realistic scale, and README/governance promise alignment. Stage H is not eligible until this integrated Stage G qualification is accepted.

#### Stage G integrated qualification matrix

The full-fabric qualification contract is maintained in `docs/STAGE_G_MEMORY_FABRIC_CANDIDATE_QUALIFICATION_MATRIX.md`.

Stage G closes by proving the **complete selected personal cognitive-memory fabric**, not by exhaustively testing every known backend and not by dropping planes to simplify deployment. Required evidence covers all logical planes, their selected implementations, the full MFM -> authority -> projection -> recollection -> EIPM -> outcome -> revision loop, correction/deletion/revocation across durable/execution/parametric influence, identity/provenance boundaries, A.L.I.C.E./Fable isolation, citation lock, consolidation stability, memory-use calibration, multimodal retrieval, multi-device continuity, realistic scale/latency, and README/governance promise alignment.

Alternative backends are evaluated only when a concrete trigger creates an unresolved implementation decision. Same-role all-pairs, all candidate substitutions, and broad combinatorial backend coverage remain explicitly unnecessary.

Stage H remains ineligible until this integrated full-fabric qualification is accepted. Physical implementations remain replaceable and create no technology ceiling.

### Stage H — Canary authority

Enable a bounded owner-authorized profile for selected namespaces, claim classes, or missions.

Canary evidence includes divergence, rollback rehearsal, recovery, failure injection, privacy, and deletion verification.

### Stage I — Cutover

Freeze the old canonical position, drain outboxes, reconcile, verify integrity, record cutover manifest, activate the new authority generation, and preserve a tested rollback window.

### Stage J — Compatibility operation

Phase 2 becomes a read-only compatibility projection, fallback, export source, or retired archive according to profile.

Stage J must itself be evaluated and owner-accepted. For this migration, final replacement or retirement is complete only after Stage J acceptance; neither Stage H canary authority nor Stage I cutover alone completes final Phase 2 replacement.

It is not deleted until retention, rollback, Stage J acceptance, and owner authority permit.

## 5. Distributed and multi-device semantics

Migration records authority namespace, owner partition, shard or stream position, logical clock, causal order, device/cluster identity, replication conflict, and reconciliation.

Partition strategy cannot leak owner or product state. Cross-owner federation requires an explicit export rather than implicit shared storage.

## 6. Deletion migration

Before cutover:

- import all active deletion requests and tombstones;
- verify ordinary retrieval exclusion;
- propagate to graph, vector, object, cache, dataset, replay, model, replica, and backup manifests;
- test archive restore with deletion replay;
- disclose model-influence limitations;
- rehearse rebuild or retirement of noncompliant derivatives.

## 7. Rollback

Rollback restores the prior authority generation or a known-good successor snapshot, replays accepted canonical events, reapplies deletion lineage, verifies current state, and records divergence.

Rollback never silently discards writes accepted after cutover. Compensation or forward repair is used when reversal would lose valid authority history.

## 8. Exit evidence

A production cutover requires:

- exact source and destination hashes;
- registered backends and profiles;
- mapping and reconciliation reports;
- synthetic and authorized benchmark results;
- failure and recovery tests;
- product-isolation and privacy review;
- deletion and restore verification;
- public-claim update;
- owner or mission authority appropriate to consequence.

## 9. Current state

The M2 contract sequence is closed at the implemented-contract and reversible-prototype level.

A deterministic synthetic cross-prototype evaluation has passed across Claim Authority, shadow adjudication, projection, bounded serving, durable workflow, and deletion propagation.

The admission review authorized preparatory read-only work for Stage A, Stage B, Stage C, and Stage E. PR #83 made Stage A inventory/registration and Stage B deterministic read-only adapters prototype-operational. PR #84 made Stage C destination-candidate profiling and Stage E deterministic synthetic or separately owner-authorized shadow-read evaluation prototype-operational.

The Stage D tranche adds deterministic historical-backfill manifests, source checkpoints, record digests, mapping versions, idempotency keys, accepted/rejected/quarantined/ambiguous accounting, evidence and deletion lineage, reconciliation receipts, and checkpoint continuation. Its repository evaluator uses synthetic records only. Real private historical batch execution requires an explicit owner-authorized manifest and remains separately auditable.

The Stage F+G successor tranche adds a nonproduction controlled-mirroring profile over an explicit canonical writer/outbox stream and generation-bound graph, vector, and workflow build receipts with deletion watermarks. Its repository evaluator is synthetic. Phase 2 remains the canonical writer and current released authority.

Owner-authorized private backfill, expanded mirroring, canary authority, canonical authority transfer, production influence, cutover, Phase 2 retirement, and P5.1e unblock remain available under their own evidence and authority gates.

## Persistent Stage F+G integration evidence after PR #86

The persistent integration profile proves restart/replay durability and persistent projection-receipt semantics through a SQLite compatibility/reference oracle. SQLite is not selected as the migration destination. The current successor architecture is the full polyglot fabric described above: KurrentDB Experience/Event Fabric + NATS JetStream federation transport, XTDB v2 Claim Authority, content-addressed objects, hardware-adaptive Cognitive Multi-Graph placement, Qdrant vector/multimodal projections, source-native retrieval, Temporal workflows, Valkey shared workspace, model/data lineage, and the new resource/recollection/deletion control planes. Existing Neo4j/Qdrant/KurrentDB-era receipts remain useful historical evidence but do not define the final physical stack. Phase 2 remains the canonical writer/current released authority. Owner-authorized private Stage D execution, full-fabric Stage G qualification, Stage H bounded canary review, canonical transfer, production serving, cutover, Stage J compatibility acceptance, Phase 2 final replacement/retirement, and P5.1e remain separate evidence gates.

The owner-ratified identity and host-learning boundary is recorded in `docs/MEMORY_IDENTITY_FORMATION_AND_HOST_LEARNING_ARCHITECTURE.md`: A.L.I.C.E. is an Elaina-derived clone, Rayan is its owner/host, ordinary Rayan learning may update host and relationship models but not the core Elaina-derived identity anchor, and the Memory Formation Model remains a separate learned non-authoritative component.
