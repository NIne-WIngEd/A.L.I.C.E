# Memory M2 Execution Plan — Contract and Full-Memory Parallelism

**Version:** 1.12.0
**Prepared:** 2026-08-06
**Status:** Owner-directed active execution clarification
**Architecture:** Memory v4.1 Capability-First Polyglot Cognitive Fabric
**Capability ceiling:** `false`

## 1. Binding clarification

M2 is a coordinated implementation program, not a declaration that A.L.I.C.E. memory contains only metadata.

Public kernel contract artifacts may carry metadata, content digests, references, authority, provenance, temporal state, deletion state, and rollback information without embedding private payload bytes. That is an artifact-local representation boundary. It does not constrain memory payloads, persistent runtimes, research, prototypes, learning, training, deployment, or the destination architecture.

## 2. Two concurrent execution lanes

### Lane A — neutral contract fabric

Continue the ratified contract sequence:

1. Store Registration, Memory Unit envelope, and Evidence Binding;
2. Claim Identity, Claim Version, and Current Claim Projection;
3. Evidence Relation, adjudication, conflict, and authority records;
4. Episode and Projection Version;
5. Context Packet and Retrieval Trace;
6. Curation Task and Receipt;
7. Durable Workflow and Activity Event;
8. Deletion Propagation Receipt.

These contracts establish authority, interoperability, lineage, deletion, rollback, and cutover semantics. They do not own private payloads or define one permanent backend.

### Lane B — full-memory research and reversible prototypes

Research and prototype work is authorized in parallel for:

- embedded, relational, distributed-SQL, event, graph, vector, object, workflow, model, and hybrid backends;
- persistent Claim Authority append and current-state materialization;
- Experience Ledger subscriptions and evidence-to-candidate pipelines;
- eligibility, rejection, deduplication, conflict detection, and shadow adjudication;
- graph, vector, lexical, symbolic, multimodal, and source expansion;
- automated selective memory formation and intentional forgetting experiments;
- episodes, missions, preferences, values, traits, goals, relationship, source-person, owner, self, world, social, causal, and prediction projections;
- skills, replay, datasets, rankers, routers, adapters, challenger models, and continual-learning experiments;
- deletion influence graphs, propagation, restore filtering, rollback, retirement, and migration rehearsal;
- edge, workstation, cluster, hybrid, remote, multi-device, and federated deployment profiles.

## 3. Concurrency rule

Contract delivery order does not impose research order.

A full-memory track may begin when its own custody, lineage, containment, evaluation, deletion, and rollback prerequisites are sufficient. It does not need to wait for every Lane A contract or later integration wave.

A contract dependency may still gate:

- production influence;
- canonical authority;
- owner-private payload use;
- irreversible migration;
- external release;
- Friday product transfer.

Any such gate must name its exact dependency, evidence, review event, removal criterion, successor path, and rollback or exit.

## 4. M2 tranche pairing

Each contract tranche should pair with at least one full-memory research or prototype objective when technically meaningful.

| Contract tranche | Parallel full-memory objective |
|---|---|
| M2.0 registration/envelope/evidence binding | backend discovery and payload-reference integrity spikes |
| M2.1 claim identity/version/current projection | embedded and alternate Claim Authority persistence prototypes |
| M2.2 evidence/adjudication/conflict | evidence-to-candidate, rejection, and shadow adjudication prototypes |
| M2.3 episode/projection version | graph, vector, owner/source/self, and temporal projection prototypes |
| M2.4 context/retrieval trace | bounded serving, fusion, expansion, and stale-index fallback prototypes |
| M2.5 curation/workflow receipts | durable Curator, migration, repair, and learning workflow prototypes |
| M2.6 deletion propagation | cross-plane deletion, restore filtering, rollback, and retirement rehearsal |

Pairing does not require both sides to ship in one pull request. It prevents contract work from becoming the only active memory workstream.

## 5. Capability-state truth

The following distinctions remain mandatory:

- `implemented` is not the same as `authorized`;
- `prototype_operational` is not the same as `production_profile_enabled`;
- `shadow_evaluated` is not canonical authority;
- a contract artifact is not a persistent store;
- a content digest is not the content;
- a projection is not evidence;
- model output is not adjudicated truth;
- a planned or researched capability is not a released feature.

## 6. Current state after M2 closeout and Stage A+B, Stage C+E, Stage D, and Stage F+G prototype activation

- M1 architecture and capability-first governance are owner-ratified.
- M2.0 through M2.6 contracts are implemented.
- The paired Claim Authority, shadow adjudication, projection, bounded serving, durable workflow, and deletion-propagation profiles remain reversible nonproduction prototypes.
- A deterministic synthetic lineage has been evaluated across claim persistence, candidate adjudication, episode and projection formation, graph and vector retrieval, Context Packet assembly, durable workflow completion, and deletion plus restore-filter rehearsal.
- The closeout report recorded zero canonical authority transfer, zero production influence, zero private payload read, and no Phase 2 migration start. PR #83 subsequently started the read-only migration program through Stage A+B without changing production authority.
- Preparatory read-only work is admitted for Stage A inventory and registration, Stage B contract adapters, Stage C destination-candidate comparison, and Stage E synthetic or separately owner-authorized shadow-read evaluation.
- Historical private-data backfill, controlled write mirroring, canary authority, production influence, cutover, and Phase 2 retirement are not activated by this closeout.
- The admission is a scoped next-work authorization rather than a permanent limit. Distributed, event, graph, vector, workflow, model, federation, training, and successor-backend research remains authorized.
- Phase 2 remains the released compatibility baseline and test oracle.
- Stage A source inventory/registration and Stage B deterministic read-only adapters are prototype-operational. This successor tranche adds Stage C destination-candidate profiles and Stage E deterministic synthetic or separately owner-authorized shadow-read evaluation contracts.
- The Stage A+B profile reads source schema, integrity metadata, and counts through read-only SQLite mode. Record adapters operate on caller-supplied synthetic or separately authorized records and emit mapping, ambiguity, information-loss, correction, deletion-lineage, and reconciliation receipts.
- The Stage C+E profile compares baseline and candidate observations across authority correctness, conflicts, corrections, deletion, privacy, product isolation, latency, staleness, and explanation traces. PR #84 made that nonproduction profile prototype-operational and its deterministic synthetic evaluation recommended eligibility for the next research gate rather than production selection.
- The Stage D profile now provides deterministic historical-backfill manifests, record and batch receipts, idempotent replay, provenance and deletion lineage, reconciliation, and checkpoint continuation. Its repository evaluation is synthetic; real private historical batches require an explicit owner-authorized manifest and external custody evidence.
- The Stage F+G profile now provides nonproduction canonical-outbox mirroring receipts plus graph/vector/workflow generation manifests, deletion watermarks, generation digests, replay evidence, and repair accounting. Phase 2 remains the canonical writer and current released authority under this profile.
- The persistent Stage F+G integration profile now proves durable restart/replay semantics through the SQLite compatibility/reference adapter and registers KurrentDB, Neo4j, Qdrant, and Temporal as non-exclusive research candidates. No external candidate is selected or contacted by the public synthetic evaluation.
- P5.1e remains paused until its exact storage-admission interaction is explicitly unblocked. That pause does not restrict Stage C/E research or any ratified distributed, graph, vector, workflow, model, training, federation, or successor-backend track.

## 7. Permanent anti-ceiling rule

The phrase `metadata-only` may describe a named public contract artifact. It may not describe the full M2 program, A.L.I.C.E. memory, the Cognitive Kernel, or the destination architecture.

Any future document, policy, test, pull-request boundary, or implementation note that turns an artifact-local representation into a phase-wide, system-wide, research, prototype, backend, payload, learning, or deployment restriction is an unresolved capability barrier and must fail governance review.
