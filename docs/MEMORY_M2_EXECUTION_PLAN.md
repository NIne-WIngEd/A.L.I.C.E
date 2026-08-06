# Memory M2 Execution Plan — Contract and Full-Memory Parallelism

**Version:** 1.4.0
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

## 6. Current state after M2.4 bounded-serving activation

- M1 architecture and capability-first governance are owner-ratified.
- M2.0 through M2.3 contracts and their paired reversible prototypes are implemented.
- M2.4 Context Packet, Context Selection, Retrieval Trace Step, and Retrieval Trace contracts are implemented by this tranche.
- A reversible persistent bounded-serving prototype stores full context-packet and retrieval-trace content outside the public repository.
- Profile-selected item and byte budgets bound ordinary serving without becoming universal architecture maxima.
- Lexical, vector, graph, temporal, and mission-aware sources may be fused and expanded with deterministic receipts.
- Stale or unavailable indexes invoke an explicit profile-selected fallback path and remain visible in the Retrieval Trace.
- Context Packets are bounded serving artifacts. They are not evidence, adjudicated Claim Authority truth, or permanent memory by themselves.
- The embedded SQLite profile is one reversible reference challenger. It does not constrain distributed serving, graph-native retrieval, external vector systems, model rankers, remote compute, or hybrid successors.
- Full-memory research remains authorized for automatic selective memory, cognitive models, curation, durable workflow, learning, training, deletion, rollback, migration, federation, and every other ratified track.
- Production serving influence remains governed by named profiles, evaluation evidence, privacy and deletion validation, cutover receipts, and owner approval.
- Phase 2 migration is not asserted started.
- P5.1e remains paused until its exact storage-admission interaction is explicitly unblocked. That pause does not prohibit other memory research or prototypes.

## 7. Permanent anti-ceiling rule

The phrase `metadata-only` may describe a named public contract artifact. It may not describe the full M2 program, A.L.I.C.E. memory, the Cognitive Kernel, or the destination architecture.

Any future document, policy, test, pull-request boundary, or implementation note that turns an artifact-local representation into a phase-wide, system-wide, research, prototype, backend, payload, learning, or deployment restriction is an unresolved capability barrier and must fail governance review.
