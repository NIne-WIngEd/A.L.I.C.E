# A.L.I.C.E. Memory Architecture v4

**Draft:** 0.2<br>
**Status:** Owner-accepted architecture direction; implementation remains held<br>
**Decision accepted:** 2026-08-03<br>
**Goal:** A coherent, bounded, inspectable, evolvable memory system capable of
supporting A.L.I.C.E.'s public destination without turning every conversation
into a slow, noisy, or irreversible database mutation.

## 1. Core design decision

A.L.I.C.E. will not use one universal "memory database."

The accepted canonical authority topology is:

```text
Experience Ledger and payload lineage
    = immutable evidence of events, sources, actions, and outcomes

Append-only bitemporal Claim Store
    = canonical adjudicated knowledge

Phase 2 Memory Core
    = compatibility projection during shadow migration

Episodes, missions, beliefs, preferences, traits, and identity models
    = versioned derived memory

Lexical, vector, graph, summary, cache, and current-state indexes
    = rebuildable derivatives

Memory Context Packet
    = bounded serving decision for one model invocation
```

It will use a memory architecture with distinct truth roles:

1. **Evidence says what was observed or executed.**
2. **Claim versions say what is asserted, by whom, with what support, and when
   it is valid.**
3. **The current-claim projection says what is presently adjudicated for fast
   serving.**
4. **Episodes organize experience without replacing evidence.**
5. **Cognitive projections summarize beliefs, models, and relationships.**
6. **Indexes make information searchable but are never authoritative.**
7. **Context packets decide what enters a model invocation.**
8. **Curation changes memory only through governed candidates.**

### 1.1 Claim Store implementation boundary

"New Claim Store" means a new logical authority layer with new Cognitive Kernel
contracts and tables. Version 1 remains inside the existing host-local SQLite
architecture. It does not introduce a network service, distributed database, or
mandatory third-party dependency.

### 1.2 Why Phase 2 is not the permanent universal authority

Phase 2 remains a valuable released foundation. Its general memory rows combine
facts, statements, preferences, episodes, relationships, reflections,
inferences, disputes, and identity information behind a shared content-oriented
shape. That was appropriate for foundation work, but it is too semantically
broad for the final authority model.

The migration preserves Phase 2 behavior through a compatibility projection
while the Claim Store introduces structured identities, append-only versions,
evidence links, adjudications, bitemporal validity, and explicit current-state
materialization.

### 1.3 Append-only does not prohibit privacy deletion

Ordinary semantic change appends a new version, supersession, dispute, or
retraction record. Authorized privacy deletion is a separate governed workflow
that may revoke, cryptographically erase, or physically excise protected
content while retaining only the minimum non-sensitive tombstone and audit
receipt permitted by policy.

## 2. Permanent principles

### 2.1 Evidence before interpretation

Raw events and source references remain available whenever policy permits.
Summaries, graphs, beliefs, and traits must link to their evidence.

### 2.2 Authority is explicit

A model-generated statement is not a fact. Every record carries an authority and
epistemic class.

### 2.3 Derived state is rebuildable

Vector indexes, lexical indexes, graph projections, summaries, current-state
tables, context caches, and owner models are derived artifacts.

### 2.4 The online path is bounded

Ordinary conversation cannot trigger unbounded scans, full-store verification,
large-model curation, full graph traversal, index rebuilds, or deep
consolidation.

### 2.5 Background work cannot block dialogue

Curation may lag. Dialogue must degrade safely to current authoritative claims,
active mission state, and raw retrieval.

### 2.6 Mutation is append-only at the semantic level

Corrections, supersessions, belief changes, and rollback create new versions.
History is not silently overwritten.

### 2.7 Personal parametric memory is blocked

Personal adapters and weight updates remain disabled until deletion, rollback,
replay lineage, contamination, and unlearning limits are ratified.

## 3. Memory planes

### Plane A — Evidence Plane

Stores immutable or logically append-only records of:

- conversation turns;
- source documents and chunks;
- tool calls and outputs;
- model, prompt, and policy versions;
- decisions and authority receipts;
- actions and execution results;
- observed outcomes;
- corrections and owner feedback;
- deletion requests and propagation receipts;
- evaluation and rollback evidence.

Primary foundations:

- Phase 5 Experience Ledger;
- raw buffer;
- content-addressed payload store;
- lifecycle journal;
- governed tier-transition receipts.

An Evidence Event proves that an event was recorded. It does not prove that every
claim inside the event is true.

### Plane B — Authoritative Claim Plane

The new logical Claim Store is the canonical layer for versioned, adjudicated
knowledge. It is append-only at the semantic level and bitemporal.

Claim Store v1 contains at minimum:

- claim identities;
- claim versions;
- claim-to-evidence relations;
- adjudication receipts;
- conflict and supersession relations;
- a materialized `current_claims` projection.

A claim is structured around:

- subject;
- predicate;
- object/value;
- scope;
- epistemic class;
- authority class;
- confidence;
- valid time;
- recorded time;
- support and opposition evidence;
- sensitivity;
- lifecycle state;
- correction and supersession lineage.

The claim plane replaces the idea that arbitrary free-text rows alone define
current truth.

The existing Phase 2 `memories` table becomes a compatibility projection during
migration, not the permanent universal schema or canonical write authority.

### Plane C — Episodic Plane

Groups evidence events into bounded episodes such as:

- one stabilized conversation segment;
- one mission attempt;
- one research investigation;
- one decision and its outcome window;
- one failure and repair sequence.

An episode contains:

- event IDs;
- start/end event time;
- start/end record time;
- participants and mission IDs;
- a concise derived summary;
- unresolved questions;
- outcome state;
- summary-generation lineage.

The summary may be regenerated. The episode's evidence bindings are permanent
subject to deletion policy.

### Plane D — Mission and Commitment Plane

Maintains the current operational state of:

- missions;
- submissions and dependencies;
- decisions;
- commitments;
- deadlines;
- blockers;
- expected outcomes;
- actual outcomes;
- owner overrides;
- parked or superseded work.

This plane joins Mission Graph contracts to evidence and claims.

### Plane E — Cognitive Projection Plane

Contains rebuildable, versioned models:

- owner model;
- source-person model;
- reconstruction hypotheses;
- owner relationship model;
- A.L.I.C.E. self-model;
- world model;
- social model;
- goals and values;
- beliefs;
- predictions;
- causal hypotheses;
- recurring failure patterns.

A projection is never promoted to historical fact merely because it is useful.

### Plane F — Memory Serving Plane

Builds a bounded Memory Context Packet for each model invocation.

The packet contains:

- mandatory constitutional/authority constraints;
- relevant active mission state;
- current authoritative claims;
- selected episodes;
- unresolved contradictions;
- uncertainty and temporal qualifiers;
- compact evidence references;
- a retrieval trace.

The serving plane controls context size. The model does not receive unlimited
memory access by default.

### Plane G — Curation and Learning Plane

Asynchronously performs:

- session stabilization;
- event segmentation;
- candidate extraction;
- deduplication;
- claim matching;
- contradiction detection;
- promotion review;
- episode summarization;
- projection refresh;
- prediction-outcome matching;
- skill nomination;
- retention nomination;
- archive and replay nomination.

It writes candidates and receipts. It does not silently rewrite authoritative
state.

### Plane H — Inspection and Governance Plane

Allows the owner to inspect:

- what A.L.I.C.E. currently believes;
- which claims are authoritative;
- supporting and opposing evidence;
- what entered a context packet;
- why a memory was selected;
- candidate promotions;
- stale or disputed state;
- correction and deletion progress;
- projection versions;
- rollback points;
- worker backlog and curation lag.

## 4. Common Memory Unit envelope

Every first-class unit uses a common envelope:

- `memory_unit_id`
- `memory_kind`
- `schema_version`
- `product_id`
- `host_instance_id`
- `owner_scope_id`
- `mission_scope_ids`
- `authority_class`
- `epistemic_class`
- `sensitivity_class`
- `observed_at`
- `recorded_at`
- `valid_from`
- `valid_to`
- `content_sha256`
- `source_event_ids`
- `derivation_id`
- `policy_version`
- `model_id`
- `prompt_version`
- `lifecycle_state`
- `deletion_state`
- `generation_id`
- `workflow_id`
- `workflow_event_id`
- `created_at`

Kind-specific records remain separate. A common envelope does not collapse facts,
episodes, beliefs, and skills into the same semantics.

## 5. Authority and epistemic classes

### Authority classes

Provisional ordering:

1. owner-ratified constitutional directive;
2. owner-attested statement;
3. verified authoritative source;
4. direct system observation with integrity receipt;
5. corroborated external evidence;
6. derived inference;
7. generated reconstruction;
8. unsupported model hypothesis.

This is not a universal numeric truth score. Domain-specific policy may decide
which source is authoritative.

### Epistemic classes

- observation;
- owner_statement;
- external_claim;
- verified_fact;
- inference;
- belief;
- prediction;
- preference_observation;
- trait_hypothesis;
- generated_reconstruction;
- dispute;
- uncertainty.

## 6. Temporal model

Every changing record must distinguish:

- **event/valid time:** when the information applies in the world;
- **record/system time:** when A.L.I.C.E. learned or changed the record.

Current state is a projection over claim versions, not a destructive overwrite.

Implicit invalidation is a separate candidate operation. It requires evidence and
propagation tests before changing related claims.

## 7. Synchronous conversation path

An ordinary turn performs only:

1. append compact evidence metadata;
2. load constitutional and protected core state;
3. load active mission snapshot;
4. classify memory needs;
5. retrieve bounded candidates;
6. batch-hydrate records and relations;
7. filter by scope, authority, validity, deletion, and sensitivity;
8. rerank and diversify;
9. build the Memory Context Packet;
10. generate the response;
11. record response and immediate execution metadata.

The synchronous path must not:

- call a large Curator model by default;
- rebuild an index;
- verify the entire store;
- scan every correction relation;
- traverse an unbounded graph;
- consolidate history;
- mutate traits or beliefs;
- decide deletion;
- train a model.

## 8. Background curation path

### Stabilization

A conversation segment enters curation only after a quiescence window. A new
turn supersedes or extends the pending job rather than producing duplicate
processing.

### Cheap-first pipeline

1. deterministic eligibility and privacy filter;
2. exact/hash duplicate filter;
3. lightweight topic/mission segmentation;
4. bounded existing-memory retrieval;
5. small/local extraction where possible;
6. larger model only for ambiguous high-value cases;
7. candidate persistence;
8. policy review and promotion;
9. derived-index update through outbox;
10. audit receipt.

### Worker requirements

Every job has:

- durable workflow ID and ordered event history;
- idempotency key;
- input generation;
- priority;
- retry budget;
- heartbeat;
- deadline;
- resource estimate;
- cancellation and supersession state;
- checkpoint;
- output receipt;
- failure quarantine;
- deterministic replay from recorded workflow events.

The workflow coordinator must be deterministic. LLM calls, database mutations,
file I/O, network calls, clock reads, and randomness are external activities
whose inputs and outputs are recorded. Memory Architecture v4 adopts Temporal's
durable-execution semantics without requiring Temporal as the first runtime.

## 9. Serving and context budgets

Provisional default:

```text
usable_context = model_context - system/tool/output reserves
memory_budget = min(4096 tokens, floor(0.12 * usable_context))
```

Within the memory budget:

- protected core: maximum 20%;
- active mission: maximum 25%;
- current claims: maximum 30%;
- episodes/evidence: maximum 20%;
- contradictions, uncertainty, and lineage: minimum reserve 5%.

These are starting limits, not permanent constants. They must be benchmarked per
model profile.

The protected core must never grow with total history.

## 10. Retrieval design

### Query plan

The serving plane decides whether a query requires:

- current-state lookup;
- temporal lookup;
- mission lookup;
- episodic similarity;
- evidence retrieval;
- relationship traversal;
- contradiction expansion;
- owner/source-person projection;
- no memory at all.

### Candidate pipeline

1. structured current-state lookup;
2. lexical retrieval;
3. vector retrieval;
4. optional graph candidates;
5. reciprocal-rank or calibrated fusion;
6. batch hydration;
7. policy filters;
8. temporal and authority adjudication;
9. diversity and source coverage;
10. optional local reranker;
11. packet assembly.

### Current Phase 2 corrections

The current retrieval implementation must eventually replace:

- global corrected-target scanning;
- per-candidate `load_memory` calls;
- per-result conflict queries;
- full index verification in every ordinary read.

With:

- a materialized current-state projection;
- batched `IN` hydration;
- batched relation loading;
- generation-bound index manifests;
- incremental verification;
- cached immutable policy state.

## 11. Storage and indexing

### Authoritative durability

Evidence and claim-version commits require the strongest configured durability.

### Derived durability

Indexes and projections may use lower-cost durability because they are
rebuildable, but every generation is bound to:

- source generation;
- embedding model/version;
- normalization version;
- schema version;
- build receipt;
- integrity digest.

### Claim Store read and write model

Normal writes append a claim version and update the materialized current-state
projection in one authoritative transaction. Derived lexical, vector, graph,
summary, and cache updates occur through the durable outbox.

Normal reads query the bounded current-state projection. Historical replay is an
explicit operation and is never required for an ordinary response.

### Write coordination

All persistent memory mutations pass through one host-scoped Memory Write
Coordinator and durable outbox.

Secondary vector, lexical, graph, and cache updates are asynchronous. Failed
secondary writes cannot roll back evidence already committed; they create
reconciliation tasks and keep the affected generation unavailable.

## 12. Correction, deletion, and rollback

### Correction

A correction creates:

- a new claim version;
- a correction edge;
- current-state projection update;
- index invalidation;
- projection refresh tasks;
- context-cache invalidation;
- audit receipt.

### Deletion

A deletion request creates a propagation graph covering:

- raw payloads;
- evidence accessibility;
- claims;
- episodes;
- summaries;
- graph edges;
- indexes;
- context caches;
- replay manifests;
- training candidates;
- skills;
- adapters/models where technically supported;
- backups and future restore filters.

Completion requires acknowledgements from every registered derivative.

### Rollback

Rollback selects a consistent version or snapshot and creates a new rollback
event. It does not erase the later history.

## 13. Degraded modes

A.L.I.C.E. remains usable when:

- vector index is stale;
- graph projection is unavailable;
- Curator is behind;
- embedding model is unavailable;
- archival store is offline;
- a projection is quarantined.

Fallback order:

1. protected core;
2. current authoritative structured lookup;
3. active mission state;
4. lexical evidence retrieval;
5. explicit disclosure of missing memory services.

## 14. Scale targets

The architecture must be evaluated at:

- 10K, 100K, 1M, and 10M evidence events;
- 1K, 10K, and 100K durable claim versions;
- 1K, 10K, and 100K episodes;
- sustained curation backlog and worker restart;
- index version migration;
- deletion propagation under load;
- model replacement with identical memory state.

## 15. Migration principle

Phase 2 and Phase 5 will not be joined by copying all rows into one table.

They will be joined through canonical identifiers and explicit roles:

```text
Experience Ledger event
    -> evidence payload/reference
    -> memory candidate
    -> claim or episode version
    -> current-state projection
    -> derived indexes/projections
    -> Memory Context Packet
    -> response/action
    -> outcome event
```

The existing Phase 2 API remains a compatibility profile until callers move to
the new kernel contracts.

## 16. Remaining ratification questions

The authority topology is resolved: Claim Store v1 is a new logical append-only,
bitemporal authority layer implemented through new Cognitive Kernel contracts
and tables in the existing host-local SQLite architecture. Phase 2 remains a
compatibility projection during shadow migration.

Before implementation resumes, decide:

1. Which writes require owner review?
2. What counts as a stable preference, value, habit, or trait?
3. Which model profiles may run curation locally?
4. What are the final context and latency budgets per hardware class?
5. What is the minimum deletion guarantee before parametric learning?
6. Which graph relationships justify graph storage?
7. Which private identity projections may be automatically updated?
8. Should the first durable Curator use the local SQLite workflow engine or
   self-hosted Temporal after comparative evaluation?
