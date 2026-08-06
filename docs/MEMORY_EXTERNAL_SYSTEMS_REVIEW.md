# External Memory Systems Code and Architecture Review

**Review date:** 2026-08-03<br>
**Purpose:** Extract concrete patterns for A.L.I.C.E. Memory Architecture v4<br>
**Method:** Review official documentation, papers, public source repositories,
release notes, and open issue reports. Product-internal details of proprietary
systems are treated as unknown.

## 1. Conclusion

No reviewed system satisfies A.L.I.C.E.'s full requirement set.

## 1.1 Accepted A.L.I.C.E. decision

On 2026-08-03, the owner accepted the following synthesis as the canonical v4
authority topology:

- Experience Ledger and payload lineage are evidence;
- a new logical append-only, bitemporal Claim Store is canonical adjudicated
  knowledge;
- Phase 2 becomes a compatibility projection during shadow migration;
- episodes and cognitive models are versioned derivatives;
- graph, vector, lexical, summary, cache, and current-state structures are
  rebuildable projections;
- The first Claim Store compatibility implementation uses SQLite inside the original host process. This implementation baseline does not define the destination topology. Backend-neutral Claim Authority contracts and challenger backends remain authorized.

This is an A.L.I.C.E.-specific synthesis. No external framework is adopted
wholesale.

The best design is not to adopt Mem0, Letta, Graphiti, Hindsight, MemOS,
LightMem, LangGraph, ChatGPT memory, or Claude memory as the architecture.
A.L.I.C.E. should combine their strongest patterns while retaining stricter
authority, provenance, privacy, correction, and owner-control semantics.

The selected synthesis is:

- **A.L.I.C.E. Phase 5:** immutable evidence and payload lineage;
- **A.L.I.C.E. Phase 2:** migration source, correction foundation, and
  compatibility projection;
- **A.L.I.C.E. Claim Store:** canonical append-only adjudicated knowledge;
- **Hindsight:** separate world facts, experiences, and mental models;
- **Graphiti:** raw-episode provenance, validity windows, and invalidation;
- **LightMem:** cheap online filtering and offline sleep-time consolidation;
- **Letta / Claude Code:** bounded core memory plus on-demand archival topics;
- **Mem0:** one-pass extraction, existing-memory shortlist, batch embeddings,
  and explicit change operations;
- **MemOS:** typed memory resources, queues, priorities, quotas, and recovery;
- **LangGraph memory service:** quiescence/debounce and rollback of duplicate
  in-flight processing;
- **Temporal:** deterministic replay, activity isolation, retries, signals, and
  durable long-running memory workflows;
- **ChatGPT memory:** explicit-vs-synthesized memory controls, source
  explanations, corrections, deletion controls, and no-memory sessions;
- **ChronoMem:** versioned rollback as an explicit memory operation.

## 2. System findings

### 2.1 Mem0

#### Source-level pattern

The current open-source V3 pipeline retrieves a bounded shortlist of existing
memories, maps opaque UUIDs to small prompt-facing identifiers, performs one
additive extraction call, batch-embeds extracted memories, stores append-only
ADD history, and optionally extracts/entities links for hybrid retrieval.
The code scopes operations through user, agent, and run identifiers and treats
identity metadata as immutable after creation.

This is a meaningful 2026 change from older mutation-heavy designs: the current
algorithm is explicitly ADD-only. Memories accumulate rather than being silently
rewritten or deleted by the extraction model.

#### Adopt

- one extraction pass per stabilized interaction batch;
- a bounded existing-memory shortlist;
- opaque-ID remapping to reduce model reference errors;
- batch embeddings with deterministic fallback;
- append-first candidate history;
- immutable product/owner/session scope identifiers;
- semantic, keyword, entity, and temporal retrieval signals.

#### Reject or constrain

- additive extraction must create candidates, not authoritative facts;
- a flat text fact must not be the only canonical representation;
- accumulation requires later governed consolidation and invalidation;
- entity maintenance must not scan large collections in the conversation path;
- destructive cleanup requires policy and authority receipts;
- managed-platform benchmark numbers are not evidence that the open-source SDK
  will perform identically.

The code currently includes entity-list operations with a `top_k=10000` ceiling
for exact matching and cleanup. That is a concrete warning against foreground
collection scans as A.L.I.C.E. grows.

### 2.2 Letta / MemGPT

#### Source-level pattern

Letta models core memory as blocks reserved in the model context and archival
memory as externally retrieved passages. Its source schemas include block labels,
limits, read-only state, tags, and metadata. Letta writes authoritative passage
state to SQL before optional secondary systems.

#### Adopt

- hard-bounded core memory blocks;
- read-only protected blocks for constitutional and identity constraints;
- archival memory separate from always-loaded context;
- SQL-first authoritative writes;
- memory inspection and version history;
- background or "sleep-time" maintenance.

#### Reject or constrain

- A.L.I.C.E. must not allow an agent to rewrite protected identity or authority
  blocks without a governed candidate flow;
- core memory cannot grow without a hard token cap;
- secondary indexes require an outbox and reconciliation, not unsafe dual write;
- archival deduplication and consolidation must exist from the beginning.

An open Letta issue requests archival deduplication and consolidation for
long-running agents. Another documents hazards when embedding configuration
changes after passages already exist. These are direct warnings for A.L.I.C.E.

### 2.3 Claude Code memory

Claude Code separates owner-written `CLAUDE.md` instructions from auto memory.
Its auto-memory entrypoint loads only the first 200 lines or 25 KB and moves
details into topic files loaded on demand.

#### Adopt

- separate owner-controlled directives from model-generated memory;
- maintain a concise always-loaded index;
- move detail into mission/topic memory loaded on demand;
- enforce a hard size limit and stale-entry compaction;
- expose what loaded into the current context.

#### Improve

A.L.I.C.E. must treat owner directives as enforceable policy where appropriate,
not merely prompt context.

### 2.4 Graphiti / Zep

#### Source-level pattern

Graphiti uses raw episodes as provenance anchors. Derived entities and edges
carry temporal metadata, and changed relationships are invalidated rather than
silently overwritten. Retrieval combines semantic, keyword, graph, and temporal
signals.

#### Adopt

- every derived relation must trace to an evidence episode;
- distinguish observation/record time from event-valid time;
- invalidate prior relationships rather than erase them;
- use graph retrieval only where multi-hop structure adds value;
- batch extraction and reranking;
- preserve processing-time and event-time watermarks separately.

#### Reject or constrain

- the graph must not be the authoritative source of all memory;
- graph extraction must run asynchronously;
- node attributes must not be destructively upserted;
- graph size and forgetting need explicit policies.

Graphiti issue reports show that edge temporal versioning does not automatically
solve node-attribute history, and users have raised graph-size/forgetting
questions. A.L.I.C.E. must version state explicitly rather than assuming a graph
database provides full temporal correctness.

### 2.5 Hindsight

#### Source-level pattern

Hindsight distinguishes world facts, agent experiences, and mental models.
Its API separates retain, recall, and reflect. It combines semantic, keyword,
graph, and temporal retrieval and supports asynchronous retain/consolidation.

#### Adopt

- separate evidence, experience, and mental-model networks;
- make reflection a separate operation from recall;
- permit LLM-free retrieval after indexing;
- attach disposition and belief updates to explicit projection versions;
- use memory-bank scope boundaries;
- use asynchronous batch retention and consolidation.

#### Reject or constrain

- mental models may not be treated as facts;
- reflection may not silently mutate authoritative claims;
- tag/scope filters must propagate through every internal retrieval;
- consolidation workers require heartbeats, bounded concurrency, and backlog
  observability.

Current open issues report consolidation backlogs that stop draining, missing
heartbeats for long-running consolidation, and a reflect-path scope-filter bug
that can synthesize from stale unfiltered data. These are direct requirements
for A.L.I.C.E.'s worker and scope design.

### 2.6 LightMem

#### Source-level pattern

LightMem uses a three-stage pipeline: lightweight sensory filtering, topic-aware
short-term consolidation, and offline long-term "sleep-time" updates. Its public
configuration separates online add/retrieve from offline update queues and
supports batch evaluation.

#### Adopt

- cheap filtering before model-based curation;
- topic/mission segmentation;
- offline consolidation;
- fixed online retrieval budget;
- micro-batched embeddings and updates;
- benchmark latency, API calls, token usage, and accuracy together.

#### Important caution

A July 2026 reproduction found that retrieval choice strongly changed results
and that raw-turn RAG often matched or exceeded constructed memories at equal
retrieval depth. Constructed memory mainly helped under tight answer-token
budgets and sometimes discarded answer-relevant information.

A.L.I.C.E. must therefore retain raw evidence and benchmark raw evidence,
constructed memory, and hybrid retrieval rather than assuming summarization is
always superior.

### 2.7 MemOS

#### Source-level pattern

MemOS treats memory as typed operational resources and introduces memory cubes,
schedulers, queues, priorities, quotas, versioning, and multiple representations
including plaintext, activation, and parametric memory.

#### Adopt

- common memory envelope with type, provenance, version, owner, and lifecycle;
- composable scope containers;
- asynchronous scheduler;
- task priority, quotas, retries, recovery, and bounded queues;
- inspectable feedback and correction;
- representation-specific governance.

#### Reject or defer

- do not import all memory types at once;
- production promotion of learned memory weights remains gated by replay, deletion, rollback, contamination, evaluation, and owner-approved activation; parametric learning research, dataset construction, challenger training, and shadow evaluation remain authorized;
- do not make a generic "memory OS" abstraction hide epistemic differences;
- performance must be proven before adding representations.

A MemOS issue states that workflows may slow as memory volume or per-session
operations grow. Generality does not remove scale engineering.

### 2.8 LangGraph and LangGraph memory service

#### Source-level pattern

LangGraph separates thread-scoped checkpoint state from long-term namespaced
memory. Its memory-service example waits for a quiescence interval and rolls
back an in-flight memory run if a new user event arrives, preventing duplicate
processing.

#### Adopt

- distinguish conversation checkpoints from durable memory;
- use custom product/host/owner/mission namespaces;
- debounce curation until a session segment stabilizes;
- cancel or supersede duplicate in-flight jobs;
- make jobs idempotent and checkpointed.

### 2.9 Temporal durable execution

Temporal is not a memory model. It is relevant because A.L.I.C.E.'s Curator,
deletion propagation, index rebuilds, migration, and long-running consolidation
are durable workflows.

#### Source-level pattern

Temporal persists an ordered event history and reconstructs workflow state by
deterministic replay. Non-deterministic or external work—network calls, database
queries, LLM calls, and file I/O—is isolated into activities whose results are
recorded. The platform supplies retries, timers, signals, cancellation, and
long-running recovery.

#### Adopt

- every long-running memory mutation has an event history;
- workflow orchestration is deterministic and replayable;
- LLM calls, file operations, and database side effects are explicit activities;
- retries never duplicate semantic writes because activities are idempotent;
- owner approval arrives as a signal rather than an ad-hoc polling loop;
- curation, deletion propagation, migration, and index generation can resume
  after process or host failure.

#### Dependency decision

Memory Architecture v4 adopts these semantics immediately. It does not yet
mandate Temporal as a runtime dependency. The first implementation should use a
local durable workflow contract and SQLite-backed task history, then compare
that implementation with self-hosted Temporal before scale activation.

### 2.10 ChatGPT memory

Only official product behavior is observable; backend source code is private.

OpenAI documents a continuously updated memory synthesis, explicit saved
memories, source explanations for personalization, user correction and deletion
controls, and Temporary Chats that neither use nor create memory. OpenAI also
documents that full deletion may require deleting every source where the
information exists.

#### Adopt

- explicit owner-pinned memory separated from inferred synthesis;
- show why a memory influenced a response;
- owner correction, deletion, and disable controls;
- temporary/no-memory sessions;
- reveal that deletion is a lineage operation, not a single-row delete;
- stale and contradictory memory repair.

#### Improve

A.L.I.C.E. should expose exact provenance, confidence, validity, and derivative
lineage rather than only a high-level memory summary.

### 2.11 ChronoMem

ChronoMem is a recent research system for whole-memory snapshots and semantic
rollback.

#### Adopt as a requirement

- every authoritative mutation must be versioned;
- the owner can select a prior state by natural-language or exact identifier;
- rollback must produce counterfactual behavior consistent with the selected
  historical state;
- rollback itself is a new auditable event.

Do not adopt whole-database snapshots as the only mechanism. A.L.I.C.E. should
support record-level versioning plus periodic consistent snapshots.

## 3. Cross-system failure patterns

The source and issue review repeatedly exposed the same failures:

1. **Accumulation without consolidation**
2. **Summaries that discard answer-relevant evidence**
3. **Stale memories retrieved after invalidation**
4. **Scope filters lost inside nested retrieval**
5. **Embedding-version drift**
6. **Unbounded background backlogs**
7. **No heartbeat or recovery for long jobs**
8. **Graph growth without forgetting**
9. **Agent-written memory becoming authority**
10. **Online curation adding latency**
11. **Dual writes diverging**
12. **Benchmarks rewarding inclusion but not penalizing stale-memory use**
13. **Non-deterministic background jobs that cannot replay after failure**

Memory Architecture v4 must treat these as first-class design constraints.

## 4. Final adoption decision

A.L.I.C.E. will use:

- immutable evidence;
- append-only claim versions;
- rebuildable current-state projections;
- explicit epistemic and authority classes;
- bounded core context;
- on-demand archival retrieval;
- asynchronous, replayable curation workflows;
- typed queues and quotas;
- bitemporal validity;
- hybrid retrieval with batch hydration;
- optional graph projections;
- source-visible context packets;
- owner correction, deletion, and rollback;
- raw-evidence preservation;
- benchmark-driven promotion.

A.L.I.C.E. will not initially use:

- a universal graph database as truth;
- direct LLM mutation of authoritative memory;
- online large-model curation;
- unrestricted context injection;
- automatic deletion;
- personal information in model weights;
- secondary indexes without generation/version tracking;
- unbounded worker queues;
- a single undifferentiated memory table as the final model.

## 5. Primary sources

- A.L.I.C.E. Memory Policy:
  https://github.com/NIne-WIngEd/A.L.I.C.E/blob/main/docs/MEMORY_POLICY.md
- A.L.I.C.E. Phase 2 retrieval:
  https://github.com/NIne-WIngEd/A.L.I.C.E/blob/main/src/alice_memory/retrieval.py
- A.L.I.C.E. Experience Ledger:
  https://github.com/NIne-WIngEd/A.L.I.C.E/blob/main/docs/PHASE_5_COMPACT_EXPERIENCE_LEDGER.md
- Mem0:
  https://github.com/mem0ai/mem0
- Letta:
  https://github.com/letta-ai/letta
- Claude Code memory:
  https://code.claude.com/docs/en/memory
- Graphiti:
  https://github.com/getzep/graphiti
- Hindsight:
  https://github.com/vectorize-io/hindsight
- LightMem:
  https://github.com/zjunlp/LightMem
- LightMem paper:
  https://arxiv.org/abs/2510.18866
- LightMem reproduction:
  https://arxiv.org/abs/2607.29104
- MemOS:
  https://github.com/MemTensor/MemOS
- LangGraph memory service:
  https://github.com/langchain-ai/langgraph-memory
- Temporal:
  https://github.com/temporalio/temporal
- Temporal workflow replay documentation:
  https://docs.temporal.io/workflows
- OpenAI Memory FAQ:
  https://help.openai.com/en/articles/8590148-memory-faq
- ChronoMem:
  https://arxiv.org/abs/2607.27773
- LongMemEval:
  https://arxiv.org/abs/2410.10813
- LoCoMo:
  https://arxiv.org/abs/2402.17753
- MemoryAgentBench:
  https://arxiv.org/abs/2507.05257
- STALE:
  https://arxiv.org/abs/2605.06527
- Memora:
  https://arxiv.org/abs/2604.20006
