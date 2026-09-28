# Frontier Watch intake — read-time curation, reconsolidation, analytic memory, and deletion echoes

**Date:** 2026-09-27  
**Status:** research intake; future-architecture implications recorded; no current N0 blocker  
**A.L.I.C.E. comparison authority:** `main@e6d0490cdf29ee56edbcb0e3d65d5783adfdf1a7`  
**Frontier-watch branch head at intake start:** `2d8ef267961d988be2cdc2c82432b46bff7f36c3`  
**Fable comparison authority:** `NIne-WIngEd/Fable_Sleight main@191b0dbf8c1c33114b56e8e5e2c4be52076cd4ce`

## 1. Scan scope

This intake reviewed 108 search-result candidates across 10 research angles covering:

- lifelong / episodic / autobiographical memory;
- graph and associative recollection;
- continual personalization;
- parametric memory;
- deletion / unlearning;
- multimodal / perceptual memory;
- working-memory and memory-control systems;
- long-horizon procedural learning;
- September 22–27 freshness.

Candidates already recorded in the repository, including MemCalib, Jev-Mem, and RPMem, were not treated as new findings.

Primary papers were deep-read before architectural comparison.

## 2. Current physical baseline correction

The frontier-watch baseline had become stale relative to current main.

Current machine policy and the latest main architecture documents establish:

- **Experience/Event authority:** KurrentDB canonical event store behind project-owned event/evidence contracts;
- **federation/edge ingress:** NATS JetStream;
- **Claim Authority:** XTDB v2;
- **local Cognitive Multi-Graph:** LadybugDB;
- **scale-out Cognitive Multi-Graph:** NebulaGraph;
- **existing graph reference:** Neo4j;
- **vector/multimodal:** Qdrant;
- **durable workflows:** Temporal;
- **shared ephemeral state:** process-local L1 + Valkey.

The frontier-watch baseline was corrected accordingly.

This does not reopen backend selection.

---

# 3. Finding A — JitMem: read-time reconstructive curation

**Paper:** *Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents*  
**arXiv:** 2609.27334  
**Submitted:** 2026-09-23  
**Primary source:** https://arxiv.org/abs/2609.27334

## Mechanism

JitMem argues that many agent-memory systems make a premature decision: they distill a trajectory into one fixed artifact at write time before the future query is known.

Its alternative is:

```text
raw/reconstructible trajectories
        ↓ retrieve
current task/query
        ↓
read-time curator
        ↓
task-specific ephemeral memory payload
        ↓
executor
        ↓
same-task outcome / reward
```

The same historical trajectory can therefore produce different memory payloads for different future tasks.

The paper also obtains a much shorter credit-assignment path because the curator's output is evaluated on the task for which it was just constructed.

Reported results show substantial gains over write-time memory methods on ALFWorld, WebShop and τ²-bench. The paper also reports lower executor token use and fewer execution steps.

## What A.L.I.C.E. already does

A.L.I.C.E. is already less exposed to JitMem's criticism than a normal write-time memory agent because the current architecture includes:

- aggressive temporary capture;
- raw evidence/object retention;
- source-native retrieval;
- an Experience/Event Fabric;
- fast/slow formation;
- episodes;
- Context Planner;
- Retrieval Orchestrator / Cognitive Recollection;
- reconstructible derived projections.

MFM also proposes semantic memory rather than being allowed to destroy source truth.

## Genuinely new delta

The current architecture does not yet explicitly define **task-conditioned read-time reconstruction as a first-class memory operation**.

That is worth adding.

The important idea is not “store everything forever.” It is:

> A write-time memory representation must not be assumed to contain every future-useful interpretation of an experience.

A.L.I.C.E. should preserve enough reconstructible evidence that a future query can derive a new task-specific representation when needed.

## Architecture implication

Add a future Track F / Stage G challenger:

### Reconstructive Read-Time Curator

Inputs:

- current query / task state;
- relevant Experience/Event ranges;
- raw/source-native evidence pointers;
- existing episodes / claims / graph / analytic views;
- current mission / host / relationship / self context;
- retrieval budget.

Output:

- an **ephemeral, query-conditioned context artifact**;
- explicit source/evidence bindings;
- coverage/confidence metadata;
- no independent truth authority.

The artifact may summarize, reorganize or extract a different “lesson” from the same experience for different tasks.

It is discarded or retained only as a derived optimization artifact. The underlying evidence remains authoritative according to normal memory policy.

## Validation requirement

Compare:

1. write-time derived memory only;
2. source/raw retrieval only;
3. hybrid write-time + read-time reconstruction.

Test:

- future queries whose relevant interpretation was not predictable when the event was written;
- rare old evidence;
- contradictory history;
- temporal changes;
- procedural reuse;
- multimodal source evidence;
- storage pressure;
- token/runtime cost.

### Retention implication

Do **not** replace selective retention with permanent raw retention.

Instead add a rule/challenger:

> Before irreversible payload destruction, evaluate whether the remaining Event/Claim/source representation is sufficient for future task-conditioned reconstruction.

This belongs in future Lifecycle Curator qualification.

**Classification:** direct future capability improvement.  
**Current N0 impact:** none.

---

# 4. Finding B — REALM: retrieval-driven memory reconsolidation

**Paper:** *Retrieval-Driven Memory Reconsolidation for Long-Term LLM Agents*  
**arXiv:** 2609.16053  
**Submitted:** 2026-09-13  
**Primary source:** https://arxiv.org/abs/2609.16053

## Mechanism

REALM treats retrieval as part of the memory lifecycle rather than a terminal read.

Its sequence is approximately:

```text
memory graph
 -> adaptive seed/expand/filter retrieval
 -> activated local subgraph
 -> task outcome / feedback
 -> reconsolidation
 -> add / strengthen / weaken derived relations
 -> changed future accessibility
```

The paper reports that retrieval-driven graph reconsolidation improves long-term memory benchmarks and that repeatedly co-used memories form more coherent local structures.

## What A.L.I.C.E. already does

Current architecture already includes:

- Cognitive Multi-Graph;
- graph views for semantic, causal, temporal, social, evidence, mission, identity and other relations;
- associative compute;
- usage-aware retrieval/accessibility projections learned from retrieval/outcome receipts;
- Retrieval Trace;
- Lifecycle Curator;
- versioned/rebuildable projections.

So REALM is **not** evidence to replace A.L.I.C.E.'s graph design.

## Genuinely new delta

The current design mentions usage-aware accessibility, but does not yet define a concrete **post-retrieval reconsolidation loop**.

Add a future challenger in which verified retrieval/outcome receipts may update:

- associative edge strength;
- accessibility;
- bridge/path priors;
- query-conditioned expansion priors;
- co-activation structure.

## Authority boundary

REALM's topology mutation must **not** be copied literally into Claim Authority.

For A.L.I.C.E.:

- Claim truth does not become stronger because it was retrieved frequently;
- evidence provenance does not change because memories co-occurred;
- repeated model mistakes must not self-reinforce;
- retrieved association is not evidence.

Only rebuildable derived topology/accessibility may reconsolidate automatically.

Changes should be conditioned on verified outcomes or sufficiently strong usage evidence.

## Validation requirement

Test:

- outcome-validated vs blind co-retrieval reconsolidation;
- popularity feedback loops;
- repeated wrong-answer contamination;
- rare-but-important evidence;
- temporal contradictions;
- deletion/revocation rollback;
- rebuild from authoritative stores;
- long-horizon retrieval improvement.

Require every reconsolidation update to carry:

- originating retrieval trace;
- outcome/feedback receipt;
- old/new weight or relation;
- policy/model version;
- rollback lineage.

**Classification:** direct future capability improvement; strengthens Track C/F/Lifecycle Curator.  
**Current N0 impact:** none.

---

# 5. Finding C — AdaMM: analytic memory as a separate derived projection

**Paper:** *Beyond Retrieval: Analytic Memory for Multimodal Agents*  
**arXiv:** 2607.29440  
**Submitted:** 2026-07-31  
**Primary source:** https://arxiv.org/abs/2607.29440

## Mechanism

AdaMM identifies a retrieval/analysis mismatch.

Semantic retrieval is appropriate for questions such as:

- “What happened when...?”
- “What did I say about...?”

But it is structurally weak for:

- average over a time range;
- all occurrences matching a condition;
- top-N ranking;
- trend/change;
- comparison between periods;
- exact counts;
- repeated multimodal observations.

AdaMM therefore builds a complementary analytic representation:

```text
dialogue / image / metadata evidence
        ↓
provenance-linked attribute/value observations
        ↓
recurring-schema induction
        ↓
materialized analytic tables/views
        ↓
filter / aggregate / rank / temporal operations
```

A memory-aware planner composes semantic retrieval and analytic operations.

## What A.L.I.C.E. already does

Current A.L.I.C.E. has:

- Claim Authority;
- Event history;
- graph projections;
- source-native exact reads;
- vector/multimodal retrieval;
- Context Planner.

These can answer some analytic questions indirectly.

But there is no named first-class **analytic-memory projection** that guarantees complete-range operations over recurring personal observations.

## Genuinely new delta

Add a rebuildable **Analytic Memory / Longitudinal View** projection.

Examples:

- sleep/activity measurements;
- recurring schedule patterns;
- project execution history;
- tool outcomes;
- spending or task metrics where authorized;
- repeated behavioral observations;
- multimodal recurring attributes;
- model/evaluation histories.

It should support:

- filters;
- grouping;
- aggregates;
- ranking;
- time windows;
- deltas/trends;
- completeness checks.

## Authority boundary

Analytic Memory is never Claim Authority.

Every materialized cell or observation should retain:

- source Event/Evidence identity;
- extraction model/version;
- occurrence time;
- confidence;
- correction/deletion state.

Every aggregate should retain:

- query/filter definition;
- row coverage;
- missing-data information;
- source-range identity;
- materialization version.

This prevents an induced table from silently becoming a new truth store.

## Retrieval-planner implication

Cognitive Recollection should classify or decompose a query into one or more of:

- Claim lookup;
- source-native exact lookup;
- semantic/vector recall;
- graph/associative traversal;
- episodic recall;
- **analytic operation**;
- live system-of-record read.

The answer may compose results from several.

## Validation requirement

Create future tests where:

- top-k semantic retrieval is provably incomplete;
- exact analytic computation requires all matching observations;
- schemas evolve over time;
- extraction errors exist;
- corrections/deletions modify historical rows;
- multimodal evidence contributes attributes.

Measure analytical correctness and source coverage, not only answer similarity.

**Classification:** new future memory plane/projection worth incorporating.  
**Current N0 impact:** none.

---

# 6. Finding D — MUTE: post-deletion influence echoes

**Paper:** *When Unlearning Fails: Reliable Data Deletion under Post-Training in Agent Networks*  
**arXiv:** 2607.28829  
**Submitted:** 2026-07-30  
**Primary source:** https://arxiv.org/abs/2607.28829

## Mechanism

MUTE studies a failure mode directly relevant to a self-improving Fable.

A memory/data item can affect a deployed policy. That policy changes later behavior. Later behavior generates new trajectories. Those trajectories are then retained and trained on.

Therefore:

```text
source to delete
 -> model / policy influence
 -> later behavior
 -> later retained trajectory
 -> later training
 -> regenerated deleted influence
```

Deleting the original source or retraining on an apparently retained dataset can still fail because some “retained” trajectories are causally downstream of the deleted material.

The paper calls this an **influence echo**.

Its proposed mitigation combines:

- lineage-based influence estimation;
- model-side erasure;
- quarantine/down-weighting of high-influence downstream trajectories;
- continued post-deletion audits;
- later erasure when influence reappears.

## What A.L.I.C.E. already does

This paper strongly validates the architecture we just ratified.

Current Track J already requires deletion across:

- source data;
- Events;
- Claims;
- episodes;
- graph/vector projections;
- contexts and plans;
- replay/training data;
- adapters/weights;
- backups/exports;
- later regenerated memory.

The product rule already states that deleted external memory must not re-teach a scrubbed model and residual model influence must not regenerate deleted external memory.

## Genuinely new delta

Add an explicit **post-deletion influence-echo qualification**.

Deletion is not complete when the deletion job finishes.

After deletion, A.L.I.C.E./Fable must continue operating and learning, then test whether the influence reappears.

## Influence-lineage extension

Where feasible, retained Experience should record enough lineage to estimate whether it was generated under:

- a model generation influenced by deleted material;
- a context containing deleted material;
- a skill/procedure derived from deleted material;
- a plan/mission state influenced by deleted material.

This does not mean every descendant must always be deleted automatically.

It means the deletion coordinator can identify high-risk descendants for:

- quarantine;
- down-weighting;
- recomputation/replay;
- retraining exclusion;
- stronger unlearning;
- inspection.

## Validation requirement

Introduce future tests:

1. train/operate with source X;
2. allow X to affect actions and later Experience;
3. delete X;
4. remove direct source/model residue;
5. continue operation and learning;
6. test whether X's behavior or information regenerates;
7. trace the regeneration path;
8. quarantine/rebuild/unlearn descendants;
9. repeat until leakage is below the accepted profile.

Track both:

- immediate deletion success;
- delayed influence-regeneration rate.

**Classification:** concrete Track J qualification improvement; validates current architecture rather than replacing it.  
**Current N0 impact:** none.

---

# 7. Strong corroboration, not new planes

## PGMem — Tightly Coupled Persona–Memory Graph

**arXiv:** 2608.01708  
**Source:** https://arxiv.org/abs/2608.01708

PGMem links event memory and persona state using explicit:

- provenance;
- support;
- contradiction;
- temporal-shift relations.

Retrieval expands through these evidence links and ranks personal-state signals by evidential validity.

This strongly supports A.L.I.C.E.'s existing separation of:

- Experience/Evidence;
- Claim Authority;
- host model;
- source-person model;
- relationship model;
- assistant-self model.

### Useful refinement

When retrieving host/relationship/self projections, score not only semantic relevance but also:

- current validity;
- supporting evidence;
- contradicting evidence;
- supersession/temporal shift.

Do not create a separate persona truth store. Derive these links from authoritative Event/Claim/Evidence state.

**Classification:** corroboration + retrieval-quality refinement.

---

## Recuris — Recursive Experiential–Working Memory Evolution

**arXiv:** 2608.24876  
**Source:** https://arxiv.org/abs/2608.24876

Recuris couples verified Working Memory with Experiential/Skill Memory.

A structured trace records:

- verified working state;
- retrieved skill;
- action;
- observation/result;
- proposed next state;
- checker decision;
- committed next state.

Failures are localized to specific memory-control components and only targeted patches are admitted through held-out regression gates.

This directly supports current Track H, which already requires evidence-grounded failure localization before mutation.

### Useful refinement

Adopt the structured trace shape as a future F8/self-evolution candidate contract.

Candidate procedural/harness changes should be:

- localized;
- versioned;
- replayable;
- validated on source failures;
- blocked if held-out tasks regress beyond policy.

**Classification:** strong corroboration and implementation guidance.

---

## FRESH — failure-aware heterogeneous procedural graph

**arXiv:** 2609.28003  
**Source:** https://arxiv.org/abs/2609.28003

FRESH represents:

- tasks;
- tools;
- actions;
- observations;
- errors;
- preconditions;
- repairs;
- successful outcomes.

It retrieves compact guidance in forms such as:

- do;
- avoid;
- check;
- repair.

A live precondition/policy gate checks risky actions against current observations.

### Useful refinement

Procedural memory should retain **failure causes, preconditions and repair structure**, not only successful skill text.

This aligns with the existing multi-graph and Track H architecture.

**Classification:** corroboration.

---

# 8. Architectural action from this intake

No current N0/N1 model-training work should be interrupted.

No selected backend should be reopened.

No new infrastructure tournament is justified.

The useful future additions are:

1. **Reconstructive Read-Time Curator** under Cognitive Recollection / Context Planner.
2. **Retrieval-driven derived-topology reconsolidation** under Lifecycle Curator / associative graph plane.
3. **Analytic Memory projection** for complete-range longitudinal and multimodal operations.
4. **Influence-echo deletion tests and descendant containment** under Track J.
5. **Validity-aware host/self/relationship retrieval** using support/contradiction/shift evidence.
6. **Structured EM–WM execution trace** for procedural/harness failure localization and gated improvement.

## Suggested destination mapping

| Finding | A.L.I.C.E. destination | Timing |
| --- | --- | --- |
| JitMem read-time curation | Track F / Stage G Context Planner | future Stage G integration |
| REALM reconsolidation | Track C + F + Lifecycle Curator | future Stage G challenger |
| AdaMM analytic memory | Track D/F derived projection | future Stage G integration |
| MUTE influence echo | Track J deletion/unlearning | Stage G/J qualification |
| PGMem validity retrieval | Track G + Retrieval Orchestrator | future host/self retrieval |
| Recuris structured trace | Track H / F8 self-evolution | future procedural/self-improvement |
| FRESH failure graph | Track H + Cognitive Multi-Graph | future procedural memory |

## Non-action

Do not:

- replace Claim Authority with graph topology;
- make retrieval frequency a truth signal;
- retain all raw payloads forever only because JitMem benefits from raw trajectories;
- create a second persona authority store;
- freeze procedural/self-improvement to an external-memory-only model;
- treat one paper's fixed schema/store/model count as a Fable limit.

---

# 9. Next research questions created by this intake

Future watch runs should look specifically for evidence on:

1. hybrid write-time + read-time memory formation;
2. safe reconsolidation without self-reinforcing retrieval bias;
3. analytic memory with provenance and correction/deletion semantics;
4. causal influence tracing through model-generated future data;
5. multimodal read-time reconstruction;
6. read-time curation under very large personal histories;
7. learned switching among semantic, graph, episodic, analytic, source-native and parametric memory;
8. evaluation of memory systems where the correct answer requires **complete coverage**, not top-k relevance.

These are higher-value follow-ups than another generic GraphRAG comparison.
